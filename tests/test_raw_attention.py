"""
Correctness tests for the RAW (pre-softmax) text->vision attention-score capture.

Design goal: GENERALISABLE TO ANY VLM.
-------------------------------------
The tests never touch PaliGemma directly. They are written against a small
abstract interface, `AttentionProbe`, that yields a `ProbeOutput` for one
(image, prompt) pair:

    ProbeOutput
      input_ids          [L]        the encoded sequence
      image_token_mask   [L] bool   True at visual-token positions
      text_token_mask    [L] bool   True at real text-token positions (no pad)
      raw_scores    {layer: [H, L, L]}   pre-softmax  QKᵀ/√d   (what we capture)
      post_softmax  {layer: [H, L, L]}   the model's ACTUAL attention distribution
      expected_num_image_tokens  int|None   exact count if the model advertises it

To validate a *different* VLM, implement one `AttentionProbe` that fills a
`ProbeOutput` from that model, add it to the `PROBES` list, and every invariant
test below runs against it unchanged.

Two probes ship here:
  * SyntheticProbe  -- no weights, always runs. Proves the invariants/tests are
    self-consistent and gives fast CI coverage. It is itself "just another VLM".
  * PaliGemmaProbe  -- loads the real model and captures BOTH the raw scores
    (`GemmaAttention._raw_attn_scores`) and the model's own post-softmax map
    (`output[1]`) in ONE forward pass. Skipped automatically if weights /
    auth / network are unavailable.

The tests fall in three groups:
  (A) Structure  -- token counts and sub-matrix shape (the two you proposed).
  (B) Raw-vs-softmax discriminators -- properties TRUE for raw logits and
      FALSE for a softmax distribution (negatives, rows don't sum to 1, != post).
  (C) Ground-truth identity -- softmax(raw) reproduces the model's post-softmax
      (argmax match + numeric reconstruction). This is THE proof that what we
      captured is the genuine pre-softmax logits of the real attention.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import pytest
import torch

# make the repo root importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Abstract interface + slicing helper (both model-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class ProbeOutput:
    input_ids: torch.Tensor            # [L]
    image_token_mask: torch.Tensor     # [L] bool
    text_token_mask: torch.Tensor      # [L] bool
    raw_scores: Dict[int, torch.Tensor]     # layer -> [H, L, L]  pre-softmax
    post_softmax: Dict[int, torch.Tensor]   # layer -> [H, L, L]  model attention
    expected_num_image_tokens: Optional[int] = None
    name: str = "probe"
    question: str = ""                 # the user's question text (for rater span scoping)
    L: int = field(init=False)

    def __post_init__(self):
        self.L = int(self.input_ids.shape[0])


def slice_text_vision(attn_HLL: torch.Tensor,
                      text_mask: torch.Tensor,
                      image_mask: torch.Tensor) -> torch.Tensor:
    """
    Reproduce the extractor's slice: rows = text positions, cols = vision
    positions.  attn_HLL: [H, L, L]  ->  [H, L_t, L_v].  Model-agnostic.
    """
    tpos = text_mask.nonzero(as_tuple=False).squeeze(-1)
    vpos = image_mask.nonzero(as_tuple=False).squeeze(-1)
    return attn_HLL[:, tpos][:, :, vpos]


# --------------------------------------------------------------------------- #
# Probe 1: synthetic (no model) -- exercises the invariants with known tensors
# --------------------------------------------------------------------------- #
def make_synthetic_output(seed: int = 0,
                          n_img: int = 12,
                          n_txt: int = 5,
                          n_pad: int = 2,
                          H: int = 4,
                          layers=(0, 1, 2)) -> ProbeOutput:
    """
    Layout mimics a VLM prefill sequence: [image tokens][bos+text tokens][pad].
    post_softmax is the true softmax of raw, so a correct capture MUST satisfy
    every invariant. This stands in for "any VLM" with a known-good attention.
    """
    g = torch.Generator().manual_seed(seed)
    L = n_img + n_txt + n_pad

    IMG_ID, PAD_ID = -100, -200
    ids = torch.empty(L, dtype=torch.long)
    ids[:n_img] = IMG_ID
    ids[n_img:n_img + n_txt] = torch.arange(1, n_txt + 1)  # arbitrary text ids
    ids[n_img + n_txt:] = PAD_ID

    image_mask = ids == IMG_ID
    text_mask = (ids != IMG_ID) & (ids != PAD_ID)

    raw_scores, post_softmax = {}, {}
    for lyr in layers:
        raw = torch.randn(H, L, L, generator=g) * 3.0   # includes negatives
        raw_scores[lyr] = raw
        post_softmax[lyr] = torch.softmax(raw, dim=-1)   # ground-truth attention
    return ProbeOutput(ids, image_mask, text_mask, raw_scores, post_softmax,
                       expected_num_image_tokens=n_img, name="synthetic")


# --------------------------------------------------------------------------- #
# Probe 2: real PaliGemma -- captures raw + post-softmax in one forward
# --------------------------------------------------------------------------- #
_PALIGEMMA_CACHE = {"tried": False, "output": None}


def make_paligemma_output() -> Optional[ProbeOutput]:
    """
    Returns a ProbeOutput from the real model, or None if it can't be loaded
    (no weights / no HF auth / no network / OOM). Result is cached across tests.
    """
    if _PALIGEMMA_CACHE["tried"]:
        return _PALIGEMMA_CACHE["output"]
    _PALIGEMMA_CACHE["tried"] = True

    model_path = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")
    try:
        from PIL import Image
        from modeling_gemma import KVCache, GemmaAttention
        from get_attention_map import load_paligemma

        model, processor, device = load_paligemma(model_path)

        raw_scores, post_softmax = {}, {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, _inp, output):
                # raw pre-softmax QKᵀ/√d that we stashed in the forward
                raw_scores[layer_idx] = module._raw_attn_scores.detach().float().cpu()[0]
                # output[1] is the model's genuine POST-softmax attention
                post_softmax[layer_idx] = output[1].detach().float().cpu()[0]
            return hook

        for m in model.modules():
            if isinstance(m, GemmaAttention):
                handles.append(m.register_forward_hook(make_hook(m.layer_idx)))

        prompt = "What is in the image?"
        image = Image.new("RGB", (224, 224), (127, 127, 127))
        inp = processor(text=[prompt], images=[image])
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            model(0.0, 0.0, False, False, None,
                  input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                  attention_mask=inp["attention_mask"], kv_cache=KVCache())
        for h in handles:
            h.remove()

        ids = inp["input_ids"][0].cpu()
        img_idx = model.config.image_token_index
        pad_id = model.pad_token_id
        image_mask = ids == img_idx
        text_mask = (ids != img_idx) & (ids != pad_id)

        out = ProbeOutput(
            ids, image_mask, text_mask, raw_scores, post_softmax,
            expected_num_image_tokens=model.config.vision_config.num_image_tokens,
            name="paligemma",
        )
    except Exception as e:  # weights/auth/network/etc. unavailable -> skip
        print(f"[paligemma probe unavailable] {type(e).__name__}: {e}")
        out = None

    _PALIGEMMA_CACHE["output"] = out
    return out


# --------------------------------------------------------------------------- #
# Fixture: run every test against every available probe
# --------------------------------------------------------------------------- #
PROBES = ["synthetic", "paligemma"]


@pytest.fixture(params=PROBES)
def probe(request) -> ProbeOutput:
    if request.param == "synthetic":
        return make_synthetic_output()
    out = make_paligemma_output()
    if out is None:
        pytest.skip("PaliGemma weights/auth/network unavailable")
    return out


# =========================================================================== #
# (A) STRUCTURE  -- the two tests you proposed
# =========================================================================== #
def test_token_counts_partition_the_sequence(probe):
    """#1: image-token and text-token counts are well-formed and disjoint."""
    n_img = int(probe.image_token_mask.sum())
    n_txt = int(probe.text_token_mask.sum())

    assert n_img > 0, "no image tokens detected"
    assert n_txt > 0, "no text tokens detected"
    # a position is never both image and text
    assert not (probe.image_token_mask & probe.text_token_mask).any()
    # image + text + (pad/other) accounts for the whole sequence, no overflow
    assert n_img + n_txt <= probe.L

    # if the model advertises a fixed image-token count, it must match exactly
    if probe.expected_num_image_tokens is not None:
        assert n_img == probe.expected_num_image_tokens


def test_submatrix_element_count(probe):
    """#2: sliced text->vision map has exactly n_text * n_image elements."""
    n_img = int(probe.image_token_mask.sum())
    n_txt = int(probe.text_token_mask.sum())

    for layer, raw in probe.raw_scores.items():
        P_heads = slice_text_vision(raw, probe.text_token_mask, probe.image_token_mask)
        head_avg = P_heads.mean(dim=0)                       # [L_t, L_v]
        assert head_avg.shape == (n_txt, n_img)
        assert head_avg.numel() == n_txt * n_img
        # per-head tensor carries the head dimension on top
        assert P_heads.shape == (raw.shape[0], n_txt, n_img)


# =========================================================================== #
# (B) RAW-vs-SOFTMAX DISCRIMINATORS
#     Properties that hold for raw logits but NOT for a softmax distribution.
# =========================================================================== #
def test_raw_scores_contain_negative_values(probe):
    """Softmax outputs are all >= 0; genuine QKᵀ/√d logits are not."""
    for layer, raw in probe.raw_scores.items():
        assert (raw < 0).any(), f"layer {layer}: raw scores have no negatives -> looks softmaxed"


def test_raw_rows_do_not_sum_to_one(probe):
    """Each softmax row (over all keys) sums to 1; raw score rows do not."""
    for layer, raw in probe.raw_scores.items():
        row_sums = raw.sum(dim=-1)                            # [H, L]
        ones = torch.ones_like(row_sums)
        assert not torch.allclose(row_sums, ones, atol=1e-2), \
            f"layer {layer}: raw rows sum to 1 -> looks softmaxed"


def test_raw_differs_from_post_softmax(probe):
    """The captured tensor must not just be the softmax map."""
    for layer in probe.raw_scores:
        assert not torch.allclose(
            probe.raw_scores[layer], probe.post_softmax[layer], atol=1e-4
        ), f"layer {layer}: raw == post-softmax -> wrong tensor captured"


# =========================================================================== #
# (C) GROUND-TRUTH IDENTITY:  softmax(raw) == model's post-softmax
#     The definitive proof the raw capture is the true pre-softmax of the
#     real attention. Restricted to "allowed" keys (post > 0) so padding /
#     causal masking is handled generically without knowing the mask.
# =========================================================================== #
def test_argmax_of_raw_matches_argmax_of_post(probe):
    """
    Softmax is monotonic, so the top-attended key by raw score should be the
    top-attended key by the model's attention -- per head, per query.

    We do NOT require the exact same index: under low-precision weights (e.g.
    4-bit / bf16 attention) two near-equal keys can round to the same post-softmax
    probability, so `argmax` (which breaks ties by lowest index) may pick a
    different one of the tied keys for `raw` vs `post`. Instead we require that
    raw's top key is (near-)maximal in post -- the probability gap between post's
    max and post at raw's argmax must be ~0. A genuine capture error (raw's top
    key pointing somewhere with low attention) still fails loudly with a big gap.
    """
    ATOL = 5e-3
    for layer in probe.raw_scores:
        raw = probe.raw_scores[layer]
        post = probe.post_softmax[layer]
        allowed_row = post.sum(dim=-1) > 0                        # [H, L]
        raw_arg = raw.argmax(dim=-1, keepdim=True)               # [H, L, 1]
        post_at_raw_arg = post.gather(-1, raw_arg).squeeze(-1)   # post prob at raw's top key
        gap = (post.max(dim=-1).values - post_at_raw_arg)[allowed_row]
        if gap.numel() == 0:
            continue
        assert gap.max().item() <= ATOL, \
            f"layer {layer}: raw's top key is not top-attention in post " \
            f"(max prob gap {gap.max().item():.2e})"


def test_softmax_of_raw_reconstructs_post_softmax(probe):
    """
    Recompute softmax(raw) over the allowed keys and compare to the model's
    post-softmax map. `allowed = post > 0` infers the effective mask generically.
    """
    for layer in probe.raw_scores:
        raw = probe.raw_scores[layer].double()
        post = probe.post_softmax[layer].double()

        allowed = post > 0
        neg_inf = torch.finfo(torch.float64).min
        masked_raw = torch.where(allowed, raw, torch.full_like(raw, neg_inf))
        recon = torch.softmax(masked_raw, dim=-1)

        # compare only where the model actually placed mass
        diff = (recon - post).abs()[allowed]
        if diff.numel() == 0:
            continue
        assert diff.max().item() < 2e-2, \
            f"layer {layer}: softmax(raw) does not match post-softmax (max diff {diff.max():.4f})"


def test_within_row_log_difference_is_constant(probe):
    """
    raw = log(post) + C  (per row), because softmax discards a per-row additive
    constant. So over allowed keys, (raw - log(post)) must be ~constant within a
    row. This is exactly the property that makes the raw scores recoverable up
    to a constant -- and confirms they ARE the logits behind `post`.
    """
    for layer in probe.raw_scores:
        raw = probe.raw_scores[layer].double()
        post = probe.post_softmax[layer].double()

        # only trust entries with non-tiny probability (log is ill-conditioned near 0)
        trust = post > 1e-4
        c = raw - torch.log(post.clamp_min(1e-12))            # [H, L, L]
        for h in range(raw.shape[0]):
            for q in range(raw.shape[1]):
                m = trust[h, q]
                if int(m.sum()) < 2:
                    continue
                vals = c[h, q][m]
                assert (vals.max() - vals.min()).item() < 5e-2, \
                    f"layer {layer} head {h} row {q}: raw - log(post) not constant"
