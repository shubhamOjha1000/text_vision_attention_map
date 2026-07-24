"""
Select the important ("rater") text tokens from a text->vision attention map.

This is the "Sample only important text tokens" step: it turns the raw text->vision
attention `P` (rows = text tokens, cols = vision tokens; produced by
`get_attention_map` / `attention_extractor`) into a per-text-token importance
score, then keeps only the visually-grounded text tokens -- SparseVLM's rater
selection, but computed from the GENUINE raw attention instead of a re-derived
embedding similarity.

Flow (matches the diagram / SparseVLM Eq. 6), with the locked design choices:
  0. keep only real text tokens        -> drop BOS, "\n", <image>/vision delims,
                                          special/added tokens          (3.3)
  1. assert the raw slice is finite     -> no causal/pad mask contamination (3.5)
  2. per (layer, head): softmax DOWN the text axis, per image column
                                          -> "distribution over text for each
                                             image token" (softmax BEFORE averaging)
  3. mean over heads (3.2), then mean over a BAND of layers (3.1)
  4. r = mean over image columns        -> importance per text token (Eq. 6)
       (optional vision_weights = gaze hook: weighted mean over image columns)
  5. top-k threshold                    -> keep  L_t' - floor(pct * L_t')  (3.4)

`P` must be the RAW pre-softmax scores: step 2's softmax is a *renormalisation
over content text tokens only*, which is the whole reason the extractor returns
raw scores.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Structural-token detection (decision 3.3)
# --------------------------------------------------------------------------- #
# Matches special-format tokens: <image>, <bos>, </s>, <loc0001>, <seg000>,
# <|im_start|>, <|vision_start|>, ...
_SPECIAL_RE = re.compile(r"^<.*>$|<\|.*\|>")

# byte / sentencepiece space+newline markers -> real chars, so we can tell a
# token that is *only* whitespace/newline (a structural token) from a real word.
_MARKER_MAP = {"▁": " ", "Ġ": " ", "Ċ": "\n", "ĉ": "\t"}


def _clean_token(tok: str) -> str:
    for k, v in _MARKER_MAP.items():
        tok = tok.replace(k, v)
    return tok.strip()


def is_structural_token(tok: Optional[str]) -> bool:
    """True if `tok` is a structural / special / whitespace token (to suppress)."""
    if tok is None:
        return True
    if _clean_token(tok) == "":          # pure whitespace / newline
        return True
    if _SPECIAL_RE.match(tok):           # <...> or <|...|> special format
        return True
    return False


def content_text_mask(text_tokens: Sequence[str],
                      tokenizer=None) -> torch.Tensor:
    """
    bool[L_t] over the text rows: True = real content token to keep, False =
    structural/special token to suppress. Uses token strings, and additionally
    `tokenizer.all_special_ids` when a tokenizer is given.
    """
    L_t = len(text_tokens)
    keep = torch.ones(L_t, dtype=torch.bool)
    for i, tok in enumerate(text_tokens):
        if is_structural_token(tok):
            keep[i] = False
    if tokenizer is not None:
        try:
            special_ids = set(tokenizer.all_special_ids)
            ids = tokenizer.convert_tokens_to_ids(list(text_tokens))
            for i, tid in enumerate(ids):
                if tid in special_ids:
                    keep[i] = False
        except Exception:
            pass
    return keep


# --------------------------------------------------------------------------- #
# Layer band (decision 3.1)
# --------------------------------------------------------------------------- #
def default_band(layers: Sequence[int]) -> List[int]:
    """Middle third of the available decoder layers."""
    layers = sorted(layers)
    n = len(layers)
    if n <= 2:
        return list(layers)
    lo = n // 3
    hi = max(lo + 1, (2 * n) // 3)
    return list(layers[lo:hi])


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class RaterResult:
    rater_mask: torch.Tensor       # bool  [L_t]  True = selected important token
    importance: torch.Tensor       # float [L_t]  per-text importance (0 if suppressed)
    content_mask: torch.Tensor     # bool  [L_t]  survived structural suppression
    band: List[int]
    pct: float
    n_content: int = field(init=False)
    n_kept: int = field(init=False)

    def __post_init__(self):
        self.n_content = int(self.content_mask.sum())
        self.n_kept = int(self.rater_mask.sum())

    def kept_tokens(self, text_tokens: Sequence[str]) -> List[str]:
        return [t for t, m in zip(text_tokens, self.rater_mask.tolist()) if m]


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #
def select_important_text_tokens(
    maps_per_head: Dict[int, torch.Tensor],   # {layer: [H, L_t, L_v]} RAW scores
    *,
    text_tokens: Optional[Sequence[str]] = None,
    tokenizer=None,
    content_mask: Optional[torch.Tensor] = None,
    band: Optional[Sequence[int]] = None,
    pct: float = 0.5,
    vision_weights: Optional[torch.Tensor] = None,   # gaze hook: [L_v], defaults uniform
    assert_finite: bool = True,
) -> RaterResult:
    """
    Returns a RaterResult marking which text tokens are the important "raters".

    `pct` is the fraction of (content) text tokens to DROP, so we keep
    `L_t' - floor(pct * L_t')` tokens, where `L_t'` = number of content tokens.
    Provide either `content_mask` directly, or `text_tokens` (+ optional
    `tokenizer`) so structural tokens can be detected and suppressed.
    """
    layers = sorted(maps_per_head.keys())
    if band is None:
        band = default_band(layers)
    band = [l for l in band if l in maps_per_head]
    if not band:
        raise ValueError(f"none of the requested band layers are in maps_per_head "
                         f"(have {layers})")

    sample = maps_per_head[band[0]]
    H, L_t, L_v = sample.shape

    # ---- Step 0: content (real text) mask, decision 3.3 ----
    if content_mask is None:
        if text_tokens is None:
            raise ValueError("provide content_mask or text_tokens for suppression")
        content_mask = content_text_mask(text_tokens, tokenizer)
    content_mask = content_mask.bool()
    content_idx = torch.nonzero(content_mask, as_tuple=False).squeeze(-1)
    L_t_prime = int(content_idx.numel())
    if L_t_prime == 0:
        raise ValueError("all text tokens were suppressed as structural; nothing to rate")

    # ---- Steps 1-3: per-(layer,head) column-softmax, then mean over heads & band ----
    acc = torch.zeros(L_t_prime, L_v, dtype=torch.float32)  # accumulate distributions
    for l in band:
        P = maps_per_head[l][:, content_idx, :].float()     # [H, L_t', L_v]
        if assert_finite and not torch.isfinite(P).all():
            # text->image is causal-allowed, so this should not happen. Don't crash:
            # push non-finite entries to -inf so softmax gives them zero mass.
            P = torch.where(torch.isfinite(P), P, torch.full_like(P, float("-inf")))
        # Step 2: softmax DOWN the text axis (dim=1), per image column -> distribution
        Ptil = F.softmax(P, dim=1)                          # [H, L_t', L_v], cols sum to 1
        acc += Ptil.mean(dim=0)                             # Step 3a: mean over heads
    Ptil_bar = acc / len(band)                              # Step 3b: mean over band

    # ---- Step 4: aggregate across image columns -> per-text importance (Eq. 6) ----
    if vision_weights is None:
        r = Ptil_bar.mean(dim=1)                            # [L_t']
    else:                                                   # gaze hook: weighted mean
        w = vision_weights.float().view(1, L_v)
        w = w / w.sum().clamp_min(1e-12)
        r = (Ptil_bar * w).sum(dim=1)                       # [L_t']

    # ---- Step 5: top-k threshold, decision 3.4 ----
    n_drop = int(pct * L_t_prime)
    n_keep = max(1, L_t_prime - n_drop)                     # guarantee >= 1 rater
    keep_local = torch.zeros(L_t_prime, dtype=torch.bool)
    keep_local[torch.topk(r, n_keep).indices] = True

    # ---- Step 6: map back to original text-row indexing ----
    rater_mask = torch.zeros(L_t, dtype=torch.bool)
    rater_mask[content_idx[keep_local]] = True
    importance = torch.zeros(L_t, dtype=torch.float32)
    importance[content_idx] = r

    return RaterResult(rater_mask=rater_mask, importance=importance,
                       content_mask=content_mask, band=list(band), pct=pct)


# --------------------------------------------------------------------------- #
# Convenience wrapper for the extractor output
# --------------------------------------------------------------------------- #
def select_from_extractor(out, *, tokenizer=None, band=None, pct=0.5,
                          vision_weights=None) -> RaterResult:
    """
    Accepts the dict returned by `get_attention_maps` OR an
    `attention_extractor.AttentionMapResult`, and runs the rater selection.
    """
    if isinstance(out, dict):
        maps_per_head = out["maps_per_head"]
        text_tokens = out["text_tokens"]
    else:  # AttentionMapResult
        maps_per_head = out.maps_per_head
        text_tokens = out.text_tokens
    return select_important_text_tokens(
        maps_per_head, text_tokens=text_tokens, tokenizer=tokenizer,
        band=band, pct=pct, vision_weights=vision_weights,
    )
