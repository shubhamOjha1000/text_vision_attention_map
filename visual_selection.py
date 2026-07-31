"""
Sample the important IMAGE tokens from the text->vision attention map.

This is the step AFTER rater selection (`rater_selection.py`): given the raw
text->vision attention `P` and the selected rater text tokens, it scores every
image token and keeps the important ones -- SparseVLM's `VisualTokenSparsifier`
(SparseVLM_module.py), but computed from the GENUINE raw attention and aggregated
over ALL decoder layers.

It is the transpose of rater selection:
  * rater selection : softmax over TEXT per image col -> aggregate over image -> per-TEXT
  * this module      : softmax over IMAGE per text row -> aggregate over text  -> per-IMAGE

Flow (matches the "Sample img tokens" diagram), over ALL layers:
  0. restrict rows to the RATER text tokens (from rater_selection)     [n_raters, L_v]
  1. assert the raw slice is finite
  2. per (layer, head): softmax DOWN the IMAGE axis, per text row
        -> "distribution over image tokens for each text token"       (rows sum to 1)
  3. mean over rater rows, then mean over heads, normalise
        -> one image-token distribution per layer                     [L_v]
  4. stack all L layers -> [L, L_v], then SUM over layers and normalise
        -> final image-token distribution                             [L_v]
  5. top-k threshold                                                   keep L_v - floor(pct*L_v)

`P` must be the RAW pre-softmax scores: step 2's softmax is a renormalisation
over image tokens only, which is why the extractor returns raw scores.

Two ways to consume the importance map
--------------------------------------
  * SELECTION  -- `select_important_image_tokens` / `select_debiased`: top-k over
    the (optionally sink-corrected) importance. What to keep at inference.
  * DISTILLATION LABEL -- `teacher_label`: a sink-corrected target for training a
    student (e.g. FRM). Corrects in LOG space and EXCLUDES sinks from the
    candidate set rather than zeroing them; see the section at the bottom for why
    the selection path is not safe as a KL target.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class VisualResult:
    vision_mask: torch.Tensor      # bool  [L_v]  True = kept important image token
    importance: torch.Tensor       # float [L_v]  final per-image distribution (sums to 1)
    per_layer: torch.Tensor        # float [n_layers, L_v]  one distribution per layer (pre-sum)
    band: List[int]
    pct: float
    n_raters: int
    L_v: int = field(init=False)
    n_layers: int = field(init=False)
    n_kept: int = field(init=False)

    def __post_init__(self):
        self.L_v = int(self.importance.numel())
        self.n_layers = int(self.per_layer.shape[0])
        self.n_kept = int(self.vision_mask.sum())


def _threshold(importance: torch.Tensor, pct: float) -> torch.Tensor:
    """Top-k mask: keep the L_v - floor(pct*L_v) highest-importance image tokens."""
    L_v = importance.numel()
    n_keep = max(1, L_v - int(pct * L_v))
    mask = torch.zeros(L_v, dtype=torch.bool)
    mask[torch.topk(importance, n_keep).indices] = True
    return mask


def image_importance(maps_per_head: Dict[int, torch.Tensor],
                     rater_mask: torch.Tensor, *,
                     band: Optional[Sequence[int]] = None,
                     assert_finite: bool = True,
                     cand_mask: Optional[torch.Tensor] = None):
    """
    The per-image-token importance distribution (steps 0-4; NO threshold).
    Returns (importance[L_v], per_layer[n_layers, L_v], band_list, n_raters).
    Use this to score several (image, question) contexts and then de-bias them.

    `cand_mask` (bool [L_v]) excludes image tokens BEFORE the softmax -- pass the
    complement of `detect_sinks(...)` to stop attention sinks from stealing
    softmax mass from real patches. Excluded tokens come back as exactly 0.
    Removing a sink here is strictly better than subtracting it afterwards: after
    the softmax the damage (every other patch scaled down) is already done.
    """
    layers = sorted(maps_per_head.keys())
    if band is None:
        band = layers                                    # ALL captured layers
    band = [l for l in band if l in maps_per_head]
    if not band:
        raise ValueError(f"none of the requested band layers are in maps_per_head "
                         f"(have {layers})")

    rater_mask = rater_mask.bool()
    rater_idx = torch.nonzero(rater_mask, as_tuple=False).squeeze(-1)
    n_raters = int(rater_idx.numel())
    if n_raters == 0:
        raise ValueError("rater_mask selects no text tokens")

    if cand_mask is not None:
        cand_mask = cand_mask.bool()
        if int(cand_mask.sum()) == 0:
            raise ValueError("cand_mask excludes every image token")

    per_layer = []
    for l in band:
        P = maps_per_head[l][:, rater_idx, :].float()    # [H, n_raters, L_v]
        if assert_finite and not torch.isfinite(P).all():
            P = torch.where(torch.isfinite(P), P, torch.full_like(P, float("-inf")))
        if cand_mask is not None:                        # drop sinks BEFORE the softmax
            P = P.masked_fill(~cand_mask.view(1, 1, -1), float("-inf"))
        Ptil = F.softmax(P, dim=-1)                      # softmax over IMAGE, rows sum to 1
        s = Ptil.mean(dim=1).mean(dim=0)                 # mean over raters, then heads -> [L_v]
        s = s / s.sum().clamp_min(1e-12)
        per_layer.append(s)
    per_layer = torch.stack(per_layer, dim=0)            # [n_layers, L_v]

    final = per_layer.sum(dim=0)                         # sum over ALL layers
    final = final / final.sum().clamp_min(1e-12)         # [L_v]
    return final, per_layer, list(band), n_raters


# --------------------------------------------------------------------------- #
# Sink-token removal:  B (baseline subtraction)  +  A (drop invariant sinks)
#
# Two consumers, two different requirements:
#   * SELECTION (which tokens to keep at inference) -> `select_debiased` below.
#     Clamping to zero is harmless here; a dropped token is simply not selected.
#   * DISTILLATION LABELS (a KL target for a student such as FRM) -> use
#     `teacher_label` / `pmi_scores` instead. A clamped-to-zero entry contributes
#     NOTHING to `KL(p || r) = sum_i p_i log(p_i / r_i)`, so the student is never
#     penalised for putting mass there -- the sinks become free real estate.
# --------------------------------------------------------------------------- #
def sink_scores(attn_full: Dict[int, torch.Tensor],
                image_mask: torch.Tensor,
                text_mask: torch.Tensor, *,
                is_post_softmax: bool = False,
                quantile: float = 0.1,
                band: Optional[Sequence[int]] = None) -> torch.Tensor:
    """
    Per-image-token ATTENTION-SINK score [L_v], from ONE forward pass. Model-agnostic.

    Definition used
    ---------------
    A sink is a token that **every** query attends to. A content patch is attended
    to strongly by a *few* queries and ignored by the rest; a sink has a high
    FLOOR of incoming attention across all of them. So we score each image token by
    the low `quantile` of the attention it receives, not the mean -- the mean cannot
    separate "one patch everybody needs" from "one patch a few queries need a lot".

        score[j] = mean over (layer, head) of
                       quantile_over_text_queries( A[text_row, j] )

    Queries are restricted to TEXT rows, which sit after the image in the sequence
    and can therefore attend to every image token -- so causal masking never makes a
    token look sink-free just because it came late.

    Parameters
    ----------
    attn_full : {layer: [H, L, L]} attention over the FULL sequence (not the
        text->vision slice). Pass `ProbeOutput.raw_scores` (pre-softmax; softmaxed
        here over the full key axis) or `ProbeOutput.post_softmax` with
        `is_post_softmax=True`.
    image_mask, text_mask : bool [L] over the full sequence.

    Note: raw scores must already have the attention mask added (the HF eager
    probes do this). For a prefill pass with full visibility it makes no difference.
    """
    if not attn_full:
        raise ValueError("attn_full is empty")
    layers = sorted(attn_full)
    band = layers if band is None else [l for l in band if l in attn_full]
    if not band:
        raise ValueError(f"no requested band layer present (have {layers})")

    tpos = torch.nonzero(text_mask.bool(), as_tuple=False).squeeze(-1)
    vpos = torch.nonzero(image_mask.bool(), as_tuple=False).squeeze(-1)
    if tpos.numel() == 0 or vpos.numel() == 0:
        raise ValueError("need at least one text row and one image column")

    acc = torch.zeros(vpos.numel(), dtype=torch.float32)
    for l in band:
        A = attn_full[l].float()                       # [H, L, L]
        if not is_post_softmax:
            A = F.softmax(A, dim=-1)                   # true attention weights
        A = A[:, tpos][:, :, vpos]                     # [H, L_t, L_v]
        acc += torch.quantile(A, quantile, dim=1).mean(dim=0)   # floor, then heads
    return acc / len(band)


def aggregate_sink_scores(scores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean sink score over several examples. Sinks are a property of the MODEL, so
    averaging a handful of (image, question) pairs gives a much steadier estimate --
    and lets you freeze the resulting mask and reuse it forever."""
    return torch.stack([s.float() for s in scores], dim=0).mean(dim=0)


def detect_sinks(score: torch.Tensor, *,
                 k: Optional[int] = None,
                 z: float = 3.5,
                 max_frac: float = 0.15) -> torch.Tensor:
    """
    bool [L_v]: True where `score` marks an attention sink.

    With `k` -> simply the top-k. Otherwise a robust outlier rule (median + MAD),
    which adapts: it returns nothing when the model has no sink and several when it
    has several, instead of always removing a fixed count. `max_frac` caps how much
    of the grid can ever be called a sink.
    """
    score = score.float()
    L_v = score.numel()
    if k is not None:
        m = torch.zeros(L_v, dtype=torch.bool)
        if k > 0:
            m[torch.topk(score, min(k, L_v)).indices] = True
        return m

    med = score.median()
    mad = (score - med).abs().median()
    if float(mad) <= 0:                                  # degenerate / constant
        return torch.zeros(L_v, dtype=torch.bool)
    mz = 0.6745 * (score - med) / mad                    # modified z-score
    m = mz > z

    cap = int(max_frac * L_v)
    if int(m.sum()) > cap:                               # keep only the worst `cap`
        m = torch.zeros(L_v, dtype=torch.bool)
        m[torch.topk(score, max(cap, 1)).indices] = True
    return m


def sink_report(score: torch.Tensor, mask: torch.Tensor, grid: Optional[int] = None) -> str:
    """One-line-per-sink human summary, with (row, col) if the grid is square."""
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    L_v = score.numel()
    if grid is None:
        g = int(round(L_v ** 0.5))
        grid = g if g * g == L_v else None
    med = float(score.median())
    lines = [f"{len(idx)}/{L_v} tokens flagged as sinks (median score {med:.2e})"]
    for i in sorted(idx, key=lambda j: -float(score[j])):
        where = f" (row {i // grid}, col {i % grid})" if grid else ""
        lines.append(f"   token {i}{where}  score {float(score[i]):.2e}  "
                     f"= {float(score[i]) / max(med, 1e-12):.0f}x median")
    return "\n".join(lines)


def make_baseline(importances: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Position-bias baseline [L_v] = mean of several importance maps (from different
    questions / a null prompt). The question-INVARIANT part -- i.e. the sinks --
    survives the mean; question-specific grounding averages out.

    IMPORTANT (label leakage): if you build this from the SAME examples you then
    debias, each example contributes 1/N of its own correction, and every label
    depends on which other examples happened to be in the set -- so the label is
    not a function of its own (image, question). For labels, use one of:
      * a baseline built ONCE on a HELD-OUT split, then frozen and reused;
      * `make_baseline_loo` (leave-one-out) if a single set is all you have;
      * a per-example NULL-PROMPT baseline (run the same image with a
        content-free prompt) -- the cleanest option, and image-conditioned, so it
        also removes per-image position bias rather than only the corpus sink.
    """
    return torch.stack([i.float() for i in importances], dim=0).mean(dim=0)


def make_baseline_loo(importances: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Leave-one-out baselines -> [N, L_v]; row i is the mean of every importance
    map EXCEPT i. Removes the self-contribution that makes `make_baseline`
    corpus-dependent, so example i's label no longer depends on example i.

    Still depends on the other N-1 examples; a frozen held-out or null-prompt
    baseline is preferable when you can afford it.
    """
    stack = torch.stack([i.float() for i in importances], dim=0)   # [N, L_v]
    n = stack.shape[0]
    if n < 2:
        raise ValueError("leave-one-out needs at least 2 importance maps")
    total = stack.sum(dim=0, keepdim=True)                          # [1, L_v]
    return (total - stack) / (n - 1)                                # [N, L_v]


def subtract_baseline(importance: torch.Tensor, baseline: torch.Tensor,
                      *, renorm: bool = True) -> torch.Tensor:
    """B: remove the sink/position bias -> importance - baseline (clamped >= 0).

    SELECTION ONLY -- do not use as a distillation label. The subtraction happens
    in probability space although the sink is an additive bias in attention-LOGIT
    space, and `clamp_min(0)` zeroes every entry where importance <= baseline
    (most of them, since importance ~ baseline). Use `pmi_scores` for labels.
    """
    deb = (importance - baseline).clamp_min(0.0)
    if renorm:
        s = deb.sum()
        if s > 0:
            deb = deb / s
    return deb


def sink_token_mask(baseline: torch.Tensor, k: int) -> torch.Tensor:
    """A: bool[L_v] marking the k highest-baseline image tokens (the sinks)."""
    m = torch.zeros(baseline.numel(), dtype=torch.bool)
    if k > 0:
        m[torch.topk(baseline, min(k, baseline.numel())).indices] = True
    return m


def select_debiased(maps_per_head: Dict[int, torch.Tensor],
                    rater_mask: torch.Tensor, *,
                    baseline: torch.Tensor,
                    drop_sink_k: int = 0,
                    pct: float = 0.5,
                    band: Optional[Sequence[int]] = None) -> VisualResult:
    """
    Sink-robust selection: subtract the position-bias `baseline` (B), optionally
    zero the top `drop_sink_k` baseline patches (A), then keep the top-k.
    Build `baseline` with `make_baseline([...])` from a few questions / a null prompt.
    (`per_layer` in the result is the PRE-baseline per-layer distribution.)
    """
    importance, per_layer, band, n_raters = image_importance(
        maps_per_head, rater_mask, band=band)
    importance = subtract_baseline(importance, baseline)
    if drop_sink_k > 0:
        importance = importance.clone()
        importance[sink_token_mask(baseline, drop_sink_k)] = 0.0
        s = importance.sum()
        if s > 0:
            importance = importance / s
    vision_mask = _threshold(importance, pct)
    return VisualResult(vision_mask=vision_mask, importance=importance,
                        per_layer=per_layer, band=list(band), pct=pct,
                        n_raters=n_raters)


def select_important_image_tokens(
    maps_per_head: Dict[int, torch.Tensor],   # {layer: [H, L_t, L_v]} RAW scores
    rater_mask: torch.Tensor,                  # bool [L_t]  which text rows are raters
    *,
    band: Optional[Sequence[int]] = None,      # layers to use; default = ALL captured layers
    pct: float = 0.5,                          # fraction of image tokens to DROP
    assert_finite: bool = True,
) -> VisualResult:
    """
    Score image tokens using only the rater text rows' attention, aggregated over
    all (or `band`) decoder layers, and keep the top `L_v - floor(pct * L_v)`.
    """
    importance, per_layer, band, n_raters = image_importance(
        maps_per_head, rater_mask, band=band, assert_finite=assert_finite)
    vision_mask = _threshold(importance, pct)
    return VisualResult(vision_mask=vision_mask, importance=importance,
                        per_layer=per_layer, band=band, pct=pct, n_raters=n_raters)


def select_from_rater(maps_per_head: Dict[int, torch.Tensor], rater_result, *,
                      band=None, pct=0.5) -> VisualResult:
    """Convenience: take a `rater_selection.RaterResult` and sample image tokens."""
    return select_important_image_tokens(
        maps_per_head, rater_result.rater_mask, band=band, pct=pct)


# --------------------------------------------------------------------------- #
# Distillation labels (teacher targets), sink-corrected in LOG space
#
# `importance` is p(patch | question) and `baseline` is p(patch) -- the
# question-invariant part. The sink is an additive bias on the attention logits,
# so remove it there:
#
#       s_i = log p(i | q) - log p(i)          # pointwise mutual information
#
# i.e. "how much more does this patch matter GIVEN this question than it usually
# does". Dense (nothing clamped), so every candidate constrains the student in
# the KL; and shift-invariant per example, so it behaves like a logit.
#
# Following the FRM spec: store the RAW `scores` and let the LOSS normalise over
# the candidate set, so labels stay decoupled from the fovea-radius / sink-k
# choices and a sweep does not force regenerating them.
# --------------------------------------------------------------------------- #
def pmi_scores(importance: torch.Tensor, baseline: torch.Tensor,
               *, eps: float = 1e-12) -> torch.Tensor:
    """
    Raw, unnormalised, sink-corrected per-image-token scores [L_v]:
        log(importance) - log(baseline)
    Both inputs are distributions over the SAME L_v grid. `eps` only floors
    exact zeros (clamped, not added, so nonzero values are unshifted).
    """
    if importance.shape != baseline.shape:
        raise ValueError(f"shape mismatch: importance {tuple(importance.shape)} "
                         f"vs baseline {tuple(baseline.shape)}")
    imp = importance.float().clamp_min(eps)
    bas = baseline.float().clamp_min(eps)
    return torch.log(imp) - torch.log(bas)


def candidate_mask(L_v: int, *, exclude: Optional[Sequence[torch.Tensor]] = None
                   ) -> torch.Tensor:
    """
    bool[L_v] candidate set: True = scored by the loss, False = excluded outright.
    `exclude` is a list of bool[L_v] masks (sinks, the fovea region, ...).

    Excluding is NOT the same as zeroing the label: an excluded token is removed
    from the student's keys AND from the teacher's normalisation, so it can never
    be predicted. A token left in with target 0 costs the student nothing and can
    still be selected at deploy time.
    """
    cand = torch.ones(L_v, dtype=torch.bool)
    for m in (exclude or []):
        cand &= ~m.bool()
    if not cand.any():
        raise ValueError("all image tokens were excluded; candidate set is empty")
    return cand


@dataclass
class TeacherLabel:
    """One example's teacher target. Store `scores` + `cand_mask`; normalise late."""
    scores: torch.Tensor        # float [L_v]  RAW sink-corrected PMI scores
    cand_mask: torch.Tensor     # bool  [L_v]  candidate set (sinks/fovea removed)
    L_v: int = field(init=False)
    n_cand: int = field(init=False)

    def __post_init__(self):
        self.L_v = int(self.scores.numel())
        self.n_cand = int(self.cand_mask.sum())

    @property
    def cand_idx(self) -> torch.Tensor:
        return torch.nonzero(self.cand_mask, as_tuple=False).squeeze(-1)

    def distribution(self, temperature: float = 1.0) -> torch.Tensor:
        """Teacher distribution over the CANDIDATES -> [n_cand], sums to 1.

        Aligned with `cand_idx`. The student's logits must be computed over the
        same candidate set, so the KL ranks context only (FRM spec sec. 3.1/4).
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        return F.softmax(self.scores[self.cand_mask] / temperature, dim=0)

    def scatter(self, values: torch.Tensor) -> torch.Tensor:
        """Map a [n_cand] vector back onto the full [L_v] grid (0 off-candidates).
        For plotting only -- never feed the padded vector to a KL."""
        full = torch.zeros(self.L_v, dtype=values.dtype)
        full[self.cand_mask] = values
        return full


def teacher_label(importance: torch.Tensor, baseline: torch.Tensor, *,
                  drop_sink_k: int = 0,
                  exclude: Optional[Sequence[torch.Tensor]] = None,
                  eps: float = 1e-12) -> TeacherLabel:
    """
    Build one distillation target from a question-conditioned importance map and
    its question-invariant `baseline` (held-out / leave-one-out / null-prompt --
    see `make_baseline`).

    `drop_sink_k` removes the k highest-baseline patches from the CANDIDATE SET
    (not by zeroing their label). `exclude` adds further bool[L_v] masks, e.g.
    the FRM fovea region.
    """
    scores = pmi_scores(importance, baseline, eps=eps)
    masks = list(exclude or [])
    if drop_sink_k > 0:
        masks.append(sink_token_mask(baseline, drop_sink_k))
    return TeacherLabel(scores=scores,
                        cand_mask=candidate_mask(scores.numel(), exclude=masks))
