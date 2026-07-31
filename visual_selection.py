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
                     assert_finite: bool = True):
    """
    The per-image-token importance distribution (steps 0-4; NO threshold).
    Returns (importance[L_v], per_layer[n_layers, L_v], band_list, n_raters).
    Use this to score several (image, question) contexts and then de-bias them.
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

    per_layer = []
    for l in band:
        P = maps_per_head[l][:, rater_idx, :].float()    # [H, n_raters, L_v]
        if assert_finite and not torch.isfinite(P).all():
            P = torch.where(torch.isfinite(P), P, torch.full_like(P, float("-inf")))
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
# --------------------------------------------------------------------------- #
def make_baseline(importances: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Position-bias baseline [L_v] = mean of several importance maps (from different
    questions / a null prompt). The question-INVARIANT part -- i.e. the sinks --
    survives the mean; question-specific grounding averages out.
    """
    return torch.stack([i.float() for i in importances], dim=0).mean(dim=0)


def subtract_baseline(importance: torch.Tensor, baseline: torch.Tensor,
                      *, renorm: bool = True) -> torch.Tensor:
    """B: remove the sink/position bias -> importance - baseline (clamped >= 0)."""
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
