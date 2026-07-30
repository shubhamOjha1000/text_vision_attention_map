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

    L_v = maps_per_head[band[0]].shape[-1]

    per_layer = []
    for l in band:
        P = maps_per_head[l][:, rater_idx, :].float()    # [H, n_raters, L_v]
        if assert_finite and not torch.isfinite(P).all():
            P = torch.where(torch.isfinite(P), P, torch.full_like(P, float("-inf")))
        # Step 2: softmax DOWN the image axis (dim=-1), per (head, text row)
        Ptil = F.softmax(P, dim=-1)                      # [H, n_raters, L_v], rows sum to 1
        # Step 3: mean over rater rows, then over heads -> per-image scores; normalise
        s = Ptil.mean(dim=1).mean(dim=0)                 # [L_v]
        s = s / s.sum().clamp_min(1e-12)                 # distribution over image tokens
        per_layer.append(s)
    per_layer = torch.stack(per_layer, dim=0)            # [n_layers, L_v]

    # Step 4: sum over all L layers, normalise -> final image-token distribution
    final = per_layer.sum(dim=0)
    final = final / final.sum().clamp_min(1e-12)         # [L_v]

    # Step 5: top-k threshold -> keep the important image tokens
    n_drop = int(pct * L_v)
    n_keep = max(1, L_v - n_drop)
    vision_mask = torch.zeros(L_v, dtype=torch.bool)
    vision_mask[torch.topk(final, n_keep).indices] = True

    return VisualResult(vision_mask=vision_mask, importance=final,
                        per_layer=per_layer, band=list(band), pct=pct,
                        n_raters=n_raters)


def select_from_rater(maps_per_head: Dict[int, torch.Tensor], rater_result, *,
                      band=None, pct=0.5) -> VisualResult:
    """Convenience: take a `rater_selection.RaterResult` and sample image tokens."""
    return select_important_image_tokens(
        maps_per_head, rater_result.rater_mask, band=band, pct=pct)
