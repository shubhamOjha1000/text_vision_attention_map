"""
Tests for visual_selection.select_important_image_tokens ("Sample img tokens").

The `check_*` functions take a `VisualCase` and are reused by both pytest (a
synthetic map) and a real-VLM notebook. Covers the diagram's three test cases:
  1. final distribution is a row over image tokens: rows == 1, cols == L_v
  2. #per-layer distributions (before the layer-sum) == #decoder layers used
  3. the input matrix has one row per RATER text token (after the threshold)
plus: distribution validity, top-k count, thresholding reduces #image tokens,
uses all layers by default, and grounded image tokens rank on top.
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import List

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_selection import (  # noqa: E402
    VisualResult,
    image_importance,
    make_baseline,
    select_debiased,
    select_important_image_tokens,
    sink_token_mask,
    subtract_baseline,
)


@dataclass
class VisualCase:
    maps_per_head: dict          # {layer: [H, L_t, L_v]}
    rater_mask: torch.Tensor     # bool [L_t]
    pct: float
    res: VisualResult
    name: str = "case"


def make_case(maps_per_head, rater_mask, *, pct=0.5, band=None, name="case") -> VisualCase:
    res = select_important_image_tokens(maps_per_head, rater_mask, band=band, pct=pct)
    return VisualCase(maps_per_head, rater_mask.bool(), pct, res, name)


# --------------------------------------------------------------------------- #
# Reusable invariant checks
# --------------------------------------------------------------------------- #
def check_final_is_row_over_images(case):
    """#1: final distribution is [L_v]; as a row it is [1, L_v]."""
    imp = case.res.importance
    L_v = case.maps_per_head[sorted(case.maps_per_head)[0]].shape[-1]
    assert imp.ndim == 1 and imp.shape[0] == L_v
    assert tuple(imp.view(1, -1).shape) == (1, L_v)


def check_num_layer_distributions_equals_layers(case):
    """#2: one prob distribution per decoder layer used (before the layer-sum)."""
    assert case.res.per_layer.shape[0] == len(case.res.band)
    assert case.res.n_layers == len(case.res.band)


def check_input_rows_equal_raters(case):
    """#3: the matrix rows fed in == number of rater text tokens (post-threshold)."""
    assert case.res.n_raters == int(case.rater_mask.sum())


def check_per_layer_rows_are_distributions(case):
    """each per-layer row is a valid distribution over image tokens."""
    pl = case.res.per_layer
    assert (pl >= 0).all()
    assert torch.allclose(pl.sum(dim=-1), torch.ones(pl.shape[0]), atol=1e-4)


def check_final_is_distribution(case):
    imp = case.res.importance
    assert (imp >= 0).all()
    assert abs(imp.sum().item() - 1.0) < 1e-4


def check_thresholding_reduces_image_tokens(case):
    assert case.res.n_kept <= case.res.L_v
    if case.pct > 0:
        assert case.res.n_kept < case.res.L_v


def check_topk_count_formula(case):
    assert case.res.n_kept == max(1, case.res.L_v - int(case.pct * case.res.L_v))


def check_uses_all_layers_by_default(case):
    assert case.res.band == sorted(case.maps_per_head.keys())


ALL_CHECKS = [
    check_final_is_row_over_images,
    check_num_layer_distributions_equals_layers,
    check_input_rows_equal_raters,
    check_per_layer_rows_are_distributions,
    check_final_is_distribution,
    check_thresholding_reduces_image_tokens,
    check_topk_count_formula,
    check_uses_all_layers_by_default,
]


# --------------------------------------------------------------------------- #
# Synthetic case: raters attend to a known image token; check it ranks top
# --------------------------------------------------------------------------- #
L_T, L_V, H, N_LAYERS = 5, 8, 3, 6
RATER_IDX = [1, 3]          # two rater text tokens
GROUNDED_IMG = 4           # both raters attend strongly to image token 4


def _synthetic_maps(seed=0):
    g = torch.Generator().manual_seed(seed)
    maps = {}
    for l in range(N_LAYERS):
        P = torch.randn(H, L_T, L_V, generator=g) * 0.5
        for i in RATER_IDX:
            P[:, i, GROUNDED_IMG] += 6.0     # raters attend to image token 4
        maps[l] = P
    return maps


def _rater_mask():
    m = torch.zeros(L_T, dtype=torch.bool)
    m[RATER_IDX] = True
    return m


def _synthetic_case(pct=0.5):
    return make_case(_synthetic_maps(), _rater_mask(), pct=pct, name="synthetic")


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_synthetic_invariant(check):
    check(_synthetic_case())


def test_grounded_image_token_is_selected_and_ranks_top():
    res = _synthetic_case(pct=0.5).res
    assert res.importance.argmax().item() == GROUNDED_IMG
    assert res.vision_mask[GROUNDED_IMG]


def test_pct_controls_kept_image_count():
    for pct, expect in [(0.0, 8), (0.25, 6), (0.5, 4), (0.9, 1)]:
        res = _synthetic_case(pct=pct).res
        assert res.n_kept == expect, f"pct={pct}: {res.n_kept} != {expect}"


def test_band_override_uses_subset():
    res = select_important_image_tokens(_synthetic_maps(), _rater_mask(), band=[0, 1])
    assert res.band == [0, 1] and res.n_layers == 2


def test_empty_rater_mask_raises():
    with pytest.raises(ValueError):
        select_important_image_tokens(_synthetic_maps(),
                                      torch.zeros(L_T, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# Sink removal:  B (baseline subtraction) + A (drop invariant sinks)
# --------------------------------------------------------------------------- #
SINK_IMG = 7          # a fixed image token that both "questions" attend to (a sink)


def _maps_with_sink(question_img, seed):
    """raters attend strongly to the shared SINK_IMG (sink) AND to a
    question-specific image token `question_img`."""
    g = torch.Generator().manual_seed(seed)
    maps = {}
    for l in range(N_LAYERS):
        P = torch.randn(H, L_T, L_V, generator=g) * 0.3
        for i in RATER_IDX:
            P[:, i, SINK_IMG] += 8.0          # shared sink (question-invariant)
            P[:, i, question_img] += 5.0      # question-specific grounding
        maps[l] = P
    return maps


def test_baseline_is_the_shared_sink():
    imp_q1, *_ = image_importance(_maps_with_sink(2, 0), _rater_mask())
    imp_q2, *_ = image_importance(_maps_with_sink(5, 1), _rater_mask())
    base = make_baseline([imp_q1, imp_q2])
    assert base.argmax().item() == SINK_IMG         # the sink is the invariant peak


def test_baseline_subtraction_recovers_question_specific_tokens():
    rmask = _rater_mask()
    imp_q1, *_ = image_importance(_maps_with_sink(2, 0), rmask)
    imp_q2, *_ = image_importance(_maps_with_sink(5, 1), rmask)
    assert imp_q1.argmax().item() == SINK_IMG       # before: sink dominates both
    assert imp_q2.argmax().item() == SINK_IMG

    base = make_baseline([imp_q1, imp_q2])
    d1 = subtract_baseline(imp_q1, base)
    d2 = subtract_baseline(imp_q2, base)
    assert d1.argmax().item() == 2                  # after: question-specific token wins
    assert d2.argmax().item() == 5
    assert d1[SINK_IMG] < d1[2] and d2[SINK_IMG] < d2[5]


def test_sink_token_mask_picks_baseline_top():
    base = torch.tensor([0.1, 0.9, 0.2, 0.7, 0.05])
    assert sink_token_mask(base, 2).tolist() == [False, True, False, True, False]
    assert sink_token_mask(base, 0).sum() == 0


def test_select_debiased_excludes_sink_and_keeps_grounded():
    rmask = _rater_mask()
    imp_q1, *_ = image_importance(_maps_with_sink(2, 0), rmask)
    imp_q2, *_ = image_importance(_maps_with_sink(5, 1), rmask)
    base = make_baseline([imp_q1, imp_q2])

    res = select_debiased(_maps_with_sink(2, 0), rmask, baseline=base,
                          drop_sink_k=1, pct=0.9)      # keep few -> must be grounded
    assert isinstance(res, VisualResult)
    assert not res.vision_mask[SINK_IMG]               # sink dropped
    assert res.vision_mask[2]                          # question-specific token kept
    assert res.importance[SINK_IMG].item() == 0.0      # A zeroed the sink
