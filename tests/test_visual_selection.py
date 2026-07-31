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
    TeacherLabel,
    VisualResult,
    aggregate_sink_scores,
    candidate_mask,
    detect_sinks,
    image_importance,
    make_baseline,
    make_baseline_loo,
    pmi_scores,
    select_debiased,
    select_important_image_tokens,
    sink_report,
    sink_scores,
    sink_token_mask,
    subtract_baseline,
    teacher_label,
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


# --------------------------------------------------------------------------- #
# Blocker 1 -- baselines must not leak the example into its own correction
# --------------------------------------------------------------------------- #
def _sink_importances(question_imgs=(2, 5, 3), seed0=0):
    rmask = _rater_mask()
    return [image_importance(_maps_with_sink(q, seed0 + i), rmask)[0]
            for i, q in enumerate(question_imgs)]


def test_loo_baseline_excludes_own_contribution():
    imps = _sink_importances()
    loo = make_baseline_loo(imps)
    assert loo.shape == (len(imps), L_V)
    for i in range(len(imps)):
        others = [x for j, x in enumerate(imps) if j != i]
        assert torch.allclose(loo[i], make_baseline(others), atol=1e-6)


def test_loo_baseline_row_is_independent_of_own_map():
    """Perturbing example i must not change example i's own baseline."""
    imps = _sink_importances()
    before = make_baseline_loo(imps)[0].clone()
    imps[0] = torch.rand(L_V); imps[0] /= imps[0].sum()   # scramble example 0 only
    assert torch.allclose(make_baseline_loo(imps)[0], before, atol=1e-6)


def test_loo_baseline_still_finds_the_sink():
    imps = _sink_importances()
    assert make_baseline_loo(imps)[0].argmax().item() == SINK_IMG


def test_loo_baseline_needs_two_maps():
    with pytest.raises(ValueError):
        make_baseline_loo([torch.rand(L_V)])


# --------------------------------------------------------------------------- #
# Blocker 2 -- labels: log-space correction, dense support, sinks EXCLUDED
# --------------------------------------------------------------------------- #
def test_pmi_recovers_question_specific_token():
    imps = _sink_importances(question_imgs=(2, 5))
    base = make_baseline(imps)
    assert imps[0].argmax().item() == SINK_IMG          # before: sink dominates
    s = pmi_scores(imps[0], base)
    assert s.argmax().item() == 2                       # after: grounded token wins
    assert s[2] > s[SINK_IMG]


def test_pmi_support_is_dense_unlike_subtraction():
    """The clamp in subtract_baseline kills entries; PMI keeps every token."""
    imps = _sink_importances(question_imgs=(2, 5))
    base = make_baseline(imps)
    sub = subtract_baseline(imps[0], base)
    lab = teacher_label(imps[0], base)
    assert (sub == 0).any()                             # subtraction zeroes entries
    assert torch.isfinite(lab.scores).all()
    assert (lab.distribution() > 0).all()               # every candidate gets mass


def test_teacher_label_excludes_sink_from_candidates_not_by_zeroing():
    imps = _sink_importances(question_imgs=(2, 5))
    base = make_baseline(imps)
    lab = teacher_label(imps[0], base, drop_sink_k=1)
    assert isinstance(lab, TeacherLabel)
    assert not lab.cand_mask[SINK_IMG]                  # removed from the set ...
    assert lab.n_cand == L_V - 1
    assert SINK_IMG not in lab.cand_idx.tolist()
    assert lab.scores[SINK_IMG].isfinite()              # ... not zeroed in the label


def test_teacher_distribution_is_over_candidates_and_normalised():
    imps = _sink_importances(question_imgs=(2, 5))
    lab = teacher_label(imps[0], make_baseline(imps), drop_sink_k=1)
    p = lab.distribution()
    assert p.shape == (lab.n_cand,)
    assert abs(p.sum().item() - 1.0) < 1e-5
    assert p.argmax().item() == lab.cand_idx.tolist().index(2)


def test_temperature_flattens_without_reordering():
    imps = _sink_importances(question_imgs=(2, 5))
    lab = teacher_label(imps[0], make_baseline(imps))
    sharp, flat = lab.distribution(0.5), lab.distribution(2.0)
    assert sharp.max() > flat.max()                                  # flatter
    assert torch.equal(sharp.argsort(), flat.argsort())              # same ranking


def test_scores_are_independent_of_candidate_choice():
    """FRM spec: store RAW scores so a sink-k / fovea sweep never regenerates labels."""
    imps = _sink_importances(question_imgs=(2, 5))
    base = make_baseline(imps)
    a = teacher_label(imps[0], base, drop_sink_k=0)
    b = teacher_label(imps[0], base, drop_sink_k=3)
    assert torch.equal(a.scores, b.scores)
    assert a.n_cand == L_V and b.n_cand == L_V - 3


def test_candidate_mask_combines_exclusions():
    fovea = torch.zeros(L_V, dtype=torch.bool); fovea[[0, 1]] = True
    sinks = torch.zeros(L_V, dtype=torch.bool); sinks[[1, 7]] = True
    cand = candidate_mask(L_V, exclude=[fovea, sinks])
    assert cand.tolist() == [False, False, True, True, True, True, True, False]


def test_candidate_mask_rejects_empty_set():
    with pytest.raises(ValueError):
        candidate_mask(L_V, exclude=[torch.ones(L_V, dtype=torch.bool)])


def test_pmi_shape_mismatch_raises():
    with pytest.raises(ValueError):
        pmi_scores(torch.rand(L_V), torch.rand(L_V + 1))


# --------------------------------------------------------------------------- #
# Attention-sink detection from a full-sequence attention tensor
#
# Synthetic "VLM": L = L_V image tokens followed by L_T text tokens. One image
# token is a SINK (every query attends to it); one is CONTENT (only two queries
# attend to it, but very strongly, so its MEAN is comparable to the sink's --
# which is exactly the case a mean-based detector cannot separate).
# --------------------------------------------------------------------------- #
SINK_TOK, CONTENT_TOK = 3, 6


def _vlm_attention(n_layers=4, heads=H, seed=0, sink=SINK_TOK, content=CONTENT_TOK):
    """Sink gets a MODERATE boost from every query (+3). Content gets a LARGER
    boost (+5) but only from 3 of the 5 text queries. The sizes are chosen so the
    content token wins on MEAN incoming attention while the sink wins on the
    low-quantile FLOOR -- the case that separates the two detectors."""
    g = torch.Generator().manual_seed(seed)
    L = L_V + L_T
    img = torch.zeros(L, dtype=torch.bool); img[:L_V] = True
    txt = torch.zeros(L, dtype=torch.bool); txt[L_V:] = True

    attn = {}
    for l in range(n_layers):
        S = torch.randn(heads, L, L, generator=g) * 0.3
        S[:, :, sink] += 3.0                              # every query -> the sink
        for r in range(L_V, L_V + 3):                     # 3 of 5 text queries only
            S[:, r, content] += 5.0
        attn[l] = S
    return attn, img, txt


def test_sink_scores_flag_the_sink_not_the_content_token():
    attn, img, txt = _vlm_attention()
    s = sink_scores(attn, img, txt)
    assert s.shape == (L_V,)
    assert int(s.argmax()) == SINK_TOK
    assert s[SINK_TOK] > s[CONTENT_TOK]


def test_mean_attention_would_be_fooled_where_the_quantile_is_not():
    """The content token is designed so its MEAN incoming attention rivals the
    sink's. Only the low-quantile floor separates them -- this is the whole reason
    sink_scores uses a quantile."""
    attn, img, txt = _vlm_attention()
    tpos, vpos = torch.nonzero(txt).squeeze(-1), torch.nonzero(img).squeeze(-1)
    A = torch.softmax(attn[0].float(), dim=-1)[:, tpos][:, :, vpos]
    mean_score = A.mean(dim=1).mean(dim=0)
    q_score = sink_scores({0: attn[0]}, img, txt)
    assert mean_score[CONTENT_TOK] > mean_score[SINK_TOK]      # mean is fooled
    assert q_score[SINK_TOK] > q_score[CONTENT_TOK]            # the floor is not


def test_detect_sinks_finds_the_planted_sink_and_not_the_content_token():
    attn, img, txt = _vlm_attention()
    m = detect_sinks(sink_scores(attn, img, txt))
    assert m[SINK_TOK] and not m[CONTENT_TOK]
    assert 1 <= int(m.sum()) <= 2


def test_detect_sinks_returns_nothing_when_there_is_no_sink():
    g = torch.Generator().manual_seed(3)
    assert detect_sinks(torch.rand(L_V, generator=g) * 1e-3 + 0.5).sum() == 0
    assert detect_sinks(torch.full((L_V,), 0.25)).sum() == 0     # constant -> MAD 0


def test_detect_sinks_k_overrides_the_outlier_rule():
    attn, img, txt = _vlm_attention()
    s = sink_scores(attn, img, txt)
    assert int(detect_sinks(s, k=3).sum()) == 3
    assert int(detect_sinks(s, k=0).sum()) == 0


def test_detect_sinks_respects_max_frac():
    s = torch.arange(L_V, dtype=torch.float32) ** 4      # many extreme outliers
    assert int(detect_sinks(s, max_frac=0.25).sum()) <= int(0.25 * L_V)


def test_sink_scores_post_softmax_matches_raw():
    attn, img, txt = _vlm_attention()
    post = {l: torch.softmax(a.float(), dim=-1) for l, a in attn.items()}
    assert torch.allclose(sink_scores(attn, img, txt),
                          sink_scores(post, img, txt, is_post_softmax=True), atol=1e-6)


def test_aggregate_sink_scores_is_steadier_across_examples():
    scores = []
    for s in range(4):
        attn, img, txt = _vlm_attention(seed=s)
        scores.append(sink_scores(attn, img, txt))
    agg = aggregate_sink_scores(scores)
    assert agg.shape == (L_V,)
    assert int(agg.argmax()) == SINK_TOK
    assert detect_sinks(agg)[SINK_TOK]


def test_sink_report_is_readable():
    attn, img, txt = _vlm_attention()
    s = sink_scores(attn, img, txt)
    txt_out = sink_report(s, detect_sinks(s))
    assert "sinks" in txt_out and f"token {SINK_TOK}" in txt_out


# --------------------------------------------------------------------------- #
# Removing the sink BEFORE the softmax
# --------------------------------------------------------------------------- #
def test_cand_mask_zeroes_excluded_and_renormalises():
    maps = _maps_with_sink(2, 0)
    cand = candidate_mask(L_V, exclude=[sink_token_mask(
        image_importance(maps, _rater_mask())[0], 1)])
    imp, _, _, _ = image_importance(maps, _rater_mask(), cand_mask=cand)
    assert float(imp[SINK_IMG]) == 0.0
    assert abs(float(imp.sum()) - 1.0) < 1e-5
    assert int(imp.argmax()) == 2                     # grounded token now wins


def test_excluding_before_softmax_beats_subtracting_after():
    """Post-hoc subtraction cannot undo the mass the sink already stole from every
    other patch; excluding it pre-softmax gives the real patch strictly more."""
    maps = _maps_with_sink(2, 0)
    rm = _rater_mask()
    plain, *_ = image_importance(maps, rm)
    cand = candidate_mask(L_V, exclude=[sink_token_mask(plain, 1)])
    excluded, *_ = image_importance(maps, rm, cand_mask=cand)
    assert excluded[2] > plain[2]


def test_cand_mask_excluding_everything_raises():
    with pytest.raises(ValueError):
        image_importance(_synthetic_maps(), _rater_mask(),
                         cand_mask=torch.zeros(L_V, dtype=torch.bool))
