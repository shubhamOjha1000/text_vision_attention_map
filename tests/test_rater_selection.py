"""
Tests for rater_selection.select_important_text_tokens.

Model-agnostic: builds a synthetic RAW text->vision map with a known ground truth
(a couple of text tokens attend strongly to the image, the rest don't, plus some
structural tokens), and checks the diagram's behaviour:
  * structural tokens (BOS, "\n", <image>-style) are suppressed before scoring,
  * the visually-grounded content tokens are the ones kept,
  * the kept count equals  L_t' - floor(pct * L_t'),
  * a gaze-style vision weighting shifts which tokens are important.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rater_selection import (  # noqa: E402
    content_text_mask,
    default_band,
    is_structural_token,
    select_important_text_tokens,
)


# tokens: two structural (<bos>, "\n") + four content (a, cat, sits, on)
TOKENS = ["<bos>", "▁a", "▁cat", "▁sits", "▁on", "Ċ"]  # Ċ = byte-level newline
CONTENT_IDX = [1, 2, 3, 4]                               # a, cat, sits, on
GROUNDED = {2, 3}                                        # cat, sits attend to image


def _synthetic_maps(H=3, L_v=8, layers=(0, 1, 2, 3, 4, 5), seed=0):
    """cat/sits get high raw scores to the image; a/on low; structural arbitrary."""
    g = torch.Generator().manual_seed(seed)
    L_t = len(TOKENS)
    maps = {}
    for l in layers:
        P = torch.randn(H, L_t, L_v, generator=g) * 0.5      # low baseline
        P[:, 2, :] += 6.0                                    # cat: strong grounding
        P[:, 3, :] += 5.0                                    # sits: strong grounding
        P[:, 0, :] += 8.0                                    # <bos> sink (must be suppressed)
        maps[l] = P
    return maps


def test_structural_detection():
    assert is_structural_token("<bos>")
    assert is_structural_token("<image>")
    assert is_structural_token("<|vision_start|>")
    assert is_structural_token("Ċ")          # newline
    assert is_structural_token("▁")          # bare space
    assert not is_structural_token("▁cat")
    assert not is_structural_token("cat")


def test_content_mask_suppresses_structural():
    keep = content_text_mask(TOKENS)
    # only a, cat, sits, on survive; <bos> and the newline are suppressed
    assert keep.nonzero().squeeze(-1).tolist() == CONTENT_IDX


def test_default_band_middle_third():
    assert default_band([0, 1, 2, 3, 4, 5]) == [2, 3]     # middle third of 6


def test_grounded_tokens_are_kept_and_bos_suppressed():
    maps = _synthetic_maps()
    res = select_important_text_tokens(
        maps, text_tokens=TOKENS, band=[0, 1, 2, 3, 4, 5], pct=0.5)
    # <bos> is a huge sink but must NOT be selected (suppressed as structural)
    assert not res.rater_mask[0]
    # content count and kept count: L_t' = 4, drop floor(0.5*4)=2 -> keep 2
    assert res.n_content == 4
    assert res.n_kept == 2
    # the two kept must be the grounded content tokens (cat, sits)
    kept = set(res.rater_mask.nonzero().squeeze(-1).tolist())
    assert kept == GROUNDED
    # suppressed tokens carry zero importance
    assert res.importance[0].item() == 0.0
    assert res.importance[5].item() == 0.0


def test_pct_controls_kept_count():
    maps = _synthetic_maps()
    for pct, expect in [(0.0, 4), (0.25, 3), (0.5, 2), (0.75, 1), (0.99, 1)]:
        res = select_important_text_tokens(
            maps, text_tokens=TOKENS, band=[0, 1, 2, 3, 4, 5], pct=pct)
        assert res.n_kept == expect, f"pct={pct}: got {res.n_kept}, want {expect}"


def test_importance_is_a_distribution_over_content():
    maps = _synthetic_maps()
    res = select_important_text_tokens(
        maps, text_tokens=TOKENS, band=[0, 1, 2, 3, 4, 5], pct=0.5)
    # r sums to 1 over the content tokens (mean of column-stochastic distributions)
    total = res.importance.sum().item()
    assert abs(total - 1.0) < 1e-4


def test_gaze_weighting_shifts_importance():
    """A vision weight concentrated on columns where 'on' happens to peak should
    raise its importance -- exercising the gaze hook path without changing shapes."""
    g = torch.Generator().manual_seed(1)
    H, L_v = 2, 6
    L_t = len(TOKENS)
    P = torch.randn(H, L_t, L_v, generator=g) * 0.3
    P[:, 2, :] += 4.0                       # cat grounded everywhere
    P[:, 4, 0] += 8.0                       # 'on' peaks only on column 0
    maps = {0: P, 1: P.clone()}

    uniform = select_important_text_tokens(
        maps, text_tokens=TOKENS, band=[0, 1], pct=0.5)
    w = torch.zeros(L_v); w[0] = 1.0        # gaze fixates column 0
    gazed = select_important_text_tokens(
        maps, text_tokens=TOKENS, band=[0, 1], pct=0.5, vision_weights=w)

    # under column-0 gaze, 'on' (idx 4) importance should rise vs uniform
    assert gazed.importance[4] > uniform.importance[4]
