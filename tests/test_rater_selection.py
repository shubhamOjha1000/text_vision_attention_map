"""
Tests for rater_selection.select_important_text_tokens.

The invariant `check_*` functions take a `RaterCase` and are reused by BOTH:
  * pytest (against a synthetic map, always runs), and
  * colab_test_rater_smolvlm.ipynb (against a real SmolVLM2 attention map).

Invariants (the two you proposed, plus more):
  1. importance is a per-text-token column vector: rows == #text tokens, col == 1
  2. thresholding reduces the count: #kept < #content
  3. every rater is a content token (rater_mask subset of content_mask)
  4. no structural/special token is ever selected
  5. exact top-k count: #kept == max(1, L_t' - floor(pct * L_t'))
  6. importance is a valid distribution over content (>=0, sums to 1, 0 elsewhere)
  7. kept tokens are exactly the top-#kept by importance
  8. always >= 1 rater
  9. pct is monotonic: higher pct -> fewer (or equal) kept
 10. the gaze vision-weight hook yields a valid distribution and same #kept
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rater_selection import (  # noqa: E402
    RaterResult,
    content_text_mask,
    default_band,
    is_structural_token,
    question_span_mask,
    select_important_text_tokens,
)


# a SmolVLM-style chat-template token layout: scaffolding around a question
CHAT_TOKENS = [
    "<|im_start|>", "User", ":",
    "How", "Ġmany", "Ġcats", "Ġare", "Ġin", "Ġthe", "Ġimage", "?",
    "<end_of_utterance>", "Ċ", "Ass", "istant", ":",
]
QUESTION = "How many cats are in the image?"
QUESTION_IDX = [3, 4, 5, 6, 7, 8, 9, 10]   # How..image? (the question span)


# --------------------------------------------------------------------------- #
# A bundle a case + its result, so the checks are model-agnostic
# --------------------------------------------------------------------------- #
@dataclass
class RaterCase:
    maps_per_head: dict          # {layer: [H, L_t, L_v]}
    text_tokens: List[str]
    pct: float
    band: List[int]
    tokenizer: object
    res: RaterResult
    name: str = "case"


def make_case(maps_per_head, text_tokens, *, pct=0.5, band=None,
              tokenizer=None, question=None, name="case") -> RaterCase:
    res = select_important_text_tokens(
        maps_per_head, text_tokens=text_tokens, tokenizer=tokenizer,
        question=question, band=band, pct=pct)
    return RaterCase(maps_per_head, list(text_tokens), pct, res.band,
                     tokenizer, res, name)


# --------------------------------------------------------------------------- #
# Reusable invariant checks
# --------------------------------------------------------------------------- #
def check_importance_is_column_vector(case):
    """#1: importance has one score per text token; as a column it is [L_t, 1]."""
    L_t = len(case.text_tokens)
    imp = case.res.importance
    assert imp.ndim == 1 and imp.shape[0] == L_t, f"importance shape {tuple(imp.shape)} != ({L_t},)"
    assert tuple(imp.view(-1, 1).shape) == (L_t, 1)


def check_thresholding_reduces_count(case):
    """#2: after the threshold, #kept < #content (for pct > 0)."""
    assert case.res.n_kept <= case.res.n_content
    if case.pct > 0:
        assert case.res.n_kept < case.res.n_content, \
            f"kept {case.res.n_kept} not < content {case.res.n_content}"


def check_raters_subset_of_content(case):
    """#3: every selected rater is a content token."""
    m = case.res
    assert torch.equal(m.rater_mask & m.content_mask, m.rater_mask)


def check_no_structural_token_selected(case):
    """#4: structural/special tokens are never selected as raters."""
    for tok, kept in zip(case.text_tokens, case.res.rater_mask.tolist()):
        if kept:
            assert not is_structural_token(tok), f"structural token selected: {tok!r}"


def check_topk_count_formula(case):
    """#5: exact top-k count."""
    Lp = case.res.n_content
    assert case.res.n_kept == max(1, Lp - int(case.pct * Lp))


def check_importance_valid_distribution(case):
    """#6: importance >= 0, sums to 1 over content, 0 on suppressed tokens."""
    imp = case.res.importance
    assert (imp >= 0).all()
    assert abs(imp.sum().item() - 1.0) < 1e-3, f"importance sums to {imp.sum().item()}"
    assert (imp[~case.res.content_mask] == 0).all()


def check_kept_are_topk_by_importance(case):
    """#7: kept tokens are the highest-importance content tokens."""
    imp = case.res.importance
    kept = case.res.rater_mask
    dropped_content = case.res.content_mask & ~kept
    if kept.any() and dropped_content.any():
        assert imp[kept].min().item() >= imp[dropped_content].max().item() - 1e-6


def check_at_least_one_rater(case):
    """#8: never returns an empty rater set."""
    assert case.res.n_kept >= 1


def check_pct_monotonic(case):
    """#9: raising pct never increases the number kept; extreme pct keeps >= 1."""
    counts = []
    for pct in (0.0, 0.25, 0.5, 0.75, 0.99):
        r = select_important_text_tokens(
            case.maps_per_head, text_tokens=case.text_tokens,
            tokenizer=case.tokenizer, band=case.band, pct=pct)
        counts.append(r.n_kept)
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts
    assert counts[-1] >= 1


def check_gaze_hook_valid(case):
    """#10: a gaze vision-weight yields a valid distribution and the same #kept."""
    L_v = next(iter(case.maps_per_head.values())).shape[-1]
    w = torch.zeros(L_v)
    w[0] = 1.0
    r = select_important_text_tokens(
        case.maps_per_head, text_tokens=case.text_tokens, tokenizer=case.tokenizer,
        band=case.band, pct=case.pct, vision_weights=w)
    assert abs(r.importance.sum().item() - 1.0) < 1e-3
    assert r.n_kept == case.res.n_kept


ALL_CHECKS = [
    check_importance_is_column_vector,
    check_thresholding_reduces_count,
    check_raters_subset_of_content,
    check_no_structural_token_selected,
    check_topk_count_formula,
    check_importance_valid_distribution,
    check_kept_are_topk_by_importance,
    check_at_least_one_rater,
    check_pct_monotonic,
    check_gaze_hook_valid,
]


# --------------------------------------------------------------------------- #
# Synthetic case (no model) -> run every invariant check
# --------------------------------------------------------------------------- #
TOKENS = ["<bos>", "▁a", "▁cat", "▁sits", "▁on", "Ċ"]  # Ċ = byte-level newline
CONTENT_IDX = [1, 2, 3, 4]
GROUNDED = {2, 3}


def _synthetic_maps(H=3, L_v=8, layers=(0, 1, 2, 3, 4, 5), seed=0):
    g = torch.Generator().manual_seed(seed)
    L_t = len(TOKENS)
    maps = {}
    for l in layers:
        P = torch.randn(H, L_t, L_v, generator=g) * 0.5
        P[:, 2, :] += 6.0   # cat: grounded
        P[:, 3, :] += 5.0   # sits: grounded
        P[:, 0, :] += 8.0   # <bos> sink -> must be suppressed as structural
        maps[l] = P
    return maps


def _synthetic_case(pct=0.5):
    return make_case(_synthetic_maps(), TOKENS, pct=pct,
                     band=[0, 1, 2, 3, 4, 5], name="synthetic")


@pytest.mark.parametrize("check", ALL_CHECKS, ids=[c.__name__ for c in ALL_CHECKS])
def test_synthetic_invariant(check):
    check(_synthetic_case())


# --------------------------------------------------------------------------- #
# Focused unit tests
# --------------------------------------------------------------------------- #
def test_structural_detection():
    for t in ("<bos>", "<image>", "<|vision_start|>", "Ċ", "▁"):
        assert is_structural_token(t)
    for t in ("▁cat", "cat"):
        assert not is_structural_token(t)


def test_content_mask_suppresses_structural():
    keep = content_text_mask(TOKENS)
    assert keep.nonzero().squeeze(-1).tolist() == CONTENT_IDX


def test_default_band_middle_third():
    assert default_band([0, 1, 2, 3, 4, 5]) == [2, 3]


def test_question_span_mask_isolates_question():
    mask = question_span_mask(CHAT_TOKENS, QUESTION)
    assert mask.nonzero().squeeze(-1).tolist() == QUESTION_IDX
    # role markers / generation prompt are excluded
    for i, tok in enumerate(CHAT_TOKENS):
        if tok in ("User", "Ass", "istant") or (tok == ":" and i not in QUESTION_IDX):
            assert not mask[i]


def test_question_scoping_drops_chat_scaffolding():
    # build a map where 'Assistant:' scaffolding attends strongly to the image,
    # so WITHOUT scoping it would be selected; scoping must exclude it.
    H, L_v = 2, 6
    L_t = len(CHAT_TOKENS)
    g = torch.Generator().manual_seed(0)
    maps = {}
    for l in range(4):
        P = torch.randn(H, L_t, L_v, generator=g) * 0.3
        P[:, 13, :] += 9.0   # 'Ass' scaffolding: huge (would win without scoping)
        P[:, 5, :] += 4.0    # 'cats': grounded question word
        maps[l] = P
    res = select_important_text_tokens(
        maps, text_tokens=CHAT_TOKENS, question=QUESTION, band=[0, 1, 2, 3], pct=0.5)
    kept = res.kept_tokens(CHAT_TOKENS)
    assert "Ass" not in kept and "istant" not in kept and "User" not in kept
    assert "Ġcats" in kept
    # every kept token lies in the question span
    for i in res.rater_mask.nonzero().squeeze(-1).tolist():
        assert i in QUESTION_IDX


def test_grounded_tokens_are_kept_and_bos_suppressed():
    case = _synthetic_case(pct=0.5)
    assert not case.res.rater_mask[0]                     # <bos> not selected
    kept = set(case.res.rater_mask.nonzero().squeeze(-1).tolist())
    assert kept == GROUNDED                               # cat, sits
