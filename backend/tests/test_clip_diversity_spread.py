"""Tests for the clip-clustering fixes in select_diverse_clips.

Symptom: with a working judge, all clips bunched into the opening ~2 minutes
(only 2-3 clips, overlapping) instead of spreading across the episode. Two
contributing bugs are fixed:
  1. select_diverse_clips MUTATED candidates' composite_score; it's called twice
     on the same objects, so the second pass saw corrupted scores.
  2. (separately) the judge's trim_suggestion could relocate clips onto the
     opening — guarded in _run_editorial_judge (not covered here).
"""

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframer_clipper import (  # noqa: E402
    select_diverse_clips, ClipCandidate,
)


def _c(start, end, score):
    return ClipCandidate(start_s=float(start), end_s=float(end),
                         duration_s=float(end - start), composite_score=float(score))


def test_does_not_mutate_composite_score():
    cands = [_c(0, 30, 1.0), _c(20, 50, 0.9), _c(100, 130, 0.8)]
    before = [c.composite_score for c in cands]
    select_diverse_clips(cands, 12, 1000, min_gap_s=60)
    assert [c.composite_score for c in cands] == before   # inputs untouched


def test_repeated_calls_are_consistent():
    # The real pipeline calls this twice on the same objects; without the
    # mutation fix the second result drifted. Must be deterministic.
    cands = [_c(i * 100, i * 100 + 30, 1.0 - i * 0.01) for i in range(10)]
    out1 = [c.start_s for c in select_diverse_clips(cands, 5, 1100, min_gap_s=60)]
    out2 = [c.start_s for c in select_diverse_clips(cands, 5, 1100, min_gap_s=60)]
    assert out1 == out2


def test_spread_candidates_are_not_collapsed():
    # 12 candidates spread across ~20 min → all distinct positions selected.
    cands = [_c(i * 100, i * 100 + 30, 1.0 - i * 0.001) for i in range(12)]
    out = select_diverse_clips(cands, 12, 1300, min_gap_s=61)
    assert len(out) == 12
    starts = sorted(c.start_s for c in out)
    assert starts[-1] - starts[0] > 600          # genuinely spread, not bunched


def test_heavily_overlapping_candidates_collapse_to_one():
    cands = [_c(0, 30, 1.0), _c(2, 32, 0.9), _c(4, 34, 0.8)]   # >80% overlap
    out = select_diverse_clips(cands, 12, 100, min_gap_s=60)
    assert len(out) == 1


def test_picks_up_to_max_clips_from_large_spread_pool():
    cands = [_c(i * 80, i * 80 + 40, 1.0 - i * 0.0005) for i in range(29)]
    out = select_diverse_clips(cands, 12, 1467, min_gap_s=61)
    assert len(out) == 12                         # honours eff_max, not just a few
