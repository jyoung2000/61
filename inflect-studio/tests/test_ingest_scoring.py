"""Pure VAD scoring + isolation heuristic (no model required)."""

from __future__ import annotations

import numpy as np
import pytest

from inflect.ingest.isolate import should_suggest_isolation, spectral_flatness
from inflect.ingest.vad import (
    ClipCandidate,
    SpeechRegion,
    _coverage_fraction,
    clipping_fraction,
    pick_candidates,
    rms_consistency,
    score_window,
    speech_ratio,
)

SR = 24_000


def _sine(freq, dur_s, amp=0.3, sr=SR):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# VAD scoring primitives
# --------------------------------------------------------------------------- #
def test_rms_consistency_steady_is_high():
    steady = _sine(200, 2.0, amp=0.3)
    assert rms_consistency(steady, SR) > 0.85


def test_rms_consistency_swinging_is_lower():
    t = np.arange(int(2.0 * SR)) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))  # amplitude swings 0..1
    swinging = (env * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    assert rms_consistency(swinging, SR) < rms_consistency(_sine(200, 2.0), SR)


def test_clipping_fraction():
    clean = _sine(200, 1.0, amp=0.5)
    assert clipping_fraction(clean) == 0.0
    clipped = np.clip(_sine(200, 1.0, amp=2.0), -1.0, 1.0)
    assert clipping_fraction(clipped) > 0.3


def test_score_window_prefers_clean_over_clipped():
    clean = _sine(200, 2.0, amp=0.4)
    clipped = np.clip(_sine(200, 2.0, amp=3.0), -1.0, 1.0).astype(np.float32)
    assert score_window(clean, SR, 1.0) > score_window(clipped, SR, 1.0)


def test_score_window_scales_with_speech_fraction():
    seg = _sine(200, 2.0)
    assert score_window(seg, SR, 1.0) > score_window(seg, SR, 0.4)


def test_coverage_fraction():
    regions = [SpeechRegion(0, 5), SpeechRegion(8, 10)]
    assert _coverage_fraction(regions, 0, 10) == pytest.approx(0.7)
    assert _coverage_fraction(regions, 5, 8) == 0.0
    assert _coverage_fraction(regions, 0, 0) == 0.0


def test_speech_ratio():
    regions = [SpeechRegion(0, 3), SpeechRegion(4, 6)]
    assert speech_ratio(regions, 10) == pytest.approx(0.5)
    assert speech_ratio([], 10) == 0.0
    assert speech_ratio(regions, 0) == 0.0


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def test_pick_candidates_returns_nonoverlapping_top_n():
    # 40 s file: clean steady speech in [0,18], quieter/clipped elsewhere.
    sr = SR
    total = 40
    audio = np.zeros(total * sr, dtype=np.float32)
    audio[: 18 * sr] = _sine(200, 18, amp=0.4)
    audio[20 * sr : 40 * sr] = np.clip(_sine(200, 20, amp=3.0), -1, 1)
    regions = [SpeechRegion(0, 18), SpeechRegion(20, 40)]
    cands = pick_candidates(regions, audio, sr, min_s=10, max_s=20, n=3)
    assert 1 <= len(cands) <= 3
    # Non-overlapping.
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            assert a.end_s <= b.start_s or b.end_s <= a.start_s
    # Best candidate should sit in the clean region and be well-scored.
    best = max(cands, key=lambda c: c.score)
    assert best.start_s < 18
    assert best.score > 0.5


def test_pick_candidates_empty_when_no_regions():
    audio = _sine(200, 30)
    assert pick_candidates([], audio, SR) == []


def test_pick_candidates_short_file():
    # Only 6 s available, less than min_s -- still returns something usable.
    audio = _sine(200, 6, amp=0.4)
    regions = [SpeechRegion(0, 6)]
    cands = pick_candidates(regions, audio, SR, min_s=10, max_s=20, n=3)
    assert len(cands) >= 1
    assert cands[0].duration <= 6 + 1e-6


# --------------------------------------------------------------------------- #
# Isolation heuristic
# --------------------------------------------------------------------------- #
def test_spectral_flatness_tone_vs_noise():
    tone = _sine(440, 1.0, amp=0.5)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR).astype(np.float32) * 0.3
    assert spectral_flatness(tone) < 0.05
    assert spectral_flatness(noise) > 0.3


def test_should_suggest_isolation():
    # Low speech ratio -> suggest.
    assert should_suggest_isolation(0.4, 0.05) is True
    # Flat spectrum (music/noise) -> suggest.
    assert should_suggest_isolation(0.9, 0.5) is True
    # Clean, speech-heavy -> don't suggest.
    assert should_suggest_isolation(0.9, 0.05) is False


# --------------------------------------------------------------------------- #
# slice_seconds (pure clip extraction)
# --------------------------------------------------------------------------- #
def test_slice_seconds():
    from inflect.ingest.source import slice_seconds

    audio = _sine(200, 4.0)  # 4 s @ 24k
    clip = slice_seconds(audio, SR, 1.0, 3.0)
    assert len(clip) == 2 * SR
    # Clamped to available range.
    assert len(slice_seconds(audio, SR, 3.5, 99.0)) == int(0.5 * SR)
    # Inverted / empty ranges return empty.
    assert slice_seconds(audio, SR, 2.0, 1.0).size == 0
    assert slice_seconds(audio, SR, 5.0, 6.0).size == 0
