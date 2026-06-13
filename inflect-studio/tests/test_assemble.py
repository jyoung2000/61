"""Assembly math: crossfade, pause insertion, normalization, resampling."""

from __future__ import annotations

import numpy as np
import pytest

from inflect.synth.assemble import (
    assemble_segments,
    equal_power_crossfade,
    master,
    normalize_loudness,
    resample,
    silence,
    true_peak_limit,
)

SR = 24_000


def _sine(freq: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# silence
# --------------------------------------------------------------------------- #
def test_silence_length():
    s = silence(100, SR)  # 100 ms
    assert len(s) == SR // 10
    assert np.all(s == 0)
    assert s.dtype == np.float32


def test_silence_zero_and_negative():
    assert len(silence(0, SR)) == 0
    assert len(silence(-50, SR)) == 0


# --------------------------------------------------------------------------- #
# equal-power crossfade
# --------------------------------------------------------------------------- #
def test_crossfade_length():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    out = equal_power_crossfade(a, b, 100)
    assert len(out) == 1000 + 1000 - 100


def test_crossfade_preserves_endpoints():
    a = np.full(500, 0.3, dtype=np.float32)
    b = np.full(500, 0.7, dtype=np.float32)
    out = equal_power_crossfade(a, b, 50)
    assert out[0] == pytest.approx(0.3, abs=1e-5)
    assert out[-1] == pytest.approx(0.7, abs=1e-5)
    assert not np.any(np.isnan(out))


def test_crossfade_equal_power_property():
    # For two identical constant signals the equal-power sum stays within
    # [1, sqrt(2)] across the overlap (never dips, characteristic of equal power).
    a = np.ones(400, dtype=np.float32)
    b = np.ones(400, dtype=np.float32)
    out = equal_power_crossfade(a, b, 200)
    overlap = out[200:400]
    assert overlap.min() >= 1.0 - 1e-4
    assert overlap.max() <= np.sqrt(2.0) + 1e-4


def test_crossfade_clamped_when_short():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32)
    out = equal_power_crossfade(a, b, 100)  # n clamped to 10
    assert len(out) == 10 + 1000 - 10


def test_crossfade_zero_overlap_is_concat():
    a = np.arange(5, dtype=np.float32)
    b = np.arange(5, dtype=np.float32)
    out = equal_power_crossfade(a, b, 0)
    assert np.array_equal(out, np.concatenate([a, b]))


# --------------------------------------------------------------------------- #
# assemble_segments
# --------------------------------------------------------------------------- #
def test_assemble_empty():
    assert len(assemble_segments([], [], SR)) == 0


def test_assemble_single_segment_passthrough():
    seg = _sine(200, 0.2)
    out = assemble_segments([seg], [0], SR)
    assert len(out) == len(seg)


def test_assemble_crossfade_reduces_length():
    a = _sine(200, 0.3)
    b = _sine(300, 0.3)
    cf_ms = 15
    out = assemble_segments([a, b], [0, 0], SR, crossfade_ms=cf_ms)
    cf_n = round(cf_ms * SR / 1000)
    assert len(out) == len(a) + len(b) - cf_n


def test_assemble_pause_inserts_silence_and_skips_crossfade():
    a = _sine(200, 0.3)
    b = _sine(300, 0.3)
    pause_ms = 250
    out = assemble_segments([a, b], [pause_ms, 0], SR, crossfade_ms=15)
    pause_n = round(pause_ms * SR / 1000)
    assert len(out) == len(a) + pause_n + len(b)
    # The silence really is silent in the middle.
    mid = out[len(a) + pause_n // 2]
    assert abs(mid) < 1e-6


def test_assemble_skips_empty_segments():
    a = _sine(200, 0.2)
    empty = np.zeros(0, dtype=np.float32)
    b = _sine(300, 0.2)
    out = assemble_segments([a, empty, b], [0, 0, 0], SR, crossfade_ms=15)
    cf_n = round(15 * SR / 1000)
    assert len(out) == len(a) + len(b) - cf_n


def test_assemble_trailing_pause_appended():
    a = _sine(200, 0.2)
    pause_ms = 100
    out = assemble_segments([a], [pause_ms], SR)
    assert len(out) == len(a) + round(pause_ms * SR / 1000)


# --------------------------------------------------------------------------- #
# loudness / peak
# --------------------------------------------------------------------------- #
def test_normalize_loudness_hits_target():
    # A quiet 1 kHz tone normalized to -16 LUFS should measure close to it.
    quiet = _sine(1000, 3.0, amp=0.02)
    out = normalize_loudness(quiet, SR, target_lufs=-16.0)
    import pyloudnorm as pyln

    measured = pyln.Meter(SR).integrated_loudness(out.astype(np.float64))
    assert measured == pytest.approx(-16.0, abs=0.6)


def test_normalize_loudness_silence_is_noop():
    s = silence(1000, SR)
    out = normalize_loudness(s, SR)
    assert np.array_equal(out, s)


def test_true_peak_limit_attenuates():
    hot = _sine(440, 1.0, amp=0.99)
    out = true_peak_limit(hot, dbtp=-1.0)
    limit = 10 ** (-1.0 / 20.0)
    assert np.max(np.abs(out)) <= limit + 1e-4


def test_true_peak_limit_does_not_boost():
    quiet = _sine(440, 1.0, amp=0.1)
    out = true_peak_limit(quiet, dbtp=-1.0)
    assert np.max(np.abs(out)) == pytest.approx(0.1, abs=1e-4)


def test_master_keeps_peak_under_limit():
    loud = _sine(440, 2.0, amp=0.9)
    out = master(loud, SR, target_lufs=-16.0, true_peak_dbtp=-1.0)
    assert np.max(np.abs(out)) <= 10 ** (-1.0 / 20.0) + 1e-3


# --------------------------------------------------------------------------- #
# resample
# --------------------------------------------------------------------------- #
def test_resample_changes_length_proportionally():
    a = _sine(440, 1.0, sr=24000)
    out = resample(a, 24000, 48000)
    assert abs(len(out) - 2 * len(a)) <= 2


def test_resample_noop_when_same_rate():
    a = _sine(440, 0.5)
    out = resample(a, SR, SR)
    assert np.array_equal(out, a)


def test_resample_down_then_up_preserves_frequency():
    # A 1 kHz tone resampled 24k->16k->24k should still be ~1 kHz dominant.
    a = _sine(1000, 1.0, sr=24000)
    down = resample(a, 24000, 16000)
    up = resample(down, 16000, 24000)
    n = min(len(a), len(up))
    spec = np.abs(np.fft.rfft(up[:n]))
    freqs = np.fft.rfftfreq(n, 1 / 24000)
    peak_freq = freqs[int(np.argmax(spec))]
    assert peak_freq == pytest.approx(1000, abs=30)
