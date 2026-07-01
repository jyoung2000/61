"""Unit tests for the One-Euro filter (reframer target-x smoothing)."""

import math

from backend.services.reframer_oneeuro import OneEuroFilter, _alpha


def test_alpha_monotonic_in_cutoff():
    # Higher cutoff at fixed dt => less smoothing => alpha closer to 1.
    dt = 1.0 / 30.0
    lo = _alpha(0.5, dt)
    hi = _alpha(5.0, dt)
    assert 0.0 < lo < hi < 1.0


def test_first_sample_passes_through():
    f = OneEuroFilter(freq=30.0)
    assert f(100.0, t=0.0) == 100.0
    assert f.initialized


def test_constant_signal_converges_to_value():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.02)
    t = 0.0
    y = 0.0
    for _ in range(120):
        t += 1.0 / 30.0
        y = f(50.0, t=t)
    assert abs(y - 50.0) < 1e-3


def test_smooths_jitter_around_static_subject():
    # A still subject with +/-5px sensor jitter should be smoothed hard.
    f = OneEuroFilter(freq=30.0, mincutoff=0.6, beta=0.01)
    t = 0.0
    outs = []
    jitter = [5, -5, 4, -6, 5, -4, 6, -5, 4, -5] * 6
    for j in jitter:
        t += 1.0 / 30.0
        outs.append(f(100.0 + j, t=t))
    tail = outs[-20:]
    spread = max(tail) - min(tail)
    assert spread < 5.0  # jitter (~11px p2p) is substantially attenuated


def test_tracks_fast_move_with_low_lag():
    # On a fast ramp the adaptive cutoff should keep lag small vs a heavy EMA.
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.05)
    t = 0.0
    pos = 0.0
    last = 0.0
    for _ in range(30):
        t += 1.0 / 30.0
        pos += 20.0  # 600 px/s ramp
        last = f(pos, t=t)
    # Lag = true - filtered; adaptive filter should trail by well under one
    # heavy-EMA time-constant worth of samples.
    lag = pos - last
    assert lag < 60.0


def test_reset_snaps_to_new_value():
    f = OneEuroFilter(freq=30.0)
    for i in range(30):
        f(0.0, t=i / 30.0)
    f.reset()
    # After reset the next sample is taken as-is (a cut / speaker switch).
    assert f(500.0, t=2.0) == 500.0


def test_nonpositive_dt_is_safe():
    f = OneEuroFilter(freq=30.0)
    f(10.0, t=1.0)
    # Duplicate / out-of-order timestamp must not divide by zero.
    out = f(20.0, t=1.0)
    assert math.isfinite(out)
