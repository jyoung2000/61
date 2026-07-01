"""Unit tests for the L1-optimal camera path solver."""

import numpy as np

from backend.services.reframer_l1_camera_path import parse_weights, solve_l1_path


def test_short_input_clamps_to_bounds():
    out = solve_l1_path([5.0, -3.0, 500.0], radius=100.0, lo=0.0, hi=100.0)
    assert out == [5.0, 0.0, 100.0]


def test_constant_target_is_a_perfect_hold():
    p = [50.0] * 40
    out = solve_l1_path(p, radius=30.0, lo=0.0, hi=200.0)
    assert out is not None
    assert all(abs(v - 50.0) < 1e-4 for v in out)


def test_jittery_static_subject_becomes_flat_hold():
    rng = np.random.default_rng(0)
    p = 100.0 + rng.normal(0, 8, size=60)
    out = solve_l1_path(p, radius=25.0, lo=0.0, hi=400.0)
    assert out is not None
    out = np.asarray(out)
    # The L1 path should be far steadier than the noisy input.
    assert np.std(np.diff(out)) < np.std(np.diff(p)) * 0.3
    # And stay within the proximity band of the (clipped) targets.
    assert np.all(np.abs(out - np.clip(p, 0, 400)) <= 25.0 + 1e-3)


def test_step_produces_holds_and_a_single_ramp():
    # A subject that holds, jumps, then holds. L1 should give hold-ramp-hold
    # with a sparse acceleration profile (few nonzero second derivatives).
    p = [20.0] * 20 + [180.0] * 20
    out = solve_l1_path(p, radius=15.0, lo=0.0, hi=300.0, weights=(1.0, 10.0, 100.0))
    assert out is not None
    out = np.asarray(out)
    accel = np.abs(np.diff(out, 2))
    # Sparse acceleration: only a handful of samples should accelerate.
    assert np.count_nonzero(accel > 1.0) <= 8
    # Endpoints respect the two hold levels within the band.
    assert out[0] <= 35.0 and out[-1] >= 165.0


def test_respects_hard_bounds():
    p = [-100.0, 500.0] * 25
    out = solve_l1_path(p, radius=1000.0, lo=0.0, hi=100.0)
    assert out is not None
    assert all(0.0 - 1e-6 <= v <= 100.0 + 1e-6 for v in out)


def test_parse_weights():
    assert parse_weights("1,10,100") == (1.0, 10.0, 100.0)
    assert parse_weights("bad") == (1.0, 10.0, 100.0)
    assert parse_weights("2,3,4", default=(1.0, 1.0, 1.0)) == (2.0, 3.0, 4.0)
