"""Phase 2 audit tests — frame like a human operator.

Covers: L1 deadband/hysteresis, saccade cut-vs-pan on the rebuilt path,
vertical eye-line composition, headroom margins, adaptive tiled trigger
config, export keypoint densification, and the new stability metrics.

All duck-typed (no cv2/models/video) per the repo's test convention.
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import tempfile

import pytest

from backend.config import settings  # noqa: E402
from backend.services import reframer_engine  # noqa: E402
from backend.services.reframer_models import ReframeTracer  # noqa: E402

Engine = reframer_engine.ReframeEngine


def _tracer():
    import os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return ReframeTracer(path)


def _fake_engine(keyframes, face_timeline=None, scene_cuts=None,
                 crop_w=200, max_x=440):
    plan = SimpleNamespace(keyframes=keyframes, crop_w=crop_w, max_x=max_x)
    perception = SimpleNamespace(
        face_timeline=face_timeline or {},
        scene_cuts=scene_cuts or [],
        is_live_action=True,
    )
    return SimpleNamespace(plan=plan, perception=perception,
                           _rdp_indices=Engine._rdp_indices,
                           _apply_deadband=Engine._apply_deadband,
                           tracer=_tracer())


# ── Deadband / hysteresis ───────────────────────────────────────────────

def test_deadband_flattens_micro_drift():
    # ±3px wobble around 100 with a 5px deadband → perfectly flat
    targets = [100.0, 103.0, 98.0, 101.0, 99.0, 102.0]
    out = Engine._apply_deadband(targets, 5.0)
    assert out == [100.0] * len(targets)


def test_deadband_passes_decisive_moves():
    targets = [100.0, 100.0, 180.0, 180.0, 181.0]
    out = Engine._apply_deadband(targets, 5.0)
    # The 80px move re-anchors; the trailing 1px wobble snaps to 180.
    assert out == [100.0, 100.0, 180.0, 180.0, 180.0]


def test_deadband_zero_is_identity():
    targets = [1.0, 2.0, 3.0]
    assert Engine._apply_deadband(targets, 0.0) is targets


def test_l1_deadband_makes_jittery_subject_a_static_hold():
    prev_flag = settings.REFRAMER_L1_PATH
    settings.REFRAMER_L1_PATH = True
    try:
        # Jittery subject: ±4px around 220 (2% of crop_w=200 → inside the
        # default 2.5% deadband). Samples every 200ms.
        kfs, face_timeline = [], {}
        import random
        rng = random.Random(42)
        for i in range(30):
            t = i * 200
            x = 220 + rng.choice([-4, -2, 0, 2, 4])
            kfs.append({"time_ms": t, "x": x,
                        "transition": "linear", "transition_ms": 100})
            face_timeline[t] = []
        eng = _fake_engine(kfs, face_timeline=face_timeline)
        Engine._apply_l1_camera_path(eng)
        xs = [kf["x"] for kf in eng.plan.keyframes]
        assert max(xs) - min(xs) <= 2, f"hold not static: {xs}"
    finally:
        settings.REFRAMER_L1_PATH = prev_flag


# ── Saccade cut-vs-pan on the rebuilt path ──────────────────────────────

def test_l1_rebuild_converts_large_fast_move_into_cut():
    prev_flag = settings.REFRAMER_L1_PATH
    settings.REFRAMER_L1_PATH = True
    try:
        # Hold at 50 then a hard jump to 400 (175% of crop_w=200 — way past
        # the 38% saccade threshold) within one 200ms sample.
        kfs, face_timeline = [], {}
        for i in range(30):
            t = i * 200
            x = 50 if i < 15 else 400
            kfs.append({"time_ms": t, "x": x,
                        "transition": "linear", "transition_ms": 100})
            face_timeline[t] = []
        eng = _fake_engine(kfs, face_timeline=face_timeline)
        Engine._apply_l1_camera_path(eng)
        out = eng.plan.keyframes
        cuts = [kf for kf in out[1:] if kf["transition"] == "cut"]
        assert cuts, (
            "large fast displacement should be emitted as a cut, got: "
            f"{[(k['x'], k['transition']) for k in out]}")
    finally:
        settings.REFRAMER_L1_PATH = prev_flag


# ── Vertical eye-line composition ───────────────────────────────────────

def _make_planner(src_w=1920, src_h=1080, ar_w=16, ar_h=9, faces=None):
    from backend.services.reframer_planner import Planner
    face_timeline = faces or {}
    perc = SimpleNamespace(
        src_w=src_w, src_h=src_h, fps=30.0, duration_ms=10000,
        face_timeline=face_timeline, is_live_action=True,
        speech_active=[], scene_cuts=[], person_timeline={},
    )
    return Planner(perc, ar_w=ar_w, ar_h=ar_h)


def test_vertical_eyeline_places_subject_at_upper_third():
    # 1440x1080 (4:3) source → 16:9 target crops HEIGHT (crop_h < src_h).
    # Faces near the top of frame (eye line ~300px).
    faces = {}
    for i in range(10):
        faces[i * 1000] = [{
            "x": 600, "y": 230, "w": 200, "h": 200, "cx": 700,
            "confidence": 0.9, "area": 40000,
        }]
    p = _make_planner(src_w=1440, src_h=1080, ar_w=16, ar_h=9, faces=faces)
    # eye_y = 230 + 0.35*200 = 300 → crop_y ≈ 300 - crop_h/3
    assert p.crop_h < 1080
    expected = int(round(300 - p.crop_h / 3))
    assert abs(p.crop_y - max(0, expected)) <= 1
    # And never blindly centered
    assert p.crop_y != max(0, (1080 - p.crop_h) // 2)


def test_vertical_eyeline_clamps_to_frame():
    # Face at the very top — crop_y must clamp to 0, not go negative.
    faces = {i * 1000: [{"x": 600, "y": 0, "w": 100, "h": 100, "cx": 650,
                         "confidence": 0.9, "area": 10000}]
             for i in range(5)}
    p = _make_planner(src_w=1440, src_h=1080, ar_w=16, ar_h=9, faces=faces)
    assert p.crop_y == 0


def test_vertical_eyeline_off_when_full_height():
    # 16:9 → 9:16 uses full height; crop_y must stay 0.
    faces = {0: [{"x": 600, "y": 500, "w": 100, "h": 100, "cx": 650,
                  "confidence": 0.9, "area": 10000}]}
    p = _make_planner(src_w=1920, src_h=1080, ar_w=9, ar_h=16, faces=faces)
    assert p.crop_y == 0


def test_vertical_eyeline_flag_off_restores_centering():
    prev = settings.REFRAMER_VERTICAL_EYELINE
    settings.REFRAMER_VERTICAL_EYELINE = False
    try:
        faces = {i * 1000: [{"x": 600, "y": 100, "w": 100, "h": 100,
                             "cx": 650, "confidence": 0.9, "area": 10000}]
                 for i in range(5)}
        p = _make_planner(src_w=1440, src_h=1080, ar_w=16, ar_h=9, faces=faces)
        assert p.crop_y == max(0, (1080 - p.crop_h) // 2)
    finally:
        settings.REFRAMER_VERTICAL_EYELINE = prev


def test_headroom_margin_floor():
    p = _make_planner()
    frac = float(getattr(settings, "REFRAMER_HEADROOM_MIN_FRAC", 0.08))
    assert p._min_edge_margin() == int(p.crop_h * frac)


# ── Export keypoint densification ───────────────────────────────────────

def test_prune_collinear_collapses_linear_ramp():
    from backend.services.reframer_bridge import _prune_collinear_keypoints
    # Perfectly linear ramp sampled at 10 Hz → endpoints only
    kp = {round(i * 0.1, 4): (100.0 + i * 10.0, 1.0) for i in range(21)}
    out = _prune_collinear_keypoints(kp)
    assert len(out) == 2
    assert min(out) == 0.0 and max(out) == 2.0


def test_prune_collinear_keeps_eased_curvature():
    from backend.services.reframer_bridge import _prune_collinear_keypoints
    # Quadratic (eased) motion — interior samples deviate from the chord
    kp = {round(i * 0.1, 4): (100.0 + (i * 0.1) ** 2 * 100.0, 1.0)
          for i in range(21)}
    out = _prune_collinear_keypoints(kp)
    assert len(out) > 5, "eased curve must keep dense samples"


# ── Stability metrics ───────────────────────────────────────────────────

def _run_evaluator(keyframes, duration_s=20, crop_w=608):
    from backend.services.reframe_evaluator import ReframeEvaluator
    plan = SimpleNamespace(
        keyframes=keyframes, crop_w=crop_w, crop_h=1080,
        duration_ms=duration_s * 1000, max_x=1920 - crop_w,
    )
    perc = SimpleNamespace(face_timeline={}, duration_ms=duration_s * 1000)
    return ReframeEvaluator(plan, perc).run()


def test_metrics_static_hold_is_calm():
    kfs = [{"time_ms": 0, "x": 300, "transition": "cut", "transition_ms": 0}]
    r = _run_evaluator(kfs)
    assert r.hold_ratio_pct == 100.0
    assert r.jerk_integral == 0.0
    assert r.cuts_per_minute == 0.0
    assert r.safe_area_pct == 100.0  # no faces → vacuous credit


def test_metrics_counts_cuts_per_minute():
    kfs = [
        {"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0},
        {"time_ms": 5000, "x": 500, "transition": "cut", "transition_ms": 0},
        {"time_ms": 10000, "x": 200, "transition": "cut", "transition_ms": 0},
    ]
    r = _run_evaluator(kfs, duration_s=60)
    assert r.cuts_per_minute == pytest.approx(2.0)
    # Cut discontinuities are excluded from jerk — a pure cut plan is smooth
    assert r.jerk_integral < 1.0


def test_metrics_pan_lowers_hold_ratio():
    # 10s constant-velocity pan across most of the frame then 10s hold
    kfs = [
        {"time_ms": 0, "x": 0, "transition": "cut", "transition_ms": 0},
        {"time_ms": 10000, "x": 1200, "transition": "linear",
         "transition_ms": 10000},
    ]
    r = _run_evaluator(kfs)
    assert r.hold_ratio_pct < 80.0
    r_hold = _run_evaluator(
        [{"time_ms": 0, "x": 300, "transition": "cut", "transition_ms": 0}])
    assert r_hold.hold_ratio_pct > r.hold_ratio_pct


# ── Flag defaults ───────────────────────────────────────────────────────

def test_phase2_flag_defaults():
    assert settings.REFRAMER_L1_DEADBAND_FRAC == pytest.approx(0.025)
    assert settings.REFRAMER_SACCADE_CUT_FRAC == pytest.approx(0.38)
    assert settings.REFRAMER_VERTICAL_EYELINE is True
    assert settings.REFRAMER_HEADROOM_MIN_FRAC == pytest.approx(0.08)
    assert settings.REFRAMER_EXPORT_KEYPOINT_HZ == pytest.approx(10.0)
    assert settings.REFRAMER_TILED_ADAPTIVE is True
    assert settings.REFRAMER_TILED_MIN_FACE_FRAC == pytest.approx(0.04)


# ── Conversation cadence: no tennis-match camera ───────────────────────────
# A measured 39-second two-person scene carried 38 keyframes — the speaker
# follow swung A→B→A→B every second. Dejitter only removes one quick bounce;
# a sustained rally survived every pass. Humans hold a two-shot for close
# subjects and hold each side for seconds when they are far apart.

from backend.services.reframer_models import RenderPlan
from backend.services.reframer_smoother import Smoother


def _plan(kfs, crop_w=600, source_w=1920):
    p = RenderPlan(source_width=source_w, crop_w=crop_w)
    p.keyframes = kfs
    return p


def _kf(t_ms, x, transition="ease_in_out"):
    return {"time_ms": int(t_ms), "x": int(x), "transition": transition,
            "transition_ms": 300}


def test_close_pingpong_becomes_a_two_shot_hold():
    # A at x=200, B at x=380 (separation 180 < 0.45×600), swap every 1.6s —
    # slow enough that dejitter's 3-second A→B→A window doesn't claim it
    # (dejitter's answer to a fast rally is to plant the camera on ONE
    # speaker; the cadence pass frames BOTH), fast enough that no leg earns
    # its 2.5-second operator hold.
    kfs = [_kf(0, 200, "cut")]
    t = 1600
    for i in range(8):
        kfs.append(_kf(t, 380 if i % 2 == 0 else 200))
        t += 1600
    out = Smoother(max_vel_px_per_sec=5000).smooth(_plan(list(kfs))).keyframes
    span = [k for k in out if 0 < k["time_ms"] <= t]
    # The rally collapses to (at most) a single move to the midpoint.
    assert len(span) <= 2, span
    assert any(abs(k["x"] - 290) <= 40 for k in span), span


def test_far_pingpong_is_thinned_to_operator_cadence():
    # A at x=0, B at x=560 — far beyond a two-shot. Swap every 1.0s for 10s.
    kfs = [_kf(0, 0, "cut")]
    t = 1000
    for i in range(10):
        kfs.append(_kf(t, 560 if i % 2 == 0 else 0))
        t += 1000
    out = Smoother(max_vel_px_per_sec=50000).smooth(_plan(list(kfs))).keyframes
    moves = [k for k in out if 0 < k["time_ms"] <= t]
    # ≥2.5s dwell per swing → at most ~4 moves survive the 10s rally.
    assert len(moves) <= 4, moves
    for a, b in zip(moves, moves[1:]):
        assert b["time_ms"] - a["time_ms"] >= 2000, (a, b)


def test_ordinary_pan_sequence_is_untouched_by_cadence():
    # Distinct forward pans with real dwell — not a rally. The cadence pass
    # must leave them alone (anticipation may shift times slightly earlier).
    kfs = [_kf(0, 100, "cut"), _kf(4000, 300), _kf(9000, 500), _kf(15000, 350)]
    out = Smoother(max_vel_px_per_sec=5000).smooth(_plan(list(kfs))).keyframes
    xs = [k["x"] for k in out]
    assert xs == [100, 300, 500, 350], out


def test_anticipation_leads_the_detection():
    from backend.config import settings
    kfs = [_kf(0, 100, "cut"), _kf(5000, 400)]
    out = Smoother(max_vel_px_per_sec=5000).smooth(_plan(list(kfs))).keyframes
    lead = int(getattr(settings, "REFRAMER_ANTICIPATE_MS", 180))
    assert out[-1]["time_ms"] == 5000 - lead, out
