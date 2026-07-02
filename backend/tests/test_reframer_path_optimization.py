"""Unit tests for the offline path-optimization passes (savgol + L1).

These exercise the pure logic of ReframeEngine._smooth_trajectory_savgol and
._apply_l1_camera_path against duck-typed plan/perception objects, so they run
without OpenCV, models or video fixtures.
"""

import sys
import types
from types import SimpleNamespace

# Stub cv2 before importing the engine (matches the repo's test convention).
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import tempfile

from backend.config import settings  # noqa: E402
from backend.services import reframer_engine  # noqa: E402
from backend.services.reframer_models import ReframeTracer  # noqa: E402

Engine = reframer_engine.ReframeEngine


def _tracer():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    import os
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
    # Attach the static helper the L1 pass calls via ``self`` + a real tracer
    # so the tests can assert the new instrumentation fires.
    return SimpleNamespace(plan=plan, perception=perception,
                           _rdp_indices=Engine._rdp_indices,
                           _apply_deadband=Engine._apply_deadband,
                           tracer=_tracer())


def test_savgol_reduces_jitter_and_preserves_cut_and_centering():
    prev = settings.REFRAMER_SAVGOL_SMOOTHING
    settings.REFRAMER_SAVGOL_SMOOTHING = True
    try:
        # A steady run with alternating +/-15px jitter, plus a cut keyframe and
        # a _centering keyframe that must be preserved exactly.
        base = 200
        kfs = []
        for i in range(21):
            j = 15 if i % 2 == 0 else -15
            kf = {"time_ms": i * 200, "x": base + j,
                  "transition": "linear", "transition_ms": 100}
            kfs.append(kf)
        kfs[10]["_centering"] = True
        kfs[10]["x"] = 320  # must survive
        # Insert a hard cut mid-way (its x must survive).
        kfs[15]["transition"] = "cut"
        kfs[15]["x"] = 90
        eng = _fake_engine(kfs)
        Engine._smooth_trajectory_savgol(eng)

        assert kfs[10]["x"] == 320           # _centering preserved
        assert kfs[15]["x"] == 90            # cut preserved
        # Jitter on a plain run keyframe is attenuated toward the mean.
        assert abs(kfs[4]["x"] - base) < 15
        # Instrumentation fired (trace coverage guard).
        assert eng.tracer.counts.get("savgol_pass") == 1
    finally:
        settings.REFRAMER_SAVGOL_SMOOTHING = prev


def test_savgol_noop_when_disabled():
    prev = settings.REFRAMER_SAVGOL_SMOOTHING
    settings.REFRAMER_SAVGOL_SMOOTHING = False
    try:
        kfs = [{"time_ms": i * 200, "x": 200 + (15 if i % 2 else -15),
                "transition": "linear", "transition_ms": 100} for i in range(21)]
        snapshot = [k["x"] for k in kfs]
        eng = _fake_engine(kfs)
        Engine._smooth_trajectory_savgol(eng)
        assert [k["x"] for k in kfs] == snapshot
    finally:
        settings.REFRAMER_SAVGOL_SMOOTHING = prev


def test_l1_rebuilds_holds_and_pan():
    prev = settings.REFRAMER_L1_PATH
    settings.REFRAMER_L1_PATH = True
    try:
        # Keyframes describing hold@50 → move → hold@300 over 40 samples.
        sample_times = [i * 200 for i in range(40)]
        face_timeline = {t: [] for t in sample_times}
        kfs = [
            {"time_ms": 0, "x": 50, "transition": "cut", "transition_ms": 0},
            {"time_ms": 20 * 200, "x": 50, "transition": "linear", "transition_ms": 400},
            {"time_ms": 25 * 200, "x": 300, "transition": "linear", "transition_ms": 1000},
            {"time_ms": 39 * 200, "x": 300, "transition": "linear", "transition_ms": 400},
        ]
        eng = _fake_engine(kfs, face_timeline=face_timeline, crop_w=200, max_x=440)
        Engine._apply_l1_camera_path(eng)
        out = eng.plan.keyframes
        assert len(out) >= 2
        # Monotonic in time, within bounds, starts near 50 ends near 300.
        assert out == sorted(out, key=lambda k: k["time_ms"])
        assert all(0 <= k["x"] <= 440 for k in out)
        assert out[0]["x"] <= 90
        assert out[-1]["x"] >= 260
        # Rebuilt path should be sparse (holds+pan), not one-per-sample.
        assert len(out) < len(sample_times)
        # Instrumentation fired (trace coverage guard).
        assert eng.tracer.counts.get("l1_path") == 1
    finally:
        settings.REFRAMER_L1_PATH = prev


def test_l1_noop_when_disabled():
    prev = settings.REFRAMER_L1_PATH
    settings.REFRAMER_L1_PATH = False
    try:
        kfs = [{"time_ms": 0, "x": 50, "transition": "cut", "transition_ms": 0},
               {"time_ms": 2000, "x": 300, "transition": "linear", "transition_ms": 400}]
        eng = _fake_engine(kfs, face_timeline={0: [], 2000: []})
        Engine._apply_l1_camera_path(eng)
        assert eng.plan.keyframes is kfs
    finally:
        settings.REFRAMER_L1_PATH = prev
