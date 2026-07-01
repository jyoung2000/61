"""Trace-coverage guards for the new reframer decision logic.

Ensures the anticipation/smoothing (planner) and saliency-source (perceiver)
telemetry actually reaches the reframe_trace / detection_overlay so an offline
reviewer (Claude Code) can debug reframing decisions. The engine-pass events
(l1_path, savgol_pass, motivated_zoom) are covered in their own suites.
"""

import os
import sys
import tempfile
import types
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings  # noqa: E402
from backend.services.reframer_models import ReframeTracer  # noqa: E402
from backend.services.reframer_planner import Planner  # noqa: E402


def _tracer():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return ReframeTracer(path)


def _planner():
    p = Planner.__new__(Planner)
    p.crop_w = 200
    p.max_x = 440
    p.sample_fps = 5.0
    p.is_live_action = True
    p._euro_filter = None
    p._prev_raw_target = None
    p._prev_raw_target_t = None
    p._filter_trace = {}
    p.tracer = _tracer()
    return p


def test_apply_target_filter_records_telemetry():
    prev = settings.REFRAMER_ONE_EURO_FILTER
    settings.REFRAMER_ONE_EURO_FILTER = True
    try:
        p = _planner()
        # A moving subject so lead-room + velocity are non-zero.
        for i, t in enumerate(range(0, 1000, 200)):
            p._apply_target_filter(100 + i * 20, t, switching=(i == 0),
                                   anchor_yaw=-0.5)
        assert set(p._filter_trace) == {0, 200, 400, 600, 800}
        d = p._filter_trace[600]
        assert set(d) >= {"raw_x", "lead_px", "v_hat", "filtered_x",
                          "filter", "yaw", "switching"}
        assert d["filter"] == "one_euro"
        assert d["yaw"] == -0.5
        # A moving subject with lead-room enabled should carry a non-zero lead.
        assert abs(d["lead_px"]) > 0 and abs(d["v_hat"]) > 0
    finally:
        settings.REFRAMER_ONE_EURO_FILTER = prev


def test_switch_records_cut_filter():
    p = _planner()
    p._apply_target_filter(300, 0, switching=True, anchor_yaw=None)
    assert p._filter_trace[0]["filter"] == "cut"
    assert p._filter_trace[0]["switching"] is True


def test_emit_keyframe_trace_emits_target_filter_events():
    p = _planner()
    # Populate two samples of filter telemetry...
    p._apply_target_filter(100, 0, switching=True, anchor_yaw=None)
    p._apply_target_filter(140, 200, switching=False, anchor_yaw=0.2)
    kfs = [
        {"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0},
        {"time_ms": 200, "x": 140, "transition": "ease_in_out", "transition_ms": 100},
    ]
    p._emit_keyframe_trace(kfs, scene_idx=0, strategy_label="adaptive_face")
    counts = p.tracer.counts
    assert counts.get("keyframe_decided") == 2
    assert counts.get("target_filter") == 2


def test_saliency_source_reaches_detection_overlay():
    from backend.services.reframer_bridge import serialize_detection_overlay
    perception = SimpleNamespace(
        face_timeline={}, person_timeline={}, motion_timeline={},
        speech_active={}, scene_cuts=[], src_w=1920, src_h=1080,
        saliency_hotspot={
            1000: {"cx": 800, "cy": 400, "intensity": 0.42, "source": "u2netp+stack"},
            2000: {"cx": 810, "cy": 402, "intensity": 0.31},  # legacy: no source
        },
    )
    overlay = serialize_detection_overlay(perception)
    sal = overlay["saliency_hotspot"]
    assert sal["1000"]["source"] == "u2netp+stack"
    # Missing source defaults to "spectral" so the field is always present.
    assert sal["2000"]["source"] == "spectral"


def test_ema_path_labels_filter_ema():
    prev = settings.REFRAMER_ONE_EURO_FILTER
    settings.REFRAMER_ONE_EURO_FILTER = False
    try:
        p = _planner()
        p._apply_target_filter(100, 0, switching=False, anchor_yaw=None)
        p._apply_target_filter(120, 200, switching=False, anchor_yaw=None)
        assert p._filter_trace[200]["filter"] == "ema"
    finally:
        settings.REFRAMER_ONE_EURO_FILTER = prev
