"""Unit tests for motivated zoom (item 10): scale contract, planner stamping,
and the zoom-aware export crop builder."""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import os
import tempfile

from backend.config import settings  # noqa: E402
from backend.services.reframer_models import interpolate_scale, ReframeTracer  # noqa: E402
from backend.services import reframer_engine  # noqa: E402
from backend.services.clip_exporter import (  # noqa: E402
    build_zoom_crop_filter, _piecewise_time_expr,
)

Engine = reframer_engine.ReframeEngine


# ── interpolate_scale (the shared contract) ──

def test_interpolate_scale_defaults_to_one_without_scale_keys():
    kfs = [{"time_ms": 0, "x": 0, "transition": "cut", "transition_ms": 0},
           {"time_ms": 1000, "x": 0, "transition": "linear", "transition_ms": 0}]
    assert interpolate_scale(kfs, 500) == 1.0


def test_interpolate_scale_eases_between_scale_keyframes():
    kfs = [
        {"time_ms": 0, "x": 0, "transition": "cut", "transition_ms": 0, "scale": 1.0},
        {"time_ms": 1000, "x": 0, "transition": "ease_in_out", "transition_ms": 0, "scale": 1.2},
    ]
    assert interpolate_scale(kfs, 0) == 1.0
    assert interpolate_scale(kfs, 1000) == 1.2
    mid = interpolate_scale(kfs, 500)
    assert 1.0 < mid < 1.2


def test_interpolate_scale_holds_across_cut():
    kfs = [
        {"time_ms": 0, "x": 0, "transition": "cut", "transition_ms": 0, "scale": 1.15},
        {"time_ms": 1000, "x": 0, "transition": "cut", "transition_ms": 0, "scale": 1.0},
    ]
    assert interpolate_scale(kfs, 500) == 1.15  # holds prev until the cut time


# ── engine motivated-zoom stamping ──

def _tracer():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return ReframeTracer(path)


def _fake_engine(keyframes, face_timeline, crop_w=200):
    plan = SimpleNamespace(keyframes=keyframes, crop_w=crop_w, max_x=440)
    perception = SimpleNamespace(face_timeline=face_timeline, scene_cuts=[])
    return SimpleNamespace(plan=plan, perception=perception, tracer=_tracer())


def test_motivated_zoom_stamps_pushin_on_held_speaker():
    prev = settings.REFRAMER_MOTIVATED_ZOOM
    prev_max = settings.REFRAMER_MOTIVATED_ZOOM_MAX
    settings.REFRAMER_MOTIVATED_ZOOM = True
    settings.REFRAMER_MOTIVATED_ZOOM_MAX = 1.15
    try:
        # A 6-second hold at x=100 with a face present throughout.
        kfs = [
            {"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0},
            {"time_ms": 6000, "x": 100, "transition": "linear", "transition_ms": 0},
        ]
        face_timeline = {t: [{"cx": 100}] for t in range(0, 6001, 500)}
        eng = _fake_engine(kfs, face_timeline)
        Engine._apply_motivated_zoom(eng)
        scales = [k.get("scale") for k in eng.plan.keyframes if k.get("_zoom")]
        assert scales, "expected zoom keyframes to be inserted"
        assert max(scales) == 1.15
        assert min(scales) == 1.0  # ramps back out
        # Peak scale should be reached somewhere in the middle of the hold.
        peak_kf = max((k for k in eng.plan.keyframes if k.get("_zoom")),
                      key=lambda k: k["scale"])
        assert 0 < peak_kf["time_ms"] < 6000
        # Instrumentation fired (trace coverage guard).
        assert eng.tracer.counts.get("motivated_zoom") == 1
    finally:
        settings.REFRAMER_MOTIVATED_ZOOM = prev
        settings.REFRAMER_MOTIVATED_ZOOM_MAX = prev_max


def test_motivated_zoom_skips_pans_and_static_graphics():
    prev = settings.REFRAMER_MOTIVATED_ZOOM
    settings.REFRAMER_MOTIVATED_ZOOM = True
    try:
        # A long span but it's a PAN (x moves a lot) → no zoom.
        kfs = [
            {"time_ms": 0, "x": 20, "transition": "cut", "transition_ms": 0},
            {"time_ms": 6000, "x": 400, "transition": "linear", "transition_ms": 0},
        ]
        face_timeline = {t: [{"cx": 100}] for t in range(0, 6001, 500)}
        eng = _fake_engine(kfs, face_timeline)
        Engine._apply_motivated_zoom(eng)
        assert not any(k.get("_zoom") for k in eng.plan.keyframes)

        # A hold but NO face present → no zoom (don't push in on graphics).
        kfs2 = [
            {"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0},
            {"time_ms": 6000, "x": 100, "transition": "linear", "transition_ms": 0},
        ]
        eng2 = _fake_engine(kfs2, {})
        Engine._apply_motivated_zoom(eng2)
        assert not any(k.get("_zoom") for k in eng2.plan.keyframes)
    finally:
        settings.REFRAMER_MOTIVATED_ZOOM = prev


# ── export zoom crop builder ──

def test_piecewise_time_expr_structure_and_downsample():
    pts = [(i * 0.1, float(i)) for i in range(50)]
    expr = _piecewise_time_expr(pts, smooth=True, max_points=10)
    assert "if(lt(t," in expr
    # Downsampled: far fewer branches than 50.
    assert expr.count("if(lt(t,") <= 12


def test_build_zoom_crop_filter_shape():
    subj = [(0.0, 50), (2.0, 60), (4.0, 50)]
    zoom = [(0.0, 1.0), (1.0, 1.15), (3.0, 1.15), (4.0, 1.0)]
    crop = build_zoom_crop_filter(subj, zoom, src_w=1920, src_h=1080,
                                  crop_w0=608, crop_h0=1080, y_offset0=0)
    assert crop.startswith("crop=")
    # Time-varying width/height (references t and the base dims), clamped x.
    assert "floor((608)" in crop and "floor((1080)" in crop
    assert "in_w-" in crop and "clip(" in crop
    assert "max(1.0," in crop  # zoom never drops below 1.0 in this path
