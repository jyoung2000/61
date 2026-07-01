"""End-to-end wire test for motivated zoom.

Proves the `scale` term survives every conversion from the reframer
RenderPlan.keyframes to the FFmpeg `crop=` expression:

    plan.keyframes[scale]  →  to_fez_render_plan (motion_path[].scale)
      →  cached-plan dict  →  zoom_keyframes_from_cached_render_plan
      →  _build_filter_chain(zoom_keyframes=...)  →  build_zoom_crop_filter crop=
"""

import sys
import types
from dataclasses import asdict
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings  # noqa: E402
from backend.services import reframer_bridge  # noqa: E402
from backend.services.render_plan_keyframes import (  # noqa: E402
    keyframes_from_cached_render_plan,
    zoom_keyframes_from_cached_render_plan,
)
from backend.services import clip_exporter  # noqa: E402


def _reframer_plan(scales):
    """A reframer RenderPlan-like object: constant x=200, zoom per `scales`."""
    times = [0, 2000, 4000]
    kfs = []
    for i, (t, s) in enumerate(zip(times, scales)):
        kfs.append({
            "time_ms": t, "x": 200,
            "transition": "cut" if i == 0 else "ease_in_out",
            "transition_ms": 0 if i == 0 else 1000,
            "scale": s,
        })
    return SimpleNamespace(
        keyframes=kfs, crop_w=608, crop_h=1080, crop_y=0,
        source_width=1920, source_height=1080, duration_ms=4000, scenes=[],
    )


def _cached_plan_dict(reframer_plan):
    fez = reframer_bridge.to_fez_render_plan(
        reframer_plan, SimpleNamespace(duration_ms=4000, src_w=1920, src_h=1080),
        src_w=1920, src_h=1080, total_duration=4.0,
    )
    return asdict(fez)


def test_scale_survives_bridge_conversion():
    plan = _reframer_plan([1.0, 1.12, 1.0])
    cached = _cached_plan_dict(plan)
    zk = zoom_keyframes_from_cached_render_plan(cached, 0.0, 4.0)
    assert zk, "zoom keyframes should be produced"
    scales = [s for _, s in zk]
    assert max(scales) > 1.1  # the 1.12 push-in survived the round-trip
    # Subject-x track stays aligned in time with the zoom track.
    sx = keyframes_from_cached_render_plan(cached, 0.0, 4.0)
    assert [round(t, 3) for t, _ in sx] == [round(t, 3) for t, _ in zk]


def test_zoom_filter_is_emitted_end_to_end():
    prev = settings.REFRAMER_MOTIVATED_ZOOM
    settings.REFRAMER_MOTIVATED_ZOOM = True
    try:
        plan = _reframer_plan([1.0, 1.12, 1.0])
        cached = _cached_plan_dict(plan)
        subject_kf = keyframes_from_cached_render_plan(cached, 0.0, 4.0)
        zoom_kf = clip_exporter._zoom_keyframes_from_cached_render_plan(
            cached, 0.0, 4.0)
        assert zoom_kf is not None  # non-trivial zoom detected

        vf, _is_complex, _sub = clip_exporter._build_filter_chain(
            "9:16", 1920, 1080, None,
            subject_keyframes=subject_kf,
            zoom_keyframes=zoom_kf,
        )
        assert vf and "crop=" in vf
        # Signature of build_zoom_crop_filter: time-varying crop with a
        # zoom-clamped, floored width — NOT a static integer crop.
        assert "crop=floor(" in vf  # zoom expression, not static crop
        assert "max(1.0," in vf
    finally:
        settings.REFRAMER_MOTIVATED_ZOOM = prev


def test_all_unit_scale_is_plain_crop():
    prev = settings.REFRAMER_MOTIVATED_ZOOM
    settings.REFRAMER_MOTIVATED_ZOOM = True
    try:
        plan = _reframer_plan([1.0, 1.0, 1.0])
        cached = _cached_plan_dict(plan)
        subject_kf = keyframes_from_cached_render_plan(cached, 0.0, 4.0)
        # No non-trivial zoom → wrapper returns None → plain crop path.
        zoom_kf = clip_exporter._zoom_keyframes_from_cached_render_plan(
            cached, 0.0, 4.0)
        assert zoom_kf is None

        vf, _is_complex, _sub = clip_exporter._build_filter_chain(
            "9:16", 1920, 1080, None,
            subject_keyframes=subject_kf,
            zoom_keyframes=zoom_kf,
        )
        assert vf and "crop=" in vf
        assert "crop=floor(" not in vf   # no zoom expression
        assert "max(1.0," not in vf
    finally:
        settings.REFRAMER_MOTIVATED_ZOOM = prev


def test_zoom_ignored_when_flag_off():
    prev = settings.REFRAMER_MOTIVATED_ZOOM
    settings.REFRAMER_MOTIVATED_ZOOM = False
    try:
        plan = _reframer_plan([1.0, 1.12, 1.0])
        cached = _cached_plan_dict(plan)
        subject_kf = keyframes_from_cached_render_plan(cached, 0.0, 4.0)
        zoom_kf = clip_exporter._zoom_keyframes_from_cached_render_plan(
            cached, 0.0, 4.0)
        vf, _is_complex, _sub = clip_exporter._build_filter_chain(
            "9:16", 1920, 1080, None,
            subject_keyframes=subject_kf,
            zoom_keyframes=zoom_kf,
        )
        # Flag off ⇒ _zoom_active guard is False ⇒ plain crop even though the
        # plan carries a scale.
        assert vf and "crop=floor(" not in vf
    finally:
        settings.REFRAMER_MOTIVATED_ZOOM = prev
