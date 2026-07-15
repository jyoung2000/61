"""Reframe debug bundle — a machine-readable per-job artifact that makes the
reframing decisions inspectable (2026-07-15).

When a reframe looks wrong, the only signal used to be the wall-clock and "146
scenes" — the per-scene math and the crop trajectory were invisible, so nobody
(human or Claude) could say WHY a scene was framed the way it was. This
serializes the decision context that already exists at the end of
``ReframeEngine.analyze()``:

  * crop geometry (crop_w/h/y, source dims, fps, duration),
  * every scene's MEASURED signals + DERIVED params + chosen strategy
    (already embedded on ``plan.scenes``) plus human-readable flags,
  * the crop-x keyframe trajectory (so jitter / wrong-subject snaps are visible),
  * a perception summary + tracer event counts + total elapsed,
  * a computed decision summary (strategy histogram, keyframe jitter, the list
    of scenes with no trackable subject → saliency/center fallback).

Serialize-only + fail-soft: it NEVER changes reframing behavior and NEVER raises
into the render path. Gated by ``REFRAMER_DEBUG_JSON`` (default on; cheap).
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("clipai.reframe_debug")

SCHEMA_VERSION = 1


def _num(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def strategy_histogram(scenes) -> dict:
    """Count scenes per chosen strategy — a one-glance profile of the reframe."""
    hist: dict = {}
    for s in scenes or []:
        k = str((s or {}).get("strategy", "?"))
        hist[k] = hist.get(k, 0) + 1
    return hist


def keyframe_x_stats(keyframes) -> dict:
    """Trajectory stats over crop-x keyframes. ``mean_abs_step`` is a jitter
    proxy (average pixel jump between consecutive keyframes); a large
    ``max_abs_step`` marks a hard snap between subjects."""
    xs = [int(k["x"]) for k in (keyframes or [])
          if isinstance(k, dict) and isinstance(k.get("x"), (int, float))]
    if not xs:
        return {"count": 0, "min_x": None, "max_x": None, "span_x": 0,
                "mean_abs_step": 0.0, "max_abs_step": 0}
    steps = [abs(xs[i] - xs[i - 1]) for i in range(1, len(xs))]
    return {
        "count": len(xs),
        "min_x": min(xs), "max_x": max(xs), "span_x": max(xs) - min(xs),
        "mean_abs_step": round(sum(steps) / len(steps), 2) if steps else 0.0,
        "max_abs_step": max(steps) if steps else 0,
    }


def scene_flags(signals) -> list:
    """Human-readable flags derived from a scene's MEASURED signals — exactly
    what a reviewer looks for when a reframe is bad. Defensive against missing
    keys so a SceneSignals schema change never breaks the bundle."""
    s = signals or {}
    flags: list = []
    fd = _num(s.get("face_density"))
    pd = _num(s.get("person_density"))
    spread = _num(s.get("spatial_spread"))
    persistence = _num(s.get("face_persistence"))
    alternation = _num(s.get("speaker_alternation"))
    motion = _num(s.get("motion_energy"))
    if fd <= 0.01 and pd <= 0.05:
        flags.append("no-subject (crop is saliency/center fallback)")
    elif fd <= 0.01:
        flags.append("no-face (person/saliency driven)")
    if spread > 1.0:
        flags.append("subjects-dont-fit (spatial_spread>1 — can't frame both)")
    if 0.0 < persistence < 0.4:
        flags.append("unstable-track (face_persistence<0.4)")
    if alternation > 0.5:
        flags.append("frequent-speaker-switch")
    if motion > 0.0 and fd <= 0.01 and pd <= 0.05:
        flags.append("motion-only")
    return flags


def build_reframe_debug_bundle(plan, *, perception_summary=None,
                               tracer_counts=None, total_elapsed=0.0,
                               job_id="", schema_version=SCHEMA_VERSION) -> dict:
    """Assemble the debug bundle dict from the final RenderPlan + context.
    Pure + defensive: reads only via getattr/.get so it can't raise on a
    partially-populated plan."""
    scenes = list(getattr(plan, "scenes", []) or [])
    keyframes = list(getattr(plan, "keyframes", []) or [])
    strategy_log = list(getattr(plan, "strategy_log", []) or [])

    scene_out: list = []
    no_subject: list = []
    for sc in scenes:
        sig = (sc or {}).get("signals") or {}
        fl = scene_flags(sig)
        entry = dict(sc or {})
        entry["flags"] = fl
        scene_out.append(entry)
        if any(f.startswith("no-subject") for f in fl):
            idx = (sc or {}).get("scene_idx")
            if idx is not None:
                no_subject.append(idx)

    summary = {
        "total_elapsed_s": round(_num(total_elapsed), 2),
        "scene_count": len(scenes),
        "keyframe_count": len(keyframes),
        "strategy_histogram": strategy_histogram(scenes),
        "strategies_in_order": [(s or {}).get("strategy") for s in strategy_log],
        "keyframe_x": keyframe_x_stats(keyframes),
        "no_subject_scenes": no_subject,
        "no_subject_scene_count": len(no_subject),
    }

    return {
        "schema_version": schema_version,
        "job_id": job_id,
        "crop": {
            "crop_w": getattr(plan, "crop_w", None),
            "crop_h": getattr(plan, "crop_h", None),
            "crop_y": getattr(plan, "crop_y", None),
            "source_width": getattr(plan, "source_width", None),
            "source_height": getattr(plan, "source_height", None),
            "target_width": getattr(plan, "target_width", None),
            "target_height": getattr(plan, "target_height", None),
            "fps": getattr(plan, "fps", None),
            "duration_ms": getattr(plan, "duration_ms", None),
            "max_x": getattr(plan, "max_x", None),
        },
        "perception": dict(perception_summary or {}),
        "tracer_counts": dict(tracer_counts or {}),
        "summary": summary,
        "scenes": scene_out,
        "keyframes": keyframes,
    }


def write_reframe_debug_bundle(path, plan, **kwargs):
    """Build + write the bundle to PATH. Returns the path on success, None on any
    failure — NEVER raises into the caller's render path."""
    try:
        bundle = build_reframe_debug_bundle(plan, **kwargs)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2, default=str)
        logger.info("reframe debug bundle written: %s (%d scenes, %d keyframes, "
                    "%d no-subject scene(s))", path, len(bundle["scenes"]),
                    len(bundle["keyframes"]),
                    bundle["summary"]["no_subject_scene_count"])
        return path
    except Exception as e:
        logger.warning("reframe debug bundle skipped (%s)", e)
        return None
