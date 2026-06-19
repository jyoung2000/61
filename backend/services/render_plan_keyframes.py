"""Extract subject-tracking keyframes from a cached RenderPlan.

The analysis pipeline runs the reframer engine (the repo-lineage planner)
and converts its smooth, per-frame camera track into a Fez ``RenderPlan``
whose ``TRACKING_CROP`` ops carry the path as ``motion_path`` keypoints.
That plan is persisted to ``/data/uploads/{job_id}/render_plan.json``.

Historically the export path never read it back: ``JobResult`` has no
``render_plan`` field and the parser was an inert stub, so every clip
silently fell through to a coarser cluster-snap re-derivation that snaps
between speaker positions and collapses to a static crop. This module is
the real parser: it turns the cached plan into the ``(relative_time_sec,
subject_x_pct)`` keyframe list the FFmpeg crop pipeline consumes, so the
export follows the subject exactly like the planner intended.

It is intentionally dependency-light (stdlib only) so the export wrapper
and the test suite can import it without pulling in cv2 / ffmpeg.

Coordinate contract
-------------------
``motion_path[i].rect`` is normalized to source dimensions (0..1). For a
crop op, ``rect.x`` is the crop's LEFT edge and ``rect.w`` its width, both
as a fraction of source width. The export's ``subject_x`` is the crop
CENTRE as a percentage of source width, so::

    subject_x_pct = (rect.x + rect.w / 2) * 100

which is exactly the inverse of the exporter's ``subject_x -> crop offset``
math (``offset = src_w * sx/100 - crop_w/2``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# RenderOp kinds whose motion_path describes a usable horizontal camera move.
_CROP_KINDS = {"tracking_crop", "motivated_push_in", "motivated_pull_out"}


def _rect_center_x_pct(rect: dict) -> Optional[float]:
    """Crop-centre x as a 0-100 percentage of source width, or None."""
    if not isinstance(rect, dict):
        return None
    try:
        x = float(rect.get("x", 0.0))
        w = float(rect.get("w", 0.0))
    except (TypeError, ValueError):
        return None
    return (x + w / 2.0) * 100.0


def _interp_sx(samples: List[Tuple[float, float]], t: float) -> Optional[int]:
    """Linear-interpolate subject_x% at absolute time ``t``.

    ``samples`` is the sorted ``(abs_t, sx_pct)`` path. Mirrors the planner's
    piecewise-linear ``interpolate_x`` so seeding a clip boundary that lands
    mid-segment reproduces the position the planner would have rendered.
    """
    if not samples:
        return None
    if t <= samples[0][0]:
        return int(round(samples[0][1]))
    if t >= samples[-1][0]:
        return int(round(samples[-1][1]))
    for i in range(1, len(samples)):
        t0, s0 = samples[i - 1]
        t1, s1 = samples[i]
        if t0 <= t <= t1:
            span = t1 - t0
            if span < 1e-9:
                return int(round(s1))
            frac = (t - t0) / span
            return int(round(s0 + (s1 - s0) * frac))
    return int(round(samples[-1][1]))


def keyframes_from_cached_render_plan(
    cached_plan: Optional[dict],
    clip_start: float = 0.0,
    clip_end: Optional[float] = None,
    *args,
    **kwargs,
) -> List[Tuple[float, int]]:
    """Convert a cached RenderPlan dict into clip-relative tracking keyframes.

    Returns a sorted list of ``(relative_time_sec, subject_x_pct)`` tuples for
    the slice of the plan overlapping ``[clip_start, clip_end]`` (absolute
    source seconds), rebased so the first keyframe is at t=0. ``subject_x_pct``
    is the crop centre as a 0-100 percentage of source width.

    Fail-soft: returns ``[]`` for any missing / malformed plan so callers fall
    back to their own keyframe derivation. Never raises.
    """
    try:
        if not isinstance(cached_plan, dict):
            return []
        ops = cached_plan.get("ops") or []
        if not ops:
            return []

        clip_start = float(clip_start or 0.0)
        clip_end_f = float(clip_end) if clip_end is not None else None

        # Absolute-time (abs_t, sx_pct) samples across the whole plan.
        samples: List[Tuple[float, float]] = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            op_start = float(op.get("start_sec", 0.0) or 0.0)
            op_end = float(op.get("end_sec", op_start) or op_start)
            motion_path = op.get("motion_path") or []
            if motion_path:
                for kp in motion_path:
                    if not isinstance(kp, dict):
                        continue
                    sx = _rect_center_x_pct(kp.get("rect") or {})
                    if sx is None:
                        continue
                    try:
                        t_rel = float(kp.get("t", 0.0))
                    except (TypeError, ValueError):
                        continue
                    samples.append((op_start + t_rel, sx))
            else:
                # Static crop op: hold its primary_rect centre over the window.
                sx = _rect_center_x_pct(op.get("primary_rect") or {})
                if sx is not None:
                    samples.append((op_start, sx))
                    samples.append((op_end, sx))

        if not samples:
            return []
        samples.sort(key=lambda s: s[0])

        if clip_end_f is None:
            clip_end_f = samples[-1][0]
        clip_dur = round(max(0.0, clip_end_f - clip_start), 3)

        # Keep only keypoints inside the clip window, rebased to clip-relative.
        out: List[Tuple[float, int]] = []
        for abs_t, sx in samples:
            if abs_t < clip_start - 1e-6 or abs_t > clip_end_f + 1e-6:
                continue
            rel_t = round(abs_t - clip_start, 3)
            out.append((rel_t, int(round(max(0.0, min(100.0, sx))))))

        # Seed boundaries by interpolating the planner's path at the clip
        # edges so a clip starting/ending mid-segment still tracks correctly.
        start_seed = _interp_sx(samples, clip_start)
        if not out:
            if start_seed is None:
                return []
            out = [(0.0, start_seed)]
        if out[0][0] > 0.01 and start_seed is not None:
            out.insert(0, (0.0, start_seed))
        if clip_dur > 0 and out[-1][0] < clip_dur - 0.01:
            end_seed = _interp_sx(samples, clip_end_f)
            out.append((clip_dur, end_seed if end_seed is not None else out[-1][1]))

        out.sort(key=lambda kf: kf[0])

        # Collapse keypoints sharing a timestamp (keep the last).
        deduped: List[Tuple[float, int]] = []
        for t, sx in out:
            if deduped and abs(deduped[-1][0] - t) < 1e-4:
                deduped[-1] = (t, sx)
            else:
                deduped.append((t, sx))
        return deduped
    except Exception:
        # Any structural surprise -> let the caller fall back, never crash.
        return []
