"""Problem-driven reframe repair — the evaluator becomes a repair loop.

The evaluator localizes every HIGH ``face_missing`` second (a face was
detected but the interpolated crop doesn't contain it) with its keyframe
bracket. Most of those come from sparse detection: at 0.14-0.23 samples/s on
long videos the camera path is blind for up to 7s between looks. This module
takes those flagged windows, RE-DETECTS densely inside just them (a few
minutes of video, YuNet-only on downscaled frames — no YOLO, no VRAM), and
inserts corrective keyframes that bring the freshly confirmed face back into
the crop. Runs BEFORE the bridge so the corrected plan is what the exports,
preview overlays, and per-clip grades all see.

Gated by ``REFRAMER_PROBLEM_REPAIR`` (default True); bounded by
``REFRAMER_REPAIR_MAX_WINDOWS``. Fail-soft: any error leaves the plan as the
planner produced it.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from backend.config import settings
from backend.services.reframer_models import clamp_x, interpolate_x

logger = logging.getLogger("clipai.reframer_repair")

# Corrections closer than this to each other are merged (don't fight the
# smoother with keyframe bursts). Matches the smoother's face-enforcement gap.
_MIN_CORRECTION_GAP_MS = 1200


def _merge_problem_windows(problems: list, pad_s: float = 1.5,
                           max_windows: int = 40) -> List[Tuple[float, float]]:
    """HIGH face_missing seconds → merged [start_s, end_s) windows."""
    secs = sorted({int(p["time_sec"]) for p in problems
                   if p.get("severity") == "HIGH"
                   and p.get("type") == "face_missing"})
    windows: List[Tuple[float, float]] = []
    for s in secs:
        start, end = s - pad_s, s + 1 + pad_s
        if windows and start <= windows[-1][1] + 0.5:
            windows[-1] = (windows[-1][0], end)
        else:
            windows.append((start, end))
    if len(windows) > max_windows:
        # Keep the LONGEST windows — they cover the most lost seconds.
        windows.sort(key=lambda w: w[1] - w[0], reverse=True)
        windows = sorted(windows[:max_windows])
    return windows


def _yunet_model_path() -> Optional[str]:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "models", "face_detection_yunet_2023mar.onnx")
    if os.path.exists(path) and os.path.getsize(path) > 50000:
        return path
    return None


def _detect_faces_in_window(cap, detector, start_s: float, end_s: float,
                            fps: float, det_w: int, det_h: int,
                            det_scale: float, step_s: float = 0.34) -> dict:
    """Dense YuNet pass over one window → {time_ms: [face dicts]}."""
    out: dict = {}
    t = max(0.0, start_s)
    while t < end_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            t += step_s
            continue
        small = cv2.resize(frame, (det_w, det_h),
                           interpolation=cv2.INTER_LINEAR)
        detector.setInputSize((det_w, det_h))
        _, faces = detector.detect(small)
        entries = []
        for f in (faces if faces is not None else []):
            x, y, w, h = (float(f[0]), float(f[1]), float(f[2]), float(f[3]))
            conf = float(f[-1])
            if w <= 2 or h <= 2:
                continue
            entries.append({
                "x": int(x / det_scale), "y": int(y / det_scale),
                "w": int(w / det_scale), "h": int(h / det_scale),
                "cx": int((x + w / 2) / det_scale),
                "cy": int((y + h / 2) / det_scale),
                "confidence": round(conf, 3),
                "track_id": -1,
                "repair": True,
            })
        if entries:
            out[int(t * 1000)] = entries
        t += step_s
    return out


def repair_high_problem_windows(video_path: str, plan, perception) -> dict:
    """Re-detect inside HIGH face_missing windows and correct the plan.

    Returns a stats dict: windows examined, dense samples added, corrective
    keyframes inserted, and the HIGH count before repair (so the post-repair
    evaluation shows the delta). Mutates ``plan.keyframes`` and
    ``perception.face_timeline`` in place.
    """
    stats = {"enabled": False, "windows": 0, "samples_added": 0,
             "keyframes_inserted": 0, "high_before": 0}
    if not bool(getattr(settings, "REFRAMER_PROBLEM_REPAIR", True)):
        return stats
    stats["enabled"] = True

    # 1. Locate the problems (quiet evaluation — the loud one runs later,
    #    post-repair, and reports the improved numbers).
    from backend.services.reframe_evaluator import ReframeEvaluator
    report = ReframeEvaluator(plan, perception).run(quiet=True)
    stats["high_before"] = sum(
        1 for p in report.problems if p.get("severity") == "HIGH")
    max_windows = int(getattr(settings, "REFRAMER_REPAIR_MAX_WINDOWS", 40))
    windows = _merge_problem_windows(report.problems, max_windows=max_windows)
    if not windows:
        return stats
    stats["windows"] = len(windows)

    model_path = _yunet_model_path()
    if model_path is None:
        logger.info("repair skipped: YuNet model not on disk")
        return stats

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.info("repair skipped: could not open %s", video_path)
        return stats
    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        det_w = min(640, src_w)
        det_scale = det_w / src_w
        det_h = max(2, int(src_h * det_scale))
        detector = cv2.FaceDetectorYN.create(
            model_path, "", (det_w, det_h), score_threshold=0.6)

        crop_w = max(1, int(getattr(plan, "crop_w", 1) or 1))
        max_x = getattr(plan, "max_x", 0)
        keyframes = plan.keyframes if isinstance(plan.keyframes, list) else []
        inserted_times: List[int] = []

        for start_s, end_s in windows:
            dense = _detect_faces_in_window(
                cap, detector, start_s, end_s, fps, det_w, det_h, det_scale)
            if not dense:
                continue
            # Feed the fresh detections to every downstream consumer
            # (evaluator, preview overlays, per-clip grades).
            for t_ms, faces in dense.items():
                if t_ms not in perception.face_timeline:
                    perception.face_timeline[t_ms] = faces
                    stats["samples_added"] += 1
            # Corrective keyframes: where the freshly confirmed face sits
            # outside the CURRENT interpolated crop, pull the crop to center
            # it. Min-gap so a burst of dense samples doesn't fight the
            # smoother with a keyframe per frame.
            for t_ms in sorted(dense.keys()):
                faces = dense[t_ms]
                best = max(faces, key=lambda f: f.get("confidence", 0))
                cur_x = clamp_x(interpolate_x(keyframes, t_ms), max_x)
                cx = best["cx"]
                if cur_x <= cx <= cur_x + crop_w:
                    continue  # already framed — no correction needed
                if inserted_times and t_ms - inserted_times[-1] < _MIN_CORRECTION_GAP_MS:
                    continue
                new_x = clamp_x(int(cx - crop_w / 2), max_x)
                keyframes.append({
                    "time_ms": int(t_ms),
                    "x": int(new_x),
                    "transition": "linear",
                    "transition_ms": 350,
                    "reason": "problem_repair",
                })
                inserted_times.append(t_ms)
                stats["keyframes_inserted"] += 1

        if stats["keyframes_inserted"]:
            keyframes.sort(key=lambda kf: kf.get("time_ms", 0))
        logger.info(
            "Problem repair: %d HIGH window(s) re-detected → %d dense "
            "sample(s) added, %d corrective keyframe(s) inserted "
            "(HIGH before repair: %d)",
            stats["windows"], stats["samples_added"],
            stats["keyframes_inserted"], stats["high_before"])
    finally:
        cap.release()
    return stats
