"""Bridge adapter — converts clipai_reframer engine output into Fez data contracts.

The reframer engine (reframer_*.py) produces its own internal structures:
  * ReframerRenderPlan — keyframes as ``[{time_ms, x, transition, transition_ms}]``
    plus ``scenes`` / ``strategy_log`` / ``crop_w`` / ``crop_h`` / ``crop_y``.
  * PerceptionResult  — face_timeline, scene_cuts, speaker_timeline,
    transcript_segments, etc.
  * clipper ClipCandidate — start_s / end_s / composite_score / judge_scores ...

The Fez frontend consumes a completely different shape (see backend/models.py
and backend/services/render_plan.py). Every converter here outputs data that
matches those contracts exactly so the React app keeps working unchanged.
"""

import logging
import os
import subprocess

from backend.services.render_plan import (
    RenderPlan, RenderOp, RenderOpKind, Rect, MotionKeypoint,
)
from backend.services.reframer_models import interpolate_x, interpolate_scale
from backend.services.caption_text import strip_cue_timestamps

logger = logging.getLogger("clipai.reframer_bridge")


# ═══════════════════════════════════════════════════════════════════════════
#  Small numeric helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clamp01(v: float) -> float:
    """Clamp to the normalized [0.0, 1.0] range RenderPlan.validate() expects."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clamp(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, v))


def _to_int(v, default=0):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _attach_track_velocity(timeline: dict) -> None:
    """Inject per-box vx/vy (source px/sec) from the previous same-track box.

    ``timeline`` maps ``"<time_ms>" -> [box, ...]`` where each box has a
    ``track_id``, ``cx``, ``cy``. Mutates boxes in place, adding ``vx``/``vy``
    used by the preview to advect boxes between sparse detections. Boxes with
    no prior sample (or ``track_id < 0``) get zero velocity.
    """
    try:
        times = sorted(timeline.keys(), key=lambda s: int(s))
    except (TypeError, ValueError):
        return
    prev = {}  # track_id -> (t_ms, cx, cy)
    for ts in times:
        t_ms = int(ts)
        for box in timeline[ts]:
            tid = box.get("track_id", -1)
            vx = vy = 0.0
            if tid is not None and tid >= 0 and tid in prev:
                pt, pcx, pcy = prev[tid]
                dt = (t_ms - pt) / 1000.0
                if dt > 1e-3:
                    vx = (box.get("cx", 0) - pcx) / dt
                    vy = (box.get("cy", 0) - pcy) / dt
            box["vx"] = round(float(vx), 2)
            box["vy"] = round(float(vy), 2)
            if tid is not None and tid >= 0:
                prev[tid] = (t_ms, box.get("cx", 0), box.get("cy", 0))


# ═══════════════════════════════════════════════════════════════════════════
#  Function 1 — reframer RenderPlan  →  Fez RenderPlan
# ═══════════════════════════════════════════════════════════════════════════

def _prune_collinear_keypoints(kp_by_t: dict, eps_px: float = 1.0,
                               eps_scale: float = 0.002) -> dict:
    """Drop keypoints that lie on the straight line between their kept
    neighbours (within ``eps_px`` for x, ``eps_scale`` for scale).

    Keeps the dense sampling honest: eased moves keep their ≥10 Hz
    samples, holds and constant-velocity pans collapse back to their
    endpoints so the FFmpeg expression stays short.
    """
    times = sorted(kp_by_t)
    if len(times) <= 2:
        return kp_by_t
    kept = [times[0]]
    for i in range(1, len(times) - 1):
        t_prev = kept[-1]
        # Find the next candidate endpoint (look ahead to the next time)
        t_next = times[i + 1]
        t = times[i]
        x0, s0 = kp_by_t[t_prev]
        x1, s1 = kp_by_t[t_next]
        x, s = kp_by_t[t]
        span = t_next - t_prev
        frac = (t - t_prev) / span if span > 0 else 0.0
        lin_x = x0 + (x1 - x0) * frac
        lin_s = s0 + (s1 - s0) * frac
        if abs(x - lin_x) > eps_px or abs(s - lin_s) > eps_scale:
            kept.append(t)
    kept.append(times[-1])
    return {t: kp_by_t[t] for t in kept}


def to_fez_render_plan(
    reframer_plan,
    perception,
    src_w: int,
    src_h: int,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: float = 30.0,
    total_duration: float = 0.0,
) -> RenderPlan:
    """Convert the reframer's keyframed RenderPlan into Fez's RenderOp timeline.

    The result is gap-free and contiguous: ``ops[i].end_sec == ops[i+1].start_sec``
    for every i, the first op starts at 0.0, and the last op ends at
    ``total_duration``. Every op is a TRACKING_CROP with a non-empty motion_path,
    so the NLE crop-window editor always has keypoints to display and drag.
    """
    src_w = max(1, _to_int(src_w, 1920))
    src_h = max(1, _to_int(src_h, 1080))
    fps = float(fps) if fps and fps > 0 else 30.0

    keyframes = sorted(
        (getattr(reframer_plan, "keyframes", None) or []),
        key=lambda k: k.get("time_ms", 0),
    )

    # Crop geometry — prefer the plan's own values, fall back to a derived 9:16.
    crop_w = _to_int(getattr(reframer_plan, "crop_w", 0))
    crop_h = _to_int(getattr(reframer_plan, "crop_h", 0))
    crop_y = _to_int(getattr(reframer_plan, "crop_y", 0))
    if crop_w <= 0:
        crop_w = int(round(src_h * (target_w / max(1, target_h))))
    if crop_h <= 0:
        crop_h = src_h
    crop_w = max(2, min(crop_w, src_w))
    crop_h = max(2, min(crop_h, src_h))
    crop_y = max(0, min(crop_y, src_h - crop_h))

    cw = _clamp01(crop_w / src_w)
    ch = _clamp01(crop_h / src_h)
    cy = _clamp01(crop_y / src_h)
    center_x = (src_w - crop_w) // 2

    # Resolve total duration from the strongest signal available.
    if total_duration <= 0:
        total_duration = (getattr(reframer_plan, "duration_ms", 0) or 0) / 1000.0
    if total_duration <= 0:
        total_duration = (getattr(perception, "duration_ms", 0) or 0) / 1000.0
    if total_duration <= 0 and keyframes:
        total_duration = keyframes[-1].get("time_ms", 0) / 1000.0
    if total_duration <= 0:
        total_duration = 1.0

    def x_at(time_ms: float) -> int:
        """Crop-x in source pixels at an arbitrary time."""
        if not keyframes:
            return center_x
        return int(interpolate_x(keyframes, time_ms))

    def scale_at(time_ms: float) -> float:
        """Motivated-zoom scale at an arbitrary time (1.0 when no zoom)."""
        if not keyframes:
            return 1.0
        return float(interpolate_scale(keyframes, time_ms))

    # ── Pass 1 — gap-free [start_sec, end_sec, strategy] segments ──
    raw_scenes = sorted(
        (getattr(reframer_plan, "scenes", None) or []),
        key=lambda s: s.get("start_ms", 0),
    )
    min_dur = 1.0 / fps
    segments = []
    cursor = 0.0
    for sc in raw_scenes:
        end_sec = sc.get("end_ms", 0) / 1000.0
        if end_sec <= cursor + min_dur:
            continue  # zero / negative / out-of-order scene — fold into neighbour
        segments.append([cursor, min(end_sec, total_duration), str(sc.get("strategy", ""))])
        cursor = segments[-1][1]
        if cursor >= total_duration:
            break
    if not segments:
        segments = [[0.0, total_duration, "adaptive"]]
    segments[0][0] = 0.0
    segments[-1][1] = total_duration

    # ── Pass 2 — build a TRACKING_CROP RenderOp per segment ──
    ops = []
    for start_sec, end_sec, strategy in segments:
        if end_sec <= start_sec:
            end_sec = start_sec + min_dur
        s_ms = int(round(start_sec * 1000))
        e_ms = int(round(end_sec * 1000))
        op_dur = end_sec - start_sec

        # Keypoint set: always the two boundaries, plus interior keyframes.
        # Each entry carries (crop_x_px, scale) so the motivated-zoom term
        # travels with the crop position through the RenderPlan.
        kp_by_t = {0.0: (x_at(s_ms), scale_at(s_ms)),
                   round(op_dur, 4): (x_at(e_ms), scale_at(e_ms))}
        for kf in keyframes:
            t_ms = kf.get("time_ms", 0)
            if s_ms < t_ms < e_ms:
                rel_t = round((t_ms - s_ms) / 1000.0, 4)
                kp_by_t[rel_t] = (kf.get("x", center_x), scale_at(t_ms))

        # Densify: the FFmpeg piecewise-x expression interpolates keypoints
        # LINEARLY, but interpolate_x eases between keyframes — sparse
        # keypoints would flatten easing into visible velocity steps. Sample
        # the eased path at ≥REFRAMER_EXPORT_KEYPOINT_HZ, then drop samples
        # that are already on the linear hull so holds stay 2 points.
        try:
            from backend.config import settings as _settings
            _kp_hz = float(getattr(_settings, 'REFRAMER_EXPORT_KEYPOINT_HZ', 10.0))
        except Exception:
            _kp_hz = 10.0
        if _kp_hz > 0 and op_dur > 0:
            step = 1.0 / _kp_hz
            t = step
            while t < op_dur - step / 2:
                rel_t = round(t, 4)
                if rel_t not in kp_by_t:
                    kp_by_t[rel_t] = (x_at(s_ms + t * 1000.0),
                                      scale_at(s_ms + t * 1000.0))
                t += step
            kp_by_t = _prune_collinear_keypoints(kp_by_t)

        motion_path = [
            MotionKeypoint(
                t=t,
                rect=Rect(x=_clamp01(kp_by_t[t][0] / src_w), y=cy, w=cw, h=ch),
                scale=round(float(kp_by_t[t][1]), 4),
            )
            for t in sorted(kp_by_t)
        ]
        primary = motion_path[0].rect
        ops.append(RenderOp(
            kind=RenderOpKind.TRACKING_CROP,
            start_sec=round(start_sec, 4),
            end_sec=round(end_sec, 4),
            primary_rect=Rect(primary.x, primary.y, primary.w, primary.h),
            motion_path=motion_path,
            ease_in_ms=0,  # scene boundaries are treated as cuts; motion is in the path
            strategy_label=strategy,
        ))

    # Final contiguity guarantee against floating-point drift.
    for i in range(len(ops) - 1):
        ops[i + 1].start_sec = ops[i].end_sec
    ops[0].start_sec = 0.0
    ops[-1].end_sec = round(total_duration, 4)

    plan = RenderPlan(
        source_width=src_w,
        source_height=src_h,
        target_width=int(target_w),
        target_height=int(target_h),
        total_duration_sec=round(total_duration, 4),
        fps=fps,
        ops=ops,
    )

    violations = plan.validate()
    if violations:
        for v in violations:
            logger.warning("RenderPlan violation: %s", v)
    else:
        logger.info("RenderPlan OK: %d ops, %.1fs", len(ops), total_duration)
    return plan


# ═══════════════════════════════════════════════════════════════════════════
#  Function 2 — PerceptionResult  →  Fez SceneDescription dicts
# ═══════════════════════════════════════════════════════════════════════════

def _extract_thumbnails_batch(video_path: str, timestamps: list,
                              out_paths: list) -> int:
    """Extract MANY scene thumbnails in ONE decode pass.

    The old path ran one `ffmpeg -ss` per scene — 224 scenes ≈ 224 seeks
    ≈ minutes of bridge_conversion (167s measured on the 128-min run).
    A single pass with a select filter decodes the file once and writes
    every thumbnail. Timestamps must be ascending (scene starts are).
    Returns the number of thumbnails written; the caller falls back to
    per-scene extraction for any that are missing.
    """
    import tempfile

    written = 0
    CHUNK = 200  # practical select-filter size limit
    for c0 in range(0, len(timestamps), CHUNK):
        ts_chunk = timestamps[c0:c0 + CHUNK]
        path_chunk = out_paths[c0:c0 + CHUNK]
        clauses = "+".join(
            f"between(t,{max(0.0, t):.3f},{max(0.0, t) + 0.05:.3f})"
            for t in ts_chunk)
        tmpdir = tempfile.mkdtemp(prefix="scene_thumbs_")
        pattern = os.path.join(tmpdir, "t_%06d.jpg")
        try:
            proc = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", video_path,
                 "-vf", f"select='{clauses}'",
                 "-vsync", "vfr", "-q:v", "4", "-an",
                 pattern],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=max(120, int(len(ts_chunk) * 2)), check=False,
            )
            produced = sorted(
                f for f in os.listdir(tmpdir) if f.endswith(".jpg"))
            # select emits frames in timestamp order — a 1:1 map when the
            # count matches. On a mismatch (overlapping windows, decode
            # hiccup) keep what aligns from the front; the caller's
            # fallback covers the rest.
            for src_name, dst in zip(produced, path_chunk):
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(os.path.join(tmpdir, src_name), dst)
                    written += 1
                except OSError:
                    pass
        except Exception as exc:  # ffmpeg missing / timeout — non-fatal
            logger.warning("batch thumbnail pass failed: %s", exc)
        finally:
            try:
                import shutil as _sh
                _sh.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
    return written


def _extract_thumbnail(video_path: str, timestamp: float, out_path: str) -> bool:
    """Best-effort single-frame grab via FFmpeg. Returns True on success."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{max(0.0, timestamp):.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=False,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as exc:  # ffmpeg missing, bad seek, etc. — non-fatal
        logger.warning("thumbnail extract failed @%.2fs: %s", timestamp, exc)
        return False


def _faces_near(perception, time_ms: int, window_ms: int = 400):
    """Faces from the perception timeline closest to ``time_ms``."""
    timeline = getattr(perception, "face_timeline", None) or {}
    if not timeline:
        return []
    if time_ms in timeline and timeline[time_ms]:
        return timeline[time_ms]
    best, best_dt = [], None
    for t, faces in timeline.items():
        dt = abs(t - time_ms)
        if dt <= window_ms and faces and (best_dt is None or dt < best_dt):
            best, best_dt = faces, dt
    return best


def to_fez_scenes(perception, reframer_plan, video_path: str, frames_dir: str) -> list:
    """One SceneDescription dict per reframer scene.

    Populates face overlays, subject position, importance and a thumbnail so
    the NLE timeline, scene cards and crop-window editor render correctly.
    """
    src_w = max(1, _to_int(getattr(perception, "src_w", 0) or getattr(reframer_plan, "source_width", 1920)))
    src_h = max(1, _to_int(getattr(perception, "src_h", 0) or getattr(reframer_plan, "source_height", 1080)))
    crop_w = max(2, min(_to_int(getattr(reframer_plan, "crop_w", 0)) or int(src_h * 9 / 16), src_w))
    keyframes = sorted(
        (getattr(reframer_plan, "keyframes", None) or []),
        key=lambda k: k.get("time_ms", 0),
    )
    speech = getattr(perception, "speech_active", None) or {}
    motion = getattr(perception, "motion_timeline", None) or {}

    scenes_src = sorted(
        (getattr(reframer_plan, "scenes", None) or []),
        key=lambda s: s.get("start_ms", 0),
    )
    if not scenes_src:
        dur_ms = int((getattr(perception, "duration_ms", 0) or 0))
        scenes_src = [{"start_ms": 0, "end_ms": dur_ms, "strategy": "adaptive"}]

    # Single-decode thumbnail pre-pass (audit Phase 5.2 follow-up): all
    # scene thumbnails from ONE ffmpeg run instead of a seek per scene.
    _thumb_paths = [os.path.join(frames_dir, f"scene_{i:04d}.jpg")
                    for i in range(len(scenes_src))]
    _thumb_ts = [max(0.0, _to_int(sc.get("start_ms", 0)) / 1000.0)
                 for sc in scenes_src]
    try:
        _n_batch = _extract_thumbnails_batch(video_path, _thumb_ts, _thumb_paths)
        if _n_batch:
            logger.info("scene thumbnails: %d/%d via single-decode batch",
                        _n_batch, len(scenes_src))
    except Exception as _bt_err:
        logger.warning("scene thumbnail batch skipped: %s", _bt_err)

    out = []
    for i, sc in enumerate(scenes_src):
        start_ms = _to_int(sc.get("start_ms", 0))
        end_ms = _to_int(sc.get("end_ms", start_ms))
        ts = start_ms / 1000.0
        strategy = str(sc.get("strategy", "adaptive"))

        faces = _faces_near(perception, start_ms)
        face_positions = []
        for slot, f in enumerate(faces):
            face_positions.append({
                "slot_id": _to_int(f.get("track_id", slot), slot),
                "x": _to_int(f.get("x", 0) / src_w * 100),
                "y": _to_int(f.get("y", 0) / src_h * 100),
                "w": _to_int(f.get("w", 0) / src_w * 100),
                "h": _to_int(f.get("h", 0) / src_h * 100),
                "is_speaking": bool(f.get("is_speaking", False)),
                "identity_id": _to_int(f.get("track_id", slot), slot),
            })

        # Subject position = crop-window centre as a 0-100 fraction of width.
        if keyframes:
            kf_x = int(interpolate_x(keyframes, start_ms))
        elif faces:
            kf_x = int(faces[0].get("cx", src_w // 2)) - crop_w // 2
        else:
            kf_x = (src_w - crop_w) // 2
        subject_x = _to_int(_clamp((kf_x + crop_w / 2) / src_w * 100, 0, 100))

        # Importance — blend face presence and local motion onto a 1-10 scale.
        win_motion = [v for t, v in motion.items() if start_ms <= t < end_ms]
        avg_motion = sum(win_motion) / len(win_motion) if win_motion else 0.0
        importance = 3 + min(4, len(faces) * 2) + min(3, int(avg_motion / 4))
        importance = max(1, min(10, importance))

        speaking = any(
            speech.get(t, False) for t in range(start_ms, end_ms, 100)
        ) if speech else False
        active_speaker_x = subject_x if speaking else None

        precise_y = 50.0
        if faces:
            precise_y = _clamp(faces[0].get("cy", src_h / 2) / src_h * 100, 0, 100)

        thumb = os.path.join(frames_dir, f"scene_{i:04d}.jpg")
        if not (os.path.exists(thumb) and os.path.getsize(thumb) > 0):
            # Batch pass missed this one (mismatch/decode hiccup) — the
            # old per-scene seek is the fallback, not the default.
            _extract_thumbnail(video_path, ts, thumb)

        out.append({
            "timestamp": ts,
            "description": f"{strategy.replace('_', ' ')} — {len(faces)} face(s)",
            "importance_score": importance,
            "thumbnail_path": thumb,
            "subject_x": subject_x,
            "active_speaker_x": active_speaker_x,
            "layout_mode": "single",
            "face_count": len(faces),
            "face_positions": face_positions,
            "has_screen_content": False,
            "precise_x": float(subject_x),
            "precise_y": float(precise_y),
        })
    return out


def to_fez_subject_track(perception, reframer_plan) -> list:
    """Dense subject-position track for the NLE crop editor.

    One sample per reframer crop keyframe: ``{t, x, source}`` where ``x``
    is the crop-window centre as a 0-100 percent of source width — the
    same encoding as ``SceneDescription.subject_x`` so the editor's dense
    and per-scene crop paths agree.
    """
    src_w = max(1, _to_int(getattr(perception, "src_w", 0) or getattr(reframer_plan, "source_width", 1920)))
    src_h = max(1, _to_int(getattr(perception, "src_h", 0) or getattr(reframer_plan, "source_height", 1080)))
    crop_w = max(2, min(_to_int(getattr(reframer_plan, "crop_w", 0)) or int(src_h * 9 / 16), src_w))
    keyframes = sorted(
        (getattr(reframer_plan, "keyframes", None) or []),
        key=lambda k: k.get("time_ms", 0),
    )
    track = []
    for kf in keyframes:
        kf_x = _to_int(kf.get("x", (src_w - crop_w) // 2))
        center_pct = _clamp((kf_x + crop_w / 2) / src_w * 100.0, 0, 100)
        track.append({
            "t": round(_to_int(kf.get("time_ms", 0)) / 1000.0, 3),
            "x": round(center_pct, 2),
            "scale": round(float(kf.get("scale", 1.0)), 4),
            "source": "reframer",
        })
    return track


# ═══════════════════════════════════════════════════════════════════════════
#  Function 3 — reframer transcript segments  →  Fez TranscriptSegment dicts
# ═══════════════════════════════════════════════════════════════════════════

def _speaker_label_map(speaker_timeline: dict) -> dict:
    """Map raw speaker ids ("SPEAKER_00", "left", ...) to "Speaker N", stably."""
    mapping = {}
    if not speaker_timeline:
        return mapping
    n = 0
    for t in sorted(speaker_timeline):
        sid = speaker_timeline[t]
        if sid and sid not in mapping:
            n += 1
            mapping[sid] = f"Speaker {n}"
    return mapping


def to_fez_transcript(reframer_segments: list, speaker_timeline: dict = None) -> list:
    """Convert reframer ``{start_sec,end_sec,text,words,...}`` segments to
    Fez TranscriptSegment dicts with "Speaker N" attribution and word timing."""
    speaker_timeline = speaker_timeline or {}
    label_map = _speaker_label_map(speaker_timeline)
    out = []

    for seg in (reframer_segments or []):
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)
        end = float(seg.get("end_sec", seg.get("end", start)) or start)
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        # Dominant speaker across the segment's millisecond span.
        speaker = "Speaker 1"
        if speaker_timeline:
            votes = {}
            for t in range(int(start * 1000), int(end * 1000) + 1, 100):
                sid = speaker_timeline.get((t // 100) * 100) or speaker_timeline.get(t)
                if sid:
                    votes[sid] = votes.get(sid, 0) + 1
            if votes:
                top = max(votes, key=votes.get)
                speaker = label_map.get(top, "Speaker 1")

        words = []
        seg_words = seg.get("words") or []
        for w in seg_words:
            if isinstance(w, dict) and w.get("word"):
                words.append({
                    "start": float(w.get("start", start) or start),
                    "end": float(w.get("end", end) or end),
                    "word": str(w.get("word", "")),
                })

        # Confidence: mean word probability, else 1 - no_speech_prob.
        if seg_words:
            confs = [float(w.get("confidence", 1.0) or 1.0)
                     for w in seg_words if isinstance(w, dict)]
            confidence = sum(confs) / len(confs) if confs else 0.9
        else:
            confidence = _clamp01(1.0 - float(seg.get("no_speech_prob", 0.1) or 0.0))

        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "speaker": speaker,
            "words": words or None,
            "confidence": round(_clamp01(confidence), 3),
            "no_speech_prob": seg.get("no_speech_prob"),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Function 4 — clipper ClipCandidate  →  Fez ClipCandidate dicts
# ═══════════════════════════════════════════════════════════════════════════

_CLIP_TYPE_BY_SOURCE = {
    "vlm_discovery": "highlight",
    "signal_peak": "reaction",
    "algo_window": "story",
}


def _judge_axis(judge_scores: dict, *keys, default: int = 0) -> int:
    """Pull a 0-100 axis score from a judge dict under any of several aliases."""
    if not isinstance(judge_scores, dict):
        return default
    for k in keys:
        if k in judge_scores and judge_scores[k] is not None:
            return _to_int(_clamp(judge_scores[k], 0, 100), default)
    return default


def to_fez_clips(clipper_candidates: list, editorial_results: list = None) -> list:
    """Convert clipper ClipCandidate objects into Fez ClipCandidate dicts.

    Populates a 1-based sequential id, a composite ``viral_score`` and all four
    decomposed axes (hook/flow/value/trend) plus their reasons, so the ClipCard
    radar chart in the React app always has data to draw.
    """
    out = []
    for idx, c in enumerate(clipper_candidates or [], start=1):
        get = (lambda k, d=None: c.get(k, d)) if isinstance(c, dict) else (
            lambda k, d=None: getattr(c, k, d))

        start_t = float(get("start_s", 0.0) or 0.0)
        end_t = float(get("end_s", start_t) or start_t)
        duration = float(get("duration_s", 0.0) or 0.0) or max(0.0, end_t - start_t)

        judge = get("judge_scores") or {}
        signal = float(get("signal_score", 0.0) or 0.0)
        composite = float(get("composite_score", 0.0) or 0.0)

        # viral_score — prefer the editorial judge, else the composite signal.
        viral = _judge_axis(judge, "viral_score", "overall", "score", default=0)
        if viral <= 0:
            viral = _to_int(_clamp(composite * 100, 1, 100), 50)
        viral = max(1, min(100, viral))

        hook = _judge_axis(judge, "hook_score", "hook", default=0) or viral
        flow = _judge_axis(judge, "flow_score", "flow", default=0) or viral
        value = _judge_axis(judge, "value_score", "value", default=0) or viral
        trend = _judge_axis(judge, "trend_score", "trend", default=0) or viral

        transcript_slice = str(get("transcript_slice", "") or "").strip()
        vlm_hook = str(get("vlm_hook", "") or "").strip()
        vlm_reason = str(get("vlm_reason", "") or "").strip()
        judge_title = str(get("judge_title", "") or "").strip()

        # ``transcript_slice`` carries inline ``[m:ss]`` cue markers; strip them
        # from the social-facing hook/caption/title (shown on cards and used as
        # on-screen overlays) — "[0:00]" is just noise there.
        _clean_slice = strip_cue_timestamps(transcript_slice)
        title = judge_title or vlm_hook[:80] or (
            " ".join(_clean_slice.split()[:8]) or f"Clip {idx}")
        hook_text = vlm_hook or (_clean_slice[:120] if _clean_slice else title)
        reasoning = vlm_reason or (
            f"Selected by signal analysis (score {composite:.2f}).")
        why = vlm_reason or (
            "Strong audio/visual engagement signals across this window.")
        caption = (_clean_slice[:150] if _clean_slice else title)

        source = str(get("source", "") or "")
        clip_type = _CLIP_TYPE_BY_SOURCE.get(source, "highlight")

        out.append({
            "id": idx,
            "title": title,
            "start_time": round(start_t, 3),
            "end_time": round(end_t, 3),
            "duration": round(duration, 3),
            "viral_score": viral,
            "viral_score_reasoning": reasoning,
            "clip_type": clip_type,
            "platform": "both",
            "suggested_caption": caption,
            "hook_text": hook_text,
            "why_this_works": why,
            "hook_score": hook,
            "flow_score": flow,
            "value_score": value,
            "trend_score": trend,
            "hook_reason": str(judge.get("hook_reason", "") or "Opening grabs attention."),
            "flow_reason": str(judge.get("flow_reason", "") or "Clean narrative arc."),
            "value_reason": str(judge.get("value_reason", "") or "Delivers a clear payoff."),
            "trend_reason": str(judge.get("trend_reason", "") or "Fits short-form formats."),
            # Preserve the three model-emitted strings the bridge used as
            # fallback sources so the post-translation refresh in
            # ``pipeline._refresh_clips_with_translation`` can rebuild
            # caption / hook_text / title against the translated transcript
            # without losing VLM-derived text.
            "vlm_hook": vlm_hook or None,
            "vlm_reason": vlm_reason or None,
            "judge_title": judge_title or None,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Function 5 — perception timelines → frontend detection-overlay sidecar
# ═══════════════════════════════════════════════════════════════════════════

def serialize_detection_overlay(perception, reframer_plan=None) -> dict:
    """Pack the perception timelines into JSON the React preview overlay
    consumes via /api/jobs/{job_id}/detection_overlay.

    All timestamps stay in milliseconds (matching the reframer's native
    units). Bounding boxes stay in *source-pixel* space so the canvas
    overlay can map them to its own display rect using scale factors.
    """
    face_tl = {}
    for t_ms, faces in (getattr(perception, "face_timeline", None) or {}).items():
        face_tl[str(int(t_ms))] = [
            {
                "x": _to_int(f.get("x", 0)),
                "y": _to_int(f.get("y", 0)),
                "w": _to_int(f.get("w", 0)),
                "h": _to_int(f.get("h", 0)),
                "cx": _to_int(f.get("cx", 0)),
                "cy": _to_int(f.get("cy", 0)),
                "confidence": round(float(f.get("confidence", 0.0) or 0.0), 3),
                "track_id": _to_int(f.get("track_id", -1), default=-1),
                "mouth_motion": round(float(f.get("mouth_motion", 0.0) or 0.0), 3),
                "source": str(f.get("source", "") or "unknown"),
                # Eye-line anchor + gaze yaw (items 5/6). Present only when the
                # detector produced landmarks; the frontend falls back to cx.
                **({"eye_cx": _to_int(f.get("eye_cx"))} if f.get("eye_cx") is not None else {}),
                **({"yaw": round(float(f.get("yaw", 0.0) or 0.0), 3)} if f.get("yaw") is not None else {}),
            }
            for f in (faces or [])
        ]

    person_tl = {}
    for t_ms, persons in (getattr(perception, "person_timeline", None) or {}).items():
        person_tl[str(int(t_ms))] = [
            {
                "x": _to_int(p.get("x", 0)),
                "y": _to_int(p.get("y", 0)),
                "w": _to_int(p.get("w", 0)),
                "h": _to_int(p.get("h", 0)),
                "cx": _to_int(p.get("cx", 0)),
                "cy": _to_int(p.get("cy", 0)),
                "class_name": str(p.get("class_name", "") or "person"),
                # Persistent per-person id (item 1) so the overlay interpolates
                # a box per subject instead of cross-fading two people.
                "track_id": _to_int(p.get("track_id", -1), default=-1),
            }
            for p in (persons or [])
        ]

    # ── Per-box velocity for optical-flow-style preview propagation (item 7) ──
    # Attach vx/vy (source px/sec) to each box from the previous sample of the
    # same track so the frontend can advect boxes smoothly between (and just
    # past) sparse detections. Gated so it can be turned off.
    try:
        from backend.config import settings as _settings
        _emit_velocity = bool(getattr(_settings, "REFRAMER_OVERLAY_VELOCITY", True))
    except Exception:
        _emit_velocity = True
    if _emit_velocity:
        _attach_track_velocity(face_tl)
        _attach_track_velocity(person_tl)

    motion_tl = {}
    for t_ms, v in (getattr(perception, "motion_timeline", None) or {}).items():
        try:
            motion_tl[str(int(t_ms))] = round(float(v), 3)
        except (TypeError, ValueError):
            pass

    speech_tl = {}
    for t_ms, v in (getattr(perception, "speech_active", None) or {}).items():
        speech_tl[str(int(t_ms))] = bool(v)

    saliency_tl = {}
    for t_ms, v in (getattr(perception, "saliency_hotspot", None) or {}).items():
        if isinstance(v, dict):
            saliency_tl[str(int(t_ms))] = {
                "cx": _to_int(v.get("cx", 0)),
                "cy": _to_int(v.get("cy", 0)),
                "intensity": round(float(v.get("intensity", 0.0) or 0.0), 3),
                "source": str(v.get("source", "") or "spectral"),
            }

    scene_cuts = [int(c) for c in (getattr(perception, "scene_cuts", None) or [])]

    track_ids = set()
    for faces in (getattr(perception, "face_timeline", None) or {}).values():
        for f in (faces or []):
            tid = _to_int(f.get("track_id", -1), default=-1)
            if tid >= 0:
                track_ids.add(tid)

    samples_with_faces = sum(
        1 for v in (getattr(perception, "face_timeline", None) or {}).values() if v
    )

    metadata = {
        "src_w": _to_int(getattr(perception, "src_w", 0)),
        "src_h": _to_int(getattr(perception, "src_h", 0)),
        "fps": round(float(getattr(perception, "fps", 0.0) or 0.0), 3),
        "duration_ms": _to_int(getattr(perception, "duration_ms", 0)),
        "is_live_action": bool(getattr(perception, "is_live_action", False)),
        "total_face_samples": samples_with_faces,
        "total_tracks": len(track_ids),
        "language": str(getattr(perception, "detected_language", "") or ""),
    }

    return {
        "face_timeline": face_tl,
        "person_timeline": person_tl,
        "motion_timeline": motion_tl,
        "speech_active": speech_tl,
        "scene_cuts": scene_cuts,
        "saliency_hotspot": saliency_tl,
        "metadata": metadata,
    }
