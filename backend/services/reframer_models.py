#!/usr/bin/env python3
"""
ClipAI Reframer — Universal 16:9 → 9:16 Video Reframing Engine

Five-stage pipeline:
  1. PERCEIVE  — Extract faces, scenes, motion signals
  2. CLASSIFY  — Route each scene to the best strategy
  3. DECIDE    — Generate keyframed crop positions (RenderPlan)
  4. SMOOTH    — Velocity clamp, deadzone, face integrity
  5. RENDER    — Export via FFmpeg pipe (guaranteed preview/export parity)

Strategies:
  S1: Single Focus Track     S2: Speaker Switch (Cut)
  S3: Dynamic Pan            S4: Split / PIP
  S5: Center Crop            S6: Artistic / Beat-Sync

Usage:
  CLI:  python clipai_reframer.py input.mp4 -o output_9x16.mp4
  GUI:  python clipai_reframer.py --gui
  Lib:  from clipai_reframer import ReframeEngine; engine = ReframeEngine("input.mp4")

Requirements:
  pip install opencv-python-headless numpy pillow
  ffmpeg on PATH
"""

import cv2
import numpy as np
import json
import subprocess
import threading
import os
import sys
import math
import argparse
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING — Structured log output for every pipeline stage
# ═══════════════════════════════════════════════════════════════════════════

class ReframeLogger:
    """
    Structured logger that writes to both console and a per-session log file.
    Log files go to ./logs/ with timestamp-based names.
    """

    def __init__(self, log_dir: str = None):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        if log_dir:
            self.log_dir = log_dir
        else:
            try:
                self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            except NameError:
                self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, f"reframe_{self.session_id}.log")
        self.entries: List[dict] = []
        self._timers: Dict[str, float] = {}

        # Python logger
        self.logger = logging.getLogger(f"clipai_reframer_{self.session_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler — full detail
        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)-5s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(fh)

        # Console handler — info and above
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('[%(levelname)-5s] %(message)s'))
        self.logger.addHandler(ch)

        # Propagate to root so the rotating /data/logs/app.log handler
        # (set up by the FastAPI app's logging config) picks these up,
        # which is what the /api/logs/export endpoint streams. Without
        # this, every line written through ``log.log_stage(...)`` lives
        # ONLY in the per-session reframe_YYYY.log under
        # backend/services/logs/ and never makes it into the export
        # bundle the user downloads.
        self.logger.propagate = True

        # Belt-and-braces: if /data/logs/app.log exists and is writable
        # but the root logger isn't writing there (e.g. CLI runs that
        # never call setup_logging), attach our own handler. Concurrent
        # writes from two file handlers to the same file are safe at
        # the line level on POSIX (each .emit() is a single write()
        # below PIPE_BUF for any realistic log line).
        app_log_path = "/data/logs/app.log"
        try:
            if os.path.isdir(os.path.dirname(app_log_path)) and os.access(
                    os.path.dirname(app_log_path), os.W_OK):
                root = logging.getLogger()
                already_attached = any(
                    isinstance(h, logging.FileHandler)
                    and os.path.abspath(getattr(h, 'baseFilename', '')) ==
                        os.path.abspath(app_log_path)
                    for h in root.handlers)
                if not already_attached:
                    app_fh = logging.FileHandler(app_log_path, encoding='utf-8')
                    app_fh.setLevel(logging.INFO)
                    app_fh.setFormatter(logging.Formatter(
                        '%(asctime)s [%(levelname)-5s] [reframer] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'))
                    self.logger.addHandler(app_fh)
        except Exception:
            # Permission denied, disk full, weird filesystem — silently
            # skip; the dedicated per-session file still works.
            pass

        self.logger.info(f"Session started: {self.session_id}")
        self.logger.info(f"Log file: {self.log_path}")

    def start_timer(self, name: str):
        self._timers[name] = _time.monotonic()
        self.logger.debug(f"Timer started: {name}")

    def stop_timer(self, name: str) -> float:
        if name not in self._timers:
            return 0
        elapsed = _time.monotonic() - self._timers[name]
        self.logger.info(f"Timer [{name}]: {elapsed:.2f}s")
        return elapsed

    def log_stage(self, stage: str, msg: str, **data):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'message': msg,
            **data
        }
        self.entries.append(entry)
        self.logger.info(f"[{stage}] {msg}")
        for k, v in data.items():
            self.logger.debug(f"  {k}: {v}")

    def log_error(self, stage: str, msg: str, exc: Exception = None):
        self.logger.error(f"[{stage}] {msg}")
        if exc:
            self.logger.error(f"  Exception: {exc}", exc_info=True)

    def log_metric(self, name: str, value, unit: str = ""):
        self.logger.info(f"  {name}: {value}{' ' + unit if unit else ''}")

    def get_summary(self) -> dict:
        """Return a summary dict of all timers and key metrics."""
        summary = {
            'session_id': self.session_id,
            'log_path': self.log_path,
            'timers': {},
            'entries': self.entries,
        }
        return summary

    def save_summary(self, path: str = None):
        """Save a JSON summary alongside the log file."""
        path = path or self.log_path.replace('.log', '_summary.json')
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        self.logger.info(f"Summary saved: {path}")
        return path


# ═══════════════════════════════════════════════════════════════════════════
#  REFRAME TRACER — Append-only JSONL of every per-keyframe decision
# ═══════════════════════════════════════════════════════════════════════════
#
#  The structured logger above captures stage transitions and aggregates —
#  "scene 7 → adaptive_face (8 keyframes)", "smoothed 45 → 42 keyframes".
#  That's enough to know WHAT happened but not WHY: which keyframe got
#  dropped, which face was chosen at t=12.3s, why crop_x = 480 vs 520.
#
#  The tracer fills that gap with one JSON line per decision. Schema is
#  intentionally flat — each event carries an ``event`` field naming the
#  decision, a ``t_ms`` (when relevant), and arbitrary kwargs describing
#  the inputs and outcome. Consumers grep / jq the file:
#
#    jq 'select(.event=="centering_nudge")' reframe_trace.jsonl
#    jq 'select(.event=="smoother_drop" and .reason=="drift_suppressed")' …
#
#  The tracer is a no-op when ``path`` is empty so callsites don't need
#  to None-check it. One file per job; the pipeline writes it next to
#  ``render_plan.json`` so the artifacts move together.

class ReframeTracer:
    """Append-only JSONL writer for per-decision reframe telemetry.

    Tracer events follow a flat schema: every line is a JSON object with
    at minimum ``{"event": "<name>", "t_ms": <int>}`` plus arbitrary
    decision-specific kwargs. The tracer never raises — write errors
    are swallowed so a full disk can't take down the pipeline.

    When ``path`` is empty / None, the tracer becomes a no-op:
    ``.event(...)`` returns immediately. Callsites pass the engine's
    tracer everywhere without having to guard each call.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Optional[str] = None):
        self.path = path or ""
        self._fh = None
        self._counts: Dict[str, int] = {}
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Open in append mode so reruns in the same job dir (e.g.
            # regenerate) don't blow away the previous trace.
            self._fh = open(self.path, 'a', encoding='utf-8')
            self._raw_write({
                'event': 'session_start',
                'schema_version': self.SCHEMA_VERSION,
                'started_at': datetime.now().isoformat(timespec='seconds'),
            })
        except Exception:
            # Could be /data/logs/jobs/.../reframe_trace.jsonl with the
            # parent dir not yet created, or a read-only filesystem —
            # either way, fall back to disabled-tracer behaviour.
            self._fh = None

    def _raw_write(self, obj: dict):
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(
                obj, default=str, separators=(',', ':')) + '\n')
            self._fh.flush()
        except Exception:
            # Best-effort — a failed write must not abort the pipeline.
            pass

    def event(self, name: str, **kwargs):
        """Emit one trace line.

        ``name`` becomes the ``event`` field. Any keyword arguments are
        included verbatim (after JSON-default coercion via ``default=str``,
        so dataclasses / numpy scalars / etc. serialise without error).
        """
        if self._fh is None:
            return
        self._counts[name] = self._counts.get(name, 0) + 1
        kwargs['event'] = name
        self._raw_write(kwargs)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    @property
    def counts(self) -> Dict[str, int]:
        """Per-event call counts. Useful for asserting in tests / smoke runs."""
        return dict(self._counts)

    def close(self, summary: Optional[dict] = None):
        """Flush + close the underlying file handle.

        Optional ``summary`` is emitted as a final ``session_end`` event
        so consumers can find the total counts without scanning the file.
        """
        if self._fh is None:
            return
        try:
            self._raw_write({
                'event': 'session_end',
                'ended_at': datetime.now().isoformat(timespec='seconds'),
                'counts': dict(self._counts),
                'summary': summary or {},
            })
            self._fh.close()
        except Exception:
            pass
        finally:
            self._fh = None


# A singleton no-op tracer for code paths that haven't been threaded with
# an explicit one yet. Tests / CLI invocations that don't care about
# tracing can use this without checking for None.
_NULL_TRACER = ReframeTracer(path=None)


def null_tracer() -> ReframeTracer:
    """Return a shared no-op tracer for callers that don't want telemetry."""
    return _NULL_TRACER


# Global logger instance — set per session
_logger: Optional[ReframeLogger] = None

def get_logger(log_dir: str = None) -> ReframeLogger:
    global _logger
    if _logger is None:
        _logger = ReframeLogger(log_dir)
    return _logger

def reset_logger(log_dir: str = None) -> ReframeLogger:
    global _logger
    _logger = ReframeLogger(log_dir)
    return _logger


# ═══════════════════════════════════════════════════════════════════════════
#  RENDERPLAN IR — The single source of truth consumed by preview + export
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RenderPlan:
    version: str = "2.0"
    source_width: int = 1920
    source_height: int = 1080
    target_width: int = 1080
    target_height: int = 1920
    fps: float = 30.0
    duration_ms: int = 0
    crop_w: int = 608
    crop_h: int = 1080
    crop_y: int = 0
    keyframes: List[dict] = field(default_factory=list)
    scenes: List[dict] = field(default_factory=list)
    strategy_log: List[dict] = field(default_factory=list)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'RenderPlan':
        with open(path) as f:
            d = json.load(f)
        p = cls()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    @property
    def max_x(self) -> int:
        return max(0, self.source_width - self.crop_w)


# ═══════════════════════════════════════════════════════════════════════════
#  INTERPOLATION — The shared contract. Both preview and export call this.
# ═══════════════════════════════════════════════════════════════════════════

def interpolate_x(keyframes: List[dict], time_ms: float) -> int:
    """
    Canonical crop-x resolver. Returns integer pixel position.
    BOTH the preview canvas AND the FFmpeg export pipe call this function.
    """
    if not keyframes:
        return 0

    n = len(keyframes)
    # Before first keyframe
    if time_ms <= keyframes[0]['time_ms']:
        return keyframes[0]['x']
    # After last keyframe
    if time_ms >= keyframes[-1]['time_ms']:
        return keyframes[-1]['x']

    # Find bracketing keyframes
    prev_idx = 0
    for i in range(n):
        if keyframes[i]['time_ms'] <= time_ms:
            prev_idx = i

    prev_kf = keyframes[prev_idx]
    next_idx = prev_idx + 1
    if next_idx >= n:
        return prev_kf['x']
    next_kf = keyframes[next_idx]

    # CUT transition: hold prev value until exact next time
    if next_kf.get('transition', 'cut') == 'cut':
        return prev_kf['x'] if time_ms < next_kf['time_ms'] else next_kf['x']

    # Eased/linear transitions
    trans_ms = next_kf.get('transition_ms', 300)
    # transition_ms == 0 means span the full prev→next interval
    if trans_ms <= 0:
        t_start = prev_kf['time_ms']
    else:
        t_start = next_kf['time_ms'] - trans_ms
    t_end = next_kf['time_ms']

    if time_ms <= t_start:
        return prev_kf['x']
    if time_ms >= t_end:
        return next_kf['x']

    dur = t_end - t_start
    if dur <= 0:
        return next_kf['x']

    p = (time_ms - t_start) / dur

    if next_kf['transition'] == 'linear':
        eased = p
    elif next_kf['transition'] == 'ease_in':
        # Slow start, fast finish — use for tracking pans (camera is catching up)
        eased = p * p
    elif next_kf['transition'] == 'ease_out':
        # Fast start, slow finish — use for correction moves (settling on target)
        eased = 1 - (1 - p) * (1 - p)
    elif next_kf['transition'] == 'ease_in_out':
        eased = 4*p*p*p if p < 0.5 else 1 - (-2*p + 2)**3 / 2
    else:
        eased = p

    return round(prev_kf['x'] + (next_kf['x'] - prev_kf['x']) * eased)


def interpolate_scale(keyframes: List[dict], time_ms: float) -> float:
    """Canonical zoom/scale resolver (item 10 — motivated zoom).

    Parallel to ``interpolate_x`` but for the optional per-keyframe ``scale``
    term (1.0 = no zoom, >1.0 = punch-in). Keyframes without a ``scale`` key
    resolve to 1.0, so this is fully backward-compatible: a plan authored
    before zoom existed always returns 1.0 and the caller renders at native
    crop width.
    """
    if not keyframes:
        return 1.0

    def _s(kf):
        try:
            v = float(kf.get('scale', 1.0))
        except (TypeError, ValueError):
            return 1.0
        return v if v > 0 else 1.0

    n = len(keyframes)
    if time_ms <= keyframes[0]['time_ms']:
        return _s(keyframes[0])
    if time_ms >= keyframes[-1]['time_ms']:
        return _s(keyframes[-1])

    prev_idx = 0
    for i in range(n):
        if keyframes[i]['time_ms'] <= time_ms:
            prev_idx = i
    prev_kf = keyframes[prev_idx]
    next_idx = prev_idx + 1
    if next_idx >= n:
        return _s(prev_kf)
    next_kf = keyframes[next_idx]

    if next_kf.get('transition', 'cut') == 'cut':
        return _s(prev_kf) if time_ms < next_kf['time_ms'] else _s(next_kf)

    trans_ms = next_kf.get('transition_ms', 300)
    if trans_ms <= 0:
        t_start = prev_kf['time_ms']
    else:
        t_start = next_kf['time_ms'] - trans_ms
    t_end = next_kf['time_ms']
    if time_ms <= t_start:
        return _s(prev_kf)
    if time_ms >= t_end:
        return _s(next_kf)
    dur = t_end - t_start
    if dur <= 0:
        return _s(next_kf)
    p = (time_ms - t_start) / dur
    # Zoom always eases smoothly regardless of the x transition style — a
    # punch-in should never snap. Use ease_in_out.
    eased = 4 * p * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2
    return _s(prev_kf) + (_s(next_kf) - _s(prev_kf)) * eased


def clamp_x(x: int, max_x: int) -> int:
    return max(0, min(x, max_x))


def _face_overlaps_person(face: dict, persons: list, margin: int = 50) -> bool:
    """Check if a face center is inside any person bounding box.

    In live-action content, real human faces overlap with YOLO person
    detections. Faces on figurines, artwork, logos do NOT overlap.
    If no persons detected at this time, accept all faces (fallback).
    """
    if not persons:
        return True
    fcx = face.get('cx', 0)
    fcy = face.get('cy', 0)
    for p in persons:
        px, py = p.get('x', 0), p.get('y', 0)
        pw, ph = p.get('w', 0), p.get('h', 0)
        if (px - margin <= fcx <= px + pw + margin and
                py - margin <= fcy <= py + ph + margin):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  TACT — Temporal Coverage Ledger
# ═══════════════════════════════════════════════════════════════════════════
#
#  The coverage ledger is a millisecond-resolution record of every audio
#  frame and what is known about it. Every bin is positively labeled:
#  speech, silence, music, noise — all accounted for, never silently dropped.
#
#  Status values:
#    covered_speech  — ASR produced text with confidence > threshold
#    covered_silence — confirmed silence (VAD says no voice, ASR agrees)
#    covered_event   — non-speech audio event (music, applause, laughter, etc.)
#    low_confidence  — ASR produced text but confidence is low
#    quarantined     — hallucination detected (repetition, boilerplate, etc.)
#    uncovered       — no pass has claimed this bin yet
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LedgerBin:
    """One bin of the Temporal Coverage Ledger."""
    status: str = 'uncovered'        # covered_speech | covered_silence | covered_event |
                                     # low_confidence | quarantined | uncovered
    text: Optional[str] = None       # word content (speech), event tag (event), None (silence)
    source: str = ''                 # 'whisper_primary' | 'gap_fill' | 'event_classifier'
    confidence: float = 0.0          # 0.0 – 1.0, from ASR model
    speaker_id: Optional[str] = None # from diarization: "SPEAKER_00", etc.
    event_type: Optional[str] = None # 'music' | 'applause' | 'laughter' | 'silence' |
                                     # 'noise' | 'crowd' | None


@dataclass
class CoverageLedger:
    """Millisecond-resolution temporal coverage for audio intelligence.

    The ledger ensures every second of audio is accounted for — speech,
    silence, music, noise — rather than silently dropped. It is the
    reframer's primary audio intelligence signal, driving speaker lock
    confidence, event-aware strategy selection, and hallucination gating.
    """
    bins: Dict[int, LedgerBin] = field(default_factory=dict)  # time_ms → bin
    bin_width_ms: int = 20
    duration_ms: int = 0

    @property
    def coverage_ratio(self) -> float:
        """Fraction of bins that are NOT uncovered (0.0–1.0)."""
        if not self.bins:
            return 0.0
        covered = sum(1 for b in self.bins.values() if b.status != 'uncovered')
        return covered / len(self.bins)

    def query_status(self, start_ms: int, end_ms: int,
                     statuses: set) -> List[Tuple[int, int]]:
        """Return contiguous intervals matching any of the given statuses."""
        intervals = []
        interval_start = None
        for t in range(start_ms, end_ms, self.bin_width_ms):
            b = self.bins.get(t)
            if b and b.status in statuses:
                if interval_start is None:
                    interval_start = t
            else:
                if interval_start is not None:
                    intervals.append((interval_start, t))
                    interval_start = None
        if interval_start is not None:
            intervals.append((interval_start, end_ms))
        return intervals

    def coverage_report(self) -> dict:
        """Compute per-status histogram and overall coverage."""
        counts = {}
        for b in self.bins.values():
            counts[b.status] = counts.get(b.status, 0) + 1
        total = max(1, len(self.bins))
        return {
            'total_bins': total,
            'coverage_ratio': self.coverage_ratio,
            'status_pct': {k: round(v / total * 100, 1) for k, v in counts.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1 — PERCEIVE  (signal extraction at sample_fps)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerceptionResult:
    """All signals extracted from a video, consumed by the planner."""
    src_w: int = 0
    src_h: int = 0
    fps: float = 30.0
    duration_ms: int = 0
    total_frames: int = 0

    # time_ms → list of face bboxes [{x,y,w,h,cx,cy}]
    face_timeline: Dict[int, list] = field(default_factory=dict)

    # Scene cut timestamps
    scene_cuts: List[int] = field(default_factory=list)

    # time_ms → motion magnitude (0-inf, typically 0-20)
    motion_timeline: Dict[int, float] = field(default_factory=dict)

    # time_ms → motion hotspot {cx, cy, intensity} for faceless content
    # This is where the most movement is happening in the frame
    motion_hotspot: Dict[int, dict] = field(default_factory=dict)

    # Speaker diarization: time_ms → speaker_id (from audio analysis)
    # Maps each moment to which speaker is active (e.g. "SPEAKER_00", "SPEAKER_01")
    speaker_timeline: Dict[int, str] = field(default_factory=dict)

    # Track-to-speaker mapping: track_id → speaker_id
    # Links face tracks to audio speakers via spatial/temporal correlation
    track_speaker_map: Dict[int, str] = field(default_factory=dict)

    # Speech activity: time_ms → bool (True = someone is speaking)
    # From Whisper transcription timestamps — more accurate than motion heuristics
    speech_active: Dict[int, bool] = field(default_factory=dict)

    # Transcription segments: [{start_sec, end_sec, text}]
    # Full transcript with timestamps, used for subtitle generation and context
    transcript_segments: List[dict] = field(default_factory=list)

    # Detected language (from Whisper auto-detect)
    detected_language: str = ""

    # Per-100ms audio RMS energy (from WAV extraction)
    # Used for audio-visual correlation to identify speakers without pyannote
    audio_rms: Dict[int, float] = field(default_factory=dict)

    # ── TACT Coverage Ledger ──
    # Replaces speech_active for TACT-enabled runs. Provides per-bin status,
    # confidence, speaker attribution, and event type for every millisecond.
    # When None, the pipeline falls back to boolean speech_active.
    coverage_ledger: Optional[CoverageLedger] = None

    # Non-speech audio event timeline: time_ms → event_type
    # Populated by acoustic event classifier (or inferred from Whisper no_speech_prob)
    # Values: 'music', 'applause', 'laughter', 'silence', 'noise', 'crowd'
    audio_events: Dict[int, str] = field(default_factory=dict)

    # ── Universal Subject Tracking (non-face signals) ──
    # YOLO person detections: time_ms → [{x, y, w, h, cx, cy, area}]
    # For non-face content (anime, sports, gaming), YOLO persons become
    # the primary tracked subject when no face detections are available.
    person_timeline: Dict[int, list] = field(default_factory=dict)

    # Saliency hotspot: time_ms → {cx, cy, intensity}
    # Spectral residual saliency for non-face/non-person frames.
    # Identifies the most visually important region regardless of motion.
    saliency_hotspot: Dict[int, dict] = field(default_factory=dict)

    @property
    def is_live_action(self) -> bool:
        """Detect if this is live-action content with real human faces.

        Live-action content (interviews, podcasts, battles, vlogs, TV shows)
        has: high percentage of frames with faces, persistent face tracks,
        and real human features. Animated/gaming/sports content does NOT.

        When True, the planner uses aggressive face-centering behavior:
        the camera should ALWAYS have a human face centered in frame."""
        if not self.face_timeline:
            return False
        total = len(self.face_timeline)
        with_faces = sum(1 for v in self.face_timeline.values() if v)
        face_pct = with_faces / max(1, total)

        # Live-action: ≥70% of samples have faces
        # (anime/gaming typically <30%, sports <10%)
        return face_pct >= 0.70


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL-DERIVED ADAPTIVE PARAMETERS — Universal reframing without genres
# ═══════════════════════════════════════════════════════════════════════════
#
#  Instead of classifying content into genres and applying preset rules,
#  measure continuous signal properties and derive parameters from them.
#  The same measurements naturally produce different behavior:
#    - Podcast: high face persistence, high speech, low motion
#              → large deadzone, cut on speaker change, strong anchoring
#    - Gaming: low face density, high motion, low speech
#              → tiny deadzone, follow motion hotspot, fast transitions
#    - TV drama: variable faces, medium motion, dialogue-driven
#              → medium deadzone, ease transitions, speaker-follows
#    - Sports: no faces, extreme motion, high cut rate
#              → follow action, minimal deadzone, linear tracking
#
#  No genre labels needed — parameters derive from signal measurements.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SceneSignals:
    """Continuous measurements of a scene's visual/audio characteristics.
    These are MEASURED from the perception data, not assumed from genre."""
    face_density: float = 0.0          # avg faces per sample (0-10+)
    face_persistence: float = 0.0      # avg fraction of scene each track is visible (0-1)
    spatial_spread: float = 0.0        # max face spread / crop_w (0-2+; >1 = can't fit both)
    motion_energy: float = 0.0         # avg frame-to-frame motion magnitude (0-inf)
    speech_density: float = 0.0        # fraction of samples with speech (0-1)
    speaker_alternation: float = 0.0   # how often the dominant mouth-mover switches (0-1)
    face_area_ratio: float = 0.0       # avg face area / frame area (0-1)
    cut_rate_per_min: float = 0.0      # source scene cuts per minute
    motion_concentration: float = 0.0  # how spatially concentrated motion is (0-1)
    dominant_track_share: float = 0.0  # fraction of scene the most-seen track is visible (0-1)

    # ── TACT-derived signals (populated when coverage ledger is available) ──
    music_density: float = 0.0         # fraction of scene with music event (0-1)
    silence_density: float = 0.0       # fraction of scene with confirmed silence (0-1)
    crowd_density: float = 0.0         # fraction with applause/crowd/laughter (0-1)
    confidence_mean: float = 1.0       # mean ASR confidence across speech bins (0-1)
    quarantine_ratio: float = 0.0      # fraction of quarantined (hallucination) bins (0-1)

    # ── Universal subject signals ──
    person_density: float = 0.0        # fraction of samples with YOLO person detections (0-1)
    saliency_strength: float = 0.0     # mean saliency hotspot intensity (0-1)


@dataclass
class AdaptiveParams:
    """Continuous parameters controlling reframe behavior.
    Derived from SceneSignals, not from genre classification."""
    deadzone_px: int = 30              # min movement to trigger crop update
    hold_ms: int = 500                 # min time to hold on a target before switching
    transition_ms: int = 200           # default transition duration
    use_cut: bool = False              # True = hard cut on target switch, False = ease
    speaker_weight: float = 0.5        # mouth/speech weight vs area/motion (0-1)
    anchor_blend: float = 0.5          # 0 = follow live face, 1 = lock to home position
    responsiveness: float = 0.5        # how quickly to react to new targets (0-1)
    motion_weight: float = 0.3         # how much to follow motion signals when no face (0-1)
    strategy_label: str = 'adaptive'   # for logging only — NOT used for behavior branching




# ═══════════════════════════════════════════════════════════════════════════
#  COMPATIBILITY ALIAS
# ═══════════════════════════════════════════════════════════════════════════
# Fez ships its own RenderPlan (backend.services.render_plan) — a different
# structure built from RenderOp/MotionKeypoint. The reframer's RenderPlan above
# stores (time_ms, x) keyframes + strategy_log. reframer_bridge.py imports this
# one under the name ``ReframerRenderPlan`` so the two never get confused.
ReframerRenderPlan = RenderPlan
