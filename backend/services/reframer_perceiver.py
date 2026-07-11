"""ClipAI Reframer — Perceiver (Stage 1 — perception).

Extracted from clipai_reframer.py (jyoung2000/60) for the Fez engine
transplant. The original Tkinter GUI is not part of this module.
"""

import cv2
import numpy as np
import json
import subprocess
import threading
import os
import sys
import math
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path

from backend.config import settings
from backend.services.reframer_models import (
    ReframeLogger, get_logger, reset_logger, RenderPlan,
    interpolate_x, clamp_x, _face_overlaps_person,
    LedgerBin, CoverageLedger, PerceptionResult, SceneSignals, AdaptiveParams,
)
from backend.services.reframer_face import FaceDetector
from backend.services.reframer_audio import AudioIntelligence
from backend.services.reframer_diarizer import SpeakerDiarizer

logger = logging.getLogger("clipai.reframer_perceiver")


class Perceiver:
    """
    Stage 1: Extract faces, scenes, motion from video.

    Face detection: DNN ResNet-10 SSD (auto-downloaded, ~50x more accurate
    than Haar cascades). Falls back to Haar if model download fails.

    Performance:
      - SEEK to sample frames (skip non-analyzed frames)
      - Downscale to 640px for detection, scale coords back
      - Frame differencing for motion + spatial hotspot
      - Temporal smoothing + persistent track IDs
    """

    def __init__(self, video_path: str, sample_fps: float = 5.0,
                 source_language: str = 'auto',
                 transcribe_audio_path: Optional[str] = None):
        self.path = video_path
        self.sample_fps = sample_fps
        self.source_language = source_language
        # Optional pre-separated vocal stem to transcribe instead of the raw
        # video audio (vocal-separation stage). None → extract from video.
        self.transcribe_audio_path = transcribe_audio_path
        self.cancelled = False

        # Tiered face detector: YOLO → DNN → Haar
        self.face_detector = FaceDetector()

        # Audio intelligence: Whisper transcription for speech detection.
        # Model is the one picked in Settings > Models (persisted as WHISPER_MODEL).
        self.audio_intel = AudioIntelligence(model_name=settings.WHISPER_MODEL or 'small')

        # Speaker diarization (optional — needs pyannote + HF token)
        self.diarizer = SpeakerDiarizer()

        # Track state
        self._next_track_id = 0
        self._active_tracks: List[dict] = []
        # Persistent person (YOLO subject) tracker — separate id space so
        # person and face track_ids never collide in the overlay.
        self._next_person_track_id = 0
        self._active_person_tracks: List[dict] = []
        self._prev_gray_for_motion = None
        self._prev_frame_for_face_motion = None

    def _sal_meshgrid(self, sal_size: int):
        """Cached ``(yy, xx) = np.mgrid[0:sal_size, 0:sal_size]``.

        The speaker-bump block in ``run()`` and the motion prior in
        ``_fuse_saliency`` rebuilt this pair on every no-face sample —
        thousands of allocations per run on gameplay/anime content. Same
        pattern as ``_center_prior_cache``; the arrays are only ever read.
        Identical numerics.
        """
        mg = getattr(self, '_sal_meshgrid_cache', None)
        if mg is None or mg[0].shape[0] != sal_size:
            yy, xx = np.mgrid[0:sal_size, 0:sal_size]
            mg = (yy, xx)
            self._sal_meshgrid_cache = mg
        return mg

    def _open_capture(self, log) -> "cv2.VideoCapture":
        """Open the source video, preferring hardware-accelerated decode.

        CPU-decoding the full source is the dominant non-Whisper cost of the
        perception band on long videos. When ``REFRAMER_CV2_HWACCEL`` is on
        (and this OpenCV build exposes the knob), ask the FFmpeg backend for
        ``VIDEO_ACCELERATION_ANY``. This changes only WHERE decoding happens:
        grab()/retrieve()/CAP_PROP_POS_FRAMES seeks and the returned BGR
        frames behave identically, so all downstream sampling/tracking logic
        is untouched.

        Fail-soft: if the hw-accel capture can't open, its first read fails,
        or it can't be rewound to frame 0, it is released and the plain
        software constructor is used — exactly the previous behavior.
        The winning path is recorded on ``self._decode_path``
        (``hw(any)`` / ``sw``) and logged at the PERCEIVE stage.
        """
        self._decode_path = 'sw'
        if (getattr(settings, 'REFRAMER_CV2_HWACCEL', True)
                and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION')
                and hasattr(cv2, 'VIDEO_ACCELERATION_ANY')):
            hw_cap = None
            try:
                hw_cap = cv2.VideoCapture(
                    self.path, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
                if hw_cap is not None and hw_cap.isOpened():
                    ret, probe = hw_cap.read()
                    if (ret and probe is not None
                            and hw_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            and int(hw_cap.get(cv2.CAP_PROP_POS_FRAMES)) == 0):
                        self._decode_path = 'hw(any)'
                        log.log_stage('PERCEIVE', 'VideoCapture decode=hw(any) '
                                      '(FFmpeg VIDEO_ACCELERATION_ANY)')
                        return hw_cap
                if hw_cap is not None:
                    hw_cap.release()
            except Exception as hw_err:
                logger.info("hw-accel VideoCapture unavailable (%s); "
                            "falling back to software decode", hw_err)
                try:
                    if hw_cap is not None:
                        hw_cap.release()
                except Exception:
                    pass
        cap = cv2.VideoCapture(self.path)
        log.log_stage('PERCEIVE', 'VideoCapture decode=sw')
        return cap

    def run(self, on_progress: Callable = None) -> PerceptionResult:
        log = get_logger()
        log.start_timer('perceive')
        log.log_stage('PERCEIVE', f'Starting analysis: {self.path}',
                       sample_fps=self.sample_fps)

        cap = self._open_capture(log)
        if not cap.isOpened():
            log.log_error('PERCEIVE', f'Cannot open: {self.path}')
            raise FileNotFoundError(f"Cannot open: {self.path}")

        r = PerceptionResult()
        r.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        r.src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        r.src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        r.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        r.duration_ms = int(r.total_frames / r.fps * 1000)

        log.log_stage('PERCEIVE', f'Source: {r.src_w}x{r.src_h} @ {r.fps:.1f}fps, '
                       f'{r.total_frames} frames, {r.duration_ms/1000:.1f}s')
        log.log_stage('PERCEIVE', f'Face detector: {self.face_detector.tier}')

        # ── Optimization #1: overlap remote Whisper with face detection ──
        # When transcription runs on a REMOTE GPU (the paired Companion), it no
        # longer competes with YOLO for the local card's VRAM — the old reason
        # these stages were serialized. Kick it off NOW, on a background thread,
        # so the local GPU detects faces while the Companion GPU transcribes.
        # Byte-identical result: the SAME transcribe() call, just started earlier
        # and joined before we assemble the transcript. Local Whisper keeps the
        # safe sequential ordering below (it shares the card with YOLO).
        _txn: dict = {"result": None, "error": None}
        _txn_thread = None
        try:
            from backend.services.reframer_audio import remote_whisper_configured
            _txn_remote = bool(remote_whisper_configured())
        except Exception:
            _txn_remote = False
        if _txn_remote:
            import threading as _threading

            def _run_remote_txn(dur_ms=r.duration_ms):
                try:
                    _stem = self.transcribe_audio_path
                    if callable(_stem):
                        try:
                            _stem = _stem()
                        except Exception:
                            _stem = None
                    # Only run CONCURRENTLY when it will actually go to the remote
                    # GPU — and remote_only=True so a remote failure never loads
                    # local Whisper here (that would fight YOLO for the card). On
                    # failure the main thread runs the normal (local-capable) pass
                    # sequentially, after the local GPU is freed.
                    if self.audio_intel.try_load() and self.audio_intel.device_used == 'remote':
                        _txn["result"] = self.audio_intel.transcribe(
                            self.path, dur_ms,
                            language=self.source_language,
                            on_progress=None,  # don't fight the face-loop's bar
                            audio_path_override=_stem,
                            remote_only=True)
                    else:
                        _txn["result"] = {'_remote_failed': True}
                except Exception as _e:  # noqa: BLE001 — reported, then retried
                    _txn["error"] = _e

            _txn_thread = _threading.Thread(
                target=_run_remote_txn, name="clipai-remote-whisper", daemon=True)
            _txn_thread.start()
            log.log_stage(
                'PERCEIVE',
                'Remote Whisper transcription started CONCURRENTLY with face '
                'detection (runs on the Companion GPU — no local VRAM contention)')

        # ── Optimization #2: overlap speaker diarization with the visual pass ──
        # Diarization is audio-only — it doesn't need faces or the transcript
        # until we LINK tracks to speakers — so run pyannote on a background
        # thread concurrently with the face loop. Its models are tiny and
        # co-reside with YOLO on the local card. Gated on remote Whisper so the
        # single-GPU path keeps its safe sequential ordering (local Whisper would
        # otherwise contend with a still-running diarization). Worst case we just
        # join and wait — never slower than the old order; the local-embedding
        # fallback (needs the transcript) still runs sequentially below.
        _diar: dict = {"timeline": None, "error": None}
        _diar_thread = None
        if _txn_remote:
            import threading as _threading_d

            def _run_diar(dur_ms=r.duration_ms):
                try:
                    if self.diarizer.try_load():
                        _diar["timeline"] = self.diarizer.diarize(self.path, dur_ms)
                except Exception as _e:  # noqa: BLE001 — degrade to the fallback
                    _diar["error"] = _e

            _diar_thread = _threading_d.Thread(
                target=_run_diar, name="clipai-diarize", daemon=True)
            _diar_thread.start()
            log.log_stage('PERCEIVE',
                          'Speaker diarization started concurrently with face detection')

        # Detection resolution: scale down for speed, but keep enough pixels
        # to resolve small faces on high-res sources. 720p/1080p → 640px wide,
        # 1440p+ → 960px wide, 4K+ → 1280px wide. The extra pixels matter
        # most for tiled and YOLO-assisted YuNet passes.
        if r.src_w >= 3840:
            target_det_w = 1280
        elif r.src_w >= 2560:
            target_det_w = 960
        else:
            target_det_w = 640
        det_scale = min(1.0, float(target_det_w) / r.src_w)
        det_w = int(r.src_w * det_scale)
        det_h = int(r.src_h * det_scale)
        log.log_stage('PERCEIVE', f'Detection resolution: {det_w}x{det_h} '
                       f'(scale {det_scale:.2f}x)')

        # Min/max face sizes at detection resolution
        min_face = max(20, int(det_w * 0.04))
        max_face = int(det_w * 0.80)

        # Compute which frames to sample (seek directly to them)
        sample_interval_ms = 1000.0 / self.sample_fps
        sample_times_ms = []
        t = 0.0
        while t < r.duration_ms:
            sample_times_ms.append(int(t))
            t += sample_interval_ms
        total_samples = len(sample_times_ms)
        log.log_stage('PERCEIVE', f'Will analyze {total_samples} samples '
                       f'({self.sample_fps} fps, every {sample_interval_ms:.0f}ms)')

        # Sparse sampling (>1.5 s between looks): the YOLO stride cache would
        # gate faces with multi-second-old person boxes — run YOLO on EVERY
        # sample instead. The no-face branch below reuses that same inference
        # (no duplicate predicts), so this trades a stale gate for a fresh
        # one at near-zero net cost on long videos.
        try:
            if (sample_interval_ms > 1500
                    and getattr(self.face_detector, '_yolo_stride', 1) > 1):
                self.face_detector._yolo_stride = 1
        except Exception:
            pass

        prev_gray_small = None
        prev_hist = None
        recent_raw: List[List[dict]] = []
        self._prev_hotspot_cx = None
        self._prev_hotspot_cy = None
        self._prev_sal_cx = None   # temporal EMA state for the saliency hotspot
        self._prev_sal_cy = None
        self._saliency_source_counts = {}  # {'u2netp+stack': n, 'spectral+stack': n, ...}

        # ════════════════════════════════════════════════════════════════
        #  YOLO-World Auto-Discovery Pass
        #  Sample a few frames, run YOLO-World with a broad vocabulary,
        #  identify what's actually in the video, then narrow the vocab
        #  to just those classes for the full analysis.
        #
        #  Like a human editor watching 30 seconds to understand:
        #  "this is about marbles" or "this is a mecha anime"
        # ════════════════════════════════════════════════════════════════
        if (self.face_detector._yolo_model is not None
                and hasattr(self.face_detector, '_yolo_classes')
                and self.face_detector._yolo_classes):

            # Broad discovery vocabulary — covers all common subjects
            _DISCOVERY_VOCAB = [
                # People
                "person", "head", "face", "character", "child",
                # Vehicles
                "car", "truck", "bus", "motorcycle", "bicycle",
                "boat", "airplane", "train", "spacecraft",
                # Animals
                "dog", "cat", "bird", "horse", "fish", "animal",
                # Objects
                "ball", "marble", "toy", "bottle", "cup", "phone",
                "book", "bag", "box", "food",
                # Equipment / tech
                "camera", "microphone", "instrument", "computer", "screen",
                # Anime / gaming
                "robot", "mecha", "weapon", "sword", "gun", "armor", "helmet",
                # Nature / structures
                "tree", "flower", "building", "sign",
                # Sports
                "goal", "net", "racket",
                # Background surfaces with face-like content (false positive suppression)
                "painting", "poster", "picture", "frame", "artwork", "display",
            ]

            try:
                # Sample 8 evenly-spaced frames for discovery
                discovery_times = []
                for di in range(8):
                    dt = int(r.duration_ms * (di + 1) / 9)
                    discovery_times.append(dt)

                # Set broad vocab for discovery. The helper re-syncs the
                # model to the GPU after set_classes — without it the new
                # class-embedding tensors land on CPU and predict() trips
                # "Expected all tensors to be on the same device".
                self.face_detector._yolo_set_classes(_DISCOVERY_VOCAB)

                class_counts = {}
                class_confs = {}

                for dt in discovery_times:
                    cap.set(cv2.CAP_PROP_POS_MSEC, dt)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    # Resize for speed
                    if frame.shape[1] > 640:
                        scale = 640 / frame.shape[1]
                        frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))

                    try:
                        # _yolo_predict() runs on the FaceDetector's chosen
                        # device (GPU when available) and handles CPU fallback.
                        results = self.face_detector._yolo_predict(
                            frame, verbose=False, conf=0.25, max_det=20)
                        for r_det in results:
                            if r_det.boxes is not None:
                                for box in r_det.boxes:
                                    cls_id = int(box.cls[0])
                                    conf_val = float(box.conf[0])
                                    if cls_id < len(_DISCOVERY_VOCAB):
                                        cls_name = _DISCOVERY_VOCAB[cls_id]
                                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                                        class_confs[cls_name] = max(
                                            class_confs.get(cls_name, 0), conf_val)
                    except Exception:
                        pass

                # Build final vocab: always include "person" + "head",
                # plus any class detected 2+ times with decent confidence
                final_classes = ["person", "head", "face"]
                for cls_name, count in sorted(class_counts.items(),
                                              key=lambda x: -x[1]):
                    if cls_name in final_classes:
                        continue
                    if count >= 2 or class_confs.get(cls_name, 0) > 0.5:
                        final_classes.append(cls_name)
                    if len(final_classes) >= 15:
                        break

                # Set the narrowed vocab for the full analysis (helper
                # re-syncs the model to GPU to avoid the mixed-device crash
                # this used to produce on CUDA runs).
                self.face_detector._yolo_set_classes(final_classes)
                self.face_detector._yolo_classes = final_classes

                discovered = [f"{c}({class_counts.get(c, 0)})"
                             for c in final_classes if c in class_counts]
                log.log_stage('PERCEIVE',
                    f'YOLO-World auto-discovery: found {len(class_counts)} object types '
                    f'in 8 sample frames → tracking: {", ".join(discovered) or "person, head, face (defaults)"}')

                # Reset video position
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            except Exception as e:
                log.log_stage('PERCEIVE',
                    f'Auto-discovery failed: {e}. Using default classes.')
                # Reset video position
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Sequential-vs-seek threshold, SAMPLE-RATE-AWARE. The old flat 60-frame
        # threshold predates long-video sampling: at 0.23 fps the stride is
        # ~100-130 frames, so EVERY one of 1800 samples took a hard
        # CAP_PROP_POS_FRAMES seek — each one rewinds to the previous keyframe
        # and re-decodes forward (~half a GOP), flushes the demuxer, and resets
        # the NVDEC pipeline on hw captures. Measured: ~0.5 s of the 0.72 s
        # per-sample cost was this acquisition path, and the gap>1 grab branch
        # (whose _track_through_gap LK bridging exists precisely to fix blind
        # inter-sample stretches) never executed — the direct cause of the
        # observed 1-2-sample face-track fragmentation. Sizing the threshold
        # to ~2.5x the actual sampling stride (bounded) decodes the same
        # target frames via sequential grab() — each source frame decoded
        # exactly once, no seek flushes — and re-enables the gap tracker.
        seek_gap = int(getattr(settings, 'REFRAMER_SEEK_GAP_FRAMES', 60))
        if len(sample_times_ms) >= 2:
            _diffs = sorted(b - a for a, b in
                            zip(sample_times_ms, sample_times_ms[1:]))
            _stride_frames = int(round(
                (_diffs[len(_diffs) // 2] / 1000.0) * max(r.fps, 1e-6)))
            _grab_cap = int(getattr(settings, 'REFRAMER_GRAB_MAX_FRAMES', 450))
            seek_gap = min(max(seek_gap, int(2.5 * _stride_frames)), _grab_cap)

        # Per-bucket wall-time accounting — one summary line at completion so
        # 'where does each sample's time go' is answerable from the job log
        # (the 0.72 s/sample regression was invisible without it).
        import time as _time_mod
        _tbuck = {'acquire': 0.0, 'detect': 0.0, 'loop': 0.0}
        _loop_t0 = _time_mod.perf_counter()

        for i, time_ms in enumerate(sample_times_ms):
            if self.cancelled:
                break

            # Smart frame reading: sequential grab() for anything up to
            # ~2.5x the sampling stride (see threshold above), hard-seek only
            # for genuinely large jumps. grab() skips frames without color
            # convert / return, which is far cheaper than a seek for these
            # gaps — and the gap branch bridges them with LK tracking.
            _t_acq = _time_mod.perf_counter()
            target_frame = int(time_ms / 1000.0 * r.fps)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            gap = target_frame - current_pos

            if gap < 0 or gap > seek_gap:
                # Behind (shouldn't happen) or a genuinely large jump
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(target_frame, r.total_frames - 1))
            elif gap > 1:
                # Skip forward by grabbing without decoding — but bridge the
                # gap with LIGHTWEIGHT TRACKING when the last sample had
                # faces. At 0.14-0.23 samples/s on long videos the camera
                # path is pure interpolation for up to 7s between looks; LK
                # optical flow on a few of the already-grabbed frames turns
                # those blind stretches into real subject positions (the
                # dominant source of HIGH face_missing problems).
                self._track_through_gap(cap, gap, r, det_w, det_h, det_scale)

            ret, frame = cap.read()
            _tbuck['acquire'] += _time_mod.perf_counter() - _t_acq
            if not ret:
                continue

            # Downscale ONCE — INTER_LINEAR is 2x faster than INTER_AREA
            # and visually identical at this scale ratio
            small_bgr = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            gray_small = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

            # ── Face detection at low res ──
            _t_det = _time_mod.perf_counter()
            raw_faces = self._detect_faces_fast(
                gray_small, small_bgr, min_face, max_face, det_w, det_h, det_scale)
            _tbuck['detect'] += _time_mod.perf_counter() - _t_det

            # Temporal smoothing
            recent_raw.append(raw_faces)
            if len(recent_raw) > 4:
                recent_raw.pop(0)
            confirmed = self._temporal_filter(recent_raw)

            # Track assignment
            tracked = self._assign_tracks(confirmed, time_ms)
            r.face_timeline[time_ms] = tracked

            # State for the inter-sample LK tracker (bridges the next gap).
            self._last_gray_small = gray_small
            self._last_sample_ms = time_ms

            # ── Cache dominant-speaker position for saliency biasing ──
            # When the speaker's face momentarily drops out of detection
            # (head turn, brief occlusion, motion blur), the saliency
            # hotspot can still pull toward the last known speaker by
            # adding a time-decaying Gaussian bump (see saliency block
            # below). We only cache when mouth motion is clearly above
            # the noise floor.
            if tracked:
                mm_face = max(tracked, key=lambda f: f.get('mouth_motion', 0))
                mm_strength = float(mm_face.get('mouth_motion', 0))
                if mm_strength > 0.10:
                    self._recent_speaker_cx = int(mm_face.get('cx', 0))
                    self._recent_speaker_cy = int(mm_face.get('cy', 0))
                    self._recent_speaker_time_ms = time_ms
                    self._recent_speaker_strength = mm_strength

            # ── Scene cuts (histogram at low res — very fast) ──
            hist = cv2.calcHist([gray_small], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > 0.35:
                    r.scene_cuts.append(time_ms)
                    # Nonhuman subject positions from the previous scene
                    # don't apply after a cut — reset the spatial cache.
                    self.face_detector.clear_nonhuman_cache()
            prev_hist = hist.copy()

            # ── Motion (dense optical flow + camera-motion compensation) ──
            # Farnebäck dense optical flow gives per-pixel motion vectors,
            # so we can locate the true center of OBJECT motion vs the old
            # absdiff approach which conflated "moving" with "high-contrast
            # texture". Subtracting the median flow removes the global
            # camera-pan / shake bias so the centroid follows real subject
            # movement instead of drifting with the camera. Falls back to
            # the previous frame-diff method on any opencv failure so a
            # missing build flag never crashes the perceiver.
            if prev_gray_small is not None:
                try:
                    # Downsample to half det resolution before computing flow.
                    # Farnebäck complexity is O(pixels), so halving each
                    # dimension gives 4x speedup. For locating the motion
                    # centroid we don't need pixel-level accuracy — 320px
                    # wide is more than enough to distinguish left vs right.
                    of_h = max(60, det_h // 2)
                    of_w = max(80, det_w // 2)
                    prev_of = cv2.resize(prev_gray_small, (of_w, of_h),
                                         interpolation=cv2.INTER_LINEAR)
                    curr_of = cv2.resize(gray_small,      (of_w, of_h),
                                         interpolation=cv2.INTER_LINEAR)
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_of, curr_of, None,
                        0.5,   # pyr_scale
                        1,     # levels  (1 = faster, sufficient for centroid)
                        9,     # winsize (smaller = faster)
                        1,     # iterations
                        5,     # poly_n
                        1.1,   # poly_sigma
                        0,     # flags
                    )
                    fx = flow[..., 0]
                    fy = flow[..., 1]
                    # Camera-motion compensation: subtract median flow
                    fx = fx - float(np.median(fx))
                    fy = fy - float(np.median(fy))
                    mag = np.sqrt(fx * fx + fy * fy)
                    mean_mag = float(np.mean(mag))
                    # Scale to roughly match the legacy motion_timeline units
                    # so downstream thresholds (e.g. motion_hotspot intensity
                    # gate of 0.01) remain meaningful.
                    motion_mag = mean_mag * 5.0
                    r.motion_timeline[time_ms] = motion_mag

                    total_mag = float(mag.sum())
                    flow_h_actual, flow_w_actual = mag.shape
                    if total_mag > 0.5:
                        gy_idx, gx_idx = np.mgrid[0:flow_h_actual, 0:flow_w_actual]
                        centroid_x_small = float(np.sum(gx_idx * mag)) / total_mag
                        centroid_y_small = float(np.sum(gy_idx * mag)) / total_mag
                    else:
                        centroid_x_small = flow_w_actual / 2.0
                        centroid_y_small = flow_h_actual / 2.0
                    # Scale centroid back to source resolution (flow was at
                    # of_w×of_h, det was det_w×det_h, source is /det_scale)
                    centroid_cx = int(centroid_x_small * det_w / max(1, flow_w_actual) / det_scale)
                    centroid_cy = int(centroid_y_small * det_h / max(1, flow_h_actual) / det_scale)
                    # Intensity ~ peak flow magnitude, normalised so 1.0 ≈
                    # large displacement. Capped to match legacy 0–1 range.
                    best_intensity = float(min(1.0, np.percentile(mag, 99) / 10.0))
                except Exception:
                    # Fallback to legacy frame-diff path
                    diff_frame = cv2.absdiff(prev_gray_small, gray_small)
                    motion_mag = float(np.mean(diff_frame)) / 255.0 * 10.0
                    r.motion_timeline[time_ms] = motion_mag
                    grid_rows, grid_cols = 3, 6
                    cell_h = det_h // grid_rows
                    cell_w = det_w // grid_cols
                    cropped_diff = diff_frame[:cell_h * grid_rows, :cell_w * grid_cols]
                    grid = cropped_diff.reshape(grid_rows, cell_h, grid_cols, cell_w)
                    cell_means = grid.mean(axis=(1, 3))
                    best_idx = np.argmax(cell_means)
                    best_gy, best_gx = divmod(best_idx, grid_cols)
                    best_intensity = float(cell_means[best_gy, best_gx]) / 255.0
                    total_motion = float(cell_means.sum())
                    if total_motion > 0.1:
                        gy_indices, gx_indices = np.mgrid[0:grid_rows, 0:grid_cols]
                        centroid_gx = float(np.sum(gx_indices * cell_means)) / total_motion
                        centroid_gy = float(np.sum(gy_indices * cell_means)) / total_motion
                        centroid_cx = int((centroid_gx + 0.5) * cell_w / det_scale)
                        centroid_cy = int((centroid_gy + 0.5) * cell_h / det_scale)
                    else:
                        centroid_cx = int((best_gx + 0.5) * cell_w / det_scale)
                        centroid_cy = int((best_gy + 0.5) * cell_h / det_scale)

                # EMA smooth the hotspot to prevent frame-to-frame jumping.
                # Dynamic alpha: high motion = more responsive tracking.
                raw_cx = centroid_cx
                raw_cy = centroid_cy
                alpha_hs = min(0.7, 0.25 + best_intensity * 0.8)  # 0.25–0.70
                if hasattr(self, '_prev_hotspot_cx') and self._prev_hotspot_cx is not None:
                    raw_cx = int(alpha_hs * centroid_cx + (1 - alpha_hs) * self._prev_hotspot_cx)
                    raw_cy = int(alpha_hs * centroid_cy + (1 - alpha_hs) * self._prev_hotspot_cy)
                self._prev_hotspot_cx = raw_cx
                self._prev_hotspot_cy = raw_cy

                r.motion_hotspot[time_ms] = {
                    'cx': raw_cx,
                    'cy': raw_cy,
                    'intensity': round(best_intensity, 4),
                }

            # ── Non-face subject tracking (YOLO-World + saliency) ──
            # When no face is detected, reuse the subject boxes the face
            # detector's OWN YOLO pass already computed for this exact frame
            # (detect() caches the full unsplit class list). The old code ran
            # a SECOND identical YOLO-World inference here on every 4th
            # faceless sample — pure duplicate work — and its 1-second
            # staleness window discarded everything at sparse (>1 s) sampling
            # strides, leaving the person timeline nearly empty on exactly
            # the content that needs it most.
            has_faces = bool(tracked)

            if not has_faces:
                person_bboxes = list(getattr(
                    self.face_detector, '_cached_subject_bboxes_all', []) or [])
                if (not person_bboxes
                        and getattr(self.face_detector, 'tier', '') != 'yunet'
                        and self.face_detector._yolo_model is not None):
                    # DNN/Haar detector tiers never feed the cache (their
                    # detect() has no YOLO gate) — keep the throttled direct
                    # call for them. In the yunet tier an empty cache means
                    # YOLO genuinely found no subjects on its last pass.
                    if (i % 4 == 0) or not hasattr(self, '_last_person_bboxes'):
                        person_bboxes = self.face_detector._get_person_bboxes(small_bgr)
                        self._last_person_bboxes = person_bboxes
                    else:
                        person_bboxes = getattr(self, '_last_person_bboxes', [])
                if person_bboxes:
                    persons = []
                    for px1, py1, px2, py2 in person_bboxes:
                        pw, ph = px2 - px1, py2 - py1
                        # Scale back to source resolution
                        sx1 = int(px1 / det_scale)
                        sy1 = int(py1 / det_scale)
                        sw = int(pw / det_scale)
                        sh = int(ph / det_scale)
                        persons.append({
                            'x': sx1, 'y': sy1, 'w': sw, 'h': sh,
                            'cx': sx1 + sw // 2, 'cy': sy1 + sh // 2,
                            'area': sw * sh,
                        })
                    # Assign persistent per-person track_ids (item 1) so the
                    # preview overlay can interpolate a box between sparse
                    # samples without cross-fading two different subjects.
                    persons = self._assign_person_tracks(persons, time_ms)
                    r.person_timeline[time_ms] = persons

                # Spectral residual saliency (pure numpy, ~2ms per frame)
                # Finds the most "visually unexpected" region — works for any
                # content type without genre-specific tuning.
                if not person_bboxes:
                    try:
                        _used_u2net = False  # saliency-source telemetry (item 8)
                        # Spectral residual: FFT → log amplitude → smooth → residual → IFFT
                        sal_size = 128  # higher res = better object localization
                        sal_gray = cv2.resize(gray_small, (sal_size, sal_size))
                        sal_f = np.fft.fft2(sal_gray.astype(np.float32))
                        log_amp = np.log(np.abs(sal_f) + 1e-5)
                        phase = np.angle(sal_f)
                        # Smooth the log amplitude spectrum
                        kernel = np.ones((3, 3)) / 9.0
                        smooth_amp = cv2.filter2D(log_amp, -1, kernel)
                        # Spectral residual = original - smoothed
                        residual = log_amp - smooth_amp
                        # Reconstruct saliency map from residual
                        sal_complex = np.exp(residual + 1j * phase)
                        sal_map = np.abs(np.fft.ifft2(sal_complex)) ** 2
                        sal_map = cv2.GaussianBlur(sal_map.astype(np.float32), (7, 7), 3.0)
                        # Normalize
                        sal_max = sal_map.max()
                        if sal_max > 0:
                            sal_map /= sal_max
                        # ── Learned salient-object base (item 8, gated) ──
                        # When u2netp is enabled + available, swap the spectral
                        # edge map for its subject mask (where a human looks,
                        # not what stands out). Suppression + fusion below still
                        # apply. Falls back to spectral on any failure.
                        if getattr(settings, 'REFRAMER_U2NET_SALIENCY', False):
                            try:
                                from backend.services.reframer_u2net import u2net_saliency
                                _u2 = u2net_saliency(small_bgr, sal_size)
                                if _u2 is not None:
                                    sal_map = _u2
                                    _used_u2net = True
                            except Exception:
                                pass
                        # ── Text/watermark saliency suppression ──
                        # Bottom 15%: subtitles, watermarks, channel logos
                        # Top 5%: letterbox bars, UI chrome
                        # Text has high edge energy that attracts the saliency
                        # detector, but a camera op ignores it.
                        sh = sal_map.shape[0]
                        sal_map[int(sh * 0.85):, :] *= 0.1   # bottom 15%
                        sal_map[:int(sh * 0.05), :] *= 0.3   # top 5%
                        # ── Mouth-motion bias ──
                        # Saliency only fires when no face is detected this
                        # frame, but the speaker may have been visible 200–
                        # 1500ms ago (head turn, blink, brief occlusion).
                        # Add a time-decaying Gaussian bump at the last
                        # known dominant-speaker position so the no-face
                        # fallback in the planner still tracks the speaker
                        # rather than wandering to a high-contrast logo.
                        rec_t = getattr(self, '_recent_speaker_time_ms', -10000)
                        gap_ms = time_ms - rec_t
                        DECAY_WINDOW_MS = 2000
                        if 0 < gap_ms < DECAY_WINDOW_MS:
                            decay = 1.0 - (gap_ms / DECAY_WINDOW_MS)
                            sp_strength = float(getattr(
                                self, '_recent_speaker_strength', 0.0))
                            sp_cx = int(getattr(self, '_recent_speaker_cx', 0))
                            sp_cy = int(getattr(self, '_recent_speaker_cy', 0))
                            sp_x_sal = int(sp_cx / max(1, r.src_w) * sal_size)
                            sp_y_sal = int(sp_cy / max(1, r.src_h) * sal_size)
                            sp_x_sal = max(0, min(sal_size - 1, sp_x_sal))
                            sp_y_sal = max(0, min(sal_size - 1, sp_y_sal))
                            yy, xx = self._sal_meshgrid(sal_size)
                            sigma = sal_size * 0.10
                            bump = np.exp(
                                -((xx - sp_x_sal) ** 2 + (yy - sp_y_sal) ** 2)
                                / (2.0 * sigma * sigma)
                            ).astype(np.float32)
                            # Max contribution: 0.5 (when fresh + strong),
                            # scaled by decay × strength so a faint cached
                            # signal never dominates real saliency peaks.
                            bump *= float(decay * min(1.0, sp_strength) * 0.5)
                            sal_map = sal_map + bump
                            sal_max2 = sal_map.max()
                            if sal_max2 > 0:
                                sal_map /= sal_max2
                        # ── Fused saliency stack (item 4) ──
                        # Raw spectral residual latches onto high-frequency
                        # novelty (logos, text, HUD edges). Fuse it with the
                        # priors a human camera op actually uses — center bias,
                        # motion energy, skin/face color — so the argmax lands
                        # on the likely subject instead of the sharpest edge.
                        if getattr(settings, 'REFRAMER_SALIENCY_STACK', True):
                            sal_map = self._fuse_saliency(
                                sal_map, sal_size, small_bgr, r, time_ms)
                        # Find peak saliency location
                        peak_idx = np.argmax(sal_map)
                        peak_y, peak_x = divmod(peak_idx, sal_size)
                        # Scale to source resolution
                        sal_cx = int(peak_x / sal_size * r.src_w)
                        sal_cy = int(peak_y / sal_size * r.src_h)
                        sal_intensity = float(sal_map[peak_y, peak_x])

                        # ── Temporal smoothing of the saliency hotspot ──
                        # The spectral map is near-flat, so a raw per-frame
                        # argmax wanders. EMA the hotspot (as the motion hotspot
                        # already is) so the no-face crop target stays steady.
                        if getattr(settings, 'REFRAMER_SALIENCY_STACK', True):
                            a_sal = 0.35
                            if getattr(self, '_prev_sal_cx', None) is not None:
                                sal_cx = int(a_sal * sal_cx + (1 - a_sal) * self._prev_sal_cx)
                                sal_cy = int(a_sal * sal_cy + (1 - a_sal) * self._prev_sal_cy)
                            self._prev_sal_cx = sal_cx
                            self._prev_sal_cy = sal_cy

                        if sal_intensity > 0.10:
                            # Saliency-source telemetry: which model produced
                            # this hotspot, so an offline reviewer can see when
                            # u2netp fired vs the spectral stack vs raw spectral.
                            _sal_src = 'u2netp' if _used_u2net else 'spectral'
                            if getattr(settings, 'REFRAMER_SALIENCY_STACK', True):
                                _sal_src += '+stack'
                            self._saliency_source_counts[_sal_src] = \
                                self._saliency_source_counts.get(_sal_src, 0) + 1
                            r.saliency_hotspot[time_ms] = {
                                'cx': sal_cx, 'cy': sal_cy,
                                'intensity': round(sal_intensity, 4),
                                'source': _sal_src,
                            }
                    except Exception:
                        pass  # Saliency is optional — never crash the pipeline

            prev_gray_small = gray_small  # cvtColor creates new array each iteration

            if on_progress and i % 20 == 0:
                on_progress(i / total_samples)

            # Periodic progress log every 25%
            if i > 0 and total_samples > 20 and i % (total_samples // 4) == 0:
                pct = int(i / total_samples * 100)
                faces_so_far = sum(1 for t2 in list(r.face_timeline.keys())[:i+1]
                                   if r.face_timeline.get(t2))
                log.log_stage('PERCEIVE',
                    f'  Face detection {pct}%: {faces_so_far} samples with faces '
                    f'({i+1}/{total_samples} processed)')

        _tbuck['loop'] = _time_mod.perf_counter() - _loop_t0
        _n = max(1, len(r.face_timeline))
        _other = max(0.0, _tbuck['loop'] - _tbuck['acquire'] - _tbuck['detect'])
        log.log_stage('PERCEIVE',
            f"  Sample-loop timing: {_tbuck['loop']:.0f}s total "
            f"({_tbuck['loop'] / _n:.2f}s/sample) — acquire "
            f"{_tbuck['acquire']:.0f}s, detect {_tbuck['detect']:.0f}s, "
            f"other {_other:.0f}s")

        cap.release()
        if on_progress:
            on_progress(1.0)

        # Face detection summary
        total_face_samples = sum(1 for faces in r.face_timeline.values() if faces)
        total_faces_detected = sum(len(faces) for faces in r.face_timeline.values())
        unique_tracks = set()
        track_sample_counts = {}
        for faces in r.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    unique_tracks.add(tid)
                    track_sample_counts[tid] = track_sample_counts.get(tid, 0) + 1

        log.log_stage('PERCEIVE',
            f'Face detection complete:\n'
            f'  Total samples: {len(r.face_timeline)}\n'
            f'  Samples with faces: {total_face_samples} '
            f'({total_face_samples*100//max(1,len(r.face_timeline))}%)\n'
            f'  Total detections: {total_faces_detected}\n'
            f'  Unique tracks: {len(unique_tracks)}\n'
            f'  Scene cuts: {len(r.scene_cuts)}\n'
            f'  Detector tier: {self.face_detector.tier}')

        # Log each track's stats
        if track_sample_counts:
            for tid in sorted(track_sample_counts.keys()):
                count = track_sample_counts[tid]
                log.log_stage('PERCEIVE',
                    f'  Track {tid}: {count} samples '
                    f'({count*100//max(1,len(r.face_timeline))}% of video)')

        # ── Track consolidation across scene cuts ──
        # Merge fragmented tracks that are clearly the same person
        self._consolidate_tracks_across_cuts(r)

        # ── Release perception models before Whisper allocates ──
        # On small GPUs (e.g. GTX 1650, 3.7 GB total) the YOLO-World
        # weights + ctranslate2 workspace can't both fit. Detection is
        # finished by this point, so drop the YOLO/SFace handles and
        # flush the CUDA allocator before Whisper loads its budget.
        self._release_perception_models()

        # ── Audio intelligence (Whisper transcription) ──
        # If it was started concurrently (remote GPU, optimization #1), JOIN it
        # now — the transcript ran while faces were detected. Otherwise run it
        # here (local Whisper: shares the card with YOLO, so it stays after the
        # visual pass). transcribe_audio_path may be a zero-arg callable (the
        # pipeline's concurrent vocal-separation handle): resolve it HERE for the
        # sequential path so Demucs got the perception stage's wall time free.
        def _apply_audio_result(audio_result):
            if not audio_result:
                return
            r.speech_active = audio_result.get('speech_active', {})
            r.transcript_segments = audio_result.get('segments', [])
            r.detected_language = audio_result.get('language', '')
            r.coverage_ledger = audio_result.get('coverage_ledger')
            r.audio_events = audio_result.get('audio_events', {})

        _txn_done = False
        if _txn_thread is not None:
            _txn_thread.join()
            _res = _txn["result"]
            # Accept the concurrent result only when it actually carries a
            # transcript. An EMPTY remote result (degenerate decode rejected
            # by the coverage gate, or a silent 200) must trigger the
            # sequential local pass — the run that shipped 0 segments on a
            # 128-min video accepted exactly such a result here. Exception:
            # an empty transcript that carries VAD evidence the audio holds
            # no substantial speech IS the correct answer — re-decoding
            # music/silence locally would only waste minutes to confirm it.
            _ok = (_txn["error"] is None and _res is not None
                   and not _res.get("_remote_failed")
                   and (bool(_res.get("segments"))
                        or bool(_res.get("no_speech_evidence"))))
            if _ok:
                _apply_audio_result(_res)
                _txn_done = True
                log.log_stage('PERCEIVE',
                              'Remote Whisper transcript ready (ran concurrently '
                              'with face detection)')
            else:
                # Concurrent remote attempt failed/deferred — run the normal
                # (local-capable) pass sequentially now that the local GPU is
                # free, so the transcript is never lost and never contends.
                if _txn["error"] is not None:
                    log.log_stage('PERCEIVE',
                                  f'Concurrent transcription failed ({_txn["error"]}) '
                                  '— retrying sequentially')
                else:
                    log.log_stage('PERCEIVE',
                                  'Remote transcription deferred — running the '
                                  'sequential pass now (local GPU is free)')

        if not _txn_done:
            _stem_path = self.transcribe_audio_path
            if callable(_stem_path):
                try:
                    _stem_path = _stem_path()
                except Exception:
                    _stem_path = None
            if self.audio_intel.try_load():
                _apply_audio_result(self.audio_intel.transcribe(
                    self.path, r.duration_ms,
                    language=self.source_language,
                    on_progress=on_progress,
                    audio_path_override=_stem_path))

        # ── Speaker diarization (audio-based) ──
        # Phase hint: diarization can run for many minutes with no per-item
        # progress; the hint lets the pipeline label the bar + heartbeat
        # "speaker diarization" instead of leaving a stale "transcription"
        # message on screen through the whole pass.
        if on_progress:
            try:
                on_progress(1.0, 'diarization')
            except TypeError:
                on_progress(1.0)   # legacy single-arg callback
        if _diar_thread is not None:
            # Concurrent (optimization #2) — join the pass that ran alongside faces.
            _diar_thread.join()
            r.speaker_timeline = _diar.get("timeline")
            if _diar.get("error") is not None:
                log.log_stage('PERCEIVE',
                              f'Concurrent diarization failed ({_diar["error"]}) '
                              '— using the fallback')
            if r.speaker_timeline:
                self._link_tracks_to_speakers(r)
        elif self.diarizer.try_load():
            r.speaker_timeline = self.diarizer.diarize(self.path, r.duration_ms)

            # Link tracks to speakers: for each tracked face, find which speaker
            # is active at the times when that face has the most mouth motion
            if r.speaker_timeline:
                self._link_tracks_to_speakers(r)

        # ── Local audio-embedding diarization (no HF token) ──
        # When pyannote can't load (no HF_TOKEN), do REAL audio diarization with
        # a local SpeechBrain ECAPA model on Whisper's speech segments — before
        # falling back to the visual-only heuristic. Works for off-screen /
        # audio-only voices too. Guarded: if speechbrain isn't installed it
        # no-ops and the visual fallback below still runs.
        if not r.speaker_timeline:
            try:
                from backend.services.local_diarizer import LocalEmbeddingDiarizer
                _local = LocalEmbeddingDiarizer()
                if _local.is_available():
                    r.speaker_timeline = _local.diarize(
                        self.path, r.transcript_segments, r.duration_ms)
                    if r.speaker_timeline:
                        get_logger().log_stage(
                            'PERCEIVE',
                            'Speaker diarization: local SpeechBrain ECAPA '
                            '(audio-based, no HF token)')
                        self._link_tracks_to_speakers(r)
            except Exception as _lde:
                get_logger().log_stage(
                    'PERCEIVE', f'Local diarizer skipped: {str(_lde)[:120]}')

        # ── Spatial pseudo-diarization (fallback when audio diarization unavailable) ──
        # If neither pyannote nor the local audio diarizer produced results,
        # cluster tracks by spatial position to build a pseudo track_speaker_map.
        if not r.track_speaker_map:
            self._build_spatial_speakers(r)

        # ── Audio-visual correlation for speaker identification ──
        # Extract per-100ms RMS energy from the audio track, then correlate
        # each face track's mouth-motion time series with the audio energy.
        # The track with the highest Pearson correlation IS the active speaker.
        # This works without pyannote, without HF tokens — just numpy.
        if r.speech_active and not r.track_speaker_map:
            self._extract_audio_rms(r)
            if r.audio_rms:
                self._correlate_audio_visual(r)

        elapsed = log.stop_timer('perceive')
        total_faces = sum(1 for faces in r.face_timeline.values() if faces)
        log.log_stage('PERCEIVE', 'Analysis complete',
                       elapsed_sec=round(elapsed, 2),
                       face_samples=len(r.face_timeline),
                       samples_with_faces=total_faces,
                       scene_cuts=len(r.scene_cuts),
                       motion_samples=len(r.motion_timeline),
                       source_resolution=f'{r.src_w}x{r.src_h}',
                       detection_resolution=f'{det_w}x{det_h}',
                       source_fps=r.fps,
                       duration_sec=round(r.duration_ms / 1000, 2))

        # Expose saliency-source telemetry so the engine can emit a trace
        # summary (u2netp vs spectral-stack vs spectral usage over the clip).
        r.saliency_source_counts = dict(self._saliency_source_counts)

        return r

    def _release_perception_models(self) -> None:
        """Drop face / YOLO weights from VRAM after detection is finished.

        Called between face-detection and Whisper-load so the
        transcriber inherits a clean allocator on small GPUs. Idempotent
        — repeated calls are no-ops because the references are nulled.
        """
        log = get_logger()
        released = []
        fd = getattr(self, 'face_detector', None)
        if fd is not None:
            for attr in ('_yolo_model', '_sface', '_yunet', '_haar_face', '_haar_eye'):
                if getattr(fd, attr, None) is not None:
                    released.append(attr)
                    try:
                        setattr(fd, attr, None)
                    except Exception:
                        pass
        try:
            import gc as _gc
            _gc.collect()
            _gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                before, _ = _torch.cuda.mem_get_info()
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
                after, _ = _torch.cuda.mem_get_info()
                freed_mb = max(0, (after - before) // (1024 * 1024))
                if released:
                    log.log_stage('PERCEIVE',
                        f'Released perception models ({", ".join(released)}); '
                        f'freed {freed_mb} MB VRAM')
            elif released:
                log.log_stage('PERCEIVE',
                    f'Released perception models ({", ".join(released)})')
        except Exception:
            pass

    def _detect_faces_fast(self, gray, frame_bgr, min_face, max_face,
                       det_w, det_h, scale) -> List[dict]:
        """Detect faces at detection resolution (BGR input, no conversion needed).
        Computes mouth motion from YuNet landmarks (MAR) when available,
        with pixel-diff fallback for non-YuNet detections."""
        raw_faces = self.face_detector.detect(frame_bgr)

        # ── REFRAMER_FIX_PREV_FRAME_MOTION (opt-in, changes outputs) ──
        # Legacy behavior assigns self._prev_frame_for_face_motion INSIDE the
        # per-face loop below, so (a) the 2nd+ face in a frame compares the
        # current gray against itself (motion/mouth_motion always 0 —
        # degrading multi-face speaker detection) and (b) zero-face frames
        # never update the reference (the next comparison spans a stale,
        # seconds-old frame). With the flag on, the previous-frame reference
        # is captured once at entry, used for ALL faces, and updated exactly
        # once per sample (including the zero-face path). Default OFF keeps
        # legacy outputs byte-identical.
        _fix_prev = bool(getattr(settings, 'REFRAMER_FIX_PREV_FRAME_MOTION', False))
        _entry_prev = self._prev_frame_for_face_motion

        validated = []
        for face in raw_faces:
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']

            if fy + fh > det_h * 0.92 and fh < det_h * 0.12:
                continue

            motion_score = 0.0
            mouth_motion = 0.0

            # ── Landmark-based MAR (Mouth Aspect Ratio) ──
            # YuNet gives us mouth corner coordinates. By tracking how the
            # mouth width and nose-to-mouth distance change frame-to-frame,
            # we get a direct speech signal instead of a noisy pixel-diff
            # proxy. This is scale-invariant and lighting-invariant.
            has_landmarks = 'right_mouth' in face and 'left_mouth' in face and 'nose' in face
            used_landmark_mar = False

            _prev_ref = _entry_prev if _fix_prev else self._prev_frame_for_face_motion
            if has_landmarks and _prev_ref is not None:
                try:
                    rm = face['right_mouth']
                    lm = face['left_mouth']
                    nose = face['nose']
                    mouth_w = math.sqrt((rm[0]-lm[0])**2 + (rm[1]-lm[1])**2)
                    mouth_mid = ((rm[0]+lm[0])/2, (rm[1]+lm[1])/2)
                    nose_mouth_dist = math.sqrt((nose[0]-mouth_mid[0])**2 +
                                                 (nose[1]-mouth_mid[1])**2)
                    # Normalize by face width for scale invariance
                    norm_mouth_w = mouth_w / max(1, fw)
                    norm_nose_mouth = nose_mouth_dist / max(1, fh)

                    # Compare with previous frame's landmarks for this face region
                    face_key = (fx // max(1, fw // 2), fy // max(1, fh // 2))  # spatial bucket
                    if hasattr(self, '_prev_landmarks') and face_key in self._prev_landmarks:
                        prev_mw, prev_nm = self._prev_landmarks[face_key]
                        # MAR delta: change in mouth width + change in mouth opening
                        mar_delta = abs(norm_mouth_w - prev_mw) + abs(norm_nose_mouth - prev_nm)
                        # Scale to 0-1 range comparable to pixel-diff values
                        # Typical speaking MAR delta is 0.02-0.15
                        mouth_motion = min(1.0, mar_delta * 5.0)
                        used_landmark_mar = True

                    # Store for next frame
                    if not hasattr(self, '_prev_landmarks'):
                        self._prev_landmarks = {}
                    self._prev_landmarks[face_key] = (norm_mouth_w, norm_nose_mouth)
                except (IndexError, ValueError, TypeError):
                    pass

            if _prev_ref is not None:
                prev_g = _prev_ref
                try:
                    cur_roi = gray[fy:fy+fh, fx:fx+fw]
                    prev_roi = prev_g[fy:fy+fh, fx:fx+fw]
                    if cur_roi.shape == prev_roi.shape and cur_roi.size > 0:
                        target = (96, 96)
                        cur_n = cv2.resize(cur_roi, target,
                                           interpolation=cv2.INTER_LINEAR)
                        prev_n = cv2.resize(prev_roi, target,
                                            interpolation=cv2.INTER_LINEAR)
                        diff = cv2.absdiff(cur_n, prev_n)
                        motion_score = float(np.mean(diff)) / 255.0

                    # Pixel-diff mouth fallback when landmarks unavailable
                    if not used_landmark_mar:
                        mouth_y = fy + int(fh * 0.6)
                        cur_m = gray[mouth_y:fy+fh, fx:fx+fw]
                        prev_m = prev_g[mouth_y:fy+fh, fx:fx+fw]
                        if cur_m.shape == prev_m.shape and cur_m.size > 0:
                            target_m = (64, 24)
                            cur_mn = cv2.resize(cur_m, target_m,
                                                interpolation=cv2.INTER_LINEAR)
                            prev_mn = cv2.resize(prev_m, target_m,
                                                 interpolation=cv2.INTER_LINEAR)
                            diff_m = cv2.absdiff(cur_mn, prev_mn)
                            mouth_motion = float(np.mean(diff_m)) / 255.0
                except (IndexError, ValueError, cv2.error):
                    pass

            if not _fix_prev:
                self._prev_frame_for_face_motion = gray  # reuse, don't copy (legacy: per-face)

            inv_scale = 1.0 / scale
            src_x = int(fx * inv_scale)
            src_y = int(fy * inv_scale)
            src_w = int(fw * inv_scale)
            src_h = int(fh * inv_scale)

            area_score = (src_w * src_h) / max(1, (det_w * det_h) * inv_scale * inv_scale)
            raw_saliency = area_score * 0.3 + motion_score * 0.3 + mouth_motion * 0.4
            # Weight saliency by detection confidence — faces with reduced
            # confidence (background artwork, failed liveness) get lower saliency
            # throughout the pipeline.
            det_conf = face.get('confidence', 0.5)
            saliency = raw_saliency * max(0.2, det_conf)

            # Compute SFace embedding for track identity matching.
            # Uses detection-resolution coordinates and frame.
            embedding = None
            if hasattr(self, '_embedding_counter'):
                self._embedding_counter += 1
            else:
                self._embedding_counter = 0
            # Sample 1 in 3 faces for embedding (balances speed vs coverage)
            if self._embedding_counter % 3 == 0:
                embedding = self.face_detector.compute_embedding(frame_bgr, face)

            # ── Eye-line anchor + gaze yaw from YuNet landmarks (items 5/6) ──
            # eye_cx: eye-midpoint x in SOURCE coords — a steadier framing
            #         anchor than the bbox center (which jitters).
            # yaw:    signed nose-vs-bbox-center asymmetry in [-1,1] used for
            #         gaze-based lead-room. Both are best-effort; absent keys
            #         simply fall back to bbox-center framing / no lead.
            eye_cx = None
            yaw = None
            if 'right_eye' in face and 'left_eye' in face:
                try:
                    re = face['right_eye']
                    le = face['left_eye']
                    eye_mid_det = (re[0] + le[0]) / 2.0
                    eye_cx = int(eye_mid_det * inv_scale)
                except (IndexError, ValueError, TypeError):
                    eye_cx = None
            if 'nose' in face and fw > 0:
                try:
                    nose_x_det = face['nose'][0]
                    bbox_center_det = fx + fw / 2.0
                    offset = (nose_x_det - bbox_center_det) / fw
                    yaw = max(-1.0, min(1.0, 2.0 * offset))
                except (IndexError, ValueError, TypeError):
                    yaw = None

            # ── Optional MediaPipe lips-based MAR (item 5, gated) ──
            # A true 478-landmark lip open/close signal beats the 5-point MAR
            # and the pixel-diff proxy for "who is talking". Runs on CPU and
            # only when explicitly enabled + available; falls back silently.
            if getattr(settings, 'REFRAMER_MEDIAPIPE_MAR', False):
                mp_mar = self._mediapipe_mouth_open(frame_bgr, face, inv_scale)
                if mp_mar is not None:
                    mouth_motion = mp_mar
                    saliency = (area_score * 0.3 + motion_score * 0.3
                                + mouth_motion * 0.4) * max(0.2, det_conf)

            result = {
                'x': src_x, 'y': src_y,
                'w': src_w, 'h': src_h,
                'cx': src_x + src_w // 2,
                'cy': src_y + src_h // 2,
                'area': src_w * src_h,
                'confidence': face.get('confidence', 0.5),
                'motion': round(motion_score, 4),
                'mouth_motion': round(mouth_motion, 4),
                'saliency': round(saliency, 4),
            }
            if eye_cx is not None:
                result['eye_cx'] = eye_cx
            if yaw is not None:
                result['yaw'] = round(yaw, 4)
            if embedding is not None:
                result['embedding'] = embedding
            validated.append(result)

        if _fix_prev:
            # Update ONCE per sample — every call, including zero-face frames,
            # so the reference never goes stale across face-less stretches.
            self._prev_frame_for_face_motion = gray  # reuse, don't copy

        return validated

    def _fuse_saliency(self, sal_map, sal_size, small_bgr, r, time_ms):
        """Fuse spectral saliency with center / motion / skin priors (item 4).

        All inputs are normalized to [0,1] on the ``sal_size × sal_size`` grid
        and combined as a weighted sum, then renormalized. This biases the
        argmax toward where a human looks (center, moving things, skin) instead
        of the highest-frequency edge (logos, captions, HUD chrome).
        """
        try:
            spectral = sal_map
            # Cache the static center-prior Gaussian (favor the frame center).
            cp = getattr(self, '_center_prior_cache', None)
            if cp is None or cp.shape[0] != sal_size:
                yy, xx = self._sal_meshgrid(sal_size)
                cy0 = cx0 = (sal_size - 1) / 2.0
                sig = sal_size * 0.35
                cp = np.exp(-((xx - cx0) ** 2 + (yy - cy0) ** 2)
                            / (2.0 * sig * sig)).astype(np.float32)
                self._center_prior_cache = cp

            # Skin-color prior (YCrCb thresholds), downsized to the sal grid.
            skin = None
            try:
                ycrcb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
                cr = ycrcb[:, :, 1].astype(np.int32)
                cb = ycrcb[:, :, 2].astype(np.int32)
                mask = ((cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
                        ).astype(np.float32)
                skin = cv2.resize(mask, (sal_size, sal_size),
                                  interpolation=cv2.INTER_AREA)
                sk_max = skin.max()
                if sk_max > 0:
                    skin /= sk_max
            except Exception:
                skin = None

            # Motion-energy prior from the (already EMA'd) motion hotspot.
            motion = None
            mh = r.motion_hotspot.get(time_ms)
            if mh:
                mx = int(mh['cx'] / max(1, r.src_w) * sal_size)
                my = int(mh['cy'] / max(1, r.src_h) * sal_size)
                mx = max(0, min(sal_size - 1, mx))
                my = max(0, min(sal_size - 1, my))
                yy, xx = self._sal_meshgrid(sal_size)
                sig = sal_size * 0.12
                motion = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2)
                                / (2.0 * sig * sig)).astype(np.float32)
                motion *= float(min(1.0, mh.get('intensity', 0.0)))

            fused = 0.50 * spectral + 0.25 * cp
            if skin is not None:
                fused = fused + 0.13 * skin
            if motion is not None:
                fused = fused + 0.20 * motion
            fmax = float(fused.max())
            if fmax > 0:
                fused /= fmax
            return fused
        except Exception:
            return sal_map

    def _mediapipe_mouth_open(self, frame_bgr, face, inv_scale) -> Optional[float]:
        """Lips-based mouth-motion via MediaPipe (item 5, gated).

        Crops the detected face from the detection-resolution BGR frame, asks
        the MediaPipe landmarker for a mouth-open ratio, and returns the
        frame-to-frame *change* mapped to the same 0-1 range the rest of the
        pipeline uses for ``mouth_motion`` (speaking = mouth moving). Returns
        ``None`` on any failure so the caller keeps its existing value.
        """
        try:
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
            if fw <= 1 or fh <= 1:
                return None
            # Pad a little so lips aren't clipped at the bbox edge.
            pad_w = int(fw * 0.15)
            pad_h = int(fh * 0.15)
            H, W = frame_bgr.shape[:2]
            x0 = max(0, fx - pad_w)
            y0 = max(0, fy - pad_h)
            x1 = min(W, fx + fw + pad_w)
            y1 = min(H, fy + fh + pad_h)
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return None
            from backend.services.reframer_mediapipe import mouth_open_ratio
            ratio = mouth_open_ratio(crop)
            if ratio is None:
                return None
            if not hasattr(self, '_prev_mp_mar'):
                self._prev_mp_mar = {}
            face_key = (fx // max(1, fw // 2), fy // max(1, fh // 2))
            prev = self._prev_mp_mar.get(face_key)
            self._prev_mp_mar[face_key] = ratio
            if prev is None:
                # First observation — seed with a small openness-derived value
                # so a wide-open mouth still reads as some activity.
                return round(min(1.0, max(0.0, ratio - 0.02) * 2.0), 4)
            return round(min(1.0, abs(ratio - prev) * 8.0), 4)
        except Exception:
            return None

    def _track_through_gap(self, cap, gap: int, r, det_w: int, det_h: int,
                           det_scale: float) -> None:
        """Bridge the inter-sample gap with LK optical-flow face tracking.

        The sampling loop grab()s ``gap-1`` frames between detection samples
        without decoding them. When the previous sample had faces, retrieve a
        few of those frames (up to REFRAMER_TRACK_POINTS_PER_GAP, ~evenly
        spaced), track each face's box forward with pyramidal Lucas-Kanade on
        the downscaled grays, and append the tracked positions to
        ``r.face_timeline`` (marked ``tracked: True``, confidence decayed).
        Detection cost is unchanged — grab() already decoded these frames;
        this adds only a retrieve + resize + sparse LK per tracked frame.
        Fail-soft: any error degrades to the plain grab() skip.
        """
        n_skip = gap - 1
        if n_skip <= 0:
            return
        prev_gray = getattr(self, '_last_gray_small', None)
        last_ms = getattr(self, '_last_sample_ms', None)
        prev_faces = (r.face_timeline.get(last_ms) or []) if last_ms is not None else []
        max_retrieves = int(getattr(settings, 'REFRAMER_TRACK_POINTS_PER_GAP', 3))
        if (not bool(getattr(settings, 'REFRAMER_INTER_SAMPLE_TRACKING', True))
                or prev_gray is None or not prev_faces or max_retrieves <= 0):
            for _ in range(n_skip):
                cap.grab()
            return

        # Working boxes in detection space.
        cur = []
        for f in prev_faces:
            try:
                cur.append({
                    'src': f,
                    'x': float(f['x']) * det_scale,
                    'y': float(f['y']) * det_scale,
                    'w': float(f['w']) * det_scale,
                    'h': float(f['h']) * det_scale,
                })
            except (KeyError, TypeError, ValueError):
                continue
        if not cur:
            for _ in range(n_skip):
                cap.grab()
            return

        stride = max(1, (n_skip + max_retrieves) // (max_retrieves + 1))
        fps = max(1e-6, float(getattr(r, 'fps', 0) or 0))
        for j in range(n_skip):
            if not cap.grab():
                return
            if (j + 1) % stride != 0 or not cur:
                continue
            # Tracking is opportunistic — any failure just skips this frame;
            # the grab loop above stays authoritative so the read position
            # is never left short of the next sample.
            try:
                ok, frame2 = cap.retrieve()
                if not ok or frame2 is None:
                    continue
                small2 = cv2.resize(frame2, (det_w, det_h),
                                    interpolation=cv2.INTER_LINEAR)
                gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
                t_ms = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps * 1000.0)
                moved = []
                entries = []
                for fbox in cur:
                    # 3x3 point grid inside the box — median flow is robust
                    # to a few bad correspondences.
                    xs = np.linspace(fbox['x'] + fbox['w'] * 0.2,
                                     fbox['x'] + fbox['w'] * 0.8, 3)
                    ys = np.linspace(fbox['y'] + fbox['h'] * 0.2,
                                     fbox['y'] + fbox['h'] * 0.8, 3)
                    pts = np.array([[[px, py]] for py in ys for px in xs],
                                   dtype=np.float32)
                    p1, st, _err = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray2, pts, None,
                        winSize=(21, 21), maxLevel=2)
                    if p1 is None or st is None:
                        continue
                    good = st.reshape(-1) == 1
                    if int(good.sum()) < 5:
                        continue  # lost the subject — stop tracking this box
                    deltas = (p1 - pts).reshape(-1, 2)[good]
                    dx = float(np.median(deltas[:, 0]))
                    dy = float(np.median(deltas[:, 1]))
                    fbox['x'] += dx
                    fbox['y'] += dy
                    ncx = fbox['x'] + fbox['w'] / 2.0
                    ncy = fbox['y'] + fbox['h'] / 2.0
                    if not (0 <= ncx < det_w and 0 <= ncy < det_h):
                        continue  # walked out of frame
                    moved.append(fbox)
                    src = fbox['src']
                    entry = {k: v for k, v in src.items()
                             if k not in ('x', 'y', 'w', 'h', 'cx', 'cy',
                                          'confidence', 'tracked')}
                    entry.update({
                        'x': int(fbox['x'] / det_scale),
                        'y': int(fbox['y'] / det_scale),
                        'w': int(fbox['w'] / det_scale),
                        'h': int(fbox['h'] / det_scale),
                        'cx': int(ncx / det_scale),
                        'cy': int(ncy / det_scale),
                        'confidence': round(
                            float(src.get('confidence', 0.5)) * 0.9, 3),
                        'tracked': True,
                    })
                    entries.append(entry)
                cur = moved
                prev_gray = gray2
                if entries and t_ms not in r.face_timeline:
                    r.face_timeline[t_ms] = entries
            except Exception:
                continue

    def _temporal_filter(self, recent_raw: List[List[dict]]) -> List[dict]:
        """Confirm faces by requiring persistence across multiple recent samples.

        A face is kept only if it has a near-position match in at least one
        of the last few samples. The current sample is the candidate set;
        we look back over recent_raw[:-1] for confirmation.

        Critical: when the previous sample is empty (e.g. right after a
        scene cut), we DO NOT auto-accept the current sample's faces —
        that previously let ghost detections through unfiltered. Instead
        we look further back to find ANY recent confirmation.
        """
        if not recent_raw:
            return []
        current = recent_raw[-1]
        if not current:
            return []

        # Look back at the last 3 samples for confirmation. With 5 fps
        # sampling that's a 600ms window — enough to bridge a single
        # missed detection without letting ghosts through.
        history = recent_raw[-4:-1] if len(recent_raw) >= 4 else recent_raw[:-1]
        prev_pool = []
        for sample in history:
            prev_pool.extend(sample)

        # If we have no history at all (literally the first sample of a
        # video), accept current — there's nothing to compare against.
        if not prev_pool and len(recent_raw) <= 1:
            return current

        # If history exists but is empty (scene cut just happened),
        # require the current sample to have multiple coherent faces
        # before accepting any. Single-face detections after an empty
        # window are the highest-risk false positives.
        if not prev_pool:
            if len(current) < 2:
                return []
            return current

        confirmed = []
        for face in current:
            for pf in prev_pool:
                # Manhattan distance is faster than sqrt and good enough
                dist = abs(face['cx'] - pf['cx']) + abs(face['cy'] - pf['cy'])
                # Slightly tighter linking (was 3.0 face widths) — at
                # 5 fps a face shouldn't move more than ~2 widths between
                # samples unless it's a scene cut, and scene cuts get
                # handled by the no-history branch above.
                if dist < face['w'] * 2.5:
                    confirmed.append(face)
                    break

        return confirmed

    def _assign_tracks(self, faces: List[dict], time_ms: int) -> List[dict]:
        """Assign persistent track IDs via position proximity.
        Uses wider linking and longer gap tolerance to avoid track fragmentation."""
        max_gap_ms = 5000    # 5 seconds before a track expires (was 3s)
        max_link_dist_ratio = 4.0  # 4x face width linking distance (was 2.5x)

        self._active_tracks = [
            t for t in self._active_tracks
            if time_ms - t['last_seen_ms'] < max_gap_ms
        ]

        result = []
        used_tracks = set()

        for face in faces:
            best_track = None
            best_dist = float('inf')
            for track in self._active_tracks:
                if track['id'] in used_tracks:
                    continue
                dist = math.sqrt(
                    (face['cx'] - track['cx'])**2 + (face['cy'] - track['cy'])**2)
                max_dist = max(face['w'], track['w']) * max_link_dist_ratio
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_track = track

            if best_track:
                track_id = best_track['id']
                # Smooth track position with exponential moving average
                # This prevents the track from jumping on noisy detections
                # Velocity-adaptive EMA: snap to detection faster when face is moving.
                # This keeps the track on the actual face position while still filtering
                # single-frame jitter from noisy detections.
                prev_cx = best_track['cx']
                prev_cy = best_track['cy']
                motion_px = abs(face['cx'] - prev_cx) + abs(face['cy'] - prev_cy)
                max_motion = max(1, face.get('w', 40))  # normalize by face width
                motion_norm = min(1.0, motion_px / max_motion)
                # alpha range: 0.60 (still face) → 0.90 (fast-moving face)
                alpha = 0.60 + motion_norm * 0.30
                best_track.update({
                    'cx': int(face['cx'] * alpha + prev_cx * (1 - alpha)),
                    'cy': int(face['cy'] * alpha + prev_cy * (1 - alpha)),
                    'w': int(face['w'] * 0.7 + best_track['w'] * 0.3),   # size changes slower
                    'h': int(face['h'] * 0.7 + best_track['h'] * 0.3),
                    'last_seen_ms': time_ms,
                })
                used_tracks.add(track_id)
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._active_tracks.append({
                    'id': track_id, 'cx': face['cx'], 'cy': face['cy'],
                    'w': face['w'], 'h': face['h'],
                    'last_seen_ms': time_ms,
                })

            result.append({**face, 'track_id': track_id})

        return result

    def _assign_person_tracks(self, persons: List[dict], time_ms: int) -> List[dict]:
        """Greedy centroid tracker for YOLO person/subject boxes (item 1).

        Mirrors ``_assign_tracks`` but for persons, in a separate id space.
        Matches each detection to the nearest un-used active track within a
        distance proportional to box width; unmatched detections start a new
        track. Enables per-``track_id`` box interpolation in the preview.
        """
        max_gap_ms = 4000
        max_link_ratio = 2.5
        self._active_person_tracks = [
            t for t in self._active_person_tracks
            if time_ms - t['last_seen_ms'] < max_gap_ms
        ]
        result = []
        used = set()
        for p in persons:
            best = None
            best_dist = float('inf')
            for track in self._active_person_tracks:
                if track['id'] in used:
                    continue
                dist = math.hypot(p['cx'] - track['cx'], p['cy'] - track['cy'])
                max_dist = max(p['w'], track['w']) * max_link_ratio
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best = track
            if best is not None:
                tid = best['id']
                best.update({
                    'cx': p['cx'], 'cy': p['cy'], 'w': p['w'], 'h': p['h'],
                    'last_seen_ms': time_ms,
                })
                used.add(tid)
            else:
                tid = self._next_person_track_id
                self._next_person_track_id += 1
                self._active_person_tracks.append({
                    'id': tid, 'cx': p['cx'], 'cy': p['cy'],
                    'w': p['w'], 'h': p['h'], 'last_seen_ms': time_ms,
                })
                used.add(tid)
            result.append({**p, 'track_id': tid})
        return result

    def _consolidate_tracks_across_cuts(self, r: PerceptionResult):
        """Merge face tracks that are clearly the same person.

        The tracker creates new track IDs whenever it loses a face for a few
        frames (occlusion, head turn, detection dropout) or when faces are
        close enough to confuse. A 2-person podcast generates 74 tracks
        instead of 2-3.

        Strategy: merge tracks that are spatially close AND never appear in
        the same frame simultaneously. If two tracks occupy the same position
        but are never on screen at the same time, they're the same person
        with fragmented tracking. If they DO appear simultaneously, they're
        definitely different people.

        This replaces the scene-cut-based approach which failed because
        tracks don't actually break at scene cuts — they break from tracker
        confusion between nearby faces."""
        log = get_logger()
        sorted_times = sorted(r.face_timeline.keys())
        if not sorted_times:
            return

        # Build per-track info
        track_info = {}
        track_active_times = {}  # track_id → set of times
        for t in sorted_times:
            for f in r.face_timeline[t]:
                tid = f.get('track_id', -1)
                if tid < 0:
                    continue
                if tid not in track_info:
                    track_info[tid] = {'cxs': [], 'cys': [], 'ws': [], 'count': 0}
                info = track_info[tid]
                info['cxs'].append(f['cx'])
                info['cys'].append(f['cy'])
                info['ws'].append(f['w'])
                info['count'] += 1
                track_active_times.setdefault(tid, set()).add(t)

        if len(track_info) < 2:
            return

        # Compute medians
        for tid, info in track_info.items():
            info['median_cx'] = int(np.median(info['cxs']))
            info['median_cy'] = int(np.median(info['cys']))
            info['median_w'] = int(np.median(info['ws']))

        # Collect per-track SFace embeddings for identity matching
        track_embeddings = {}  # track_id → list of embedding arrays
        for t in sorted_times:
            for f in r.face_timeline[t]:
                tid = f.get('track_id', -1)
                emb = f.get('embedding')
                if tid >= 0 and emb is not None:
                    track_embeddings.setdefault(tid, []).append(emb)

        # Compute representative embedding per track (mean of up to 10 samples)
        track_rep_embedding = {}
        sface_available = False
        for tid, embs in track_embeddings.items():
            if len(embs) >= 2:
                # Use up to 10 embeddings for a robust representative
                sampled = embs[:10]
                rep = np.mean(sampled, axis=0)
                # L2 normalize
                norm = np.linalg.norm(rep)
                if norm > 0:
                    track_rep_embedding[tid] = rep / norm
                    sface_available = True

        if sface_available:
            log.log_stage('PERCEIVE',
                f'SFace embeddings: {len(track_rep_embedding)} tracks with identity vectors')

        # Sort tracks by sample count (descending) — merge small into large
        sorted_tids = sorted(track_info.keys(),
                             key=lambda t: track_info[t]['count'], reverse=True)

        # Greedy merge: for each track, try to merge it into a larger track
        # that has a similar position and no temporal overlap
        merge_map = {}  # small_tid → large_tid

        for i, small_tid in enumerate(sorted_tids):
            if small_tid in merge_map:
                continue
            small = track_info[small_tid]
            small_times = track_active_times[small_tid]

            for large_tid in sorted_tids[:i]:  # only larger tracks
                if large_tid in merge_map:
                    # Chase to canonical
                    canon = large_tid
                    seen = set()
                    while canon in merge_map and canon not in seen:
                        seen.add(canon)
                        canon = merge_map[canon]
                    large_tid = canon

                large = track_info.get(large_tid, None)
                if large is None:
                    continue
                large_times = track_active_times.get(large_tid, set())

                # Check spatial proximity OR embedding similarity
                # SFace embedding match is the primary check — it works even
                # when camera angles change (same person, different position).
                # Spatial proximity is the fallback when embeddings unavailable.
                identity_match = False

                if small_tid in track_rep_embedding and large_tid in track_rep_embedding:
                    # Cosine similarity between representative embeddings
                    sim = float(np.dot(track_rep_embedding[small_tid],
                                       track_rep_embedding[large_tid]))
                    if sim > 0.50:  # same person threshold (0.30 was too loose)
                        identity_match = True

                if not identity_match:
                    # Fall back to spatial proximity: within 3x face width
                    dx = abs(small['median_cx'] - large['median_cx'])
                    dy = abs(small['median_cy'] - large['median_cy'])
                    dist = math.sqrt(dx*dx + dy*dy)
                    avg_w = (small['median_w'] + large['median_w']) / 2
                    if dist > avg_w * 3.0:
                        continue  # too far apart spatially and no embedding match

                # ── Spatial consistency guard ──
                # Even if SFace says "same person," reject the merge if it would
                # create a track with positions spread across the frame. Real
                # people sit in one spot; a track jumping from cx=800 to cx=3000
                # means two different people got merged.
                combined_cxs = list(track_info[small_tid]['cxs'])
                # Include the large track's positions (already in track_info)
                large_info = track_info.get(large_tid)
                if large_info:
                    combined_cxs.extend(large_info['cxs'])
                if len(combined_cxs) >= 5:
                    cx_std = float(np.std(combined_cxs))
                    if cx_std > r.src_w * 0.25:
                        continue

                # Check temporal overlap: do they EVER appear simultaneously?
                overlap = small_times & large_times
                overlap_ratio = len(overlap) / max(1, min(len(small_times), len(large_times)))

                # Allow up to 5% overlap (detection noise can put both tracks
                # in the same frame briefly). Strict zero overlap is too rigid.
                if overlap_ratio > 0.05:
                    continue  # they appear together → different people

                # Merge small into large
                merge_map[small_tid] = large_tid
                # Merge the active times so subsequent comparisons see the
                # combined temporal footprint
                track_active_times.setdefault(large_tid, set()).update(small_times)
                break

        if not merge_map:
            log.log_stage('PERCEIVE',
                f'Track consolidation: {len(track_info)} tracks, 0 merges '
                f'(all tracks are temporally overlapping or spatially distant)')
            return

        # Chase merge chains to canonical roots
        def canonical(tid):
            seen = set()
            while tid in merge_map and tid not in seen:
                seen.add(tid)
                tid = merge_map[tid]
            return tid

        # Apply
        reassigned = 0
        for t in sorted_times:
            for f in r.face_timeline[t]:
                old_tid = f.get('track_id', -1)
                if old_tid >= 0 and old_tid in merge_map:
                    f['track_id'] = canonical(old_tid)
                    reassigned += 1

        # Count unique tracks after consolidation
        post_tracks = set()
        for t in sorted_times:
            for f in r.face_timeline[t]:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    post_tracks.add(tid)

        pre_count = len(track_info)
        post_count = len(post_tracks)
        log.log_stage('PERCEIVE',
            f'Track consolidation: {pre_count} → {post_count} tracks '
            f'({len(merge_map)} merges, {reassigned} face records reassigned)')

        # ── Static-face filter ──
        # A figurine, poster, or mannequin produces a face track whose
        # per-scene positional variance is near-zero (only detector jitter).
        # A real human — even seated — has head micro-movements, expression
        # changes, and breathing that create measurably higher variance.
        # This filter catches non-human faces that slipped past the YOLO
        # gates (e.g. YOLO misclassified the figurine as "person").
        self._filter_static_face_tracks(r)

    def _filter_static_face_tracks(self, r: PerceptionResult):
        """Remove face tracks whose per-scene positional variance is
        near-zero, indicating a static object (figurine, poster, mask)
        rather than a living person.

        Strategy:
          - Group each track's detections by scene (between scene cuts)
          - For each scene segment with ≥10 samples, compute cx variance
          - If a track's MEDIAN per-scene cx-variance is below threshold,
            it's static — remove it from the face_timeline

        Threshold rationale (at detection resolution 1280×720):
          - YuNet jitter on a static object: σ ≈ 2-4px → var ≈ 4-16
          - Seated human micro-movements:    σ ≈ 8-20px → var ≈ 64-400
          - Threshold of 30 sits between jitter (16) and movement (64)
        """
        import numpy as np
        log = get_logger()

        STATIC_VAR_THRESHOLD = 30  # px² at detection resolution

        # Build scene boundaries
        scene_bounds = sorted(set([0] + r.scene_cuts + [r.duration_ms]))

        # Collect per-track, per-scene face positions
        track_scene_cxs = {}  # {track_id: {scene_idx: [cx, cx, ...]}}
        for t_ms, faces in r.face_timeline.items():
            # Find which scene this timestamp belongs to
            scene_idx = 0
            for i in range(len(scene_bounds) - 1):
                if scene_bounds[i] <= t_ms < scene_bounds[i + 1]:
                    scene_idx = i
                    break
            for f in faces:
                tid = f.get('track_id', -1)
                if tid < 0:
                    continue
                track_scene_cxs.setdefault(tid, {}).setdefault(
                    scene_idx, []).append(f['cx'])

        # Evaluate each track
        static_tracks = set()
        for tid, scene_data in track_scene_cxs.items():
            # Compute per-scene variance (only scenes with ≥10 samples)
            scene_vars = []
            for scene_idx, cxs in scene_data.items():
                if len(cxs) >= 10:
                    scene_vars.append(float(np.var(cxs)))

            if not scene_vars:
                continue  # not enough data to judge

            median_var = float(np.median(scene_vars))
            if median_var < STATIC_VAR_THRESHOLD:
                static_tracks.add(tid)

        if not static_tracks:
            return

        # Remove static tracks from face_timeline
        removed = 0
        for t_ms in list(r.face_timeline.keys()):
            orig = r.face_timeline[t_ms]
            filtered = [f for f in orig if f.get('track_id', -1)
                        not in static_tracks]
            if len(filtered) < len(orig):
                removed += len(orig) - len(filtered)
                if filtered:
                    r.face_timeline[t_ms] = filtered
                else:
                    del r.face_timeline[t_ms]

        if removed:
            log.log_stage('PERCEIVE',
                f'Static-face filter: removed {len(static_tracks)} static '
                f'tracks ({removed} face records) — likely figurines/posters')

    def _build_spatial_speakers(self, r: PerceptionResult):
        """Build pseudo-diarization from spatial clustering when pyannote is unavailable.

        For multi-person content (especially 2-person podcasts), the face tracks
        cluster into spatial groups — left seat, right seat. By assigning each
        cluster a pseudo-speaker ID and linking tracks to their cluster, we get
        a track_speaker_map that the speaker lock and S2 can use.

        Only activates when:
          - No pyannote diarization produced any results
          - There are 2+ real tracks with clear spatial separation
        """
        log = get_logger()
        if r.track_speaker_map:
            return  # real diarization already ran

        # Build per-track median positions using only tracks with enough samples
        sorted_times = sorted(r.face_timeline.keys())
        track_positions = {}
        for t in sorted_times:
            for f in r.face_timeline[t]:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_positions.setdefault(tid, []).append(f['cx'])

        # Filter to tracks with ≥25 samples (real tracks)
        real_tracks = {tid: cxs for tid, cxs in track_positions.items()
                       if len(cxs) >= 25}
        if len(real_tracks) < 2:
            return

        # Compute median x for each real track
        track_medians = {tid: int(np.median(cxs)) for tid, cxs in real_tracks.items()}

        # Simple 2-cluster split: sort tracks by median_cx, split at the largest gap
        sorted_tids = sorted(track_medians.keys(), key=lambda t: track_medians[t])
        sorted_cxs = [track_medians[t] for t in sorted_tids]

        # Find the largest gap between consecutive track medians
        if len(sorted_cxs) < 2:
            return
        gaps = [(sorted_cxs[i+1] - sorted_cxs[i], i)
                for i in range(len(sorted_cxs) - 1)]
        max_gap, split_idx = max(gaps, key=lambda g: g[0])

        # Require meaningful spatial separation (at least 8% of source width).
        # Lowered from 15% — podcast guests on a couch can be quite close
        # (492px gap on 3840px = 12.8%, which was rejected at 15%).
        if max_gap < r.src_w * 0.08:
            log.log_stage('PERCEIVE',
                f'Spatial speakers: gap too small ({max_gap}px), skipping')
            return

        # Assign pseudo-speaker IDs
        left_tids = set(sorted_tids[:split_idx + 1])
        right_tids = set(sorted_tids[split_idx + 1:])

        for tid in left_tids:
            r.track_speaker_map[tid] = 'SPEAKER_LEFT'
        for tid in right_tids:
            r.track_speaker_map[tid] = 'SPEAKER_RIGHT'

        # Build speaker_timeline from mouth motion: at each sample, the
        # track with the most mouth motion determines the active speaker
        for t in sorted_times:
            faces = r.face_timeline.get(t, [])
            best_mouth = 0.0
            best_speaker = None
            for f in faces:
                tid = f.get('track_id', -1)
                if tid in r.track_speaker_map:
                    mm = f.get('mouth_motion', 0)
                    if mm > best_mouth:
                        best_mouth = mm
                        best_speaker = r.track_speaker_map[tid]
            if best_speaker and best_mouth > 0.005:
                r.speaker_timeline[t] = best_speaker

        log.log_stage('PERCEIVE',
            f'Spatial speakers: {len(left_tids)} left tracks, '
            f'{len(right_tids)} right tracks, '
            f'{len(r.speaker_timeline)} timeline entries '
            f'(gap={max_gap}px)')

    def _extract_audio_rms(self, r: PerceptionResult):
        """Extract per-100ms RMS audio energy from the video's audio track.
        Used for audio-visual correlation — no external dependencies, just numpy."""
        log = get_logger()
        try:
            import tempfile, wave, struct
            audio_path = tempfile.mktemp(suffix='.wav')
            result = subprocess.run([
                'ffmpeg', '-y', '-i', self.path,
                '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                audio_path
            ], capture_output=True, timeout=120)

            if not os.path.exists(audio_path):
                return

            with wave.open(audio_path, 'rb') as wf:
                n_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                raw_data = wf.readframes(n_frames)

            try:
                os.remove(audio_path)
            except Exception:
                pass

            # Convert to numpy float array
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Compute RMS per 100ms window
            window_samples = int(sample_rate * 0.1)
            for i in range(0, len(samples) - window_samples, window_samples):
                time_ms = int(i / sample_rate * 1000)
                chunk = samples[i:i + window_samples]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                r.audio_rms[time_ms] = rms

            log.log_stage('PERCEIVE',
                f'Audio RMS: {len(r.audio_rms)} windows extracted')
        except Exception as e:
            log.log_stage('PERCEIVE', f'Audio RMS extraction skipped: {str(e)[:80]}')

    def _correlate_audio_visual(self, r: PerceptionResult):
        """Identify speakers by correlating audio RMS with per-track mouth motion.

        For each face track, build a time series of mouth-motion values. Then
        compute Pearson correlation between that series and the audio RMS series.
        The track with the highest correlation is the one whose mouth moves in
        sync with the audio — i.e., the speaker.

        This replaces pyannote for speaker identification using only numpy.
        Works because a speaking person's mouth motion is temporally correlated
        with audio energy; a listener's mouth motion has near-zero correlation.
        """
        log = get_logger()
        sorted_times = sorted(r.face_timeline.keys())

        # Build per-track mouth motion time series aligned with audio RMS
        track_mouth_series = {}
        common_times = sorted(set(sorted_times) & set(r.audio_rms.keys()))

        if len(common_times) < 20:  # not enough data for meaningful correlation
            return

        # Build aligned audio array
        audio_arr = np.array([r.audio_rms.get(t, 0) for t in common_times])

        # Build per-track arrays
        track_cxs = {}
        for t in common_times:
            for f in r.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_mouth_series.setdefault(tid, {})[t] = f.get('mouth_motion', 0)
                    track_cxs.setdefault(tid, []).append(f['cx'])

        # Compute Pearson correlation for each track
        track_correlations = {}
        for tid, mouth_dict in track_mouth_series.items():
            if len(mouth_dict) < 20:
                continue
            mouth_arr = np.array([mouth_dict.get(t, 0) for t in common_times])
            # Pearson correlation
            if np.std(audio_arr) > 0 and np.std(mouth_arr) > 0:
                corr = float(np.corrcoef(audio_arr, mouth_arr)[0, 1])
                track_correlations[tid] = corr

        if not track_correlations:
            return

        # Sort tracks by correlation
        sorted_tracks = sorted(track_correlations.items(), key=lambda x: x[1], reverse=True)

        # Build track_speaker_map: tracks with positive correlation are speakers
        # Cluster by spatial position (left vs right)
        speaker_tracks = [(tid, corr) for tid, corr in sorted_tracks if corr > 0.05]
        if not speaker_tracks:
            return

        # Assign speaker labels by spatial position
        for tid, corr in speaker_tracks:
            if tid in track_cxs:
                median_cx = int(np.median(track_cxs[tid]))
                if median_cx < r.src_w * 0.5:
                    r.track_speaker_map[tid] = 'SPEAKER_LEFT'
                else:
                    r.track_speaker_map[tid] = 'SPEAKER_RIGHT'

        # Build speaker timeline from correlations
        for t in sorted_times:
            faces = r.face_timeline.get(t, [])
            best_corr = -1
            best_speaker = None
            for f in faces:
                tid = f.get('track_id', -1)
                if tid in track_correlations and tid in r.track_speaker_map:
                    corr = track_correlations[tid]
                    mouth = f.get('mouth_motion', 0)
                    # Weight by both correlation strength and current mouth motion
                    score = corr * 0.5 + mouth * 0.5
                    if score > best_corr:
                        best_corr = score
                        best_speaker = r.track_speaker_map[tid]
            if best_speaker and best_corr > 0:
                r.speaker_timeline[t] = best_speaker

        log.log_stage('PERCEIVE',
            f'Audio-visual correlation: {len(speaker_tracks)} speaker tracks identified, '
            f'{len(r.speaker_timeline)} timeline entries, '
            f'top correlation={sorted_tracks[0][1]:.3f}')

    def _link_tracks_to_speakers(self, r: PerceptionResult):
        """Link face tracks to audio speakers.
        For each track, find which speaker is most often active when that
        track's face has high mouth motion (= that face is speaking)."""
        track_speaker_votes = {}  # track_id → {speaker_id: vote_count}

        for time_ms, faces in r.face_timeline.items():
            # Find nearest speaker label (within 200ms)
            speaker = None
            for offset in [0, -200, 200, -400, 400]:
                t = time_ms + offset
                if t in r.speaker_timeline:
                    speaker = r.speaker_timeline[t]
                    break
            if not speaker:
                continue

            # Vote: the face with most mouth motion at this time gets the speaker label
            speaking_faces = [f for f in faces if f.get('mouth_motion', 0) > 0.01]
            if speaking_faces:
                best = max(speaking_faces, key=lambda f: f.get('mouth_motion', 0))
                tid = best.get('track_id', -1)
                if tid >= 0:
                    track_speaker_votes.setdefault(tid, {})
                    track_speaker_votes[tid][speaker] = track_speaker_votes[tid].get(speaker, 0) + 1

        # Assign each track to its most-voted speaker
        for tid, votes in track_speaker_votes.items():
            if votes:
                best_speaker = max(votes, key=votes.get)
                r.track_speaker_map[tid] = best_speaker

        log = get_logger()
        log.log_stage('PERCEIVE', f'Track-speaker links: {len(r.track_speaker_map)} tracks mapped',
                       mapping={str(k): v for k, v in r.track_speaker_map.items()})


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2+3 — CLASSIFY + DECIDE  (scene → strategy → keyframes)
# ═══════════════════════════════════════════════════════════════════════════

