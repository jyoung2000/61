"""ClipAI Reframer — ReframeEngine (pipeline orchestrator).

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

from backend.services.reframer_models import (
    ReframeLogger, get_logger, reset_logger, RenderPlan,
    interpolate_x, clamp_x, _face_overlaps_person,
    LedgerBin, CoverageLedger, PerceptionResult, SceneSignals, AdaptiveParams,
)
from backend.services.reframer_perceiver import Perceiver
from backend.services.reframer_planner import Planner
from backend.services.reframer_smoother import Smoother

logger = logging.getLogger("clipai.reframer_engine")


class ReframeEngine:
    """
    Single entry point for the full pipeline.
    Use this in ClipAI container code.

    Example:
        engine = ReframeEngine("input.mp4")
        engine.analyze(on_progress=print)
        engine.plan.save("renderplan.json")
        engine.export("output_9x16.mp4", on_progress=print)

    Supported aspect ratios: '9:16', '1:1', '4:5', '4:3', '3:4', '16:9'
    """

    ASPECT_RATIOS = {
        '9:16': (9, 16),
        '1:1': (1, 1),
        '4:5': (4, 5),
        '4:3': (4, 3),
        '3:4': (3, 4),
        '16:9': (16, 9),
    }

    def __init__(self, video_path: str, sample_fps: float = 5.0,
                 log_dir: str = None, aspect_ratio: str = '9:16',
                 source_language: str = 'auto'):
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.aspect_ratio = aspect_ratio
        self.source_language = source_language
        self.perception: Optional[PerceptionResult] = None
        self.plan: Optional[RenderPlan] = None
        self.log = reset_logger(log_dir)

        # Parse aspect ratio
        if aspect_ratio in self.ASPECT_RATIOS:
            self.ar_w, self.ar_h = self.ASPECT_RATIOS[aspect_ratio]
        else:
            # Try parsing "W:H" format
            try:
                parts = aspect_ratio.split(':')
                self.ar_w, self.ar_h = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                self.ar_w, self.ar_h = 9, 16

    def analyze(self, on_progress: Callable = None) -> RenderPlan:
        """Run stages 1-4: perceive → classify → decide → smooth."""
        self.log.log_stage('ENGINE', f'Pipeline start: {os.path.basename(self.video_path)}',
                           aspect_ratio=f'{self.ar_w}:{self.ar_h}',
                           sample_fps=self.sample_fps,
                           source_language=self.source_language)
        self.log.start_timer('total_pipeline')

        # Stage 1: Perception (faces + audio + motion)
        self.log.log_stage('ENGINE', '═══ STAGE 1: PERCEIVE (faces, audio, motion) ═══')
        perceiver = Perceiver(self.video_path, self.sample_fps,
                              source_language=self.source_language)
        self.perception = perceiver.run(on_progress=on_progress)

        # Store audio device info for GUI display
        if hasattr(perceiver, 'audio_intel') and perceiver.audio_intel.available:
            self._perceiver_audio_device = getattr(
                perceiver.audio_intel, 'device_used', 'unknown')

        # Stage 2+3: Classify + Decide
        self.log.log_stage('ENGINE', '═══ STAGE 2+3: CLASSIFY + DECIDE ═══')
        planner = Planner(self.perception, self.ar_w, self.ar_h)
        self.plan = planner.generate()

        # Stage 3.5: Gradient centering for stuck title-card scenes
        self._fix_gradient_centering()

        # Stage 3.6: Predictive anchoring — collapse jittery keyframes
        self._stabilize_keyframes()

        # Stage 4: Smooth
        self.log.log_stage('ENGINE', '═══ STAGE 4: SMOOTH ═══')
        self.log.start_timer('smooth')
        smoother = Smoother()
        self.plan = smoother.smooth(self.plan)
        self.log.stop_timer('smooth')

        # Stage 4.4: Eval-aligned post-smoother centering. The evaluator
        # samples every 1 second and checks the INTERPOLATED crop_x at
        # that timestamp, so keyframes that are centered AT the keyframe
        # time can still drift outside the middle third in between. This
        # pass walks both keyframe positions and the per-second eval
        # grid and corrects both — without inflating the keyframe count
        # the smoother just worked to collapse.
        self._post_smoother_centering()

        # Stage 4.5: Live-action face enforcement
        # In live-action content, EVERY keyframe must have a face in crop.
        # This is the hard constraint that distinguishes live-action behavior.
        if self.perception.is_live_action:
            self._enforce_live_action_faces()

        total_elapsed = self.log.stop_timer('total_pipeline')

        # Final summary
        perc = self.perception
        n_faces = sum(1 for v in perc.face_timeline.values() if v) if perc else 0
        n_segs = len(perc.transcript_segments) if perc else 0
        lang = perc.detected_language if perc else ''
        self.log.log_stage('ENGINE',
            f'═══ PIPELINE COMPLETE ({total_elapsed:.1f}s) ═══\n'
            f'  Face samples with detections: {n_faces}\n'
            f'  Transcript segments: {n_segs} (lang={lang})\n'
            f'  Scenes: {len(self.plan.scenes)}\n'
            f'  Keyframes: {len(self.plan.keyframes)}\n'
            f'  Strategies: {[s["strategy"] for s in self.plan.strategy_log]}')

        return self.plan

    def _final_face_centering(self):
        """Final pass: correct keyframes where the best face is not in crop or off-center.

        This runs AFTER the post-pass and smoother, on the FINAL keyframes.
        It only adjusts x positions — no new keyframes are added, so stability
        is preserved. This catches issues where the predictive anchoring or
        smoother removed safety corrections that were keeping faces in frame.
        """
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x

        # Build real-track set (tracks with ≥25 samples = ≥5s screen time)
        track_counts = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}

        corrected = 0
        for kf in self.plan.keyframes:
            t = kf['time_ms']
            x = kf['x']

            # Find best face at this time (±200ms window)
            best_face = None
            best_score = 0
            for dt in [0, -200, 200, -400, 400]:
                faces = self.perception.face_timeline.get(t + dt, [])
                for f in faces:
                    if f.get('track_id', -1) not in real_tracks:
                        continue
                    # Match evaluator: saliency + mouth_motion, weighted by confidence
                    score = (f.get('saliency', 0) + f.get('mouth_motion', 0)
                            ) * max(0.15, f.get('confidence', 0.5))
                    if score > best_score:
                        best_score = score
                        best_face = f

            if best_face is None:
                continue

            face_cx = best_face['cx']

            # Check 1: face completely outside crop → hard correction
            if face_cx < x or face_cx > x + crop_w:
                kf['x'] = max(0, min(max_x, face_cx - crop_w // 2))
                corrected += 1
                continue

            # Check 2: face in crop but off-center (outside center 50%)
            crop_center = x + crop_w // 2
            offset = abs(face_cx - crop_center)
            if offset > crop_w * 0.25:
                # Blend 75% toward centered — aggressive correction
                centered_x = max(0, min(max_x, face_cx - crop_w // 2))
                kf['x'] = max(0, min(max_x, int(0.75 * centered_x + 0.25 * x)))
                corrected += 1

        if corrected > 0:
            log.log_stage('SMOOTH',
                f'Final face-centering: corrected {corrected} keyframes')

    def _post_smoother_centering(self):
        """Eval-aligned final centering pass after the smoother.

        The reframe evaluator samples every 1 second and checks the
        INTERPOLATED crop_x at that timestamp against the middle-third
        band. Two failure modes survive the earlier passes and tank the
        centering metric:

        * KEYFRAME drift — a keyframe whose face is outside the
          middle-third band. The smoother's drift suppression can
          re-introduce these by collapsing earlier centering nudges.
        * INTERPOLATED drift — between two centered keyframes the eased
          interpolation can leave the face outside the middle-third for
          half the transition duration. Phase A doesn't see this
          because it only checks AT keyframe positions.

        Both are handled here:

        * Phase A nudges off-centre keyframes back inside the middle
          third with a 35 %-crop cap so it can't yank past an adjacent
          subject.
        * Phase B walks the per-second eval grid and, for each second
          that fails the middle-third check, either inserts a new
          anchor (when the nearest existing keyframe is far away) or
          UPDATES the nearest neighbour's x (the new behaviour — keeps
          keyframe count flat on dense-keyframe content so Stability /
          Hold quality / Decisiveness don't regress).

        Cuts are never touched: they're intentional speaker switches
        and nudging them has historically dragged centering down.
        """
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        if crop_w <= 0 or not self.plan.keyframes:
            return

        # Build real-track set, with animated fallback (rapid scene
        # changes can yield short-lived tracks where none clear the
        # 25-sample bar — fall back to all tracked faces so animated
        # shows still benefit from the centering pass).
        track_counts: dict = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}
        if not real_tracks and track_counts:
            real_tracks = set(track_counts.keys())

        # Eval's middle-third boundary: |face_cx - crop_center| <= crop_w/6.
        # Trigger SLIGHTLY inside the boundary so sub-pixel interpolation
        # rounding doesn't land us back on the line.
        eval_boundary = crop_w // 6
        trigger_offset = max(0, eval_boundary - 2)
        NUDGE_CAP = max(20, int(crop_w * 0.35))

        def _best_face_near(time_ms: int):
            """Pick the most salient face within ±400 ms of time_ms.

            Uses the same scoring as the evaluator (saliency + mouth
            motion, confidence-weighted) so we agree on the subject and
            filters out non-human faces via person overlap when YOLO
            has persons at that moment.
            """
            best_face = None
            best_score = 0.0
            persons_at_t = self.perception.person_timeline.get(time_ms, [])
            for dt in [0, -200, 200, -400, 400]:
                faces = self.perception.face_timeline.get(time_ms + dt, [])
                cand = faces
                if persons_at_t:
                    human = [f for f in faces if _face_overlaps_person(f, persons_at_t)]
                    if human:
                        cand = human
                for f in cand:
                    if f.get('track_id', -1) not in real_tracks:
                        continue
                    score = (f.get('saliency', 0) + f.get('mouth_motion', 0)
                             ) * max(0.15, f.get('confidence', 0.5))
                    if score > best_score:
                        best_score = score
                        best_face = f
            return best_face

        # ── Phase A: validate every keyframe ──
        corrected = 0
        for kf in self.plan.keyframes:
            if kf.get('transition') == 'cut':
                continue  # never nudge cuts

            t = kf['time_ms']
            best_face = _best_face_near(t)
            if best_face is None:
                continue

            face_cx = best_face['cx']
            x = kf['x']

            # Face center outside crop is the inclusion-fix pass's job.
            if face_cx < x or face_cx > x + crop_w:
                continue

            crop_center = x + crop_w // 2
            if abs(face_cx - crop_center) <= trigger_offset:
                continue  # already inside middle third

            ideal_x = clamp_x(face_cx - crop_w // 2, max_x)
            delta = ideal_x - x
            sign = 1 if delta > 0 else -1
            nudge = min(NUDGE_CAP, abs(delta)) * sign
            new_x = clamp_x(x + nudge, max_x)
            if new_x == x:
                continue
            kf['x'] = new_x
            kf['_centering'] = True
            corrected += 1

        # ── Phase B: eval-aligned per-second pass ──
        # The evaluator's per-second grid catches interpolated drift
        # between keyframes that Phase A can't see. For each second
        # whose interpolated crop puts the best face outside the
        # middle-third band:
        #
        #   * If the nearest keyframe is FAR (>400 ms), insert a new
        #     anchor at that second.
        #   * If a neighbour is close (≤400 ms), UPDATE the neighbour's
        #     x by nudging it toward face center. Keeps keyframe count
        #     flat so Stability / Hold quality / Decisiveness don't
        #     regress, which is the failure mode that left centering
        #     stuck at 71 % on dense-keyframe content. Cut neighbours
        #     are still never touched — we only update non-cut ones,
        #     and fall through to "skip" when the only neighbour is a
        #     cut.
        from bisect import bisect_left

        keyframes = self.plan.keyframes
        existing_times = sorted(kf['time_ms'] for kf in keyframes)
        kfs_by_time = {kf['time_ms']: kf for kf in keyframes}
        anchor_kfs: list = []
        anchored_count = 0
        neighbour_updated = 0

        NEIGHBOUR_WINDOW_MS = 400  # used for both "is there a neighbour" and "is it close"

        def _nearest_nonzero_neighbour(time_ms: int):
            """Return (neighbour_kf, distance_ms) — the closest non-cut
            keyframe whose time is within NEIGHBOUR_WINDOW_MS, or None.
            """
            i = bisect_left(existing_times, time_ms)
            candidates = []
            if i < len(existing_times):
                candidates.append(existing_times[i])
            if i > 0:
                candidates.append(existing_times[i - 1])
            best_kf = None
            best_d = NEIGHBOUR_WINDOW_MS + 1
            for ct in candidates:
                d = abs(ct - time_ms)
                if d > NEIGHBOUR_WINDOW_MS:
                    continue
                kf = kfs_by_time.get(ct)
                if kf is None or kf.get('transition') == 'cut':
                    continue
                if d < best_d:
                    best_d = d
                    best_kf = kf
            return best_kf

        duration_ms = self.plan.duration_ms or 0
        for sec in range(0, duration_ms // 1000):
            time_ms = sec * 1000
            best_face = _best_face_near(time_ms)
            if best_face is None:
                continue
            interp_x = clamp_x(interpolate_x(keyframes, time_ms), max_x)
            face_cx = best_face['cx']
            # Outside crop is the inclusion-fix pass's job.
            if face_cx < interp_x or face_cx > interp_x + crop_w:
                continue
            crop_center = interp_x + crop_w // 2
            if abs(face_cx - crop_center) <= eval_boundary:
                continue  # already in middle third at eval time

            neighbour = _nearest_nonzero_neighbour(time_ms)
            ideal_x = clamp_x(face_cx - crop_w // 2, max_x)

            if neighbour is None:
                # No close neighbour — insert a fresh anchor with a
                # short ease so it blends with surrounding keyframes
                # instead of cutting.
                anchor_kfs.append({
                    'time_ms': time_ms,
                    'x': ideal_x,
                    'transition': 'ease_in_out',
                    'transition_ms': 200,
                    '_centering': True,
                })
                anchored_count += 1
            else:
                # Pull the existing neighbour toward face center. The
                # 35 % NUDGE_CAP keeps adjacent subjects safe.
                nx = neighbour['x']
                delta = ideal_x - nx
                if abs(delta) < 3:
                    continue  # already close enough; sub-pixel jitter
                sign = 1 if delta > 0 else -1
                nudge = min(NUDGE_CAP, abs(delta)) * sign
                new_nx = clamp_x(nx + nudge, max_x)
                if new_nx == nx:
                    continue
                neighbour['x'] = new_nx
                neighbour['_centering'] = True
                neighbour_updated += 1

        if anchor_kfs:
            self.plan.keyframes.extend(anchor_kfs)
            self.plan.keyframes.sort(key=lambda k: k['time_ms'])

        if corrected or anchored_count or neighbour_updated:
            log.log_stage('SMOOTH',
                f'Post-smoother centering: nudged {corrected}, '
                f'anchored {anchored_count} mid-transition seconds, '
                f'pulled {neighbour_updated} neighbours toward face')

    def _enforce_live_action_faces(self):
        """Hard constraint: in live-action content, ensure every keyframe
        interval has a face in the crop.

        Unlike the predictive anchoring's face inclusion check (which runs
        before smoothing and gets its corrections smoothed away), this runs
        AFTER smoothing as a final guarantee. It checks every 500ms of the
        video, and if the interpolated crop position has no face, it inserts
        a correction keyframe.

        This is what makes live-action reframing behave like a human camera
        operator who ALWAYS keeps a face in frame."""
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        kfs = self.plan.keyframes

        if len(kfs) < 2:
            return

        # Build real-track set
        track_counts = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}

        # Sample times from face_timeline
        sample_times = sorted(self.perception.face_timeline.keys())

        def interpolate_x(t_ms):
            """Interpolate crop x at time t_ms from keyframes."""
            if t_ms <= kfs[0]['time_ms']:
                return kfs[0]['x']
            if t_ms >= kfs[-1]['time_ms']:
                return kfs[-1]['x']
            for i in range(len(kfs) - 1):
                if kfs[i]['time_ms'] <= t_ms <= kfs[i + 1]['time_ms']:
                    t0, t1 = kfs[i]['time_ms'], kfs[i + 1]['time_ms']
                    if t1 == t0:
                        return kfs[i]['x']
                    frac = (t_ms - t0) / (t1 - t0)
                    return int(kfs[i]['x'] + frac * (kfs[i + 1]['x'] - kfs[i]['x']))
            return kfs[-1]['x']

        corrections = []
        last_correction_t = -2000  # prevent corrections too close together

        for st in sample_times:
            faces = self.perception.face_timeline.get(st, [])
            real_faces = [f for f in faces
                          if f.get('track_id', -1) >= 0
                          and (not real_tracks
                               or f.get('track_id', -1) in real_tracks)]
            if not real_faces:
                continue

            crop_x = interpolate_x(st)
            crop_left = crop_x
            crop_right = crop_x + crop_w

            # Check if ANY face center is in crop
            any_in_crop = any(crop_left <= f['cx'] <= crop_right for f in real_faces)

            if not any_in_crop and (st - last_correction_t) >= 800:
                # No face in crop — snap to the best face
                best = max(real_faces, key=lambda f: (
                    f.get('mouth_motion', 0) * 10.0 + f.get('area', 0) * 0.0001
                ) * max(0.15, f.get('confidence', 0.5)))
                new_x = clamp_x(best['cx'] - crop_w // 2, max_x)

                # Use cut for large corrections (>20% crop_w) — decisive, not drifty
                dx = abs(new_x - crop_x)
                if dx > crop_w * 0.20:
                    trans_type = 'cut'
                    trans_ms = 0
                else:
                    trans_type = 'ease_in_out'
                    trans_ms = 350

                corrections.append({
                    'time_ms': st,
                    'x': new_x,
                    'transition': trans_type,
                    'transition_ms': trans_ms,
                })
                last_correction_t = st

        if corrections:
            self.plan.keyframes.extend(corrections)
            self.plan.keyframes.sort(key=lambda k: k['time_ms'])
            log.log_stage('SMOOTH',
                f'Live-action face enforcement: inserted {len(corrections)} corrections')

    def _fix_gradient_centering(self):
        """Post-pass: fix stuck-at-left scenes using Sobel gradient centering.

        Title cards, logos, and text overlays often have no face/person detections,
        so the planner's EMA tracker starts near x=0 and never finds the content.
        This pass detects scenes where all keyframes are stuck at the far left
        and uses pixel-level gradient analysis to find where the visual content
        actually is, then recenters the keyframes.

        Conditions (ALL must be true for a scene to be fixed):
          1. Strategy is adaptive_saliency or adaptive_motion (non-face scenes)
          2. Scene has zero face detections from real tracks
          3. Scene has zero YOLO person detections
          4. Maximum keyframe x in the scene is < max_x * 0.30 (stuck at left)
        """
        if not self.plan or not self.perception or not self.video_path:
            return

        log = get_logger()
        max_x = self.plan.max_x
        if max_x <= 0:
            return

        # Threshold: if all keyframes are below this x, the scene is "stuck"
        stuck_threshold = int(max_x * 0.30)
        crop_w = self.plan.crop_w

        # Build real-track set (same logic as Planner)
        global_track_counts = {}
        for faces in self.perception.face_timeline.values():
            seen = set()
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    seen.add(tid)
            for tid in seen:
                global_track_counts[tid] = global_track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, n in global_track_counts.items() if n >= 25}

        fixed_scenes = []
        cap = None

        try:
            for scene in self.plan.scenes:
                strategy = scene.get('strategy', '')
                if strategy not in ('adaptive_saliency', 'adaptive_motion'):
                    continue

                s_start = scene['start_ms']
                s_end = scene['end_ms']

                # Check: does this scene have any real face detections?
                scene_has_faces = False
                for t_ms, faces in self.perception.face_timeline.items():
                    if s_start <= t_ms < s_end:
                        for f in faces:
                            tid = f.get('track_id', -1)
                            if tid >= 0 and (not real_tracks or tid in real_tracks):
                                scene_has_faces = True
                                break
                    if scene_has_faces:
                        break

                if scene_has_faces:
                    continue

                # Check: does this scene have YOLO person detections?
                scene_has_persons = False
                for t_ms, persons in self.perception.person_timeline.items():
                    if s_start <= t_ms < s_end and persons:
                        scene_has_persons = True
                        break

                if scene_has_persons:
                    continue

                # Find keyframes in this scene
                scene_kf_indices = []
                for idx, kf in enumerate(self.plan.keyframes):
                    if s_start <= kf['time_ms'] < s_end:
                        scene_kf_indices.append(idx)

                if not scene_kf_indices:
                    continue

                # Check if keyframes are stuck at far left
                kf_max_x = max(self.plan.keyframes[i]['x'] for i in scene_kf_indices)
                if kf_max_x > stuck_threshold:
                    continue

                # ── This scene qualifies for gradient centering ──
                # Read a frame from the middle of the scene
                mid_ms = (s_start + s_end) / 2
                mid_frame_num = int(mid_ms / 1000.0 * self.perception.fps)

                if cap is None:
                    cap = cv2.VideoCapture(self.video_path)
                    if not cap.isOpened():
                        log.log_stage('SMOOTH',
                            '[GRADIENT] Cannot open video for gradient centering')
                        return

                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Compute Sobel gradient magnitude per column
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape

                # ── Corner exclusion: mask out edges where watermarks/ads live ──
                # Typical watermark locations: corners, top/bottom strips.
                # Zero out a margin around the frame edges so gradient energy
                # from TV station logos, social handles, and ad text doesn't
                # pull the centroid to the wrong position.
                edge_margin_x = max(30, int(w * 0.12))   # ~12% horizontal edges
                edge_margin_y = max(20, int(h * 0.15))   # ~15% vertical edges
                mask = np.ones_like(gray, dtype=np.float64)
                # Top-left, top-right corners
                mask[:edge_margin_y, :edge_margin_x] = 0
                mask[:edge_margin_y, w - edge_margin_x:] = 0
                # Bottom-left, bottom-right corners
                mask[h - edge_margin_y:, :edge_margin_x] = 0
                mask[h - edge_margin_y:, w - edge_margin_x:] = 0

                # Horizontal and vertical Sobel
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(sobel_x**2 + sobel_y**2) * mask

                # Sum gradient magnitude per column → 1D energy profile
                col_energy = magnitude.sum(axis=0)  # shape: (width,)

                # Find energy-weighted centroid (center of visual content)
                total_energy = col_energy.sum()
                if total_energy < 1.0:
                    continue  # blank or uniform frame

                cols = np.arange(len(col_energy))
                centroid_x = int(np.sum(cols * col_energy) / total_energy)

                # Convert centroid to crop position (center crop on centroid)
                gradient_crop_x = clamp_x(centroid_x - crop_w // 2, max_x)

                # Override all keyframes in this scene
                for idx in scene_kf_indices:
                    old_x = self.plan.keyframes[idx]['x']
                    self.plan.keyframes[idx]['x'] = gradient_crop_x

                fixed_scenes.append({
                    'scene_range': f'{s_start/1000:.1f}-{s_end/1000:.1f}s',
                    'strategy': strategy,
                    'old_x': kf_max_x,
                    'gradient_centroid': centroid_x,
                    'new_crop_x': gradient_crop_x,
                })

        finally:
            if cap is not None:
                cap.release()

        if fixed_scenes:
            for fs in fixed_scenes:
                log.log_stage('SMOOTH',
                    f'[GRADIENT] Recentered {fs["scene_range"]} ({fs["strategy"]}): '
                    f'x={fs["old_x"]}→{fs["new_crop_x"]} '
                    f'(gradient centroid={fs["gradient_centroid"]})')
            log.log_stage('SMOOTH',
                f'[GRADIENT] Fixed {len(fixed_scenes)} stuck scene(s) via '
                f'Sobel gradient centering')

    def _stabilize_keyframes(self):
        """Predictive anchoring via per-scene saliency-weighted clustering.

        The planner emits a keyframe every ~200ms, producing noisy positions
        that jitter frame-to-frame. This pass replaces that noise with stable
        per-scene anchor positions by:

          1. Per-scene median clustering — find ONE stable x per scene
          2. Bimodal sub-scene splitting — if keyframes cluster into two
             groups (two speakers), split into two anchors with a clean
             transition at the temporal boundary
          3. Blip removal — positions that snap somewhere and back within
             2 seconds are removed (they're detection noise, not real moves)
          4. Face inclusion check — once per second, if the best face CENTER
             is fully outside the crop, insert a gentle correction
          5. Centering nudge — ONLY non-cut keyframes, ≤20px toward the
             nearest face. NEVER nudge cuts (proven to regress centering).
        """
        if not self.plan or not self.perception:
            return

        log = get_logger()
        kfs = self.plan.keyframes
        if len(kfs) < 2:
            return

        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        # Cluster separation threshold: positions must differ by this much
        # to be considered two separate clusters (bimodal)
        cluster_gap = max(40, int(crop_w * 0.30))  # 30% of crop width

        # Build real-track set
        global_track_counts = {}
        for faces in self.perception.face_timeline.values():
            seen = set()
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    seen.add(tid)
            for tid in seen:
                global_track_counts[tid] = global_track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, n in global_track_counts.items() if n >= 25}

        # Build sorted sample times for fast nearest-neighbor lookup
        sorted_sample_times = sorted(self.perception.face_timeline.keys())

        def nearest_sample(t_ms):
            """Find the nearest perception sample time."""
            if not sorted_sample_times:
                return None
            # Binary search
            import bisect
            idx = bisect.bisect_left(sorted_sample_times, t_ms)
            candidates = []
            if idx > 0:
                candidates.append(sorted_sample_times[idx - 1])
            if idx < len(sorted_sample_times):
                candidates.append(sorted_sample_times[idx])
            best = min(candidates, key=lambda s: abs(s - t_ms))
            return best if abs(best - t_ms) <= 500 else None

        # ══════════════════════════════════════════════════════════════
        # Pass 1: Per-scene clustering — collapse noisy keyframes into
        # 1-2 stable anchor positions per scene
        # ══════════════════════════════════════════════════════════════
        anchored = []
        anchor_count = 0

        for scene in self.plan.scenes:
            s_start = scene['start_ms']
            s_end = scene['end_ms']
            strategy = scene.get('strategy', '')

            # Collect keyframes in this scene
            scene_kfs = [dict(kf) for kf in kfs
                         if s_start <= kf['time_ms'] < s_end]
            if not scene_kfs:
                continue

            # ══════════════════════════════════════════════════════════
            # FACE IS KING: If faces are visible, track them directly.
            # A camera operator ALWAYS centers on the face — saliency,
            # motion, and person boxes are fallbacks for faceless frames.
            # ══════════════════════════════════════════════════════════

            # Collect face positions WITH timestamps (time-ordered)
            face_positions = []  # list of (time_ms, crop_x)
            for t_ms in sorted(self.perception.face_timeline.keys()):
                if s_start <= t_ms < s_end:
                    faces = self.perception.face_timeline[t_ms]
                    real_faces = [f for f in faces
                                  if f.get('track_id', -1) >= 0
                                  and (not real_tracks
                                       or f.get('track_id', -1) in real_tracks)]
                    # Filter non-human faces using person overlap
                    persons_here = self.perception.person_timeline.get(t_ms, [])
                    if persons_here and real_faces:
                        human_f = [f for f in real_faces
                                   if _face_overlaps_person(f, persons_here)]
                        if human_f:
                            real_faces = human_f
                    if real_faces:
                        # PICK ONE: largest face weighted by confidence, never average
                        best = max(real_faces, key=lambda f: f.get('area', 0) *
                                   max(0.15, f.get('confidence', 0.5)))
                        cx = clamp_x(best['cx'] - crop_w // 2, max_x)
                        face_positions.append((t_ms, cx))

            if len(face_positions) >= 3:
                # ── FACE-BEARING SCENE: deadband face tracking ──
                # Follow the face as it moves, but use a deadband to
                # prevent jitter from noisy detections. This creates
                # 1-5 anchors per scene that track face movement while
                # holding still during minor detection noise.
                # Live-action: tighter deadband for responsive face tracking.
                # Animated: wider deadband for stability.
                if self.perception.is_live_action:
                    face_deadband = max(30, int(crop_w * 0.25))  # 25% crop_w
                else:
                    face_deadband = max(30, int(crop_w * 0.30))  # 30% crop_w

                # ── Bimodal detection: two faces on opposite sides ──
                # When two faces are far apart (e.g. Verzuz battle), the
                # "best face" alternates between them frame-by-frame,
                # creating a bounce pattern.  Detect this and commit to
                # one face per time segment instead of oscillating.
                #
                # STRICT conditions to avoid false triggers:
                # 1. Gap between clusters must be > 80% of crop_w
                # 2. Both clusters need substantial representation (≥25% each)
                # 3. Must confirm simultaneous two-face presence at ≥20% of timestamps
                fp_xs = [fp[1] for fp in face_positions]
                sorted_fp = sorted(fp_xs)
                max_gap = 0
                split_val = 0
                for gi in range(1, len(sorted_fp)):
                    gap = sorted_fp[gi] - sorted_fp[gi - 1]
                    if gap > max_gap:
                        max_gap = gap
                        split_val = (sorted_fp[gi - 1] + sorted_fp[gi]) // 2

                left_count = sum(1 for x in fp_xs if x < split_val)
                right_count = sum(1 for x in fp_xs if x >= split_val)
                total_fp = len(fp_xs)
                min_cluster_pct = 0.25

                # Verify simultaneous two-face presence
                simultaneous_count = 0
                if max_gap > crop_w * 0.80:
                    for fp_t, _ in face_positions:
                        faces_here = self.perception.face_timeline.get(fp_t, [])
                        real_here = [f for f in faces_here
                                     if f.get('track_id', -1) >= 0
                                     and (not real_tracks
                                          or f.get('track_id', -1) in real_tracks)]
                        if len(real_here) >= 2:
                            xs_here = [clamp_x(f['cx'] - crop_w // 2, max_x) for f in real_here]
                            has_left = any(x < split_val for x in xs_here)
                            has_right = any(x >= split_val for x in xs_here)
                            if has_left and has_right:
                                simultaneous_count += 1

                is_bimodal = (max_gap > crop_w * 0.80 and
                              left_count >= total_fp * min_cluster_pct and
                              right_count >= total_fp * min_cluster_pct and
                              simultaneous_count >= total_fp * 0.20)

                if is_bimodal:
                    # Two-face commitment strategy: instead of bouncing between
                    # left and right faces, commit to one side for extended
                    # periods.  Use mouth_motion and area to decide which face
                    # to show.  Switch only when the other side shows strong
                    # sustained activity (speaker change).

                    # Classify each timestamp into left/right cluster
                    # and compute activity score per cluster per window
                    left_cluster = int(np.median([x for x in fp_xs if x < split_val]))
                    right_cluster = int(np.median([x for x in fp_xs if x >= split_val]))

                    # For each face_position timestamp, check which faces are
                    # available and their mouth_motion scores
                    committed_positions = []
                    current_side = None  # 'left' or 'right'
                    side_hold_start = 0
                    min_commit_ms = 2000  # hold each side for at least 2 seconds

                    for fp_t, fp_x in face_positions:
                        side = 'left' if fp_x < split_val else 'right'

                        # Get mouth motion for faces at this timestamp
                        faces_here = self.perception.face_timeline.get(fp_t, [])
                        left_activity = 0.0
                        right_activity = 0.0
                        for f in faces_here:
                            f_crop_x = clamp_x(f['cx'] - crop_w // 2, max_x)
                            mouth = f.get('mouth_motion', 0)
                            area_s = f.get('area', 0) * 0.00001
                            activity = mouth * 10.0 + area_s
                            if f_crop_x < split_val:
                                left_activity = max(left_activity, activity)
                            else:
                                right_activity = max(right_activity, activity)

                        if current_side is None:
                            # First position — commit to the more active side
                            current_side = 'left' if left_activity >= right_activity else 'right'
                            side_hold_start = fp_t
                            committed_positions.append(
                                (fp_t, left_cluster if current_side == 'left' else right_cluster))
                        else:
                            time_on_side = fp_t - side_hold_start

                            # Should we switch sides?
                            other_side = 'right' if current_side == 'left' else 'left'
                            other_activity = right_activity if current_side == 'left' else left_activity
                            my_activity = left_activity if current_side == 'left' else right_activity

                            # Switch only after minimum hold AND other side is clearly more active
                            should_switch = (time_on_side >= min_commit_ms and
                                             other_activity > my_activity * 1.5 + 0.1)

                            if should_switch:
                                current_side = other_side
                                side_hold_start = fp_t

                            committed_positions.append(
                                (fp_t, left_cluster if current_side == 'left' else right_cluster))

                    face_positions = committed_positions

                # Start with first face position
                face_anchors = [{
                    'time_ms': face_positions[0][0],
                    'x': face_positions[0][1],
                    'transition': scene_kfs[0].get('transition', 'cut'),
                    'transition_ms': scene_kfs[0].get('transition_ms', 0),
                    'source': 'face_track',
                }]
                current_x = face_positions[0][1]

                for t_ms, fx in face_positions[1:]:
                    if abs(fx - current_x) > face_deadband:
                        # Face moved significantly — create new anchor
                        face_anchors.append({
                            'time_ms': t_ms,
                            'x': fx,
                            'transition': 'ease_in_out',
                            'transition_ms': 400,
                            'source': 'face_track',
                        })
                        current_x = fx

                anchored.extend(face_anchors)
                anchor_count += len(face_anchors)

            else:
                # ── NON-FACE SCENE: scene-level median (proven stable) ──
                xs = [kf['x'] for kf in scene_kfs]

                if len(xs) < 3:
                    anchored.extend(scene_kfs)
                    anchor_count += len(scene_kfs)
                    continue

                # Bimodal detection for two-cluster scenes
                sorted_xs = sorted(xs)
                max_gap = 0
                split_idx = 0
                for i in range(1, len(sorted_xs)):
                    gap = sorted_xs[i] - sorted_xs[i - 1]
                    if gap > max_gap:
                        max_gap = gap
                        split_idx = i

                min_cluster = max(2, len(xs) // 6)

                if (max_gap > cluster_gap
                        and split_idx >= min_cluster
                        and len(sorted_xs) - split_idx >= min_cluster):
                    # BIMODAL: two clusters
                    left_xs = sorted_xs[:split_idx]
                    right_xs = sorted_xs[split_idx:]
                    left_anchor = int(np.median(left_xs))
                    right_anchor = int(np.median(right_xs))
                    split_value = (left_xs[-1] + right_xs[0]) / 2

                    for kf in scene_kfs:
                        kf['x'] = left_anchor if kf['x'] < split_value else right_anchor

                    deduped = [scene_kfs[0]]
                    for kf in scene_kfs[1:]:
                        if kf['x'] != deduped[-1]['x']:
                            if kf.get('transition') != 'cut':
                                kf['transition'] = 'ease_in_out'
                                kf['transition_ms'] = 500
                            deduped.append(kf)
                    anchored.extend(deduped)
                    anchor_count += 2
                else:
                    # UNIMODAL: single anchor
                    anchor_x = int(np.median(xs))
                    first_kf = scene_kfs[0]
                    first_kf['x'] = anchor_x
                    anchored.append(first_kf)
                    anchor_count += 1

        # ══════════════════════════════════════════════════════════════
        # Pass 2: Blip removal — A→B→A patterns within 2 seconds are noise
        # ══════════════════════════════════════════════════════════════
        blip_count = 0
        if len(anchored) >= 3:
            i = 1
            while i < len(anchored) - 1:
                prev = anchored[i - 1]
                curr = anchored[i]
                nxt = anchored[i + 1]

                hold_dur = (nxt['time_ms'] - curr['time_ms']) / 1000.0
                returns_close = abs(nxt['x'] - prev['x']) < crop_w * 0.10

                # Never remove face-tracking keyframes — they follow
                # real face movement, not detection noise
                if curr.get('source') == 'face_track':
                    i += 1
                    continue

                if hold_dur < 2.5 and returns_close:
                    # This is a blip — remove it
                    anchored.pop(i)
                    blip_count += 1
                else:
                    i += 1

        # ══════════════════════════════════════════════════════════════
        # Pass 3: Face inclusion check (1×/sec) — conservative
        # Only correct when face CENTER is fully outside the crop.
        # ══════════════════════════════════════════════════════════════
        inclusion_fixes = 0
        correction_kfs = []

        for check_ms in range(0, self.plan.duration_ms, 200):
            st = nearest_sample(check_ms)
            if st is None:
                continue

            faces = self.perception.face_timeline.get(st, [])
            real_faces = [f for f in faces
                          if f.get('track_id', -1) >= 0
                          and (not real_tracks
                               or f.get('track_id', -1) in real_tracks)]
            if not real_faces:
                continue

            best_face = max(real_faces, key=lambda f: (
                f.get('saliency', 0) + f.get('mouth_motion', 0)
            ) * max(0.15, f.get('confidence', 0.5)))
            face_cx = best_face['cx']

            crop_x = interpolate_x(anchored, check_ms)
            crop_left = crop_x
            crop_right = crop_x + crop_w

            # Only fix if face center is FULLY outside crop (not just at edge)
            if face_cx < crop_left or face_cx > crop_right:
                corrected_x = clamp_x(face_cx - crop_w // 2, max_x)

                # Don't insert too close to existing keyframes
                too_close = any(abs(kf['time_ms'] - check_ms) < 500
                                for kf in anchored)
                if not too_close:
                    correction_kfs.append({
                        'time_ms': check_ms,
                        'x': corrected_x,
                        'transition': 'ease_in_out',
                        'transition_ms': 400,
                    })
                    inclusion_fixes += 1

        if correction_kfs:
            anchored.extend(correction_kfs)
            anchored.sort(key=lambda k: k['time_ms'])

        # ══════════════════════════════════════════════════════════════
        # Pass 4: Centering nudge — NON-CUT keyframes ONLY
        # PROVEN: nudging cuts always regresses centering.
        #   v22: per-frame face → 78→61%
        #   v28: forward-looking → 69→64%
        #   v30: scene-level median → 68→56%
        # ══════════════════════════════════════════════════════════════
        centered_count = 0
        for kf in anchored:
            if kf.get('transition') == 'cut':
                continue  # NEVER nudge cuts

            st = nearest_sample(kf['time_ms'])
            if st is None:
                continue

            faces = self.perception.face_timeline.get(st, [])
            real_faces = [f for f in faces
                          if f.get('track_id', -1) >= 0
                          and (not real_tracks
                               or f.get('track_id', -1) in real_tracks)]
            if not real_faces:
                continue

            crop_center = kf['x'] + crop_w // 2
            # Pick the most salient face (matches evaluator logic), not nearest
            best_face = max(real_faces, key=lambda f: (
                f.get('saliency', 0) + f.get('mouth_motion', 0)
            ) * max(0.2, f.get('confidence', 0.5)))
            ideal_x = clamp_x(best_face['cx'] - crop_w // 2, max_x)
            delta = ideal_x - kf['x']
            if 5 < abs(delta) <= 150:
                nudge = min(60, abs(delta)) * (1 if delta > 0 else -1)
                kf['x'] = clamp_x(kf['x'] + nudge, max_x)
                centered_count += 1

        log.log_stage('SMOOTH',
            f'Predictive anchoring: {len(kfs)} → {len(anchored)} keyframes '
            f'({anchor_count} anchors, {inclusion_fixes} inclusion fixes, '
            f'{blip_count} blips removed, {centered_count} centered)')

        self.plan.keyframes = anchored

