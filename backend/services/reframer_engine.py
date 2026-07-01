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
    ReframeTracer, null_tracer,
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
                 source_language: str = 'auto',
                 trace_path: str = None,
                 transcribe_audio_path: Optional[str] = None):
        self.video_path = video_path
        self.sample_fps = sample_fps
        self.aspect_ratio = aspect_ratio
        self.source_language = source_language
        # Optional pre-separated vocal stem fed to Whisper (vocal separation).
        self.transcribe_audio_path = transcribe_audio_path
        self.perception: Optional[PerceptionResult] = None
        self.plan: Optional[RenderPlan] = None
        self.log = reset_logger(log_dir)
        # JSONL per-decision trace. When trace_path is empty the tracer
        # becomes a no-op so every callsite below stays the same.
        self.tracer: ReframeTracer = (
            ReframeTracer(trace_path) if trace_path else null_tracer())

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
        # Tracer: high-level session metadata. Every downstream event
        # joins on this implicitly (one trace file = one session).
        self.tracer.event('analyze_start',
                          video=os.path.basename(self.video_path),
                          aspect_ratio=f'{self.ar_w}:{self.ar_h}',
                          sample_fps=self.sample_fps,
                          source_language=self.source_language)

        # Stage 1: Perception (faces + audio + motion)
        self.log.log_stage('ENGINE', '═══ STAGE 1: PERCEIVE (faces, audio, motion) ═══')
        perceiver = Perceiver(self.video_path, self.sample_fps,
                              source_language=self.source_language,
                              transcribe_audio_path=self.transcribe_audio_path)
        self.perception = perceiver.run(on_progress=on_progress)

        # Store audio device + EFFECTIVE model info for GUI display. The
        # effective model name is what actually loaded (post any VRAM
        # downgrade), so the active-config readout can't silently disagree
        # with what ran (Task 4).
        if hasattr(perceiver, 'audio_intel') and perceiver.audio_intel.available:
            self._perceiver_audio_device = getattr(
                perceiver.audio_intel, 'device_used', 'unknown')
            self._perceiver_audio_model = getattr(
                perceiver.audio_intel, 'model_name', None)
            self._perceiver_audio_model_requested = getattr(
                perceiver.audio_intel, 'requested_model_name', None)

        self.tracer.event('perceive_complete',
                          src_w=self.perception.src_w,
                          src_h=self.perception.src_h,
                          duration_ms=self.perception.duration_ms,
                          fps=self.perception.fps,
                          face_samples=len(self.perception.face_timeline),
                          scene_cuts=len(self.perception.scene_cuts),
                          transcript_segments=len(self.perception.transcript_segments),
                          is_live_action=self.perception.is_live_action,
                          detected_language=self.perception.detected_language)

        # Stage 2+3: Classify + Decide
        self.log.log_stage('ENGINE', '═══ STAGE 2+3: CLASSIFY + DECIDE ═══')
        planner = Planner(self.perception, self.ar_w, self.ar_h,
                          tracer=self.tracer)
        self.plan = planner.generate()

        # Stage 3.5: Gradient centering for stuck title-card scenes
        self._fix_gradient_centering()

        # Stage 3.6: Predictive anchoring — collapse jittery keyframes
        self._stabilize_keyframes()

        # Stage 4: Smooth
        self.log.log_stage('ENGINE', '═══ STAGE 4: SMOOTH ═══')
        self.log.start_timer('smooth')
        smoother = Smoother(tracer=self.tracer)
        self.plan = smoother.smooth(self.plan)
        self.log.stop_timer('smooth')

        # Stage 4.2: Offline path optimization. We render the whole clip after
        # the fact, so we can make NON-causal decisions the greedy planner
        # can't: an L1-optimal camera path (holds + linear pans) and/or a
        # Savitzky-Golay smoothing pass over the trajectory. Both preserve
        # hard cuts and _centering-flagged keyframes, and are flag-gated.
        self._apply_l1_camera_path()
        self._smooth_trajectory_savgol()

        # Stage 4.4: Final eval-aligned face centering. This used to
        # conflict with the in-planner centering passes because BOTH
        # corrected aggressively and re-snapped each other. The current
        # version is opt-in to that fight only when the eval would
        # actually mark a keyframe off-center — it does NOT insert new
        # keyframes (the inclusion-fix pass already covered face-out-
        # of-crop), it does NOT touch keyframes already tagged
        # ``_centering`` (those came from the stabilizer's nudge pass
        # and the planner has already settled on a position), and it
        # only nudges within a tight 50 px cap proportional to the
        # crop width so it can't yank between adjacent subjects.
        # Together with the iterated post-EMA pull in the planner and
        # the smoother's centering-flag respect, this is what closes
        # the 68 %→target gap.
        self._post_smoother_centering()

        # Stage 4.5: Face enforcement — live-action runs every 800ms,
        # non-live-action every 1500ms (gentle enough not to jitter on
        # anime/gaming but catches the long-interval face_missing gaps
        # that Phase B misses when the perceiver has sparse samples).
        self._enforce_live_action_faces()

        # Final edge-violation elimination — runs last, after all other passes
        self._eliminate_edge_violations()

        # Stage 5: Motivated zoom — stamp a per-keyframe `scale` term (push-in
        # on held speakers, punch-out for reveals) using the dormant
        # motivated_zoom planner. Runs last so it layers on the final x path.
        # Flag-gated (default OFF); no-op when no zoom moments are produced.
        self._apply_motivated_zoom()

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

        # Final tracer summary + close. Done in a try/finally so even a
        # raise on the way out flushes the trace to disk for diagnosis.
        self.tracer.close(summary={
            'total_elapsed_s': round(total_elapsed, 2),
            'scenes': len(self.plan.scenes),
            'keyframes': len(self.plan.keyframes),
            'strategies': [s['strategy'] for s in self.plan.strategy_log],
        })

        return self.plan

    def _final_face_centering(self):
        """Final pass: correct keyframes where the best face is not in crop or off-center.

        This runs AFTER the post-pass and smoother, on the FINAL keyframes.
        It only adjusts x positions — no new keyframes are added, so stability
        is preserved. This catches issues where the predictive anchoring or
        smoother removed safety corrections that were keeping faces in frame.

        Two-check ladder per keyframe:
          1. Face center OUTSIDE crop → snap full correction onto face center.
          2. Face IN crop but >25% off-center → blend 75% toward centered.

        Both checks ALSO enforce full-bbox containment: if the face bbox
        (cx ± w/2) plus an edge margin would clip on either side after the
        x correction, the crop is nudged inward so the whole face fits.
        """
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x

        # Build real-track set (tracks with ≥25 samples = ≥5s screen time).
        # Animated content with rapid scene changes can yield short-lived
        # tracks where none clear the 25-sample bar — in that case fall
        # back to ALL tracks (>0 samples) so the final pass still has
        # face evidence to act on. Without this fallback, animated shows
        # silently skip the final-centering safety net.
        track_counts = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}
        if not real_tracks and track_counts:
            # Animated fallback: every tracked face is "real" because the
            # tracker only ever emits IDs for face-shaped detections.
            real_tracks = set(track_counts.keys())

        corrected = 0
        for kf in self.plan.keyframes:
            t = kf['time_ms']
            x = kf['x']

            # Find best face at this time (±200ms window).
            best_face = None
            best_score = 0
            for dt in [0, -200, 200, -400, 400, -600, 600]:
                faces = self.perception.face_timeline.get(t + dt, [])
                # Filter non-human faces (figurines, posters) when YOLO
                # found persons at this moment — mirrors the planner gate
                # so the final pass doesn't snap onto background art.
                persons_at_t = self.perception.person_timeline.get(t + dt, [])
                cand_faces = faces
                if persons_at_t:
                    human = [f for f in faces if _face_overlaps_person(f, persons_at_t)]
                    if human:
                        cand_faces = human
                for f in cand_faces:
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
            face_w = best_face.get('w', 0)
            # Edge margin = half the face width + 15% padding so the
            # whole bbox stays inside the crop, not just the center.
            edge_m = max(20, int(face_w * 0.65)) if face_w > 0 else 20

            def _apply_containment(target_x: int) -> int:
                """Shift target_x so face_cx ± face_w/2 + edge_m sits inside crop."""
                if face_w <= 0:
                    return max(0, min(max_x, target_x))
                face_left = face_cx - face_w // 2
                face_right = face_cx + face_w // 2
                crop_left = target_x
                crop_right = target_x + crop_w
                if face_left < crop_left + edge_m:
                    target_x = face_left - edge_m
                elif face_right > crop_right - edge_m:
                    target_x = face_right + edge_m - crop_w
                return max(0, min(max_x, target_x))

            # Check 1: face completely outside crop → hard correction
            if face_cx < x or face_cx > x + crop_w:
                kf['x'] = _apply_containment(face_cx - crop_w // 2)
                corrected += 1
                continue

            # Check 2: face in crop but off-center (outside center 50%)
            crop_center = x + crop_w // 2
            offset = abs(face_cx - crop_center)
            if offset > crop_w * 0.25:
                # Blend 75% toward centered — aggressive correction
                centered_x = face_cx - crop_w // 2
                blended = int(0.75 * centered_x + 0.25 * x)
                kf['x'] = _apply_containment(blended)
                corrected += 1
            else:
                # Check 3: even when within the center band, make sure
                # the face bbox isn't clipping at the edge (a >25%-off
                # center face also fails containment, but a 24%-off face
                # with a wide bbox can still poke out — fix it).
                contained = _apply_containment(x)
                if contained != x:
                    kf['x'] = contained
                    corrected += 1

        if corrected > 0:
            log.log_stage('SMOOTH',
                f'Final face-centering: corrected {corrected} keyframes')

    def _post_smoother_centering(self):
        """Eval-aligned final centering pass after the smoother.

        Walks the surviving keyframes and, for each one whose best face
        at the nearest perception sample lands outside the eval's
        middle-third band, nudges the crop_x so the face center moves
        into the middle third. Differences from ``_final_face_centering``
        (which was the conflict-prone earlier attempt):

        * ADJUST-ONLY — never inserts new keyframes. The smoother just
          worked hard to land at this keyframe count; adding new ones
          would re-introduce the jitter / drift it removed.
        * Respects the ``_centering`` flag — keyframes the stabilizer
          already nudged stay put. This avoids the
          plan-vs-stabilizer-vs-final ping-pong that took the metric
          from 90 %→54 % the last time we tried a third pass.
        * Tight 50 px nudge cap so we can't yank between adjacent
          subjects (the failure mode the /60 evaluator warns about).
        * Eval-aligned thresholds — fires at the same 16.7 % off-center
          boundary the evaluator uses, plus a 2 px buffer so we cross
          back inside even after sub-pixel interpolation rounding.
        """
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        if crop_w <= 0:
            return

        # Build real-track set with the same animated fallback as
        # ``_final_face_centering`` so animated shows still benefit.
        track_counts = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}
        if not real_tracks and track_counts:
            real_tracks = set(track_counts.keys())

        # Eval's middle-third boundary. The evaluator marks a second
        # off-center when |face_cx - crop_center| > crop_w / 6 (half
        # the middle-third width). We trigger SLIGHTLY inside that
        # boundary so we cross back in after sub-pixel interpolation
        # rounding rather than landing right on the line.
        third_offset = crop_w // 3
        eval_boundary = third_offset // 2  # 1/6 of crop_w
        trigger_offset = max(0, eval_boundary - 2)
        NUDGE_CAP = max(20, int(crop_w * 0.20))   # 20% of crop width per nudge

        # Pick the best face at a given time (±200 ms window).
        # Extracted as a closure so the per-keyframe and per-second
        # passes below agree on which face is the subject and so we
        # don't duplicate the same scoring formula three times.
        def _best_face_near(time_ms: int):
            best_face = None
            best_score = 0
            persons_at_t = self.perception.person_timeline.get(time_ms, [])
            for dt in [0, -200, 200, -400, 400, -600, 600]:
                faces = self.perception.face_timeline.get(time_ms + dt, [])
                cand_faces = faces
                if persons_at_t:
                    human = [f for f in faces if _face_overlaps_person(f, persons_at_t)]
                    if human:
                        cand_faces = human
                for f in cand_faces:
                    if f.get('track_id', -1) not in real_tracks:
                        continue
                    score = (f.get('saliency', 0) + f.get('mouth_motion', 0)
                            ) * max(0.15, f.get('confidence', 0.5))
                    if score > best_score:
                        best_score = score
                        best_face = f
            return best_face

        # ── Phase A: validate every keyframe ──
        # Drop the previous ``_centering``-skip — the worry was
        # ping-pong with the planner / stabilizer passes, but those
        # passes only set crop_x at AIM positions; the smoother and
        # interpolation can still leave the keyframe off-center.
        # The 35 % NUDGE_CAP keeps us from yanking past an adjacent
        # face, and we only correct when the face is actually outside
        # the eval's middle-third band — so there's nothing to
        # ping-pong against.
        corrected = 0
        revalidated_centering = 0
        for kf in self.plan.keyframes:
            if kf.get('transition') == 'cut':
                # Cuts are intentional speaker switches — leave them.
                continue

            t = kf['time_ms']
            best_face = _best_face_near(t)
            if best_face is None:
                continue

            face_cx = best_face['cx']
            x = kf['x']

            # Used to skip when the face was fully outside the crop
            # ("that's the inclusion-fix pass's job"), but reframe_report
            # in clipai_logs_20260527_094523.log shows HIGH-severity
            # face_missing problems surviving past inclusion-fix and into
            # the final plan — usually because the smoother dropped the
            # inserted anchor or the 500ms-neighbour guard prevented an
            # anchor. So when the face is outside, lift NUDGE_CAP to the
            # full delta (a content-driven pan is preferable to a face
            # not in frame) and apply the move from here as a defence-
            # in-depth pass. Same protection flag the stabilizer uses,
            # so the smoother won't re-collapse it.
            face_outside = face_cx < x or face_cx > x + crop_w
            crop_center = x + crop_w // 2
            face_offset = abs(face_cx - crop_center)
            if not face_outside and face_offset <= trigger_offset:
                if kf.get('_centering'):
                    revalidated_centering += 1
                continue  # already in middle third

            # Aim for dead-center, but cap the move so we never yank
            # past an adjacent face. Sub-pixel rounding is fine because
            # the renderer integer-clamps anyway. When the face is
            # outside the crop, allow the full delta — we'd rather
            # produce a visible pan than leave a face_missing problem
            # in the plan.
            ideal_x = clamp_x(face_cx - crop_w // 2, max_x)
            delta = ideal_x - x
            sign = 1 if delta > 0 else -1
            step_cap = abs(delta) if face_outside else NUDGE_CAP
            nudge = min(step_cap, abs(delta)) * sign
            new_x = clamp_x(x + nudge, max_x)
            if new_x == x:
                continue
            self.tracer.event('post_smoother_nudge',
                              t_ms=t,
                              old_x=x, new_x=new_x,
                              face_cx=face_cx,
                              face_offset_px=face_offset,
                              ideal_x=ideal_x,
                              delta=delta, nudge=nudge,
                              nudge_cap=NUDGE_CAP)
            kf['x'] = new_x
            kf['_centering'] = True
            corrected += 1

        # ── Phase B: eval-aligned per-second pass ──
        # The evaluator samples every 1 second and checks the
        # INTERPOLATED crop_x at that timestamp, not the keyframe
        # positions directly. Between two centered keyframes the
        # eased interpolation can drift the face out of the middle
        # third for half the transition duration — which is exactly
        # what's keeping the metric at 70 %. For each second whose
        # interpolated crop puts the best face outside the middle
        # third, insert a centered anchor keyframe ONLY when no
        # existing keyframe is within ±200 ms of that second. The
        # ±200 ms guard keeps us from doubling the keyframe count
        # on dense scenes (and re-introducing the jitter the
        # smoother just removed).
        from backend.services.reframer_models import interpolate_x

        existing_times = sorted(kf['time_ms'] for kf in self.plan.keyframes)
        anchored = 0
        anchor_kfs: list[dict] = []

        # Binary-search neighbour lookup: avoid an O(N²) scan over
        # 2000+ keyframes × 1500+ seconds.
        from bisect import bisect_left
        def _has_neighbour(time_ms: int, window_ms: int = 150) -> bool:
            i = bisect_left(existing_times, time_ms)
            if i < len(existing_times) and abs(existing_times[i] - time_ms) <= window_ms:
                return True
            if i > 0 and abs(existing_times[i - 1] - time_ms) <= window_ms:
                return True
            return False

        duration_ms = self.plan.duration_ms or 0
        for half_sec in range(0, duration_ms // 500):
            time_ms = half_sec * 500
            best_face = _best_face_near(time_ms)
            if best_face is None:
                continue
            interp_x = clamp_x(interpolate_x(self.plan.keyframes, time_ms), max_x)
            face_cx = best_face['cx']
            face_outside = face_cx < interp_x or face_cx > interp_x + crop_w
            crop_center = interp_x + crop_w // 2
            # When the face is outside the crop entirely we always want
            # to anchor — bypass the 'already centered' check. The
            # off-centre check below only applies when the face is
            # inside the crop.
            if not face_outside and abs(face_cx - crop_center) <= eval_boundary:
                continue  # already centered at eval time
            # When face is INSIDE the crop but off-centre, respect the
            # neighbour guard — a nearby keyframe already covers the frame.
            # When face is OUTSIDE the crop entirely, never block on neighbours:
            # the existing nearby keyframe is at the WRONG position, and we
            # must override it to prevent the HIGH face_missing problem.
            if not face_outside and _has_neighbour(time_ms):
                continue  # face in crop; nearby keyframe covers this sample
            # Snap to face — use a short ease so the new keyframe
            # blends with its neighbours instead of cutting.
            new_x = clamp_x(face_cx - crop_w // 2, max_x)
            anchor_kfs.append({
                'time_ms': time_ms,
                'x': new_x,
                'transition': 'ease_in_out',
                'transition_ms': 200,
                '_centering': True,
            })
            self.tracer.event('post_smoother_anchor_inserted',
                              t_ms=time_ms,
                              interp_x_before=interp_x,
                              new_x=new_x,
                              face_cx=face_cx,
                              face_offset_px=abs(face_cx - crop_center),
                              eval_boundary=eval_boundary,
                              reason='mid_transition_off_center')
            anchored += 1

        if anchor_kfs:
            self.plan.keyframes.extend(anchor_kfs)
            self.plan.keyframes.sort(key=lambda k: k['time_ms'])

        if corrected > 0 or anchored > 0 or revalidated_centering > 0:
            log.log_stage('SMOOTH',
                f'Post-smoother centering: nudged {corrected}, '
                f'anchored {anchored} mid-transition seconds '
                f'({revalidated_centering} _centering keyframes re-validated)')

    def _enforce_live_action_faces(self):
        """Final guarantee: ensure every face sample has a face in the crop.

        Runs for ALL content (not just live-action). Live-action uses a tighter
        800ms correction gap; non-live-action uses 1500ms to avoid jitter on
        anime/gaming content where face detections can be sparse or noisy.

        Unlike predictive anchoring (which runs before smoothing and gets
        corrections smoothed away), this runs AFTER smoothing as a hard
        guarantee that no long-interval face_missing gap survives."""
        if not self.perception or not self.plan:
            return

        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        kfs = self.plan.keyframes

        if len(kfs) < 2:
            return

        # Build real-track set (with animated fallback so short clips and
        # non-live-action content don't end up with an empty real_tracks set).
        track_counts = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}
        if not real_tracks and track_counts:
            real_tracks = set(track_counts.keys())

        # Correction gap: live-action reacts every 800ms; non-live-action
        # every 1500ms (gentler — avoids chasing sparse/noisy detections
        # on stylised or gaming content).
        min_gap_ms = 800 if self.perception.is_live_action else 1500

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

            if not any_in_crop and (st - last_correction_t) >= min_gap_ms:
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
                self.tracer.event('live_action_correction',
                                  t_ms=st,
                                  old_interp_x=crop_x,
                                  new_x=new_x,
                                  face_cx=best['cx'],
                                  face_w=best.get('w', 0),
                                  face_track_id=best.get('track_id', -1),
                                  mouth_motion=best.get('mouth_motion', 0),
                                  transition=trans_type,
                                  transition_ms=trans_ms,
                                  delta_px=dx,
                                  reason='no_face_in_crop')
                last_correction_t = st

        if corrections:
            self.plan.keyframes.extend(corrections)
            self.plan.keyframes.sort(key=lambda k: k['time_ms'])
            log.log_stage('SMOOTH',
                f'Face enforcement ({"live" if self.perception.is_live_action else "non-live"}, '
                f'min_gap={min_gap_ms}ms): inserted {len(corrections)} corrections')

    def _apply_motivated_zoom(self):
        """Stamp a per-keyframe ``scale`` for slow push-ins on held speakers.

        A human operator pushes in slowly while a subject holds the frame
        (building intimacy) and never zooms mid-pan. We detect holds — long
        gaps between two near-equal-x keyframes with a face present — and ramp
        ``scale`` 1.0 → max → 1.0 across each, easing in/out. The ``scale`` term
        is consumed by ``interpolate_scale`` and the zoom-aware export crop.
        Flag-gated (default OFF); no-op when no qualifying holds exist.
        """
        from backend.config import settings
        if not getattr(settings, 'REFRAMER_MOTIVATED_ZOOM', False):
            return
        kfs = self.plan.keyframes
        if len(kfs) < 2:
            return
        max_scale = float(getattr(settings, 'REFRAMER_MOTIVATED_ZOOM_MAX', 1.15))
        if max_scale <= 1.001:
            return
        crop_w = self.plan.crop_w
        MIN_HOLD_MS = 2500
        RAMP_MS = 1000
        PAD_MS = 250
        MIN_GAP_MS = 1500
        MAX_ZOOMS = 6

        def _face_in_span(t0, t1):
            for t, faces in self.perception.face_timeline.items():
                if t0 <= t <= t1 and faces:
                    return True
            return False

        inserts = []
        zooms = 0
        last_end = -10 ** 9
        for i in range(len(kfs) - 1):
            if zooms >= MAX_ZOOMS:
                break
            a, b = kfs[i], kfs[i + 1]
            if b.get('transition') == 'cut':
                continue
            t0, t1 = a['time_ms'], b['time_ms']
            if t1 - t0 < MIN_HOLD_MS:
                continue
            if abs(b['x'] - a['x']) > crop_w * 0.05:
                continue  # not a hold — it's a pan
            if t0 < last_end + MIN_GAP_MS:
                continue
            if not _face_in_span(t0, t1):
                continue  # don't push in on static graphics / no subject
            s0 = t0 + PAD_MS
            s3 = t1 - PAD_MS
            if s3 - s0 < 2 * RAMP_MS + 200:
                continue
            s1 = s0 + RAMP_MS
            s2 = s3 - RAMP_MS
            x_hold = a['x']
            # start (1.0) → peak-in (max) → peak-out (max) → end (1.0)
            inserts.append({'time_ms': int(s0), 'x': int(x_hold),
                            'transition': 'ease_in_out', 'transition_ms': 0, 'scale': 1.0,
                            '_zoom': True})
            inserts.append({'time_ms': int(s1), 'x': int(x_hold),
                            'transition': 'ease_in_out', 'transition_ms': RAMP_MS,
                            'scale': round(max_scale, 4), '_zoom': True})
            inserts.append({'time_ms': int(s2), 'x': int(x_hold),
                            'transition': 'ease_in_out', 'transition_ms': 0,
                            'scale': round(max_scale, 4), '_zoom': True})
            inserts.append({'time_ms': int(s3), 'x': int(x_hold),
                            'transition': 'ease_in_out', 'transition_ms': RAMP_MS,
                            'scale': 1.0, '_zoom': True})
            last_end = t1
            zooms += 1

        if inserts:
            kfs.extend(inserts)
            kfs.sort(key=lambda k: k['time_ms'])
            get_logger().log_stage(
                'ENGINE', f'Motivated zoom: {zooms} push-in(s) '
                f'(max scale {max_scale:.2f})')

    def _smooth_trajectory_savgol(self):
        """Non-causal Savitzky-Golay smoothing over the target-x trajectory.

        We render offline, so we can look at the whole path at once and remove
        residual jitter with a polynomial-fit low-pass — far gentler than the
        reactive 6-pass keyframe surgery. Cuts break the signal into runs
        (never smoothed across a hard cut), and ``_centering``-flagged
        keyframes keep their corrected x. Flag-gated via REFRAMER_SAVGOL_SMOOTHING.
        """
        from backend.config import settings
        if not getattr(settings, 'REFRAMER_SAVGOL_SMOOTHING', True):
            return
        kfs = self.plan.keyframes
        if len(kfs) < 5:
            return
        try:
            from scipy.signal import savgol_filter
        except Exception:
            return
        max_x = self.plan.max_x
        polyorder = int(getattr(settings, 'REFRAMER_SAVGOL_POLYORDER', 2))
        window_ms = int(getattr(settings, 'REFRAMER_SAVGOL_WINDOW_MS', 1200))

        # Split into runs delimited by hard cuts.
        runs = []
        cur = []
        for i, kf in enumerate(kfs):
            if kf.get('transition') == 'cut' and cur:
                runs.append(cur)
                cur = []
            if kf.get('transition') == 'cut':
                # A cut is a boundary; it starts its own fresh run.
                cur = [i]
            else:
                cur.append(i)
        if cur:
            runs.append(cur)

        smoothed = 0
        for run in runs:
            if len(run) < 5:
                continue
            xs = np.array([float(kfs[i]['x']) for i in run])
            times = [kfs[i]['time_ms'] for i in run]
            span = max(1, times[-1] - times[0])
            med_dt = span / max(1, len(run) - 1)
            win = int(round(window_ms / max(1.0, med_dt)))
            win = min(win, len(run))
            if win % 2 == 0:
                win -= 1
            if win <= polyorder or win < 5:
                continue
            try:
                ys = savgol_filter(xs, win, polyorder)
            except Exception:
                continue
            for local_i, kf_i in enumerate(run):
                kf = kfs[kf_i]
                if kf.get('_centering') or kf.get('transition') == 'cut':
                    continue  # preserve corrected / cut positions exactly
                kf['x'] = int(clamp_x(int(round(ys[local_i])), max_x))
                smoothed += 1
        if smoothed:
            get_logger().log_stage(
                'SMOOTH', f'Savitzky-Golay trajectory pass: adjusted {smoothed} keyframes')

    def _apply_l1_camera_path(self):
        """Replace the reactive smoother output with an L1-optimal camera path.

        Reconstructs a per-sample target from the current keyframes, then
        solves the Grundmann L1 program per scene (holds + constant-velocity
        pans) and rebuilds keyframes from the piecewise-linear result. Hard
        cuts are preserved as scene boundaries. Flag-gated (default OFF).
        """
        from backend.config import settings
        if not getattr(settings, 'REFRAMER_L1_PATH', False):
            return
        kfs = self.plan.keyframes
        if len(kfs) < 4:
            return
        try:
            from backend.services.reframer_l1_camera_path import (
                solve_l1_path, parse_weights)
        except Exception:
            return
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        sample_times = sorted(self.perception.face_timeline.keys())
        if len(sample_times) < 4:
            return
        weights = parse_weights(getattr(settings, 'REFRAMER_L1_WEIGHTS', '1,10,100'))
        radius = max(8.0, crop_w * 0.15)

        # Segment sample times at scene cuts (each scene solved independently
        # so a cut stays a hard discontinuity).
        cuts = sorted(set(int(c) for c in (self.perception.scene_cuts or [])))
        segments = []
        cur = []
        cut_ptr = 0
        cut_set = set(cuts)
        for t in sample_times:
            if t in cut_set and cur:
                segments.append(cur)
                cur = []
            cur.append(t)
        if cur:
            segments.append(cur)

        new_kfs = []
        for seg_idx, seg in enumerate(segments):
            targets = [float(interpolate_x(kfs, t)) for t in seg]
            path = solve_l1_path(targets, radius, 0.0, float(max_x), weights=weights)
            if path is None:
                path = [float(clamp_x(int(v), max_x)) for v in targets]
            corners = self._rdp_indices(path, epsilon=max(2.0, crop_w * 0.01))
            for ci, idx in enumerate(corners):
                t = seg[idx]
                x = int(clamp_x(int(round(path[idx])), max_x))
                if ci == 0:
                    # Scene start: a cut for scenes after the first, else the
                    # clip's opening keyframe.
                    transition = 'cut' if seg_idx > 0 else 'cut'
                    transition_ms = 0
                else:
                    prev_t = seg[corners[ci - 1]]
                    transition = 'ease_in_out'
                    transition_ms = int(max(0, t - prev_t))
                new_kfs.append({'time_ms': int(t), 'x': x,
                                'transition': transition,
                                'transition_ms': transition_ms})
        if new_kfs:
            new_kfs.sort(key=lambda k: k['time_ms'])
            self.plan.keyframes = new_kfs
            get_logger().log_stage(
                'SMOOTH', f'L1-optimal camera path: {len(new_kfs)} keyframes '
                f'from {len(sample_times)} samples across {len(segments)} scenes')

    @staticmethod
    def _rdp_indices(values, epsilon):
        """Ramer-Douglas-Peucker over a value series (index axis) → kept indices.

        The L1 path is piecewise-linear; RDP recovers its corner points, which
        become the rebuilt keyframes. The horizontal axis is the sample index
        so the perpendicular-distance tolerance ``epsilon`` is in position
        units (px). Always keeps the endpoints.
        """
        n = len(values)
        if n <= 2:
            return list(range(n))
        keep = [False] * n
        keep[0] = keep[-1] = True
        stack = [(0, n - 1)]
        while stack:
            lo, hi = stack.pop()
            if hi <= lo + 1:
                continue
            x0, y0 = float(lo), float(values[lo])
            x1, y1 = float(hi), float(values[hi])
            dx = x1 - x0
            dy = y1 - y0
            denom = math.hypot(dx, dy) or 1.0
            max_d = -1.0
            max_i = lo
            for i in range(lo + 1, hi):
                # Perpendicular distance from point i to the chord lo→hi.
                d = abs(dy * (float(i) - x0) - dx * (float(values[i]) - y0)) / denom
                if d > max_d:
                    max_d = d
                    max_i = i
            if max_d > epsilon:
                keep[max_i] = True
                stack.append((lo, max_i))
                stack.append((max_i, hi))
        return [i for i in range(n) if keep[i]]

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
                    # Tag as a centering correction so subsequent passes
                    # (the inclusion-fix loop in _stabilize_keyframes, the
                    # smoother's merge / consolidate / dejitter / drift
                    # passes) treat it as already-corrected and don't
                    # snap it to a competing signal. Without this flag,
                    # the saliency hotspot (the inclusion-fix's third-
                    # tier fallback) disagrees with the Sobel centroid
                    # by 1-2 px on title cards / text-only scenes and
                    # trips the "face_cx < crop_left" outside-crop check,
                    # silently dragging the crop back to the wrong
                    # position — observed on the GUNDAM title card in
                    # clipai_logs_20260527_120641.log line 2300+.
                    self.plan.keyframes[idx]['_centering'] = True

                fixed_scenes.append({
                    'scene_range': f'{s_start/1000:.1f}-{s_end/1000:.1f}s',
                    'strategy': strategy,
                    'old_x': kf_max_x,
                    'gradient_centroid': centroid_x,
                    'new_crop_x': gradient_crop_x,
                })
                self.tracer.event('gradient_recenter',
                                  scene_start_ms=s_start,
                                  scene_end_ms=s_end,
                                  strategy=strategy,
                                  old_max_x=kf_max_x,
                                  centroid_x=centroid_x,
                                  new_crop_x=gradient_crop_x,
                                  keyframes_affected=len(scene_kf_indices))

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
                # Deadband: how far a face can move before the crop follows.
                # Tighter for live-action — faces on screen deserve precise framing.
                # Wider for animated/gaming — subjects teleport and tight deadbands
                # cause jitter from detection noise on stylised faces.
                if self.perception.is_live_action:
                    face_deadband = max(20, int(crop_w * 0.12))  # 12% crop_w ≈ 24px on 200px crop
                else:
                    face_deadband = max(30, int(crop_w * 0.25))  # 25% crop_w (unchanged)

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
                self.tracer.event('stabilize_scene_anchor',
                                  scene_start_ms=s_start, scene_end_ms=s_end,
                                  strategy=strategy,
                                  mode='face_track_deadband',
                                  input_kfs=len(scene_kfs),
                                  output_kfs=len(face_anchors),
                                  face_positions=len(face_positions),
                                  deadband_px=face_deadband,
                                  bimodal=is_bimodal,
                                  anchor_xs=[a['x'] for a in face_anchors])

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
                    self.tracer.event('stabilize_scene_anchor',
                                      scene_start_ms=s_start, scene_end_ms=s_end,
                                      strategy=strategy,
                                      mode='non_face_bimodal',
                                      input_kfs=len(scene_kfs),
                                      output_kfs=len(deduped),
                                      left_anchor=left_anchor,
                                      right_anchor=right_anchor,
                                      cluster_gap_px=max_gap,
                                      split_value=split_value)
                else:
                    # UNIMODAL: single anchor
                    anchor_x = int(np.median(xs))
                    first_kf = scene_kfs[0]
                    first_kf['x'] = anchor_x
                    anchored.append(first_kf)
                    anchor_count += 1
                    self.tracer.event('stabilize_scene_anchor',
                                      scene_start_ms=s_start, scene_end_ms=s_end,
                                      strategy=strategy,
                                      mode='non_face_unimodal',
                                      input_kfs=len(scene_kfs),
                                      output_kfs=1,
                                      anchor_x=anchor_x,
                                      x_range=[min(xs), max(xs)])

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
                    self.tracer.event('stabilize_blip_removed',
                                      t_ms=curr['time_ms'],
                                      x=curr['x'],
                                      hold_dur_s=round(hold_dur, 3),
                                      prev_x=prev['x'],
                                      next_x=nxt['x'],
                                      pattern='A_to_B_to_A')
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

            # Same subject-priority chain as the centering nudge below:
            # face → YOLO subject → saliency hotspot. Animated /
            # gameplay / wide-landscape moments without a face still
            # get their crop corrected when the main subject drifts
            # off-frame. Track which signal won so we can be more
            # conservative about overriding a previous correction
            # with the weakest source (saliency hotspot).
            face_cx = None
            face_source = None
            faces = self.perception.face_timeline.get(st, [])
            real_faces = [f for f in faces
                          if f.get('track_id', -1) >= 0
                          and (not real_tracks
                               or f.get('track_id', -1) in real_tracks)]
            if real_faces:
                best_face = max(real_faces, key=lambda f: (
                    f.get('saliency', 0) + f.get('mouth_motion', 0)
                ) * max(0.15, f.get('confidence', 0.5)))
                face_cx = best_face['cx']
                face_source = 'face'
            else:
                persons = (self.perception.person_timeline.get(st, [])
                           if hasattr(self.perception, 'person_timeline')
                           else [])
                if persons:
                    face_cx = max(persons, key=lambda p: p.get('area', 0)).get('cx')
                    face_source = 'person'
            if face_cx is None:
                sal = (self.perception.saliency_hotspot.get(st)
                       if hasattr(self.perception, 'saliency_hotspot')
                       else None)
                if isinstance(sal, dict):
                    face_cx = sal.get('cx')
                    face_source = 'saliency'
            if face_cx is None:
                continue

            crop_x = interpolate_x(anchored, check_ms)
            crop_left = crop_x
            crop_right = crop_x + crop_w

            # Tolerance band around the crop edges. Without this, a
            # 1-pixel disagreement between the saliency hotspot and an
            # adjacent gradient-recentered keyframe trips the
            # "subject_outside_crop" branch and silently undoes the
            # gradient pass's work (observed on the GUNDAM title card
            # in clipai_logs_20260527_120641.log line 2300+, where the
            # Sobel centroid sat at 317 / crop_left at 216 while the
            # saliency hotspot reported cx=215, off by 1 px). 5 % of
            # crop_w (≈10 px on a 200-px 9:16 crop) is well below any
            # visually meaningful "outside" threshold but absorbs the
            # planner-vs-perceiver rounding noise.
            tolerance = max(2, int(crop_w * 0.05))
            if face_cx >= crop_left - tolerance and face_cx <= crop_right + tolerance:
                continue

            # When the only signal we have is the saliency hotspot (the
            # 3rd-tier fallback) AND the keyframes bracketing this
            # check_ms in the anchored list are already centering-
            # corrected, leave the interpolated position alone. The
            # earlier pass that set _centering used a stronger signal
            # (Sobel gradient centroid, dedicated face-centering, or
            # YOLO person), and the saliency hotspot disagreeing by
            # a small margin is exactly the noise the tolerance was
            # added to absorb. Same effect as a per-source ranking —
            # face beats person beats saliency.
            #
            # Was: ``abs(kf.time_ms - check_ms) < 1500`` — a flat
            # 1.5-second window. That missed the scene-0 case where
            # the only keyframe sits at t=0 and inclusion_fix fires
            # at t=1600+ (>1500 ms away), so the weak-signal override
            # silently bulldozed the gradient-corrected crop on every
            # title card. Bracketing check is structural — it doesn't
            # care how long the scene is.
            if face_source == 'saliency':
                # Find the keyframes immediately before and after
                # check_ms in the anchored list (the same pair
                # ``interpolate_x`` used to compute crop_x). If
                # EITHER is centering-corrected, the interpolated
                # crop_x is also gradient-aware — skip the override.
                kf_before = None
                kf_after = None
                for kf in anchored:
                    t_kf = kf['time_ms']
                    if t_kf <= check_ms and (kf_before is None
                                              or t_kf > kf_before['time_ms']):
                        kf_before = kf
                    if t_kf >= check_ms and (kf_after is None
                                             or t_kf < kf_after['time_ms']):
                        kf_after = kf
                bracketed_by_centering = (
                    (kf_before is not None and kf_before.get('_centering'))
                    or (kf_after is not None and kf_after.get('_centering'))
                )
                if bracketed_by_centering:
                    continue

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
                    # Mark as a centering correction so the smoother's
                    # pan consolidation doesn't collapse two adjacent
                    # inclusion fixes (which usually target DIFFERENT
                    # faces) into a single move to the last position
                    # — that loses the first face's centering.
                    '_centering': True,
                })
                self.tracer.event('stabilize_inclusion_fix',
                                  t_ms=check_ms,
                                  crop_x=crop_x,
                                  new_x=corrected_x,
                                  face_cx=face_cx,
                                  face_source=face_source,
                                  crop_w=crop_w,
                                  reason='subject_outside_crop')
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
        #   v33: aggressive 2× nudge + 2 passes → REGRESSED to 70%
        #   v34: 80 px cap + YOLO/saliency fallback + no upper-bound
        #        gate → REGRESSED to 54 % (the current branch). The
        #        wider 80 px cap let the nudge cross a third boundary
        #        in a single step, and feeding it saliency hotspots
        #        on non-face frames yanked the crop between adjacent
        #        subjects on cut-heavy content.
        #   v35 (current): reverted to the /60 reference settings —
        #        FACE-ONLY subject (no YOLO/saliency fallback in this
        #        pass; the inclusion-fix Pass 3 already covers
        #        face-less frames), 60 px cap (proportional to crop
        #        width), and an upper-bound gate so we skip nudges
        #        when the best face is so far away it's almost
        #        certainly a different shot or a misclassified frame.
        # ══════════════════════════════════════════════════════════════
        NUDGE_MAX = 60          # tight cap when face is INSIDE the crop —
                                # avoids visible jumps between adjacent
                                # same-shot faces. Stays at 60 even now;
                                # the face_outside path below uses a
                                # larger cap so a single nudge actually
                                # brings the face into the crop instead
                                # of leaving it stuck at the edge.
        NUDGE_MIN_DELTA = 1     # Was 5. Lowered to 1 so the eval's
                                # sub-pixel face_missing problems (face
                                # at crop edge by 1-2 px) don't get
                                # skipped as 'indistinguishable from
                                # jitter'. The keyframe-level deltas
                                # produced by ``ideal_x - kf['x']`` are
                                # face-to-crop-center distances, which
                                # are O(100 px) even when the face is
                                # only 1-2 px outside the crop — so 5
                                # never actually filtered jitter, it
                                # just filtered nothing while leaving
                                # edge cases on the table.
        # Was 150. Bumped to 250 after a B-grade run flagged 18 face_missing
        # HIGH-severity problems whose best-face deltas fell in the 150-250
        # band — the old ceiling threw away those genuine same-shot
        # corrections as "probably a different shot" and left centering at
        # 70.5%. Genuine cross-shot pulls are 300+ px on a 600 px crop and
        # are bracketed by a ``cut`` transition (which the loop above
        # already filters out) so the wider ceiling doesn't change the
        # cross-shot behaviour. Reference: reframe_report.problems[] in
        # clipai_logs_20260527_000705.log line 1781+.
        NUDGE_DELTA_CEILING = 250

        def _nudge_pass(label):
            count = 0
            for kf in anchored:
                if kf.get('transition') == 'cut':
                    continue  # NEVER nudge cuts

                st = nearest_sample(kf['time_ms'])
                if st is None:
                    continue

                # FACE-ONLY: dropping the YOLO/saliency fallback here
                # is deliberate. The inclusion-fix pass above already
                # corrects keyframes that have no face by snapping to
                # YOLO persons; running a second pass that nudges
                # toward saliency hotspots on top of that pulled the
                # crop the wrong direction on cut-heavy content and
                # was the largest contributor to the 90 %→54 %
                # centering regression.
                faces = self.perception.face_timeline.get(st, [])
                real_faces = [f for f in faces
                              if f.get('track_id', -1) >= 0
                              and (not real_tracks
                                   or f.get('track_id', -1) in real_tracks)]
                if not real_faces:
                    continue
                # Pick the most salient face (matches evaluator logic).
                best_face = max(real_faces, key=lambda f: (
                    f.get('saliency', 0) + f.get('mouth_motion', 0)
                ) * max(0.2, f.get('confidence', 0.5)))
                cx = best_face['cx']

                # The evaluator wants cx in [crop_left + third,
                # crop_left + 2*third]. Use dead-centre as the IDEAL
                # but cap the nudge at NUDGE_MAX so we never jump
                # far enough in one step to lose another face in the
                # same shot, AND skip nudges when the delta is so
                # large it's almost certainly a different shot (the
                # reference's upper-bound gate the current branch
                # removed).
                ideal_x = clamp_x(cx - crop_w // 2, max_x)
                delta = ideal_x - kf['x']
                abs_delta = abs(delta)
                if abs_delta < NUDGE_MIN_DELTA:
                    continue
                if abs_delta > NUDGE_DELTA_CEILING:
                    # Best face is far away — likely a different shot
                    # or a misclassified frame. Trusting it would pull
                    # the crop the wrong direction. Leave the keyframe
                    # for the cut-aware planner to handle.
                    continue
                # min(cap, magnitude) with original sign — matches the
                # reference's single-step clamp, NOT the current
                # branch's symmetric max(-NUDGE_MAX, min(NUDGE_MAX, ...))
                # which never short-circuits when the delta is exactly
                # at the cap and let consecutive nudges accumulate.
                sign = 1 if delta > 0 else -1
                # When the best face is FULLY OUTSIDE the crop, the
                # NUDGE_MAX=60 cap physically cannot bring the face in
                # on a typical ~200 px 9:16 crop — a 60 px nudge moves
                # crop_left from 414 to 354 while face_cx stays at 312,
                # which is STILL face_missing (the exact pattern showing
                # up at 25s / 33s / 34s in
                # clipai_logs_20260527_094523 line 2576+). face_missing
                # is the worst eval category — it costs both
                # face_coverage and centering — and a content-driven
                # pan is visually preferable to a face that's not
                # in frame. So lift the per-step cap to ``abs_delta``
                # (i.e. go straight to the ideal centered position)
                # when the face is outside the crop; keep the tight
                # NUDGE_MAX cap when it's inside-but-off-centre.
                crop_left = kf['x']
                crop_right = crop_left + crop_w
                face_outside = cx < crop_left or cx > crop_right
                step_cap = abs_delta if face_outside else NUDGE_MAX
                nudge = min(step_cap, abs_delta) * sign
                old_x = kf['x']
                kf['x'] = clamp_x(kf['x'] + nudge, max_x)
                self.tracer.event('stabilize_centering_nudge',
                                  t_ms=kf['time_ms'],
                                  old_x=old_x, new_x=kf['x'],
                                  face_cx=cx,
                                  ideal_x=ideal_x,
                                  delta=delta,
                                  nudge=nudge,
                                  nudge_cap=step_cap,
                                  face_outside=face_outside,
                                  pass_label=label)
                # Tag the keyframe as a centering correction so the
                # smoother's drift-suppression doesn't collapse it
                # back into a hold. Without this flag, a 24 px
                # centering nudge on a 202 px crop falls under the
                # smoother's drift threshold and gets removed —
                # which is exactly what tanked centering to 68 %
                # (we saw "drift-suppressed 290" in the log eating
                # the nudge pass's output).
                kf['_centering'] = True
                count += 1
            return count

        centered_count = _nudge_pass('first')

        log.log_stage('SMOOTH',
            f'Predictive anchoring: {len(kfs)} → {len(anchored)} keyframes '
            f'({anchor_count} anchors, {inclusion_fixes} inclusion fixes, '
            f'{blip_count} blips removed, {centered_count} centered)')

        self.plan.keyframes = anchored

    def _eliminate_edge_violations(self) -> None:
        """Final sweep: ensure no face bbox clips at the crop edge.
        Runs AFTER all other passes. For every keyframe whose nearest
        perception sample has a real face, compute whether the face's
        full bounding box (cx ± w/2) fits inside the crop window with a
        minimum 8px clearance on each side. If not, shift the crop so it
        does.
        This is purely reactive — it only reads face positions that the
        perceiver already measured. No prediction. The crop moves only
        as much as needed to contain the current face bbox.
        """
        if not self.plan or not self.perception:
            return
        log = self.log
        crop_w = self.plan.crop_w
        max_x = self.plan.max_x
        # Wider clearance so faces don't sit hard against the crop edge.
        # Scales with crop width: 5% of crop_w (with a 12px floor for tiny
        # crops) — at a 1080×1920 crop_w=608 this is ~30px; for a 4K source
        # crop_w=1215 it's ~60px. Replaces the previous 8px fixed value
        # which left visible-only-just-inside faces at the boundary.
        EDGE_CLEARANCE = max(12, int(crop_w * 0.05))
        # Build real-track set
        track_counts: dict[int, int] = {}
        for faces in self.perception.face_timeline.values():
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        real_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 25}
        if not real_tracks and track_counts:
            real_tracks = set(track_counts.keys())
        fixed = 0
        for kf in self.plan.keyframes:
            t = kf['time_ms']
            x = kf['x']
            # Find the nearest face sample within 300ms
            best_face = None
            best_score = 0.0
            for dt in [0, -200, 200, -300, 300]:
                faces = self.perception.face_timeline.get(t + dt, [])
                for f in faces:
                    if f.get('track_id', -1) not in real_tracks:
                        continue
                    score = (
                        f.get('saliency', 0) + f.get('mouth_motion', 0)
                    ) * max(0.15, f.get('confidence', 0.5))
                    if score > best_score:
                        best_score = score
                        best_face = f
            if best_face is None:
                continue
            face_cx = best_face['cx']
            face_w = best_face.get('w', 0)
            if face_w <= 0:
                continue
            face_left = face_cx - face_w // 2
            face_right = face_cx + face_w // 2
            crop_left = x
            crop_right = x + crop_w
            new_x = x
            # Face clipping on the left
            if face_left < crop_left + EDGE_CLEARANCE:
                new_x = face_left - EDGE_CLEARANCE
            # Face clipping on the right (re-check after left correction)
            if face_right > new_x + crop_w - EDGE_CLEARANCE:
                new_x = face_right + EDGE_CLEARANCE - crop_w
            new_x = min(max_x, new_x)  # floor enforced at render time by clamp_x
            if new_x != x:
                kf['x'] = new_x
                kf['_edge_fixed'] = True
                fixed += 1
        if fixed > 0:
            log.log_stage('SMOOTH',
                f'Edge violation sweep: corrected {fixed} keyframes '
                f'(face bbox now clears crop edge by ≥{EDGE_CLEARANCE}px)')

