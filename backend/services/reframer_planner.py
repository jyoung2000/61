"""ClipAI Reframer — Planner (Stages 2-3 — classify + decide).

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
from backend.config import settings

logger = logging.getLogger("clipai.reframer_planner")


def _saliency_or_static_center_x(sal_cx, has_motion, crop_w, max_x,
                                 center_static=True):
    """Horizontal crop-x for a no-face / no-person frame whose only signal is a
    single spectral-saliency peak.

    On a motion-less STATIC GRAPHIC (title card / logo / credits) the spectral
    peak latches onto the highest-contrast EDGE (e.g. a centered logo's wing-
    tip), so following it mis-frames the crop off to the side — center instead.
    With motion present (dynamic content) the peak is meaningful, so follow it.
    Gated by ``center_static`` (REFRAMER_CENTER_STATIC_GRAPHICS)."""
    if center_static and not has_motion:
        return clamp_x(max_x // 2, max_x)
    return clamp_x(sal_cx - crop_w // 2, max_x)


class Planner:
    """Route scenes to strategies, emit keyframed RenderPlan."""

    def __init__(self, perc: PerceptionResult, ar_w: int = 9, ar_h: int = 16,
                 tracer: Optional[ReframeTracer] = None):
        self.p = perc
        self.ar_w = ar_w
        self.ar_h = ar_h
        self.tracer: ReframeTracer = tracer if tracer is not None else null_tracer()

        # Compute crop dimensions from aspect ratio
        # Key insight: we're cropping FROM a 16:9 source.
        # For any target ratio narrower than source: crop width (full height)
        # For any target ratio wider/equal to source: crop height (full width)
        src_ratio = perc.src_w / perc.src_h  # e.g. 1.778 for 16:9
        target_ratio = ar_w / ar_h

        if target_ratio < src_ratio:
            # Target is narrower/taller than source → crop width, keep full height
            self.crop_h = perc.src_h
            self.crop_w = round(perc.src_h * ar_w / ar_h)
            self.crop_y = 0
        else:
            # Target is wider/same as source → crop height, keep full width
            self.crop_w = perc.src_w
            self.crop_h = round(perc.src_w * ar_h / ar_w)
            self.crop_y = max(0, (perc.src_h - self.crop_h) // 2)

        self.crop_w = min(self.crop_w, perc.src_w)
        self.crop_h = min(self.crop_h, perc.src_h)
        self.max_x = max(0, perc.src_w - self.crop_w)
        self.is_live_action = perc.is_live_action

        # Target output resolution (scale to common sizes)
        if ar_h >= ar_w:
            self.target_h = 1920
            self.target_w = round(1920 * ar_w / ar_h)
        else:
            self.target_w = 1920
            self.target_h = round(1920 * ar_h / ar_w)

        self.sample_times = sorted(perc.face_timeline.keys())

    def generate(self) -> RenderPlan:
        log = get_logger()
        log.start_timer('plan')
        log.log_stage('CLASSIFY', 'Segmenting scenes and routing strategies',
                       crop_w=self.crop_w, crop_h=self.crop_h,
                       crop_y=self.crop_y, max_x=self.max_x,
                       target=f'{self.target_w}x{self.target_h}',
                       aspect=f'{self.ar_w}:{self.ar_h}')

        plan = RenderPlan(
            source_width=self.p.src_w,
            source_height=self.p.src_h,
            target_width=self.target_w,
            target_height=self.target_h,
            fps=self.p.fps,
            duration_ms=self.p.duration_ms,
            crop_w=self.crop_w,
            crop_h=self.crop_h,
            crop_y=self.crop_y,
        )

        scenes = self._segment_scenes()
        log.log_stage('CLASSIFY', f'Segmented into {len(scenes)} scenes')

        # Log what audio intelligence data is available for reframing
        n_speech = len(self.p.speech_active)
        n_transcript = len(self.p.transcript_segments)
        n_speaker_map = len(self.p.track_speaker_map)
        if n_speech > 0:
            log.log_stage('CLASSIFY',
                f'Using Whisper speech data for reframing: '
                f'{n_speech} speech timestamps, {n_transcript} segments, '
                f'{n_speaker_map} track-speaker links')
        else:
            log.log_stage('CLASSIFY',
                'No audio data available — reframing uses visual-only face scoring')

        # ── REAL TRACKS FILTER ──
        # A "real" track is one that has accumulated enough samples across the
        # whole video to be plausibly a person, not a phantom. With 5 fps
        # sampling and many short-lived ghost tracks, anything under ~25
        # samples (5 seconds of cumulative screen time) is almost certainly
        # noise — toys, posters, art, edge artifacts, or briefly-glimpsed
        # background figures. Both the classifier and the safety pass use
        # this gate so phantom tracks don't drive strategy or corrective cuts.
        global_track_counts = {}
        for faces in self.p.face_timeline.values():
            seen_tracks = set()
            for f in faces:
                tid = f.get('track_id', -1)
                if tid >= 0:
                    seen_tracks.add(tid)
            for tid in seen_tracks:
                global_track_counts[tid] = global_track_counts.get(tid, 0) + 1
        self._real_tracks = {tid for tid, n in global_track_counts.items() if n >= 25}
        log.log_stage('CLASSIFY',
            f'Real-track filter: {len(self._real_tracks)}/{len(global_track_counts)} '
            f'tracks have ≥25 samples (≥5s of screen time)')

        # ── SPEAKER LOCK ──
        # Use Whisper transcript segments as natural utterance boundaries.
        # For each segment, find the real track with the most mouth motion
        # during that segment — that's the active speaker. Build a map
        # of (segment_start_ms, segment_end_ms) → locked_track_id.
        # Both S1 and S2 consult this map and snap the crop to the locked
        # track for the duration of the segment, producing crisp
        # speaker-locked cuts instead of frame-by-frame drift.
        self._speaker_lock = self._build_speaker_lock()
        # Count words that have timestamps
        total_words = sum(len(seg.get('words', [])) for seg in self.p.transcript_segments)
        log.log_stage('CLASSIFY',
            f'Speaker lock: {len(self._speaker_lock)} locks '
            f'({"word-level" if total_words > 0 else "segment-level"}, '
            f'{total_words} words, {len(self.p.transcript_segments)} segments)')

        all_kf = []
        # Measure global content signals once for parameter derivation context
        global_signals = self._measure_global_signals()
        log.log_stage('CLASSIFY',
            f'Global signals: face_density={global_signals.face_density:.2f}, '
            f'motion_energy={global_signals.motion_energy:.2f}, '
            f'speech_density={global_signals.speech_density:.2f}, '
            f'face_persistence={global_signals.face_persistence:.2f}')
        log.log_stage('CLASSIFY',
            f'Content type: {"LIVE-ACTION (face-priority mode)" if self.is_live_action else "animated/other (standard mode)"}')

        # ── TACT: Log coverage ledger report ──
        if self.p.coverage_ledger:
            report = self.p.coverage_ledger.coverage_report()
            status_str = ', '.join(f'{k}: {v}%' for k, v in report['status_pct'].items())
            log.log_stage('CLASSIFY',
                f'TACT coverage: {report["coverage_ratio"]:.1%} '
                f'({report["total_bins"]} bins) — {status_str}')
            if global_signals.quarantine_ratio > 0:
                log.log_stage('CLASSIFY',
                    f'TACT events: music={global_signals.music_density:.1%}, '
                    f'crowd={global_signals.crowd_density:.1%}, '
                    f'silence={global_signals.silence_density:.1%}, '
                    f'quarantined={global_signals.quarantine_ratio:.1%}, '
                    f'confidence_mean={global_signals.confidence_mean:.2f}')

        for i, (s_start, s_end) in enumerate(scenes):
            signals = self._measure_scene_signals(s_start, s_end, global_signals)
            params = self._derive_params(signals)
            kfs = self._decide_adaptive(s_start, s_end, params, scene_idx=i)
            all_kf.extend(kfs)
            plan.strategy_log.append({
                'time_range': [s_start, s_end],
                'strategy': params.strategy_label,
            })
            # Embed measured signals + derived params on every scene
            # entry so the artifact captures WHY the planner picked the
            # strategy it did. Without this the only output of
            # _measure_scene_signals / _derive_params was the strategy
            # label — leaving the per-scene math invisible to anyone
            # reviewing a bad reframe. asdict() is safe here because
            # both SceneSignals and AdaptiveParams are plain dataclasses.
            plan.scenes.append({
                'scene_idx': i,
                'start_ms': s_start,
                'end_ms': s_end,
                'strategy': params.strategy_label,
                'keyframe_count': len(kfs),
                'signals': asdict(signals),
                'params': asdict(params),
            })
            self.tracer.event('scene_signals',
                              scene_idx=i,
                              start_ms=s_start, end_ms=s_end,
                              strategy=params.strategy_label,
                              keyframe_count=len(kfs),
                              signals=asdict(signals),
                              params=asdict(params))
            dur_sec = (s_end - s_start) / 1000
            log.log_stage('DECIDE',
                f'Scene {i+1}/{len(scenes)}: '
                f'{s_start/1000:.1f}-{s_end/1000:.1f}s ({dur_sec:.1f}s) → '
                f'{params.strategy_label} ({len(kfs)} keyframes)')

        # Sort and deduplicate by time
        all_kf.sort(key=lambda k: k['time_ms'])
        deduped = []
        seen = set()
        for kf in all_kf:
            if kf['time_ms'] not in seen:
                seen.add(kf['time_ms'])
                deduped.append(kf)
        plan.keyframes = deduped

        # ── SPEAKER LOCK APPLICATION ──
        # Walk the keyframes and override positions inside speaker-locked
        # utterance segments. For each lock, we:
        #   1. Insert a hard-cut keyframe at the lock's start_ms snapping to
        #      the locked anchor_x (the median position of the active speaker
        #      during this utterance).
        #   2. Replace the x-position of every existing keyframe inside the
        #      lock window with the locked anchor_x.
        #   3. Remove keyframes that would jitter the position within the lock.
        #
        # The result: the crop sits on the active speaker for the duration
        # of their utterance, then cuts cleanly to the next speaker on the
        # next lock boundary. No drift, no per-sample reactions to fidgety
        # listeners.
        if self._speaker_lock:
            locked_count = 0
            inserted_anchors = 0

            # Build a fast lookup: for each kf, find which lock (if any) covers it
            new_kfs = []
            for kf in plan.keyframes:
                t = kf['time_ms']
                lock = self._get_speaker_lock_at(t)
                if lock is not None:
                    # Override the kf's x position with the speaker anchor.
                    # Keep its time and transition style so we don't mess up
                    # transition timing across non-locked → locked boundaries.
                    if kf['x'] != lock['anchor_x']:
                        self.tracer.event('speaker_lock_override',
                                          t_ms=t,
                                          old_x=kf['x'],
                                          new_x=lock['anchor_x'],
                                          lock_start_ms=lock['start_ms'],
                                          lock_end_ms=lock['end_ms'],
                                          track_id=lock['track_id'],
                                          mouth_score=lock.get('mouth_score', 0))
                        kf = {**kf, 'x': lock['anchor_x']}
                        locked_count += 1
                new_kfs.append(kf)

            # Insert a hard-cut anchor at the start of each speaker lock that
            # doesn't already have a keyframe within ±50ms. This guarantees
            # the lock activates cleanly.
            existing_times = {kf['time_ms'] for kf in new_kfs}
            for lock in self._speaker_lock:
                anchor_t = lock['start_ms']
                # Skip if a kf already exists very near this time
                if any(abs(t - anchor_t) < 50 for t in existing_times):
                    continue
                new_kfs.append({
                    'time_ms': anchor_t,
                    'x': lock['anchor_x'],
                    'transition': 'cut',
                    'transition_ms': 0,
                })
                self.tracer.event('speaker_lock_anchor_inserted',
                                  t_ms=anchor_t,
                                  x=lock['anchor_x'],
                                  lock_end_ms=lock['end_ms'],
                                  track_id=lock['track_id'],
                                  word_count=lock.get('word_count', 0))
                existing_times.add(anchor_t)
                inserted_anchors += 1

            new_kfs.sort(key=lambda k: k['time_ms'])
            plan.keyframes = new_kfs

            log.log_stage('DECIDE',
                f'Speaker lock applied: {locked_count} keyframes overridden, '
                f'{inserted_anchors} segment-start anchors inserted')

        # ── FINAL SAFETY: Scan every sample, insert corrections where faces are missed ──
        # The strategies may leave gaps where the interpolated crop doesn't cover faces.
        # This pass walks every sample time, checks the interpolated crop, and inserts
        # a correction keyframe whenever a face is detected but not in the crop.
        # Only acts on real tracks (precomputed above) — phantoms are ignored.
        corrected = 0
        inserted = 0

        def _real_faces_at(t):
            """Return only faces from real tracks at time t."""
            return [f for f in self.p.face_timeline.get(t, [])
                    if f.get('track_id', -1) in self._real_tracks]

        # Pass 1: Fix existing keyframes
        for kf in plan.keyframes:
            t = kf['time_ms']
            x = kf['x']

            # Speaker-locked keyframes: only correct if face is completely
            # outside the crop (HIGH severity). Don't correct off-center
            # positioning within speaker locks — the lock knows who to show.
            is_speaker_locked = self._get_speaker_lock_at(t) is not None

            best_face = None
            best_score = -1
            speech_at_t = self.p.speech_active.get(t, False)
            if not speech_at_t:
                for dt2 in [-100, 100, -200, 200]:
                    if self.p.speech_active.get(t + dt2, False):
                        speech_at_t = True
                        break

            for dt in [0, -100, 100, -200, 200]:
                faces = _real_faces_at(t + dt)
                for f in faces:
                    conf_w = max(0.15, f.get('confidence', 0.5))
                    if speech_at_t:
                        # Hybrid (Option B): mouth-first, motion as tiebreaker
                        score = (f.get('mouth_motion', 0) * 15.0 +
                                 f.get('motion', 0) * 2.0 +
                                 f.get('area', 0) * 0.0001) * conf_w
                    else:
                        score = (f.get('saliency', 0) +
                                 f.get('mouth_motion', 0) * 2 +
                                 f.get('area', 0) * 0.0001) * conf_w
                    if score > best_score:
                        best_score = score
                        best_face = f

            if best_face is None:
                continue

            face_cx = best_face['cx']
            # Check if face is in the crop at all
            if not (x <= face_cx <= x + self.crop_w):
                # Face completely outside crop — always correct, even speaker-locked
                new_x = clamp_x(face_cx - self.crop_w // 2, self.max_x)
                self.tracer.event('safety_correction',
                                  t_ms=t, old_x=x, new_x=new_x,
                                  face_cx=face_cx,
                                  face_track_id=best_face.get('track_id', -1),
                                  best_score=round(best_score, 4),
                                  speaker_locked=is_speaker_locked,
                                  reason='face_outside_crop')
                kf['x'] = new_x
                corrected += 1
            elif not is_speaker_locked:
                # Check centering (center half) — only for non-locked keyframes
                center_left = x + self.crop_w // 4
                center_right = x + (3 * self.crop_w) // 4
                if not (center_left <= face_cx <= center_right):
                    new_x = clamp_x(face_cx - self.crop_w // 2, self.max_x)
                    self.tracer.event('safety_correction',
                                      t_ms=t, old_x=x, new_x=new_x,
                                      face_cx=face_cx,
                                      face_track_id=best_face.get('track_id', -1),
                                      best_score=round(best_score, 4),
                                      speaker_locked=False,
                                      reason='off_center_in_crop')
                    kf['x'] = new_x
                    corrected += 1

        # Pass 2: Check interpolated positions at every sample time
        # and insert new keyframes where faces are missed.
        # Only consider real-track faces here too — phantom detections
        # shouldn't trigger corrective cuts.
        existing_times = {kf['time_ms'] for kf in plan.keyframes}
        new_kfs = []

        for t in self.sample_times:
            if t in existing_times:
                continue  # already has a keyframe

            faces = _real_faces_at(t)
            if not faces:
                continue

            # What would the interpolated crop position be at this time?
            interp_x = clamp_x(interpolate_x(plan.keyframes, t), self.max_x)

            # Is the best face in the interpolated crop?
            speech_at_t = self.p.speech_active.get(t, False)
            if speech_at_t:
                # Hybrid (Option B): mouth-first, motion as tiebreaker
                best_face = max(faces, key=lambda f: (
                    f.get('mouth_motion', 0) * 15.0 +
                    f.get('motion', 0) * 2.0 +
                    f.get('area', 0) * 0.0001
                ) * max(0.15, f.get('confidence', 0.5)))
            else:
                best_face = max(faces, key=lambda f: (
                    f.get('saliency', 0) + f.get('mouth_motion', 0) * 2 +
                    f.get('area', 0) * 0.0001
                ) * max(0.15, f.get('confidence', 0.5)))
            face_cx = best_face['cx']

            if not (interp_x <= face_cx <= interp_x + self.crop_w):
                # Face completely outside the interpolated crop — insert correction
                new_x = clamp_x(face_cx - self.crop_w // 2, self.max_x)
                new_kfs.append({
                    'time_ms': t, 'x': new_x,
                    'transition': 'cut', 'transition_ms': 0
                })
                self.tracer.event('safety_insertion',
                                  t_ms=t, x=new_x,
                                  interp_x_before=interp_x,
                                  face_cx=face_cx,
                                  face_track_id=best_face.get('track_id', -1),
                                  speech_active=speech_at_t,
                                  reason='interp_crop_missed_face')
                inserted += 1

        if new_kfs:
            plan.keyframes.extend(new_kfs)
            plan.keyframes.sort(key=lambda k: k['time_ms'])
            # Dedupe
            seen = set()
            deduped2 = []
            for kf in plan.keyframes:
                if kf['time_ms'] not in seen:
                    seen.add(kf['time_ms'])
                    deduped2.append(kf)
            plan.keyframes = deduped2

        if corrected > 0 or inserted > 0:
            log.log_stage('DECIDE',
                f'Safety pass: corrected {corrected} keyframes, '
                f'inserted {inserted} new corrections '
                f'(total: {len(plan.keyframes)} keyframes)')

        elapsed = log.stop_timer('plan')
        log.log_stage('DECIDE', f'RenderPlan complete: {len(deduped)} keyframes, '
                       f'{len(plan.scenes)} scenes in {elapsed:.2f}s',
                       strategies=[s['strategy'] for s in plan.strategy_log])

        return plan

    # ── Scene segmentation ──

    def _segment_scenes(self) -> List[Tuple[int, int]]:
        """Segment based on visual cuts only — NOT face count changes.
        Face count changes are too noisy for real content (detections flicker)."""
        raw_cuts = sorted(set([0] + self.p.scene_cuts + [self.p.duration_ms]))

        # Deduplicate cuts within 1 second
        deduped = [raw_cuts[0]]
        for c in raw_cuts[1:]:
            if c - deduped[-1] >= 1000:
                deduped.append(c)
        if deduped[-1] != self.p.duration_ms:
            deduped.append(self.p.duration_ms)

        scenes = []
        for i in range(len(deduped) - 1):
            if deduped[i+1] - deduped[i] >= 500:
                scenes.append((deduped[i], deduped[i+1]))

        if not scenes:
            scenes = [(0, self.p.duration_ms)]

        # Merge tiny scenes (< 3s) into neighbors — real podcast scenes are longer
        merged = [scenes[0]]
        for i in range(1, len(scenes)):
            s, e = scenes[i]
            if e - s < 3000 and merged:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        return merged if merged else [(0, self.p.duration_ms)]

    def _build_speaker_lock(self) -> List[dict]:
        """Build speaker-locked segments using word-level timestamps.

        Instead of one lock per transcript segment (which can span 30+
        seconds), this uses per-word timing to identify speaker changes
        mid-sentence. For each word's time window, the track with the most
        mouth motion is the speaker. Consecutive words with the same speaker
        are grouped into a single lock.

        This means if speaker A talks for 5 seconds then speaker B
        interrupts, the lock switches to B within 200ms of the interruption
        — not after 30 seconds when the segment ends.

        When word-level data isn't available, falls back to segment-level
        locks (same as before).
        """
        if not self.p.transcript_segments:
            return []

        real_tracks = getattr(self, '_real_tracks', set())
        log = get_logger()

        # Collect all words across all segments
        all_words = []
        for seg in self.p.transcript_segments:
            if seg.get('words'):
                for w in seg['words']:
                    all_words.append(w)

        if all_words:
            return self._build_word_level_locks(all_words, real_tracks, log)
        else:
            return self._build_segment_level_locks(real_tracks)

    def _build_word_level_locks(self, all_words, real_tracks, log) -> List[dict]:
        """Build locks from word-level timestamps — sub-second speaker tracking.

        TACT enhancements:
          - Quarantined words (hallucinations) are skipped entirely
          - Mouth motion is weighted by word confidence (high conf = strong signal)
          - Speaker switches require minimum confidence (prevents noisy switches)
        """
        locks = []
        ledger = self.p.coverage_ledger  # may be None for non-TACT runs

        # For each word, find which track has the most mouth motion during
        # that word's time window. This identifies the speaker per word.
        word_speakers = []  # [{start, end, track_id, anchor_cx, confidence}, ...]
        quarantined_skipped = 0

        for w in all_words:
            w_start_ms = int(w['start'] * 1000)
            w_end_ms = int(w['end'] * 1000)
            if w_end_ms - w_start_ms < 50:
                w_end_ms = w_start_ms + 200  # minimum 200ms window

            # ── TACT: Skip quarantined words (hallucinations) ──
            if ledger:
                bin_key = (w_start_ms // ledger.bin_width_ms) * ledger.bin_width_ms
                b = ledger.bins.get(bin_key)
                if b and b.status == 'quarantined':
                    quarantined_skipped += 1
                    continue

            # ── TACT: Get word confidence for weighting ──
            word_confidence = w.get('confidence', 1.0)
            if ledger:
                bin_key = (w_start_ms // ledger.bin_width_ms) * ledger.bin_width_ms
                b = ledger.bins.get(bin_key)
                if b and b.confidence > 0:
                    word_confidence = b.confidence

            # Find track with most mouth motion during this word,
            # weighted by word confidence
            track_mouth = {}
            track_cx = {}
            for t in self.sample_times:
                if t < w_start_ms - 100 or t > w_end_ms + 100:
                    continue
                for f in self.p.face_timeline.get(t, []):
                    tid = f.get('track_id', -1)
                    if tid < 0 or (real_tracks and tid not in real_tracks):
                        continue
                    # Weight mouth motion by word confidence
                    weighted_motion = f.get('mouth_motion', 0) * word_confidence
                    track_mouth[tid] = track_mouth.get(tid, 0) + weighted_motion
                    track_cx.setdefault(tid, []).append(f['cx'])

            if not track_mouth:
                continue

            best_tid = max(track_mouth, key=track_mouth.get)
            if track_cx.get(best_tid):
                # Use position clustering for the anchor
                cxs = track_cx[best_tid]
                if len(cxs) >= 2:
                    bucket_size = max(100, self.crop_w // 3)
                    buckets = {}
                    for cx in cxs:
                        buckets.setdefault(cx // bucket_size, []).append(cx)
                    best_bucket = max(buckets.values(), key=len)
                    anchor_cx = int(np.median(best_bucket))
                else:
                    anchor_cx = cxs[0]
            else:
                continue

            word_speakers.append({
                'start': w['start'],
                'end': w['end'],
                'track_id': best_tid,
                'anchor_cx': anchor_cx,
                'confidence': word_confidence,
                'mouth_score': track_mouth.get(best_tid, 0),
            })

        if not word_speakers:
            return []

        # Pre-compute scene cut set for fast lookup
        scene_cuts_set = set(self.p.scene_cuts)

        # ── Majority-vote smoothing: 3-word sliding window ──
        # Instead of trusting per-word speaker attribution (noisy),
        # take the majority track over a 3-word window. This prevents
        # single-frame mouth-motion spikes from switching the speaker.
        if len(word_speakers) >= 3:
            smoothed = []
            for i in range(len(word_speakers)):
                window_start = max(0, i - 1)
                window_end = min(len(word_speakers), i + 2)
                window = word_speakers[window_start:window_end]
                # Count track_id votes in the window
                vote_counts = {}
                for ws in window:
                    tid = ws['track_id']
                    vote_counts[tid] = vote_counts.get(tid, 0) + 1
                majority_tid = max(vote_counts, key=vote_counts.get)
                ws_copy = dict(word_speakers[i])
                ws_copy['track_id'] = majority_tid
                smoothed.append(ws_copy)
            word_speakers = smoothed

        # Group consecutive words with the same speaker into locks
        # Each lock accumulates evidence: mouth scores, confidences, word count
        current_lock = {
            'start_ms': int(word_speakers[0]['start'] * 1000),
            'end_ms': int(word_speakers[0]['end'] * 1000),
            'track_id': word_speakers[0]['track_id'],
            'anchor_cxs': [word_speakers[0]['anchor_cx']],
            'mouth_scores': [word_speakers[0].get('mouth_score', 0)],
            'confidences': [word_speakers[0].get('confidence', 1.0)],
            'word_count': 1,
        }

        for ws in word_speakers[1:]:
            ws_start = int(ws['start'] * 1000)
            ws_end = int(ws['end'] * 1000)

            # ── Scene-cut awareness: break locks at shot boundaries ──
            # A scene cut means a new camera angle — the speaker lock from
            # the previous shot shouldn't carry over because face positions
            # change completely. Check if any scene cut falls between the
            # current lock's end and this word's start.
            crosses_cut = False
            for cut_ms in scene_cuts_set:
                if current_lock['end_ms'] < cut_ms <= ws_start:
                    crosses_cut = True
                    break

            # Same speaker, close in time, no scene cut → extend current lock
            if (ws['track_id'] == current_lock['track_id']
                    and ws_start - current_lock['end_ms'] < 1000
                    and not crosses_cut):
                current_lock['end_ms'] = ws_end
                current_lock['anchor_cxs'].append(ws['anchor_cx'])
                current_lock['mouth_scores'].append(ws.get('mouth_score', 0))
                current_lock['confidences'].append(ws.get('confidence', 1.0))
                current_lock['word_count'] += 1
            else:
                # ── TACT: Require minimum confidence for speaker switches ──
                # Low-confidence words shouldn't trigger camera jumps
                if ws.get('confidence', 1.0) < 0.6 and not crosses_cut:
                    # Extend current lock instead of switching
                    current_lock['end_ms'] = ws_end
                    current_lock['word_count'] += 1
                    continue

                # Speaker changed (or scene cut) → finalize and switch
                self._finalize_lock(current_lock, locks)
                current_lock = {
                    'start_ms': ws_start,
                    'end_ms': ws_end,
                    'track_id': ws['track_id'],
                    'anchor_cxs': [ws['anchor_cx']],
                    'mouth_scores': [ws.get('mouth_score', 0)],
                    'confidences': [ws.get('confidence', 1.0)],
                    'word_count': 1,
                }

        self._finalize_lock(current_lock, locks)

        log.log_stage('CLASSIFY',
            f'Word-level speaker lock: {len(locks)} locks from '
            f'{len(word_speakers)} word-speaker pairs'
            + (f' ({quarantined_skipped} hallucinated words skipped)'
               if quarantined_skipped > 0 else ''))

        return locks

    def _finalize_lock(self, current_lock, locks):
        """Convert a raw lock dict into a finalized speaker lock entry.

        Evidence-based validation — a lock is valid when the accumulated
        evidence proves this face IS the active speaker, regardless of how
        long they appear on screen. A speaker on camera for 100ms with clear
        mouth motion during a high-confidence word is a legitimate lock.
        A face visible for 3 seconds with zero mouth motion is noise.

        Evidence score = accumulated_mouth_motion × mean_confidence × word_count_factor
          - mouth_motion: sum of per-word mouth motion scores (from face timeline)
          - confidence: mean TACT word confidence (0–1)
          - word_count_factor: log(1 + word_count) — one word can be enough,
            but more words increase confidence
        """
        duration_ms = current_lock['end_ms'] - current_lock['start_ms']
        mouth_scores = current_lock.get('mouth_scores', [])
        confidences = current_lock.get('confidences', [])
        word_count = current_lock.get('word_count', len(mouth_scores))

        # Compute evidence score
        total_mouth = sum(mouth_scores) if mouth_scores else 0
        mean_conf = float(np.mean(confidences)) if confidences else 0.5
        word_factor = float(np.log1p(word_count))  # log(1 + n): 1 word → 0.69, 3 → 1.39, 10 → 2.4

        evidence = total_mouth * mean_conf * word_factor

        # Minimum evidence threshold:
        #   - 1 word with strong mouth motion (0.5) and high confidence (0.9):
        #     evidence = 0.5 * 0.9 * 0.69 = 0.31 → VALID
        #   - 1 word with weak mouth motion (0.05) and low confidence (0.3):
        #     evidence = 0.05 * 0.3 * 0.69 = 0.01 → INVALID (noise)
        #   - 5 words with moderate mouth motion (0.2 each) and decent confidence (0.7):
        #     evidence = 1.0 * 0.7 * 1.79 = 1.25 → VALID
        min_evidence = 0.15

        if evidence < min_evidence:
            return  # insufficient evidence — this lock is noise

        # Use clustered position for the anchor
        cxs = current_lock['anchor_cxs']
        if len(cxs) >= 2:
            bucket_size = max(100, self.crop_w // 3)
            buckets = {}
            for cx in cxs:
                buckets.setdefault(cx // bucket_size, []).append(cx)
            best_bucket = max(buckets.values(), key=len)
            anchor_cx = int(np.median(best_bucket))
        else:
            anchor_cx = cxs[0]

        locks.append({
            'start_ms': current_lock['start_ms'],
            'end_ms': current_lock['end_ms'],
            'track_id': current_lock['track_id'],
            'anchor_x': clamp_x(anchor_cx - self.crop_w // 2, self.max_x),
            'mouth_score': round(evidence, 3),
            'word_count': word_count,
        })

    def _build_segment_level_locks(self, real_tracks) -> List[dict]:
        """Fallback: segment-level locks when word timestamps unavailable."""
        locks = []
        for seg in self.p.transcript_segments:
            seg_start_ms = int(seg['start_sec'] * 1000)
            seg_end_ms = int(seg['end_sec'] * 1000)
            if seg_end_ms - seg_start_ms < 300:
                continue

            track_mouth_total = {}
            track_positions = {}
            for t in self.sample_times:
                if t < seg_start_ms or t > seg_end_ms:
                    continue
                for f in self.p.face_timeline.get(t, []):
                    tid = f.get('track_id', -1)
                    if tid < 0 or (real_tracks and tid not in real_tracks):
                        continue
                    track_mouth_total[tid] = (
                        track_mouth_total.get(tid, 0) + f.get('mouth_motion', 0))
                    track_positions.setdefault(tid, []).append(f['cx'])

            if not track_mouth_total:
                continue

            speaker_tid = max(track_mouth_total, key=track_mouth_total.get)
            speaker_score = track_mouth_total[speaker_tid]
            seg_dur_sec = (seg_end_ms - seg_start_ms) / 1000.0
            if speaker_score < 0.02 * max(1.0, seg_dur_sec):
                continue

            if speaker_tid in track_positions and track_positions[speaker_tid]:
                cxs = track_positions[speaker_tid]
                if len(cxs) >= 3:
                    bucket_size = max(100, self.crop_w // 3)
                    buckets = {}
                    for cx in cxs:
                        buckets.setdefault(cx // bucket_size, []).append(cx)
                    best_bucket = max(buckets.values(), key=len)
                    anchor_cx = int(np.median(best_bucket))
                else:
                    anchor_cx = int(np.median(cxs))
                anchor_x = clamp_x(anchor_cx - self.crop_w // 2, self.max_x)
            else:
                anchor_x = self.max_x // 2

            locks.append({
                'start_ms': seg_start_ms,
                'end_ms': seg_end_ms,
                'track_id': speaker_tid,
                'anchor_x': anchor_x,
                'mouth_score': round(speaker_score, 4),
            })

        return locks

    def _get_speaker_lock_at(self, time_ms: int) -> Optional[dict]:
        """Return the speaker lock active at time_ms, or None."""
        for lock in self._speaker_lock:
            if lock['start_ms'] <= time_ms <= lock['end_ms']:
                return lock
        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  SIGNAL MEASUREMENT — Content tells us how to reframe it
    # ═══════════════════════════════════════════════════════════════════════

    def _is_speech_active(self, t: int) -> bool:
        """Check if speech is active at time t.
        Uses TACT coverage ledger when available (confidence-gated),
        falls back to boolean speech_active with ±200ms tolerance."""
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            bin_key = (t // ledger.bin_width_ms) * ledger.bin_width_ms
            b = ledger.bins.get(bin_key)
            if b:
                return b.status == 'covered_speech' and b.confidence > 0.4
            return False
        # Fallback: existing boolean speech_active (±200ms tolerance)
        if self.p.speech_active.get(t, False):
            return True
        for dt in [-100, 100, -200, 200]:
            if self.p.speech_active.get(t + dt, False):
                return True
        return False

    def _speech_confidence_at(self, t: int) -> float:
        """Return ASR confidence at time t (0.0–1.0).
        Returns 1.0 for non-TACT runs where speech is active, 0.0 otherwise."""
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            bin_key = (t // ledger.bin_width_ms) * ledger.bin_width_ms
            b = ledger.bins.get(bin_key)
            if b and b.status in ('covered_speech', 'low_confidence'):
                return b.confidence
            return 0.0
        return 1.0 if self._is_speech_active(t) else 0.0

    def _audio_event_at(self, t: int) -> Optional[str]:
        """Return the audio event type at time t (music/applause/laughter/etc).
        Returns None if no event or TACT not available."""
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            bin_key = (t // ledger.bin_width_ms) * ledger.bin_width_ms
            b = ledger.bins.get(bin_key)
            if b and b.event_type:
                return b.event_type
        return self.p.audio_events.get(t)

    def _is_quarantined_at(self, t: int) -> bool:
        """Check if audio at time t is quarantined (hallucination)."""
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            bin_key = (t // ledger.bin_width_ms) * ledger.bin_width_ms
            b = ledger.bins.get(bin_key)
            if b:
                return b.status == 'quarantined'
        return False

    def _measure_global_signals(self) -> SceneSignals:
        """Measure content-wide signal characteristics."""
        s = SceneSignals()
        total = len(self.sample_times) or 1

        face_counts = [len(self.p.face_timeline.get(t, [])) for t in self.sample_times]
        s.face_density = float(np.mean(face_counts)) if face_counts else 0

        track_counts = {}
        for t in self.sample_times:
            for f in self.p.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid >= 0 and tid in self._real_tracks:
                    track_counts[tid] = track_counts.get(tid, 0) + 1
        if track_counts:
            s.face_persistence = float(np.mean([c / total for c in track_counts.values()]))
            s.dominant_track_share = max(track_counts.values()) / total
        else:
            s.face_persistence = 0
            s.dominant_track_share = 0

        motions = [self.p.motion_timeline.get(t, 0) for t in self.sample_times]
        s.motion_energy = float(np.mean(motions)) if motions else 0

        speech_count = sum(1 for t in self.sample_times if self._is_speech_active(t))
        s.speech_density = speech_count / total

        # ── TACT: Populate event-based signal densities ──
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            all_bins = list(ledger.bins.values())
            n_bins = max(1, len(all_bins))
            s.music_density = sum(1 for b in all_bins
                                  if b.event_type == 'music') / n_bins
            s.silence_density = sum(1 for b in all_bins
                                    if b.status == 'covered_silence') / n_bins
            s.crowd_density = sum(1 for b in all_bins
                                  if b.event_type in ('applause', 'crowd', 'laughter')) / n_bins
            speech_bins = [b for b in all_bins if b.status == 'covered_speech']
            s.confidence_mean = (float(np.mean([b.confidence for b in speech_bins]))
                                 if speech_bins else 0.0)
            s.quarantine_ratio = sum(1 for b in all_bins
                                     if b.status == 'quarantined') / n_bins

        areas = []
        frame_area = max(1, self.p.src_w * self.p.src_h)
        for t in self.sample_times:
            for f in self.p.face_timeline.get(t, []):
                areas.append(f.get('area', 0) / frame_area)
        s.face_area_ratio = float(np.mean(areas)) if areas else 0

        dur_min = max(0.01, self.p.duration_ms / 60000.0)
        s.cut_rate_per_min = len(self.p.scene_cuts) / dur_min

        # ── Universal subject signals ──
        person_count = sum(1 for t in self.sample_times
                          if self.p.person_timeline.get(t))
        s.person_density = person_count / total

        sal_intensities = [self.p.saliency_hotspot[t]['intensity']
                          for t in self.sample_times
                          if t in self.p.saliency_hotspot]
        s.saliency_strength = float(np.mean(sal_intensities)) if sal_intensities else 0

        return s

    def _measure_scene_signals(self, start: int, end: int,
                                global_sig: SceneSignals) -> SceneSignals:
        """Measure signal characteristics for a specific scene."""
        s = SceneSignals()
        samples = [t for t in self.sample_times if start <= t < end]
        n = len(samples) or 1

        face_counts = [len(self.p.face_timeline.get(t, [])) for t in samples]
        s.face_density = float(np.mean(face_counts)) if face_counts else 0

        scene_tracks = {}
        for t in samples:
            for f in self.p.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid >= 0 and tid in self._real_tracks:
                    scene_tracks[tid] = scene_tracks.get(tid, 0) + 1
        if scene_tracks:
            s.face_persistence = float(np.mean([c / n for c in scene_tracks.values()]))
            s.dominant_track_share = max(scene_tracks.values()) / n
        else:
            s.face_persistence = 0
            s.dominant_track_share = 0

        # Spatial spread
        if len(scene_tracks) >= 2:
            track_medians = {}
            for t in samples:
                for f in self.p.face_timeline.get(t, []):
                    tid = f.get('track_id', -1)
                    if tid in scene_tracks:
                        track_medians.setdefault(tid, []).append(f['cx'])
            if len(track_medians) >= 2:
                medians = [int(np.median(cxs)) for cxs in track_medians.values()]
                s.spatial_spread = (max(medians) - min(medians)) / max(1, self.crop_w)

        motions = [self.p.motion_timeline.get(t, 0) for t in samples]
        s.motion_energy = float(np.mean(motions)) if motions else 0

        speech_count = sum(1 for t in samples if self._is_speech_active(t))
        s.speech_density = speech_count / n

        # ── TACT: Populate event-based signal densities for this scene ──
        if self.p.coverage_ledger:
            ledger = self.p.coverage_ledger
            scene_bins = [ledger.bins.get((t // ledger.bin_width_ms) * ledger.bin_width_ms)
                          for t in samples]
            scene_bins = [b for b in scene_bins if b is not None]
            n_bins = max(1, len(scene_bins))
            s.music_density = sum(1 for b in scene_bins
                                  if b.event_type == 'music') / n_bins
            s.silence_density = sum(1 for b in scene_bins
                                    if b.status == 'covered_silence') / n_bins
            s.crowd_density = sum(1 for b in scene_bins
                                  if b.event_type in ('applause', 'crowd', 'laughter')) / n_bins
            speech_bins = [b for b in scene_bins if b.status == 'covered_speech']
            s.confidence_mean = (float(np.mean([b.confidence for b in speech_bins]))
                                 if speech_bins else 0.0)
            s.quarantine_ratio = sum(1 for b in scene_bins
                                     if b.status == 'quarantined') / n_bins

        # Speaker alternation
        if len(scene_tracks) >= 2 and speech_count > 0:
            prev_best = None
            switches = 0
            for t in samples:
                if not self._is_speech_active(t):
                    continue
                best_tid = None
                best_mouth = 0.0
                for f in self.p.face_timeline.get(t, []):
                    tid = f.get('track_id', -1)
                    if tid in scene_tracks:
                        m = f.get('mouth_motion', 0)
                        if m > best_mouth:
                            best_mouth = m
                            best_tid = tid
                if best_tid is not None and best_mouth > 0.005:
                    if prev_best is not None and best_tid != prev_best:
                        switches += 1
                    prev_best = best_tid
            s.speaker_alternation = switches / max(1, speech_count)

        areas = []
        frame_area = max(1, self.p.src_w * self.p.src_h)
        for t in samples:
            for f in self.p.face_timeline.get(t, []):
                areas.append(f.get('area', 0) / frame_area)
        s.face_area_ratio = float(np.mean(areas)) if areas else 0

        hotspots = [self.p.motion_hotspot.get(t) for t in samples]
        active_hs = [h for h in hotspots if h and h['intensity'] > 0.02]
        if len(active_hs) >= 2:
            hxs = [h['cx'] for h in active_hs]
            spread = max(hxs) - min(hxs)
            s.motion_concentration = 1.0 - min(1.0, spread / max(1, self.p.src_w))
        else:
            s.motion_concentration = 0.5

        scene_cuts = [c for c in self.p.scene_cuts if start <= c <= end]
        scene_dur_min = max(0.01, (end - start) / 60000.0)
        s.cut_rate_per_min = len(scene_cuts) / scene_dur_min

        # ── Universal subject signals ──
        person_count = sum(1 for t in samples if self.p.person_timeline.get(t))
        s.person_density = person_count / n

        sal_intensities = [self.p.saliency_hotspot[t]['intensity']
                          for t in samples if t in self.p.saliency_hotspot]
        s.saliency_strength = float(np.mean(sal_intensities)) if sal_intensities else 0

        return s

    def _derive_params(self, s: SceneSignals) -> AdaptiveParams:
        """Derive continuous adaptive parameters from measured signals.

        NO genre labels, NO strategy strings. Just continuous functions
        mapping signal measurements to behavior parameters.

        stability_demand = how much the content wants the crop to hold still
        speaker_demand = how much the content is speech-driven
        action_demand = how much the content wants responsive tracking
        """
        p = AdaptiveParams()

        inv_motion = 1.0 - min(1.0, s.motion_energy / 5.0)
        stability_demand = (s.face_persistence * 0.35
                            + s.speech_density * 0.35
                            + inv_motion * 0.30)

        speaker_demand = (s.speech_density * 0.40
                          + s.speaker_alternation * 0.35
                          + min(1.0, s.spatial_spread) * 0.25)

        inv_face = 1.0 - min(1.0, s.face_density / 3.0)
        action_demand = (min(1.0, s.motion_energy / 5.0) * 0.50
                         + inv_face * 0.25
                         + min(1.0, s.cut_rate_per_min / 30.0) * 0.25)

        # Deadzone
        # Deadzone — minimum movement to trigger a crop update.
        # PROPORTIONAL to crop width so it works on all resolutions:
        #   640px source (crop_w=202): deadzone = 6–16px (3–8%)
        #   3840px source (crop_w=1215): deadzone = 36–97px (3–8%)
        base_dz = 0.03 + stability_demand * 0.04 - action_demand * 0.02
        base_dz = max(0.03, min(0.08, base_dz))
        p.deadzone_px = max(8, int(self.crop_w * base_dz))

        # Hold time
        p.hold_ms = int(200 + stability_demand * 600 - action_demand * 200)
        p.hold_ms = max(100, min(1000, p.hold_ms))

        # ── Fast-cut damper ──
        # When the source has rapid cuts (>10/min), increase hold time and
        # deadzone so the camera doesn't chase every split-second appearance.
        # Proportional to crop width so it works on all resolutions.
        if s.cut_rate_per_min > 10:
            cut_damper = min(1.0, (s.cut_rate_per_min - 10) / 20.0)
            p.hold_ms = max(p.hold_ms, int(400 + cut_damper * 400))
            p.deadzone_px = max(p.deadzone_px, int(self.crop_w * (0.03 + cut_damper * 0.03)))

        # Transition style
        if speaker_demand > 0.4 and s.spatial_spread > 0.3:
            p.use_cut = True
            p.transition_ms = 0
        elif action_demand > 0.6:
            p.use_cut = False
            p.transition_ms = int(100 + (1.0 - action_demand) * 300)
        else:
            p.use_cut = False
            p.transition_ms = int(150 + stability_demand * 200)

        # Speaker weight
        p.speaker_weight = min(1.0, s.speech_density * 0.8 + s.speaker_alternation * 0.4)

        # Anchor blend (0=live, 1=home) — keep moderate to follow face
        # through camera angle changes within a scene. Deadzone handles
        # frame-to-frame noise; anchor_blend just adds gentle stabilization.
        # After SFace consolidation, tracks can span multiple camera angles,
        # so high anchor_blend would fight the face's actual position.
        p.anchor_blend = min(0.40, stability_demand * 0.35 + speaker_demand * 0.15)

        # Responsiveness
        p.responsiveness = max(0.2, min(1.0, 0.3 + action_demand * 0.5 +
                                         (1.0 - stability_demand) * 0.3))

        # ── Universal subject tracking: motion_weight ──
        # How much to trust motion/saliency signals when no face is detected.
        # Derived from signal measurements, not genre labels.
        #   High motion + low faces → follow motion strongly
        #   High person density → follow persons (YOLO), less motion chasing
        #   High saliency → stable anchor on visually important region
        #   Low everything → hold center / last good position
        if s.face_density < 0.1:
            # No faces: signal-derive the motion weight
            motion_factor = min(1.0, s.motion_energy / 5.0)
            # Persons (YOLO) reduce motion_weight because we have a better signal
            person_dampening = s.person_density * 0.4
            p.motion_weight = max(0.1, min(0.9,
                motion_factor * 0.6 + (1.0 - s.motion_concentration) * 0.2
                - person_dampening))
        else:
            # Faces present: motion is barely used (face scoring dominates)
            p.motion_weight = 0.1

        # Label for logging only
        if s.face_density < 0.1 and s.person_density > 0.3:
            p.strategy_label = 'adaptive_person'
        elif s.face_density < 0.1 and action_demand > 0.5:
            p.strategy_label = 'adaptive_motion'
        elif s.face_density < 0.1 and s.saliency_strength > 0.3:
            p.strategy_label = 'adaptive_saliency'
        elif speaker_demand > 0.4 and s.spatial_spread > 0.3:
            p.strategy_label = 'adaptive_speaker'
        elif s.face_density > 0.05:
            p.strategy_label = 'adaptive_face'
        else:
            p.strategy_label = 'adaptive_center'

        # ── TACT: Event-aware strategy modifiers ──
        # Override strategy when the coverage ledger reveals non-speech events.
        # These modifiers only activate when the ledger provides event data.

        # MUSIC SCENE: de-prioritize mouth-motion, follow movement/saliency
        if s.music_density > 0.5:
            p.speaker_weight = min(p.speaker_weight, 0.1)
            p.responsiveness = max(p.responsiveness, 0.6)
            p.strategy_label = 'tact_music_follow'

        # CROWD/APPLAUSE: hold steady, don't chase individual faces
        elif s.crowd_density > 0.4:
            p.speaker_weight = 0.0
            p.deadzone_px = max(p.deadzone_px, int(self.crop_w * 0.12))
            p.hold_ms = max(p.hold_ms, 800)
            p.strategy_label = 'tact_crowd_hold'

        # HIGH-CONFIDENCE DIALOGUE: trust speaker lock fully
        elif s.speech_density > 0.6 and s.confidence_mean > 0.8:
            p.speaker_weight = min(1.0, s.speech_density * 0.9 +
                                    s.speaker_alternation * 0.4)
            p.strategy_label = 'tact_speaker_high_conf'

        # LOW-CONFIDENCE SPEECH: blend speaker lock with face heuristics
        elif s.speech_density > 0.3 and s.confidence_mean < 0.5:
            p.speaker_weight = min(0.6, s.speech_density * 0.4)
            p.strategy_label = 'tact_speaker_low_conf'

        # QUARANTINE-HEAVY: this scene has hallucination issues
        # Discount speaker lock to avoid ghost-lock camera snaps
        if s.quarantine_ratio > 0.15:
            p.speaker_weight *= 0.3
            if 'tact' not in p.strategy_label:
                p.strategy_label = 'tact_quarantine_discount'

        return p

    # ═══════════════════════════════════════════════════════════════════════
    #  UNIVERSAL DECIDE — One function for all content types
    # ═══════════════════════════════════════════════════════════════════════

    def _decide_adaptive(self, start: int, end: int, params: AdaptiveParams,
                         scene_idx: int = -1) -> List[dict]:
        """Universal keyframe generator — snappy, locked on the active subject.

        Core behavior:
          - LOCK: crop dead-centers on the active face at all times
          - SNAP: when the active subject changes, hard cut instantly
          - HOLD: resist switching targets for hold_ms to prevent flicker
          - ANCHOR: while holding on a target, blend toward home position
                    for stability (no drift from small head movements)
          - PROTECT: ensure face edges aren't clipped at crop boundaries
        """
        samples = [t for t in self.sample_times if start <= t < end]
        if not samples:
            x = clamp_x(self.max_x // 2, self.max_x)
            return [{'time_ms': start, 'x': x, 'transition': 'cut', 'transition_ms': 0}]

        # Reset EMA target at scene boundary (each scene starts fresh)
        if hasattr(self, '_ema_target'):
            del self._ema_target

        # Build per-track home positions (median cx → crop position)
        track_cxs = {}
        track_widths = {}
        for t in samples:
            for f in self.p.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid >= 0 and (not self._real_tracks or tid in self._real_tracks):
                    track_cxs.setdefault(tid, []).append(f['cx'])
                    track_widths.setdefault(tid, []).append(f.get('w', 0))

        track_homes = {}
        for tid, cxs in track_cxs.items():
            # Cluster positions to avoid averaging across camera angles
            if len(cxs) >= 3:
                bucket_size = max(100, self.crop_w // 3)
                buckets = {}
                for cx in cxs:
                    key = cx // bucket_size
                    buckets.setdefault(key, []).append(cx)
                best_bucket = max(buckets.values(), key=len)
                home_cx = int(np.median(best_bucket))
            else:
                home_cx = int(np.median(cxs))
            track_homes[tid] = clamp_x(home_cx - self.crop_w // 2, self.max_x)

        kfs = []
        prev_x = None
        prev_target_tid = None
        last_switch_ms = start
        last_face_x = self.max_x // 2
        last_face_seen_ms = start - 10000  # long ago (no face seen yet)
        samples_holding = 0  # how many consecutive samples the camera has held position
        # ── Margin-based hysteresis state ──
        challenger_tid = None
        challenger_wins = 0

        # ── Temporal mouth motion accumulator ──
        # Track accumulated mouth motion per track over a sliding window.
        # This separates "has been speaking for the last second" (sustained
        # high mouth motion) from "smiled once" (single-frame spike).
        # At 5fps, window_size=5 = 1 second lookback.
        window_size = 5
        # Pre-build per-track mouth motion history for the entire scene
        track_mouth_history: Dict[int, List[float]] = {}
        for t in samples:
            for f in self.p.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid >= 0 and (not self._real_tracks or tid in self._real_tracks):
                    track_mouth_history.setdefault(tid, []).append(
                        f.get('mouth_motion', 0))

        # Pre-compute per-sample accumulated mouth motion for each track
        # accumulated_mouth[sample_index] = {track_id: accumulated_value}
        accumulated_mouth: List[Dict[int, float]] = []
        for si, t in enumerate(samples):
            acc = {}
            for f in self.p.face_timeline.get(t, []):
                tid = f.get('track_id', -1)
                if tid < 0 or (self._real_tracks and tid not in self._real_tracks):
                    continue
                # Look back up to window_size samples for this track
                history = track_mouth_history.get(tid, [])
                if not history:
                    acc[tid] = f.get('mouth_motion', 0)
                    continue
                # Find how far into this track's history we are
                # Count how many times this track has appeared up to sample si
                count = 0
                for prev_si in range(max(0, si - window_size), si + 1):
                    prev_t = samples[prev_si]
                    for pf in self.p.face_timeline.get(prev_t, []):
                        if pf.get('track_id', -1) == tid:
                            count += 1
                            acc[tid] = acc.get(tid, 0) + pf.get('mouth_motion', 0)
                            break
                # Normalize by window count so the value is an average, not a sum
                if tid in acc and count > 0:
                    acc[tid] = acc[tid] / count
            accumulated_mouth.append(acc)

        for si, t in enumerate(samples):
            faces = self.p.face_timeline.get(t, [])
            real_faces = [f for f in faces
                          if f.get('track_id', -1) >= 0
                          and (not self._real_tracks
                               or f.get('track_id', -1) in self._real_tracks)]

            # Filter out non-human faces (figurines, artwork, logos)
            # by requiring overlap with a YOLO person detection
            persons_at_t = self.p.person_timeline.get(t, [])
            if persons_at_t and real_faces:
                human_faces = [f for f in real_faces
                               if _face_overlaps_person(f, persons_at_t)]
                if human_faces:  # only filter if we'd keep at least one
                    real_faces = human_faces

            speech_active = self._is_speech_active(t)
            switching = False
            subject_source = None  # reset each sample

            if real_faces:
                # Score faces using ACCUMULATED mouth motion (who's BEEN speaking)
                # instead of instantaneous mouth motion (who moved their mouth THIS frame)
                sw = params.speaker_weight if speech_active else params.speaker_weight * 0.2
                acc_mouth = accumulated_mouth[si] if si < len(accumulated_mouth) else {}

                # Compute mouth motion dominance: what fraction of total mouth
                # motion does each face have? If one face has 90% of the
                # accumulated mouth motion, it's clearly the speaker.
                total_acc_mouth = sum(acc_mouth.get(f.get('track_id', -1), 0)
                                     for f in real_faces) or 0.001

                # When speech is not active: face AREA is the primary signal.
                # The bigger face (closer to camera) is the subject to frame.
                # Without this, two similar-sized faces produce near-equal scores
                # and the camera sits between them, framing neither.
                area_weight = 8.0 if not speech_active else 2.0

                def face_score(f):
                    tid = f.get('track_id', -1)
                    # Use accumulated mouth motion (sustained speaking signal)
                    mouth_acc = acc_mouth.get(tid, f.get('mouth_motion', 0))
                    # Also keep instantaneous for responsiveness
                    mouth_inst = f.get('mouth_motion', 0)
                    # Blend: 70% accumulated (stability), 30% instantaneous (responsiveness)
                    mouth = mouth_acc * 0.7 + mouth_inst * 0.3

                    # Dominance bonus: if this face has most of the mouth motion,
                    # boost it further. This is the "who's the speaker" signal.
                    dominance = acc_mouth.get(tid, 0) / total_acc_mouth
                    dominance_bonus = dominance * 8.0  # strong bonus for dominant speaker

                    motion = f.get('motion', 0)
                    area_norm = f.get('area', 0) * 0.0001
                    saliency = f.get('saliency', 0)
                    speech_s = mouth * 20.0 + dominance_bonus
                    visual_s = motion * 3.0 + area_norm * area_weight + saliency
                    raw_score = sw * speech_s + (1 - sw) * visual_s
                    # Weight by detection confidence — validated false positives
                    # (background artwork, posters) have reduced confidence and
                    # should not compete with real human faces.
                    conf_weight = max(0.15, f.get('confidence', 0.5))
                    return raw_score * conf_weight

                # Score all faces
                scored = [(f, face_score(f)) for f in real_faces]
                scored.sort(key=lambda x: -x[1])
                best_face, best_score = scored[0]
                best_tid = best_face.get('track_id', -1)

                # ── Margin-based hysteresis ──
                # When speech is active, a small margin is enough to switch
                # (speaker changes are meaningful). When speech is NOT active,
                # require a much larger margin — visual-only differences between
                # similar faces are noise, not a reason to move the camera.
                margin_threshold = 5.0 if speech_active else 12.0

                if best_tid != prev_target_tid and prev_target_tid is not None:
                    # Check if the current target is still visible
                    incumbent_faces = [(f, s) for f, s in scored
                                       if f.get('track_id', -1) == prev_target_tid]

                    if not incumbent_faces:
                        # Current target LEFT THE FRAME → switch immediately
                        challenger_tid = None
                        challenger_wins = 0
                        # Allow the switch to proceed
                    else:
                        incumbent_score = incumbent_faces[0][1]
                        margin = best_score - incumbent_score

                        if margin > margin_threshold:
                            # LARGE margin → clear change, switch NOW
                            challenger_tid = None
                            challenger_wins = 0
                        else:
                            # SMALL margin → ambiguous, require confirmation
                            if best_tid == challenger_tid:
                                challenger_wins += 1
                            else:
                                challenger_tid = best_tid
                                challenger_wins = 1

                            if challenger_wins < 2:
                                # Not enough confirmation — keep incumbent
                                best_face = incumbent_faces[0][0]
                                best_tid = prev_target_tid
                                best_score = incumbent_score
                else:
                    # Same target as before — reset challenger
                    challenger_tid = None
                    challenger_wins = 0

                face_w = best_face.get('w', 0)
                face_h = best_face.get('h', 0)

                # ── True-center targeting with full-face containment ──
                # Goal: the ENTIRE face (plus breathing room) must be inside
                # the crop. Not just the face center — the full bounding box
                # including forehead and chin.
                #
                # Edge margin = half the face width + 15% padding. This ensures
                # the face is never clipped by the crop boundary, even when the
                # face is near the edge of the source frame.
                edge_margin = max(20, int(face_w * 0.65))

                # Start by centering the crop on the face center
                center_x = clamp_x(best_face['cx'] - self.crop_w // 2, self.max_x)

                # Compute face edges in source coordinates
                face_left = best_face['cx'] - face_w // 2
                face_right = best_face['cx'] + face_w // 2
                crop_left = center_x
                crop_right = center_x + self.crop_w

                # Full-face containment: if the face bbox + margin extends
                # beyond the crop boundary, shift the crop to contain it.
                if face_left < crop_left + edge_margin:
                    # Face is being clipped on the left side of the crop
                    center_x = clamp_x(face_left - edge_margin, self.max_x)
                elif face_right > crop_right - edge_margin:
                    # Face is being clipped on the right side of the crop
                    center_x = clamp_x(face_right + edge_margin - self.crop_w, self.max_x)

                # ── Hold enforcement ──
                if best_tid != prev_target_tid:
                    if (prev_x is not None
                            and t - last_switch_ms < params.hold_ms
                            and prev_target_tid is not None):
                        # Still in hold period — keep previous position
                        target_x = prev_x
                    else:
                        # Hold expired — SWITCH to new target with hard cut
                        switching = True
                        last_switch_ms = t
                        prev_target_tid = best_tid
                        # On switch: go directly to face center (no blending)
                        target_x = center_x
                else:
                    # Same target as before — blend home vs live.
                    # Reduced from 0.5 → 0.3 multiplier so the live face
                    # position dominates over the static "home" anchor.
                    # The home anchor was originally added to stabilize
                    # against jitter, but the EMA already handles that —
                    # over-weighting home pulls the crop away from where
                    # the face actually is right now.
                    ab = params.anchor_blend * 0.3
                    if best_tid in track_homes:
                        home_x = track_homes[best_tid]
                        target_x = clamp_x(int(home_x * ab + center_x * (1 - ab)),
                                           self.max_x)
                    else:
                        target_x = center_x

                    # ── Post-blend face containment check ──
                    # The anchor_blend can pull the crop away from the face.
                    # Re-verify the face is fully inside the crop after blending.
                    blended_left = target_x
                    blended_right = target_x + self.crop_w
                    if face_left < blended_left + edge_margin:
                        target_x = clamp_x(face_left - edge_margin, self.max_x)
                    elif face_right > blended_right - edge_margin:
                        target_x = clamp_x(face_right + edge_margin - self.crop_w,
                                           self.max_x)

                last_face_x = target_x
                last_face_seen_ms = t  # remember when we last had a face
            else:
                # ── UNIVERSAL SUBJECT TRACKING (no face detected) ──
                subject_x = None
                subject_source = None
                subject_cx = None  # source-frame center of the chosen subject

                # 1. YOLO person detection — best non-face signal.
                #    Only used when no face signal is available.
                persons = self.p.person_timeline.get(t, [])
                if subject_x is None and persons:
                    # ── Face proxy: if faces are visible, use the face
                    # position instead of person-box centroid ──
                    # Person boxes cover head-to-toe, so their cx may be
                    # offset from the face by 40-80px in dynamic poses.
                    all_faces_here = self.p.face_timeline.get(t, [])
                    high_conf = [f for f in all_faces_here
                                 if f.get('confidence', 0) >= 0.5
                                 and f.get('area', 0) > 400]
                    # Filter non-human faces using person overlap
                    if high_conf and persons:
                        human_hc = [f for f in high_conf
                                    if _face_overlaps_person(f, persons)]
                        if human_hc:
                            high_conf = human_hc
                    if high_conf:
                        # Use face position as proxy for person centering
                        proxy_face = max(high_conf, key=lambda f: f.get('area', 0))
                        person_cx = proxy_face['cx']
                    else:
                        # No face visible — use largest/most central person box
                        def person_score(p):
                            center_dist = abs(p['cx'] - self.p.src_w / 2) / max(1, self.p.src_w)
                            area_norm = p['area'] / max(1, self.p.src_w * self.p.src_h)
                            return area_norm * 3.0 + (1.0 - center_dist) * 1.0
                        best_person = max(persons, key=person_score)
                        person_cx = best_person['cx']

                    raw_x = clamp_x(person_cx - self.crop_w // 2, self.max_x)

                    # Person containment (only when using person box, not face proxy)
                    if not high_conf:
                        person_margin = max(20, int(best_person['w'] * 0.3))
                        person_left = best_person['x']
                        person_right = best_person['x'] + best_person['w']
                        if person_left < raw_x + person_margin:
                            raw_x = clamp_x(person_left - person_margin, self.max_x)
                        elif person_right > raw_x + self.crop_w - person_margin:
                            raw_x = clamp_x(person_right + person_margin - self.crop_w,
                                           self.max_x)

                    subject_x = raw_x
                    subject_source = 'person'
                    subject_cx = person_cx

                # 2. Spectral saliency — finds visually important regions.
                #    Handles: static anime shots, text overlays, game HUDs,
                #    bright subjects on dark backgrounds.
                if subject_x is None:
                    sal = self.p.saliency_hotspot.get(t)
                    if sal and sal['intensity'] > 0.10:
                        _mhs = self.p.motion_hotspot.get(t)
                        _has_motion = bool(_mhs and _mhs.get('intensity', 0) > 0.01)
                        _center_static = bool(getattr(
                            settings, "REFRAMER_CENTER_STATIC_GRAPHICS", True))
                        subject_x = _saliency_or_static_center_x(
                            sal['cx'], _has_motion, self.crop_w, self.max_x,
                            center_static=_center_static)
                        if _center_static and not _has_motion:
                            # Centered a static graphic (no face/person/motion):
                            # a title card / logo, framed center not on the
                            # saliency edge. A face here is overridden by
                            # FACE-ALWAYS-WINS below.
                            subject_cx = self.max_x // 2 + self.crop_w // 2
                            subject_source = 'static_center'
                        else:
                            subject_cx = sal['cx']
                            subject_source = 'saliency'

                # 3. Motion centroid — follows where the action is.
                #    Handles: racing, sports wide shots, anime fights,
                #    explosions, any fast-moving content.
                if subject_x is None:
                    hs = self.p.motion_hotspot.get(t)
                    if hs and hs['intensity'] > 0.01:
                        # Blend with hold position using motion_weight
                        mw = params.motion_weight
                        hotspot_x = clamp_x(hs['cx'] - self.crop_w // 2, self.max_x)
                        hold_x = prev_x if prev_x is not None else last_face_x
                        subject_x = clamp_x(int(mw * hotspot_x + (1 - mw) * hold_x),
                                           self.max_x)
                        subject_source = 'motion'
                        subject_cx = hs['cx']

                # 4. Brightness/contrast center — when nothing else works,
                #    center on the brightest region of the frame.
                #    Handles: dark scenes with one bright element, explosions,
                #    energy beams, UI elements, any content where the "subject"
                #    is simply what stands out visually.
                if subject_x is None:
                    # Use the motion hotspot as a proxy for brightness center
                    # (the motion grid's cell_means correlate with visual activity)
                    hs = self.p.motion_hotspot.get(t)
                    sal = self.p.saliency_hotspot.get(t)
                    if sal and sal['intensity'] > 0.08:
                        subject_x = clamp_x(sal['cx'] - self.crop_w // 2, self.max_x)
                        subject_source = 'brightness'
                        subject_cx = sal['cx']
                    elif hs:
                        subject_x = clamp_x(hs['cx'] - self.crop_w // 2, self.max_x)
                        subject_source = 'brightness'
                        subject_cx = hs['cx']

                # 5. Hold — absolute last resort
                if subject_x is None:
                    subject_x = prev_x if prev_x is not None else last_face_x
                    subject_source = 'hold'

                # ═══════════════════════════════════════════════════════
                # FACE ALWAYS WINS: even when the strategy is non-face
                # (adaptive_saliency, adaptive_person, adaptive_motion),
                # if a face is clearly visible RIGHT NOW, center on it.
                # A camera operator ALWAYS frames the face — saliency,
                # person boxes, and motion are fallbacks for faceless frames.
                # ═══════════════════════════════════════════════════════
                all_faces_at_t = self.p.face_timeline.get(t, [])
                high_conf_faces = [f for f in all_faces_at_t
                                   if f.get('confidence', 0) >= 0.5
                                   and f.get('area', 0) > 400]
                # Filter non-human faces using YOLO person overlap
                if high_conf_faces and persons_at_t:
                    human_hc = [f for f in high_conf_faces
                                if _face_overlaps_person(f, persons_at_t)]
                    if human_hc:
                        high_conf_faces = human_hc
                if high_conf_faces:
                    # PICK ONE: largest face (most prominent in frame)
                    winner = max(high_conf_faces, key=lambda f: f.get('area', 0))
                    face_cx = winner['cx']
                    face_w = winner.get('w', 0)
                    # Center crop on this face with containment
                    override_x = clamp_x(face_cx - self.crop_w // 2, self.max_x)
                    # Containment check
                    face_left = face_cx - face_w // 2
                    face_right = face_cx + face_w // 2
                    edge_m = max(15, int(face_w * 0.5))
                    if face_left < override_x + edge_m:
                        override_x = clamp_x(face_left - edge_m, self.max_x)
                    elif face_right > override_x + self.crop_w - edge_m:
                        override_x = clamp_x(face_right + edge_m - self.crop_w,
                                            self.max_x)
                    subject_x = override_x
                    subject_source = 'face_override'
                    subject_cx = face_cx
                    last_face_x = override_x
                    last_face_seen_ms = t  # face override counts as face seen

                target_x = subject_x

            # ════════════════════════════════════════════════════════════════
            #  SNAP-OR-LOCK: the camera either moves decisively or holds
            #  perfectly still. No micro-adjustments, no gradual drift.
            #
            #  This is how professional editors work:
            #    - Small difference from current position → LOCK (hold still)
            #    - Large difference → SNAP (move decisively with eased transition)
            #    - Speaker switch → CUT (instant snap, intentional)
            #
            #  Thresholds are proportional to crop width:
            #    Lock zone:  < 8% of crop width → hold completely still
            #    Snap zone:  > 8% of crop width → move to target with ease
            #    Cut zone:   > 30% of crop width → hard cut (scene change)
            # ════════════════════════════════════════════════════════════════

            # ── Human-editor-style thresholds ──
            # Live-action: small lock zone (8%) so face corrections are applied quickly.
            # The code comment previously said "< 8% → hold still" but used 15%;
            # this aligns implementation with documentation.
            # Animated: wider lock zone for stability (less critical to center faces).
            if self.is_live_action:
                lock_threshold = max(8, int(self.crop_w * 0.08))    # 8% = responsive face tracking
            else:
                lock_threshold = max(12, int(self.crop_w * 0.20))   # 20% = stable for animated

            # ── Velocity-adaptive EMA on target position ──
            # When the face is far from the current crop center, use a higher alpha
            # to snap quickly. When already close, use a low alpha for smooth panning.
            # This prevents the crop from lagging a second behind a fast-moving subject
            # while still looking smooth during small adjustments.
            if not switching:
                if not hasattr(self, '_ema_target'):
                    self._ema_target = target_x
                else:
                    ema_center_now = self._ema_target + self.crop_w // 2
                    target_center  = target_x       + self.crop_w // 2
                    dist_pct = abs(target_center - ema_center_now) / max(1, self.crop_w)
                    if dist_pct > 0.30:
                        ema_alpha = 0.55   # far: snap quickly
                    elif dist_pct > 0.15:
                        ema_alpha = 0.40   # moderate: track steadily
                    else:
                        ema_alpha = 0.25 if not self.is_live_action else 0.30  # close: smooth
                    self._ema_target = clamp_x(
                        int(ema_alpha * target_x + (1 - ema_alpha) * self._ema_target),
                        self.max_x)
                target_x = self._ema_target
            else:
                # On speaker switch: reset EMA to snap immediately
                self._ema_target = target_x

            # ═══════════════════════════════════════════════════════
            # POST-EMA CENTERING CONSTRAINT (iterative)
            # If the chosen FACE is outside the eval's middle-third
            # boundary (16.7 % off-center), pull the crop toward it.
            # Iterate up to 3× so extreme off-center cases (face at
            # frame edge after EMA) converge into the middle third —
            # a single blend at 0.70 only closes 70 % of the gap, so
            # a face at 100 % off lands at 30 % which is STILL outside
            # the middle third. Three iterations of 0.70 close to 97 %
            # of the gap, putting even the worst case at 3 % off-center.
            #
            # ``real_faces`` gates on ``_real_tracks`` (track has ≥25
            # samples across the video), which filters out brief
            # cutaway shots — exactly the case the screenshot at
            # t=8:51.84 shows, where a face is on the LEFT side but
            # the crop sits at the RIGHT speaker's previous position.
            # FACE_WINS earlier in the loop already accepted this
            # cutaway face into ``subject_x`` (it uses ``high_conf_faces``,
            # NOT ``real_faces``); fall back to those here too so the
            # post-EMA centering pulls the cutaway face into the middle
            # third instead of leaving the crop pinned to the previous
            # speaker's position. A human camera op would always frame
            # the visible face, even a brief one.
            # ═══════════════════════════════════════════════════════
            faces_for_centering = real_faces or high_conf_faces
            if faces_for_centering:
                face_cx_now = (best_face['cx'] if real_faces
                               else max(faces_for_centering,
                                        key=lambda f: f.get('area', 0))['cx'])
                # Trigger at the eval's middle-third boundary plus a
                # small buffer (0.18 vs 0.167) so we catch cases the
                # eval would mark off-center. Reference used 0.28/0.30
                # which left a 12 % dead-band where the eval failed but
                # the planner did nothing.
                # Fire at 12% off-center (tightened from 18%) to close the dead-band
                # between the 8% lock zone and the centering trigger. 5 iterations
                # at blend=0.70 close 99.9% of the gap vs 97% at 3 iterations.
                trigger_pct = 0.12
                trigger_offset = int(self.crop_w * trigger_pct)
                blend = 0.70  # per-iteration pull
                max_iters = 5

                # If the face is FULLY OUTSIDE the post-EMA crop, no
                # amount of blend at 0.70 will pull it back inside in
                # one sample (the EMA bias from the previous speaker
                # position dominates). Snap directly to the centered
                # position and let downstream smoothing decide whether
                # to ease the transition. Mirrors the
                # face_outside-cap-lift in the stabilizer's centering
                # nudge (reframer_engine.py) and is the planner-side
                # equivalent of the inclusion-fix that re-centers
                # face-missing keyframes after the smoother runs.
                crop_left_now = target_x
                crop_right_now = target_x + self.crop_w
                if face_cx_now < crop_left_now or face_cx_now > crop_right_now:
                    target_x = clamp_x(
                        face_cx_now - self.crop_w // 2, self.max_x)
                else:
                    for _it in range(max_iters):
                        crop_center = target_x + self.crop_w // 2
                        face_offset = abs(face_cx_now - crop_center)
                        if face_offset <= trigger_offset:
                            break
                        centered_x = clamp_x(face_cx_now - self.crop_w // 2, self.max_x)
                        new_target = clamp_x(
                            int(blend * centered_x + (1 - blend) * target_x), self.max_x)
                        if new_target == target_x:
                            break  # clamped — can't pull further
                        target_x = new_target
                self._ema_target = target_x

            if prev_x is None:
                # First keyframe — just set it
                kfs.append({'time_ms': t, 'x': target_x,
                            'transition': 'cut', 'transition_ms': 0})
                prev_x = target_x
            elif switching:
                # Speaker switch — always a deliberate hard cut
                kfs.append({'time_ms': t, 'x': target_x,
                            'transition': 'cut', 'transition_ms': 0})
                prev_x = target_x
            else:
                delta = abs(target_x - prev_x)

                # When a visible high-confidence face would fall OUTSIDE
                # the current prev_x crop, force a snap regardless of
                # lock_threshold. The lock zone exists to suppress
                # micro-jitter on a held subject, not to ignore a
                # cutaway whose face is fully off-screen. Without this
                # override the planner can lock into the previous
                # speaker's position even when the visible frame's
                # face is at the opposite edge — the failure mode
                # shown at t=8:51.84 in the user's screenshot
                # (crop_x=411 while face_cx≈300 on a 437-wide max_x).
                force_snap = False
                if faces_for_centering:
                    crop_left_prev = prev_x
                    crop_right_prev = prev_x + self.crop_w
                    if face_cx_now < crop_left_prev or face_cx_now > crop_right_prev:
                        force_snap = True

                if delta <= lock_threshold and not force_snap:
                    # ── LOCK: hold completely still ──
                    # The target is close enough. Don't move at all.
                    pass  # prev_x stays exactly where it is

                else:
                    # ── PAN or CUT: distance-based transition ──
                    # Reverted to /60-reference thresholds. The current
                    # branch's 0.28 cut threshold + 400-950 ms durations
                    # let medium eases drift the crop off the best face
                    # mid-transition; the reference's tighter 0.20 cut +
                    # shorter eases keep the crop on-target during
                    # transitions.
                    dist_ratio = delta / max(1, self.crop_w)

                    if self.is_live_action and dist_ratio > 0.20:
                        kfs.append({'time_ms': t, 'x': target_x,
                                    'transition': 'cut', 'transition_ms': 0})
                        prev_x = target_x
                    else:
                        if dist_ratio < 0.15:
                            trans_ms = 300
                        elif dist_ratio < 0.30:
                            trans_ms = 500
                        elif dist_ratio < 0.50:
                            trans_ms = 700
                        else:
                            trans_ms = 800

                        kfs.append({'time_ms': t, 'x': target_x,
                                    'transition': 'ease_in_out',
                                    'transition_ms': trans_ms})
                        prev_x = target_x

        if not kfs:
            if track_homes:
                best_tid = max(track_cxs.keys(), key=lambda t: len(track_cxs[t]))
                x = track_homes[best_tid]
            else:
                x = clamp_x(self.max_x // 2, self.max_x)
            kfs.append({'time_ms': start, 'x': x,
                        'transition': 'cut', 'transition_ms': 0})

        kfs = self._validate_face_in_crop(kfs, start, end)
        # Emit one trace line per keyframe so consumers can join later
        # smoother / centering events on the same time_ms. We do this
        # in one batch at function exit rather than per-append to keep
        # the inner loop clean.
        if self.tracer.enabled:
            for kf in kfs:
                self.tracer.event('keyframe_decided',
                                  scene_idx=scene_idx,
                                  t_ms=kf['time_ms'],
                                  x=kf['x'],
                                  transition=kf.get('transition', 'cut'),
                                  transition_ms=kf.get('transition_ms', 0),
                                  strategy=params.strategy_label)
        return kfs


    def _validate_face_in_crop(self, kfs: List[dict], start: int, end: int) -> List[dict]:
        """Safety net: ensure faces are FULLY inside the crop, not clipped.

        Two-level check:
          Level 1: If a face CENTER is completely outside the crop → snap to it
          Level 2: If a face is IN the crop but its edges are CLIPPED by the
                   crop boundary → shift the crop to contain the full face

        This catches:
          - Faces that drifted outside due to EMA smoothing or anchor blending
          - Faces partially clipped at the crop edge (forehead or chin cut off)
          - New faces that appeared after the keyframe was generated
        """
        if not kfs:
            return kfs

        for kf in kfs:
            t = kf['time_ms']
            x = kf['x']

            # Find faces near this keyframe time
            nearby_faces = []
            for sample_t in self.sample_times:
                if abs(sample_t - t) <= 400:
                    faces = self.p.face_timeline.get(sample_t, [])
                    nearby_faces.extend(faces)

            if not nearby_faces:
                continue

            crop_left = x
            crop_right = x + self.crop_w

            # Level 1: Check if ANY face center is inside the crop
            faces_in_crop = [f for f in nearby_faces
                            if crop_left <= f['cx'] <= crop_right]

            if not faces_in_crop:
                # No face in crop at all — snap to the best face
                best_face = max(nearby_faces,
                                key=lambda f: (f.get('mouth_motion', 0) * 10.0 +
                                               f.get('area', 0) * 0.0001
                                              ) * max(0.15, f.get('confidence', 0.5)))
                face_w = best_face.get('w', 0)
                margin = max(20, int(face_w * 0.65))
                new_x = clamp_x(best_face['cx'] - self.crop_w // 2, self.max_x)
                # Containment check
                face_left = best_face['cx'] - face_w // 2
                face_right = best_face['cx'] + face_w // 2
                if face_left < new_x + margin:
                    new_x = clamp_x(face_left - margin, self.max_x)
                elif face_right > new_x + self.crop_w - margin:
                    new_x = clamp_x(face_right + margin - self.crop_w, self.max_x)
                kf['x'] = new_x
                continue

            # Level 2: Check if any face in the crop is partially clipped
            # Find the most important face in the crop (highest score)
            best_in_crop = max(faces_in_crop,
                              key=lambda f: (f.get('mouth_motion', 0) * 10.0 +
                                             f.get('area', 0) * 0.0001))
            face_w = best_in_crop.get('w', 0)
            if face_w < 10:
                continue  # Too small to worry about clipping

            margin = max(20, int(face_w * 0.5))
            face_left = best_in_crop['cx'] - face_w // 2
            face_right = best_in_crop['cx'] + face_w // 2

            # Is the face clipped on either side?
            needs_shift = False
            new_x = x

            if face_left < crop_left + margin:
                # Face clipped on the left — shift crop left
                new_x = clamp_x(face_left - margin, self.max_x)
                needs_shift = True
            elif face_right > crop_right - margin:
                # Face clipped on the right — shift crop right
                new_x = clamp_x(face_right + margin - self.crop_w, self.max_x)
                needs_shift = True

            if needs_shift:
                # Only shift if the correction is small (< 15% of crop width)
                # Large corrections likely mean a different scene — don't fight
                if abs(new_x - x) < self.crop_w * 0.15:
                    kf['x'] = new_x

        return kfs

