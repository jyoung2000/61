"""ClipAI Reframer — Smoother (Stage 4 — camera path refinement).

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
    ReframeTracer, null_tracer,
)

logger = logging.getLogger("clipai.reframer_smoother")


class Smoother:
    """Post-process keyframes for smooth, natural-feeling output."""

    def __init__(self, max_vel_px_per_sec: float = 2000,
                 tracer: Optional[ReframeTracer] = None):
        self.max_vel = max_vel_px_per_sec
        self.tracer: ReframeTracer = tracer if tracer is not None else null_tracer()

    def smooth(self, plan: RenderPlan) -> RenderPlan:
        log = get_logger()
        kfs = plan.keyframes
        if len(kfs) < 2:
            log.log_stage('SMOOTH', f'Skipped (only {len(kfs)} keyframe)')
            self.tracer.event('smoother_skipped',
                              keyframes=len(kfs),
                              reason='not_enough_keyframes')
            return plan
        self.tracer.event('smoother_start',
                          input_keyframes=len(kfs),
                          crop_w=plan.crop_w,
                          max_velocity_px_per_s=self.max_vel)

        # Pass 1: Remove temporal duplicates and clamp velocity
        smoothed = [kfs[0]]
        clamped_count = 0
        removed_count = 0
        for i in range(1, len(kfs)):
            kf = dict(kfs[i])
            prev = smoothed[-1]
            dt = (kf['time_ms'] - prev['time_ms']) / 1000.0
            if dt <= 0.001:
                self.tracer.event('smoother_drop',
                                  pass_name='dedup',
                                  t_ms=kf['time_ms'], x=kf['x'],
                                  prev_t_ms=prev['time_ms'],
                                  reason='temporal_duplicate')
                removed_count += 1
                continue

            if kf['transition'] != 'cut':
                dx = abs(kf['x'] - prev['x'])
                vel = dx / dt
                if vel > self.max_vel:
                    max_dx = int(self.max_vel * dt)
                    sign = 1 if kf['x'] > prev['x'] else -1
                    new_x = clamp_x(prev['x'] + sign * max_dx, plan.max_x)
                    self.tracer.event('smoother_clamp',
                                      pass_name='velocity',
                                      t_ms=kf['time_ms'],
                                      original_x=kf['x'], clamped_x=new_x,
                                      prev_x=prev['x'],
                                      velocity_px_per_s=round(vel, 1),
                                      max_velocity_px_per_s=self.max_vel,
                                      reason='velocity_exceeds_max')
                    kf['x'] = new_x
                    clamped_count += 1

            smoothed.append(kf)

        # Pass 2: Remove jitter — if A→B→A pattern within 1 second, remove B
        # This is the most common cause of Stability=0
        dejittered = [smoothed[0]]
        jitter_count = 0
        centering_protected = 0
        for i in range(1, len(smoothed)):
            if i >= 2:
                prev2 = dejittered[-2] if len(dejittered) >= 2 else None
                prev1 = dejittered[-1]
                curr = smoothed[i]

                if prev2 is not None:
                    # Check for A→B→A pattern (oscillation)
                    dt_total = (curr['time_ms'] - prev2['time_ms']) / 1000.0
                    dx_back = abs(curr['x'] - prev2['x'])
                    dx_out = abs(prev1['x'] - prev2['x'])

                    # If we bounced back to near where we were within 2.0s
                    if (dt_total < 3.0 and dx_back < plan.crop_w * 0.20
                            and dx_out > plan.crop_w * 0.04):
                        # ``_centering`` keyframes are the engine's
                        # face-centering corrections; the merge, pan-
                        # consolidation and drift-suppress passes already
                        # honor the flag, but dejitter used to wipe them
                        # whenever they sat between two anchor frames at
                        # similar x. That's exactly the case the engine's
                        # centering nudge produces (anchor → centering
                        # nudge → next anchor), so dejitter was eating
                        # most of the work and leaving the metric at
                        # 70.5%. Keep the flagged keyframe; it's a real
                        # correction, not random jitter.
                        if prev1.get('_centering'):
                            centering_protected += 1
                            self.tracer.event(
                                'smoother_keep_centering',
                                pass_name='dejitter',
                                t_ms=prev1['time_ms'],
                                x=prev1['x'],
                                prev_x=prev2['x'],
                                next_x=curr['x'],
                                dt_total_s=round(dt_total, 3),
                                dx_back=dx_back, dx_out=dx_out,
                                reason='centering_kf_protected',
                            )
                        else:
                            # Remove the middle keyframe (the bounce)
                            popped = dejittered[-1]
                            self.tracer.event('smoother_drop',
                                              pass_name='dejitter',
                                              t_ms=popped['time_ms'],
                                              x=popped['x'],
                                              prev_x=prev2['x'],
                                              next_x=curr['x'],
                                              dt_total_s=round(dt_total, 3),
                                              dx_back=dx_back, dx_out=dx_out,
                                              reason='A_B_A_oscillation')
                            dejittered.pop()
                            jitter_count += 1

            dejittered.append(smoothed[i])

        # Pass 2.5: Hold enforcement — after a cut, suppress movement for 800ms.
        # This prevents the "snap then immediately drift" pattern that looks jumpy.
        held = [dejittered[0]]
        hold_enforced = 0
        for i in range(1, len(dejittered)):
            kf = dejittered[i]
            # Find the most recent cut before this keyframe
            last_cut_t = None
            for j in range(len(held) - 1, -1, -1):
                if held[j].get('transition') == 'cut':
                    last_cut_t = held[j]['time_ms']
                    break

            if (last_cut_t is not None
                    and kf.get('transition') != 'cut'
                    and kf['time_ms'] - last_cut_t < 1200):
                # Within hold period after cut — suppress this movement,
                # UNLESS this is a centering correction. The merge,
                # consolidation and drift-suppress passes already honor
                # _centering; hold_enforce was the last gap that was
                # silently eating centering nudges placed in the 800ms
                # window after a cut (a frequent need when the speaker
                # appears off-centre on the post-cut frame).
                if kf.get('_centering'):
                    self.tracer.event(
                        'smoother_keep_centering',
                        pass_name='hold_enforce',
                        t_ms=kf['time_ms'], x=kf['x'],
                        last_cut_t_ms=last_cut_t,
                        delta_from_cut_ms=kf['time_ms'] - last_cut_t,
                        reason='centering_kf_protected_post_cut',
                    )
                    held.append(kf)
                    continue
                self.tracer.event('smoother_drop',
                                  pass_name='hold_enforce',
                                  t_ms=kf['time_ms'], x=kf['x'],
                                  last_cut_t_ms=last_cut_t,
                                  delta_from_cut_ms=kf['time_ms'] - last_cut_t,
                                  reason='within_800ms_post_cut')
                hold_enforced += 1
                continue

            held.append(kf)

        # Pass 3: Merge keyframes that are very close in both time and position
        merged = [held[0]]
        merge_count = 0

        for i in range(1, len(held)):
            kf = held[i]
            prev = merged[-1]
            dt = (kf['time_ms'] - prev['time_ms']) / 1000.0
            dx = abs(kf['x'] - prev['x'])

            # If less than 500ms apart and less than 50px movement, merge —
            # UNLESS one of them is a centering correction (added by
            # ``_stabilize_keyframes`` Pass 4). Centering nudges are small
            # by definition (≤60 px) so they'd always trip this rule and
            # disappear, which is the regression that left the metric at
            # 68 %. Faces near the edge of the crop need that nudge to
            # survive into the rendered output.
            if (dt < 0.5 and dx < 50
                    and not kf.get('_centering')
                    and not prev.get('_centering')):
                self.tracer.event('smoother_drop',
                                  pass_name='merge',
                                  t_ms=kf['time_ms'], x=kf['x'],
                                  prev_t_ms=prev['time_ms'], prev_x=prev['x'],
                                  dt_s=round(dt, 3), dx=dx,
                                  reason='close_in_time_and_position')
                merge_count += 1
                continue

            merged.append(kf)

        # Pass 4: Pan consolidation — merge consecutive same-direction
        # moves within 1.5s into a single smooth arc.
        #
        # Before: A→B(+20) → C(+15) → D(+25) in 1.2s = 3 stuttery pans
        # After:  A → D(+60) in 1.2s = 1 smooth cinematic pan
        #
        # Only consolidates ease_in_out transitions (not cuts — those are
        # intentional speaker switches).
        consolidated = [merged[0]]
        consol_count = 0
        i = 1
        while i < len(merged):
            kf = merged[i]
            prev = consolidated[-1]

            # Only consolidate eased moves (cuts are intentional).
            # Centering corrections are also off-limits — collapsing a
            # chain of small per-face centering nudges into a single
            # move to the LAST position loses the intermediate
            # centering on every face except the last one, which is
            # one of the biggest contributors to the 68 % cap.
            if kf.get('transition') != 'ease_in_out' or kf.get('_centering'):
                consolidated.append(kf)
                i += 1
                continue

            # Look ahead for consecutive same-direction eased moves
            direction = 1 if kf['x'] > prev['x'] else -1 if kf['x'] < prev['x'] else 0
            if direction == 0:
                consolidated.append(kf)
                i += 1
                continue

            # Collect run of same-direction eased moves within the window —
            # stop at any centering-tagged keyframe so it stays in place.
            # 2.5s (configurable) — the code shipped at 1.5s while its own
            # comment documented 2.5s, and the measured path still stuttered
            # (jerk 4693, stability 74 on the 24-min eval): three same-direction
            # steps 1.6-2.4s apart survived as separate mini-pans. The wider
            # window folds those into one cinematic move; centering keyframes
            # still break the run so face-coverage nudges stay in place.
            _consol_window_s = float(getattr(
                settings, "REFRAMER_PAN_CONSOLIDATE_WINDOW_S", 2.5))
            run_end = i
            for j in range(i + 1, len(merged)):
                nxt = merged[j]
                dt_total = (nxt['time_ms'] - prev['time_ms']) / 1000.0
                if dt_total > _consol_window_s:
                    break
                if nxt.get('transition') != 'ease_in_out':
                    break
                if nxt.get('_centering'):
                    break
                nxt_dir = 1 if nxt['x'] > merged[j-1]['x'] else -1 if nxt['x'] < merged[j-1]['x'] else 0
                if nxt_dir != direction:
                    break
                run_end = j

            if run_end > i:
                # Consolidate: skip intermediate keyframes, go directly
                # to the final position with distance-scaled duration
                final = merged[run_end]
                total_dist = abs(final['x'] - prev['x'])
                dist_ratio = total_dist / max(1, plan.crop_w)
                # Linear interpolation between 200ms (tiny nudge) and 600ms (full-width pan).
                # Capped at 600ms — anything slower feels like lag, not cinema.
                trans_ms = int(200 + min(400, dist_ratio * 400 / 0.50))

                # Centering corrections should ease_out (settle gently onto the face).
                # Tracking pans should ease_in (start slow as the eye follows).
                # All other consolidated pans stay ease_in_out.
                _transition_type = 'ease_in_out'
                if final.get('_centering'):
                    _transition_type = 'ease_out'
                elif all(merged[j].get('source') == 'face_track' for j in range(i, run_end + 1)):
                    _transition_type = 'ease_in'
                consolidated.append({
                    'time_ms': final['time_ms'],
                    'x': final['x'],
                    'transition': _transition_type,
                    'transition_ms': trans_ms,
                })
                # Emit one drop event per intermediate keyframe absorbed
                # into the consolidated pan. Distinct events so the trace
                # can be filtered exactly like the other passes.
                dropped_count = run_end - i
                for j in range(i, run_end):
                    dk = merged[j]
                    self.tracer.event('smoother_drop',
                                      pass_name='consolidate',
                                      t_ms=dk['time_ms'], x=dk['x'],
                                      consolidated_into_t_ms=final['time_ms'],
                                      consolidated_x=final['x'],
                                      run_length=dropped_count,
                                      reason='intermediate_pan_step')
                consol_count += dropped_count
                i = run_end + 1
            else:
                consolidated.append(kf)
                i += 1

        # Pass 4.7: Conversation cadence — collapse ping-pong.
        #
        # A two-person dialogue drives the speaker-follow into strict
        # alternation: A→B→A→B every second or so (a measured 39-second
        # classroom scene carried 38 keyframes). Dejitter only removes a
        # single quick A→B→A bounce; a sustained rally survives every pass
        # and reads as a tennis-match camera. No human operator does this —
        # for a close two-shot they frame BOTH speakers and hold; for a wide
        # separation they still hold each side for a couple of seconds
        # instead of chasing every line.
        _pp_hold_s = float(getattr(settings, "REFRAMER_PINGPONG_MIN_HOLD_S", 2.5))
        _pp_twoshot = float(getattr(settings, "REFRAMER_PINGPONG_TWOSHOT_PCT", 0.45))
        cadenced = [consolidated[0]]
        pingpong_collapsed = 0
        i = 1
        while i < len(consolidated):
            kf = consolidated[i]
            if kf.get('transition') == 'cut' or kf.get('_centering'):
                cadenced.append(kf)
                i += 1
                continue
            # Measure the rally window starting here: consecutive eased moves
            # each landing sooner than the hold. Membership on a side is
            # decided by CLUSTERING afterwards, not by per-step direction —
            # the strict flip-every-step test detected NOTHING on a measured
            # two-speaker episode (827 keyframes, 0 collapses) because a real
            # speaker-follow rally contains small same-side corrections
            # (A→B→B′→A) that broke the run at step two every time.
            run = [i]
            j = i + 1
            while j < len(consolidated):
                nkf = consolidated[j]
                if nkf.get('transition') == 'cut' or nkf.get('_centering'):
                    break
                dt = (nkf['time_ms'] - consolidated[j - 1]['time_ms']) / 1000.0
                if dt >= _pp_hold_s:
                    break
                run.append(j)
                j += 1
            if len(run) >= 4:
                xs = [consolidated[k]['x'] for k in run]
                # Two sides = the clusters left/right of the window's midline.
                _mid_x = (max(xs) + min(xs)) / 2.0
                _sides = [0 if x <= _mid_x else 1 for x in xs]
                side_a = [x for x, s in zip(xs, _sides) if s == 0]
                side_b = [x for x, s in zip(xs, _sides) if s == 1]
                # A rally must actually ALTERNATE between the clusters — a
                # slow monotonic drift also lands inside the window but only
                # crosses the midline once, and must not be collapsed.
                _swings = sum(1 for a, b in zip(_sides, _sides[1:]) if a != b)
                spread_ok = (bool(side_a) and bool(side_b) and _swings >= 3
                             and max(side_a) - min(side_a) <= plan.crop_w * 0.15
                             and max(side_b) - min(side_b) <= plan.crop_w * 0.15)
                sep = (abs(sum(side_a) / len(side_a) - sum(side_b) / len(side_b))
                       if side_a and side_b else 0.0)
                if spread_ok and sep <= plan.crop_w * _pp_twoshot:
                    # Close enough to frame both: ONE move to the midpoint,
                    # then hold for the whole rally.
                    mid = clamp_x(int(round(sum(xs) / len(xs))), plan.max_x)
                    first = consolidated[run[0]]
                    cadenced.append({
                        'time_ms': first['time_ms'], 'x': mid,
                        'transition': 'ease_in_out', 'transition_ms': 450,
                    })
                    for k in run:
                        dk = consolidated[k]
                        self.tracer.event('smoother_drop',
                                          pass_name='pingpong',
                                          t_ms=dk['time_ms'], x=dk['x'],
                                          twoshot_x=mid, run_length=len(run),
                                          reason='pingpong_two_shot_hold')
                    pingpong_collapsed += len(run)
                    i = run[-1] + 1
                    continue
                if spread_ok:
                    # Too far apart for a two-shot: keep the rally but slow
                    # it to operator cadence — each side holds ≥ the minimum
                    # before the camera swings back.
                    last_kept_t = cadenced[-1]['time_ms']
                    for k in run:
                        dk = consolidated[k]
                        if dk['time_ms'] - last_kept_t >= _pp_hold_s * 1000.0:
                            cadenced.append(dk)
                            last_kept_t = dk['time_ms']
                        else:
                            self.tracer.event('smoother_drop',
                                              pass_name='pingpong',
                                              t_ms=dk['time_ms'], x=dk['x'],
                                              run_length=len(run),
                                              reason='pingpong_hold_not_met')
                            pingpong_collapsed += 1
                    i = run[-1] + 1
                    continue
            cadenced.append(kf)
            i += 1

        # Pass 4.8: Anticipation — begin each eased move slightly BEFORE the
        # detection that motivated it. Perception lags the event (a face is
        # found some frames after it appears; a speaker change lands after the
        # first word), so an unshifted pan always arrives late and reads as
        # chasing. A human operator leads the action. Bounded by the previous
        # keyframe so ordering is preserved.
        _lead_ms = int(getattr(settings, "REFRAMER_ANTICIPATE_MS", 180))
        anticipated = 0
        if _lead_ms > 0:
            for idx in range(1, len(cadenced)):
                kf = cadenced[idx]
                if kf.get('transition') == 'cut':
                    continue
                floor_t = cadenced[idx - 1]['time_ms'] + 50
                new_t = max(floor_t, kf['time_ms'] - _lead_ms)
                if new_t < kf['time_ms']:
                    kf['time_ms'] = new_t
                    anticipated += 1
        consolidated = cadenced

        # Pass 5: Micro-drift suppression — remove eased moves smaller than
        # the drift threshold. These are too small for the viewer to notice
        # but they add perceived jitter and hurt the stability score. A
        # human editor would never make a 30-pixel adjustment — they'd
        # either hold or make a real move.
        #
        # The 0.12 multiplier was too aggressive on small crops
        # (202 px → 24 px threshold), which is the same band the
        # centering nudge produces. The log showed 290 drift-suppressed
        # moves on a video that only got 24 centering nudges into the
        # keyframe list — most of those nudges died here. Tightened to
        # 0.08 + 15 px floor so small centering corrections survive
        # without re-introducing the original jitter problem. Centering-
        # tagged keyframes are skipped unconditionally so the eval's
        # middle-third check still passes on faces near the edge.
        # 8% of crop width with a 15px floor — the values this comment block
        # documents (the code shipped at 0.05/10px, so ~5-8% wobble survived
        # and read as jitter: stability 74, hold_ratio 80.8% on the 24-min
        # eval). Centering-tagged keyframes bypass this pass entirely, so the
        # face-coverage machinery is untouched. Configurable for tuning
        # against reframe_debug.json without a rebuild.
        _drift_pct = float(getattr(settings, "REFRAMER_DRIFT_SUPPRESS_PCT", 0.08))
        drift_threshold = max(15, int(plan.crop_w * _drift_pct))
        stabilized = [consolidated[0]]
        drift_suppressed = 0
        for i in range(1, len(consolidated)):
            kf = consolidated[i]
            prev = stabilized[-1]
            # Skip drift suppression for centering corrections.
            if kf.get('_centering'):
                stabilized.append(kf)
                continue
            # Only suppress eased moves (cuts are intentional)
            if kf.get('transition') == 'ease_in_out':
                dx = abs(kf['x'] - prev['x'])
                if dx < drift_threshold:
                    self.tracer.event('smoother_drop',
                                      pass_name='drift_suppress',
                                      t_ms=kf['time_ms'], x=kf['x'],
                                      prev_x=prev['x'], dx=dx,
                                      drift_threshold_px=drift_threshold,
                                      reason='movement_below_drift_threshold')
                    drift_suppressed += 1
                    continue
            stabilized.append(kf)

        plan.keyframes = stabilized
        log.log_stage('SMOOTH',
            f'Smoothed {len(kfs)} → {len(stabilized)} keyframes '
            f'(clamped {clamped_count} vel, removed {removed_count} dupes, '
            f'dejittered {jitter_count}, '
            f'centering-protected {centering_protected}, '
            f'merged {merge_count}, '
            f'consolidated {consol_count} pans, hold-enforced {hold_enforced}, '
            f'ping-pong-collapsed {pingpong_collapsed}, '
            f'anticipated {anticipated}, '
            f'drift-suppressed {drift_suppressed})')
        self.tracer.event('smoother_complete',
                          input_keyframes=len(kfs),
                          output_keyframes=len(stabilized),
                          velocity_clamped=clamped_count,
                          temporal_dupes_removed=removed_count,
                          dejittered=jitter_count,
                          centering_protected=centering_protected,
                          merged=merge_count,
                          consolidated=consol_count,
                          pingpong_collapsed=pingpong_collapsed,
                          anticipated=anticipated,
                          hold_enforced=hold_enforced,
                          drift_suppressed=drift_suppressed)
        return plan


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 5 — RENDER  (FFmpeg pipe — guaranteed parity with preview)
# ═══════════════════════════════════════════════════════════════════════════

