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
                    if (dt_total < 2.0 and dx_back < plan.crop_w * 0.15
                            and dx_out > plan.crop_w * 0.05):
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
                    and kf['time_ms'] - last_cut_t < 800):
                # Within hold period after cut — suppress this movement
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

            # Collect run of same-direction eased moves within 2.5s —
            # stop at any centering-tagged keyframe so it stays in place.
            run_end = i
            for j in range(i + 1, len(merged)):
                nxt = merged[j]
                dt_total = (nxt['time_ms'] - prev['time_ms']) / 1000.0
                if dt_total > 2.5:
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
                if dist_ratio < 0.15:
                    trans_ms = 350
                elif dist_ratio < 0.30:
                    trans_ms = 550
                elif dist_ratio < 0.50:
                    trans_ms = 750
                else:
                    trans_ms = 900

                consolidated.append({
                    'time_ms': final['time_ms'],
                    'x': final['x'],
                    'transition': 'ease_in_out',
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
        drift_threshold = max(15, int(plan.crop_w * 0.08))
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
                          hold_enforced=hold_enforced,
                          drift_suppressed=drift_suppressed)
        return plan


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 5 — RENDER  (FFmpeg pipe — guaranteed parity with preview)
# ═══════════════════════════════════════════════════════════════════════════

