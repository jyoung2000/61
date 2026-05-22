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
)

logger = logging.getLogger("clipai.reframer_smoother")


class Smoother:
    """Post-process keyframes for smooth, natural-feeling output."""

    def __init__(self, max_vel_px_per_sec: float = 2000):
        self.max_vel = max_vel_px_per_sec

    def smooth(self, plan: RenderPlan) -> RenderPlan:
        log = get_logger()
        kfs = plan.keyframes
        if len(kfs) < 2:
            log.log_stage('SMOOTH', f'Skipped (only {len(kfs)} keyframe)')
            return plan

        # Pass 1: Remove temporal duplicates and clamp velocity
        smoothed = [kfs[0]]
        clamped_count = 0
        removed_count = 0
        for i in range(1, len(kfs)):
            kf = dict(kfs[i])
            prev = smoothed[-1]
            dt = (kf['time_ms'] - prev['time_ms']) / 1000.0
            if dt <= 0.001:
                removed_count += 1
                continue

            if kf['transition'] != 'cut':
                dx = abs(kf['x'] - prev['x'])
                vel = dx / dt
                if vel > self.max_vel:
                    max_dx = int(self.max_vel * dt)
                    sign = 1 if kf['x'] > prev['x'] else -1
                    kf['x'] = clamp_x(prev['x'] + sign * max_dx, plan.max_x)
                    clamped_count += 1

            smoothed.append(kf)

        # Pass 2: Remove jitter — if A→B→A pattern within 1 second, remove B
        # This is the most common cause of Stability=0
        dejittered = [smoothed[0]]
        jitter_count = 0
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
                        # Remove the middle keyframe (the bounce)
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

            # If less than 500ms apart and less than 50px movement, merge
            if dt < 0.5 and dx < 50:
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

            # Only consolidate eased moves (cuts are intentional)
            if kf.get('transition') != 'ease_in_out':
                consolidated.append(kf)
                i += 1
                continue

            # Look ahead for consecutive same-direction eased moves
            direction = 1 if kf['x'] > prev['x'] else -1 if kf['x'] < prev['x'] else 0
            if direction == 0:
                consolidated.append(kf)
                i += 1
                continue

            # Collect run of same-direction eased moves within 2.5s
            run_end = i
            for j in range(i + 1, len(merged)):
                nxt = merged[j]
                dt_total = (nxt['time_ms'] - prev['time_ms']) / 1000.0
                if dt_total > 2.5:
                    break
                if nxt.get('transition') != 'ease_in_out':
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
                consol_count += (run_end - i)
                i = run_end + 1
            else:
                consolidated.append(kf)
                i += 1

        # Pass 5: Micro-drift suppression — remove eased moves smaller than
        # 5% of crop width. These are too small for the viewer to notice but
        # they add perceived jitter and hurt the stability score. A human editor
        # would never make a 30-pixel adjustment — they'd either hold or
        # make a real move.
        drift_threshold = max(20, int(plan.crop_w * 0.12))
        stabilized = [consolidated[0]]
        drift_suppressed = 0
        for i in range(1, len(consolidated)):
            kf = consolidated[i]
            prev = stabilized[-1]
            # Only suppress eased moves (cuts are intentional)
            if kf.get('transition') == 'ease_in_out':
                dx = abs(kf['x'] - prev['x'])
                if dx < drift_threshold:
                    drift_suppressed += 1
                    continue
            stabilized.append(kf)

        plan.keyframes = stabilized
        log.log_stage('SMOOTH',
            f'Smoothed {len(kfs)} → {len(stabilized)} keyframes '
            f'(clamped {clamped_count} vel, removed {removed_count} dupes, '
            f'dejittered {jitter_count}, merged {merge_count}, '
            f'consolidated {consol_count} pans, hold-enforced {hold_enforced}, '
            f'drift-suppressed {drift_suppressed})')
        return plan


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 5 — RENDER  (FFmpeg pipe — guaranteed parity with preview)
# ═══════════════════════════════════════════════════════════════════════════

