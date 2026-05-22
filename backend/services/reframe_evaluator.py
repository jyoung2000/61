"""Reframe quality evaluator — grades a finished reframer RenderPlan.

Ported from clipai_reframer.py (jyoung2000/56). Once the reframer has
produced a keyframed crop plan, this audits it second-by-second against
the perception's face timeline and emits a :class:`ReframeReport`: an
A-F grade, a 0-100 overall score, per-axis sub-scores (face coverage,
saliency focus, centering, stability, cut coherence, edge violations,
watchability) and a list of flagged problems.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

import numpy as np

from backend.services.reframer_models import (
    PerceptionResult, RenderPlan, clamp_x, interpolate_x,
)

logger = logging.getLogger("clipai.reframe_evaluator")


@dataclass
class ReframeReport:
    """Structured quality report from the evaluator."""
    overall_score: float = 0.0           # 0-100
    grade: str = 'F'                     # A/B/C/D/F
    face_coverage_pct: float = 0.0       # % of face-present seconds with face in crop
    saliency_accuracy_pct: float = 0.0   # % of seconds with BEST face in crop
    centering_pct: float = 0.0           # % of seconds with best face in center third
    stability_score: float = 0.0         # 0-100 (100 = no jitter)
    cut_coherence_score: float = 0.0     # 0-100
    edge_violation_pct: float = 0.0      # % of seconds with face cut at edge

    # Watchability — measures how "human-editor-like" the reframe feels
    watchability_score: float = 0.0      # 0-100 composite
    hold_quality: float = 0.0            # 0-100 (do targets hold long enough?)
    transition_decisiveness: float = 0.0  # 0-100 (are moves committed, not drifty?)
    motion_budget: float = 0.0           # 0-100 (total movement reasonable?)

    # Per-second detail (not persisted on the job — used to derive scores)
    second_scores: List[dict] = field(default_factory=list)

    # Flagged problems (timestamps + descriptions)
    problems: List[dict] = field(default_factory=list)

    # Summary stats
    total_seconds: int = 0
    seconds_with_faces: int = 0
    seconds_face_in_crop: int = 0
    seconds_best_face_in_crop: int = 0
    seconds_face_centered: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ReframeEvaluator:
    """Automated quality auditor for a reframed video."""

    def __init__(self, plan: RenderPlan, perception: PerceptionResult):
        self.plan = plan
        self.perc = perception

    def run(self, on_progress: Optional[Callable] = None) -> ReframeReport:
        report = ReframeReport()
        duration_ms = (
            getattr(self.plan, "duration_ms", 0)
            or getattr(self.perc, "duration_ms", 0)
            or 0
        )
        report.total_seconds = int(duration_ms / 1000.0)
        if report.total_seconds == 0:
            return report

        crop_w = max(1, int(getattr(self.plan, "crop_w", 0) or 1))
        max_x = self.plan.max_x
        keyframes = getattr(self.plan, "keyframes", None) or []
        face_timeline = getattr(self.perc, "face_timeline", None) or {}

        prev_x = None
        x_deltas: List[float] = []
        cut_intervals: List[int] = []
        last_cut_sec = 0

        # ── Evaluate every second ──
        for sec in range(report.total_seconds):
            time_ms = sec * 1000
            crop_x = clamp_x(interpolate_x(keyframes, time_ms), max_x)

            nearest_t = self._nearest_sample(time_ms, face_timeline)
            faces = face_timeline.get(nearest_t, []) if nearest_t is not None else []
            faces = [f for f in faces if isinstance(f, dict) and "cx" in f]

            has_faces = len(faces) > 0
            face_in_crop = False
            best_face_in_crop = False
            edge_violation = False
            face_centered = False
            best_face = None

            if has_faces:
                report.seconds_with_faces += 1
                crop_left = crop_x
                crop_right = crop_x + crop_w

                # Any face center inside the crop?
                for f in faces:
                    if crop_left <= f["cx"] <= crop_right:
                        face_in_crop = True
                        fx = f.get("x", f["cx"])
                        if fx < crop_left or fx + f.get("w", 0) > crop_right:
                            edge_violation = True
                        break
                if face_in_crop:
                    report.seconds_face_in_crop += 1

                # Is the BEST (most salient) face in the crop? Weight by
                # confidence so low-confidence false positives don't mislead.
                best_face = max(faces, key=lambda f: (
                    f.get("saliency", 0) + f.get("mouth_motion", 0)
                ) * max(0.2, f.get("confidence", 0.5)))
                if crop_left <= best_face["cx"] <= crop_right:
                    best_face_in_crop = True
                    report.seconds_best_face_in_crop += 1
                    center_left = crop_left + crop_w // 3
                    center_right = crop_left + (2 * crop_w) // 3
                    face_centered = center_left <= best_face["cx"] <= center_right
                    if face_centered:
                        report.seconds_face_centered += 1

            # Stability: track crop-position delta + detect cuts (large jumps).
            if prev_x is not None:
                delta = abs(crop_x - prev_x)
                x_deltas.append(delta)
                if delta > crop_w * 0.3:
                    cut_intervals.append(sec - last_cut_sec)
                    last_cut_sec = sec
            prev_x = crop_x

            report.second_scores.append({
                "time_sec": sec,
                "crop_x": crop_x,
                "n_faces": len(faces),
                "face_in_crop": face_in_crop,
                "best_face_in_crop": best_face_in_crop,
                "edge_violation": edge_violation,
            })

            # Flag problems
            if has_faces and not face_in_crop:
                report.problems.append({
                    "time_sec": sec, "severity": "HIGH", "type": "face_missing",
                    "message": "Face detected but not in the crop window",
                })
            elif has_faces and face_in_crop and not best_face_in_crop:
                report.problems.append({
                    "time_sec": sec, "severity": "MED", "type": "wrong_face",
                    "message": "A less salient face is in the crop; the best face is elsewhere",
                })
            if edge_violation:
                report.problems.append({
                    "time_sec": sec, "severity": "LOW", "type": "edge_cut",
                    "message": "Face partially cut off at the crop edge",
                })
            if has_faces and best_face_in_crop and not face_centered:
                report.problems.append({
                    "time_sec": sec, "severity": "MED", "type": "off_center",
                    "message": "Best face is off-center within the crop",
                })

            if on_progress and sec % 10 == 0:
                on_progress(sec / report.total_seconds)

        # ── Compute scores ──

        # Face coverage / saliency / centering
        if report.seconds_with_faces > 0:
            report.face_coverage_pct = (
                report.seconds_face_in_crop / report.seconds_with_faces * 100)
            report.saliency_accuracy_pct = (
                report.seconds_best_face_in_crop / report.seconds_with_faces * 100)
            report.centering_pct = (
                report.seconds_face_centered / report.seconds_with_faces * 100)
        else:
            report.face_coverage_pct = 100.0
            report.saliency_accuracy_pct = 100.0
            report.centering_pct = 100.0

        # Stability — jitter, excluding intentional speaker-switch cuts.
        if x_deltas:
            jitter_deltas = [d for d in x_deltas if d < crop_w * 0.3]
            if jitter_deltas:
                avg_jitter = float(np.mean(jitter_deltas))
                max_reasonable_jitter = crop_w * 0.08  # 8% of crop width / sec
                report.stability_score = max(0.0, min(100.0,
                    100 - (avg_jitter / max(1.0, max_reasonable_jitter)) * 50))
            else:
                report.stability_score = 100.0
        else:
            report.stability_score = 100.0

        # Cut coherence — penalize rapid (<1s apart) cuts.
        if cut_intervals:
            rapid_cuts = sum(1 for i in cut_intervals if i < 1)
            report.cut_coherence_score = max(0.0, min(100.0, 100 - rapid_cuts * 15))
        else:
            report.cut_coherence_score = 100.0

        # Edge violations
        edge_violations = sum(1 for s in report.second_scores if s["edge_violation"])
        report.edge_violation_pct = edge_violations / max(1, report.total_seconds) * 100

        # ── Watchability — does this feel like a human editor made it? ──

        still_thresh = crop_w * 0.05  # 5% of crop width = essentially still

        # Hold quality: how long the crop stays settled on each target.
        if x_deltas:
            hold_lengths: List[int] = []
            current_hold = 0
            for d in x_deltas:
                if d < still_thresh:
                    current_hold += 1
                else:
                    if current_hold > 0:
                        hold_lengths.append(current_hold)
                    current_hold = 0
            if current_hold > 0:
                hold_lengths.append(current_hold)
            if hold_lengths:
                avg_hold = float(np.mean(hold_lengths))
                report.hold_quality = min(100.0, max(0.0, avg_hold * 33))
            else:
                report.hold_quality = 0.0
        else:
            report.hold_quality = 100.0

        # Transition decisiveness: moves should be quick and committed.
        if x_deltas:
            still_count = sum(1 for d in x_deltas if d < still_thresh)
            decisive_count = sum(1 for d in x_deltas if d >= crop_w * 0.10)
            total = len(x_deltas)
            committed_pct = (still_count + decisive_count) / max(1, total)
            report.transition_decisiveness = min(100.0, committed_pct * 100)
        else:
            report.transition_decisiveness = 100.0

        # Motion budget: total drift movement (cuts excluded) per minute.
        if x_deltas and report.total_seconds > 0:
            drift_deltas = [d for d in x_deltas if d < crop_w * 0.20]
            drift_movement = sum(drift_deltas)
            drift_per_min = drift_movement / max(1.0, report.total_seconds / 60)
            budget_ratio = drift_per_min / 4000
            report.motion_budget = max(0.0, min(100.0, (1 - budget_ratio) * 100))
        else:
            report.motion_budget = 100.0

        report.watchability_score = (
            report.hold_quality * 0.35
            + report.transition_decisiveness * 0.35
            + report.motion_budget * 0.30
        )

        # ── Overall score + letter grade ──
        report.overall_score = (
            report.face_coverage_pct * 0.25
            + report.saliency_accuracy_pct * 0.20
            + report.centering_pct * 0.10
            + report.stability_score * 0.10
            + report.cut_coherence_score * 0.10
            + report.watchability_score * 0.20
            + (100 - report.edge_violation_pct) * 0.05
        )

        if report.overall_score >= 90:
            report.grade = 'A'
        elif report.overall_score >= 75:
            report.grade = 'B'
        elif report.overall_score >= 60:
            report.grade = 'C'
        elif report.overall_score >= 40:
            report.grade = 'D'
        else:
            report.grade = 'F'

        if on_progress:
            on_progress(1.0)
        return report

    @staticmethod
    def _nearest_sample(time_ms: int, face_timeline: dict) -> Optional[int]:
        """Nearest perception sample within 1s of ``time_ms``."""
        best_t = None
        best_d = float('inf')
        for t in face_timeline:
            d = abs(t - time_ms)
            if d < best_d:
                best_d = d
                best_t = t
        return best_t if best_d < 1000 else None
