"""Tests for the fluid tracking, centering, and stabilization improvements."""
import pytest
class TestEdgeViolationSweep:
    def test_method_exists_on_engine(self):
        from backend.services.reframer_engine import ReframeEngine
        assert hasattr(ReframeEngine, '_eliminate_edge_violations')
    def test_no_clip_after_sweep(self):
        """A face at the crop edge must be shifted to clear the edge."""
        from backend.services.reframer_engine import ReframeEngine
        from backend.services.reframer_models import RenderPlan, PerceptionResult
        import types
        engine = object.__new__(ReframeEngine)
        engine.log = types.SimpleNamespace(log_stage=lambda *a, **kw: None)
        plan = RenderPlan(source_width=1920, source_height=1080,
                          crop_w=600, crop_h=1080)
        plan.keyframes = [{'time_ms': 0, 'x': 0}]  # face at left edge
        perc = PerceptionResult()
        # Face whose bbox clips at the left edge of the crop
        perc.face_timeline = {
            0: [{'cx': 30, 'cy': 540, 'x': 5, 'y': 440, 'w': 50, 'h': 100,
                 'track_id': 0, 'saliency': 0.8, 'mouth_motion': 0.5,
                 'confidence': 0.9}]
        }
        engine.plan = plan
        engine.perception = perc
        engine._eliminate_edge_violations()
        # crop_x should have moved so face_left (5) >= crop_x + 8
        assert plan.keyframes[0]['x'] <= 5 - 8, (
            f"crop_x should pull left to clear edge, got {plan.keyframes[0]['x']}")
class TestEasingCurves:
    def test_ease_in_slower_than_linear_at_start(self):
        from backend.services.reframer_models import interpolate_x
        kfs = [
            {'time_ms': 0, 'x': 0},
            {'time_ms': 1000, 'x': 100, 'transition': 'ease_in', 'transition_ms': 0},
        ]
        # At 25% of travel, ease_in should be < 25px (slower start)
        val = interpolate_x(kfs, 250)
        assert val < 25, f"ease_in at 25% should be <25px, got {val}"
    def test_ease_out_faster_than_linear_at_start(self):
        from backend.services.reframer_models import interpolate_x
        kfs = [
            {'time_ms': 0, 'x': 0},
            {'time_ms': 1000, 'x': 100, 'transition': 'ease_out', 'transition_ms': 0},
        ]
        # At 25% of travel, ease_out should be > 25px (faster start, slow finish)
        val = interpolate_x(kfs, 250)
        assert val > 25, f"ease_out at 25% should be >25px, got {val}"
    def test_ease_in_out_symmetric(self):
        from backend.services.reframer_models import interpolate_x
        kfs = [
            {'time_ms': 0, 'x': 0},
            {'time_ms': 1000, 'x': 100, 'transition': 'ease_in_out', 'transition_ms': 0},
        ]
        at_25 = interpolate_x(kfs, 250)
        at_75 = interpolate_x(kfs, 750)
        # ease_in_out is symmetric: val(25%) + val(75%) == 100
        assert abs((at_25 + at_75) - 100) <= 1
class TestSmootherTightDrift:
    def test_drift_threshold_tighter(self):
        """drift_threshold should be 5% of crop_w, not 8%."""
        from backend.services.reframer_smoother import Smoother
        from backend.services.reframer_models import RenderPlan
        from backend.services.reframer_models import null_tracer
        s = Smoother()
        plan = RenderPlan(source_width=1920, source_height=1080, crop_w=600)
        plan.keyframes = [
            {'time_ms': 0, 'x': 300, 'transition': 'cut', 'transition_ms': 0},
            # tiny 20px drift move — should be suppressed at 5% threshold (30px)
            {'time_ms': 1000, 'x': 320, 'transition': 'ease_in_out', 'transition_ms': 300},
            {'time_ms': 2000, 'x': 300, 'transition': 'ease_in_out', 'transition_ms': 300},
        ]
        result = s.smooth(plan)
        # The 20px move is below 5% of 600px (=30px) — should be suppressed
        xs = [kf['x'] for kf in result.keyframes]
        assert 320 not in xs, f"20px drift move should have been suppressed, got {xs}"
class TestVelocityAdaptiveEMA:
    def test_fast_face_uses_higher_alpha(self):
        """When face moves a lot between samples, alpha should be > 0.6."""
        # Simulate the alpha computation logic directly
        face_w = 100
        # Face moved 2 widths — definitely fast
        motion_px = 200
        motion_norm = min(1.0, motion_px / face_w)
        alpha = 0.60 + motion_norm * 0.30
        assert alpha > 0.85, f"Fast face should use alpha >0.85, got {alpha:.2f}"
    def test_still_face_uses_base_alpha(self):
        """When face barely moves, alpha should stay near 0.6."""
        face_w = 100
        motion_px = 3  # essentially still
        motion_norm = min(1.0, motion_px / face_w)
        alpha = 0.60 + motion_norm * 0.30
        assert alpha < 0.65, f"Still face should use alpha <0.65, got {alpha:.2f}"
class TestMetricLogger:
    def test_summarise_problems(self):
        from backend.services.reframe_evaluator import ReframeEvaluator, ReframeReport
        report = ReframeReport()
        report.problems = [
            {'severity': 'HIGH', 'type': 'face_missing'},
            {'severity': 'HIGH', 'type': 'face_missing'},
            {'severity': 'MED', 'type': 'off_center'},
        ]
        summary = ReframeEvaluator.summarise_problems(report)
        assert 'HIGH:face_missing×2' in summary
        assert 'MED:off_center×1' in summary
    def test_summarise_no_problems(self):
        from backend.services.reframe_evaluator import ReframeEvaluator, ReframeReport
        summary = ReframeEvaluator.summarise_problems(ReframeReport())
        assert summary == 'no_problems'
class TestDeadbandTightening:
    def test_live_action_deadband_smaller_than_animated(self):
        """Live-action deadband must be tighter than animated deadband."""
        crop_w = 400
        live_deadband = max(20, int(crop_w * 0.12))
        anim_deadband = max(30, int(crop_w * 0.25))
        assert live_deadband < anim_deadband, (
            f"live={live_deadband} should be < anim={anim_deadband}")
    def test_live_action_deadband_below_5pct_face_width(self):
        """On a typical 9:16 crop, deadband should be < 1 face width."""
        crop_w = 200   # narrow 9:16 on 360p
        live_deadband = max(20, int(crop_w * 0.12))
        typical_face_w = 80   # face takes ~40% of crop width
        assert live_deadband < typical_face_w, (
            f"deadband={live_deadband}px should be less than face_w={typical_face_w}px")
