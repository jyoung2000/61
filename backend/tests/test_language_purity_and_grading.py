"""Language purity + coherence, user-accurate grading, reframing upgrades.

Covers the 2026-07-04 change set driven by the run-13 audit:
  A) transcript: verified purity loop (cloud escalation on echo), the
     translated-track polish on the LLM path, ASR-confidence marks in the
     polish prompt, and the wrong-script hallucination gate;
  B) grading: evaluator window slices (per-clip), p10 worst-moments score,
     the HIGH-density grade cap, per-clip aggregation, the VLM spot-check
     plumbing, and operator-correction telemetry;
  C) reframing: inter-sample LK tracking, problem-window merging for the
     repair pass, per-scene vertical eye-line through the bridge, and the
     corroborated live-action classifier.
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _name, _attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                     ("anthropic", "AsyncAnthropic")):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod
if "google.generativeai" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.generativeai")
    _gg.configure = lambda *a, **k: None
    _gg.GenerativeModel = object
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules["google.generativeai"] = _gg

import inspect

import pytest

from backend.config import settings


def _pipeline_src():
    from backend.services import pipeline
    return inspect.getsource(pipeline)


# ═══ A) Transcript language + coherence ══════════════════════════════════

def test_wrong_script_gate_flags_latin_prose_in_ja():
    from backend.services.reframer_audio import _wrong_script_for_language
    assert _wrong_script_for_language(
        "See you next time in the video.", "ja") is True
    assert _wrong_script_for_language(
        "Thank you for watching, please subscribe now", "japanese") is True


def test_wrong_script_gate_conservative():
    from backend.services.reframer_audio import _wrong_script_for_language
    # Short interjections never trip it.
    assert _wrong_script_for_language("OK", "ja") is False
    assert _wrong_script_for_language("Bye bye", "ja") is False
    # Mixed lines with real Japanese script are kept.
    assert _wrong_script_for_language(
        "それで one more time お願いします yeah", "ja") is False
    # Latin-script languages can't have a "wrong script".
    assert _wrong_script_for_language(
        "This is a normal English sentence here.", "en") is False
    assert _wrong_script_for_language(
        "This is a normal English sentence here.", "") is False


def test_purity_loop_escalates_echoes_and_reports():
    from backend.services import pipeline
    src = inspect.getsource(pipeline._llm_cleanup_untranslated)
    # Echo-failures (not just exceptions) must reach the cloud net…
    assert "_validate" in src
    assert "_cloud_polish_completion" in src
    # …and survivors become a VISIBLE job warning with timestamps.
    assert "_record_pipeline_warning" in src
    assert "Purity check" in src


def test_llm_translation_gets_post_edit_polish():
    src = _pipeline_src()
    assert "AI post-edit START on LLM-translated text" in src
    assert "TRANSLATION_POLISH_LLM_OUTPUT" in src
    assert settings.TRANSLATION_POLISH_LLM_OUTPUT is True


def test_polish_prompt_marks_low_confidence_lines():
    from backend.services.transcript_polisher import _build_user_prompt
    batch = [
        {"index": 0, "text": "Hand kimchi.", "avg_logprob": -1.4,
         "start": 0.0, "end": 2.0},
        {"index": 1, "text": "How old are you now?", "avg_logprob": -0.2,
         "start": 2.0, "end": 4.0},
    ]
    prompt = _build_user_prompt(batch, [], [], language="en")
    assert '"asr_confidence": "low"' in prompt
    assert "unreliable speech" in prompt
    # The reliable line is NOT marked (one mark in the segments block; the
    # instruction note mentions the key once more).
    seg_block = prompt.split("SEGMENTS TO POLISH", 1)[1]
    assert seg_block.count('"asr_confidence"') == 1


def test_polish_prompt_no_conf_note_when_all_reliable():
    from backend.services.transcript_polisher import _build_user_prompt
    batch = [{"index": 0, "text": "All good here.", "avg_logprob": -0.1,
              "start": 0.0, "end": 2.0}]
    prompt = _build_user_prompt(batch, [], [], language="en")
    assert "asr_confidence" not in prompt


def test_language_flags_defaults():
    assert settings.WHISPER_SCRIPT_FILTER is True
    assert settings.TRANSLATION_POLISH_LLM_OUTPUT is True


# ═══ B) User-accurate grading ═══════════════════════════════════════════

def _grading_fixture(lost_from_sec=None, total_sec=120):
    """Plan + perception where the face sits at cx=300 and the crop follows —
    except (optionally) after ``lost_from_sec`` where the crop points away."""
    kfs = [{"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0}]
    if lost_from_sec is not None:
        kfs.append({"time_ms": lost_from_sec * 1000, "x": 1300,
                    "transition": "cut", "transition_ms": 0})
    plan = SimpleNamespace(
        keyframes=kfs, crop_w=608, crop_h=1080, crop_y=0, max_x=1312,
        duration_ms=total_sec * 1000, scenes=[])
    face_timeline = {
        t * 1000: [{"cx": 300, "x": 260, "w": 80, "h": 100, "y": 200,
                    "confidence": 0.9, "saliency": 1.0, "mouth_motion": 0.2}]
        for t in range(0, total_sec, 2)
    }
    perc = SimpleNamespace(duration_ms=total_sec * 1000,
                           face_timeline=face_timeline)
    return plan, perc


def test_evaluator_window_slice_isolates_problems():
    from backend.services.reframe_evaluator import ReframeEvaluator
    plan, perc = _grading_fixture(lost_from_sec=60)
    good = ReframeEvaluator(plan, perc).run(start_sec=0, end_sec=60, quiet=True)
    bad = ReframeEvaluator(plan, perc).run(start_sec=60, end_sec=120, quiet=True)
    assert good.face_coverage_pct > 95
    assert bad.face_coverage_pct < 5
    assert good.window_start_sec == 0 and good.window_end_sec == 60


def test_high_density_caps_the_grade():
    from backend.services.reframe_evaluator import ReframeEvaluator
    plan, perc = _grading_fixture(lost_from_sec=90)
    rep = ReframeEvaluator(plan, perc).run(quiet=True)
    # 30 lost seconds in 2 minutes = 15 HIGH/min — far over the cap line.
    assert rep.high_problems_per_min > 4
    assert rep.grade in ("D", "F")
    assert rep.grade_uncapped  # the pre-cap grade is preserved for the report


def test_p10_reflects_worst_windows():
    from backend.services.reframe_evaluator import ReframeEvaluator
    plan, perc = _grading_fixture(lost_from_sec=90)
    rep = ReframeEvaluator(plan, perc).run(quiet=True)
    clean = ReframeEvaluator(*_grading_fixture()).run(quiet=True)
    assert rep.p10_window_score < clean.p10_window_score
    assert clean.p10_window_score > 95


def test_per_clip_aggregate_weights_by_duration():
    from backend.services.reframe_evaluator import evaluate_per_clip
    plan, perc = _grading_fixture(lost_from_sec=60)
    clips = [
        {"id": 1, "start_time": 0.0, "end_time": 50.0},     # clean window
        {"id": 2, "start_time": 70.0, "end_time": 80.0},    # lost window
    ]
    pairs, agg = evaluate_per_clip(plan, perc, clips)
    assert len(pairs) == 2
    scores = {c["id"]: r.overall_score for c, r in pairs}
    assert scores[1] > scores[2]
    assert agg["clips_graded"] == 2
    assert agg["worst_clip_id"] == 2
    # Duration weighting: the clean 50s clip dominates the 10s bad one.
    assert agg["weighted_score"] > (scores[1] + scores[2]) / 2


def test_vlm_spotcheck_sample_times_spread_across_clips():
    from backend.services.reframe_vlm_judge import _sample_times
    clips = [{"start_time": 0.0, "end_time": 30.0},
             {"start_time": 100.0, "end_time": 130.0}]
    times = _sample_times(clips, 12)
    assert len(times) == 12
    assert sum(1 for t in times if t < 30) == 6
    assert sum(1 for t in times if t >= 100) == 6


def test_vlm_spotcheck_disabled_returns_none(monkeypatch):
    import asyncio
    from backend.services import reframe_vlm_judge as J
    monkeypatch.setattr(settings, "REFRAME_VLM_SPOTCHECK", False)
    out = asyncio.run(J.vlm_spotcheck_framing("/v.mp4", SimpleNamespace(), []))
    assert out is None


def test_operator_corrections_plumbing():
    from backend.models import ExportRequest
    assert ExportRequest.model_fields["manual_crop_overrides"].default == 0
    with open("backend/routers/clips.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "REFRAME-CORRECTION" in src
    assert "operator_corrections" in src
    with open("frontend/src/pages/Analysis.jsx", encoding="utf-8") as fh:
        fsrc = fh.read()
    assert "manual_crop_overrides" in fsrc


def test_pipeline_attaches_per_clip_reports():
    src = _pipeline_src()
    assert "evaluate_per_clip" in src
    assert "clip_weighted" in src
    assert "vlm_spotcheck_framing" in src


# ═══ C) Reframing upgrades ═══════════════════════════════════════════════

def test_live_action_corroborated_by_persistent_track():
    from backend.services.reframer_models import PerceptionResult
    p = PerceptionResult()
    # 60% face coverage (below the old flat 70% bar) with one PERSISTENT
    # track → live action via corroboration.
    for t in range(100):
        p.face_timeline[t * 1000] = (
            [{"cx": 300, "track_id": 7}] if t % 5 < 3 else [])
    assert p.is_live_action is True


def test_low_coverage_still_not_live_action():
    from backend.services.reframer_models import PerceptionResult
    p = PerceptionResult()
    # 30% coverage — anime-like — even with a recurring track id.
    for t in range(100):
        p.face_timeline[t * 1000] = (
            [{"cx": 300, "track_id": 7}] if t % 10 < 3 else [])
    assert p.is_live_action is False


def test_midrange_coverage_without_persistence_not_live_action():
    from backend.services.reframer_models import PerceptionResult
    p = PerceptionResult()
    # 60% coverage but every detection is a different ghost track.
    for t in range(100):
        p.face_timeline[t * 1000] = (
            [{"cx": 300, "track_id": t}] if t % 5 < 3 else [])
    assert p.is_live_action is False


def test_problem_windows_merge_and_cap():
    from backend.services.reframer_repair import _merge_problem_windows
    problems = ([{"time_sec": s, "severity": "HIGH", "type": "face_missing"}
                 for s in (10, 11, 12, 40)]
                + [{"time_sec": 99, "severity": "MED", "type": "off_center"}])
    windows = _merge_problem_windows(problems)
    # 10-12 merge into one window; 40 is its own; the MED is ignored.
    assert len(windows) == 2
    assert windows[0][0] <= 10 and windows[0][1] >= 13
    many = [{"time_sec": s * 10, "severity": "HIGH", "type": "face_missing"}
            for s in range(100)]
    assert len(_merge_problem_windows(many, max_windows=40)) == 40


def test_inter_sample_tracking_wired():
    with open("backend/services/reframer_perceiver.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "_track_through_gap" in src
    assert "REFRAMER_INTER_SAMPLE_TRACKING" in src
    assert "calcOpticalFlowPyrLK" in src
    assert "'tracked': True" in src


def test_repair_wired_before_bridge():
    src = _pipeline_src()
    i_repair = src.index("repair_high_problem_windows")
    i_bridge = src.index("asyncio.to_thread(_run_bridge)")
    assert i_repair < i_bridge


def test_per_scene_eyeline_reaches_bridge_rects():
    from backend.services import reframer_bridge
    kfs = [
        {"time_ms": 0, "x": 100, "y": 0, "transition": "cut", "transition_ms": 0},
        {"time_ms": 2000, "x": 100, "y": 300, "transition": "cut", "transition_ms": 0},
    ]
    plan = SimpleNamespace(
        keyframes=kfs, crop_w=960, crop_h=540, crop_y=100,
        source_width=1920, source_height=1080, duration_ms=4000,
        scenes=[{"start_ms": 0, "end_ms": 2000, "strategy": "a"},
                {"start_ms": 2000, "end_ms": 4000, "strategy": "b"}])
    fez = reframer_bridge.to_fez_render_plan(
        plan, SimpleNamespace(duration_ms=4000, src_w=1920, src_h=1080),
        src_w=1920, src_h=1080, total_duration=4.0)
    y0 = fez.ops[0].motion_path[0].rect.y
    y1 = fez.ops[1].motion_path[0].rect.y
    assert y0 == pytest.approx(0.0, abs=0.01)
    assert y1 == pytest.approx(300 / 1080, abs=0.01)


def test_bridge_y_falls_back_to_plan_crop_y():
    from backend.services import reframer_bridge
    kfs = [{"time_ms": 0, "x": 100, "transition": "cut", "transition_ms": 0}]
    plan = SimpleNamespace(
        keyframes=kfs, crop_w=960, crop_h=540, crop_y=270,
        source_width=1920, source_height=1080, duration_ms=2000, scenes=[])
    fez = reframer_bridge.to_fez_render_plan(
        plan, SimpleNamespace(duration_ms=2000, src_w=1920, src_h=1080),
        src_w=1920, src_h=1080, total_duration=2.0)
    assert fez.ops[0].motion_path[0].rect.y == pytest.approx(270 / 1080, abs=0.01)


def test_planner_tags_scene_keyframes_with_y():
    with open("backend/services/reframer_planner.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "_scene_crop_y" in src
    assert "REFRAMER_PER_SCENE_EYELINE" in src


def test_exporter_reads_plan_vertical_framing():
    with open("backend/services/clip_exporter.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "vertical framing from cached RenderPlan" in src.lower() or \
        "framing from cached RenderPlan" in src


def test_reframing_flag_defaults():
    assert settings.REFRAMER_INTER_SAMPLE_TRACKING is True
    assert settings.REFRAMER_TRACK_POINTS_PER_GAP == 3
    assert settings.REFRAMER_PROBLEM_REPAIR is True
    assert settings.REFRAMER_REPAIR_MAX_WINDOWS == 40
    assert settings.REFRAMER_PER_SCENE_EYELINE is True
    assert settings.REFRAME_VLM_SPOTCHECK is True
    assert settings.REFRAME_VLM_FRAMES == 12
