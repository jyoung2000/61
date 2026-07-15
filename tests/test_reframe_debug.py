"""Reframe debug bundle (2026-07-15).

A machine-readable per-job artifact so a bad reframe is diagnosable without a
re-run: per-scene measured signals + derived params + chosen strategy + flags,
the crop-x keyframe trajectory, and a computed decision summary. Pure + fail-soft.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframe_debug import (  # noqa: E402
    strategy_histogram,
    keyframe_x_stats,
    scene_flags,
    build_reframe_debug_bundle,
    write_reframe_debug_bundle,
)


class _FakePlan:
    """Minimal RenderPlan-shaped object (the bundle reads via getattr/.get)."""
    def __init__(self, scenes, keyframes, strategy_log):
        self.scenes = scenes
        self.keyframes = keyframes
        self.strategy_log = strategy_log
        self.crop_w, self.crop_h, self.crop_y = 608, 1080, 0
        self.source_width, self.source_height = 1920, 1080
        self.target_width, self.target_height = 1080, 1920
        self.fps, self.duration_ms = 30.0, 60000
        self.max_x = 1312


def _scene(idx, strategy, **signals):
    return {"scene_idx": idx, "start_ms": idx * 1000, "end_ms": idx * 1000 + 1000,
            "strategy": strategy, "keyframe_count": 2,
            "signals": signals, "params": {"strategy_label": strategy}}


# ── strategy_histogram ──────────────────────────────────────────────────────

def test_strategy_histogram():
    scenes = [_scene(0, "single_face"), _scene(1, "single_face"), _scene(2, "center")]
    assert strategy_histogram(scenes) == {"single_face": 2, "center": 1}
    assert strategy_histogram([]) == {}


# ── keyframe_x_stats ────────────────────────────────────────────────────────

def test_keyframe_x_stats_jitter():
    kfs = [{"time_ms": 0, "x": 100}, {"time_ms": 500, "x": 140},
           {"time_ms": 1000, "x": 100}]
    st = keyframe_x_stats(kfs)
    assert st["count"] == 3
    assert st["min_x"] == 100 and st["max_x"] == 140 and st["span_x"] == 40
    assert st["mean_abs_step"] == 40.0            # |140-100| , |100-140| → 40
    assert st["max_abs_step"] == 40


def test_keyframe_x_stats_empty():
    st = keyframe_x_stats([])
    assert st["count"] == 0 and st["mean_abs_step"] == 0.0 and st["min_x"] is None


# ── scene_flags ─────────────────────────────────────────────────────────────

def test_scene_flags_no_subject():
    fl = scene_flags({"face_density": 0.0, "person_density": 0.0, "motion_energy": 0.0})
    assert any(f.startswith("no-subject") for f in fl)


def test_scene_flags_no_face_but_person():
    fl = scene_flags({"face_density": 0.0, "person_density": 0.6})
    assert any("no-face" in f for f in fl)
    assert not any(f.startswith("no-subject") for f in fl)


def test_scene_flags_cant_fit_and_unstable_and_switch():
    fl = scene_flags({"face_density": 2.0, "spatial_spread": 1.4,
                      "face_persistence": 0.2, "speaker_alternation": 0.8})
    assert any("subjects-dont-fit" in f for f in fl)
    assert any("unstable-track" in f for f in fl)
    assert any("frequent-speaker-switch" in f for f in fl)


def test_scene_flags_clean_scene_has_none():
    fl = scene_flags({"face_density": 1.0, "person_density": 0.9,
                      "spatial_spread": 0.4, "face_persistence": 0.95,
                      "speaker_alternation": 0.1})
    assert fl == []


def test_scene_flags_defensive_on_missing_signals():
    assert scene_flags({}) != []            # empty signals → no-subject flagged
    assert scene_flags(None) != []          # None → treated as no-subject


# ── build + write ───────────────────────────────────────────────────────────

def _plan():
    scenes = [
        _scene(0, "single_face", face_density=1.0, person_density=0.9,
               spatial_spread=0.3, face_persistence=0.9),
        _scene(1, "center", face_density=0.0, person_density=0.0, motion_energy=0.0),
        _scene(2, "two_face", face_density=2.0, spatial_spread=1.5),
    ]
    kfs = [{"time_ms": 0, "x": 200}, {"time_ms": 1000, "x": 900},
           {"time_ms": 2000, "x": 210}]
    slog = [{"time_range": [0, 1000], "strategy": s["strategy"]} for s in scenes]
    return _FakePlan(scenes, kfs, slog)


def test_build_bundle_shape_and_summary():
    b = build_reframe_debug_bundle(
        _plan(), perception_summary={"detected_language": "ja", "src_w": 1920},
        tracer_counts={"scene_signals": 3}, total_elapsed=712.4, job_id="abc")
    assert b["schema_version"] == 1 and b["job_id"] == "abc"
    assert b["crop"]["crop_w"] == 608 and b["crop"]["max_x"] == 1312
    assert b["perception"]["detected_language"] == "ja"
    assert b["tracer_counts"]["scene_signals"] == 3
    s = b["summary"]
    assert s["scene_count"] == 3 and s["keyframe_count"] == 3
    assert s["total_elapsed_s"] == 712.4
    assert s["strategy_histogram"] == {"single_face": 1, "center": 1, "two_face": 1}
    assert s["no_subject_scenes"] == [1]        # scene 1 has no subject
    assert s["keyframe_x"]["max_abs_step"] == 700   # 200→900 snap is visible
    # Per-scene flags are attached.
    assert any(f.startswith("no-subject") for f in b["scenes"][1]["flags"])
    assert any("subjects-dont-fit" in f for f in b["scenes"][2]["flags"])


def test_write_bundle_fail_soft(tmp_path):
    p = str(tmp_path / "sub" / "reframe_debug.json")
    out = write_reframe_debug_bundle(p, _plan(), total_elapsed=1.0)
    assert out == p
    import json
    with open(p) as f:
        data = json.load(f)
    assert data["summary"]["scene_count"] == 3


def test_write_bundle_never_raises_on_bad_plan(tmp_path):
    # A plan missing everything must not raise — build is defensive.
    class _Broken:
        pass
    b = build_reframe_debug_bundle(_Broken(), total_elapsed=1.0)
    assert b["summary"]["scene_count"] == 0 and b["summary"]["keyframe_count"] == 0
    # A path whose PARENT is a regular file → makedirs fails (even as root) →
    # fail-soft None, never raises.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    assert write_reframe_debug_bundle(str(blocker / "nested" / "x.json"), _Broken()) is None
