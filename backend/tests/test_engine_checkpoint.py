"""Tests for the engine RESUME checkpoint (perception + plan serialization).

A failed / interrupted job resumes from this checkpoint instead of re-running
the reframer engine (face/motion detection + Whisper transcription + planning).
The round-trip must be faithful — most critically, the int-keyed timeline dicts
(``face_timeline`` … ) have to come back with INTEGER keys, because downstream
code does ``int(t_ms / 1000)`` on them and would crash on the string keys JSON
produces.

The pure-serialization + signature tests run anywhere (the module's heavy
imports are function-local). The full save→load round-trip needs the real
``PerceptionResult`` / ``RenderPlan`` dataclasses, which pull in OpenCV via
``reframer_models``, so it is skipped where cv2 is unavailable.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend.services import pipeline_checkpoint as pc


# ── Pure helpers (no cv2) ────────────────────────────────────────────


def test_int_keyed_restores_integer_keys():
    assert pc._int_keyed({"0": "a", "500": "b"}) == {0: "a", 500: "b"}
    # Non-integer keys are passed through unchanged rather than dropped.
    assert pc._int_keyed({"x": "y"}) == {"x": "y"}
    assert pc._int_keyed(None) == {}


def test_serialize_perception_is_json_safe_and_keeps_fields():
    p = SimpleNamespace(
        src_w=1920, src_h=1080, fps=30.0, duration_ms=10_000, total_frames=300,
        scene_cuts=[1000, 3000],
        transcript_segments=[{"start_sec": 0.0, "end_sec": 1.0, "text": "hi"}],
        detected_language="en",
        coverage_ledger=None,
        face_timeline={0: [{"x": 1, "cx": 5}], 500: []},
        motion_timeline={500: 0.5},
        motion_hotspot={},
        speaker_timeline={0: "SPEAKER_00"},
        track_speaker_map={1: "SPEAKER_00"},
        speech_active={0: True},
        audio_rms={0: 0.1},
        audio_events={0: "music"},
        person_timeline={},
        saliency_hotspot={},
    )
    d = pc._serialize_perception(p)
    # Must survive a JSON round-trip (the on-disk format).
    back = json.loads(json.dumps(d))
    assert back["detected_language"] == "en"
    assert back["transcript_segments"][0]["text"] == "hi"
    # JSON stringifies dict keys — that's exactly what _int_keyed undoes on load.
    assert set(back["face_timeline"].keys()) == {"0", "500"}


def test_serialize_perception_round_trips_int_keys():
    p = SimpleNamespace(
        src_w=0, src_h=0, fps=30.0, duration_ms=0, total_frames=0,
        scene_cuts=[], transcript_segments=[], detected_language="",
        coverage_ledger=None,
        face_timeline={0: [{"cx": 5}], 1000: [{"cx": 9}]},
        motion_timeline={}, motion_hotspot={}, speaker_timeline={},
        track_speaker_map={}, speech_active={}, audio_rms={}, audio_events={},
        person_timeline={}, saliency_hotspot={},
    )
    serialized = json.loads(json.dumps(pc._serialize_perception(p)))
    restored = pc._int_keyed(serialized["face_timeline"])
    assert set(restored.keys()) == {0, 1000}
    assert all(isinstance(k, int) for k in restored)


def test_signature_match_rules():
    a = pc.checkpoint_signature(source_sha="abc", source_language="en",
                                sample_fps=5.0, aspect_ratio="9:16")
    assert pc._signatures_match(a, a) is True
    # An empty source SHA can never be trusted to identify the source.
    b = pc.checkpoint_signature(source_sha="", source_language="en",
                                sample_fps=5.0, aspect_ratio="9:16")
    assert pc._signatures_match(b, b) is False
    # Any differing field misses.
    c = pc.checkpoint_signature(source_sha="abc", source_language="ja",
                                sample_fps=5.0, aspect_ratio="9:16")
    assert pc._signatures_match(a, c) is False
    d = pc.checkpoint_signature(source_sha="abc", source_language="en",
                                sample_fps=2.0, aspect_ratio="9:16")
    assert pc._signatures_match(a, d) is False


def test_signature_includes_planner_fingerprint():
    base = pc.checkpoint_signature(source_sha="abc", source_language="en",
                                   sample_fps=5.0, aspect_ratio="9:16",
                                   planner_fingerprint="X=1")
    other = pc.checkpoint_signature(source_sha="abc", source_language="en",
                                    sample_fps=5.0, aspect_ratio="9:16",
                                    planner_fingerprint="X=2")
    assert pc._signatures_match(base, other) is False


def test_signature_diff_names_the_differing_field():
    a = pc.checkpoint_signature(source_sha="abc123def456xyz", source_language="en",
                                sample_fps=5.0, aspect_ratio="9:16")
    b = pc.checkpoint_signature(source_sha="abc123def456xyz", source_language="ja",
                                sample_fps=5.0, aspect_ratio="9:16")
    diff = pc._signature_diff(a, b)
    assert "source_language" in diff and "sample_fps" not in diff
    # An empty EXPECTED sha is the "can never trust a checkpoint" case and is
    # called out explicitly rather than as a field mismatch.
    empty = pc.checkpoint_signature(source_sha="", source_language="en",
                                    sample_fps=5.0, aspect_ratio="9:16")
    assert "source_sha is empty" in pc._signature_diff(a, empty)


# ── Serialization hardening (the actual silent-failure cause) ─────────
#
# A single un-serializable dict KEY made the whole save raise — and the
# best-effort ``except`` swallowed it, so the next revive re-ran detection +
# transcription from scratch ("died at translate → restarted faces"). The
# ``default=`` hook can't help: it only ever sees VALUES. ``_json_safe`` must
# normalize keys too. These run without numpy by using stdlib types json also
# rejects as keys (tuples) / values (sets, custom objects).


def test_json_safe_key_normalizes_integral_and_stringifies_the_rest():
    assert pc._json_safe_key(5) == 5
    assert pc._json_safe_key("k") == "k"
    assert pc._json_safe_key(True) is True            # bool stays distinct from 1
    assert pc._json_safe_key(3.0) == 3                # integral float → int (round-trips)
    assert pc._json_safe_key(2.5) == "2.5"            # fractional float → str
    assert pc._json_safe_key((1, 2)) == "(1, 2)"      # tuple key → str


def test_json_safe_makes_an_unserializable_payload_dumpable():
    class Weird:
        def __str__(self):
            return "weird"

    payload = {
        (1, 2): "tuple-key",              # tuple key → json.dumps raises without coercion
        3: {"nested": {1.0: "x"}},        # integral float key deep down
        "set": {Weird()},                 # set of a custom object as a value
        "n": None,
    }
    # The raw payload is genuinely un-dumpable — proves the coercion is load-bearing.
    with pytest.raises(TypeError):
        json.dumps(payload)
    # After _json_safe it dumps + reloads cleanly.
    out = json.dumps(pc._json_safe(payload))
    back = json.loads(out)
    assert back["3"]["nested"]["1"] == "x"
    assert back["set"] == ["weird"]
    assert back["n"] is None


# ── Verify-after-save + numpy round-trip (need the real dataclasses) ──


def test_save_with_empty_sha_fails_verify(tmp_path, monkeypatch):
    """A checkpoint the loader can never accept (empty source SHA) must be
    reported as a FAILED save, not a silent success that strands the next
    revive with no usable checkpoint."""
    pytest.importorskip("cv2")
    from backend.services.reframer_models import PerceptionResult, RenderPlan

    monkeypatch.setattr(pc, "checkpoint_dir", lambda jid: str(tmp_path / jid))
    sig = pc.checkpoint_signature(source_sha="", source_language="en",
                                  sample_fps=5.0, aspect_ratio="9:16")
    ok = asyncio.run(pc.save_engine_checkpoint(
        "jobEmptySha", PerceptionResult(), RenderPlan(), signature=sig, audio_meta={}))
    assert ok is False


def test_save_survives_numpy_keys_and_values(tmp_path, monkeypatch):
    """The real engine emits numpy scalars in timeline values (and a future
    detector could emit numpy KEYS). The save must still produce a checkpoint
    that reloads with integer keys."""
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from backend.services.reframer_models import PerceptionResult, RenderPlan

    monkeypatch.setattr(pc, "checkpoint_dir", lambda jid: str(tmp_path / jid))
    perception = PerceptionResult(
        src_w=1920, src_h=1080, fps=30.0, duration_ms=2000, total_frames=60,
        face_timeline={np.int64(0): [{"cx": np.int32(5), "score": np.float32(0.9)}],
                       np.int64(500): []},
        motion_timeline={np.int64(500): np.float64(1.25)},
    )
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        keyframes=[{"time_ms": np.int64(0), "x": np.float32(1.5)}],
    )
    sig = pc.checkpoint_signature(source_sha="np-sha", source_language="en",
                                  sample_fps=5.0, aspect_ratio="9:16")
    ok = asyncio.run(pc.save_engine_checkpoint(
        "jobNumpy", perception, plan, signature=sig, audio_meta={}))
    assert ok is True  # would have been False (silent serialization failure) before

    loaded = asyncio.run(pc.load_engine_checkpoint("jobNumpy", signature=sig))
    assert loaded is not None
    pr, _rp, _stub = loaded
    assert all(isinstance(k, int) for k in pr.face_timeline)
    assert all(isinstance(k, int) for k in pr.motion_timeline)


# ── Full round-trip (needs the real dataclasses → OpenCV) ────────────


def test_full_checkpoint_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from backend.services.reframer_models import (
        PerceptionResult, RenderPlan, CoverageLedger, LedgerBin,
    )

    # Redirect the checkpoint dir into the tmp path (default is /data/uploads).
    monkeypatch.setattr(pc, "checkpoint_dir", lambda jid: str(tmp_path / jid))

    perception = PerceptionResult(
        src_w=1920, src_h=1080, fps=30.0, duration_ms=5000, total_frames=150,
        scene_cuts=[1000, 3000],
        transcript_segments=[{"start_sec": 0.0, "end_sec": 1.0, "text": "hello"}],
        detected_language="en",
        face_timeline={0: [{"x": 1, "y": 2, "w": 3, "h": 4, "cx": 5, "cy": 6}], 500: []},
        motion_timeline={0: 0.5, 500: 1.25},
        motion_hotspot={0: {"cx": 1, "cy": 2, "intensity": 0.3}},
        speaker_timeline={0: "SPEAKER_00", 500: "SPEAKER_01"},
        track_speaker_map={1: "SPEAKER_00"},
        speech_active={0: True, 500: False},
        audio_rms={0: 0.1, 500: 0.2},
        audio_events={0: "music"},
        person_timeline={0: [{"x": 1, "y": 2, "w": 3, "h": 4}]},
        saliency_hotspot={0: {"cx": 7, "cy": 8, "intensity": 0.6}},
        coverage_ledger=CoverageLedger(
            bins={0: LedgerBin(status="covered_speech", text="hello", confidence=0.9),
                  20: LedgerBin(status="covered_silence")},
            bin_width_ms=20, duration_ms=5000,
        ),
    )
    plan = RenderPlan(
        source_width=1920, source_height=1080,
        keyframes=[{"time_ms": 0, "x": 100, "transition": "linear"}],
        scenes=[{"start_ms": 0, "end_ms": 5000}],
        strategy_log=[{"stage": "decide", "note": "ok"}],
    )

    sig = pc.checkpoint_signature(source_sha="sha-1", source_language="en",
                                  sample_fps=5.0, aspect_ratio="9:16")
    saved = asyncio.run(pc.save_engine_checkpoint(
        "jobA", perception, plan, signature=sig,
        audio_meta={"device": "cuda_float16", "model": "base", "requested": "base"}))
    assert saved is True

    loaded = asyncio.run(pc.load_engine_checkpoint("jobA", signature=sig))
    assert loaded is not None
    pr, rp, stub = loaded

    # Int keys restored (the crash-class bug this guards against).
    assert set(pr.face_timeline.keys()) == {0, 500}
    assert all(isinstance(k, int) for k in pr.motion_timeline)
    assert all(isinstance(k, int) for k in pr.speaker_timeline)
    assert all(isinstance(k, int) for k in pr.track_speaker_map)
    # The is_live_action property (derived from face_timeline) still works.
    _ = pr.is_live_action

    # Scalar + list fields preserved.
    assert pr.detected_language == "en"
    assert pr.scene_cuts == [1000, 3000]
    assert pr.transcript_segments[0]["text"] == "hello"

    # coverage_ledger is deliberately NOT checkpointed (planner-only, and its
    # tens-of-thousands of 20ms bins froze the event loop on resume) — it comes
    # back at its None default. The bins set above must not survive the round-trip.
    assert pr.coverage_ledger is None

    # Plan preserved.
    assert rp.source_width == 1920
    assert rp.keyframes == plan.keyframes
    assert rp.strategy_log == plan.strategy_log

    # Engine stub carries the Whisper device/model for the Compute card.
    assert stub._perceiver_audio_device == "cuda_float16"
    assert stub._perceiver_audio_model == "base"
    assert stub._resumed_from_checkpoint is True


def test_checkpoint_signature_mismatch_returns_none(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from backend.services.reframer_models import PerceptionResult, RenderPlan

    monkeypatch.setattr(pc, "checkpoint_dir", lambda jid: str(tmp_path / jid))
    sig = pc.checkpoint_signature(source_sha="sha-1", source_language="en",
                                  sample_fps=5.0, aspect_ratio="9:16")
    asyncio.run(pc.save_engine_checkpoint(
        "jobB", PerceptionResult(), RenderPlan(), signature=sig, audio_meta={}))

    # Source changed → different SHA → must NOT reuse the stale checkpoint.
    other = pc.checkpoint_signature(source_sha="sha-2", source_language="en",
                                    sample_fps=5.0, aspect_ratio="9:16")
    assert asyncio.run(pc.load_engine_checkpoint("jobB", signature=other)) is None


def test_missing_checkpoint_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "checkpoint_dir", lambda jid: str(tmp_path / jid))
    sig = pc.checkpoint_signature(source_sha="sha-1", source_language="en",
                                  sample_fps=5.0, aspect_ratio="9:16")
    assert asyncio.run(pc.load_engine_checkpoint("nope", signature=sig)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
