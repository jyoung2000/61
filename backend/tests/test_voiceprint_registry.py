"""Acceptance tests for Task 3 — voiceprint persistence & matching."""

import pytest

from backend.services import voiceprint_registry as vpr
from backend.services.voiceprint_registry import VoiceprintRegistry


@pytest.fixture
def registry(tmp_path):
    return VoiceprintRegistry(path=str(tmp_path / "voiceprints.json"))


# A pair of clearly-different unit-ish vectors.
EMB_A = [1.0, 0.0, 0.0, 0.0]
EMB_A_SIMILAR = [0.95, 0.05, 0.02, 0.01]
EMB_B = [0.0, 1.0, 0.0, 0.0]


# ── Enroll → match ─────────────────────────────────────────────────────────

def test_enroll_then_match_above_threshold(registry):
    vid = registry.enroll_or_update(EMB_A, "Eric")
    assert vid is not None
    m = registry.match(EMB_A_SIMILAR)
    assert m is not None
    assert m[1] == "Eric"


def test_dissimilar_embedding_returns_none(registry):
    registry.enroll_or_update(EMB_A, "Eric")
    assert registry.match(EMB_B) is None


def test_match_empty_registry_returns_none(registry):
    assert registry.match(EMB_A) is None


# ── Centroid running-mean update ───────────────────────────────────────────

def test_centroid_update_moves_toward_new_samples(registry):
    vid = registry.enroll_or_update([1.0, 0.0], "Eric")
    cent0 = list(registry._data[vid]["centroid_embedding"])
    assert cent0 == [1.0, 0.0]
    # Update with a sample pulling toward [0, 1].
    registry.enroll_or_update([0.0, 1.0], "Eric")
    cent1 = registry._data[vid]["centroid_embedding"]
    # new = (centroid*1 + emb) / 2  → [0.5, 0.5]
    assert cent1 == [0.5, 0.5]
    assert registry._data[vid]["sample_count"] == 2
    # Moved toward the new sample on the second axis, away on the first.
    assert cent1[1] > cent0[1]
    assert cent1[0] < cent0[0]


def test_same_name_updates_existing_not_duplicate(registry):
    registry.enroll_or_update(EMB_A, "Eric")
    registry.enroll_or_update(EMB_A_SIMILAR, "Eric")
    assert len(registry.list_voiceprints()) == 1


# ── Persistence round-trip ─────────────────────────────────────────────────

def test_persistence_survives_reload(tmp_path):
    path = str(tmp_path / "voiceprints.json")
    reg = VoiceprintRegistry(path=path)
    vid = reg.enroll_or_update(EMB_A, "Eric")
    # Simulate a restart: new instance reads the same mount-backed file.
    reg2 = VoiceprintRegistry(path=path)
    assert vid in reg2._data
    m = reg2.match(EMB_A_SIMILAR)
    assert m is not None and m[1] == "Eric"


def test_delete_and_clear(registry):
    vid = registry.enroll_or_update(EMB_A, "Eric")
    registry.enroll_or_update(EMB_B, "Alice")
    assert registry.delete(vid) is True
    assert registry.delete("nonexistent") is False
    assert len(registry.list_voiceprints()) == 1
    assert registry.clear() == 1
    assert registry.list_voiceprints() == []


# ── Graceful degradation: missing embedding backend ─────────────────────────

def test_missing_backend_extract_noop(monkeypatch, tmp_path):
    # No HF_TOKEN / pyannote → SpeakerEmbedder.try_load returns False and
    # extraction yields {} without raising.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"\x00")
    out = vpr.extract_speaker_embeddings(str(audio), {0: "SPEAKER_00", 200: "SPEAKER_00"})
    assert out == {}


def test_missing_backend_apply_names_noop(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    transcript = [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": "Speaker 1"}]
    out = vpr.apply_voiceprint_names(transcript, {0: "SPEAKER_00"}, "/nonexistent.wav")
    assert out is transcript


def test_missing_backend_enroll_noop(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    assert vpr.enroll_from_segments(
        "/nonexistent.wav",
        [{"start": 0.0, "end": 1.0, "speaker": "Speaker 1"}],
        "Eric",
    ) is None


def test_match_empty_embedding_returns_none(registry):
    registry.enroll_or_update(EMB_A, "Eric")
    assert registry.match([]) is None
    assert registry.enroll_or_update([], "Nobody") is None
