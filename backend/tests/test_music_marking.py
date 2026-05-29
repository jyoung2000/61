"""Tests for music-region marking ([♪ music ♪]) instead of transcribing
sung lyrics. Driven by the Gundam Wing OP/ED themes being dropped entirely
(and Whisper hallucinating credits filler over them)."""

import asyncio

import pytest

from backend.services import audio_analyzer as aa
from backend.services.audio_analyzer import (
    MUSIC_MARKER, is_subtitle_marker, merge_markers, detect_music_markers,
)


# ── marker detection ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("[♪ music ♪]", True),
    ("[music]", True),
    ("[applause]", True),
    ("[laughter]", True),
    ("Just wild beat communication", False),   # actual lyric line
    ("I am Heero Yuy.", False),
    ("", False),
    ("[He turned to the door]", False),         # stage-direction-like, not a tag
])
def test_is_subtitle_marker(text, expected):
    assert is_subtitle_marker(text) is expected


# ── merge_markers ───────────────────────────────────────────────────────────

def _seg(s, e, text="hi"):
    return {"start": s, "end": e, "text": text, "speaker": "Speaker 1"}


def test_merge_inserts_in_gaps_sorted():
    transcript = [_seg(60, 64, "narration"), _seg(120, 124, "more")]
    markers = [{"start": 26, "end": 56, "text": MUSIC_MARKER, "speaker": ""}]
    out = merge_markers(transcript, markers)
    assert len(out) == 3
    assert out[0]["text"] == MUSIC_MARKER       # earliest, sorted first
    assert [s["start"] for s in out] == [26, 60, 120]


def test_merge_drops_marker_overlapping_speech():
    transcript = [_seg(60, 90, "talking")]
    markers = [{"start": 70, "end": 100, "text": MUSIC_MARKER}]  # overlaps speech
    out = merge_markers(transcript, markers)
    assert len(out) == 1
    assert out[0]["text"] == "talking"


def test_merge_empty_markers_noop():
    transcript = [_seg(0, 4)]
    assert merge_markers(transcript, []) == transcript


# ── detect_music_markers (classify mocked) ──────────────────────────────────

def test_detect_music_markers_filters_short_and_speech(monkeypatch):
    async def _fake_classify(audio_path, window_seconds=1.0):
        return [
            {"timestamp": 26.0, "duration": 30.0, "type": "music", "confidence": 0.9},
            {"timestamp": 60.0, "duration": 50.0, "type": "speech", "confidence": 0.9},
            {"timestamp": 200.0, "duration": 2.0, "type": "music", "confidence": 0.8},  # too short
        ]
    monkeypatch.setattr(aa, "classify_audio_events", _fake_classify)
    transcript = [_seg(61, 65, "dialogue")]
    markers = asyncio.get_event_loop().run_until_complete(
        detect_music_markers("/x.wav", transcript, min_seconds=5.0))
    assert len(markers) == 1
    assert markers[0]["text"] == MUSIC_MARKER
    assert abs(markers[0]["start"] - 26.0) < 1e-6


def test_detect_music_markers_noop_on_classify_failure(monkeypatch):
    async def _boom(audio_path, window_seconds=1.0):
        raise RuntimeError("numpy unavailable")
    monkeypatch.setattr(aa, "classify_audio_events", _boom)
    markers = asyncio.get_event_loop().run_until_complete(
        detect_music_markers("/x.wav", [], min_seconds=5.0))
    assert markers == []
