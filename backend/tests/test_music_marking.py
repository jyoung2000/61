"""Tests for music-region marking ([♪ music ♪]) instead of transcribing
sung lyrics. Driven by the Gundam Wing OP/ED themes being dropped entirely
(and Whisper hallucinating credits filler over them)."""

import asyncio

import pytest

def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


from backend.services import audio_analyzer as aa
from backend.services.audio_analyzer import (
    MUSIC_MARKER, is_subtitle_marker, merge_markers, detect_music_markers,
    suppress_speech_in_music_spans, _is_nonlexical_vocalization,
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
    markers = _aiorun(
        detect_music_markers("/x.wav", transcript, min_seconds=5.0))
    assert len(markers) == 1
    assert markers[0]["text"] == MUSIC_MARKER
    assert abs(markers[0]["start"] - 26.0) < 1e-6


def test_detect_music_markers_noop_on_classify_failure(monkeypatch):
    async def _boom(audio_path, window_seconds=1.0):
        raise RuntimeError("numpy unavailable")
    monkeypatch.setattr(aa, "classify_audio_events", _boom)
    markers = _aiorun(
        detect_music_markers("/x.wav", [], min_seconds=5.0))
    assert markers == []


# ── _is_nonlexical_vocalization ─────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("ああああ", True),                # elongated single mora
    ("ラララララ", True),               # sung "la"
    ("la la la la", True),            # latin sung filler (spaces stripped)
    ("aaaaah", True),
    ("mmmm", True),
    ("あーーー", True),                # vowel + prolonged-sound marks
    ("na-na-na-na", True),
    ("", False),                      # empty → nothing to suppress
    ("はい", False),                   # short real word, kept
    ("ok", False),
    ("I am Heero Yuy.", False),        # real dialogue
    ("作戦名はオペレーション・メテオ", False),   # real CJK dialogue
    ("We have to stop OZ from taking the colonies.", False),
])
def test_is_nonlexical_vocalization(text, expected):
    assert _is_nonlexical_vocalization(text) is expected


# ── suppress_speech_in_music_spans (vocalizations-only guard) ────────────────

def _mseg(s, e, text):
    return {"start": s, "end": e, "text": text, "speaker": "Speaker 1"}


def test_suppress_keeps_real_dialogue_over_misclassified_music():
    # A loud action cue the spectral classifier mislabelled "music"; the real
    # dialogue inside it must survive, only the hallucinated sung filler goes.
    span = [(183.0, 260.0)]
    transcript = [
        _mseg(185, 191, "We have to destroy the mobile suits before they reach the colony."),
        _mseg(193, 197, "ラララララ"),
    ]
    kept, suppressed = suppress_speech_in_music_spans(transcript, span)
    assert [s["text"] for s in suppressed] == ["ラララララ"]
    assert any("mobile suits" in k["text"] for k in kept)


def test_suppress_blanket_when_vocalizations_only_false():
    # Opt back into the old behaviour: every cue inside the span is dropped.
    span = [(183.0, 260.0)]
    transcript = [_mseg(185, 191, "Real dialogue that normally survives.")]
    kept, suppressed = suppress_speech_in_music_spans(
        transcript, span, vocalizations_only=False)
    assert kept == []
    assert len(suppressed) == 1


def test_suppress_keeps_markers_and_cues_outside_spans():
    span = [(183.0, 260.0)]
    transcript = [
        {"start": 183, "end": 260, "text": MUSIC_MARKER, "speaker": ""},
        _mseg(300, 305, "ああああ"),   # a vocalization, but OUTSIDE the music span
    ]
    kept, suppressed = suppress_speech_in_music_spans(transcript, span)
    assert suppressed == []
    assert len(kept) == 2


def test_mark_and_suppress_keeps_dialogue_drops_vocalization(monkeypatch):
    async def _fake_classify(audio_path, window_seconds=1.0):
        return [{"timestamp": 180.0, "duration": 80.0, "type": "music", "confidence": 0.9}]
    monkeypatch.setattr(aa, "classify_audio_events", _fake_classify)
    transcript = [
        _mseg(185, 191, "We can't let OZ win this battle."),
        _mseg(210, 214, "ラララララ"),
    ]
    out, n_supp, _n_mark = _aiorun(
        aa.mark_and_suppress_music("/x.wav", transcript, min_seconds=5.0))
    texts = [o["text"] for o in out]
    assert any("OZ win this battle" in t for t in texts)   # dialogue survived
    assert "ラララララ" not in texts                          # vocalization dropped
    assert n_supp == 1
