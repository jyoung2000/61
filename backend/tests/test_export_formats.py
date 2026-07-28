"""Acceptance tests for Task 5 — WebVTT + bilingual SRT + export toggles."""

import re

from backend.models import TranscriptSegment
from backend.services.srt_generator import (
    generate_srt, generate_vtt, generate_bilingual_srt,
)


def _seg(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _segments():
    # Named speakers: the diarizer's "Speaker N" placeholders are suppressed on
    # export (they name nobody and eat the line budget), so a fixture that wants
    # to exercise the label toggles has to carry real names.
    return [
        _seg(1.2, 4.8, "Hello everyone, welcome to the show.", "Ada"),
        _seg(65.0, 68.3, "Thanks for having me!", "Grace"),
    ]


# ── WebVTT ────────────────────────────────────────────────────────────────

def test_vtt_header_and_dot_ms_timing():
    out = generate_vtt(_segments(), include_speakers=True, enforce_readability_rules=False)
    assert out.startswith("WEBVTT")
    # Dot-millisecond timing (HH:MM:SS.mmm).
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", out)
    # Voice tags when speakers included.
    assert "<v Ada>" in out
    assert "<v Grace>" in out


def test_vtt_strips_speakers_when_disabled():
    out = generate_vtt(_segments(), include_speakers=False, enforce_readability_rules=False)
    assert "<v " not in out


def test_vtt_inline_timestamps():
    out = generate_vtt(
        _segments(), include_speakers=False,
        include_timestamps_in_text=True, enforce_readability_rules=False)
    assert "[00:01]" in out      # first cue at 1.2s
    assert "[01:05]" in out      # second cue at 65.0s


# ── Bilingual SRT ─────────────────────────────────────────────────────────

def test_bilingual_translation_top():
    source = [
        _seg(1.0, 4.0, "Hello everyone, welcome to the show.", "Speaker 1"),
        _seg(65.0, 68.0, "Thanks for having me!", "Speaker 2"),
    ]
    # Translated segments carry bogus timings to prove source timings win.
    translated = [
        _seg(999.0, 1000.0, "Hola a todos, bienvenidos al programa.", "Speaker 1"),
        _seg(999.0, 1000.0, "¡Gracias por invitarme!", "Speaker 2"),
    ]
    out = generate_bilingual_srt(
        source, translated, order="translation_top",
        include_speakers=False, enforce_readability_rules=False)
    cues = [c for c in out.strip().split("\n\n") if c.strip()]
    assert len(cues) == 2
    for cue in cues:
        body = cue.split("\n")[2:]   # index, timing, then text lines
        assert len(body) == 2        # exactly two text lines per cue
    # Translation is the first text line.
    first_cue = cues[0].split("\n")
    assert "Hola a todos" in first_cue[2]
    assert "Hello everyone" in first_cue[3]
    # Source timing preserved (not the translated segment's bogus timing).
    assert "00:00:01,000 --> 00:00:04,000" in cues[0]
    assert "999" not in out


def test_bilingual_original_top():
    source = _segments()
    translated = ["Hola", "Gracias"]
    out = generate_bilingual_srt(
        source, translated, order="original_top",
        include_speakers=False, enforce_readability_rules=False)
    first_cue = out.strip().split("\n\n")[0].split("\n")
    assert "Hello everyone" in first_cue[2]   # original first
    assert "Hola" in first_cue[3]             # translation below


# ── Cross-format toggles ──────────────────────────────────────────────────

def test_srt_speaker_and_timestamp_toggles():
    segs = _segments()
    with_labels = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Ada]" in with_labels
    no_labels = generate_srt(segs, include_speakers=False, enforce_readability_rules=False)
    assert "[Ada]" not in no_labels
    inline = generate_srt(
        segs, include_speakers=False,
        include_timestamps_in_text=True, enforce_readability_rules=False)
    assert "[00:01]" in inline
    assert "[01:05]" in inline


def test_srt_backward_compatible_signature():
    # Existing positional callers must keep working.
    out = generate_srt(_segments())
    assert "-->" in out
