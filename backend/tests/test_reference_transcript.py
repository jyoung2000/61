"""Conform ClipAI's subtitle track to an operator reference (YouTube captions):
parse SRT/VTT/plain-timestamped, then adopt (words+timing+segmentation) or snap
timing only. The surest way to make the shipped track match a known-good source.
"""
from backend.services.reference_transcript import (
    parse_reference,
    conform_to_reference,
)


def _srt(n):
    blocks = [f"{i+1}\n00:00:{30+i*4:02d},000 --> 00:00:{34+i*4:02d},000\nReference line {i}."
              for i in range(n)]
    return "\n\n".join(blocks)


def test_parses_srt():
    cues = parse_reference(_srt(10))
    assert len(cues) == 10
    assert cues[0]["start"] == 30.0 and cues[0]["end"] == 34.0
    assert cues[0]["text"] == "Reference line 0."


def test_parses_vtt():
    vtt = "WEBVTT\n\n" + "\n\n".join(
        f"00:00:{30+i:02d}.000 --> 00:00:{33+i:02d}.000\nVtt {i}" for i in range(9))
    cues = parse_reference(vtt)
    assert len(cues) == 9
    assert cues[0]["text"] == "Vtt 0"


def test_parses_plain_timestamped_with_speaker_prefix():
    # ClipAI's own .txt export / a YouTube paste: "[0:30] Speaker 1: text".
    plain = "\n".join(f"[0:{30+i:02d}] Speaker 1: hello {i}" for i in range(10))
    cues = parse_reference(plain)
    assert len(cues) == 10
    assert cues[0]["text"] == "hello 0"          # speaker prefix + timestamp stripped
    assert cues[0]["start"] == 30.0
    # start-only format: end derived from the next start
    assert cues[0]["end"] == cues[1]["start"]


def test_too_short_reference_is_ignored():
    assert parse_reference(_srt(3)) == []          # below the trust floor
    clip = [{"start": 0, "end": 5, "text": "x", "speaker": "A"}]
    assert conform_to_reference(clip, _srt(3))[1] is False


def test_adopt_replaces_words_and_timing_and_inherits_speaker():
    clip = [
        {"start": 30, "end": 45, "speaker": "Speaker 1", "text": "garbled A",
         "words": [{"start": 30, "end": 31}]},
        {"start": 46, "end": 60, "speaker": "Speaker 2", "text": "garbled B"},
    ]
    out, changed = conform_to_reference(clip, _srt(10), mode="adopt")
    assert changed
    assert len(out) == 10
    # Reference wording + timing win…
    assert out[0]["text"] == "Reference line 0."
    assert out[0]["start"] == 30.0
    # …ClipAI's speaker is inherited by time overlap…
    assert out[0]["speaker"] == "Speaker 1"
    # …and word timings are cleared (reference text ≠ ClipAI word times).
    assert out[0]["words"] == []


def test_timing_mode_keeps_words_snaps_boundaries():
    clip = [{"start": 30.4, "end": 33.1, "speaker": "Speaker 1", "text": "keep me"}]
    clip += [{"start": 40 + i, "end": 41 + i, "speaker": "Speaker 1", "text": f"c{i}"}
             for i in range(9)]
    out, changed = conform_to_reference(clip, _srt(10), mode="timing")
    assert changed
    assert out[0]["text"] == "keep me"            # ClipAI wording kept
    assert out[0]["start"] == 30.0                # snapped to the reference boundary


def test_blank_reference_is_noop():
    clip = [{"start": 0, "end": 5, "text": "x", "speaker": "A"}]
    out, changed = conform_to_reference(clip, "", mode="adopt")
    assert changed is False and out == clip
