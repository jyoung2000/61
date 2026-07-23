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


# ── Timestamp-less references (DownloadYoutubeSubtitles.com plain text) ─────

def test_parse_reference_lines_blocks():
    from backend.services.reference_transcript import parse_reference_lines
    txt = "\n\n".join(f"Caption line number {i},\nwrapped for display" for i in range(10))
    lines = parse_reference_lines(txt)
    assert len(lines) == 10
    assert lines[0] == "Caption line number 0, wrapped for display"
    # A timestamped text is NOT for this parser.
    assert parse_reference_lines("\n".join(f"[0:{30+i}] hi {i}" for i in range(10))) == []


def test_timestampless_adopt_rewords_matched_cues_only():
    ref = "\n\n".join([
        "I told you.", "I'm a true soldier.", "All areas functioning.",
        "Commencing operations in seven minutes.", "A civilian shuttle...",
        "Mr. Darlian.", "The shuttle will soon enter the atmosphere.",
        "Please fasten your seat belt and remain seated.",
        "What's the matter Relena?", "Aren't you glad to be coming home to Earth?",
    ])
    clip = [
        {"start": 240, "end": 243, "speaker": "S2", "text": "Didn't I say I'm a soldier?"},
        {"start": 250, "end": 254, "speaker": "S1", "text": "All systems normal."},
        {"start": 255, "end": 259, "speaker": "S1", "text": "We'll avoid the operation in seven minutes."},
        {"start": 262, "end": 265, "speaker": "S1", "text": "A civilian shuttle."},
        {"start": 268, "end": 272, "speaker": "S1", "text": "This shuttle will now enter Earth's atmosphere."},
        {"start": 274, "end": 278, "speaker": "S1", "text": "Please fasten your seatbelt and remain seated, thank you."},
        {"start": 280, "end": 283, "speaker": "S2", "text": "What's wrong, Lily?"},
        {"start": 283, "end": 287, "speaker": "S2", "text": "Do you hate going back to Earth so much?"},
        {"start": 289, "end": 291, "speaker": "S1", "text": "Yes, very much."},
        {"start": 292, "end": 295, "speaker": "S2", "text": "Sorry about that."},
    ]
    out, changed = conform_to_reference(clip, ref, mode="adopt")
    assert changed
    texts = [r["text"] for r in out]
    # Strong matches adopt YouTube's exact wording (speaker kept; indices may
    # shift because a cue spanning two reference lines splits in two).
    assert "The shuttle will soon enter the atmosphere." in texts
    shuttle = next(r for r in out if r["text"] == "The shuttle will soon enter the atmosphere.")
    assert shuttle["speaker"] == "S1"
    # …and a cue with no good match keeps ClipAI's own text.
    assert "Yes, very much." in texts


def test_timestampless_pair_match_splits_cue_for_youtube_pacing():
    """A ClipAI cue that covers TWO reference lines splits into two cues at a
    char-proportional cut, reproducing YouTube's finer pacing."""
    ref = "\n\n".join([
        "I told you.", "I'm a true soldier.", "All areas functioning.",
        "Commencing operations in seven minutes.", "A civilian shuttle...",
        "Mr. Darlian.", "The shuttle will soon enter the atmosphere.",
        "Please fasten your seat belt and remain seated.",
    ])
    clip = [
        {"start": 240, "end": 244, "speaker": "S2", "text": "I told you, I'm a true soldier."},
        {"start": 250, "end": 254, "speaker": "S1", "text": "All areas functioning."},
        {"start": 255, "end": 259, "speaker": "S1", "text": "Commencing operations in seven minutes."},
        {"start": 262, "end": 265, "speaker": "S1", "text": "A civilian shuttle..."},
        {"start": 266, "end": 267, "speaker": "S1", "text": "Mr. Darlian."},
        {"start": 268, "end": 272, "speaker": "S1", "text": "The shuttle will soon enter the atmosphere."},
        {"start": 274, "end": 278, "speaker": "S1", "text": "Please fasten your seat belt and remain seated."},
    ]
    out, changed = conform_to_reference(clip, ref, mode="adopt")
    assert changed
    texts = [r["text"] for r in out]
    # The double-line cue split into YouTube's two cues…
    assert "I told you." in texts and "I'm a true soldier." in texts
    a = next(r for r in out if r["text"] == "I told you.")
    b = next(r for r in out if r["text"] == "I'm a true soldier.")
    # …contiguous in time, inside the original span, same speaker.
    assert a["end"] == b["start"]
    assert a["start"] == 240 and b["end"] == 244
    assert a["speaker"] == b["speaker"] == "S2"
