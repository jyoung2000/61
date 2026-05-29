"""Formatting-parity tests: make ClipAI's SRT structurally match a
hand-authored SRT — chronological cues, no backwards/degenerate cues,
no speaker labels for single-speaker content, and no wall-of-text cues.

Driven by an export whose cues were out of order (cue 2 started before
cue 1), carried "[Speaker 1]" on every line, and packed most of the
episode into one 0.8s cue.
"""

import re

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.srt_generator import generate_srt, effective_include_speakers
from backend.services.subtitle_formatter import enforce_readability


def _seg(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _starts(srt: str):
    return [m.group(0) for m in re.finditer(r"\d{2}:\d{2}:\d{2},\d{3}(?= -->)", srt)]


def test_cues_emitted_in_chronological_order():
    # Deliberately out of order (mirrors the broken export).
    segs = [
        _seg(24.0, 25.0, "fourth"),
        _seg(8.0, 9.0, "second"),
        _seg(1.0, 2.0, "first"),
        _seg(20.0, 21.0, "third"),
    ]
    out = generate_srt(segs, enforce_readability_rules=False)
    starts = _starts(out)
    assert starts == sorted(starts)
    # "first" cue comes before "fourth".
    assert out.index("first") < out.index("fourth")


def test_backwards_cue_dropped():
    segs = [_seg(1.0, 2.0, "ok"), _seg(9.0, 5.0, "backwards")]  # end < start
    out = generate_srt(segs, enforce_readability_rules=False)
    assert "backwards" not in out
    assert "ok" in out


def test_single_speaker_suppresses_labels():
    segs = [_seg(1.0, 2.0, "alpha"), _seg(3.0, 4.0, "beta")]   # all Speaker 1
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Speaker 1]" not in out
    assert "alpha" in out


def test_multi_speaker_keeps_labels():
    segs = [_seg(1.0, 2.0, "alpha", "Speaker 1"),
            _seg(3.0, 4.0, "beta", "Speaker 2")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Speaker 1]" in out and "[Speaker 2]" in out


def test_effective_include_speakers_logic():
    assert effective_include_speakers([_seg(0, 1, "a")], True) is False
    assert effective_include_speakers(
        [_seg(0, 1, "a", "Speaker 1"), _seg(1, 2, "b", "Speaker 2")], True) is True
    assert effective_include_speakers([_seg(0, 1, "a", "Speaker 1")], False) is False


def test_oversized_short_cue_is_split():
    # A wall of text with an implausibly short (0.8s) duration — the corrupt
    # case. enforce_readability must explode it into several readable cues
    # rather than leaving one giant block.
    sentences = ["This is sentence number %d about the battle." % i for i in range(12)]
    text = " ".join(sentences)
    big = _seg(100.0, 100.8, text)   # ~500 chars in 0.8s
    out = enforce_readability([big])
    assert len(out) > 3, f"giant block not split (got {len(out)} cues)"
    # No resulting cue should still be the whole wall of text.
    assert all(len(s.text) < len(text) for s in out)
    # Output is chronological.
    for i in range(len(out) - 1):
        assert out[i].start <= out[i + 1].start


def test_srt_from_oversized_block_has_many_cues():
    text = " ".join("Line %d of the report here." % i for i in range(15))
    out = generate_srt([_seg(50.0, 50.8, text)])
    cue_count = len([l for l in out.split("\n\n") if l.strip()])
    assert cue_count > 3
