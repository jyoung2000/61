"""Formatting-parity tests: make ClipAI's SRT structurally match a
hand-authored SRT — chronological cues, no backwards/degenerate cues,
no speaker labels for single-speaker content, and no wall-of-text cues.

Driven by an export whose cues were out of order (cue 2 started before
cue 1), carried "[Speaker 1]" on every line, and packed most of the
episode into one 0.8s cue.
"""

import re

from backend.config import settings
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
    # Two NAMED speakers: the labels carry information, so they ship.
    segs = [_seg(1.0, 2.0, "alpha", "Zechs"),
            _seg(3.0, 4.0, "beta", "Noin")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Zechs]" in out and "[Noin]" in out


def test_placeholder_speaker_labels_are_suppressed():
    # Two speakers, but both still carry the diarizer's own placeholder.
    # "[Speaker 2]" names nobody and costs 12 of a 34-character line, so a
    # reference-grade track omits it — same as the single-speaker case.
    segs = [_seg(1.0, 2.0, "alpha", "Speaker 1"),
            _seg(3.0, 4.0, "beta", "Speaker 2")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Speaker 1]" not in out and "[Speaker 2]" not in out
    assert "alpha" in out and "beta" in out


def test_naming_one_speaker_turns_labels_back_on():
    segs = [_seg(1.0, 2.0, "alpha", "Zechs"),
            _seg(3.0, 4.0, "beta", "Speaker 2")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Zechs]" in out and "[Speaker 2]" in out


def test_placeholder_suppression_is_reversible(monkeypatch):
    from backend.services import srt_generator as sg
    monkeypatch.setattr(
        sg.settings, "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES", False, raising=False)
    segs = [_seg(1.0, 2.0, "alpha", "Speaker 1"),
            _seg(3.0, 4.0, "beta", "Speaker 2")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    assert "[Speaker 1]" in out and "[Speaker 2]" in out


def test_speaker_label_does_not_break_the_line_budget():
    # The label is prepended AFTER the readability pass wrapped the cue, so it
    # has to be re-wrapped or line 1 ships over the on-screen budget.
    budget = int(settings.SUBTITLE_MAX_CHARS_PER_LINE)
    # A one-line cue sitting right at the budget: undecorated it is legal, and
    # "[Zechs] " must not be allowed to push it over.
    line = "Their vanguard has already landed"
    assert len(line) <= budget
    segs = [_seg(1.0, 6.0, line, "Zechs"),
            _seg(7.0, 9.0, "beta", "Noin")]
    out = generate_srt(segs, include_speakers=True, enforce_readability_rules=False)
    body = [ln for ln in out.splitlines()
            if ln and "-->" not in ln and not ln.isdigit()]
    assert body and all(len(ln) <= budget for ln in body), body


def test_effective_include_speakers_logic():
    assert effective_include_speakers([_seg(0, 1, "a")], True) is False
    assert effective_include_speakers(
        [_seg(0, 1, "a", "Zechs"), _seg(1, 2, "b", "Noin")], True) is True
    assert effective_include_speakers(
        [_seg(0, 1, "a", "Speaker 1"), _seg(1, 2, "b", "Speaker 2")], True) is False
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


def test_baked_speaker_prefix_is_stripped_even_with_enforcement_off():
    # A measured download shipped "Speaker 1:" verbatim on every cue —
    # including the music markers — from text that arrived pre-labelled.
    # Attribution lives in the ``speaker`` field; text labels are stripped
    # UNCONDITIONALLY, even when the readability formatter is bypassed.
    segs = [
        _seg(1.0, 3.0, "Speaker 1: [♪ Opening theme ♪]", speaker="Speaker 1"),
        _seg(4.0, 6.0, "Speaker 2: Believe in myself", speaker="Speaker 2"),
        _seg(7.0, 9.0, "[Speaker 1] Hello there", speaker="Speaker 1"),
        _seg(10.0, 12.0, "Relena: Who are you?", speaker="Relena"),
    ]
    out = generate_srt(segs, include_speakers=False, enforce_readability_rules=False)
    assert "Speaker 1:" not in out
    assert "Speaker 2:" not in out
    assert "[Speaker 1]" not in out
    assert "Relena:" not in out
    assert "[♪ Opening theme ♪]" in out
    assert "Believe in myself" in out
    assert "Who are you?" in out


def test_baked_prefix_never_eats_real_dialogue():
    from backend.services.subtitle_formatter import strip_baked_speaker_label
    # A different speaker's name mid-transcript is CONTENT, not a label.
    assert strip_baked_speaker_label("Zechs: report in.", "Speaker 1") == \
        "Zechs: report in."
    # Colon-free text is untouched.
    assert strip_baked_speaker_label("Speaker of the house", "Speaker 1") == \
        "Speaker of the house"
    # Own-name label is presentation → stripped.
    assert strip_baked_speaker_label("Zechs: report in.", "Zechs") == "report in."
    # Generic placeholder is stripped regardless of the speaker field.
    assert strip_baked_speaker_label("Speaker 7: hello", None) == "hello"
