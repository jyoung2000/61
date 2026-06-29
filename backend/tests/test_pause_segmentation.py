"""P1 — pause-based (acoustic) sentence segmentation.

Local-mode (and especially CJK) transcripts often arrive with NO sentence
punctuation, so terminator-based resegmentation leaves the whole block as one
wall-of-text cue. The pause-based splitter breaks such word-timed blocks at
inter-word silence — language-agnostic, no model — with word-accurate
boundaries.
"""

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services.sentence_segmenter import (
    resegment_by_sentence, _split_words_by_pause,
)


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def test_cjk_no_punctuation_splits_on_pauses():
    # Zero punctuation, two phrases separated by a ~0.6 s silence.
    spec = [
        (0.0, 0.4, "今日"), (0.4, 0.8, "は"), (0.8, 1.4, "いい天気"),
        # 0.7 s pause here ↓
        (2.1, 2.5, "明日"), (2.5, 2.9, "は"), (2.9, 3.5, "雨です"),
    ]
    seg = TranscriptSegment(
        start=0.0, end=3.5, text="今日はいい天気明日は雨です",
        speaker="Speaker 1", words=_words(spec))
    out = resegment_by_sentence([seg])
    assert len(out) == 2, f"pause split did not fire (got {len(out)})"
    # Word-accurate boundaries (not char-proportional).
    assert abs(out[0].end - 1.4) < 1e-6
    assert abs(out[1].start - 2.1) < 1e-6
    # CJK joins without spaces and text is preserved.
    assert out[0].text == "今日はいい天気"
    assert out[1].text == "明日は雨です"
    assert " " not in out[0].text


def test_no_pause_no_terminator_stays_whole_until_max_cue():
    # Continuous speech, no pauses, no punctuation, under the max-cue guard.
    spec = [(i * 0.5, i * 0.5 + 0.4, "word%d" % i) for i in range(6)]  # ~3 s
    seg = TranscriptSegment(
        start=0.0, end=3.0, text=" ".join("word%d" % i for i in range(6)),
        speaker="Speaker 1", words=_words(spec))
    out = resegment_by_sentence([seg])
    assert len(out) == 1


def test_long_block_split_by_max_cue_guard(monkeypatch):
    monkeypatch.setattr(settings, "SENTENCE_SPLIT_MAX_CUE_MS", 3000)
    # No pauses, no terminators, but 6 s long → must break at the max-cue guard.
    spec = [(i * 0.5, i * 0.5 + 0.5, "w%d" % i) for i in range(12)]  # 6 s
    seg = TranscriptSegment(
        start=0.0, end=6.0, text=" ".join("w%d" % i for i in range(12)),
        speaker="Speaker 1", words=_words(spec))
    out = resegment_by_sentence([seg])
    assert len(out) >= 2


def test_terminators_still_respected_when_present():
    spec = [
        (0.0, 0.5, "Hello"), (0.6, 1.0, "there."),
        (1.1, 1.6, "Welcome"), (1.7, 2.2, "back."),
    ]
    seg = TranscriptSegment(
        start=0.0, end=2.2, text="Hello there. Welcome back.",
        speaker="Speaker 1", words=_words(spec))
    out = resegment_by_sentence([seg])
    assert len(out) == 2
    assert out[0].text == "Hello there."
    assert out[1].text == "Welcome back."


def test_flag_off_keeps_punctuationless_block_whole(monkeypatch):
    monkeypatch.setattr(settings, "SENTENCE_SPLIT_PAUSE_ENABLED", False)
    spec = [
        (0.0, 0.4, "今日"), (0.4, 0.8, "は"), (0.8, 1.4, "いい天気"),
        (2.1, 2.5, "明日"), (2.5, 2.9, "は"), (2.9, 3.5, "雨です"),
    ]
    seg = TranscriptSegment(
        start=0.0, end=3.5, text="今日はいい天気明日は雨です",
        speaker="Speaker 1", words=_words(spec))
    out = resegment_by_sentence([seg])
    assert len(out) == 1


def test_split_words_by_pause_monotonic():
    spec = [
        (0.0, 0.4, "a"), (0.5, 0.9, "b"),
        (2.0, 2.4, "c"), (2.5, 2.9, "d"),
    ]
    groups = _split_words_by_pause(_words(spec), pause_s=0.4, max_cue_s=8.0)
    assert len(groups) == 2
    assert [w.word for w in groups[0]] == ["a", "b"]
    assert [w.word for w in groups[1]] == ["c", "d"]
