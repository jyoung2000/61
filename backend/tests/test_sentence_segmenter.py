"""Acceptance tests for Task 4 — sentence-aware resegmentation."""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.sentence_segmenter import resegment_by_sentence


def _seg(start, end, text, speaker="Speaker 1", words=None):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker, words=words)


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def test_30s_block_splits_into_sentences():
    # A single 30 s block carrying three sentences worth of word timing.
    words = _words([
        (0.0, 1.0, "Hello"), (1.0, 2.0, "everyone."),
        (10.0, 11.0, "Welcome"), (11.0, 12.0, "back."),
        (20.0, 21.0, "Let's"), (21.0, 22.0, "begin."),
    ])
    seg = _seg(0.0, 30.0, "Hello everyone. Welcome back. Let's begin.", words=words)
    out = resegment_by_sentence([seg])
    assert len(out) == 3
    # Per-sentence timing is taken from the word timestamps.
    assert out[0].text == "Hello everyone."
    assert out[1].text == "Welcome back."
    assert out[2].text == "Let's begin."
    # Monotonic, non-overlapping.
    for i in range(len(out) - 1):
        assert out[i].start <= out[i].end <= out[i + 1].start
    assert abs(out[0].end - 2.0) < 1e-6
    assert abs(out[1].start - 10.0) < 1e-6


def test_speaker_boundaries_not_merged():
    a_words = _words([(0.0, 1.0, "Question"), (1.0, 2.0, "here.")])
    b_words = _words([(2.5, 3.5, "Answer"), (3.5, 4.5, "there.")])
    segs = [
        _seg(0.0, 2.0, "Question here.", speaker="Speaker 1", words=a_words),
        _seg(2.5, 4.5, "Answer there.", speaker="Speaker 2", words=b_words),
    ]
    out = resegment_by_sentence(segs)
    # Two speakers, never glued together.
    assert len(out) == 2
    assert out[0].speaker == "Speaker 1"
    assert out[1].speaker == "Speaker 2"


def test_same_speaker_merge_then_resplit():
    # Two adjacent same-speaker windows that each hold a fragment; merging
    # then sentence-splitting should yield clean per-sentence segments.
    w1 = _words([(0.0, 0.5, "The"), (0.5, 1.0, "first"), (1.0, 1.5, "sentence.")])
    w2 = _words([(2.0, 2.5, "The"), (2.5, 3.0, "second"), (3.0, 3.5, "one.")])
    segs = [
        _seg(0.0, 1.5, "The first sentence.", words=w1),
        _seg(2.0, 3.5, "The second one.", words=w2),
    ]
    out = resegment_by_sentence(segs)
    assert len(out) == 2
    assert out[0].text == "The first sentence."
    assert out[1].text == "The second one."
    assert out[1].start >= out[0].end


def test_cjk_punctuation_handled():
    words = _words([
        (0.0, 1.0, "こんにちは。"),
        (1.0, 2.0, "元気ですか。"),
    ])
    seg = _seg(0.0, 5.0, "こんにちは。元気ですか。", speaker="Speaker 1", words=words)
    out = resegment_by_sentence([seg])
    assert len(out) == 2
    assert out[0].text == "こんにちは。"
    assert out[1].text == "元気ですか。"
    # CJK joins without spaces.
    assert " " not in out[0].text


def test_single_sentence_unchanged():
    words = _words([(0.0, 1.0, "Just"), (1.0, 2.0, "one"), (2.0, 3.0, "sentence.")])
    seg = _seg(0.0, 3.0, "Just one sentence.", words=words)
    out = resegment_by_sentence([seg])
    assert len(out) == 1
    assert out[0].text == "Just one sentence."


def test_empty_unchanged():
    assert resegment_by_sentence([]) == []
