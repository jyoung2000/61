"""P3 — timing-provenance observability helper."""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.sentence_segmenter import timing_provenance_report


def _seg(text, words=None):
    return TranscriptSegment(start=0.0, end=1.0, text=text,
                             speaker="Speaker 1", words=words)


def test_counts_word_timed_vs_proportional():
    segs = [
        _seg("a", words=[WordTimestamp(start=0.0, end=1.0, word="a")]),
        _seg("b", words=[WordTimestamp(start=0.0, end=1.0, word="b")]),
        _seg("c", words=None),
        _seg("   ", words=None),   # blank — ignored
    ]
    rep = timing_provenance_report(segs)
    assert rep["total"] == 3
    assert rep["word_timed"] == 2
    assert rep["proportional"] == 1
    assert rep["pct_word_timed"] == round(2 / 3 * 100, 1)


def test_empty_is_fully_word_timed():
    rep = timing_provenance_report([])
    assert rep["total"] == 0
    assert rep["pct_word_timed"] == 100.0


def test_accepts_dict_segments():
    segs = [
        {"text": "x", "words": [{"start": 0.0, "end": 1.0, "word": "x"}]},
        {"text": "y", "words": []},
    ]
    rep = timing_provenance_report(segs)
    assert rep["word_timed"] == 1
    assert rep["proportional"] == 1
