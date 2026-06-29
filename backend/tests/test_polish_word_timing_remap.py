"""P1 — polish preserves word-level timestamps via re-mapping.

The polisher used to NULL a segment's word timestamps whenever it changed the
text. Because polish punctuates almost every segment, that wiped nearly all word
timing and forced the sentence segmenter onto its char-proportional fallback.
Now the original timestamps are re-mapped onto the edited text.
"""

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services.transcript_polisher import (
    _emit_segment, _remap_words_onto_text,
)
from backend.services.sentence_segmenter import resegment_by_sentence


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def _seg():
    spec = [
        (0.0, 0.5, "hello"), (0.6, 1.0, "there"), (1.2, 1.8, "everyone"),
        (2.0, 2.4, "welcome"), (2.5, 3.0, "back"),
    ]
    return TranscriptSegment(
        start=0.0, end=3.0, text="hello there everyone welcome back",
        speaker="Speaker 1", words=_words(spec))


def test_punctuation_only_edit_keeps_word_timing():
    seg = _seg()
    # Punctuation-only edit (the dominant polish case).
    edited = "Hello there, everyone. Welcome back."
    out = _emit_segment(seg, edited)
    assert out.text == edited
    assert out.words, "word timestamps were wiped on a punctuation-only edit"
    # One timed token per original word survived.
    assert len(out.words) == 5
    # Original start/end carried verbatim onto the matching tokens.
    starts = [w.start for w in out.words]
    ends = [w.end for w in out.words]
    assert starts == [0.0, 0.6, 1.2, 2.0, 2.5]
    assert ends == [0.5, 1.0, 1.8, 2.4, 3.0]
    # Monotonic.
    for i in range(len(out.words) - 1):
        assert out.words[i].end <= out.words[i + 1].start + 1e-9


def test_terminator_rides_on_token_for_segmenter():
    # The polish-added "." must ride on its word so the segmenter can split.
    seg = _seg()
    out = _emit_segment(seg, "Hello there, everyone. Welcome back.")
    # Feed straight into the resegmenter — it must take the WORD-timed path
    # (not the char-proportional fallback) and split at the real timestamp.
    res = resegment_by_sentence([out])
    assert len(res) == 2
    assert res[0].text == "Hello there, everyone."
    assert res[1].text == "Welcome back."
    # Boundary comes from the word timing (everyone ends at 1.8), not a
    # char-proportional guess.
    assert abs(res[0].end - 1.8) < 1e-6
    assert abs(res[1].start - 2.0) < 1e-6


def test_substituted_word_keeps_text_and_interpolates_timing():
    seg = _seg()
    # One word substituted (no original counterpart) — text must stay complete,
    # its timing interpolated between neighbours, still monotonic.
    edited = "hello there friends welcome back"
    out = _emit_segment(seg, edited)
    assert " ".join(w.word for w in out.words) == edited
    for i in range(len(out.words) - 1):
        assert out.words[i].end <= out.words[i + 1].start + 1e-9
    assert out.words[0].start == 0.0
    assert out.words[-1].end <= 3.0 + 1e-9


def test_dict_shape_remaps_too():
    seg = _seg().model_dump()
    out = _emit_segment(seg, "Hello there, everyone. Welcome back.")
    assert out["words"], "dict path did not remap words"
    assert out["words"][0]["start"] == 0.0


def test_flag_off_restores_nulling(monkeypatch):
    monkeypatch.setattr(settings, "POLISH_REMAP_WORD_TIMESTAMPS", False)
    out = _emit_segment(_seg(), "Hello there, everyone. Welcome back.")
    assert out.words in (None, [])


def test_low_confidence_drops_timing():
    seg = _seg()
    # A complete rewrite with no shared tokens → confidence 0 → words cleared.
    out = _emit_segment(seg, "totally different sentence content here now")
    assert out.words in (None, [])


def test_over_90pct_segments_retain_word_timing():
    # A batch of punctuation-only edits: >90% must keep monotonic word timing.
    segs = [_seg() for _ in range(20)]
    edits = ["Hello there, everyone. Welcome back."] * 20
    retained = 0
    for s, e in zip(segs, edits):
        out = _emit_segment(s, e)
        if out.words and all(
            out.words[i].end <= out.words[i + 1].start + 1e-9
            for i in range(len(out.words) - 1)
        ):
            retained += 1
    assert retained / len(segs) > 0.9


def test_timing_monotonic_helper_direct():
    seg = _seg()
    words, conf = _remap_words_onto_text(
        seg.words, "Hello there, everyone. Welcome back.", 0.0, 3.0)
    assert conf == 1.0
    assert all(words[i]["end"] <= words[i + 1]["start"] + 1e-9
               for i in range(len(words) - 1))
