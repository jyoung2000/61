"""Span algebra: apply / clear / remap and Inflection invariants.

These mirror the cases the spec explicitly calls out: insert inside, overlap
left, overlap right, engulf, exact match, adjacent -- plus offset remapping on
text edits.
"""

from __future__ import annotations

import pytest

from inflect.document.spans import (
    EMOTIONS,
    Document,
    Inflection,
    InflectionSpan,
    _carve_out,
    _remap_point,
)


def _doc(text: str) -> Document:
    return Document(text=text)


def _styled(start: int, end: int, *, speed: float = 1.2) -> InflectionSpan:
    return InflectionSpan(start, end, Inflection(speed=speed))


def _ranges(doc: Document) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in doc.sorted_spans()]


# --------------------------------------------------------------------------- #
# Inflection validation & serialization
# --------------------------------------------------------------------------- #
def test_emotion_vector_wrong_length_rejected():
    with pytest.raises(ValueError):
        Inflection(emotion_vector=[0.1, 0.2, 0.3])


def test_emotion_vector_clamped():
    inf = Inflection(emotion_vector=[2.0, -1.0] + [0.5] * 6)
    assert inf.emotion_vector[0] == 1.0
    assert inf.emotion_vector[1] == 0.0


def test_speed_and_alpha_clamped():
    assert Inflection(speed=99).speed == 1.5
    assert Inflection(speed=0.01).speed == 0.5
    assert Inflection(emo_alpha=5).emo_alpha == 1.0


def test_audio_signature_excludes_pause_and_engine():
    a = Inflection(speed=1.1, pause_after_ms=0, engine=None)
    b = Inflection(speed=1.1, pause_after_ms=900, engine="fish")
    assert a.audio_signature() == b.audio_signature()


def test_audio_signature_reflects_emotion():
    a = Inflection(emotion_vector=[0.0] * 8)
    b = Inflection(emotion_vector=[1.0] + [0.0] * 7)
    assert a.audio_signature() != b.audio_signature()


def test_is_neutral():
    assert Inflection().is_neutral()
    assert not Inflection(speed=1.2).is_neutral()
    assert not Inflection(emo_text="whispering").is_neutral()
    assert not Inflection(emotion_vector=[0.4] + [0.0] * 7).is_neutral()


def test_inflection_round_trip():
    inf = Inflection(
        emotion_vector=[0.1 * i for i in range(8)],
        emo_text="hesitant",
        emo_alpha=0.6,
        speed=0.9,
        pause_after_ms=250,
        engine="hybrid",
    )
    assert Inflection.from_dict(inf.to_dict()) == inf
    assert len(EMOTIONS) == 8


# --------------------------------------------------------------------------- #
# apply_inflection -- the six required intersection cases
# --------------------------------------------------------------------------- #
def test_apply_into_empty_document():
    doc = _doc("hello world this is text")
    doc.apply_inflection(0, 5, Inflection(speed=1.3))
    assert _ranges(doc) == [(0, 5)]


def test_apply_insert_inside_splits_existing():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(12, 15, Inflection(emo_text="angry"))
    assert _ranges(doc) == [(10, 12), (12, 15), (15, 20)]
    # The middle span is the new one.
    assert doc.span_at(13).inflection.emo_text == "angry"
    # The flanking pieces retain the original styling.
    assert doc.span_at(11).inflection.speed == pytest.approx(1.2)


def test_apply_overlap_left_truncates():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(5, 15, Inflection(emo_text="new"))
    assert _ranges(doc) == [(5, 15), (15, 20)]
    assert doc.span_at(6).inflection.emo_text == "new"
    assert doc.span_at(17).inflection.speed == pytest.approx(1.2)


def test_apply_overlap_right_truncates():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(15, 25, Inflection(emo_text="new"))
    assert _ranges(doc) == [(10, 15), (15, 25)]
    assert doc.span_at(11).inflection.speed == pytest.approx(1.2)
    assert doc.span_at(20).inflection.emo_text == "new"


def test_apply_engulf_removes_old():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(5, 25, Inflection(emo_text="big"))
    assert _ranges(doc) == [(5, 25)]
    assert doc.span_at(12).inflection.emo_text == "big"


def test_apply_exact_match_replaces():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(10, 20, Inflection(emo_text="replaced"))
    assert _ranges(doc) == [(10, 20)]
    assert doc.span_at(12).inflection.emo_text == "replaced"


def test_apply_adjacent_left_and_right_keep_both():
    doc = _doc("x" * 40)
    doc.spans = [_styled(10, 20)]
    doc.apply_inflection(20, 30, Inflection(emo_text="right"))
    doc.apply_inflection(0, 10, Inflection(emo_text="left"))
    assert _ranges(doc) == [(0, 10), (10, 20), (20, 30)]


def test_apply_engulfing_multiple_existing():
    doc = _doc("x" * 50)
    doc.spans = [_styled(5, 10), _styled(15, 20), _styled(25, 30)]
    doc.apply_inflection(8, 27, Inflection(emo_text="sweep"))
    # First span truncated on the right, middle engulfed, last truncated left.
    assert _ranges(doc) == [(5, 8), (8, 27), (27, 30)]


def test_apply_empty_selection_is_noop():
    doc = _doc("x" * 20)
    doc.spans = [_styled(5, 10)]
    assert doc.apply_inflection(7, 7, Inflection()) is None
    assert _ranges(doc) == [(5, 10)]


def test_apply_reversed_selection_normalized():
    doc = _doc("x" * 20)
    doc.apply_inflection(15, 5, Inflection(emo_text="rev"))
    assert _ranges(doc) == [(5, 15)]


def test_apply_clamps_to_text_length():
    doc = _doc("x" * 10)
    doc.apply_inflection(5, 999, Inflection())
    assert _ranges(doc) == [(5, 10)]


# --------------------------------------------------------------------------- #
# clear_inflection
# --------------------------------------------------------------------------- #
def test_clear_inside_splits():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.clear_inflection(12, 15)
    assert _ranges(doc) == [(10, 12), (15, 20)]


def test_clear_engulf_removes():
    doc = _doc("x" * 30)
    doc.spans = [_styled(10, 20)]
    doc.clear_inflection(5, 25)
    assert _ranges(doc) == []


def test_carve_out_pure_function_disjoint():
    spans = [InflectionSpan(0, 5, Inflection()), InflectionSpan(10, 15, Inflection())]
    out = _carve_out(spans, 6, 9)
    assert [(s.start, s.end) for s in out] == [(0, 5), (10, 15)]


# --------------------------------------------------------------------------- #
# Offset remap on edits (QTextDocument.contentsChange semantics)
# --------------------------------------------------------------------------- #
def _remap(span: tuple[int, int], pos: int, removed: int, added: int, new_text_len: int):
    doc = Document(text="y" * new_text_len, spans=[InflectionSpan(*span, Inflection())])
    doc.remap_for_edit(pos, removed, added)
    return _ranges(doc)


def test_remap_insert_before_shifts():
    assert _remap((10, 20), pos=5, removed=0, added=3, new_text_len=33) == [(13, 23)]


def test_remap_insert_at_start_boundary_keeps_text_outside():
    # Typing exactly at a span's start pushes the span right (inserted text is
    # unstyled / before the span).
    assert _remap((10, 20), pos=10, removed=0, added=3, new_text_len=33) == [(13, 23)]


def test_remap_insert_inside_grows_span():
    assert _remap((10, 20), pos=15, removed=0, added=3, new_text_len=33) == [(10, 23)]


def test_remap_insert_at_end_boundary_keeps_text_outside():
    assert _remap((10, 20), pos=20, removed=0, added=3, new_text_len=33) == [(10, 20)]


def test_remap_delete_before_shifts_left():
    assert _remap((10, 20), pos=2, removed=3, added=0, new_text_len=27) == [(7, 17)]


def test_remap_delete_inside_shrinks():
    assert _remap((10, 20), pos=12, removed=3, added=0, new_text_len=27) == [(10, 17)]


def test_remap_delete_covering_span_drops_it():
    assert _remap((10, 20), pos=8, removed=15, added=0, new_text_len=15) == []


def test_remap_delete_overlapping_start():
    assert _remap((10, 20), pos=5, removed=8, added=0, new_text_len=22) == [(5, 12)]


def test_remap_replace_combination():
    # Replace 4 chars at pos 12 with 2 chars: span [10,20) -> end shifts by -2.
    assert _remap((10, 20), pos=12, removed=4, added=2, new_text_len=28) == [(10, 18)]


def test_remap_multiple_spans_preserve_order_and_nonoverlap():
    doc = Document(
        text="z" * 33,
        spans=[InflectionSpan(0, 5, Inflection()), InflectionSpan(10, 15, Inflection())],
    )
    doc.remap_for_edit(position=3, chars_removed=0, chars_added=3)
    rngs = _ranges(doc)
    # Insertion at pos 3 is strictly inside the first span (grows it to (0,8))
    # and entirely before the second span (shifts it by +3).
    assert rngs == [(0, 8), (13, 18)]
    # Non-overlapping & sorted invariant holds.
    for (a1, b1), (a2, b2) in zip(rngs, rngs[1:]):
        assert b1 <= a2


def test_remap_point_unit():
    # Pure boundary math sanity checks.
    assert _remap_point(10, 5, 0, 3, is_start=True) == 13
    assert _remap_point(10, 15, 0, 3, is_start=False) == 10
    assert _remap_point(20, 15, 0, 3, is_start=False) == 23


# --------------------------------------------------------------------------- #
# Document queries / serialization
# --------------------------------------------------------------------------- #
def test_span_at_caret_semantics():
    doc = _doc("x" * 20)
    doc.spans = [_styled(5, 10)]
    assert doc.span_at(5) is not None
    assert doc.span_at(9) is not None
    assert doc.span_at(10) is None  # exclusive end
    assert doc.span_at(4) is None


def test_inflection_for_falls_back_to_default():
    doc = Document(text="x" * 20, default_inflection=Inflection(speed=0.8))
    doc.spans = [_styled(5, 10, speed=1.4)]
    assert doc.inflection_for(7).speed == pytest.approx(1.4)
    assert doc.inflection_for(0).speed == pytest.approx(0.8)


def test_set_pause_after_creates_or_updates_span():
    doc = _doc("Hello world.")
    span = doc.set_pause_after(len(doc.text), 400)
    assert span.inflection.pause_after_ms == 400
    assert span.end == len(doc.text)


def test_document_round_trip():
    doc = Document(
        text="Hello there. General Kenobi.",
        voice_profile_id="abc-123",
        default_inflection=Inflection(speed=0.95),
    )
    doc.apply_inflection(0, 5, Inflection(emotion_vector=[0.7] + [0.0] * 7), color_idx=2)
    restored = Document.from_dict(doc.to_dict())
    assert restored.text == doc.text
    assert restored.voice_profile_id == "abc-123"
    assert _ranges(restored) == _ranges(doc)
    assert restored.spans[0].color_idx == 2
    assert restored.default_inflection.speed == pytest.approx(0.95)
