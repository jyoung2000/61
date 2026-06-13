"""Segmenter: region coverage, sentence splitting, pause placement, hashing."""

from __future__ import annotations

import pytest

from inflect.document.segmenter import (
    MAX_SEG_CHARS,
    iter_regions,
    segment_document,
    segment_hash,
)
from inflect.document.spans import Document, Inflection


def _doc(text: str) -> Document:
    return Document(text=text, voice_profile_id="voice-1")


# --------------------------------------------------------------------------- #
# Region coverage
# --------------------------------------------------------------------------- #
def test_regions_cover_whole_text_no_spans():
    doc = _doc("Hello world.")
    regions = iter_regions(doc)
    assert regions == [(0, len(doc.text), doc.default_inflection)]


def test_regions_fill_gaps_with_default():
    doc = _doc("abcdefghij")  # 10 chars
    doc.apply_inflection(3, 6, Inflection(emo_text="mid"))
    regions = iter_regions(doc)
    bounds = [(s, e) for s, e, _ in regions]
    assert bounds == [(0, 3), (3, 6), (6, 10)]
    assert regions[1][2].emo_text == "mid"
    assert regions[0][2] is doc.default_inflection
    assert regions[2][2] is doc.default_inflection


def test_regions_span_at_start_and_end():
    doc = _doc("abcdefghij")
    doc.apply_inflection(0, 4, Inflection(emo_text="a"))
    doc.apply_inflection(7, 10, Inflection(emo_text="b"))
    bounds = [(s, e) for s, e, _ in iter_regions(doc)]
    assert bounds == [(0, 4), (4, 7), (7, 10)]


# --------------------------------------------------------------------------- #
# Job production & boundaries
# --------------------------------------------------------------------------- #
def test_one_job_per_short_region():
    doc = _doc("First part. Second part.")
    doc.apply_inflection(0, 11, Inflection(emo_text="x"))  # "First part."
    jobs = segment_document(doc, engine="chatterbox")
    assert len(jobs) == 2
    assert jobs[0].text.strip() == "First part."
    assert jobs[1].text.strip() == "Second part."
    assert jobs[0].char_start == 0
    assert all(j.voice_profile_id == "voice-1" for j in jobs)
    assert all(j.engine == "chatterbox" for j in jobs)


def test_blank_segments_skipped():
    doc = _doc("Hello.       \n\n   World.")
    jobs = segment_document(doc, engine="chatterbox")
    texts = [j.text.strip() for j in jobs]
    assert "" not in texts
    assert any("Hello" in t for t in texts)
    assert any("World" in t for t in texts)


def test_char_offsets_reference_source_text():
    doc = _doc("Alpha beta gamma.")
    doc.apply_inflection(6, 10, Inflection(emo_text="beta"))  # "beta"
    jobs = segment_document(doc, engine="chatterbox")
    beta_job = next(j for j in jobs if j.text.strip() == "beta")
    assert doc.text[beta_job.char_start:beta_job.char_end] == "beta"


# --------------------------------------------------------------------------- #
# Sentence splitting on long runs
# --------------------------------------------------------------------------- #
def test_long_run_is_split_at_sentences():
    sentence = "This is a fairly ordinary sentence with some length. "
    text = sentence * 20  # well over 400 chars, ~1060
    doc = _doc(text)
    jobs = segment_document(doc, engine="chatterbox")
    assert len(jobs) > 1
    for j in jobs:
        assert len(j.text) <= MAX_SEG_CHARS + 5  # small slack for trailing space
    # Reassembling stripped chunks reproduces the stripped sentences.
    joined = " ".join(j.text.strip() for j in jobs)
    assert joined.replace("  ", " ").startswith("This is a fairly ordinary")


def test_short_span_inside_long_doc_stays_single():
    long_tail = "Filler sentence number {}. ".format
    body = "".join(long_tail(i) for i in range(40))
    doc = _doc("Quick angry bit. " + body)
    doc.apply_inflection(0, 16, Inflection(emotion_vector=[0, 0.9] + [0] * 6))
    jobs = segment_document(doc, engine="chatterbox")
    angry_jobs = [j for j in jobs if j.char_start < 16]
    assert len(angry_jobs) == 1
    assert angry_jobs[0].text.strip() == "Quick angry bit."


def test_oversized_single_sentence_falls_back_to_clauses():
    # One sentence, no terminators until the end, with clause commas.
    clause = "and then something else happened, "
    text = clause * 30 + "the end."  # > 400, single sentence
    doc = _doc(text)
    jobs = segment_document(doc, engine="chatterbox")
    assert len(jobs) > 1
    assert all(len(j.text) <= 600 for j in jobs)


# --------------------------------------------------------------------------- #
# Pause placement
# --------------------------------------------------------------------------- #
def test_pause_only_on_last_subsegment_of_split_region():
    sentence = "A medium sentence that is reasonably wordy here. "
    region_text = sentence * 15  # forces a split
    doc = _doc(region_text)
    doc.default_inflection = Inflection(pause_after_ms=500)
    jobs = segment_document(doc, engine="chatterbox")
    assert len(jobs) > 1
    assert all(j.pause_after_ms == 0 for j in jobs[:-1])
    assert jobs[-1].pause_after_ms == 500


def test_pause_on_styled_span():
    doc = _doc("Hello there. Goodbye now.")
    doc.apply_inflection(0, 12, Inflection(emo_text="x", pause_after_ms=300))
    jobs = segment_document(doc, engine="chatterbox")
    first = next(j for j in jobs if "Hello" in j.text)
    assert first.pause_after_ms == 300


def test_pause_on_blank_region_folds_onto_previous():
    # A styled, whitespace-only span carrying a pause should not create a job;
    # its pause attaches to the preceding real segment.
    doc = _doc("Word one.    Word two.")
    # Style the run of spaces between the sentences with a pause.
    gap_start = doc.text.index("    ")
    doc.apply_inflection(gap_start, gap_start + 4, Inflection(pause_after_ms=750))
    jobs = segment_document(doc, engine="chatterbox")
    assert all(j.text.strip() for j in jobs)  # no blank jobs
    prev = next(j for j in jobs if "Word one" in j.text)
    assert prev.pause_after_ms == 750


# --------------------------------------------------------------------------- #
# Hashing & caching semantics
# --------------------------------------------------------------------------- #
def test_identical_docs_identical_hashes():
    doc1 = _doc("Hello world. This is text.")
    doc2 = _doc("Hello world. This is text.")
    h1 = [j.hash for j in segment_document(doc1, engine="indextts2")]
    h2 = [j.hash for j in segment_document(doc2, engine="indextts2")]
    assert h1 == h2


def test_emotion_change_changes_hash():
    doc = _doc("Hello world.")
    base = segment_document(doc, engine="indextts2")[0].hash
    doc.default_inflection = Inflection(emotion_vector=[0.9] + [0] * 7)
    changed = segment_document(doc, engine="indextts2")[0].hash
    assert base != changed


def test_pause_change_does_not_change_hash():
    doc = _doc("Hello world.")
    base = segment_document(doc, engine="indextts2")[0].hash
    doc.default_inflection = Inflection(pause_after_ms=999)
    same = segment_document(doc, engine="indextts2")[0]
    assert same.hash == base
    assert same.pause_after_ms == 999


def test_engine_change_changes_hash():
    doc = _doc("Hello world.")
    a = segment_document(doc, engine="indextts2")[0].hash
    b = segment_document(doc, engine="chatterbox")[0].hash
    assert a != b


def test_engine_params_change_changes_hash():
    doc = _doc("Hello world.")
    a = segment_document(doc, engine="chatterbox", engine_params={"cfg_weight": 0.5})[0].hash
    b = segment_document(doc, engine="chatterbox", engine_params={"cfg_weight": 0.9})[0].hash
    assert a != b


def test_per_span_engine_override():
    doc = _doc("Calm intro. FURIOUS PART.")
    doc.apply_inflection(12, 25, Inflection(emo_text="rage", engine="fish"))
    jobs = segment_document(doc, engine="indextts2")
    intro = next(j for j in jobs if "Calm" in j.text)
    rage = next(j for j in jobs if "FURIOUS" in j.text)
    assert intro.engine == "indextts2"
    assert rage.engine == "fish"


def test_segment_hash_helper_matches_job():
    doc = _doc("Hello world.")
    job = segment_document(doc, engine="indextts2", engine_params={"k": 1})[0]
    expected = segment_hash(job.text, job.inflection, "voice-1", "indextts2", {"k": 1})
    assert job.hash == expected
