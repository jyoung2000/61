"""Defensive sanitizer for translated transcripts corrupted by an interrupted run.

A container restart mid-pipeline + flaky reconnects left the stored
``translated_transcript`` a union of source + translated cues, each duplicated
~10× (584 cues, 37% Japanese). The pipeline persists a clean track; this repairs
damage that happens afterwards. These pin: drop source-language relapse, collapse
gross duplication, leave a clean track untouched, never touch a CJK target.
"""

from backend.services.transcript_sanitize import (
    sanitize_translated_transcript,
    merge_transcript_fragments,
)


def test_drops_source_language_cues_for_english_target():
    rows = [
        {"start": 0.0, "end": 2.0, "text": "Hello there.", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 5.0, "text": "え?そんなことも知らないの?", "speaker": "Speaker 1"},
        {"start": 5.0, "end": 7.0, "text": "How are you?", "speaker": "Speaker 1"},
    ]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert changed
    assert [c["text"] for c in clean] == ["Hello there.", "How are you?"]


def test_collapses_substantial_duplicate_to_one():
    # A substantial line (a real sentence) that repeats verbatim is Whisper
    # repetition / corruption — collapse to a single occurrence.
    rows = [{"start": i * 3.0, "end": i * 3 + 2.0,
             "text": "Just for a little while. Really?", "speaker": "Speaker 1"}
            for i in range(10)]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert changed
    assert len(clean) == 1


def test_keeps_two_short_repeats():
    # Short lines can legitimately recur — keep a couple, drop the gross excess.
    rows = [{"start": i * 3.0, "end": i * 3 + 2.0, "text": "Yes?", "speaker": "Speaker 1"}
            for i in range(10)]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert changed
    assert len(clean) == 2


def test_clean_track_is_untouched():
    rows = [{"start": i * 3.0, "end": i * 3 + 2.0,
             "text": f"This is line number {i}.", "speaker": "Speaker 1"}
            for i in range(40)]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert not changed
    assert len(clean) == 40


def test_preserves_a_couple_of_legit_repeats():
    rows = [
        {"start": 0.0, "end": 1.0, "text": "Yeah.", "speaker": "Speaker 1"},
        {"start": 10.0, "end": 11.0, "text": "Something else entirely.", "speaker": "Speaker 1"},
        {"start": 20.0, "end": 21.0, "text": "Yeah.", "speaker": "Speaker 1"},
    ]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert not changed
    assert len(clean) == 3  # two "Yeah." is within the allowed repeat budget


def test_cjk_target_keeps_cjk():
    rows = [
        {"start": 0.0, "end": 2.0, "text": "こんにちは、元気ですか？", "speaker": "Speaker 1"},
        {"start": 2.0, "end": 4.0, "text": "はい、元気です。", "speaker": "Speaker 1"},
    ]
    clean, changed = sanitize_translated_transcript(rows, "ja")
    assert not changed
    assert len(clean) == 2


def test_idempotent():
    rows = [{"start": 0.0, "end": 2.0, "text": "Hi.", "speaker": "Speaker 1"},
            {"start": 2.0, "end": 4.0, "text": "今日も食べないよ", "speaker": "Speaker 1"}]
    rows += [{"start": 4.0 + i, "end": 5.0 + i, "text": "Dup line here.", "speaker": "Speaker 1"}
             for i in range(6)]
    clean1, _ = sanitize_translated_transcript(rows, "en")
    clean2, changed2 = sanitize_translated_transcript(clean1, "en")
    assert not changed2
    assert len(clean1) == len(clean2)


def test_sorts_by_start_time():
    rows = [
        {"start": 9.0, "end": 10.0, "text": "Third.", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 2.0, "text": "First.", "speaker": "Speaker 1"},
        {"start": 5.0, "end": 6.0, "text": "Second.", "speaker": "Speaker 1"},
    ]
    clean, changed = sanitize_translated_transcript(rows, "en")
    assert changed
    assert [c["text"] for c in clean] == ["First.", "Second.", "Third."]


def test_empty_is_safe():
    assert sanitize_translated_transcript([], "en") == ([], False)
    assert sanitize_translated_transcript(None, "en") == ([], False)


# ── merge_transcript_fragments ───────────────────────────────────────────────
# Whisper splits one spoken sentence into 2-4 word cues on acoustic pauses; this
# folds them back when they're clearly one utterance, and ONLY then.

def _cues(*triples, speaker="Speaker 1"):
    """(start, end, text) triples → cue dicts."""
    return [{"start": s, "end": e, "text": t, "speaker": speaker} for s, e, t in triples]


def test_merges_split_sentence():
    rows = _cues((2.0, 2.5, "It's just the"), (2.6, 3.0, "number 21."))
    out, changed = merge_transcript_fragments(rows, "en")
    assert changed
    assert len(out) == 1
    assert out[0]["text"] == "It's just the number 21."
    assert out[0]["start"] == 2.0 and out[0]["end"] == 3.0  # span preserved


def test_merges_multiple_fragments():
    rows = _cues(
        (1.0, 1.4, "But it looks"),
        (1.5, 1.9, "like this is"),
        (2.0, 2.6, "a good birthday."),
    )
    out, changed = merge_transcript_fragments(rows, "en")
    assert changed
    assert [c["text"] for c in out] == ["But it looks like this is a good birthday."]


def test_complete_lines_are_not_merged():
    # Each ends with terminal punctuation → independent utterances.
    rows = _cues(
        (2.0, 2.4, "How old are you now?"),
        (2.6, 3.0, "Is it okay?"),
        (3.2, 3.8, "You're my sister, right?"),
    )
    out, changed = merge_transcript_fragments(rows, "en")
    assert not changed
    assert len(out) == 3


def test_no_merge_across_speakers():
    rows = [
        {"start": 1.0, "end": 1.4, "text": "Can I have a", "speaker": "Speaker 1"},
        {"start": 1.5, "end": 1.9, "text": "turn with you?", "speaker": "Speaker 2"},
    ]
    out, changed = merge_transcript_fragments(rows, "en")
    assert not changed
    assert len(out) == 2


def test_no_merge_across_large_gap():
    # A trailing unfinished fragment must not glue onto the next scene's speech.
    rows = _cues((3.0, 3.5, "so I won't eat"), (60.0, 60.4, "what?"))
    out, changed = merge_transcript_fragments(rows, "en")
    assert not changed
    assert len(out) == 2


def test_respects_length_cap():
    # Two long unfinished cues whose join exceeds the readable cap stay split.
    a = "this is a fairly long unfinished clause that keeps going and going"
    b = "and here is even more text that would blow well past the length cap"
    rows = _cues((1.0, 3.0, a), (3.1, 5.0, b))
    out, _ = merge_transcript_fragments(rows, "en")
    assert len(out) == 2


def test_respects_duration_cap():
    # Same-speaker continuation but the combined span is too long to show as one.
    rows = _cues((0.0, 4.0, "this clause keeps"), (4.2, 12.0, "going for ages"))
    out, _ = merge_transcript_fragments(rows, "en")
    assert len(out) == 2


def test_marker_is_a_boundary():
    rows = [
        {"start": 0.0, "end": 2.0, "text": "[♪ music ♪]", "speaker": "Speaker 1"},
        {"start": 2.1, "end": 2.5, "text": "I'll have", "speaker": "Speaker 1"},
        {"start": 2.6, "end": 3.0, "text": "some cake.", "speaker": "Speaker 1"},
    ]
    out, changed = merge_transcript_fragments(rows, "en")
    assert changed
    assert [c["text"] for c in out] == ["[♪ music ♪]", "I'll have some cake."]


def test_cjk_target_is_untouched():
    rows = _cues((0.0, 1.0, "これは"), (1.1, 2.0, "テストです"))
    out, changed = merge_transcript_fragments(rows, "ja")
    assert not changed
    assert len(out) == 2


def test_merge_is_idempotent():
    # The critical property: applying twice == applying once (no read-time churn).
    rows = _cues(
        (2.0, 2.4, "How old are you now?"),
        (3.0, 3.4, "It's just the"),
        (3.5, 3.9, "number 21."),
        (4.0, 4.4, "Can I have a"),
        (4.5, 4.9, "turn with you?"),
        (10.0, 10.4, "a long unfinished tail clause that never gets its ending here"),
        (10.5, 12.0, "and just keeps on extending past every cap we set for it"),
    )
    once, c1 = merge_transcript_fragments(rows, "en")
    twice, c2 = merge_transcript_fragments(once, "en")
    assert c1
    assert not c2  # second pass is a no-op
    assert [c["text"] for c in once] == [c["text"] for c in twice]


def test_clean_sentences_unchanged():
    rows = _cues(*[(float(i), i + 0.5, f"This is line number {i}.") for i in range(20)])
    out, changed = merge_transcript_fragments(rows, "en")
    assert not changed
    assert len(out) == 20


def test_merge_concatenates_word_timestamps():
    rows = [
        {"start": 2.0, "end": 2.5, "text": "It's just the", "speaker": "Speaker 1",
         "words": [{"word": "It's", "start": 2.0, "end": 2.2}]},
        {"start": 2.6, "end": 3.0, "text": "number 21.", "speaker": "Speaker 1",
         "words": [{"word": "number", "start": 2.6, "end": 2.8}]},
    ]
    out, changed = merge_transcript_fragments(rows, "en")
    assert changed and len(out) == 1
    assert [w["word"] for w in out[0]["words"]] == ["It's", "number"]


def test_merge_empty_is_safe():
    assert merge_transcript_fragments([], "en") == ([], False)
    assert merge_transcript_fragments(None, "en") == ([], False)
