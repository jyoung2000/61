"""Defensive sanitizer for translated transcripts corrupted by an interrupted run.

A container restart mid-pipeline + flaky reconnects left the stored
``translated_transcript`` a union of source + translated cues, each duplicated
~10× (584 cues, 37% Japanese). The pipeline persists a clean track; this repairs
damage that happens afterwards. These pin: drop source-language relapse, collapse
gross duplication, leave a clean track untouched, never touch a CJK target.
"""

from backend.services.transcript_sanitize import sanitize_translated_transcript


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
