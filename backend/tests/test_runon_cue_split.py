"""YouTube-style one-thought-per-cue segmentation: split_run_on_cues.

Official subs put one sentence/thought per cue; Whisper + 1:1 translation can
cram 3-4 sentences into an 11-second cue. The splitter breaks those at
sentence boundaries with proportional time allocation — and must NEVER touch
short cues, markers, CJK targets, or cues too brief to split readably.
"""

from backend.services.transcript_sanitize import (
    merge_transcript_fragments,
    split_run_on_cues,
)


def _cue(start, end, text, **kw):
    return {"start": float(start), "end": float(end), "text": text,
            "speaker": kw.get("speaker", "Speaker 1"),
            "words": kw.get("words")}


def test_splits_the_run54_capsule_runon():
    # The real shipped cue: 4 sentences over ~11s. One-thought-per-cue now
    # gives each finished sentence its own cue (YouTube cadence) → 4 pieces.
    segs = [_cue(355.0, 366.0,
                 "The capsule has altered its course. Does it have suicidal "
                 "tendencies? If it burns out, even secrets can be protected. "
                 "I suppose that's about right.")]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and 3 <= len(out) <= 4
    # Chronological, gap-free, same overall window.
    assert out[0]["start"] == 355.0 and out[-1]["end"] == 366.0
    for a, b in zip(out, out[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6
        assert a["end"] > a["start"]
    # Every piece keeps the speaker and ends at a sentence boundary.
    joined = " ".join(p["text"] for p in out)
    assert joined == segs[0]["text"]
    for p in out:
        assert p["speaker"] == "Speaker 1"
        assert p["text"].rstrip()[-1] in ".!?…"


def test_time_allocation_is_proportional():
    segs = [_cue(0.0, 10.0, "A" * 100 + ". " + "Second sentence here to split off.")]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and len(out) == 2
    # The long first sentence gets the lion's share of the window.
    assert (out[0]["end"] - out[0]["start"]) > (out[1]["end"] - out[1]["start"])


def test_never_touches_short_cues_markers_or_cjk():
    # A single short thought (one sentence, under one line) stays whole.
    short = [_cue(0, 3, "Just one short thought here.")]
    out, changed = split_run_on_cues(short, "en")
    assert not changed and out[0]["text"] == "Just one short thought here."

    marker = [_cue(0, 30, "[♪ music ♪]" + " " * 90)]
    _, changed = split_run_on_cues(marker, "en")
    assert not changed

    cjk = [_cue(0, 12, "これは長い文です。" * 12)]
    _, changed = split_run_on_cues(cjk, "ja")
    assert not changed

    brief = [_cue(0.0, 1.5, "One. " * 30)]  # too brief for ≥1s pieces
    _, changed = split_run_on_cues(brief, "en")
    assert not changed


def test_short_two_sentence_cue_now_splits_one_thought_per_cue():
    # Two finished thoughts on one ≥2s cue → each gets its own cue (the
    # YouTube-cadence behavior; old contract kept them welded).
    segs = [_cue(0.0, 3.0, "Two thoughts. Both short.")]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and len(out) == 2
    assert out[0]["text"] == "Two thoughts." and out[1]["text"] == "Both short."
    assert out[0]["start"] == 0.0 and out[-1]["end"] == 3.0
    for p in out:
        assert p["end"] - p["start"] >= 1.0 - 1e-9  # min-piece floor honored


def test_long_single_sentence_splits_at_clauses_when_word_timed():
    # A long single clause-run sentence with NO sentence-final punctuation still
    # splits — at strong clause boundaries — when the cue carries 1:1 words.
    text = ("M Plan, as long as there's a civilian shuttle in front of us, "
            "we have no choice but to decelerate.")
    toks = text.split()
    words = [{"start": i * 6.0 / len(toks), "end": (i + 1) * 6.0 / len(toks),
              "word": toks[i]} for i in range(len(toks))]
    segs = [_cue(0.0, 6.0, text, words=words)]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and len(out) >= 2
    # Text round-trips and word partition stays exact (no dropped/duplicated).
    assert " ".join(p["text"] for p in out) == text
    assert sum(len(p["words"] or []) for p in out) == len(words)
    for p in out:
        assert p["end"] > p["start"]


def test_long_single_sentence_kept_whole_when_word_less():
    # Same long sentence but WITHOUT 1:1 words (tier C): clause splitting is
    # suppressed to avoid scrambling char-proportional timing, so it stays whole.
    text = ("M Plan, as long as there's a civilian shuttle in front of us, "
            "we have no choice but to decelerate.")
    segs = [_cue(0.0, 6.0, text)]  # no words
    out, changed = split_run_on_cues(segs, "en")
    assert not changed and len(out) == 1


def test_words_are_partitioned_into_their_piece():
    # Words are 1:1 with the text tokens — the real case, whether they come
    # straight off Whisper or 1:1 from the aligner. Each piece takes its own
    # slice IN ORDER (by token count), so no straddling word is duplicated
    # across the boundary the way the old time-overlap filter did.
    text = ("This is the first long sentence of the run-on cue we split. "
            "And here is the second sentence that lands in piece two.")
    toks = text.split()
    words = [{"start": i * 8.0 / len(toks), "end": (i + 1) * 8.0 / len(toks),
              "word": toks[i]} for i in range(len(toks))]
    segs = [_cue(0.0, 8.0, text, words=words)]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and len(out) == 2
    w0, w1 = out[0]["words"] or [], out[1]["words"] or []
    # Exact, in-order partition: each piece's words == its own tokens.
    assert [w["word"] for w in w0] == out[0]["text"].split()
    assert [w["word"] for w in w1] == out[1]["text"].split()
    # Concatenation reproduces the input with nothing duplicated or dropped.
    assert [w["word"] for w in w0] + [w["word"] for w in w1] == toks


def test_dense_multisentence_cue_is_idempotent_and_never_zero_length():
    # Many short sentences over a long cue: each finished thought keeps its own
    # cue and the confetti cap must NOT weld two whole sentences (that would
    # re-split on a second pass). No emitted cue may be zero-length.
    text = ("First short thought here now. Second brief idea over here. "
            "Third little notion appears. Fourth quick concept lands. "
            "Fifth swift remark follows. Sixth fast point arrives. "
            "Seventh final statement ends.")
    segs = [_cue(0.0, 20.0, text)]
    out1, ch1 = split_run_on_cues(segs, "en")
    assert ch1 and len(out1) >= 6
    assert all(p["end"] > p["start"] for p in out1)          # no zero-length
    # Idempotent: a second pass changes nothing (no welded run-on to re-split).
    out2, ch2 = split_run_on_cues(out1, "en")
    assert not ch2
    assert [p["text"] for p in out1] == [p["text"] for p in out2]


def test_word_timed_tail_never_starves_last_cue():
    # Word starts clustered near the tail must not march the boundary to `end`
    # and collapse the last cue to zero length; every piece keeps a real span.
    text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
    toks = text.split()
    # starts bunched at the very end of the 3s window
    starts = [0.0, 0.05, 0.1, 2.90, 2.92, 2.94, 2.96, 2.97, 2.98, 2.985, 2.99, 2.995][:len(toks)]
    words = [{"start": starts[i], "end": starts[i], "word": toks[i]} for i in range(len(toks))]
    out, changed = split_run_on_cues([_cue(0.0, 3.0, text, words=words)], "en")
    assert changed
    for p in out:
        assert p["end"] > p["start"]                          # strictly positive span
    assert out[-1]["end"] == 3.0 and out[0]["start"] == 0.0


def test_mismatched_word_count_leaves_pieces_wordless():
    # When the word list is NOT 1:1 with the text tokens the partition is
    # ambiguous, so pieces ship word-less (the char-proportional highlighter
    # fills them) rather than duplicate a straddling word into two cues.
    text = ("This is the first long sentence of the run-on cue we split. "
            "And here is the second sentence that lands in piece two.")
    words = [{"start": 0.2 + i * 0.4, "end": 0.5 + i * 0.4, "word": f"w{i}"}
             for i in range(20)]  # 20 words vs 23 tokens
    segs = [_cue(0.0, 8.0, text, words=words)]
    out, changed = split_run_on_cues(segs, "en")
    assert changed and len(out) == 2
    assert not out[0]["words"] and not out[1]["words"]


def test_split_and_merge_are_disjoint():
    # merge only touches cues WITHOUT sentence-final punctuation; split only
    # cues WITH ≥2 sentences — running both must be stable (no fighting).
    segs = [
        _cue(0.0, 2.0, "It's just the"),          # fragment → merge folds it
        _cue(2.2, 4.0, "number 21."),
        _cue(10.0, 21.0,
             "The capsule has altered its course. Does it have suicidal "
             "tendencies? If it burns out, even secrets can be protected. "
             "I suppose that's about right."),     # run-on → split breaks it
    ]
    merged, m_changed = merge_transcript_fragments(segs, "en")
    split, s_changed = split_run_on_cues(merged, "en")
    assert m_changed and s_changed
    # The merged fragment stays merged; the run-on got split.
    texts = [c["text"] for c in split]
    assert "It's just the number 21." in texts
    assert not any(len(t) > 200 for t in texts)
    # Idempotent: a second pass changes nothing further.
    again, changed2 = split_run_on_cues(split, "en")
    assert not changed2
