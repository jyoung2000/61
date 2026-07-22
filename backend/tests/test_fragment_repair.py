"""repair_fragment_cues + the ${Name} template-leak unwrap: heal the over-splits
that hurt readability AND hand the karaoke highlight a junk target (a bare ".",
a dangling "Mr.", an ellipsis-split word), and strip a leaked "${Zechs}"
placeholder — all so the transcript reads closer to the reference subs."""

from backend.services.transcript_sanitize import repair_fragment_cues, _is_marker
from backend.services.translator import tidy_punctuation_artifacts


def _cue(start, end, text, speaker="A", words=None):
    r = {"start": float(start), "end": float(end), "text": text, "speaker": speaker}
    if words is not None:
        r["words"] = words
    return r


def _w(word, start, end):
    return {"word": word, "start": float(start), "end": float(end)}


def test_bare_punctuation_cue_is_dropped_and_time_folds_into_prev():
    rows = [
        _cue(15.0, 15.9, "Relena Dorian."),
        _cue(16.07, 16.1, "."),          # bare-punct junk cue
        _cue(16.11, 16.9, "And you?"),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    assert changed
    texts = [r["text"] for r in out]
    assert "." not in texts
    assert texts == ["Relena Dorian.", "And you?"]
    # The dropped cue's on-screen time folded into the previous cue.
    assert out[0]["end"] == 16.1


def test_dangling_title_abbreviation_merges_with_name():
    rows = [
        _cue(4.32, 4.33, "Mr."),
        _cue(4.33, 4.36, "Dorian, this shuttle will now enter the atmosphere."),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    assert changed
    assert len(out) == 1
    assert out[0]["text"] == "Mr. Dorian, this shuttle will now enter the atmosphere."
    assert out[0]["start"] == 4.32 and out[0]["end"] == 4.36


def test_abbrev_merge_repartitions_word_timings():
    rows = [
        _cue(4.32, 4.33, "Mr.", words=[_w("Mr.", 4.32, 4.33)]),
        _cue(4.33, 4.90, "Darlian speaks.",
             words=[_w("Darlian", 4.33, 4.6), _w("speaks.", 4.6, 4.9)]),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    assert changed and len(out) == 1
    # 3 displayed tokens ("Mr." "Darlian" "speaks.") → 3 word timings, in order.
    assert out[0]["words"] is not None
    assert [w["word"] for w in out[0]["words"]] == ["Mr.", "Darlian", "speaks."]
    assert len(out[0]["text"].split()) == len(out[0]["words"])


def test_ellipsis_split_word_is_bridged():
    rows = [
        _cue(18.16, 18.25, "Am Wu…"),
        _cue(18.25, 18.40, "…Fey."),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    assert changed and len(out) == 1
    # Doubled ellipsis at the seam collapses to one.
    assert out[0]["text"] == "Am Wu…Fey."


def test_pure_dialogue_is_untouched():
    rows = [
        _cue(0.0, 2.0, "Are we under attack?"),
        _cue(2.0, 4.0, "This is Duo here."),
        _cue(4.0, 6.0, "Drop your weapons and surrender."),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    assert not changed and out == rows


def test_short_complete_lines_not_merged():
    # "Yes." / "No." are legitimate short cues, not abbreviations — never merged.
    rows = [_cue(0, 1, "Yes."), _cue(1, 2, "No."), _cue(2, 3, "Understood.")]
    out, changed = repair_fragment_cues(rows, "en")
    assert not changed and len(out) == 3


def test_idempotent():
    rows = [
        _cue(4.32, 4.33, "Mr."),
        _cue(4.33, 4.36, "Dorian is here."),
        _cue(16.07, 16.1, "."),
        _cue(16.11, 16.9, "And you?"),
        _cue(18.16, 18.25, "Am Wu…"),
        _cue(18.25, 18.40, "…Fey."),
    ]
    once, _ = repair_fragment_cues(rows, "en")
    twice, changed2 = repair_fragment_cues(once, "en")
    assert not changed2 and once == twice


def test_cjk_target_passthrough():
    rows = [_cue(0, 1, "。"), _cue(1, 2, "こんにちは")]
    out, changed = repair_fragment_cues(rows, "ja")
    assert not changed and out == rows


def test_markers_never_bridged_or_dropped():
    rows = [
        _cue(0, 1, "[♪ music ♪]"),
        _cue(1, 2, "Mr."),
        _cue(2, 3, "Dorian arrives."),
    ]
    out, changed = repair_fragment_cues(rows, "en")
    # The marker survives; the Mr. + name still merge.
    assert any(_is_marker(r["text"]) for r in out)
    assert any(r["text"] == "Mr. Dorian arrives." for r in out)


# ── ${Name} template-leak unwrap ─────────────────────────────────────────────

def test_letter_led_template_placeholder_unwrapped():
    assert tidy_punctuation_artifacts("${Zechs} Six?") == "Zechs Six?"


def test_numeric_snippet_placeholder_still_unwrapped():
    assert tidy_punctuation_artifacts("${1:Moreover}, we go.") == "Moreover, we go."


def test_bare_numeric_literal_left_alone():
    # A "${5}" price/variable literal is NOT a name placeholder — leave it.
    assert tidy_punctuation_artifacts("It costs ${5} today.") == "It costs ${5} today."
