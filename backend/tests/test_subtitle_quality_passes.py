"""The five parity fixes measured against a professional reference track.

A graded comparison of a measured run against the official YouTube subtitles
for the same episode scored Content C, Timing C, Readability C, Artifacts D.
Each pass here targets one of those findings:

  1. ``suppress_echo_cues``   — one line shipped 2-4 times (artifacts, content)
  2. ``drop_junk_cues``       — hole-filling residue (artifacts, readability)
  3. ``snap_cues_to_voice_onsets`` + ``enforce_anchor_brackets`` — a one-sided
     early bias of 20 cues >1s early vs 1 late (timing)
  4. ``mark_sentence_continuations`` + ``cap_stub_dwell`` + the hard-CPS split
     escape + the min-duration floor (readability, structure)
  5. ``_micro_voice_gaps``    — short missing lines (content)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.models import TranscriptSegment, WordTimestamp  # noqa: E402
from backend.services.subtitle_aligner import enforce_anchor_brackets  # noqa: E402
from backend.services.subtitle_formatter import (  # noqa: E402
    cap_stub_dwell, clamp_cue_durations, mark_sentence_continuations)
from backend.services.transcript_sanitize import (  # noqa: E402
    drop_junk_cues, suppress_echo_cues)


# ── 1. Echo suppression ────────────────────────────────────────────────────

def test_echo_suppression_collapses_the_measured_meteor_cluster():
    # The reference covers 3:22-3:34 with ONE line; the measured run emitted
    # four independently-translated renderings of it.
    rows = [
        {"start": 202.9, "end": 205.2, "speaker": "Speaker 1",
         "text": "Reporting meteor strikes."},
        {"start": 205.2, "end": 208.7, "speaker": "Speaker 1",
         "text": "Meteors falling, they say It's being reported as a meteor strike"},
        {"start": 208.7, "end": 211.2, "speaker": "Speaker 1",
         "text": "The meteorites are being reported."},
        {"start": 211.2, "end": 214.2, "speaker": "Speaker 1",
         "text": "Reported as falling meteorites"},
        {"start": 215.9, "end": 217.6, "speaker": "Speaker 2",
         "text": "The surveillance satellites are useless."},
    ]
    out, dropped = suppress_echo_cues(rows)
    texts = [r["text"] for r in out]
    assert len(dropped) >= 2, dropped
    assert "The surveillance satellites are useless." in texts
    # Exactly one rendering of the meteor report survives.
    assert sum(1 for t in texts if "meteor" in t.lower()) == 1


def test_echo_suppression_keeps_deliberate_short_repeats():
    # The professional reference itself ships "Fire! Fire!!" and "Enemy
    # attack! Enemy attack!" — short repeated exclamations are drama.
    rows = [
        {"start": 10.0, "end": 11.0, "speaker": "S1", "text": "Fire!"},
        {"start": 11.1, "end": 12.0, "speaker": "S1", "text": "Fire!!"},
        {"start": 12.2, "end": 13.2, "speaker": "S1", "text": "Hey!"},
        {"start": 13.3, "end": 14.2, "speaker": "S1", "text": "Hey!"},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert out is rows and dropped == []


def test_echo_suppression_never_crosses_a_speaker_change():
    rows = [
        {"start": 10.0, "end": 12.0, "speaker": "Speaker 1",
         "text": "So it WAS a Gundam after all."},
        {"start": 12.5, "end": 14.5, "speaker": "Speaker 2",
         "text": "So it was a Gundam after all!"},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert len(out) == 2 and dropped == []


def test_echo_suppression_keeps_the_audio_anchored_copy():
    w = [{"word": "x", "start": 1.0, "end": 2.0}]
    rows = [
        {"start": 10.0, "end": 12.0, "speaker": "S1", "words": None,
         "text": "The combat data analysis is complete now."},
        {"start": 13.0, "end": 15.0, "speaker": "S1", "words": w,
         "text": "Combat data analysis is complete."},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert len(out) == 1 and len(dropped) == 1
    assert out[0]["words"] == w, "the measured copy must be the survivor"


def test_echo_suppression_respects_the_time_window():
    # The same stock phrase five minutes apart is two different moments.
    rows = [
        {"start": 10.0, "end": 12.0, "speaker": "S1",
         "text": "We have reached the attack altitude."},
        {"start": 310.0, "end": 312.0, "speaker": "S1",
         "text": "We have reached the attack altitude."},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert len(out) == 2 and dropped == []


# ── 2. Junk-cue filter ─────────────────────────────────────────────────────

def test_junk_filter_drops_markup_and_parked_vocalizations():
    rows = [
        {"start": 923.8, "end": 928.3, "text": "*Grunt* *grunt*"},
        {"start": 605.5, "end": 611.3, "text": "Ha ha"},          # 5.8s
        {"start": 1179.7, "end": 1184.6, "text": "Ah"},           # 4.8s
        {"start": 450.3, "end": 451.0, "text": "Gah!"},           # short: keep
        {"start": 500.0, "end": 502.0, "text": "Real dialogue here."},
    ]
    out, dropped = drop_junk_cues(rows)
    texts = [r["text"] for r in out]
    assert "*Grunt* *grunt*" not in texts
    assert "Ha ha" not in texts
    assert "Ah" not in texts
    assert "Gah!" in texts, "a brief interjection is real"
    assert "Real dialogue here." in texts
    assert len(dropped) == 3


def test_junk_filter_keeps_real_words_the_reference_also_captions():
    # The professional track captions all of these — dropping by shortness
    # would have thrown away real dialogue.
    rows = [
        {"start": 10.0, "end": 14.0, "text": "Hey!"},
        {"start": 15.0, "end": 19.0, "text": "Huh?"},
        {"start": 20.0, "end": 24.0, "text": "What?!"},
        {"start": 25.0, "end": 29.0, "text": "Yes, sir."},
        {"start": 30.0, "end": 34.0, "text": "[♪ Opening theme ♪]"},
    ]
    out, dropped = drop_junk_cues(rows)
    assert out is rows and dropped == []


# ── 3. Timing: anchor brackets ─────────────────────────────────────────────

def _anchored(start, end, text):
    return {"start": start, "end": end, "text": text,
            "words": [{"word": "w", "start": start, "end": end}]}


def test_anchor_brackets_pull_a_drifted_run_back_onto_its_audio():
    # The measured failure: a run of projected cues leading its audio by ~9s
    # between two CTC-aligned neighbours.
    cues = [
        _anchored(1130.0, 1132.0, "anchored before"),
        {"start": 1121.0, "end": 1124.0, "text": "This was taken by Oz drones."},
        {"start": 1124.0, "end": 1127.0, "text": "The suits we fought are similar."},
        _anchored(1150.0, 1152.0, "anchored after"),
    ]
    out = enforce_anchor_brackets(cues)
    assert out["runs"] == 1 and out["cues"] == 2
    assert cues[1]["start"] >= 1132.0 - 1e-6
    assert cues[2]["end"] <= 1150.0 + 1e-6
    assert cues[1]["end"] <= cues[2]["start"] + 1e-6      # still monotonic


def test_anchor_brackets_leave_a_well_placed_run_untouched():
    cues = [
        _anchored(100.0, 101.0, "A"),
        {"start": 102.0, "end": 104.0, "text": "comfortably inside"},
        _anchored(110.0, 111.0, "B"),
    ]
    before = (cues[1]["start"], cues[1]["end"])
    out = enforce_anchor_brackets(cues)
    assert out["cues"] == 0
    assert (cues[1]["start"], cues[1]["end"]) == before


def test_anchor_brackets_never_move_an_anchored_cue():
    cues = [
        _anchored(100.0, 101.0, "A"),
        _anchored(90.0, 92.0, "measured but early — not ours to move"),
        _anchored(110.0, 111.0, "B"),
    ]
    out = enforce_anchor_brackets(cues)
    assert out["cues"] == 0
    assert cues[1]["start"] == 90.0


def test_anchor_brackets_need_two_anchors():
    cues = [{"start": 1.0, "end": 2.0, "text": "a"},
            {"start": 3.0, "end": 4.0, "text": "b"}]
    assert enforce_anchor_brackets(cues)["cues"] == 0


# ── 3b. Timing: voice-onset snap ───────────────────────────────────────────

def test_voice_onset_snap_pulls_a_silent_start_onto_speech(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(182.5, 186.0)])
    SC._VAD_CACHE.clear()
    rows = [{"start": 181.0, "end": 185.0, "text": "It's not just one."}]
    out, shifted = SC.snap_cues_to_voice_onsets(rows, str(wav))
    assert len(shifted) == 1
    assert out[0]["start"] == 182.5 and out[0]["end"] == 185.0


def test_voice_onset_snap_skips_measured_and_on_voice_cues(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(20.0, 25.0)])
    SC._VAD_CACHE.clear()
    rows = [
        {"start": 18.0, "end": 24.0, "text": "measured — leave it",
         "words": [{"word": "w", "start": 18.0, "end": 24.0}]},
        {"start": 21.0, "end": 24.0, "text": "already on voice"},
    ]
    out, shifted = SC.snap_cues_to_voice_onsets(rows, str(wav))
    assert shifted == []
    assert out[0]["start"] == 18.0 and out[1]["start"] == 21.0


def test_voice_onset_snap_is_capped_and_fails_soft(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    # Onset 30s away — far beyond the cap; a snap that big would be a guess.
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(50.0, 55.0)])
    SC._VAD_CACHE.clear()
    rows = [{"start": 20.0, "end": 22.0, "text": "way out"}]
    out, shifted = SC.snap_cues_to_voice_onsets(rows, str(wav))
    assert shifted == [] and out[0]["start"] == 20.0
    # No VAD at all → untouched.
    monkeypatch.setattr(SC, "voice_activity_regions", lambda p, **k: [])
    SC._VAD_CACHE.clear()
    rows2 = [{"start": 20.0, "end": 22.0, "text": "x"}]
    out2, shifted2 = SC.snap_cues_to_voice_onsets(rows2, str(wav))
    assert out2 is rows2 and shifted2 == []


# ── 4. Readability ─────────────────────────────────────────────────────────

def test_continuations_are_marked_the_way_the_reference_marks_them():
    rows = [
        {"start": 0.0, "end": 2.0, "text": "The surveillance"},
        {"start": 2.1, "end": 4.0, "text": "satellites are useless."},
    ]
    mark_sentence_continuations(rows)
    assert rows[0]["text"] == "The surveillance…"
    assert rows[1]["text"] == "…satellites are useless."


def test_continuations_skip_new_sentences_markers_and_dashed_cues():
    rows = [
        {"start": 0.0, "end": 2.0, "text": "Understood"},
        {"start": 2.1, "end": 4.0, "text": "Mission changed acknowledged."},
        {"start": 4.1, "end": 6.0, "text": "[♪ music ♪]"},
        {"start": 6.1, "end": 8.0, "text": "still music"},
        {"start": 8.1, "end": 10.0, "text": "- Is the Leo ready?"},
        {"start": 10.1, "end": 12.0, "text": "- yes it is"},
    ]
    mark_sentence_continuations(rows)
    assert rows[0]["text"] == "Understood", "capitalized opener = new sentence"
    assert rows[2]["text"] == "[♪ music ♪]"
    assert rows[4]["text"] == "- Is the Leo ready?"


def test_continuations_are_idempotent():
    rows = [{"start": 0.0, "end": 2.0, "text": "The surveillance"},
            {"start": 2.1, "end": 4.0, "text": "satellites are useless."}]
    mark_sentence_continuations(rows)
    once = [r["text"] for r in rows]
    mark_sentence_continuations(rows)
    assert [r["text"] for r in rows] == once


def test_stub_dwell_cap_trims_parked_stubs_only():
    rows = [
        {"start": 605.5, "end": 611.3, "text": "Ha ha"},
        {"start": 781.5, "end": 786.5, "text": "Come on"},
        {"start": 100.0, "end": 101.5, "text": "Understood."},
        {"start": 200.0, "end": 206.0, "text": "This is a full sentence that "
                                               "legitimately needs its time."},
        {"start": 300.0, "end": 306.0, "text": "[♪ Opening theme ♪]"},
    ]
    cap_stub_dwell(rows)
    assert rows[0]["end"] == 607.5          # 2.0s cap
    assert rows[1]["end"] == 783.5
    assert rows[2]["end"] == 101.5          # already short — untouched
    assert rows[3]["end"] == 206.0          # long text — untouched
    assert rows[4]["end"] == 306.0          # markers have their own ceiling


def test_stub_dwell_cap_never_moves_the_start_or_crosses_the_floor():
    rows = [{"start": 10.0, "end": 20.0, "text": "Yes."}]
    cap_stub_dwell(rows)
    assert rows[0]["start"] == 10.0
    assert rows[0]["end"] - rows[0]["start"] >= 0.833


def test_min_duration_floor_grows_a_sub_minimum_cue_into_its_gap():
    # A measured run shipped three 0.709s cues (17 frames vs a 20-frame
    # minimum) boxed in by 1-frame gaps, which no earlier pass could grow.
    rows = [
        {"start": 216.889, "end": 217.598, "text": "satellites are useless."},
        {"start": 220.000, "end": 222.000, "text": "next cue, far away"},
    ]
    clamp_cue_durations(rows)
    assert rows[0]["end"] - rows[0]["start"] >= 0.833 - 1e-6
    assert rows[0]["end"] < rows[1]["start"], "must not touch the next cue"


def test_min_duration_floor_respects_a_tight_neighbour():
    rows = [
        {"start": 100.000, "end": 100.709, "text": "short"},
        {"start": 100.800, "end": 102.000, "text": "right behind it"},
    ]
    clamp_cue_durations(rows)
    assert rows[0]["end"] <= 100.800 - 0.041, "no overlap, gap preserved"


def test_hard_cps_escape_splits_an_unreadable_wordless_cue():
    # word_timed_split_only normally holds a word-less cue whole; above the
    # hard CPS ceiling an approximate cut is the smaller error — PROVIDED
    # both halves still clear the display minimum. (A shipped run proved the
    # unconditional version harmful: it cut a 0.96 s cue into two flashes of
    # 0.539 s and 0.421 s. The too-short case is covered separately by
    # test_cps_split_will_not_manufacture_unreadable_flashes.)
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(
        start=482.44, end=484.44, speaker="Speaker 1",
        text="which is faster than O, would that be more suitable for this")
    # 60 chars in 2.0 s = 30 CPS, past the 25 ceiling, and long enough that
    # both halves clear the 0.833 s display floor.
    pieces = _split_segment(
        seg, target_cps=20.0, max_chars_per_line=34,
        word_timed_split_only=True)
    assert len(pieces) > 1, "an unreadable cue must be cut even without words"
    assert " ".join(p.text for p in pieces).split() == seg.text.split()
    for p in pieces:
        assert p.end - p.start >= 0.833 - 1e-6, "halves must stay readable"


def test_hard_cps_escape_leaves_readable_wordless_cues_whole():
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(start=100.0, end=104.0, speaker="Speaker 1",
                            text="A comfortable line to read.")
    pieces = _split_segment(
        seg, target_cps=20.0, max_chars_per_line=34,
        word_timed_split_only=True)
    assert len(pieces) == 1


# ── 5. Micro-gap recovery ──────────────────────────────────────────────────

def test_micro_gaps_find_the_short_uncovered_line():
    # "A body!" is ~1s between two well-transcribed cues: it opens no hole
    # large enough for any other selector, which is why it went missing.
    from backend.services.vocal_gap_recovery import _micro_voice_gaps
    segments = [{"start": 780.0, "end": 785.0, "text": "before"},
                {"start": 790.0, "end": 795.0, "text": "after"}]
    voice = [(780.0, 785.0), (785.8, 786.8), (790.0, 795.0)]
    out = _micro_voice_gaps(segments, voice, pad_s=0.3, existing=[])
    assert len(out) == 1
    lo, hi = out[0]
    assert lo <= 785.8 and hi >= 786.8, "the voiced moment must be inside"


def test_micro_gaps_ignore_covered_audio_and_spans_already_queued():
    from backend.services.vocal_gap_recovery import _micro_voice_gaps
    segments = [{"start": 0.0, "end": 100.0, "text": "covers everything"}]
    voice = [(10.0, 20.0), (30.0, 40.0)]
    assert _micro_voice_gaps(segments, voice, pad_s=0.3, existing=[]) == []
    # Uncovered, but an earlier tier already queued it.
    segments2 = [{"start": 0.0, "end": 10.0, "text": "a"}]
    voice2 = [(12.0, 14.0)]
    assert _micro_voice_gaps(segments2, voice2, pad_s=0.3,
                             existing=[(11.0, 15.0)]) == []


def test_micro_gaps_need_a_vad_map_and_enough_voice():
    from backend.services.vocal_gap_recovery import _micro_voice_gaps
    segments = [{"start": 0.0, "end": 10.0, "text": "a"}]
    assert _micro_voice_gaps(segments, [], pad_s=0.3, existing=[]) == []
    # 0.2s of voice is a breath, not a line.
    assert _micro_voice_gaps(segments, [(12.0, 12.2)], pad_s=0.3,
                             existing=[]) == []


def test_micro_gaps_are_capped():
    from backend.services.vocal_gap_recovery import _micro_voice_gaps
    segments = [{"start": 0.0, "end": 1.0, "text": "a"}]
    voice = [(10.0 + 5 * i, 11.0 + 5 * i) for i in range(80)]
    out = _micro_voice_gaps(segments, voice, pad_s=0.3, existing=[])
    assert 0 < len(out) <= 24


def test_echo_suppression_protects_a_line_carrying_a_new_name():
    # The reference stages this beat deliberately: "My name..." then "My
    # name is Relena Darlian." An early build deleted the surname because
    # the fragment before it scored as a match.
    rows = [
        {"start": 953.0, "end": 958.8, "speaker": "S1",
         "text": "It's me... I'm Relena..."},
        {"start": 959.9, "end": 964.9, "speaker": "S1",
         "text": "Relena Dorlian."},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert len(out) == 2 and dropped == []


def test_echo_suppression_ignores_a_single_shared_content_word():
    # "Father, what's that?" reduces to one content word and would otherwise
    # score 1.0 against any other line mentioning a father.
    rows = [
        {"start": 295.2, "end": 301.8, "speaker": "S1",
         "text": "Father, next time when you go to space, please take it easy."},
        {"start": 301.8, "end": 303.1, "speaker": "S1",
         "text": "Father, what's that?"},
    ]
    out, dropped = suppress_echo_cues(rows)
    assert len(out) == 2 and dropped == []


def test_continuation_marks_a_binding_tail_even_before_a_capital():
    # "…manufacturing technologies" after "the Alliance and" is one sentence
    # however the next cue is capitalized.
    rows = [{"start": 0.0, "end": 2.0, "text": "Apart from the Alliance and"},
            {"start": 2.1, "end": 4.0, "text": "Oz, there were others."}]
    mark_sentence_continuations(rows)
    assert rows[0]["text"].endswith("…")
    assert rows[1]["text"].startswith("…")


def test_continuation_still_refuses_two_independent_sentences():
    rows = [{"start": 0.0, "end": 2.0, "text": "Understood"},
            {"start": 2.1, "end": 4.0, "text": "Mission changed"}]
    mark_sentence_continuations(rows)
    assert rows[0]["text"] == "Understood"


# ── Run-13 follow-ups: four regressions found in the shipped output ────────

def test_cps_split_will_not_manufacture_unreadable_flashes():
    # The escape cut a 0.96s over-CPS cue into 0.539s + 0.421s — both under
    # the display minimum. Two flashes nobody can read is not an improvement
    # on one cue that reads fast.
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(start=100.0, end=100.96, speaker="S1",
                            text="which is faster than O, be more suitable?")
    pieces = _split_segment(seg, target_cps=20.0, max_chars_per_line=34,
                            word_timed_split_only=True)
    assert len(pieces) == 1, "too short to split readably — keep it whole"
    for p in pieces:
        assert p.end - p.start >= 0.833 - 1e-6


def test_cps_split_still_fires_when_both_halves_stay_readable():
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(
        start=100.0, end=102.4, speaker="S1",
        text="which is faster than O, would that be more suitable for this mission")
    pieces = _split_segment(seg, target_cps=20.0, max_chars_per_line=34,
                            word_timed_split_only=True)
    assert len(pieces) == 2
    for p in pieces:
        assert p.end - p.start >= 0.833 - 1e-6


def test_over_box_split_keeps_its_exemption():
    # Text running off the screen is still worse than an approximate cut, so
    # an over-BOX cue may still produce short pieces.
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(
        start=1150.0, end=1150.67, speaker="S1",
        text="It looks similar to the Suits we fought before and now there are two")
    pieces = _split_segment(seg, target_cps=20.0, max_chars_per_line=34,
                            word_timed_split_only=True)
    assert len(pieces) > 1


def test_heal_split_ellipsis_collapses_a_welded_marker_pair():
    from backend.services.subtitle_formatter import heal_split_ellipsis
    rows = [
        {"text": "The capsule has changed course,… …does it want to commit suicide?"},
        {"text": "even more,... ...trying to escape?"},
        {"text": "A cue that legitimately trails off…"},
        {"text": "…and its separate continuation."},
        {"text": "Wait... what?"},
    ]
    heal_split_ellipsis(rows)
    assert rows[0]["text"] == "The capsule has changed course, does it want to commit suicide?"
    assert rows[1]["text"] == "even more, trying to escape?"
    assert rows[2]["text"] == "A cue that legitimately trails off…"
    assert rows[3]["text"] == "…and its separate continuation."
    assert rows[4]["text"] == "Wait... what?"
    before = [r["text"] for r in rows]
    heal_split_ellipsis(rows)
    assert [r["text"] for r in rows] == before, "must be idempotent"


def test_readability_pass_heals_the_pairs_it_welds():
    # enforce_readability is what merges a marked pair back together, so the
    # heal has to be part of what it RETURNS.
    from backend.services.subtitle_formatter import enforce_readability
    segs = [
        TranscriptSegment(start=10.0, end=11.2, speaker="S1",
                          text="The capsule has changed course,…"),
        TranscriptSegment(start=11.3, end=13.0, speaker="S1",
                          text="…does it want to commit suicide?"),
    ]
    out = enforce_readability(segs, max_cps=20.0, max_chars_per_line=34)
    for cue in out:
        assert "… …" not in cue.text and "... ..." not in cue.text


def test_onset_snap_uses_a_confident_threshold_not_the_gate_s(monkeypatch, tmp_path):
    # The attestation gate decodes at 0.25 to protect whispers; at that
    # sensitivity almost nothing reads as silence and this pass found zero
    # work on a real run.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    seen = {}

    def _fake(p, **k):
        seen.update(k)
        return [(10.0, 12.0)]

    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    SC._VAD_CACHE.clear()
    SC.snap_cues_to_voice_onsets(
        [{"start": 8.0, "end": 12.0, "text": "x"}], str(wav))
    assert seen.get("threshold") == 0.5


# ── Audio-keyed theme collapse ─────────────────────────────────────────────

def _sung(n, t0, step=4.0):
    return [{"start": t0 + step * i, "end": t0 + step * i + step,
             "text": f"sung line {i}", "speaker": "S1"} for i in range(n)]


def test_audio_theme_collapse_marks_both_themes_without_reading_the_words():
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = (_sung(8, 30.0)
            + [{"start": 300.0, "end": 303.0, "text": "Real dialogue.",
                "speaker": "S2"}]
            + _sung(6, 1360.0, 5.0)
            + [{"start": 1450.0, "end": 1455.0, "text": "Closing line.",
                "speaker": "S2"}])
    out, changed = collapse_theme_by_music_spans(
        rows, [(26.0, 92.0), (1358.0, 1415.0)])
    assert changed
    texts = [r["text"] for r in out]
    assert "[♪ Opening theme ♪]" in texts and "[♪ Ending theme ♪]" in texts
    assert "Real dialogue." in texts and "Closing line." in texts
    assert not any(t.startswith("sung line") for t in texts)


def test_theme_collapse_logs_the_anchor_drift_it_introduces(caplog):
    """The theme marker's placement has oscillated between runs — a correct
    ~30s on some, 0:00 on others — and the mechanism was only ever reproduced
    synthetically. The backwards walk over absorbable cues is what moves the
    anchor off its chorus, so the run must say by how much and on account of
    which cues, in ROW coordinates. (The walk indexes the contiguous GROUP;
    reporting a group offset as a row offset would name an unrelated cue two
    thousand seconds away and send the next investigation to the wrong place.)
    """
    import logging
    from backend.services.transcript_sanitize import collapse_song_choruses
    rows = [{"start": 200.0 + 20.0 * i, "end": 203.0 + 20.0 * i,
             "speaker": "S1",
             "text": f"Heero warns Relena about the Alliance, part {i}."}
            for i in range(10)]
    rows += [{"start": 1360.0 + 4.0 * i, "end": 1363.5 + 4.0 * i,
              "speaker": "S2", "text": t}
             for i, t in enumerate([
                 "a soft night wind",           # absorbed head
                 "the refrain returns again",   # chorus A
                 "and echoes far away",         # chorus B
                 "the refrain returns again",   # chorus A reprise
                 "and echoes far away",         # chorus B reprise
                 "trailing soft line",          # absorbed tail
             ])]
    with caplog.at_level(logging.INFO,
                         logger="backend.services.transcript_sanitize"):
        out, changed = collapse_song_choruses([dict(r) for r in rows])
    assert changed
    line = next(m for m in caplog.messages if m.startswith("theme collapse:"))
    assert "4→6 cue(s)" in line
    # Anchor 1360.0 vs a chorus that starts at 1364.0 — the four seconds the
    # walk gave away, not a row index picked out of the wrong list.
    assert "anchor 1360.000s (chorus starts 1364.000s, drift -4.000s)" in line
    assert "a soft night wind" in line and "trailing soft line" in line
    assert "Heero warns Relena" not in line


def test_theme_collapse_is_quiet_when_it_does_not_extend_the_run(caplog):
    import logging
    from backend.services.transcript_sanitize import collapse_song_choruses
    rows = [{"start": 200.0 + 20.0 * i, "end": 203.0 + 20.0 * i,
             "speaker": "S1",
             "text": f"Heero warns Relena about the Alliance, part {i}."}
            for i in range(10)]
    # Same song, but bounded by dialogue on both sides instead of by lyrics:
    # nothing is absorbable, so the anchor IS the chorus and there is no drift
    # to report.
    rows += [{"start": 1360.0 + 4.0 * i, "end": 1363.5 + 4.0 * i,
              "speaker": "S2", "text": t}
             for i, t in enumerate([
                 "the refrain returns again",
                 "and echoes far away",
                 "the refrain returns again",
                 "and echoes far away",
             ])]
    with caplog.at_level(logging.INFO,
                         logger="backend.services.transcript_sanitize"):
        collapse_song_choruses([dict(r) for r in rows])
    assert not [m for m in caplog.messages if m.startswith("theme collapse:")]


def test_source_chorus_spans_survive_translation_phrasing_variance():
    """The same episode's ED collapsed on runs whose translation left lyrics
    unpunctuated and shipped as dialogue on runs that punctuated them — with
    an identical Whisper-JA track under both. The source repetition is the
    phrasing-independent evidence, and its span must collapse the punctuated
    TRANSLATED lyrics the text pass refuses."""
    from backend.services.transcript_sanitize import (
        collapse_theme_by_music_spans, source_chorus_spans)
    src = [{"start": 200.0 + 30.0 * i, "end": 204.0 + 30.0 * i,
            "text": f"作戦の状況を報告する、その{i}。"} for i in range(10)]
    ed = ["ただ愛のせいで眠れない", "君を呼ぶ声が響く",
          "ただ愛のせいで眠れない", "夜の風が頬を撫でる",
          "君を呼ぶ声が響く", "ただ愛のせいで眠れない"]
    src += [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i, "text": t}
            for i, t in enumerate(ed)]
    spans = source_chorus_spans(src)
    assert len(spans) == 1 and 1355.0 < spans[0][0] <= 1360.5
    # A Whisper hallucination LOOP repeats ONE line — never a chorus.
    loop = [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
             "text": "ご視聴ありがとうございました"} for i in range(6)]
    assert source_chorus_spans(
        src[:10] + loop) == []
    # The span collapses the PUNCTUATED translated lyrics (run-21/23 shape,
    # which the chorus/soft text passes measurably refused).
    translated = [{"start": 200.0 + 30.0 * i, "end": 204.0 + 30.0 * i,
                   "speaker": "S1", "text": f"Reporting operation status {i}."}
                  for i in range(10)]
    translated += [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
                    "speaker": "S2", "text": t}
                   for i, t in enumerate([
                       "I really don't like it when you act this way.",
                       "You're so full of yourself.",
                       "Why are you making me wait?",
                       "I know you ran over here.",
                       "That's just how things work around here.",
                       "It irritates me.",
                   ])]
    out, changed = collapse_theme_by_music_spans(translated, spans)
    assert changed
    texts = [r["text"] for r in out]
    assert "[♪ Ending theme ♪]" in texts
    assert not any("full of yourself" in t for t in texts)
    assert f"Reporting operation status 3." in texts


def test_theme_anchor_ignores_an_isolated_zero_hallucination():
    from backend.services.transcript_sanitize import _theme_anchor_start
    # The 0:00 phantom is temporally ISOLATED from the real song.
    assert _theme_anchor_start([0.0, 30.4, 35.2, 40.0]) == 30.4
    # A theme that genuinely opens the video keeps its first cue.
    assert _theme_anchor_start([0.4, 4.2, 8.9]) == 0.4
    assert _theme_anchor_start([26.0, 30.0]) == 26.0


def test_marker_clamp_never_moves_a_sane_marker_later(caplog):
    """Measured failure: the text pass anchored the opening theme at 30.342s
    (correct) and the classifier's first onset was 61.0s (its silence floor
    slept through the song's quiet intro) — the unrestricted clamp shoved a
    right answer half a minute late. Only a ~0:00 marker may be clamped."""
    from backend.services.transcript_sanitize import clamp_theme_marker_starts
    rows = [{"start": 30.342, "end": 34.342, "speaker": "",
             "text": "[♪ Opening theme ♪]"}]
    out, notes = clamp_theme_marker_starts([dict(r) for r in rows], 61.0)
    assert notes == [] and out[0]["start"] == 30.342
    # The one pathology it exists for still heals.
    z, n = clamp_theme_marker_starts(
        [{"start": 0.0, "end": 4.0, "speaker": "",
          "text": "[♪ Opening theme ♪]"}], 61.0)
    assert len(n) == 1 and z[0]["start"] == 61.0


def test_near_gap_rescue_keeps_reclocked_short_lines():
    """A measured run decoded 36 recovered cue(s) and culled 34 as
    "outside-gap" — among them the short interjections the reference
    captions ("Roger!", "It moved!"). A short decode within 2s of the span
    over uncaptioned audio is a find at a re-clocked time; a long decode far
    away is a boundary re-hearing and stays culled."""
    from backend.services.vocal_gap_recovery import _clip_to_gap
    gap, pad = (100.0, 104.0), 0.4
    # 1.4s past the span end, 1.1s long, nothing existing there → kept AT
    # ITS DECODED TIME (the decode's own clock is the only evidence).
    got = _clip_to_gap({"start": 105.4, "end": 106.5, "text": "Roger!"},
                       gap, pad, existing=[])
    assert got is not None and got["start"] == 105.4 and got["end"] == 106.5
    # Same shape but an existing cue already covers that audio → cull.
    assert _clip_to_gap({"start": 105.4, "end": 106.5, "text": "Roger!"},
                        gap, pad,
                        existing=[{"start": 105.0, "end": 107.0}]) is None
    # Too far outside → cull; too long even when near → cull.
    assert _clip_to_gap({"start": 108.5, "end": 109.4, "text": "x"},
                        gap, pad, existing=[]) is None
    assert _clip_to_gap({"start": 104.5, "end": 109.0, "text": "long line"},
                        gap, pad, existing=[]) is None


def test_asr_decode_is_seeded_on_both_paths():
    """Greedy decoding is deterministic; the 277-vs-283-segment variance
    between identical runs enters when the temperature FALLBACK samples.
    The remote request carries the pinned seed as a tuning field and the
    local path seeds CTranslate2 directly."""
    from backend.services import reframer_audio as ra
    fields = ra._remote_tuning_fields()
    assert fields.get("seed") == "42"
    ra._seed_ct2_sampling()   # must never raise, with or without ct2
    import inspect
    # The helper is only worth anything if the local decode actually calls
    # it before transcribing.
    assert inspect.getsource(ra).count("_seed_ct2_sampling()") >= 2


def test_rank_titles_pin_the_renderings_localizations_use():
    """ゼクス中尉 shipped as "Zechs Unique" three times on a measured run —
    the glossary pinned the name and left the model to guess the title.
    Ranks are a closed vocabulary with one accepted rendering each."""
    from backend.services.canonical_names import rank_title_pairs_in
    got = rank_title_pairs_in("ゼクス中尉、報告します。外務次官ドーリアン閣下が到着。")
    assert got["中尉"] == "Lieutenant"
    assert got["外務次官"] == "Vice Foreign Minister"
    assert got["閣下"] == "Excellency"
    assert rank_title_pairs_in("こんにちは、元気ですか。") == {}


def test_junk_filter_drops_meta_leaks_but_not_reference_shapes():
    """"Mrs. Ifc, check translation.", "Title Strange." and bare "Episode 1"
    all shipped as subtitles on a measured run. The professional reference's
    own "Next Episode" and "Next, on Gundam Wing, Episode 2." must survive —
    zero reference damage was measured across the full track."""
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": float(i), "end": float(i) + 2.0, "text": t}
            for i, t in enumerate([
                "Mrs. Ifc, check translation.",
                "Title Strange.",
                "Episode 1",
                "Episode",
                "(TN: this is a pun)",
                "Next Episode",
                "Next, on Gundam Wing, Episode 2.",
                "The Gundam Deathscythe",
            ])]
    kept, dropped = drop_junk_cues(rows)
    kept_texts = [r["text"] for r in kept]
    assert len(dropped) == 5
    assert "Next Episode" in kept_texts
    assert "Next, on Gundam Wing, Episode 2." in kept_texts
    assert "The Gundam Deathscythe" in kept_texts
    assert not any("check translation" in t for t in kept_texts)


def test_junk_filter_drops_credit_cards_preambles_and_crumbs():
    """Run-24 leaks, verbatim: the ED's credit-card readouts ("Lyrics by…"),
    the model announcing its answer instead of giving it, and a batch
    reply's JSON crumbs glued to real words. All measured at ZERO drops on
    the full professional reference."""
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": float(i), "end": float(i) + 2.0, "text": t}
            for i, t in enumerate([
                "Lyrics by…",
                "…Composition and…",
                "…arrangement by Initial composition Miku",
                "Here's the translated subtitle line in English:",
                "Gundams in total now. }`[",
                "The Gundam Deathscythe",
            ])]
    kept, dropped = drop_junk_cues(rows)
    kept_texts = [r["text"] for r in kept]
    assert len(dropped) == 4
    assert "Gundams in total now." in kept_texts          # crumbs stripped
    assert "The Gundam Deathscythe" in kept_texts


def test_recovered_cues_inside_a_song_window_are_dropped():
    """The post-COMPLETE merge runs AFTER the theme collapse decided where
    the songs are, so a recovered lyric/credit fragment walks past every
    guard — run 24 shipped the ED credits 25s BEFORE the marker's own start
    (the marker anchors on the chorus; credits ride the intro). Scoped by
    provenance and preview-exempt: next-episode narration is real."""
    from backend.services.transcript_sanitize import (
        drop_recovered_near_theme_markers)
    rows = [
        {"start": 1371.0, "end": 1375.0, "text": "[♪ Ending theme ♪]"},
        {"start": 1346.0, "end": 1350.0, "text": "Sorry, Jagdster",
         "recovered": True},
        {"start": 1425.0, "end": 1429.0, "recovered": True,
         "text": "The Alliance sends troops to find the sunken Gundam."},
        {"start": 1350.0, "end": 1354.0, "text": "That is terrible."},
    ]
    kept, dropped = drop_recovered_near_theme_markers(rows)
    kept_texts = [r["text"] for r in kept]
    assert len(dropped) == 1 and "Jagdster" in dropped[0]
    # Preview-shaped recovered cue survives; NON-recovered cue in the window
    # is never touched (the collapse already ruled on it).
    assert any("sunken Gundam" in t for t in kept_texts)
    assert "That is terrible." in kept_texts
    # No markers → pure no-op.
    same, none = drop_recovered_near_theme_markers(rows[1:])
    assert none == [] and len(same) == 3


def test_source_chorus_accepts_a_single_stable_hook():
    """Real EDs defeat the two-distinct-repeats gate: every verse transcribes
    slightly differently on every Whisper pass, and only the HOOK is stable
    enough to repeat verbatim. One line repeating 2-3 times among distinct
    verses is a song; one line repeating six times over nothing else is a
    Whisper loop and still yields nothing."""
    from backend.services.transcript_sanitize import source_chorus_spans
    src = [{"start": 200.0 + 30.0 * i, "end": 204.0 + 30.0 * i,
            "text": f"作戦の状況を報告する、その{i}。"} for i in range(10)]
    ed = ["ただ愛だけが", "夜風が頬を撫でていく", "君を呼ぶ声がどこかで",
          "ただ愛だけが", "遠い空の下で歌う", "誰もいない街角で"]
    src_hook = src + [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
                       "text": t} for i, t in enumerate(ed)]
    spans = source_chorus_spans(src_hook)
    assert len(spans) == 1 and spans[0][0] == 1360.0
    loop = src + [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
                   "text": "ご視聴ありがとうございました"} for i in range(6)]
    assert source_chorus_spans(loop) == []


def test_post_merge_resanitize_now_covers_the_late_leaks():
    """Run 24's 'Mr. Dorlian' ×4 were recovery-merged AFTER the main-track
    respell ran — the alias that fixes them sat unused. The re-sanitize must
    carry the theme-window guard and the glossary respell."""
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline._resanitize_after_merge)
    assert "drop_recovered_near_theme_markers" in src
    assert "respell_text_from_glossary" in src
    assert "kana_aliases_for_job" in src


def test_respeller_never_corrupts_correct_text():
    """Run-25 regression, verbatim: the mined-names list carried junk
    ("Report", "Aretha") and the respeller REWROTE CORRECT TEXT with it —
    "Reporting"→"Report" and "Aries"→"Aretha" ×3. Three guards now hold:
    ordinary English (including by stem) is never a garble; when kana data
    exists, only kana-backed names may be orthographic targets; and short
    words need ≥0.85 on the alias loose-match ("aries"→"aresa" was 0.80
    exactly). The real fixes still fire."""
    from backend.services.canonical_names import respell_text_from_glossary
    gloss = ["Report", "Aretha", "Quatre Raberba Winner", "Relena Darlian"]
    aliases = {"katoru": "Quatre", "kato": "Quatre", "darian": "Darlian",
               "dorian": "Darlian", "aresa": "Aretha"}
    texts = ["Reporting strikes to Agent Zechs.",
             "Then wouldn't the Aries mobile suit be better?",
             "This is Kato.",
             "Mr. Dorian, sir.",
             "I trust Dorlian completely."]
    out, n, _ = respell_text_from_glossary(texts, gloss, aliases)
    assert "Reporting" in out[0]              # inflection of junk 'Report'
    assert "Aries" in out[1]                  # 0.80 alias match, word too short
    assert out[2] == "This is Quatre."        # exact alias still fires
    assert out[3].startswith("Mr. Darlian")   # position-exempt exact alias
    assert "Darlian" in out[4]                # 0.92 loose alias still fires
    # A junk mined name is never an orthographic AUTHORITY once kana exists.
    out2, _, _ = respell_text_from_glossary(
        ["He said Repord filed it."], gloss, aliases)
    assert "Repord" in out2[0]


def test_theme_window_guard_ignores_bare_music_markers():
    """Run-25 regression: the guard around a mid-episode "[♪ music ♪]"
    marker dropped a recovered line at 682s that was plausibly real
    dialogue — score under a scene has dialogue right beside it. Only the
    opening/ending THEME labels testify to a song region."""
    from backend.services.transcript_sanitize import (
        drop_recovered_near_theme_markers)
    rows = [{"start": 684.0, "end": 689.0, "text": "[♪ music ♪]"},
            {"start": 682.4, "end": 686.0, "recovered": True,
             "text": "There is absolutely no absolute good"}]
    kept, dropped = drop_recovered_near_theme_markers(rows)
    assert dropped == [] and len(kept) == 2


def test_junk_filter_drops_preview_stub_readouts():
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": float(i), "end": float(i) + 2.0, "text": t}
            for i, t in enumerate([
                "Next time preview Part 1 Part 2 Part 2 Part 2",
                "Next episode preview",
                "Next Episode",
                "Next, on Gundam Wing, Episode 2.",
            ])]
    kept, dropped = drop_junk_cues(rows)
    assert len(dropped) == 2
    assert all("preview-stub" in d for d in dropped)
    kept_texts = [r["text"] for r in kept]
    assert "Next Episode" in kept_texts
    assert "Next, on Gundam Wing, Episode 2." in kept_texts


def test_source_chorus_finds_a_hook_glued_onto_different_verses():
    """Whisper attaches the hook to different verse text on each repeat, so
    the full-cue key never matches twice — the punctuation-split PIECE does.
    This is why the detector stayed silent on two real EDs."""
    from backend.services.transcript_sanitize import source_chorus_spans
    src = [{"start": 200.0 + 30.0 * i, "end": 204.0 + 30.0 * i,
            "text": f"作戦の状況を報告する、その{i}。"} for i in range(10)]
    ed = ["ジャストラブ、君の名前を呼ぶ", "夜風が頬を撫でていく",
          "君を呼ぶ声がどこかで", "ジャストラブ、待たせないでよ",
          "遠い空の下で歌う", "誰もいない街角で待つ"]
    src += [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i, "text": t}
            for i, t in enumerate(ed)]
    spans = source_chorus_spans(src)
    assert len(spans) == 1 and spans[0][0] == 1360.0


def test_glossary_miner_rejects_acronyms():
    """A measured run's glossary shipped 'DVD' as a cast name (the franchise
    page mentions the format constantly, mid-sentence, TitleCase-shaped by
    the regex's lights). Cast names are Xxxx-shaped; ALL-CAPS is a format."""
    from backend.services.canonical_names import _mine_glossary_names
    text = ("The series was released on DVD in 2000. Critics praised the DVD "
            "release. The pilot Heero Yuy flies the machine, and later "
            "the same Heero Yuy returns. Fans bought the DVD again.")
    mined = _mine_glossary_names(text)
    assert "DVD" not in mined
    assert "Heero Yuy" in mined


def test_audio_theme_collapse_ignores_mid_episode_score():
    # A sustained music cue under a battle scene is not a theme.
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = _sung(6, 600.0) + [{"start": 1450.0, "end": 1452.0,
                               "text": "End.", "speaker": "S2"}]
    out, changed = collapse_theme_by_music_spans(rows, [(595.0, 640.0)])
    assert changed is False and out == rows


def test_audio_theme_collapse_preserves_preview_narration():
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = (_sung(4, 1360.0, 5.0)
            + [{"start": 1382.0, "end": 1387.0,
                "text": "Next episode: the Gundam appears.", "speaker": "S3"}]
            + _sung(3, 1390.0, 5.0))
    out, changed = collapse_theme_by_music_spans(rows, [(1358.0, 1410.0)])
    assert changed
    assert "Next episode: the Gundam appears." in [r["text"] for r in out]


def test_audio_theme_collapse_is_a_noop_without_spans():
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = _sung(8, 30.0)
    for spans in ([], None, [(26.0, 30.0)]):     # last: span too short
        out, changed = collapse_theme_by_music_spans(rows, spans)
        assert changed is False


def test_audio_theme_collapse_needs_enough_cues():
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = _sung(2, 30.0)
    out, changed = collapse_theme_by_music_spans(rows, [(26.0, 92.0)])
    assert changed is False


def test_two_signal_tier_collapses_a_sung_theme_the_strong_tier_cannot_see():
    """The classifier reads sung vocals as speech, so a vocal theme NEVER
    yields a strong music span (measured: 205 speech blockers, only
    mid-episode instrumentals qualified). What survives are raw fragments
    threaded between the vocal windows — enough only when the text agrees."""
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = ([{"start": 200.0, "end": 203.0, "speaker": "S1",
              "text": "Heero warns Relena about the Alliance."}]
            + [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
                "speaker": "S2", "text": t}
               for i, t in enumerate([
                   "even before i call out your name",
                   "you seem to have a lot of energy",
                   "why are you making me wait for you?",
                   "i know you ran over here.",
                   "that's just how things are.",
                   "i hate it when you do that.",
               ])])
    frags = [(1361.0, 1364.0), (1370.0, 1373.0), (1378.0, 1381.0),
             (1386.0, 1388.0)]   # 11s of music over a 29s run — ~38%
    out, changed = collapse_theme_by_music_spans(
        rows, [], raw_music=frags)
    assert changed
    texts = [r["text"] for r in out]
    assert "[♪ Ending theme ♪]" in texts
    assert not any("call out your name" in t for t in texts)
    assert "Heero warns Relena about the Alliance." in texts


def test_two_signal_tier_never_eats_a_dialogue_scene():
    """Same music fragments, but the cues are a SCENE: proper-noun-rich and
    gapped the way conversation is. Score under dialogue must survive even
    when the classifier fragments exactly like a theme."""
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    rows = [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
             "speaker": "S2", "text": t}
            for i, t in enumerate([
                "Zechs is in the atmosphere; let him know.",
                "Lieutenant Zechs, are you all right?",
                "The Marina Mother Ship is offering to salvage it.",
                "Tell Treize it sank near the JAP point.",
                "Heero must destroy the Gundam first.",
                "Relena just returned from space yesterday.",
            ])]
    frags = [(1361.0, 1364.0), (1370.0, 1373.0), (1378.0, 1381.0),
             (1386.0, 1388.0)]
    out, changed = collapse_theme_by_music_spans(rows, [], raw_music=frags)
    assert changed is False
    # And lyric-shaped cues with almost NO music under them stay too — the
    # audio signal is required, not decorative.
    lyric_rows = [{"start": 1360.0 + 5.0 * i, "end": 1364.0 + 5.0 * i,
                   "speaker": "S2", "text": f"soft line number {i}"}
                  for i in range(6)]
    out2, changed2 = collapse_theme_by_music_spans(
        lyric_rows, [], raw_music=[(1361.0, 1363.0)])   # ~7% coverage
    assert changed2 is False


def test_theme_marker_start_clamps_to_the_first_audio_onset():
    """A cue hallucinated at 0:00.000 can anchor the chorus run, but it
    cannot conjure sound: with the first classified onset at 25.4s, the
    opening marker moves there. Legitimate placements and dialogue cues are
    untouched, and no onset means no-op."""
    from backend.services.transcript_sanitize import clamp_theme_marker_starts
    rows = [
        {"start": 0.0, "end": 4.003, "speaker": "", "text": "[♪ Opening theme ♪]"},
        {"start": 87.9, "end": 92.7, "speaker": "S1", "text": "First real line."},
        {"start": 1377.1, "end": 1381.1, "speaker": "", "text": "[♪ Ending theme ♪]"},
    ]
    out, notes = clamp_theme_marker_starts([dict(r) for r in rows], 25.4)
    assert len(notes) == 1 and "0.000s → 25.400s" in notes[0]
    assert out[0]["text"] == "[♪ Opening theme ♪]" and out[0]["start"] == 25.4
    assert out[0]["end"] >= 25.4 + 1.5
    assert out[1]["start"] == 87.9                      # dialogue untouched
    assert out[2]["start"] == 1377.1                    # already after onset
    # Marker already at/after the onset → untouched; unknown onset → no-op.
    ok, n2 = clamp_theme_marker_starts(
        [{"start": 30.0, "end": 34.0, "speaker": "", "text": "[♪ Opening theme ♪]"}], 25.4)
    assert n2 == [] and ok[0]["start"] == 30.0
    same, n3 = clamp_theme_marker_starts([dict(r) for r in rows], None)
    assert n3 == [] and same[0]["start"] == 0.0


# ── Run-14 follow-ups ─────────────────────────────────────────────────────

def test_music_spans_bridge_classifier_fragments_into_a_theme():
    # The classifier works in ~2s windows (699 events across 1467 s on a
    # measured run), and a SUNG theme keeps flipping between the music and
    # speech labels because the vocal is voice. Unbridged, no span could ever
    # reach a meaningful duration bar and the audio theme pass never fired.
    from backend.services.audio_analyzer import _bridge_spans
    fragments = [(26.0 + 2.1 * i, 28.0 + 2.1 * i) for i in range(12)]
    fragments.append((53.0, 92.0))
    assert _bridge_spans(fragments, 8.0) == [(26.0, 92.0)]


def test_music_span_bridge_will_not_swallow_the_following_narration():
    # The opening narration begins about 20 s after the theme ends and is
    # real, captioned content — the bridge must tolerate a sung phrase, not
    # a scene.
    from backend.services.audio_analyzer import _bridge_spans
    theme = [(26.0 + 2.1 * i, 28.0 + 2.1 * i) for i in range(12)]
    theme.append((53.0, 92.0))
    out = _bridge_spans(theme + [(112.0, 134.0)], 8.0)
    assert out == [(26.0, 92.0), (112.0, 134.0)]


def test_bridged_spans_feed_the_audio_theme_collapse():
    # End to end: fragments in, one theme marker out.
    from backend.services.audio_analyzer import _bridge_spans
    from backend.services.transcript_sanitize import collapse_theme_by_music_spans
    spans = _bridge_spans([(26.0 + 2.1 * i, 28.0 + 2.1 * i)
                           for i in range(30)], 8.0)
    rows = _sung(8, 30.0) + [{"start": 300.0, "end": 303.0,
                              "text": "Real dialogue.", "speaker": "S2"}]
    out, changed = collapse_theme_by_music_spans(rows, spans)
    assert changed
    assert "[♪ Opening theme ♪]" in [r["text"] for r in out]
    assert "Real dialogue." in [r["text"] for r in out]


def test_onset_snap_reports_its_reasons_even_when_nothing_moves(monkeypatch, tmp_path, caplog):
    # Two measured runs logged nothing at all, leaving "found no work" and
    # "wired wrong" indistinguishable.
    import logging
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(20.0, 25.0)])
    SC._VAD_CACHE.clear()
    rows = [{"start": 21.0, "end": 24.0, "text": "already on voice"}]
    with caplog.at_level(logging.INFO, logger=SC.logger.name):
        _, shifted = SC.snap_cues_to_voice_onsets(rows, str(wav))
    assert shifted == []
    assert "voice-onset snap" in caplog.text
    assert "1 already on voice" in caplog.text


def test_repetition_burst_drops_the_measured_narration_replay():
    from backend.services.transcript_sanitize import drop_repetition_bursts
    rows = [
        {"start": 130.0, "end": 134.0,
         "text": "With overwhelming military power, they subjugated each colony."},
        {"start": 134.0, "end": 138.0,
         "text": "In After Colony 195, the operation name is Operation Meteor."},
        {"start": 138.0, "end": 142.0,
         "text": "Colonies rebel against the Union and send weapons as meteorites."},
        {"start": 210.5, "end": 210.75,
         "text": "With force, they took each colony In AC195,"},
        {"start": 210.8, "end": 211.1,
         "text": "Operation Meteor Residents oppose the Union"},
        {"start": 211.2, "end": 211.8,
         "text": "Operation Meteor starts now Overwhelming"},
        {"start": 211.9, "end": 212.5,
         "text": "power subjugates colonies. In AC 195"},
        {"start": 213.0, "end": 216.0,
         "text": "The surveillance satellites are useless."},
    ]
    out, dropped = drop_repetition_bursts(rows)
    assert len(dropped) == 1 and len(out) == 4
    texts = [r["text"] for r in out]
    assert "The surveillance satellites are useless." in texts
    assert not any("AC195" in t for t in texts)


def test_repetition_burst_spares_a_rapid_exchange():
    # The reference captions every one of these; they are short because the
    # exchange is fast, not because a decode looped.
    from backend.services.transcript_sanitize import drop_repetition_bursts
    rows = [
        {"start": 100.0, "end": 104.0, "text": "Drop your weapons and surrender."},
        {"start": 105.0, "end": 105.6, "text": "Yes, sir."},
        {"start": 105.7, "end": 106.3, "text": "What?!"},
        {"start": 106.4, "end": 107.0, "text": "Hurry!"},
        {"start": 107.1, "end": 107.7, "text": "Captain!"},
    ]
    out, dropped = drop_repetition_bursts(rows)
    assert out is rows and dropped == []


def test_repetition_burst_spares_short_cues_carrying_new_content():
    from backend.services.transcript_sanitize import drop_repetition_bursts
    rows = [
        {"start": 100.0, "end": 104.0,
         "text": "The Alliance is monitoring space closely."},
        {"start": 200.0, "end": 200.5, "text": "Torpedo bay flooded!"},
        {"start": 200.6, "end": 201.2, "text": "Reactor breach imminent!"},
        {"start": 201.3, "end": 201.9, "text": "Abandon the bridge!"},
    ]
    out, dropped = drop_repetition_bursts(rows)
    assert out is rows and dropped == []


def test_repetition_burst_needs_a_long_enough_run_and_prior_content():
    from backend.services.transcript_sanitize import drop_repetition_bursts
    prior = {"start": 10.0, "end": 14.0,
             "text": "Operation Meteor begins in After Colony 195 with force."}
    two = [prior,
           {"start": 100.0, "end": 100.4, "text": "Operation Meteor force"},
           {"start": 100.5, "end": 100.9, "text": "After Colony 195 begins"}]
    assert drop_repetition_bursts(two)[1] == [], "two cues is not a burst"
    # Same burst with no prior content to have copied.
    orphan = [{"start": 100.0, "end": 100.4, "text": "Operation Meteor force"},
              {"start": 100.5, "end": 100.9, "text": "After Colony 195 begins"},
              {"start": 101.0, "end": 101.4, "text": "colonies subjugated now"}]
    assert drop_repetition_bursts(orphan)[1] == []


# ── Run-15 regressions ─────────────────────────────────────────────────────

def test_bridge_does_not_weld_score_fragments_across_narration():
    """The run-15 defect: 2 s score fragments under the opening narration
    bridged into one 142-209 s "music" span, and the theme collapse then
    deleted sixty-five seconds of captioned dialogue."""
    from backend.services.audio_analyzer import _bridge_spans
    music = [(142 + 6 * i, 142 + 6 * i + 2) for i in range(11)]
    speech = [(142 + 6 * i + 2, 142 + 6 * i + 6) for i in range(11)]
    assert len(_bridge_spans(music, 8.0)) == 1, "unblocked, they weld"
    assert len(_bridge_spans(music, 8.0, blockers=speech)) == 11


def test_bridge_still_assembles_an_instrumental_bed():
    from backend.services.audio_analyzer import _bridge_spans, _covered_seconds
    bed = [(26 + 2 * i, 26 + 2 * i + 2) for i in range(33)]
    out = _bridge_spans(bed, 8.0, blockers=[])
    assert len(out) == 1 and out[0] == (26.0, 92.0)
    a, b = out[0]
    assert _covered_seconds(bed, a, b) == b - a


def test_covered_seconds_handles_overlap_and_clipping():
    from backend.services.audio_analyzer import _covered_seconds
    assert _covered_seconds([(0, 10), (5, 15)], 0, 20) == 15.0
    assert _covered_seconds([(0, 10)], 5, 20) == 5.0
    assert _covered_seconds([], 0, 10) == 0.0
    assert _covered_seconds([(30, 40)], 0, 10) == 0.0


def test_asr_boilerplate_catches_the_leaked_header():
    from backend.services.transcript_sanitize import looks_like_asr_boilerplate
    assert looks_like_asr_boilerplate("] sync:20 plain:no-commentary The")
    assert looks_like_asr_boilerplate("Subtitles by the Amara.org community")
    assert looks_like_asr_boilerplate("Synced and corrected by someone")


def test_asr_boilerplate_spares_ordinary_dialogue():
    from backend.services.transcript_sanitize import looks_like_asr_boilerplate
    for line in ("Zechs is in the atmosphere; let him know.",
                 "It's 10:30 already.",
                 "Relena: what's your name?",
                 "The odds are 3:1 against us.",
                 "Mr. Darlian!",
                 ""):
        assert not looks_like_asr_boilerplate(line), line


def test_junk_filter_drops_boilerplate_cue():
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": 210.9, "end": 214.1,
             "text": "] sync:20 plain:no-commentary The"},
            {"start": 214.1, "end": 217.6,
             "text": "The surveillance satellites are useless."}]
    kept, dropped = drop_junk_cues(rows)
    assert len(kept) == 1 and "boilerplate" in dropped[0]


def test_duplicate_theme_markers_collapse_to_the_first():
    from backend.services.transcript_sanitize import dedupe_theme_markers
    rows = [{"start": 30.4, "end": 34.4, "text": "[♪ Opening theme ♪]"},
            {"start": 112.2, "end": 114.0, "text": "But the Alliance..."},
            {"start": 142.2, "end": 146.1, "text": "[♪ Opening theme ♪]"},
            {"start": 1350.2, "end": 1354.2, "text": "[♪ Ending theme ♪]"}]
    out = dedupe_theme_markers(rows)
    assert [r["start"] for r in out] == [30.4, 112.2, 1350.2]


def test_dedupe_theme_markers_leaves_other_markers_alone():
    from backend.services.transcript_sanitize import dedupe_theme_markers
    rows = [{"start": 685.0, "end": 690.0, "text": "[♪ music ♪]"},
            {"start": 1272.0, "end": 1279.0, "text": "[♪ music ♪]"}]
    assert len(dedupe_theme_markers(rows)) == 2


def test_tail_echo_window_catches_the_repeated_preview():
    from backend.services.transcript_sanitize import suppress_echo_cues
    rows = [{"start": 1379.7, "end": 1385.2, "speaker": "",
             "text": "The Gundam sank to the ocean floor and they have "
                     "started recovering it now"},
            {"start": 1445.2, "end": 1451.4, "speaker": "",
             "text": "Mobile Suit Gundam Wing Episode 2"},
            {"start": 1451.5, "end": 1458.3, "speaker": "",
             "text": "The Gundam sank to the ocean floor, and they've "
                     "started recovering it now"}]
    kept, dropped = suppress_echo_cues(rows)
    assert len(kept) == 2 and len(dropped) == 1
    assert "1451.50s" in dropped[0]


def test_wide_echo_window_is_confined_to_the_tail():
    """Mid-episode, a line 70 s later is a callback, not decode residue."""
    from backend.services.transcript_sanitize import suppress_echo_cues
    rows = [{"start": 300.0, "end": 305.0, "speaker": "",
             "text": "The Gundam sank to the ocean floor and they have "
                     "started recovering it now"},
            {"start": 375.0, "end": 380.0, "speaker": "",
             "text": "The Gundam sank to the ocean floor, and they've "
                     "started recovering it now"},
            {"start": 1400.0, "end": 1405.0, "speaker": "", "text": "Fin."}]
    kept, dropped = suppress_echo_cues(rows)
    assert len(kept) == 3 and dropped == []


# ── Run-16 fixes ───────────────────────────────────────────────────────────

def _q(rows, floor=0.833, fps=23.98):
    from backend.services.subtitle_formatter import quantize_to_frames
    return quantize_to_frames([dict(r) for r in rows], fps, 1, floor)


def _invariants(rows, floor=0.833):
    gaps = [round(rows[i + 1]["start"] - rows[i]["end"], 4)
            for i in range(len(rows) - 1)]
    return {
        "overlaps": sum(1 for g in gaps if g < 0),
        "ordered": all(rows[i]["start"] <= rows[i + 1]["start"]
                       for i in range(len(rows) - 1)),
        "under": sum(1 for r in rows if r["end"] - r["start"] < floor - 1e-6),
    }


def test_frame_quantize_no_longer_mints_one_frame_cues():
    """The run-16 defect: enforce_readability extended a cue to the floor, the
    extension overlapped its neighbour, and quantizing cut it back to 0.042s —
    one frame, holding 39 characters at 929 cps."""
    rows = [{"start": 468.5, "end": 469.349},
            {"start": 469.391, "end": 470.224},
            {"start": 469.475, "end": 472.0}]
    legacy = _q(rows, floor=0.0)
    assert round(legacy[1]["end"] - legacy[1]["start"], 3) == 0.042
    out = _q(rows)
    assert out[1]["end"] - out[1]["start"] >= 0.833
    assert _invariants(out) == {"overlaps": 0, "ordered": True, "under": 0}


def test_frame_quantize_never_trades_the_floor_for_an_overlap():
    """A run too packed for any cue to reach the floor leaves cues short —
    never overlapping. Zero overlaps is the invariant that already matches the
    professional reference and must survive this pass."""
    packed = [{"start": 100.0 + i * 0.30, "end": 100.0 + i * 0.30 + 0.25}
              for i in range(8)]
    out = _q(packed)
    inv = _invariants(out)
    assert inv["overlaps"] == 0 and inv["ordered"]
    assert inv["under"] > 0, "this run is too dense to fix by timing alone"


def test_frame_quantize_stays_idempotent_with_the_floor_on():
    packed = [{"start": 100.0 + i * 0.30, "end": 100.0 + i * 0.30 + 0.25}
              for i in range(8)]
    once = _q(packed)
    twice = _q(once)
    assert [(r["start"], r["end"]) for r in once] == \
           [(r["start"], r["end"]) for r in twice]


def test_frame_quantize_floor_off_is_the_legacy_behaviour():
    rows = [{"start": 100.0 + i * 0.30, "end": 100.0 + i * 0.30 + 0.25}
            for i in range(6)]
    from backend.services.subtitle_formatter import quantize_to_frames
    a = quantize_to_frames([dict(r) for r in rows], 23.98, 1)
    b = _q(rows, floor=0.0)
    assert [(r["start"], r["end"]) for r in a] == [(r["start"], r["end"]) for r in b]


def test_annotation_artifacts_are_convicted():
    from backend.services.transcript_sanitize import looks_like_annotation_artifact
    for t in ("(Alarm sound)", "(音楽)", "（効果音）", "Dialogue end",
              "Music starts", "Sound effect", "Music ends"):
        assert looks_like_annotation_artifact(t), t


def test_fused_annotation_pair_is_convicted_before_the_splitter_minted_it():
    """Run-26: "Dialogue end. Sound effect" travelled as ONE cue — four
    words, over the old 3-word cap — survived the junk filter, and the
    run-on splitter then cut it into two junk cues that SHIPPED. All-words-
    in-vocabulary is the conviction test; the cap only bounds the scan."""
    from backend.services.transcript_sanitize import looks_like_annotation_artifact
    assert looks_like_annotation_artifact("Dialogue end. Sound effect")
    assert looks_like_annotation_artifact("Music starts. Dialogue begins.")
    # Six words of pure vocabulary still convicts; one real word acquits.
    assert looks_like_annotation_artifact("More dialogue music sound effects end")
    assert not looks_like_annotation_artifact("No sound came from the room")


def test_meta_editing_note_cue_is_dropped():
    """Run-26 shipped "Misspelled: 'Capturing' corrected." as a subtitle —
    the model narrating its own copy-editing. Dropped by the meta-note net;
    a character SAYING the word mid-sentence is untouched."""
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": 1, "end": 2, "text": "Misspelled: 'Capturing' corrected."},
            {"start": 3, "end": 4, "text": "You misspelled my name!"}]
    kept, dropped = drop_junk_cues(rows)
    assert [r["text"] for r in kept] == ["You misspelled my name!"]
    assert len(dropped) == 1


def test_annotation_filter_spares_dialogue_and_markers():
    from backend.services.transcript_sanitize import looks_like_annotation_artifact
    for t in ("[♪ music ♪]", "[♪ Opening theme ♪]", "Music to my ears",
              "The sound of it", "Sound the alarm!", "Effect confirmed, sir.",
              "(He whispers something and then leaves the room)", ""):
        assert not looks_like_annotation_artifact(t), t


def test_annotation_prefix_is_stripped_without_losing_the_line():
    from backend.services.transcript_sanitize import strip_annotation_prefix
    assert strip_annotation_prefix("(Emotion) Come on! Hurry up!") == "Come on! Hurry up!"
    assert strip_annotation_prefix("(Ren) T-, t-,") == "T-, t-,"
    # A cue that is ONLY a tag is left for the artifact test, not gutted here.
    assert strip_annotation_prefix("(Alarm sound)") == "(Alarm sound)"
    assert strip_annotation_prefix("Come on!") == "Come on!"


def test_bare_music_tag_folds_onto_the_styled_marker():
    from backend.services.transcript_sanitize import normalize_markers
    rows = [{"start": 401.2, "end": 405.2, "text": "[Music]"},
            {"start": 684.9, "end": 689.9, "text": "[♪ music ♪]"},
            {"start": 692.2, "end": 696.2, "text": "[Music]"}]
    out, notes = normalize_markers(rows)
    assert [r["text"] for r in out] == ["[♪ music ♪]", "[♪ music ♪]"]
    assert any("adjacent" in n for n in notes)


def test_distant_music_markers_both_survive():
    from backend.services.transcript_sanitize import normalize_markers
    rows = [{"start": 100.0, "end": 104.0, "text": "[♪ music ♪]"},
            {"start": 900.0, "end": 904.0, "text": "[♪ music ♪]"}]
    out, _ = normalize_markers(rows)
    assert len(out) == 2


def test_junk_filter_drops_annotations_and_keeps_their_dialogue():
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": 1, "end": 2, "text": "(Alarm sound)"},
            {"start": 3, "end": 4, "text": "(Emotion) Come on! Hurry up!"},
            {"start": 5, "end": 6, "text": "Dialogue end"},
            {"start": 7, "end": 8, "text": "[♪ music ♪]"},
            {"start": 9, "end": 10, "text": "It is a real line of dialogue."}]
    kept, dropped = drop_junk_cues(rows)
    assert [r["text"] for r in kept] == [
        "Come on! Hurry up!", "[♪ music ♪]", "It is a real line of dialogue."]
    assert len(dropped) == 2


def test_restatement_gate_only_touches_recovered_cues():
    """Scoped by provenance, not by threshold. Measured: run whole-track, this
    test deletes 4 genuine lines from the professional reference while catching
    4 restatements — a one-to-one trade against the track we are matching."""
    from backend.services.transcript_sanitize import drop_restatement_cues
    rows = [
        {"start": 1400.0, "end": 1404.0, "speaker": "",
         "text": "Another shadow emerges from the depths."},
        {"start": 1404.5, "end": 1408.0, "speaker": "",
         "text": "in the darkness of the deep sea."},
        {"start": 1408.5, "end": 1410.0, "speaker": "", "recovered": True,
         "text": "Another shadow emerges from depths"},
    ]
    kept, dropped = drop_restatement_cues([dict(r) for r in rows])
    assert len(kept) == 2 and len(dropped) == 1
    # The identical cue WITHOUT the recovered stamp is untouchable.
    unstamped = [dict(r) for r in rows]
    unstamped[2].pop("recovered")
    assert drop_restatement_cues(unstamped)[1] == []


def test_restatement_gate_spares_a_recovered_cue_that_adds_content():
    from backend.services.transcript_sanitize import drop_restatement_cues
    rows = [
        {"start": 100.0, "end": 103.0, "speaker": "",
         "text": "The Alliance is watching the colonies closely."},
        {"start": 104.0, "end": 107.0, "speaker": "", "recovered": True,
         "text": "Their carrier reached the eastern seaboard at dawn."},
    ]
    assert drop_restatement_cues([dict(r) for r in rows])[1] == []


def test_merge_recovered_stamps_provenance():
    from backend.services.vocal_gap_recovery import merge_recovered
    existing = [{"start": 0.0, "end": 2.0, "text": "An existing line."}]
    rec = [{"start": 10.0, "end": 12.0, "text": "A recovered line."}]
    out, added = merge_recovered(existing, rec)
    assert added == 1
    assert [r.get("recovered", False) for r in out] == [False, True]


def test_served_model_mismatch_detection():
    from backend.services.reframer_audio import _model_differs
    assert _model_differs("medium", "large-v3-turbo")
    assert _model_differs("small", "medium")
    for served, req in (("large-v3-turbo", "large-v3-turbo"),
                        ("ggml-large-v3-turbo.bin", "large-v3-turbo"),
                        ("whisper-large-v3-turbo", "large-v3-turbo"),
                        ("Systran/faster-whisper-large-v3", "large-v3"),
                        ("", "large-v3-turbo")):
        assert not _model_differs(served, req), (served, req)


# ── Run-17 fixes ───────────────────────────────────────────────────────────

def test_bare_meta_prose_is_convicted():
    from backend.services.transcript_sanitize import looks_like_annotation_artifact
    for t in ("More dialogue", "Dialogue end", "Music starts", "Sound effect"):
        assert looks_like_annotation_artifact(t), t


def test_qualifier_alone_convicts_nothing():
    from backend.services.transcript_sanitize import looks_like_annotation_artifact
    for t in ("Final line of defence, sir!", "More dialogue is needed",
              "No more!", "A sound plan.", "The sound of it"):
        assert not looks_like_annotation_artifact(t), t


def test_trailing_annotation_is_stripped_from_real_dialogue():
    from backend.services.transcript_sanitize import strip_trailing_annotation
    assert strip_trailing_annotation(
        "starting new lives in space colonies Dialogue end."
    ) == "starting new lives in space colonies"
    assert strip_trailing_annotation(
        "Get to the shelter now Music ends") == "Get to the shelter now"
    # An ordinary sentence that merely mentions music keeps every word.
    for t in ("Get to the shelter now.", "The music ends at dawn, sir."):
        assert strip_trailing_annotation(t) == t


def test_junk_filter_strips_both_ends_and_keeps_the_line():
    from backend.services.transcript_sanitize import drop_junk_cues
    rows = [{"start": 1, "end": 2,
             "text": "starting new lives in space colonies Dialogue end."},
            {"start": 3, "end": 4, "text": "More dialogue"},
            {"start": 5, "end": 6, "text": "[♪ music ♪]"},
            {"start": 7, "end": 8, "text": "Get to the shelter now."}]
    kept, dropped = drop_junk_cues(rows)
    assert [r["text"] for r in kept] == [
        "starting new lives in space colonies", "[♪ music ♪]",
        "Get to the shelter now."]
    assert len(dropped) == 1


def test_roster_keeps_every_real_romanization():
    """The consonant skeleton is now mandatory; these must all still pass."""
    from backend.services.canonical_names import _canonical_sounds_plausible
    for kana, name in (("セックス", "Zechs"), ("ゼクス", "Zechs"),
                       ("ヒイロ", "Heero"), ("リリーナ", "Relena"),
                       ("カトル", "Quatre"), ("トロワ", "Trowa"),
                       ("ウーフェイ", "Wufei"), ("エアリーズ", "Aries"),
                       ("トレーズ", "Treize"), ("リーオー", "Leo"),
                       ("デュオ", "Duo"), ("ガンダニュウム", "Gundanium")):
        assert _canonical_sounds_plausible(kana, name), (kana, name)


def test_roster_rejects_a_name_whose_consonants_disagree():
    """オラゴン scored 0.46 against "Emotion" on the full-string ratio and
    shipped a literal "(Emotion)" tag into subtitles. Consonants: 0.33."""
    from backend.services.canonical_names import _canonical_sounds_plausible
    for kana, name in (("オラゴン", "Emotion"), ("エアリーズ", "Peacecraft"),
                       ("レン", "Heero"), ("マリーナ", "Relena")):
        assert not _canonical_sounds_plausible(kana, name), (kana, name)


def test_roster_rejects_initials_and_labels_as_names():
    from backend.services.canonical_names import _sanitize_mapping
    terms = ["Just", "Love", "Wing", "Uing", "Dorian"]
    out = _sanitize_mapping(
        {"Just": "J", "Love": "L2", "Wing": "W1ng",
         "Uing": "Wing", "Dorian": "Dorlian"}, terms)
    assert out == {"Uing": "Wing", "Dorian": "Dorlian"}


# ── Glossary as a spelling authority over the subtitle text ────────────────

_GLOSS = ["Heero Yuy", "Relena Darlian", "Zechs Merquise", "Duo Maxwell",
          "Trowa Barton", "Quatre Winner", "Wufei Chang", "Treize Khushrenada",
          "Septem", "Marina", "Gundanium", "Gundam", "Deathscythe", "Leo",
          "Aries", "Cancer"]


def test_glossary_respells_misspelled_names_in_the_text():
    """The roster declined every candidate on a measured run while the
    glossary held the right spellings. Apply it to the text directly."""
    from backend.services.canonical_names import respell_text_from_glossary
    out, n, samples = respell_text_from_glossary(
        ["I am Relena Dorlian.",
         "General Septum is waiting.",
         "Maybe the Aires mobile suit would be better?"], _GLOSS)
    assert n == 3, samples
    assert out == ["I am Relena Darlian.",
                   "General Septem is waiting.",
                   "Maybe the Aries mobile suit would be better?"]


def test_glossary_respell_leaves_correct_names_alone():
    from backend.services.canonical_names import respell_text_from_glossary
    lines = ["Lieutenant Zechs, are you all right?",
             "My name is Relena Darlian.",
             "So, it WAS a Gundam.",
             "Five Gundams?!",
             "The Alliance's Marina is on the way.",
             "General Septem's expecting you."]
    out, n, _ = respell_text_from_glossary(list(lines), _GLOSS)
    assert n == 0 and out == lines


def test_glossary_respell_spares_plurals_and_possessives():
    """Measured on the professional reference, every near-miss hit was a
    correct plural or possessive — never an error."""
    from backend.services.canonical_names import respell_text_from_glossary
    lines = ["Five Gundams?!", "Marina's carrier is coming.",
             "Relena's birthday is tomorrow.", "Treize's subordinate lost three."]
    out, n, _ = respell_text_from_glossary(list(lines), _GLOSS)
    assert n == 0 and out == lines


def test_glossary_respell_refuses_an_ambiguous_target():
    """A near-tie between two official names means identity is unknown, and
    guessing puts one character's name onto another."""
    from backend.services.canonical_names import respell_text_from_glossary
    # "Trois" sits between Trowa and Treize; neither may be chosen.
    out, n, _ = respell_text_from_glossary(["I'll go by Trois here."], _GLOSS)
    assert n == 0 and out == ["I'll go by Trois here."]


def test_glossary_respell_ignores_sentence_initial_capitals():
    from backend.services.canonical_names import respell_text_from_glossary
    lines = ["Ladies and gentlemen.", "True enough.", "Area secured."]
    out, n, _ = respell_text_from_glossary(list(lines), _GLOSS)
    assert n == 0 and out == lines


def test_glossary_respell_is_idempotent_and_fail_soft():
    from backend.services.canonical_names import respell_text_from_glossary
    once, n1, _ = respell_text_from_glossary(["I am Relena Dorlian."], _GLOSS)
    twice, n2, _ = respell_text_from_glossary(once, _GLOSS)
    assert n1 == 1 and n2 == 0 and once == twice
    assert respell_text_from_glossary(["Dorlian here"], [])[1] == 0
    assert respell_text_from_glossary([], _GLOSS)[1] == 0


# ── Run-18 parity fixes ────────────────────────────────────────────────────

def test_ellipsis_heal_no_longer_unwraps_a_correctly_wrapped_cue():
    """heal_split_ellipsis ran AFTER the guaranteed wrap; its `\\s*` matched
    the newline and the join destroyed the break, with nothing to re-wrap
    after. That was the sole cause of every over-width line in a measured
    run — 61 chars against a reference whose 503 lines never exceed 34."""
    from backend.services.subtitle_formatter import heal_split_ellipsis

    class _S:
        def __init__(self, t):
            self.text, self.start, self.end, self.words = t, 0.0, 4.0, None

    # A doubled marker split across the wrap is healed, and the result is flat
    # (single line) so the wrap that follows can re-break it legally.
    seg = _S("Of course, that\n… …goes without saying.")
    heal_split_ellipsis([seg])
    assert "\n" not in seg.text
    assert seg.text == "Of course, that goes without saying."
    # A legitimately wrapped continuation pair keeps its break untouched.
    keep = _S("…and had already captured\nthe Gundam securely Heero…")
    heal_split_ellipsis([keep])
    assert keep.text == "…and had already captured\nthe Gundam securely Heero…"


def test_quantize_pools_slack_across_a_short_run():
    """Cue 208 shipped at 0.625s because its immediate neighbour was itself at
    the floor, while the cue after that held 13 frames of slack."""
    from backend.services.subtitle_formatter import quantize_to_frames
    rows = [{"start": 902.5, "end": 903.125}, {"start": 903.167, "end": 903.999},
            {"start": 904.041, "end": 904.874}, {"start": 904.916, "end": 906.292}]
    legacy = quantize_to_frames([dict(r) for r in rows], 23.98, 1, 0.0)
    assert round(legacy[0]["end"] - legacy[0]["start"], 3) == 0.626
    out = quantize_to_frames([dict(r) for r in rows], 23.98, 1, 0.833)
    assert all(r["end"] - r["start"] >= 0.833 for r in out)
    gaps = [round(out[i + 1]["start"] - out[i]["end"], 4)
            for i in range(len(out) - 1)]
    assert all(g > 0 for g in gaps) and min(gaps) == 0.041
    again = quantize_to_frames([dict(r) for r in out], 23.98, 1, 0.833)
    assert [(r["start"], r["end"]) for r in out] == \
           [(r["start"], r["end"]) for r in again]


def test_quantize_leaves_an_unfixable_run_alone():
    """A run with no slack anywhere stays a fixed point rather than walking."""
    from backend.services.subtitle_formatter import quantize_to_frames
    packed = [{"start": 100.0 + i * 0.30, "end": 100.0 + i * 0.30 + 0.25}
              for i in range(8)]
    out = quantize_to_frames([dict(r) for r in packed], 23.98, 1, 0.833)
    gaps = [round(out[i + 1]["start"] - out[i]["end"], 4)
            for i in range(len(out) - 1)]
    assert all(g > 0 for g in gaps)
    assert sum(1 for r in out if r["end"] - r["start"] < 0.833) > 0
    again = quantize_to_frames([dict(r) for r in out], 23.98, 1, 0.833)
    assert [(r["start"], r["end"]) for r in out] == \
           [(r["start"], r["end"]) for r in again]


def test_trailing_paren_tag_is_stripped_but_a_whispered_aside_survives():
    from backend.services.transcript_sanitize import strip_trailing_annotation
    assert strip_trailing_annotation("Brrr (sound effect)") == "Brrr"
    assert strip_trailing_annotation("Captain! (alarm)") == "Captain!"
    long_aside = "He nods (he whispers something and then leaves the room)"
    assert strip_trailing_annotation(long_aside) == long_aside
    assert strip_trailing_annotation("Get to the shelter now.") == \
        "Get to the shelter now."


def test_echo_surface_floor_spares_dialogue_but_keeps_the_collapse():
    """Shared vocabulary alone must not convict: on the professional
    reference that deleted eight real cues. Two translations of ONE line
    share wording as well, so the surface floor separates them."""
    from backend.services.transcript_sanitize import suppress_echo_cues
    # Genuine four-rendering duplicate cluster — must collapse to one.
    dup = [{"start": 202.9, "end": 205.2, "speaker": "S1",
            "text": "Reporting meteor strikes."},
           {"start": 205.2, "end": 208.7, "speaker": "S1",
            "text": "Meteors falling, they say It's being reported as a meteor strike"},
           {"start": 208.7, "end": 211.2, "speaker": "S1",
            "text": "The meteorites are being reported."},
           {"start": 211.2, "end": 214.2, "speaker": "S1",
            "text": "Reported as falling meteorites"}]
    kept, dropped = suppress_echo_cues([dict(r) for r in dup])
    assert len(dropped) >= 2
    assert sum(1 for r in kept if "meteor" in r["text"].lower()) == 1
    # Two lines sharing vocabulary but not WORDING are no longer convicted on
    # containment alone — that combination is ordinary dialogue.
    real = [{"start": 100.0, "end": 104.0, "speaker": "",
             "text": "The colonies are demanding their independence now."},
            {"start": 104.5, "end": 107.5, "speaker": "",
             "text": "Independence for the colonies? Never."}]
    assert suppress_echo_cues([dict(r) for r in real])[1] == []
    # Honest bound: the floor REDUCES reference damage, it does not remove it.
    # A short reply whose wording is literally contained in the line before it
    # still scores above the floor and is still dropped.
    still = [{"start": 200.0, "end": 204.0, "speaker": "",
              "text": "You're wasting the military's valuable combat resources!"},
             {"start": 204.5, "end": 206.5, "speaker": "",
              "text": "Valuable combat resources?"}]
    assert len(suppress_echo_cues([dict(r) for r in still])[1]) == 1


def test_series_glossary_falls_back_when_no_hint_is_configured():
    """series_roster_terms() returns [] without TRANSLATION_SERIES_HINT, which
    made the text respeller dead code on every run that had no hint."""
    from backend.services import canonical_names as cn
    cn._cache_put("glossary:job-x", {"a": "Heero Yuy", "b": "Relena Darlian"})
    got = cn.series_glossary_for_job("job-x")
    assert sorted(got) == ["Heero Yuy", "Relena Darlian"]
    assert cn.series_glossary_for_job("no-such-job") == cn.series_roster_terms()


def test_series_glossary_survives_a_restart_via_the_durable_store(tmp_path,
                                                                  monkeypatch):
    """The cast list must be reachable BEFORE the series is identified.

    Identification runs after transcription, so within one run the names are
    already mangled by the time we know whose they are. The durable pointer is
    what lets the next run bias Whisper's decoder — process-local state cannot,
    because it is empty at the moment transcription starts.
    """
    from backend.services import canonical_names as cn
    store = tmp_path / "glossary.json"
    monkeypatch.setattr(cn, "_GLOSSARY_STORE_PATH", str(store))
    monkeypatch.setattr(cn, "_LAST_SERIES_KEY", "")   # simulate a fresh process
    cn._glossary_store_save({"mobile suit gundam wing":
                             ["Heero Yuy", "Relena Darlian", "Zechs Merquise"]})
    # No pointer yet → nothing to bias with, exactly as on a first-ever run.
    assert cn.series_glossary_for_job() == cn.series_roster_terms()
    # A resolved series writes the pointer, and it is written only once.
    cn._remember_series(cn._glossary_store_load(), "mobile suit gundam wing")
    assert cn._glossary_store_load()["__last_series__"] == \
        "mobile suit gundam wing"
    assert sorted(cn.series_glossary_for_job()) == \
        ["Heero Yuy", "Relena Darlian", "Zechs Merquise"]
    # An explicit series argument still wins over the remembered one.
    cn._glossary_store_save({**cn._glossary_store_load(),
                             "other show": ["Amuro Ray"]})
    assert cn.series_glossary_for_job(series="Other Show") == ["Amuro Ray"]


def test_asr_vocabulary_bias_reads_the_resolved_glossary():
    """The decoder bias must read the resolved chain, not the configured hint.

    ``series_roster_terms()`` is empty without TRANSLATION_SERIES_HINT, which is
    unset by default — so on every un-hand-configured run this biasing was off,
    and names like Quatre came back as "Kato". No downstream spelling test can
    undo that: the two share 40% of their letters, and correcting on sound
    alone maps one character's name onto another's.
    """
    import inspect
    from backend.services import reframer_audio, cloud_transcription
    for fn in (reframer_audio._vocab_bias_kwargs,
               cloud_transcription._vocab_prompt):
        src = inspect.getsource(fn)
        assert "series_glossary_for_job" in src
        assert "series_roster_terms" not in src


def test_deterministic_text_options_pin_a_seed_and_respect_qwen3():
    """Two identical runs produced materially different subtitle tracks (334
    vs 352 cues; the ED collapsed on one and shipped as dialogue on the
    other) — sampled decoding was the entry point. Non-Qwen3 models decode
    greedily; Qwen3 keeps its anti-repetition profile (its card warns that
    near-greedy decoding loops — measured here too) and relies on the pinned
    seed alone for reproducibility."""
    from backend.services.local_models import deterministic_text_options
    o = deterministic_text_options("qwen2.5:14b")
    assert o["seed"] == 42 and o["temperature"] == 0.0 and o["top_p"] == 1.0
    q3 = deterministic_text_options("qwen3:4b-instruct-2507-q4_K_M")
    assert q3["seed"] == 42
    assert "temperature" not in q3 and "top_p" not in q3


def test_transcript_shaping_callers_request_deterministic_decoding():
    """The seed only helps where it is actually sent: the translator's chat
    options, the polish batch, and the gap-recovery per-cue translation are
    the three passes whose output ships as subtitles."""
    import inspect
    from backend.services import translator, transcript_polisher
    assert "deterministic_text_options" in inspect.getsource(
        translator._translate_batch_via_ollama)
    assert "deterministic=True" in inspect.getsource(
        transcript_polisher._polish_batch)
    from backend.services.providers import ollama_provider
    src = inspect.getsource(ollama_provider.OllamaProvider._call_text)
    assert "deterministic_text_options" in src


def test_kana_mining_and_romaji_aliases_reach_the_unreachable_garbles():
    """カトル→"Kato" shares 40% of its letters with "Quatre" — no orthographic
    threshold reaches it, and phonetic matching mapped Trois onto Treize.
    The kana reading is the evidence both of those lacked: mined from the
    same wiki text as the names, romanized the way Whisper's translate head
    actually writes it."""
    from backend.services import canonical_names as cn
    text = ("Quatre Raberba Winner (カトル・ラバーバ・ウィナー, Katoru) pilots "
            "Sandrock. Relena Darlian (リリーナ・ドーリアン, Rirīna) appears. "
            "Trowa Barton (トロワ・バートン, Torowa) and the Aries (エアリーズ) suit. "
            "Duo Maxwell (デュオ・マックスウェル, Dyuo).")
    pairs = cn._mine_kana_pairs(text)
    assert pairs["カトル"] == "Quatre"
    assert pairs["ドーリアン"] == "Darlian"
    assert pairs["エアリーズ"] == "Aries"          # leading article stripped
    assert cn._kana_reading_romaji("カトル") == "katoru"
    assert cn._kana_reading_romaji("ダーリアン") == "darian"
    assert cn._kana_reading_romaji("デュオ") == "dyuo"
    aliases = cn._alias_map_from_pairs(pairs)
    assert aliases["kato"] == "Quatre"
    assert aliases["dorian"] == "Darlian"
    # A kana form claimed by two different names is ambiguous — dropped.
    two = cn._mine_kana_pairs(
        "Alpha One (アルファ) fights. Alpha Prime (アルファ) returns.")
    assert "アルファ" not in two


def test_respeller_alias_path_fixes_kato_and_dorian_but_not_trois():
    from backend.services.canonical_names import respell_text_from_glossary
    glossary = ["Quatre Raberba Winner", "Relena Darlian", "Trowa Barton",
                "Treize Khushrenada", "Zechs Merquise"]
    aliases = {"katoru": "Quatre", "kato": "Quatre", "darian": "Darlian",
               "dorian": "Darlian", "torowa": "Trowa", "torezu": "Treize"}
    texts = ["This is Kato.",
             "Mr. Dorian, sir, I’ve been waiting for you.",
             "But Trois wouldn't have acted so irresponsibly.",
             "Calm down, Trowa."]
    out, n, samples = respell_text_from_glossary(texts, glossary, aliases)
    assert out[0] == "This is Quatre."
    assert out[1].startswith("Mr. Darlian")
    # "Trois" reaches no alias at 0.8 with a first-letter guard — mapping it
    # onto Treize was the exact phonetic failure this path must not repeat.
    assert out[2] == texts[2]
    assert out[3] == texts[3]                       # already official


def test_kana_pairs_survive_a_restart_via_the_durable_store(tmp_path,
                                                            monkeypatch):
    from backend.services import canonical_names as cn
    store = tmp_path / "glossary.json"
    monkeypatch.setattr(cn, "_GLOSSARY_STORE_PATH", str(store))
    monkeypatch.setattr(cn, "_LAST_SERIES_KEY", "")
    cn._glossary_store_save({
        "mobile suit gundam wing": ["Quatre Raberba Winner"],
        "__last_series__": "mobile suit gundam wing",
        cn._KANA_STORE_PREFIX + "mobile suit gundam wing": {"カトル": "Quatre"},
    })
    assert cn.kana_pairs_for_job() == {"カトル": "Quatre"}
    assert cn.kana_aliases_for_job()["kato"] == "Quatre"
    # No pointer and no explicit series → nothing to answer with.
    cn._glossary_store_save({"other": ["X"]})
    assert cn.kana_pairs_for_job() == {}


def test_condense_threshold_leaves_merely_brisk_cues_alone():
    """At 20 cps the condenser rewrote cues that were readable, taking total
    text 656 characters below the professional reference and reflowing shortened
    two-line cues to one line. The trigger must sit well clear of the target the
    rewrite is asked to hit, or the pass eats prose it was never meant to see.
    """
    from backend.config import settings
    assert settings.SUBTITLE_CONDENSE_CPS == 24.0
    assert settings.SUBTITLE_CONDENSE_CPS > settings.SUBTITLE_CONDENSE_TARGET_CPS
    # A 21-cps cue — brisk, still under the trigger.
    brisk = "Colonies do not surrender to Alliance threats."   # 45 chars
    assert len(brisk) / 2.1 < settings.SUBTITLE_CONDENSE_CPS
