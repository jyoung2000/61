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
