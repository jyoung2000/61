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
    # hard CPS ceiling an approximate cut is the smaller error.
    from backend.services.subtitle_formatter import _split_segment
    seg = TranscriptSegment(
        start=482.44, end=483.40, speaker="Speaker 1",
        text="which is faster than O, would that be more suitable for this")
    pieces = _split_segment(
        seg, target_cps=20.0, max_chars_per_line=34,
        word_timed_split_only=True)
    assert len(pieces) > 1, "an unreadable cue must be cut even without words"
    assert " ".join(p.text for p in pieces).split() == seg.text.split()


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
