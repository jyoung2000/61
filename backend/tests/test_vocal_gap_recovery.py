"""Targeted vocal-separation gap recovery: pure logic + the post-COMPLETE
merge path. The heavy pieces (Demucs, remote ASR) are exercised only through
their fail-soft seams — recovery must never be able to break a finished job.
"""

import asyncio
import json

import pytest

from backend.services.vocal_gap_recovery import (
    _JUNK_RE,
    _clip_to_gap,
    _concat_offsets,
    find_coverage_gaps,
    merge_recovered,
)


def test_concat_offsets_track_silence_separators():
    # All spans separated into ONE Demucs pass: each clip's start offset inside
    # the concatenation = sum of prior clip durations + one separator between.
    assert _concat_offsets([20.0, 15.0, 32.0], 1.0) == [0.0, 21.0, 37.0]
    assert _concat_offsets([10.0], 1.0) == [0.0]   # lone span → no separator
    assert _concat_offsets([], 1.0) == []


def _track(*spans):
    return [{"start": float(a), "end": float(b), "text": t}
            for a, b, t in spans]


def test_gap_finder_interior_holes_only():
    segs = _track((0, 4, "a"), (5, 9, "b"), (40, 44, "c"), (44, 300, "d"))
    # 9→40 is a hole; nothing before the first or after the last cue counts.
    gaps = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=2.0)
    assert gaps == [(7.0, 42.0)]


def test_gap_finder_ignores_small_holes_and_respects_caps():
    # 136s / 196s holes are non-speech SCENES — the default per-gap cap
    # (max_span_s=45) skips both; only short buried-dialogue holes qualify.
    segs = _track((0, 10, "a"), (14, 56, "b"),      # 4s hole — too small
                  (60, 64, "c"),                    # 4s hole — too small
                  (94, 98, "d"),                    # 30s hole — kept
                  (200, 204, "e"),                  # 102s hole — too long
                  (400, 404, "f"))                  # 196s hole — too long
    gaps = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=0.0,
                              max_spans=8, max_total_s=150.0)
    assert gaps == [(64.0, 94.0)]


def test_gap_finder_per_span_cap_can_be_disabled():
    # With the per-gap cap OFF, every hole is eligible and only the seconds
    # budget limits the picks.
    #
    # This used to assert largest-first (taking the 136s hole and skipping the
    # 196s one). The priority is now SMALLEST-first, which inverts it: the 50s
    # hole is taken and the 136s one no longer fits the 150s budget. That is a
    # deliberate change — largest-first contradicted this module's own premise
    # that music-buried dialogue arrives as SHORT holes, and on a reference
    # episode it spent the entire budget on 14-25s spans that held only theme
    # music while every real miss (2.1-5.7s) went unexamined.
    segs = _track((0, 10, "a"), (60, 64, "c"), (200, 204, "d"), (400, 404, "e"))
    gaps = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=0.0,
                              max_spans=8, max_total_s=150.0, max_span_s=0)
    assert gaps == [(10.0, 60.0)]
    # A budget big enough for two takes the two SMALLEST, in timeline order.
    gaps2 = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=0.0,
                               max_spans=8, max_total_s=200.0, max_span_s=0)
    assert gaps2 == [(10.0, 60.0), (64.0, 200.0)]


def test_gap_finder_skips_the_long_music_scene():
    # The run-1 case: a single 179s hole (76:34-79:33) that cost 8 min of
    # Demucs for zero recovery. It must not be scheduled by default.
    segs = _track((0, 4594, "dialogue"), (4773, 9000, "more dialogue"))
    assert find_coverage_gaps(segs, min_gap_s=8.0, pad_s=2.0) == []


def test_gap_finder_needs_two_cues():
    assert find_coverage_gaps(_track((0, 5, "only"))) == []
    assert find_coverage_gaps([]) == []


def test_clip_to_gap_strips_the_padded_leadin():
    gap = (95.0, 132.0)  # padded by 2s on each side → real hole 97..130
    inside = _clip_to_gap({"start": 100.0, "end": 104.0, "text": "x"}, gap, 2.0)
    assert inside == {"start": 100.0, "end": 104.0, "text": "x"}
    # A cue entirely inside the lead-in pad that re-hears an EXISTING cue —
    # drop. (Pad decodes with NO covering cue are kept now: the pad guards
    # against double-captioning, not against recovery itself.)
    assert _clip_to_gap({"start": 95.2, "end": 96.8, "text": "x"}, gap, 2.0,
                        existing=[{"start": 94.0, "end": 96.5, "text": "y"}]) is None
    # Straddling the pad boundary: clipped to the real hole.
    edge = _clip_to_gap({"start": 96.0, "end": 101.0, "text": "x"}, gap, 2.0)
    assert edge["start"] == 97.0 and edge["end"] == 101.0


def test_merge_is_additive_and_drops_boundary_echoes():
    existing = _track((0, 4, "The Colony Summit begins."),
                      (40, 44, "General Septem is waiting."))
    merged, added = merge_recovered(existing, [
        {"start": 10.0, "end": 13.0, "text": "Any comments, Vice Minister?"},
        {"start": 14.0, "end": 17.0, "text": "Any comment, Vice Minister?!"},  # dup of accepted
        {"start": 38.0, "end": 41.0, "text": "General Septem is waiting"},     # echo of neighbor
    ])
    assert added == 1
    assert len(merged) == 3
    # Sorted by start, existing cues untouched.
    assert [m["text"] for m in merged] == [
        "The Colony Summit begins.",
        "Any comments, Vice Minister?",
        "General Septem is waiting.",
    ]


def test_junk_hallucinations_regex():
    assert _JUNK_RE.match("Thank you for watching.")
    assert _JUNK_RE.match("ご視聴ありがとうございました")
    assert not _JUNK_RE.match("Thank you, Lieutenant.")


def test_post_complete_recovery_merges_and_translates(monkeypatch, tmp_path):
    from backend import database
    from backend.models import JobResult, JobStatus, TranscriptSegment
    from backend.services import pipeline as pl

    monkeypatch.setattr(database, "_job_dir",
                        lambda job_id: str(tmp_path / job_id))
    now = "2026-07-21T00:00:00+00:00"
    d = tmp_path / "j-rec"
    d.mkdir()
    src = [TranscriptSegment(start=float(i * 10), end=float(i * 10 + 4),
                             text=f"セリフ{i}", speaker="Speaker 1")
           for i in range(6)]
    tt = [TranscriptSegment(start=s.start, end=s.end,
                            text=f"line {i}", speaker="Speaker 1")
          for i, s in enumerate(src)]
    job = JobResult(job_id="j-rec", filename="v.mp4", file_path="/x/v.mp4",
                    status=JobStatus.COMPLETE, created_at=now, updated_at=now,
                    transcript=src, translated_transcript=tt,
                    subtitle_language="en")
    (d / "job.json").write_text(job.model_dump_json())

    async def _fake_recover(job_id, audio, segments, lang, work_dir):
        return [{"start": 21.0, "end": 24.0, "text": "埋もれた台詞",
                 "speaker": "Speaker 1"}]

    monkeypatch.setattr(
        "backend.services.vocal_gap_recovery.recover_gap_dialogue",
        _fake_recover)

    events = []

    async def _fake_broadcast(job_id, msg):
        events.append(msg)

    monkeypatch.setattr(pl, "broadcast_ws", _fake_broadcast)

    class _Orch:
        async def text_completion(self, prompt, **kw):
            assert "埋もれた台詞" in prompt
            return "The buried line."

    asyncio.run(pl._post_complete_gap_recovery("j-rec", _Orch()))

    saved = json.loads((d / "job.json").read_text())
    src_texts = [s["text"] for s in saved["transcript"]]
    tt_texts = [s["text"] for s in saved["translated_transcript"]]
    assert "埋もれた台詞" in src_texts
    assert "The buried line." in tt_texts
    assert len(saved["transcript"]) == 7
    # Existing cues untouched, order preserved by start time.
    assert src_texts[0] == "セリフ0" and src_texts[-1] == "セリフ5"
    assert any(e.get("task") == "vocal_recovery"
               and e.get("status") == "complete" for e in events)


def test_post_complete_recovery_noop_without_gap_hits(monkeypatch, tmp_path):
    from backend import database
    from backend.models import JobResult, JobStatus, TranscriptSegment
    from backend.services import pipeline as pl

    monkeypatch.setattr(database, "_job_dir",
                        lambda job_id: str(tmp_path / job_id))
    now = "2026-07-21T00:00:00+00:00"
    d = tmp_path / "j-none"
    d.mkdir()
    src = [TranscriptSegment(start=float(i), end=float(i) + 0.9, text=f"t{i}",
                             speaker="Speaker 1") for i in range(6)]
    job = JobResult(job_id="j-none", filename="v.mp4", file_path="/x/v.mp4",
                    status=JobStatus.COMPLETE, created_at=now, updated_at=now,
                    transcript=src)
    before = job.model_dump_json()
    (d / "job.json").write_text(before)

    async def _fake_recover(*a, **k):
        return []

    monkeypatch.setattr(
        "backend.services.vocal_gap_recovery.recover_gap_dialogue",
        _fake_recover)
    asyncio.run(pl._post_complete_gap_recovery("j-none", None))
    assert (d / "job.json").read_text() == before  # nothing rewritten


# ── Voice-gated span selection ────────────────────────────────────────────
# The selector computes holes against OUR OWN transcript, so "no cue here"
# also covers every second of theme music. A measured run spent its whole
# 228 s budget separating song and title spans and correctly recovered
# nothing, while the five holes that did contain speech (2.1-5.7 s) were
# below the 8 s floor and never considered at all.

def _seg(s, e):
    return {"start": s, "end": e, "text": "x"}


def _track_with_holes(holes):
    """A transcript whose interior holes are exactly ``holes``."""
    segs, t = [], 10.0
    for a, b in holes:
        segs.append(_seg(t, a))
        t = b
    segs.append(_seg(t, t + 5.0))
    return segs


# Measured on the reference episode: three song holes, one narration hole and
# four dialogue holes.
_SONG = [(26.0, 41.8), (47.9, 62.0), (1358.9, 1365.1)]
_SPEECH = [(129.2, 134.9), (211.0, 214.6), (556.6, 558.7),
           (583.8, 586.2), (1001.9, 1004.7)]
_VOICE = [(a + 0.3, b - 0.3) for a, b in _SPEECH]      # VAD sees only speech


def test_voice_check_rejects_song_holes_and_keeps_speech_holes():
    segs = _track_with_holes(sorted(_SONG + _SPEECH))
    picked = find_coverage_gaps(segs, min_gap_s=2.0, pad_s=2.0, max_spans=14,
                                max_total_s=240.0, max_span_s=45.0,
                                voice_regions=_VOICE)
    assert len(picked) == len(_SPEECH), picked
    for a, _b in _SPEECH:
        assert any(g[0] <= a <= g[1] for g in picked), a
    for a, _b in _SONG:
        assert not any(g[0] <= a <= g[1] for g in picked), a
    # And it costs a fraction of the old budget.
    assert sum(b - a for a, b in picked) < 60.0


def test_the_old_floor_could_not_see_any_real_miss():
    """Pins WHY the pass never recovered anything: every measured miss was
    shorter than the 8 s floor, so no amount of budget or sorting helped."""
    segs = _track_with_holes(sorted(_SPEECH))
    assert find_coverage_gaps(segs, min_gap_s=8.0, pad_s=2.0, max_spans=14,
                              max_total_s=240.0, max_span_s=45.0) == []
    assert find_coverage_gaps(segs, min_gap_s=2.0, pad_s=2.0, max_spans=14,
                              max_total_s=240.0, max_span_s=45.0)


def test_smallest_first_so_short_holes_survive_the_budget():
    """Largest-first contradicted the module's own premise and spent the budget
    on the spans least likely to hold dialogue."""
    segs = _track_with_holes([(100.0, 103.0), (200.0, 240.0)])
    picked = find_coverage_gaps(segs, min_gap_s=2.0, pad_s=0.0, max_spans=14,
                                max_total_s=10.0, max_span_s=45.0)
    assert picked == [(100.0, 103.0)], picked


def test_no_voice_regions_leaves_selection_unfiltered():
    """Fail-soft: a box without VAD keeps the previous behaviour."""
    segs = _track_with_holes(sorted(_SONG + _SPEECH))
    assert find_coverage_gaps(segs, min_gap_s=2.0, pad_s=2.0, max_spans=14,
                              max_total_s=240.0, max_span_s=45.0,
                              voice_regions=None)


def test_overlaps_voice_measures_the_unpadded_interior():
    from backend.services.vocal_gap_recovery import _overlaps_voice
    gap = (10.0, 20.0)          # padded; interior is 12.0-18.0 at pad=2
    # Voice only inside the PAD must not qualify — those seconds already
    # belong to the neighbouring cues.
    assert not _overlaps_voice(gap, [(10.2, 11.8)], pad_s=2.0)
    assert _overlaps_voice(gap, [(14.0, 16.0)], pad_s=2.0)
    # A brush of voice below the minimum is not enough.
    assert not _overlaps_voice(gap, [(15.0, 15.1)], pad_s=2.0)
    # Malformed regions are ignored, not crashed on.
    assert not _overlaps_voice(gap, [None, ("x", "y"), ()], pad_s=2.0)


# ── Most missing dialogue is not a HOLE ────────────────────────────────────
# Measured against a reference track for the same episode: 30 of its 347 cues
# had no ClipAI counterpart, 985 characters, 93 % of the whole content deficit.
# The largest single miss was a 17-second press scrum the transcript nominally
# COVERED with two cues totalling fifteen characters — no hole to find, so a
# hole-based scan never looked at it.

def test_density_detector_finds_an_under_transcribed_voiced_run():
    segs = _track(
        (0.0, 3.0, "A perfectly ordinary line of dialogue here."),
        (5.0, 8.0, "Another perfectly ordinary line of dialogue."),
        (10.0, 13.0, "And a third ordinary line to set the median."),
        (15.0, 18.0, "A fourth ordinary line of dialogue as well."),
        (20.0, 23.0, "A fifth ordinary line of dialogue as well."),
        (25.0, 28.0, "A sixth ordinary line of dialogue as well."),
        (30.0, 33.0, "A seventh ordinary line of dialogue too."),
        (35.0, 38.0, "An eighth ordinary line of dialogue too."),
        # 40-60s: continuous voice, CONTINUOUSLY COVERED by cues (so there is
        # no hole to find), but only a handful of characters transcribed.
        (40.0, 43.0, "Over"),
        (43.0, 46.0, "Yes"),
        (46.0, 49.0, "No"),
        (49.0, 52.0, "Wait"),
        (52.0, 55.0, "First"),
        (55.0, 58.0, "Sir"),
        (58.0, 60.0, "Hm"),
        (62.0, 65.0, "Back to ordinary dialogue after the scrum."),
    )
    voice = [(40.0 + i * 0.5, 40.4 + i * 0.5) for i in range(40)]   # 40-60s
    off = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=1.0, max_spans=14,
                             max_total_s=240.0, max_span_s=45.0,
                             voice_regions=voice, density_ratio=0.0)
    on = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=1.0, max_spans=14,
                            max_total_s=240.0, max_span_s=45.0,
                            voice_regions=voice, density_ratio=0.35)
    def _covers(picked, t):
        return any(a <= t <= b for a, b in picked)
    assert not _covers(off, 46.0), "hole scan should not see a covered span"
    assert _covers(on, 46.0), f"density scan missed the scrum: {on}"


def test_density_detector_leaves_a_normally_transcribed_run_alone():
    segs = _track(*[(i * 5.0, i * 5.0 + 3.0,
                     "A perfectly ordinary line of dialogue here.")
                    for i in range(12)])
    voice = [(i * 5.0, i * 5.0 + 3.0) for i in range(12)]
    picked = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=1.0, max_spans=14,
                                max_total_s=240.0, max_span_s=45.0,
                                voice_regions=voice, density_ratio=0.35)
    assert picked == [], picked


def test_overlong_hole_is_reduced_to_its_voiced_parts_not_dropped():
    """"Longer than the cap" was a proxy for "no speech in it". On a reference
    episode it was wrong: a 70-second hole before the ending theme held two
    lines of dialogue and the "to be continued" card, and the cap dropped
    them along with the hole."""
    segs = _track((0.0, 5.0, "before the hole"), (80.0, 85.0, "after the hole"))
    voice = [(30.0, 33.0), (40.0, 42.0)]      # two utterances inside a 75s hole
    dropped = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=1.0, max_spans=14,
                                 max_total_s=240.0, max_span_s=45.0,
                                 voice_regions=None)
    kept = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=1.0, max_spans=14,
                              max_total_s=240.0, max_span_s=45.0,
                              voice_regions=voice, density_ratio=0.0)
    assert dropped == [], "a 75s hole is over the cap"
    assert kept, "the voiced parts of that hole must survive"
    assert all(b - a <= 45.0 + 2.0 for a, b in kept), kept
    assert any(a <= 31.0 <= b for a, b in kept), kept


def test_selection_prefers_the_span_holding_more_missing_speech():
    """Yield ranking, not size ranking. Size was a proxy adopted when long
    spans meant song; with both detectors feeding the selector the candidate
    list outgrew the budget and the proxy started evicting the biggest real
    misses in favour of short spans holding nothing."""
    segs = _track((0.0, 5.0, "one"), (15.0, 20.0, "two"),
                  (30.0, 35.0, "three"), (45.0, 50.0, "four"))
    # Hole 5-15s: barely any voice. Hole 35-45s: voice throughout.
    voice = [(6.0, 6.4)] + [(35.5 + i * 0.5, 35.9 + i * 0.5) for i in range(18)]
    # Budget fits exactly ONE of the two 10s holes, so the ranking decides.
    picked = find_coverage_gaps(segs, min_gap_s=4.0, pad_s=0.0, max_spans=14,
                                max_total_s=10.0, max_span_s=45.0,
                                voice_regions=voice, density_ratio=0.0)
    assert len(picked) == 1, picked
    assert picked[0][0] >= 35.0, \
        f"the voice-dense span should win the budget, not the emptier one: {picked}"


def test_chunk_splits_a_long_run_instead_of_discarding_it():
    from backend.services.vocal_gap_recovery import _chunk
    assert _chunk(0.0, 10.0, 45.0) == [(0.0, 10.0)]
    assert _chunk(0.0, 100.0, 45.0) == [(0.0, 45.0), (45.0, 90.0), (90.0, 100.0)]
    assert _chunk(0.0, 100.0, 0.0) == [(0.0, 100.0)]      # cap disabled


def test_relisten_tier_needs_no_demucs(monkeypatch, tmp_path):
    """The cheap tier is what lets recovered lines reach the shipped file: it
    runs inside the job, before translation. Requiring Demucs would have kept
    it in the post-COMPLETE task with everything else."""
    from backend.services import vocal_gap_recovery as V
    import backend.services.vocal_separator as VS
    monkeypatch.setattr(VS, "is_available", lambda: False)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    segs = _track((0.0, 5.0, "one"), (60.0, 65.0, "two"))
    # separate=True bails on the missing Demucs; separate=False gets past it
    # and only stops later, at the (unstubbed) slicing step.
    assert asyncio.run(V.recover_gap_dialogue(
        "j", str(audio), segs, "ja", str(tmp_path / "w"), separate=True)) == []
    assert asyncio.run(V.recover_gap_dialogue(
        "j", str(audio), segs, "ja", str(tmp_path / "w2"), separate=False)) == []


def test_clip_to_gap_keeps_pad_edge_decode_when_no_existing_cue_covers_it():
    # The pad exists to avoid double-captioning audio an existing cue already
    # covers. A decode landing in the pad where the transcript has NOTHING is
    # exactly the dialogue the pass exists to recover — a measured run culled
    # 37/37 decoded segments (real lines like 了解) by pure geometry.
    from backend.services.vocal_gap_recovery import _clip_to_gap
    gap, pad = (100.0, 113.0), 2.0
    seg = {"start": 100.3, "end": 101.6, "text": "了解"}
    kept = _clip_to_gap(seg, gap, pad, existing=[])
    assert kept is not None
    assert kept["start"] >= gap[0] and kept["end"] <= gap[1]


def test_clip_to_gap_drops_pad_edge_decode_that_rehears_an_existing_cue():
    from backend.services.vocal_gap_recovery import _clip_to_gap
    gap, pad = (100.0, 113.0), 2.0
    seg = {"start": 100.3, "end": 101.6, "text": "了解"}
    existing = [{"start": 99.0, "end": 101.2, "text": "了解です"}]
    assert _clip_to_gap(seg, gap, pad, existing=existing) is None


def test_clip_to_gap_interior_decode_still_clips_to_interior():
    from backend.services.vocal_gap_recovery import _clip_to_gap
    gap, pad = (100.0, 113.0), 2.0
    seg = {"start": 101.0, "end": 106.0, "text": "本当の話"}
    kept = _clip_to_gap(seg, gap, pad, existing=[])
    assert kept is not None
    assert abs(kept["start"] - 102.0) < 1e-6  # clipped to interior lo


def test_clip_to_gap_still_rejects_fully_outside_decodes():
    from backend.services.vocal_gap_recovery import _clip_to_gap
    gap, pad = (100.0, 113.0), 2.0
    assert _clip_to_gap({"start": 90.0, "end": 95.0, "text": "x"},
                        gap, pad, existing=[]) is None
    assert _clip_to_gap({"start": 120.0, "end": 125.0, "text": "x"},
                        gap, pad, existing=[]) is None


def test_repair_stem_times_distributes_degenerate_decodes():
    # A measured run culled 86/86 recovered segments as "no-times": every
    # relisten segment came back with start == end. The repair distributes
    # the stem window by text weight so the recovered lines survive.
    from backend.services.vocal_gap_recovery import _repair_stem_times
    segs = [
        {"start": 0.0, "end": 0.0, "text": "連合本部に察知されていた"},
        {"start": 0.0, "end": 0.0, "text": "了解"},
    ]
    out, n = _repair_stem_times(segs, 10.0)
    assert n == 2
    assert out[0]["start"] == 0.0 and out[-1]["end"] == 10.0
    assert out[0]["end"] == out[1]["start"]          # contiguous, ordered
    assert out[0]["end"] - out[0]["start"] > out[1]["end"] - out[1]["start"]
    # +gap[0] then produces in-gap absolute times → kept, not culled.
    from backend.services.vocal_gap_recovery import _clip_to_gap
    for s in out:
        s2 = dict(s)
        s2["start"] += 100.0
        s2["end"] += 100.0
        assert _clip_to_gap(s2, (100.0, 110.0), 2.0, existing=[]) is not None


def test_repair_stem_times_leaves_valid_decodes_alone():
    from backend.services.vocal_gap_recovery import _repair_stem_times
    segs = [{"start": 0.5, "end": 2.0, "text": "a"},
            {"start": 2.2, "end": 4.0, "text": "b"}]
    out, n = _repair_stem_times(segs, 10.0)
    assert n == 0 and out is segs


def test_repair_stem_times_never_touches_a_partially_valid_decode():
    # Whisper commonly emits ONE zero-length trailing artifact beside
    # well-timed segments. Redistributing everything for it drifted real
    # lines by seconds (and the nearest-cue speaker guess with them) — the
    # repair fires only on the measured failure mode: NO usable time at all.
    from backend.services.vocal_gap_recovery import _repair_stem_times
    segs = [
        {"start": 0.8, "end": 2.1, "text": "A line with exact times"},
        {"start": 3.0, "end": 4.4, "text": "second"},
        {"start": 7.5, "end": 9.2, "text": "third exact line"},
        {"start": 9.2, "end": 9.2, "text": ""},        # the artifact
    ]
    out, n = _repair_stem_times(segs, 10.0)
    assert n == 0 and out is segs
    assert out[0]["start"] == 0.8 and out[2]["end"] == 9.2


def test_repair_stem_times_distributes_inside_the_pad_trimmed_interior():
    # The stem includes 2s pads that overlap existing cues by construction;
    # a short first line placed inside the lead pad was culled as a boundary
    # re-hearing — the exact line the repair exists to save.
    from backend.services.vocal_gap_recovery import _repair_stem_times, _clip_to_gap
    segs = [{"start": 0.0, "end": 0.0, "text": "了解"},
            {"start": 0.0, "end": 0.0, "text": "こちらは長い台詞でありますから続きます"}]
    out, n = _repair_stem_times(segs, 12.0, pad_s=2.0)
    assert n == 2
    assert out[0]["start"] >= 2.0 - 1e-6          # never inside the lead pad
    assert out[-1]["end"] <= 10.0 + 1e-6          # never inside the tail pad
    # End-to-end: with a boundary cue covering the pad, the short line survives.
    gap = (100.0, 112.0)
    existing = [{"start": 96.0, "end": 100.5, "text": "previous cue"}]
    for s in out:
        s2 = dict(s)
        s2["start"] += gap[0]
        s2["end"] += gap[0]
        assert _clip_to_gap(s2, gap, 2.0, existing=existing) is not None
