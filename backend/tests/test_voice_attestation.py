"""Voice attestation + LLM meta-response guard — the run-9 phantom fixes.

A measured run shipped two garbage blocks: an 11-cue block at 0:00-0:09 of
the timeline rendered over the silence before the opening theme, and a
12-cue block that was the LLM's refusal paragraph ("This sentence appears
to be in Japanese and seems to contain…") split across frame-width cues.
Two independent gates kill both classes at the last writer:

  * ``attest_cues_to_voice`` — a speech cue must overlap VAD-detected voice;
  * ``looks_like_meta_response`` — a "translation" that talks ABOUT the line
    is rejected instead of shipped.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services.speech_coverage import (  # noqa: E402
    attest_cues_to_voice, voice_activity_regions_cached)
from backend.services.translator import looks_like_meta_response  # noqa: E402


def _rows():
    return [
        {"start": 0.0, "end": 0.6, "text": "Cut! Zechs rules!", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 1.7, "text": "Mission accomplished!", "speaker": "Speaker 1"},
        {"start": 27.8, "end": 31.8, "text": "[♪ Opening theme ♪]", "speaker": "Speaker 1"},
        {"start": 127.6, "end": 129.9, "text": "an operation called 'Operation Meteor'",
         "speaker": "Speaker 2"},
    ]


def test_attestation_drops_head_cues_over_silence(monkeypatch, tmp_path):
    # Voice exists only from 2:07 on — the head block has no audio under it.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(127.0, 135.0)])
    kept, dropped = attest_cues_to_voice(_rows(), str(wav))
    texts = [r["text"] for r in kept]
    assert "Cut! Zechs rules!" not in texts
    assert "Mission accomplished!" not in texts
    assert len(dropped) == 2 and "Zechs" in dropped[0]
    # The dialogue that sits on voice survives.
    assert "an operation called 'Operation Meteor'" in texts


def test_attestation_exempts_markers_over_music(monkeypatch, tmp_path):
    # [♪ Opening theme ♪] annotates MUSIC on purpose — it must never need
    # voice under it.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(127.0, 135.0)])
    kept, _ = attest_cues_to_voice(_rows(), str(wav))
    assert "[♪ Opening theme ♪]" in [r["text"] for r in kept]


def test_attestation_keeps_readability_extended_cues(monkeypatch, tmp_path):
    # The extension pass legitimately stretches a cue far past its voiced
    # audio (CPS relief). A small ABSOLUTE overlap must suffice — no
    # fractional requirement.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(10.0, 10.5)])
    rows = [{"start": 10.0, "end": 14.0, "text": "Yes.", "speaker": "Speaker 1"}]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept == rows and not dropped


def test_attestation_drops_zero_width_and_timeless_cues(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(0.0, 600.0)])
    rows = [
        {"start": 5.0, "end": 5.0, "text": "Effect canceled", "speaker": "Speaker 1"},
        {"start": None, "end": None, "text": "Dialogue start", "speaker": "Speaker 1"},
        {"start": 5.0, "end": 6.0, "text": "Real line.", "speaker": "Speaker 1"},
    ]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert [r["text"] for r in kept] == ["Real line."]
    assert len(dropped) == 2


def test_attestation_fails_soft_without_vad(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions", lambda p, **k: [])
    rows = _rows()
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept is rows and dropped == []
    # Missing file: same fail-soft contract.
    kept2, dropped2 = attest_cues_to_voice(rows, str(tmp_path / "nope.wav"))
    assert kept2 is rows and dropped2 == []


def test_vad_cache_one_pass_per_file(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    calls = {"n": 0}

    def _fake(p, **k):
        calls["n"] += 1
        return [(1.0, 2.0)]

    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    SC._VAD_CACHE.clear()
    assert voice_activity_regions_cached(str(wav)) == [(1.0, 2.0)]
    assert voice_activity_regions_cached(str(wav)) == [(1.0, 2.0)]
    assert calls["n"] == 1, "second consult must hit the cache"
    # An empty result is NEVER cached — VAD can become available later.
    monkeypatch.setattr(SC, "voice_activity_regions", lambda p, **k: [])
    wav2 = tmp_path / "b.wav"
    wav2.write_bytes(b"RIFF" + b"\0" * 64)
    assert voice_activity_regions_cached(str(wav2)) == []
    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    assert voice_activity_regions_cached(str(wav2)) == [(1.0, 2.0)]


# ── The meta-response guard ─────────────────────────────────────────────────

_SHIPPED_REFUSAL = (
    "This sentence appears to be in Japanese and seems to contain multiple "
    "characters that do not form coherent words or sentences. Given the "
    "context provided, it's unclear what specific phrase is intended for "
    "translation as \"今回の殺さ\" does not appear to be a valid Japanese "
    "word. If we were to interpret this literally (which would likely result "
    "in nonsensical English), it could approximate: \"This time killing\".")


def test_meta_guard_catches_the_shipped_refusal_paragraph():
    assert looks_like_meta_response(_SHIPPED_REFUSAL, "今回の殺さ") is True
    # Pattern alone suffices — even without the source for the ratio check.
    assert looks_like_meta_response(_SHIPPED_REFUSAL) is True


def test_meta_guard_passes_real_dialogue():
    for line in (
        "Mission accomplished!",
        "That's definitely a Gundam.",
        "It seems our enemy possesses quite advanced technology.",
        "I'm sorry.",                       # a real apology LINE, not a refusal
        "Please, tell me what happened.",
    ):
        assert looks_like_meta_response(line, "了解、任務完了だ") is False, line


def test_meta_guard_rejects_paragraph_blowup_with_source():
    src = "今回の殺さ"
    blown = "word " * 60                    # 300 chars for a 5-char source
    assert looks_like_meta_response(blown, src) is True
    # The same text without a source is not flagged by ratio alone.
    assert looks_like_meta_response(blown) is False
    # A normal-length translation of a short line is fine.
    assert looks_like_meta_response("This time, the killing begins.", src) is False


def test_validate_translation_uses_the_meta_guard():
    from backend.services.pipeline import _validate_translation
    assert _validate_translation(_SHIPPED_REFUSAL, "ja", src_text="今回の殺さ") == ""
    assert _validate_translation(
        "Mission accomplished!", "ja", src_text="任務完了") == "Mission accomplished!"


# ── Evidence-aware recall (run-10 regression: gate ate real whispers) ──────

def test_attestation_keeps_whispers_with_measured_words(monkeypatch, tmp_path):
    # "I'll kill you" — whispered, invisible to Silero — but its word rows
    # were measured against real audio (decode/CTC). Evidence beats the
    # VAD's opinion of a whisper.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(10.0, 12.0)])
    rows = [
        {"start": 1324.0, "end": 1325.4, "text": "I'll kill you.",
         "speaker": "Speaker 3",
         "words": [{"word": "I'll", "start": 1324.0, "end": 1324.4},
                   {"word": "kill", "start": 1324.4, "end": 1324.9},
                   {"word": "you.", "start": 1324.9, "end": 1325.4}]},
    ]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept == rows and not dropped


def test_attestation_synthetic_words_are_not_evidence(monkeypatch, tmp_path):
    # Phantom cues carry SYNTHETIC word rows (char-weight projections built
    # for whatever window they ended up in) — those prove nothing.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(100.0, 105.0)])
    rows = [
        {"start": 0.5, "end": 2.1, "text": "Got it! Normal here.",
         "speaker": "Speaker 1", "words_synthetic": True,
         "words": [{"word": "Got", "start": 0.5, "end": 1.0},
                   {"word": "it!", "start": 1.0, "end": 2.1}]},
    ]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept == [] and len(dropped) == 1


def test_attestation_margin_tolerates_offset_cue_edges(monkeypatch, tmp_path):
    # Onset bias / extension legitimately shift a cue slightly off its
    # voiced audio — the ±0.5s margin keeps such cues.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(9.5, 10.5)])
    rows = [{"start": 10.8, "end": 11.8, "text": "Slightly late cue.",
             "speaker": "Speaker 1"}]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept == rows and not dropped


def test_attestation_uses_the_sensitive_vad_threshold(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    seen = {}

    def _fake(p, **k):
        seen.update(k)
        return [(1.0, 2.0)]

    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    SC._VAD_CACHE.clear()
    attest_cues_to_voice([{"start": 1.0, "end": 2.0, "text": "hi",
                           "speaker": "Speaker 1"}], str(wav))
    assert seen.get("threshold") == 0.25, "whisper-sensitive map required"


# ── Translation window attestation (restore, don't lose) ──────────────────

def test_restore_windows_repairs_degenerate_and_drifted_cues():
    from backend.services.subtitle_aligner import restore_translation_windows
    source = [
        {"start": 128.0, "end": 130.5, "text": "了解、任務完了だ"},
        {"start": 200.0, "end": 203.0, "text": "ゼクス、頼む"},
        {"start": 300.0, "end": 302.0, "text": "はい"},
    ]
    translated = [
        {"start": 0.0, "end": 0.0, "text": "Got it! Mission complete.",
         "words": [{"word": "Got", "start": 0.0, "end": 0.0}],
         "words_synthetic": True},                      # degenerate window
        {"start": 0.56, "end": 2.17, "text": "Zechs, please."},  # drifted
        {"start": 300.1, "end": 301.9, "text": "Yes."},          # healthy
    ]
    out = restore_translation_windows(translated, source)
    assert out["restored"] == 2 and out["source_degenerate"] == 0
    assert (translated[0]["start"], translated[0]["end"]) == (128.0, 130.5)
    assert translated[0]["words"] is None                # wrong-window words die
    assert (translated[1]["start"], translated[1]["end"]) == (200.0, 203.0)
    assert (translated[2]["start"], translated[2]["end"]) == (300.1, 301.9)
    assert len(out["samples"]) == 2


def test_restore_windows_leaves_legit_timing_refinement_alone():
    # CTC tightening moves edges by fractions of a second — never restored.
    from backend.services.subtitle_aligner import restore_translation_windows
    source = [{"start": 100.0, "end": 104.0, "text": "ソース"}]
    translated = [{"start": 100.6, "end": 103.2, "text": "Tightened."}]
    out = restore_translation_windows(translated, source)
    assert out["restored"] == 0
    assert (translated[0]["start"], translated[0]["end"]) == (100.6, 103.2)


def test_restore_windows_requires_one_to_one_lists():
    from backend.services.subtitle_aligner import restore_translation_windows
    source = [{"start": 1.0, "end": 2.0, "text": "a"}]
    translated = [{"start": 0.0, "end": 0.0, "text": "x"},
                  {"start": 0.0, "end": 0.0, "text": "y"}]
    out = restore_translation_windows(translated, source)
    assert out["restored"] == 0
    assert translated[0]["start"] == 0.0                 # untouched


def test_restore_windows_counts_source_side_corruption():
    # A degenerate SOURCE window means the corruption is upstream of
    # translation — nothing to restore from, but the count names it.
    from backend.services.subtitle_aligner import restore_translation_windows
    source = [{"start": 5.0, "end": 5.0, "text": "壊れた"}]
    translated = [{"start": 0.0, "end": 0.0, "text": "Broken."}]
    out = restore_translation_windows(translated, source)
    assert out["restored"] == 0 and out["source_degenerate"] == 1


def test_restore_windows_handles_model_objects():
    from backend.models import TranscriptSegment, WordTimestamp
    from backend.services.subtitle_aligner import restore_translation_windows
    src = TranscriptSegment(start=50.0, end=53.0, text="ソース行",
                            speaker="Speaker 1")
    bad = TranscriptSegment(
        start=0.0, end=0.0, text="A line.", speaker="Speaker 1",
        words=[WordTimestamp(word="A", start=0.0, end=0.0)],
        words_synthetic=True)
    out = restore_translation_windows([bad], [src])
    assert out["restored"] == 1
    assert (bad.start, bad.end) == (50.0, 53.0)
    assert bad.words is None and bad.words_synthetic is None


# ── Stale-words guard + degenerate-window sweep (run-11 injector) ─────────

def test_resegmenter_ignores_word_rows_outside_the_cue_window():
    # A recovery cue at 4:11 carried stem-relative words (0-3s) — the
    # word-timed split path re-timed it to the head of the video. Words
    # that live outside their own cue are evidence of a bug, not timing.
    from backend.models import TranscriptSegment, WordTimestamp
    from backend.services.sentence_segmenter import _split_segment_by_sentence
    seg = TranscriptSegment(
        start=251.0, end=258.0, speaker="Speaker 1",
        text="Got it! All units fine. Roger that!",
        words=[WordTimestamp(word="Got", start=0.5, end=0.8),
               WordTimestamp(word="it!", start=0.8, end=1.1),
               WordTimestamp(word="All", start=1.1, end=1.4),
               WordTimestamp(word="units", start=1.4, end=1.8),
               WordTimestamp(word="fine.", start=1.8, end=2.2),
               WordTimestamp(word="Roger", start=2.2, end=2.6),
               WordTimestamp(word="that!", start=2.6, end=3.0)])
    pieces = _split_segment_by_sentence(seg)
    for p in pieces:
        assert 251.0 - 1e-6 <= p.start <= p.end <= 258.0 + 1e-6, \
            f"piece escaped its cue window: {p.start}-{p.end}"


def test_resegmenter_still_uses_in_window_words():
    from backend.models import TranscriptSegment, WordTimestamp
    from backend.services.sentence_segmenter import _split_segment_by_sentence
    seg = TranscriptSegment(
        start=10.0, end=16.0, speaker="Speaker 1",
        text="First line. Second line.",
        words=[WordTimestamp(word="First", start=10.0, end=10.5),
               WordTimestamp(word="line.", start=10.5, end=11.0),
               WordTimestamp(word="Second", start=14.0, end=14.5),
               WordTimestamp(word="line.", start=14.5, end=15.0)])
    pieces = _split_segment_by_sentence(seg)
    assert len(pieces) == 2
    assert abs(pieces[1].start - 14.0) < 0.5     # real word-timed boundary


def test_degenerate_window_sweep_drops_echoes_and_retimes_unique_text():
    from backend.services.transcript_sanitize import (
        repair_degenerate_cue_windows)
    rows = [
        {"start": 10.0, "end": 12.0, "text": "Roger that, all units fine."},
        {"start": 0.0, "end": 0.0, "text": "Roger that all units fine"},  # echo
        {"start": 0.0, "end": 0.0, "text": "A unique lost line."},        # keep
        {"start": 30.0, "end": 32.0, "text": "Next scene starts."},
    ]
    out, dropped, repaired = repair_degenerate_cue_windows(rows)
    texts = [r["text"] for r in out]
    assert "Roger that all units fine" not in texts
    assert len(dropped) == 1 and "echo:" in dropped[0]
    assert len(repaired) == 1
    fixed = next(r for r in out if r["text"] == "A unique lost line.")
    # Re-timed into the silence between its neighbours, inside (12, 30).
    assert 12.0 < fixed["start"] < fixed["end"] < 30.0


def test_degenerate_window_sweep_leaves_valid_rows_untouched():
    from backend.services.transcript_sanitize import (
        repair_degenerate_cue_windows)
    rows = [{"start": 1.0, "end": 2.0, "text": "fine"},
            {"start": 3.0, "end": 4.0, "text": "also fine"}]
    out, dropped, repaired = repair_degenerate_cue_windows(rows)
    assert out is rows and not dropped and not repaired


def test_degenerate_window_sweep_drops_when_no_room():
    from backend.services.transcript_sanitize import (
        repair_degenerate_cue_windows)
    rows = [
        {"start": 10.0, "end": 12.0, "text": "Before."},
        {"start": 0.0, "end": 0.0, "text": "Unique but squeezed."},
        {"start": 12.1, "end": 14.0, "text": "After."},
    ]
    out, dropped, repaired = repair_degenerate_cue_windows(rows)
    assert len(out) == 2 and not repaired
    assert len(dropped) == 1 and "no-room:" in dropped[0]
