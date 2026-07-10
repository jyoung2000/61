"""Tests for the degenerate remote-decode gate.

A live 128-min run produced ZERO transcript segments: whisper.cpp on the
Companion looped (257 segments, 254 verbatim repeats of the same line), the
server-side filters correctly discarded the garbage — and the pipeline then
ACCEPTED the empty result as success, skipped translation/summary, and logged
"the remainder is music/silence" at 0% coverage. These pin the fixes:

  * ``_finalize_cloud_transcription`` returns None (→ local fallback) when
    the FINAL filtered list is empty, not a 0-segment "success" — UNLESS
    Silero confirms the audio holds almost no speech, in which case a valid
    EMPTY result (marked ``no_speech_evidence``) is correct and the pipeline
    skips the redundant local decode.
  * ``_remote_transcript_acceptable`` rejects a transcript covering almost
    none of the VAD-detected speech on a talky video — but BOTH the ratio
    floor and the absolute covered-seconds floor must fail, so a concert VOD
    (huge sung "voice", small real-dialogue transcript) is never rejected.
  * a rejected 200-OK decode sets ``_remote_decode_rejected`` and try_load()
    then skips the remote server — no deterministic second garbage decode.
  * the coverage log stops claiming "music/silence" at low ratios.
"""

import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings
from backend.services import reframer_audio as RA


class _Log:
    def __init__(self):
        self.lines = []

    def log_stage(self, stage, msg, **kw):
        self.lines.append(str(msg))

    def log_error(self, stage, msg, **kw):
        self.lines.append(str(msg))

    def text(self):
        return "\n".join(self.lines)


def _ai():
    return RA.AudioIntelligence.__new__(RA.AudioIntelligence)


# ─────────────────────────────────────────────────────────────────────────────
# _finalize_cloud_transcription: empty-after-filter → None
# ─────────────────────────────────────────────────────────────────────────────

def _cloud(segments):
    return {"segments": segments, "language": "japanese",
            "provider": "remote", "model": "large-v3-turbo"}


def _seg(i, text, ns=0.05):
    return {"start_sec": float(i * 10), "end_sec": float(i * 10 + 4),
            "text": text, "words": [], "no_speech_prob": ns,
            "avg_logprob": -0.3, "is_hallucination": False}


def test_finalize_returns_none_when_every_segment_is_filtered(tmp_path):
    # Every cue carries no_speech_prob above the 0.7 drop → all flagged as
    # hallucinated → the final cross-validated list is empty → None, which is
    # what triggers the caller's local fallback.
    segs = [_seg(i, f"text {i}", ns=0.95) for i in range(6)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=120_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is None


def test_finalize_returns_none_on_looped_decode(tmp_path):
    # The live failure shape: the SAME line repeated hundreds of times. The
    # repetition-loop filter keeps a couple of copies; the survivors carry a
    # high no_speech_prob (loops live over music) so nothing real remains.
    segs = [_seg(i, "作戦名オペレーション・メテオ", ns=0.75) for i in range(60)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=1_200_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is None


def test_finalize_keeps_a_real_transcript(tmp_path):
    segs = [_seg(i, f"ちゃんとした台詞です {i}", ns=0.05) for i in range(6)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=120_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is not None
    assert len(out["segments"]) == 6


def test_finalize_empty_with_no_speech_is_a_valid_empty_result(tmp_path, monkeypatch):
    # Everything filtered AND Silero heard almost no speech (40 s < the 120 s
    # floor): the empty transcript is CORRECT — return a valid empty result
    # (marked no_speech_evidence) instead of None, so legitimately
    # speech-free content never pays a redundant local decode.
    from backend.services import speech_coverage as SC
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda path: [(10.0, 50.0)])
    segs = [_seg(i, f"text {i}", ns=0.95) for i in range(6)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=120_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is not None
    assert out["segments"] == []
    assert out.get("no_speech_evidence") is True


def test_finalize_empty_with_substantial_speech_is_a_failed_decode(tmp_path, monkeypatch):
    # Everything filtered but Silero heard 900 s of speech — that is a FAILED
    # decode, not an empty video: None → the caller runs the local fallback.
    from backend.services import speech_coverage as SC
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda path: [(0.0, 900.0)])
    segs = [_seg(i, f"text {i}", ns=0.95) for i in range(6)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=1_200_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is None


def test_finalize_empty_with_unknown_vad_is_a_failed_decode(tmp_path, monkeypatch):
    # VAD unavailable ([] is ambiguous: no speech OR no VAD) → be
    # conservative: treat the empty transcript as a failed decode.
    from backend.services import speech_coverage as SC
    monkeypatch.setattr(SC, "voice_activity_regions", lambda path: [])
    segs = [_seg(i, f"text {i}", ns=0.95) for i in range(6)]
    out = _ai()._finalize_cloud_transcription(
        _cloud(segs), str(tmp_path / "audio.wav"), duration_ms=120_000,
        log=_Log(), on_progress=None, pinned_language="ja")
    assert out is None


# ─────────────────────────────────────────────────────────────────────────────
# _remote_transcript_acceptable
# ─────────────────────────────────────────────────────────────────────────────

def _result(n_segments=10, voice_sec=1725.0, ratio=0.9,
            with_coverage=True):
    res = {"segments": [_seg(i, f"line {i}") for i in range(n_segments)]}
    if with_coverage:
        res["speech_coverage"] = {
            "voice_sec": voice_sec, "coverage_ratio": ratio,
            "covered_voice_sec": voice_sec * ratio,
            "uncovered_voice_sec": voice_sec * (1 - ratio),
            "gap_count": 1.0,
        }
    return res


def test_gate_rejects_zero_coverage_on_talky_video():
    # The live failure: 1725 s of VAD speech, 0% covered.
    assert _ai()._remote_transcript_acceptable(
        _result(n_segments=3, voice_sec=1725.0, ratio=0.0), _Log()) is False


def test_gate_rejects_empty_segments():
    assert _ai()._remote_transcript_acceptable(
        {"segments": []}, _Log()) is False
    assert _ai()._remote_transcript_acceptable({}, _Log()) is False


def test_gate_accepts_vad_confirmed_empty_transcript():
    # Finalize already proved (via Silero) the audio holds no substantial
    # speech — the empty transcript is the CORRECT answer, not a failure.
    assert _ai()._remote_transcript_acceptable(
        {"segments": [], "no_speech_evidence": True}, _Log()) is True


def test_gate_keeps_concert_vod_with_real_mc_talk():
    # Silero counts SINGING as voice: 790 s of "voice" on a concert VOD, and
    # the filters (by design) dropped the lyric cues — but the 90 s of real
    # MC dialogue survived. The RATIO is tiny (11%), yet the transcript is
    # perfectly good: the absolute covered-seconds floor must keep it.
    res = _result(n_segments=30, voice_sec=790.0, ratio=0.114)
    assert res["speech_coverage"]["covered_voice_sec"] >= 30.0  # sanity
    assert _ai()._remote_transcript_acceptable(res, _Log()) is True


def test_gate_accepts_healthy_coverage():
    assert _ai()._remote_transcript_acceptable(
        _result(ratio=0.93), _Log()) is True


def test_gate_accepts_sparse_speech_video():
    # Under the voice floor (a 90 s of speech in a music video) the gate
    # never fires — sparse-but-real transcripts must pass.
    assert _ai()._remote_transcript_acceptable(
        _result(voice_sec=90.0, ratio=0.05), _Log()) is True


def test_gate_accepts_when_no_coverage_evidence():
    # Audit disabled / VAD unavailable → no evidence → no rejection.
    assert _ai()._remote_transcript_acceptable(
        _result(with_coverage=False), _Log()) is True


def test_gate_floors_are_configurable(monkeypatch):
    # Rejection needs BOTH floors to fail. Raise both so a mid-coverage
    # transcript (ratio 0.3 → 517 s covered of 1725 s) trips them, then relax
    # each floor in turn and watch the gate flip back to accept.
    monkeypatch.setattr(settings, "REMOTE_TRANSCRIPT_MIN_COVERAGE", 0.5)
    monkeypatch.setattr(settings, "REMOTE_TRANSCRIPT_MIN_COVERED_S", 600.0)
    assert _ai()._remote_transcript_acceptable(
        _result(ratio=0.3), _Log()) is False       # 30% < 50% AND 517s < 600s
    monkeypatch.setattr(settings, "REMOTE_TRANSCRIPT_MIN_COVERED_S", 30.0)
    assert _ai()._remote_transcript_acceptable(
        _result(ratio=0.3), _Log()) is True        # 517s ≥ 30s → kept
    monkeypatch.setattr(settings, "REMOTE_TRANSCRIPT_MIN_COVERED_S", 600.0)
    monkeypatch.setattr(settings, "REMOTE_TRANSCRIPT_MIN_COVERAGE", 0.15)
    assert _ai()._remote_transcript_acceptable(
        _result(ratio=0.3), _Log()) is True        # 30% ≥ 15% → kept


# ─────────────────────────────────────────────────────────────────────────────
# Honest coverage messaging
# ─────────────────────────────────────────────────────────────────────────────

def _run_audit(monkeypatch, segments, voice_regions):
    from backend.services import speech_coverage as SC
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda path: voice_regions)
    monkeypatch.setattr(settings, "SPEECH_GAP_RECOVERY_ENABLED", False)
    log = _Log()
    result = {"segments": segments}
    _ai()._audit_and_recover_speech(result, "audio.wav", 600_000, "ja",
                                    None, log)
    return log.text()


def test_low_coverage_no_longer_claims_music_silence(monkeypatch):
    # 600 s of voice, transcript covers none of it.
    txt = _run_audit(monkeypatch, [], [(0.0, 600.0)])
    assert "LOW COVERAGE" in txt
    assert "music/silence" not in txt


def test_high_coverage_keeps_the_silence_note(monkeypatch):
    segs = [{"start_sec": 0.0, "end_sec": 590.0, "text": "x"}]
    txt = _run_audit(monkeypatch, segs, [(0.0, 600.0)])
    assert "music/silence" in txt
    assert "LOW COVERAGE" not in txt


# ─────────────────────────────────────────────────────────────────────────────
# No deterministic second remote decode after a rejection
# ─────────────────────────────────────────────────────────────────────────────

def _loadable_ai(monkeypatch):
    """A real AudioIntelligence with a healthy 'remote' in front of it, and a
    stub logger so try_load() doesn't create session log files."""
    log = _Log()
    monkeypatch.setattr(RA, "get_logger", lambda *a, **k: log)
    monkeypatch.setattr(RA, "remote_whisper_configured", lambda: True)
    monkeypatch.setattr(RA, "remote_whisper_healthy",
                        lambda force=False: True)
    return RA.AudioIntelligence(model_name="small"), log


def test_try_load_selects_remote_normally(monkeypatch):
    ai, _ = _loadable_ai(monkeypatch)
    assert ai.try_load() is True
    assert ai.device_used == "remote"


def test_try_load_skips_remote_after_rejected_decode(monkeypatch):
    # The perceiver's sequential retry reuses the SAME instance — once a
    # 200-OK decode was rejected as degenerate, re-sending the same audio is
    # deterministic garbage, so the retry must go straight to the local
    # ladder (whatever of it is installed) instead of remote.
    ai, log = _loadable_ai(monkeypatch)
    ai._remote_decode_rejected = True
    ai.try_load()                      # local ladder may or may not load here
    assert ai.device_used != "remote"
    assert "skipped" in log.text().lower()


def test_rejected_flag_starts_clear():
    assert RA.AudioIntelligence(model_name="small")._remote_decode_rejected \
        is False
