"""Tests for the transcription coverage-loss fixes (Netflix-grade coverage).

Four root causes of dropped quiet speech / skipped stretches are pinned here:

  1. SIDECAR PARITY — the Companion whisper sidecar used to run faster-whisper
     DEFAULTS (Silero 0.5 / min_silence 2000 ms / no_speech 0.6 /
     condition_on_previous_text=True) while the local path ran tuned values.
     The sidecar now accepts optional decode-tuning form fields
     (feature-detected like ``_vocab_bias_kwargs``) and ``RemoteWhisperEngine``
     sends the SAME tuned values the local path computes.
  2. BOOSTED GAP RECOVERY — a voice-active gap slice is re-extracted through a
     speech-boost ffmpeg chain (highpass → afftdn → speechnorm) before the
     retry decode, so quiet/off-mic speech is lifted.
  3. FILTER/COVERAGE TENSION — a cue the TACT phantom gate would drop is
     routed to the low-confidence redecode queue (with a raised budget) when
     the independent Silero voice map confirms voice under its span.
  4. POST-TRANSLATION CPS RE-CHECK — translated text that now exceeds the
     CPS / chars-per-line caps is re-split before persisting.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

# reframer_audio imports cv2 at module scope; the tests never touch it.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings
from backend.services import reframer_audio as RA
from backend.services import speech_coverage as SC


# ─────────────────────────────────────────────────────────────────────────────
# 1a. Sidecar: feature-detected tuning kwargs
# ─────────────────────────────────────────────────────────────────────────────

_SIDECAR_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "companion", "sidecars", "whisper-server", "server.py")


@pytest.fixture(scope="module")
def sidecar():
    spec = importlib.util.spec_from_file_location(
        "whisper_sidecar_server_under_test", _SIDECAR_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _modern_transcribe(audio, beam_size=5, vad_filter=True, vad_parameters=None,
                       word_timestamps=False, no_speech_threshold=0.6,
                       condition_on_previous_text=True, no_repeat_ngram_size=0,
                       log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
                       hallucination_silence_threshold=None, language=None,
                       initial_prompt=None, temperature=0.0):
    """Signature mirror of a current faster-whisper ``transcribe``."""


def _old_transcribe(audio, beam_size=5, vad_filter=True, language=None,
                    word_timestamps=False, **kwargs):
    """An old build: no tuning params, only a ``**kwargs`` catch-all (which
    must NOT count as support — passing unknown kwargs would raise later)."""


def test_sidecar_tuned_kwargs_modern_build(sidecar):
    kw = sidecar._tuned_transcribe_kwargs(
        _modern_transcribe,
        beam_size=5,
        vad_threshold=0.10,
        vad_min_silence_ms=300,
        vad_speech_pad_ms=150,
        no_speech_threshold=0.4,
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=2.0,
    )
    assert kw["beam_size"] == 5
    assert kw["no_speech_threshold"] == 0.4
    assert kw["condition_on_previous_text"] is False
    assert kw["no_repeat_ngram_size"] == 3
    assert kw["log_prob_threshold"] == -1.0
    assert kw["compression_ratio_threshold"] == 2.4
    assert kw["hallucination_silence_threshold"] == 2.0
    assert kw["vad_parameters"] == {
        "threshold": 0.10,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 150,
    }


def test_sidecar_tuned_kwargs_old_build_drops_unknown(sidecar):
    """On an old faster-whisper only the explicitly named params survive —
    the ``**kwargs`` catch-all does not count as support."""
    kw = sidecar._tuned_transcribe_kwargs(
        _old_transcribe,
        beam_size=8,
        vad_threshold=0.10,
        no_speech_threshold=0.4,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    assert kw == {"beam_size": 8}


def test_sidecar_tuned_kwargs_none_means_engine_default(sidecar):
    """A client that sends nothing gets the loaded build's own defaults —
    the backward-compatible contract for speaches / whisper.cpp clients."""
    assert sidecar._tuned_transcribe_kwargs(_modern_transcribe) == {}


def test_sidecar_tuned_kwargs_uninspectable_callable(sidecar):
    # inspect.signature raises on some builtins — must return {} not raise.
    assert sidecar._tuned_transcribe_kwargs(dict, beam_size=5) == {}


def test_sidecar_tuned_kwargs_rejects_out_of_range_vad_threshold(sidecar):
    kw = sidecar._tuned_transcribe_kwargs(_modern_transcribe, vad_threshold=1.5)
    assert "vad_parameters" not in kw
    kw = sidecar._tuned_transcribe_kwargs(_modern_transcribe, vad_threshold=0.0)
    assert "vad_parameters" not in kw


def test_sidecar_parse_optional_bool(sidecar):
    f = sidecar._parse_optional_bool
    assert f(None) is None
    assert f("") is None
    assert f("  ") is None
    for truthy in ("true", "True", "1", "yes", "on"):
        assert f(truthy) is True
    for falsy in ("false", "0", "no", "off", "banana"):
        assert f(falsy) is False


# ─────────────────────────────────────────────────────────────────────────────
# 1b. RemoteWhisperEngine sends the local path's tuned values
# ─────────────────────────────────────────────────────────────────────────────

def test_remote_tuning_fields_match_local_path(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_SEND_TUNING", True)
    fields = RA._remote_tuning_fields()

    vad = RA._vad_parameters()
    assert fields["beam_size"] == str(RA._beam_size())
    assert fields["vad_min_silence_ms"] == str(int(vad["min_silence_duration_ms"]))
    assert fields["vad_speech_pad_ms"] == str(int(vad["speech_pad_ms"]))
    assert float(fields["no_speech_threshold"]) == float(
        settings.WHISPER_NO_SPEECH_THRESHOLD)
    assert fields["condition_on_previous_text"] == (
        "true" if settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT else "false")
    assert int(fields["no_repeat_ngram_size"]) == int(
        settings.WHISPER_NO_REPEAT_NGRAM_SIZE)
    assert float(fields["log_prob_threshold"]) == float(
        settings.WHISPER_LOG_PROB_THRESHOLD)
    assert float(fields["compression_ratio_threshold"]) == float(
        settings.WHISPER_COMPRESSION_RATIO_THRESHOLD)
    assert float(fields["hallucination_silence_threshold"]) == float(
        settings.WHISPER_HALLUCINATION_SILENCE_S)
    # WHISPER_VAD_ONSET (0.10 default) must ride along as the Silero threshold
    # — the sidecar's faster-whisper default of 0.5 is what dropped soft speech.
    assert "threshold" in vad
    assert float(fields["vad_threshold"]) == float(vad["threshold"])


def test_remote_tuning_fields_disabled(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_SEND_TUNING", False)
    assert RA._remote_tuning_fields() == {}


def test_transcribe_wav_posts_tuning_fields(monkeypatch, tmp_path):
    """The remote POST carries the tuned decode fields end-to-end."""
    monkeypatch.setattr(settings, "WHISPER_REMOTE_SEND_TUNING", True)
    wav = tmp_path / "probe.wav"
    wav.write_bytes(b"RIFF0000WAVEfmt ")

    captured = {}

    def _fake_payload(self, wav_bytes, filename, data, headers, language,
                      model, offset_s=0.0):
        captured.update(data)
        return {"segments": [{"start_sec": 0.0, "end_sec": 1.0,
                              "text": "hi", "words": []}],
                "language": "en"}

    monkeypatch.setattr(RA.RemoteWhisperEngine, "_transcribe_payload",
                        _fake_payload)
    engine = RA.RemoteWhisperEngine(model="large-v3")
    result = engine.transcribe_wav(str(wav), language="en")
    assert result is not None and result["segments"]
    for key in ("beam_size", "vad_threshold", "vad_min_silence_ms",
                "vad_speech_pad_ms", "no_speech_threshold",
                "condition_on_previous_text", "no_repeat_ngram_size",
                "log_prob_threshold", "compression_ratio_threshold",
                "hallucination_silence_threshold"):
        assert key in captured, f"tuning field {key} missing from the POST"


def test_transcribe_wav_tuning_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_SEND_TUNING", False)
    wav = tmp_path / "probe.wav"
    wav.write_bytes(b"RIFF0000WAVEfmt ")
    captured = {}

    def _fake_payload(self, wav_bytes, filename, data, headers, language,
                      model, offset_s=0.0):
        captured.update(data)
        return {"segments": [{"start_sec": 0.0, "end_sec": 1.0,
                              "text": "hi", "words": []}],
                "language": "en"}

    monkeypatch.setattr(RA.RemoteWhisperEngine, "_transcribe_payload",
                        _fake_payload)
    RA.RemoteWhisperEngine(model="large-v3").transcribe_wav(str(wav), "en")
    assert "vad_threshold" not in captured
    assert "no_speech_threshold" not in captured


# ─────────────────────────────────────────────────────────────────────────────
# 2. Boosted gap recovery: the speech-boost filter chain builder
# ─────────────────────────────────────────────────────────────────────────────

def test_gap_boost_default_chain(monkeypatch):
    monkeypatch.setattr(settings, "SPEECH_GAP_BOOST_ENABLED", True)
    monkeypatch.setattr(settings, "SPEECH_GAP_BOOST_FILTER", "")
    args = RA._gap_boost_af_args()
    assert args[0] == "-af"
    assert args[1] == "highpass=f=80,afftdn=nf=-25,speechnorm=e=6.25:r=0.0001:l=1"


def test_gap_boost_disabled(monkeypatch):
    monkeypatch.setattr(settings, "SPEECH_GAP_BOOST_ENABLED", False)
    assert RA._gap_boost_af_args() == []


def test_gap_boost_custom_chain(monkeypatch):
    monkeypatch.setattr(settings, "SPEECH_GAP_BOOST_ENABLED", True)
    monkeypatch.setattr(settings, "SPEECH_GAP_BOOST_FILTER",
                        "speechnorm=e=3.0")
    assert RA._gap_boost_af_args() == ["-af", "speechnorm=e=3.0"]


# ─────────────────────────────────────────────────────────────────────────────
# 3a. Voice-map overlap check (pure interval math)
# ─────────────────────────────────────────────────────────────────────────────

def test_overlaps_voice_no_voice_map():
    assert SC.overlaps_voice([], 1.0, 2.0) is False


def test_overlaps_voice_full_overlap():
    assert SC.overlaps_voice([(0.5, 3.0)], 1.0, 2.0) is True


def test_overlaps_voice_trivial_brush_rejected():
    # 50 ms brush against a voice region on a 1 s cue — below the 0.2 s floor.
    assert SC.overlaps_voice([(0.0, 1.05)], 1.0, 2.0) is False


def test_overlaps_voice_long_cue_needs_proportional_overlap():
    # A 10 s cue with only 0.5 s of voice under it is NOT voice-confirmed
    # (needs ≥ 30% = 3 s), so a long invented run-on over music stays dropped.
    assert SC.overlaps_voice([(0.0, 0.5)], 0.0, 10.0) is False
    assert SC.overlaps_voice([(0.0, 4.0)], 0.0, 10.0) is True


def test_overlaps_voice_sums_disjoint_regions():
    # Three 0.4 s voice bursts under a 2 s cue → 1.2 s total ≥ max(0.2, 0.6).
    voice = [(0.0, 0.4), (0.7, 1.1), (1.4, 1.8)]
    assert SC.overlaps_voice(voice, 0.0, 2.0) is True


def test_overlaps_voice_invalid_span():
    assert SC.overlaps_voice([(0.0, 10.0)], 5.0, 5.0) is False
    assert SC.overlaps_voice([(0.0, 10.0)], None, 5.0) is False


# ─────────────────────────────────────────────────────────────────────────────
# 3b. VAD-confirmed phantom cues get a raised redecode budget
# ─────────────────────────────────────────────────────────────────────────────

class _FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class _FakeSeg:
    def __init__(self, start, end, text, avg_logprob=-0.1):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        step = max(0.01, (end - start) / 2)
        self.words = [
            _FakeWord("clear", start, start + step),
            _FakeWord("speech", start + step, end),
        ]


class _FakeEngine:
    """Sequential engine whose redecode always comes back confident."""

    def __init__(self):
        self.clips = []

    def transcribe(self, audio_path, language=None, vad_filter=True,
                   word_timestamps=True, clip_timestamps=None, beam_size=5,
                   patience=1.0, condition_on_previous_text=True,
                   no_repeat_ngram_size=0, compression_ratio_threshold=2.4,
                   log_prob_threshold=-1.0, repetition_penalty=1.0,
                   temperature=0.0, hallucination_silence_threshold=None):
        self.clips.append(list(clip_timestamps or []))
        a, b = (clip_timestamps or [0.0, 1.0])[:2]
        return iter([_FakeSeg(a, b, "clear speech")]), None


def _seg(i, *, phantom=False, vad_confirmed=False, logprob=0.0,
         hallucination=False):
    start = float(i)
    entry = {
        "start_sec": start,
        "end_sec": start + 0.9,
        "text": f"segment {i} words here",
        "words": [
            {"word": "segment", "start": start, "end": start + 0.4,
             "confidence": 0.9},
            {"word": "words", "start": start + 0.45, "end": start + 0.9,
             "confidence": 0.9},
        ],
        "is_hallucination": hallucination,
        "no_speech_prob": 0.1,
        "avg_logprob": logprob,
    }
    if phantom:
        entry["phantom"] = True
    if vad_confirmed:
        entry["vad_confirmed_voice"] = True
    return entry


def _make_ai():
    ai = RA.AudioIntelligence.__new__(RA.AudioIntelligence)
    ai.engine = _FakeEngine()
    return ai


def test_vad_confirmed_phantoms_get_raised_redecode_budget(monkeypatch):
    """With a base budget of 1, the 5 VAD-confirmed phantoms are STILL all
    redecoded — the raised budget applies to them specifically — instead of
    being crowded out and dropped."""
    monkeypatch.setattr(settings, "WHISPER_REDECODE_MAX_FRAC", 0.05)
    monkeypatch.setattr(settings, "WHISPER_REDECODE_VAD_MAX_FRAC", 0.25)

    segments = [_seg(i) for i in range(14)]
    # One ordinary low-logprob candidate (fills the base budget of 1)...
    segments.append(_seg(14, logprob=-2.0, hallucination=True))
    # ...and five VAD-confirmed phantom cues that must ALL be redecoded.
    phantoms = [
        _seg(15 + j, phantom=True, vad_confirmed=True, logprob=-1.5,
             hallucination=True)
        for j in range(5)
    ]
    segments.extend(phantoms)

    ai = _make_ai()
    replaced = ai._redecode_difficult_segments("dummy.wav", segments, "en", None)

    assert replaced == 6  # 1 regular (base budget) + 5 VAD-confirmed
    for entry in segments[15:]:
        assert entry["is_hallucination"] is False
        assert entry.get("redecoded") is True
        assert "phantom" not in entry  # rescue clears the phantom verdict


def test_unconfirmed_phantoms_stay_within_base_budget(monkeypatch):
    """Without VAD confirmation the old bound holds: only the base budget's
    worth of candidates is redecoded — silence-phantoms don't burn compute."""
    monkeypatch.setattr(settings, "WHISPER_REDECODE_MAX_FRAC", 0.05)
    monkeypatch.setattr(settings, "WHISPER_REDECODE_VAD_MAX_FRAC", 0.25)

    segments = [_seg(i) for i in range(14)]
    segments.extend(
        _seg(14 + j, phantom=True, logprob=-1.5, hallucination=True)
        for j in range(6)
    )

    ai = _make_ai()
    replaced = ai._redecode_difficult_segments("dummy.wav", segments, "en", None)
    assert replaced == 1  # max(1, int(20 * 0.05)) — unchanged legacy bound


# ─────────────────────────────────────────────────────────────────────────────
# 4. Post-translation CPS / line-length re-check
# ─────────────────────────────────────────────────────────────────────────────

def _over_cps_translated_cue():
    from backend.models import TranscriptSegment
    # ~160 chars in 2 s — far beyond any CPS cap after translation inflated it.
    text = ("This translated sentence has become dramatically longer than the "
            "original source line and now reads far too fast for anyone to "
            "follow on screen comfortably.")
    return TranscriptSegment(start=0.0, end=2.0, text=text, speaker="Speaker 1")


def test_post_translation_recheck_splits_over_cps_cue(monkeypatch):
    from backend.services.pipeline import recheck_translated_readability
    monkeypatch.setattr(settings, "SUBTITLE_POST_TRANSLATION_CPS_RECHECK", True)
    monkeypatch.setattr(settings, "SUBTITLE_MAX_CPS", 20.0)
    monkeypatch.setattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)

    out = recheck_translated_readability([_over_cps_translated_cue()])
    assert len(out) > 1, "an over-CPS translated cue must be re-split"
    for seg in out:
        lines = (seg.text or "").split("\n")
        # The enforcer's contract: ≤ 2 lines per cue, sized to the two-line
        # character budget (its greedy wrapper may leave one line slightly
        # unbalanced, so assert the budget, not per-line width).
        assert len(lines) <= 2
        assert sum(len(l) for l in lines) <= 2 * 42
        # Reading speed must come down from the input's ~80 CPS to (near)
        # the 20 CPS cap — the enforcer allows a small tolerance.
        chars = len((seg.text or "").replace("\n", " "))
        cps = chars / max(0.001, seg.end - seg.start)
        assert cps <= 20.0 * 1.5, f"cue still reads at {cps:.1f} CPS"


def test_post_translation_recheck_disabled(monkeypatch):
    from backend.services.pipeline import recheck_translated_readability
    monkeypatch.setattr(settings, "SUBTITLE_POST_TRANSLATION_CPS_RECHECK", False)
    cue = _over_cps_translated_cue()
    out = recheck_translated_readability([cue])
    assert out == [cue]


def test_post_translation_recheck_failsoft_on_bad_schema(monkeypatch):
    from backend.services.pipeline import recheck_translated_readability
    monkeypatch.setattr(settings, "SUBTITLE_POST_TRANSLATION_CPS_RECHECK", True)
    bad = [{"not_a_segment": True}]
    assert recheck_translated_readability(bad) == bad
