"""Phase 3 audit tests — local transcription quality.

Covers: hallucination_silence_threshold feature detection, VAD tuning
settings, degenerate-word detection, the difficult-segment redecode
pass (fake engine), forced-alignment fail-safety, distil model VRAM
registry entries, and the VRAM ledger.
"""

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import pytest

from backend.config import settings  # noqa: E402
from backend.services.reframer_audio import (  # noqa: E402
    _decoding_kwargs, _vad_parameters, _words_degenerate, AudioIntelligence,
)


# ── Decode kwargs feature detection ─────────────────────────────────────

def test_hallucination_silence_threshold_passed_when_supported():
    def transcribe(audio, hallucination_silence_threshold=None,
                   condition_on_previous_text=True, temperature=0.0):
        pass
    kwargs = _decoding_kwargs(transcribe)
    assert kwargs["hallucination_silence_threshold"] == pytest.approx(
        settings.WHISPER_HALLUCINATION_SILENCE_S)


def test_hallucination_silence_threshold_dropped_when_unsupported():
    def transcribe(audio, condition_on_previous_text=True):
        pass
    kwargs = _decoding_kwargs(transcribe)
    assert "hallucination_silence_threshold" not in kwargs


def test_vad_parameters_defaults():
    vp = _vad_parameters()
    assert vp["min_silence_duration_ms"] == 300
    assert vp["speech_pad_ms"] == 150


# ── Degenerate word-timestamp detection (batched word-timing bug) ───────

def test_words_degenerate_monotonic_ok():
    words = [{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}]
    assert _words_degenerate(words, 0.0, 1.0) is False


def test_words_degenerate_backwards():
    words = [{"start": 1.0, "end": 2.0}, {"start": 0.2, "end": 0.4}]
    assert _words_degenerate(words, 0.0, 2.0) is True


def test_words_degenerate_inverted_span():
    words = [{"start": 1.0, "end": 0.5}]
    assert _words_degenerate(words, 0.0, 2.0) is True


def test_words_degenerate_zero_width_majority():
    words = [{"start": 1.0, "end": 1.0}, {"start": 1.0, "end": 1.0},
             {"start": 1.0, "end": 2.0}]
    assert _words_degenerate(words, 0.0, 2.0) is True


def test_words_degenerate_empty_is_not_degenerate():
    assert _words_degenerate([], 0.0, 1.0) is False


# ── Difficult-segment redecode (fake engine) ────────────────────────────

class _FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word, self.start, self.end = word, start, end
        self.probability = probability


class _FakeSeg:
    def __init__(self, text, start, end, avg_logprob=-0.2, words=None):
        self.text, self.start, self.end = text, start, end
        self.avg_logprob = avg_logprob
        self.words = words or []


class _FakeEngine:
    def __init__(self, result_text="corrected words here"):
        self.calls = []
        self.result_text = result_text

    def transcribe(self, audio_path, language=None, vad_filter=True,
                   word_timestamps=False, clip_timestamps=None,
                   beam_size=5, patience=1.0,
                   condition_on_previous_text=True, temperature=0.0):
        self.calls.append({"clip_timestamps": clip_timestamps,
                           "beam_size": beam_size, "patience": patience})
        start, end = clip_timestamps
        words = []
        t = start
        step = (end - start) / 3
        for w in self.result_text.split():
            words.append(_FakeWord(w, round(t, 3), round(t + step, 3)))
            t += step
        seg = _FakeSeg(self.result_text, start, end,
                       avg_logprob=-0.1, words=words)
        return iter([seg]), SimpleNamespace(language=language)


def _audio_intel_with(engine):
    ai = AudioIntelligence.__new__(AudioIntelligence)
    ai.engine = engine
    return ai


class _NullLog:
    def log_stage(self, *a, **k):
        pass


def test_redecode_targets_low_logprob_and_respects_budget():
    engine = _FakeEngine()
    ai = _audio_intel_with(engine)
    # 20 segments: 5 bad (avg_logprob -2.0), rest good. Budget = 10% = 2.
    segments = []
    for i in range(20):
        bad = i < 5
        segments.append({
            "start_sec": float(i), "end_sec": float(i) + 0.9,
            "text": f"seg {i}", "words": [
                {"word": "seg", "start": float(i), "end": float(i) + 0.4,
                 "confidence": 0.9},
                {"word": str(i), "start": float(i) + 0.4,
                 "end": float(i) + 0.8, "confidence": 0.9},
            ],
            "is_hallucination": False,
            "no_speech_prob": 0.1,
            "avg_logprob": -2.0 if bad else -0.2,
        })
    n = ai._redecode_difficult_segments("fake.wav", segments, "en", _NullLog())
    assert n == 2  # bounded to 10% of 20
    assert len(engine.calls) == 2
    assert all(c["beam_size"] == settings.WHISPER_REDECODE_BEAM
               for c in engine.calls)
    assert all(c["patience"] == 1.5 for c in engine.calls)
    redecoded = [s for s in segments if s.get("redecoded")]
    assert len(redecoded) == 2
    for s in redecoded:
        assert s["text"] == "corrected words here"
        assert s["avg_logprob"] == pytest.approx(-0.1)


def test_redecode_skips_when_engine_lacks_clip_timestamps():
    class OldEngine:
        def transcribe(self, audio_path, language=None):
            raise AssertionError("must not be called")
    ai = _audio_intel_with(OldEngine())
    segments = [{"start_sec": 0.0, "end_sec": 1.0, "text": "x",
                 "words": [], "is_hallucination": True,
                 "no_speech_prob": 0.9, "avg_logprob": -3.0}]
    assert ai._redecode_difficult_segments("f.wav", segments, "en", _NullLog()) == 0


def test_redecode_noop_when_all_segments_good():
    engine = _FakeEngine()
    ai = _audio_intel_with(engine)
    segments = [{"start_sec": 0.0, "end_sec": 1.0, "text": "fine",
                 "words": [{"word": "fine", "start": 0.0, "end": 0.5,
                            "confidence": 0.95}],
                 "is_hallucination": False, "no_speech_prob": 0.05,
                 "avg_logprob": -0.15}]
    assert ai._redecode_difficult_segments("f.wav", segments, "en", _NullLog()) == 0
    assert engine.calls == []


def test_redecode_picks_up_degenerate_words():
    engine = _FakeEngine()
    ai = _audio_intel_with(engine)
    segments = [{
        "start_sec": 0.0, "end_sec": 2.0, "text": "one two three",
        # zero-width word spans — the batched word-timing failure mode
        "words": [{"word": "one", "start": 1.0, "end": 1.0, "confidence": 0.9},
                  {"word": "two", "start": 1.0, "end": 1.0, "confidence": 0.9},
                  {"word": "three", "start": 1.0, "end": 2.0, "confidence": 0.9}],
        "is_hallucination": False, "no_speech_prob": 0.1,
        "avg_logprob": -0.2,  # NOT low — degeneracy alone must trigger
    }]
    n = ai._redecode_difficult_segments("f.wav", segments, "en", _NullLog())
    assert n == 1
    assert segments[0].get("redecoded") is True


# ── Forced alignment fail-safety ────────────────────────────────────────

def test_forced_align_flag_off_is_noop():
    from backend.services import forced_aligner
    prev = settings.SUBTITLE_FORCED_ALIGN
    settings.SUBTITLE_FORCED_ALIGN = False
    try:
        segs = [{"start_sec": 0.0, "end_sec": 1.0, "text": "hi",
                 "words": [{"word": "hi", "start": 0.0, "end": 0.5}]}]
        stats = forced_aligner.refine_word_timestamps("missing.wav", segs, "en")
        assert stats["enabled"] is False
        assert segs[0]["words"][0]["start"] == 0.0
    finally:
        settings.SUBTITLE_FORCED_ALIGN = prev


def test_forced_align_never_raises_without_backend(monkeypatch):
    from backend.services import forced_aligner
    monkeypatch.setattr(forced_aligner, "_get_backend", lambda *a: None)
    segs = [{"start_sec": 0.0, "end_sec": 1.0, "text": "hi",
             "words": [{"word": "hi", "start": 0.0, "end": 0.5}]}]
    stats = forced_aligner.refine_word_timestamps("missing.wav", segs, "en")
    assert stats["enabled"] is False


# ── Distil registry / VRAM tables ───────────────────────────────────────

def test_distil_models_in_vram_table():
    tbl = AudioIntelligence._VRAM_LOAD_GB
    assert ("distil-large-v3", "int8_float16") in tbl
    assert ("distil-large-v3.5", "int8_float16") in tbl
    # Must fit the 4 GB card class: int8 load under large-v3-turbo's
    assert tbl[("distil-large-v3", "int8_float16")] <= tbl[
        ("large-v3-turbo", "int8_float16")]


# ── VRAM ledger ─────────────────────────────────────────────────────────

def test_vram_ledger_graceful_without_gpu(monkeypatch):
    from backend.services import vram_ledger
    monkeypatch.setattr(vram_ledger, "_query_vram", lambda: None)
    assert vram_ledger.snapshot("test_stage", "job1") is None
    assert vram_ledger.get_ledger("job1") == []


def test_vram_ledger_records_stages(monkeypatch):
    from backend.services import vram_ledger
    monkeypatch.setattr(vram_ledger, "_query_vram", lambda: (2048, 4096))
    vram_ledger._ledgers.clear()
    vram_ledger.snapshot("pre_whisper", "job2")
    vram_ledger.snapshot("post_whisper_release", "job2")
    ledger = vram_ledger.get_ledger("job2")
    assert [e["stage"] for e in ledger] == ["pre_whisper", "post_whisper_release"]
    assert ledger[0]["free_mb"] == 2048


# ── Flag defaults ───────────────────────────────────────────────────────

def test_phase3_flag_defaults():
    assert settings.WHISPER_VAD_MIN_SILENCE_MS == 300
    assert settings.WHISPER_VAD_SPEECH_PAD_MS == 150
    assert settings.WHISPER_HALLUCINATION_SILENCE_S == pytest.approx(2.0)
    assert settings.WHISPER_REDECODE_ENABLED is True
    assert settings.WHISPER_REDECODE_LOGPROB == pytest.approx(-0.8)
    assert settings.WHISPER_REDECODE_MAX_FRAC == pytest.approx(0.10)
    assert settings.SUBTITLE_FORCED_ALIGN is True
    assert settings.WHISPER_PREFER_DISTIL_ENGLISH is False
