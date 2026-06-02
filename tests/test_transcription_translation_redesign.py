"""Regression tests for the offline-translation + transcription-quality redesign.

Covers the units that don't need the heavy ML stack (Whisper/CTranslate2 model
weights, a GPU, or media): the NMT convert bug fix, the :free-model translation
policy, the anti-repetition decoding kwargs, music-span speech suppression, the
fuzzy repetition-loop quarantine, and source resegmentation.

Heavy third-party deps are stubbed so the target modules import without a GPU:
  * ``cv2`` / ``numpy``  → so ``reframer_audio`` (Task 3 helper) imports.
  * ``backend.services.ai_orchestrator`` → so ``translator`` imports without the
    provider SDK chain (Task 2). ``translator`` only needs ``AIOrchestrator`` as
    a type hint and ``ProviderRateLimitError`` from ``providers.base``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Data-dir-at-import modules resolve from HOME.
_TMP = Path(tempfile.mkdtemp(prefix="clipai_redesign_test_"))
os.environ["HOME"] = str(_TMP)

# ── stubs so the target modules import without GPU/provider deps ──
for _name in ("cv2", "numpy"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

if "backend.services.ai_orchestrator" not in sys.modules:
    _orch_mod = types.ModuleType("backend.services.ai_orchestrator")

    class _AIOrchestrator:  # minimal stand-in (only used as a type hint)
        ...

    _orch_mod.AIOrchestrator = _AIOrchestrator
    sys.modules["backend.services.ai_orchestrator"] = _orch_mod

from backend.models import TranscriptSegment, WordTimestamp  # noqa: E402
import backend.services.nmt_translator as nmt  # noqa: E402
import backend.services.transcript_dedup as dedup  # noqa: E402
import backend.services.audio_analyzer as aa  # noqa: E402
import backend.services.reframer_audio as ra  # noqa: E402
import backend.services.translator as translator  # noqa: E402
from backend.services.sentence_segmenter import resegment_by_sentence  # noqa: E402


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ════════════════════════════════════════════════════════════════════════
#  Task 1 — NMT convert bug: temp-dir then atomic rename
# ════════════════════════════════════════════════════════════════════════

def _install_fake_converter(monkeypatch, *, boom: bool = False):
    """Install a fake ``ctranslate2.converters`` whose ``convert`` mimics the
    REAL behaviour: it refuses a pre-existing output dir unless force=True.
    With ``boom=True`` it writes a partial dir then raises (interrupted DL)."""
    mod = types.ModuleType("ctranslate2.converters")

    class _Converter:
        def __init__(self, model_id):
            self.model_id = model_id

        def convert(self, out_dir, quantization="int8", force=False):
            # Faithful to CTranslate2: a pre-existing dir is refused.
            if os.path.exists(out_dir) and not force:
                raise RuntimeError(
                    f"output directory {out_dir} already exists, use --force")
            if boom:
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "model.bin"), "w") as f:
                    f.write("partial")
                raise RuntimeError("download interrupted")
            os.makedirs(out_dir, exist_ok=False)
            with open(os.path.join(out_dir, "model.bin"), "wb") as f:
                f.write(b"X" * 64)
            with open(os.path.join(out_dir, "sentencepiece.bpe.model"), "wb") as f:
                f.write(b"Y" * 32)

    mod.TransformersConverter = _Converter
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", mod)


def _model_present(d):
    return (os.path.exists(os.path.join(d, "model.bin"))
            and os.path.exists(os.path.join(d, "sentencepiece.bpe.model")))


def test_convert_into_fresh_target_succeeds(monkeypatch, tmp_path):
    """The everyday path that USED to fail: target dir does not yet exist.

    The old code did ``makedirs(target_dir)`` then ``convert(force=False)``,
    which the converter refuses — so this asserts the fix (convert into a temp
    dir, then rename onto target)."""
    monkeypatch.setattr(nmt, "_models_dir", lambda: str(tmp_path / "models"))
    _install_fake_converter(monkeypatch)
    target = str(tmp_path / "models" / "nllb" / "facebook_nllb")
    nmt._convert_with_cleanup("facebook/nllb-200-distilled-600M", target, "NLLB")
    assert _model_present(target)
    # No leftover .convert-* temp holders next to the target.
    parent = os.path.dirname(target)
    assert not [d for d in os.listdir(parent) if d.startswith(".convert-")]


def test_convert_replaces_stale_target(monkeypatch, tmp_path):
    """A pre-existing/partial target is cleared and replaced, not refused."""
    monkeypatch.setattr(nmt, "_models_dir", lambda: str(tmp_path / "models"))
    _install_fake_converter(monkeypatch)
    target = str(tmp_path / "stale")
    os.makedirs(target)
    with open(os.path.join(target, "garbage.txt"), "w") as f:
        f.write("stale half-write")
    nmt._convert_with_cleanup("m", target, "NLLB")
    assert _model_present(target)
    assert not os.path.exists(os.path.join(target, "garbage.txt"))


def test_convert_failure_cleans_up_and_unblocks_retry(monkeypatch, tmp_path):
    """On failure: no partial target, no temp holder — and a retry succeeds."""
    monkeypatch.setattr(nmt, "_models_dir", lambda: str(tmp_path / "models"))
    target = str(tmp_path / "models" / "nllb" / "x")
    os.makedirs(os.path.dirname(target), exist_ok=True)

    _install_fake_converter(monkeypatch, boom=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        nmt._convert_with_cleanup("m", target, "NLLB")
    assert not os.path.exists(target)
    parent = os.path.dirname(target)
    assert not [d for d in os.listdir(parent) if d.startswith(".convert-")]

    # Retry with a healthy converter — a stale dir would have blocked it before.
    _install_fake_converter(monkeypatch, boom=False)
    nmt._convert_with_cleanup("m", target, "NLLB")
    assert _model_present(target)


# ════════════════════════════════════════════════════════════════════════
#  Task 2 — never grind a :free OpenRouter model; offline is the default
# ════════════════════════════════════════════════════════════════════════

def test_is_free_openrouter_model():
    assert translator._is_free_openrouter_model("qwen/qwen3-next-80b-a3b-instruct:free")
    assert translator._is_free_openrouter_model("openrouter:google/gemma-3-27b-it:free")
    assert not translator._is_free_openrouter_model("google/gemini-2.5-pro")
    assert not translator._is_free_openrouter_model("")
    assert not translator._is_free_openrouter_model(None)


class _FakeProvider:
    def __init__(self, provider_name, text_model_name):
        self.provider_name = provider_name
        self.text_model_name = text_model_name


class _FakeOrchestrator:
    """Records text_completion calls so a test can prove the free model was
    NOT ground through."""

    def __init__(self, chain):
        self._chain = chain
        self.text_completion_calls = 0

    def _get_active_chain(self):
        return self._chain

    async def text_completion(self, *a, **k):
        self.text_completion_calls += 1
        from backend.services.providers.base import ProviderRateLimitError
        raise ProviderRateLimitError("429 rate limited (should never be reached)")


def test_llm_translation_model_is_free_detection():
    free_chain = [_FakeProvider("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free")]
    paid_chain = [_FakeProvider("openrouter", "google/gemini-2.5-pro")]
    ollama_first = [_FakeProvider("ollama", "qwen2.5:3b"),
                    _FakeProvider("openrouter", "x:free")]

    # explicit override wins
    is_free, label = translator._llm_translation_model_is_free(
        _FakeOrchestrator(paid_chain), "anything:free")
    assert is_free and label == "anything:free"

    # chain inspection: free OpenRouter first → free
    is_free, _ = translator._llm_translation_model_is_free(_FakeOrchestrator(free_chain), None)
    assert is_free
    # paid OpenRouter first → not free
    is_free, _ = translator._llm_translation_model_is_free(_FakeOrchestrator(paid_chain), None)
    assert not is_free
    # a non-OpenRouter provider first (local Ollama) → not the free-grind case
    is_free, _ = translator._llm_translation_model_is_free(_FakeOrchestrator(ollama_first), None)
    assert not is_free


def _segs(n=3):
    return [TranscriptSegment(text=f"セリフ{i}", start=float(i), end=float(i) + 1.0,
                              speaker="Speaker 1") for i in range(n)]


def test_free_model_skips_llm_and_fails_fast(monkeypatch):
    """LLM engine + free OpenRouter model + no Ollama → fail FAST with an
    actionable TranslationFailedError, and text_completion is never called
    (no 429 storm)."""
    monkeypatch.setattr(translator.settings, "TRANSLATION_ENGINE", "llm", raising=False)
    monkeypatch.setattr(translator.settings, "OPENROUTER_TRANSLATION_MODEL", "", raising=False)
    monkeypatch.setattr(translator.settings, "OLLAMA_TRANSLATION_MODEL", "", raising=False)
    orch = _FakeOrchestrator([_FakeProvider("openrouter", "qwen/qwen3:free")])

    with pytest.raises(translator.TranslationFailedError) as ei:
        _run(translator.translate_segments_with_fallback(
            _segs(), "ja", "en", orch))
    msg = str(ei.value).lower()
    assert "free" in msg and ("nmt" in msg or "offline" in msg)
    assert orch.text_completion_calls == 0  # never ground the free model


def test_free_model_skips_openrouter_but_uses_ollama(monkeypatch):
    """With a free OpenRouter model AND an Ollama model configured, the router
    skips OpenRouter (no text_completion) and proceeds to the Ollama path."""
    monkeypatch.setattr(translator.settings, "TRANSLATION_ENGINE", "llm", raising=False)
    monkeypatch.setattr(translator.settings, "OPENROUTER_TRANSLATION_MODEL", "", raising=False)
    monkeypatch.setattr(translator.settings, "OLLAMA_TRANSLATION_MODEL", "qwen2.5:3b", raising=False)
    orch = _FakeOrchestrator([_FakeProvider("openrouter", "qwen/qwen3:free")])

    # Make the Ollama model "unavailable" so we get a distinct, Ollama-specific
    # failure — proving control reached the Ollama path, not the OpenRouter LLM.
    async def _no_ollama(_model):
        return False
    monkeypatch.setattr(translator, "_ensure_ollama_model", _no_ollama)

    with pytest.raises(RuntimeError) as ei:
        _run(translator.translate_segments_with_fallback(_segs(), "ja", "en", orch))
    assert "qwen2.5:3b" in str(ei.value)          # the Ollama model name
    assert orch.text_completion_calls == 0         # OpenRouter LLM was skipped


# ════════════════════════════════════════════════════════════════════════
#  Task 3 — anti-repetition decoding kwargs
# ════════════════════════════════════════════════════════════════════════

def _modern_transcribe(audio, *, language=None, beam_size=5, word_timestamps=False,
                       condition_on_previous_text=True, no_repeat_ngram_size=0,
                       compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                       repetition_penalty=1.0, temperature=0.0):  # pragma: no cover
    ...


def _old_transcribe(audio, *, language=None, condition_on_previous_text=True,
                    compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                    temperature=0.0):  # pragma: no cover
    ...


def _kwargs_only(audio, **kw):  # pragma: no cover
    ...


def test_decoding_kwargs_default_condition_off():
    out = ra._decoding_kwargs(_modern_transcribe)
    assert out["condition_on_previous_text"] is False
    assert out["no_repeat_ngram_size"] == 3
    assert "compression_ratio_threshold" in out and "log_prob_threshold" in out
    assert "repetition_penalty" in out and "temperature" in out


def test_decoding_kwargs_filters_unsupported():
    """An older build without no_repeat_ngram_size / repetition_penalty must not
    receive them (would raise at call time)."""
    out = ra._decoding_kwargs(_old_transcribe)
    assert "no_repeat_ngram_size" not in out
    assert "repetition_penalty" not in out
    assert out["condition_on_previous_text"] is False
    assert "temperature" in out


def test_decoding_kwargs_kwargs_only_is_empty():
    # A **kwargs catch-all does not count as explicit support → splat nothing.
    assert ra._decoding_kwargs(_kwargs_only) == {}


def test_decoding_kwargs_gapfill_override(monkeypatch):
    # Even if the global default is flipped ON, gap-fill forces it OFF.
    monkeypatch.setattr(ra.settings, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", True, raising=False)
    assert ra._decoding_kwargs(_modern_transcribe)["condition_on_previous_text"] is True
    assert ra._decoding_kwargs(
        _modern_transcribe, condition_on_previous_text=False
    )["condition_on_previous_text"] is False


# ════════════════════════════════════════════════════════════════════════
#  Task 4 — fuzzy repetition-loop quarantine + music suppression
# ════════════════════════════════════════════════════════════════════════

def test_fuzzy_repetition_loop_collapses_near_identical_long_blocks():
    base = "アフターコロニー195 人類は宇宙に進出しコロニーで暮らすようになった"
    segs = [
        {"text": base + "。", "start": 48.0, "end": 55.0},
        {"text": "本編のセリフです", "start": 60.0, "end": 62.0},     # real dialogue
        {"text": base + "、", "start": 91.0, "end": 98.0},            # near-dup (punct)
        {"text": base.replace("人類", "人 類"), "start": 168.0, "end": 175.0},  # near-dup (space)
        {"text": "了解", "start": 200.0, "end": 200.5},
        {"text": base, "start": 350.0, "end": 357.0},
        {"text": "了解", "start": 360.0, "end": 360.5},
        {"text": base + "。", "start": 445.0, "end": 452.0},
        {"text": "了解", "start": 500.0, "end": 500.5},
        {"text": base + "。", "start": 591.0, "end": 598.0},
        {"text": "了解", "start": 610.0, "end": 610.5},                # 4th → dropped (cap 3)
    ]
    kept, dropped = dedup.drop_repetition_loops(segs)
    texts = [s["text"] for s in kept]
    assert sum(1 for t in texts if t.startswith("アフターコロニー195")) == 1
    assert sum(1 for t in texts if dedup._normalize_text(t) == "了解") == 3
    assert "本編のセリフです" in texts
    assert dropped == 6


def test_normalize_text_strips_punctuation():
    assert dedup._normalize_text("fine,") == dedup._normalize_text("fine")
    assert dedup._normalize_text("テスト。") == dedup._normalize_text("テスト、")


def test_music_suppression_drops_lyrics_keeps_dialogue_and_markers():
    music_spans = [(26.0, 96.0)]  # a sustained OP-song span
    transcript = [
        {"text": "本編前のセリフ", "start": 10.0, "end": 13.0},     # before song → keep
        {"text": "ああああああ", "start": 30.0, "end": 34.0},        # hallucinated → drop
        {"text": "そらをかけるよ", "start": 40.0, "end": 46.0},      # fake lyric → drop
        {"text": "[♪ music ♪]", "start": 50.0, "end": 60.0},  # marker → keep
        {"text": "ラララ", "start": 70.0, "end": 92.0},             # fake lyric → drop
        {"text": "作戦を開始する", "start": 100.0, "end": 104.0},    # after song → keep
        {"text": "半分だけ", "start": 90.0, "end": 110.0},          # 30% in music → keep
    ]
    kept, suppressed = aa.suppress_speech_in_music_spans(
        transcript, music_spans, min_overlap_frac=0.6)
    kt = [s["text"] for s in kept]
    assert len(suppressed) == 3
    assert "[♪ music ♪]" in kt
    assert "本編前のセリフ" in kt and "作戦を開始する" in kt and "半分だけ" in kt
    assert all(s["text"] in ("ああああああ", "そらをかけるよ", "ラララ") for s in suppressed)


def test_music_suppression_noops_without_spans():
    t = [{"text": "x", "start": 0.0, "end": 1.0}]
    kept, sup = aa.suppress_speech_in_music_spans(t, [])
    assert kept == t and sup == []


# ════════════════════════════════════════════════════════════════════════
#  Task 5 — source resegmentation into one-utterance cues
# ════════════════════════════════════════════════════════════════════════

def test_resegment_splits_runon_into_one_utterance_cues():
    def W(w, s, e):
        return WordTimestamp(word=w, start=s, end=e)
    seg = TranscriptSegment(
        text="Hello there. How are you? I am fine.",
        start=0.0, end=6.0, speaker="Speaker 1",
        words=[W("Hello", 0.0, 0.4), W(" there.", 0.4, 1.0),
               W(" How", 1.2, 1.5), W(" are", 1.5, 1.7), W(" you?", 1.7, 2.2),
               W(" I", 3.0, 3.1), W(" am", 3.1, 3.3), W(" fine.", 3.3, 4.0)],
    )
    out = resegment_by_sentence([seg])
    assert len(out) == 3
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 1e-6  # monotonic, non-overlapping
