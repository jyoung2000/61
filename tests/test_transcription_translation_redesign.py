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
import importlib.machinery  # noqa: E402
for _name in ("cv2", "numpy"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        # A valid __spec__ so importlib.util.find_spec(_name) (used by some
        # libraries' availability probes) doesn't raise on the bare stub.
        _stub.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = _stub

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

def _install_fake_converter(monkeypatch, *, boom: bool = False, with_tokenizer: bool = True):
    """Install a fake ``ctranslate2.converters`` whose ``convert`` mimics the
    REAL behaviour: it refuses a pre-existing output dir unless force=True, and
    (like real CTranslate2) writes ``model.bin`` + vocab but the SentencePiece
    tokenizer only when ``with_tokenizer`` (real CT2 does NOT — see the
    tokenizer-fetch test). With ``boom=True`` it writes a partial dir then
    raises (interrupted DL)."""
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
            if with_tokenizer:
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


def test_convert_fetches_missing_tokenizer(monkeypatch, tmp_path):
    """Real CTranslate2 writes model.bin but NOT the .spm tokenizer the runtime
    needs — without it the model is on disk yet is_available() is False and the
    job silently falls back to the LLM (the user's 599 MB-model-no-tokenizer
    stall). The convert must fetch + save the tokenizer alongside the model."""
    monkeypatch.setattr(nmt, "_models_dir", lambda: str(tmp_path / "models"))
    _install_fake_converter(monkeypatch, with_tokenizer=False)

    # Stand-in HF download → returns a path to a fake sentencepiece file.
    import huggingface_hub
    src_spm = tmp_path / "src_sentencepiece.bpe.model"
    src_spm.write_bytes(b"SPM" * 100)

    def _fake_dl(model_id, filename, *a, **k):
        if filename == "sentencepiece.bpe.model":
            return str(src_spm)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_dl)

    target = str(tmp_path / "models" / "nllb" / "m")
    nmt._convert_with_cleanup(
        "facebook/nllb-200-distilled-600M", target, "NLLB",
        tokenizer_files=("sentencepiece.bpe.model",),
    )
    # Both files present → _model_files_present() would now be True.
    assert os.path.exists(os.path.join(target, "model.bin"))
    assert os.path.exists(os.path.join(target, "sentencepiece.bpe.model"))


def test_ensure_nllb_repairs_missing_tokenizer_without_reconvert(monkeypatch, tmp_path):
    """A pre-fix model dir (model.bin, no tokenizer) is repaired by fetching the
    ~5 MB tokenizer — NOT by re-downloading + re-converting 2.4 GB."""
    monkeypatch.setattr(nmt, "_models_dir", lambda: str(tmp_path / "models"))
    model_id = "facebook/nllb-200-distilled-600M"
    target = nmt._nllb_dir(model_id)
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "model.bin"), "wb") as f:
        f.write(b"X" * 64)  # model present, tokenizer absent (the broken state)

    import huggingface_hub
    src = tmp_path / "spm"
    src.write_bytes(b"SPM" * 50)

    def _fake_dl(mid, filename, *a, **k):
        if filename == "sentencepiece.bpe.model":
            return str(src)
        raise FileNotFoundError(filename)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_dl)

    # The converter must NOT run for a tokenizer-only repair.
    boom = types.ModuleType("ctranslate2.converters")

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("convert must NOT run during a tokenizer-only repair")
    boom.TransformersConverter = _Boom
    monkeypatch.setitem(sys.modules, "ctranslate2.converters", boom)
    # _has_dependencies() probes these — provide light stand-ins so the repair's
    # final check passes without a real ctranslate2/sentencepiece install.
    monkeypatch.setitem(sys.modules, "ctranslate2", types.ModuleType("ctranslate2"))
    monkeypatch.setitem(sys.modules, "sentencepiece", types.ModuleType("sentencepiece"))

    out = nmt.ensure_nllb_downloaded(model_id)
    assert os.path.exists(os.path.join(out, "model.bin"))
    assert os.path.exists(os.path.join(out, "sentencepiece.bpe.model"))


def test_torch_load_guard_neutralised_during_convert(monkeypatch):
    """The convert temporarily neutralises transformers' torch<2.6 torch.load
    CVE guard (so NLLB/Opus-MT .bin checkpoints load on the pinned torch 2.5.1)
    and restores it afterwards. Uses injected fake transformers modules so it is
    deterministic without the torch-only real import; the mechanism is also
    verified against the real transformers guard during development."""
    def _guard(*a, **k):
        raise ValueError("torch<2.6 torch.load guard (CVE-2025-32434)")
    pkg = types.ModuleType("transformers"); pkg.__path__ = []
    utils = types.ModuleType("transformers.utils"); utils.__path__ = []
    mu = types.ModuleType("transformers.modeling_utils")
    iu = types.ModuleType("transformers.utils.import_utils")
    mu.check_torch_load_is_safe = _guard
    iu.check_torch_load_is_safe = _guard
    for name, mod in [("transformers", pkg), ("transformers.utils", utils),
                      ("transformers.modeling_utils", mu),
                      ("transformers.utils.import_utils", iu)]:
        monkeypatch.setitem(sys.modules, name, mod)

    with pytest.raises(ValueError):
        mu.check_torch_load_is_safe()              # guard fires (simulating torch<2.6)
    with nmt._allow_trusted_torch_load():
        mu.check_torch_load_is_safe()              # neutralised at the call-site binding
        iu.check_torch_load_is_safe()              # and at the definition site
    assert mu.check_torch_load_is_safe is _guard   # restored after the convert
    assert iu.check_torch_load_is_safe is _guard
    with pytest.raises(ValueError):
        mu.check_torch_load_is_safe()


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


def test_repetition_filter_keeps_distinct_longer_extension():
    # A distinct longer line that merely CONTAINS an earlier line must NOT be
    # dropped (the containment-branch false-positive). Their lengths differ by
    # more than the similarity ratio, so the length gate excludes them.
    segs = [
        {"text": "I will protect this colony from the enemy", "start": 1.0, "end": 4.0},
        {"text": "I will protect this colony from the enemy until my dying breath",
         "start": 300.0, "end": 304.0},
    ]
    kept, dropped = dedup.drop_repetition_loops(segs)
    assert dropped == 0 and len(kept) == 2


def test_repetition_filter_is_bounded_on_many_distinct_long_cues():
    # 1500 distinct long cues must not trigger an O(n^2) blow-up.
    import time
    segs = [{"text": f"This is distinct dialogue line number {i:04d} here.",
             "start": float(i), "end": float(i) + 1.0} for i in range(1500)]
    t0 = time.perf_counter()
    kept, dropped = dedup.drop_repetition_loops(segs)
    assert dropped == 0 and len(kept) == 1500
    # Bounded (exact-key O(1) + fixed fuzzy window) vs the ~24s full-history
    # O(n^2) scan — generous ceiling so it's not flaky on slow CI but still
    # unambiguously rules out the quadratic blow-up.
    assert (time.perf_counter() - t0) < 6.0


def test_repetition_filter_keeps_all_music_markers():
    # OP + ED + two insert songs → four identical [♪ music ♪] cues. All must
    # survive the repetition-loop filter (markers are intentional, not loops).
    segs = [
        {"text": "[♪ music ♪]", "start": 26.0, "end": 96.0},
        {"text": "本編のセリフ", "start": 120.0, "end": 123.0},
        {"text": "[♪ music ♪]", "start": 300.0, "end": 340.0},
        {"text": "[♪ music ♪]", "start": 700.0, "end": 760.0},
        {"text": "[♪ music ♪]", "start": 1300.0, "end": 1380.0},
    ]
    kept, dropped = dedup.drop_repetition_loops(segs)
    assert sum(1 for s in kept if s["text"] == "[♪ music ♪]") == 4
    assert dropped == 0


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

# ════════════════════════════════════════════════════════════════════════
#  Translation AI dropdown — only translation-capable models
# ════════════════════════════════════════════════════════════════════════

def test_translation_capability_filter():
    pytest.importorskip("fastapi")
    try:
        from backend.routers.settings import _is_translation_capable as cap
    except Exception:
        pytest.skip("settings router deps unavailable in this environment")
    # Excluded: reasoning / 'thinking' models (break the JSON translator) and
    # image / audio generators — the classes that "can't be used for translation".
    assert cap("liquid/lfm-2.5-1.2b-thinking:free") is False   # the model the user hit
    assert cap("openai/o3") is False
    assert cap("openai/o4-mini") is False
    assert cap("openai/o1-preview") is False
    assert cap("deepseek/deepseek-r1") is False
    assert cap("openai/gpt-5-image", ["image", "text"]) is False
    assert cap("openai/gpt-audio", ["audio", "text"]) is False
    # Kept: standard instruct chat models — including ids that merely contain
    # an 'o' (must not false-match the o1/o3/o4 reasoning rule).
    assert cap("google/gemini-3.1-flash-lite") is True
    assert cap("openai/gpt-4o-mini") is True
    assert cap("anthropic/claude-opus-4") is True
    assert cap("x-ai/grok-2-1212") is True


# ════════════════════════════════════════════════════════════════════════
#  Translation polish — readability only, preserve meaning + timing
# ════════════════════════════════════════════════════════════════════════

def test_translation_polish_mode_is_readability_only():
    import backend.services.transcript_polisher as tp
    batch = [{"index": 0, "text": "the colony destroyed by enemy", "start": 1.0, "end": 3.0}]
    asr = tp._build_user_prompt(batch, [], [], "en", None, mode="asr")
    tr = tp._build_user_prompt(batch, [], [], "en", None, mode="translation")
    # Translation prompt is readability-only and NOT Whisper/ASR-framed.
    assert "phonetic" not in tr.lower() and "whisper" not in tr.lower()
    assert "re-translate" in tr.lower() and "meaning" in tr.lower()
    # The ASR prompt keeps its phonetic-correction framing.
    assert "phonetic" in asr.lower() or "whisper" in asr.lower()
    # Distinct translation system prompt that still forbids timing changes.
    assert tp._SYSTEM_PROMPT_TRANSLATION != tp._SYSTEM_PROMPT
    assert "Never change timing" in tp._SYSTEM_PROMPT_TRANSLATION
    # CJK target → CJK punctuation guidance, never Western punctuation.
    tr_ja = tp._build_user_prompt(batch, [], [], "ja", None, mode="translation")
    assert "。" in tr_ja


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
