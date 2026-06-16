"""Regression test for the NLLB / Opus-MT repetition fix (Task 1).

NLLB-distilled and Opus-MT loop on long run-on cues — the same degenerate
n-gram repeat the Whisper decode path already guards against. The NMT decode
sites now pass ``beam_size=5`` plus the anti-repetition CTranslate2 kwargs
(``repetition_penalty`` + ``no_repeat_ngram_size``), feature-detected so an
older build never raises on an unknown kwarg.

These tests exercise the pure-Python wiring with a fake CTranslate2 translator
(no ctranslate2 / torch needed):

  * the kwargs are detected from the binding's signature OR docstring,
  * the decode call sites actually pass them (so the loop guard is active),
  * the call self-heals (drops the kwargs, retries) on an older build, and
  * the resulting translation has no repeated 3-gram.
"""

import sys
import types

import pytest


def _install_config_stub(**overrides):
    mod = types.ModuleType("backend.config")
    defaults = dict(
        NMT_NLLB_MODEL="facebook/nllb-200-distilled-1.3B",
        NMT_OPUS_MT_TEMPLATE="Helsinki-NLP/opus-mt-{src}-{tgt}",
        NMT_MAX_OPUS_PAIRS=5,
        NMT_AUTODOWNLOAD=True,
        NMT_DEVICE="auto",
    )
    defaults.update(overrides)
    mod.settings = types.SimpleNamespace(**defaults)
    sys.modules["backend.config"] = mod
    return mod


@pytest.fixture()
def N(monkeypatch):
    _install_config_stub()
    from backend.services import nmt_translator as N
    # Always start from an un-cached detection so each test controls it.
    monkeypatch.setattr(N, "_CT2_DECODE_KWARGS", None, raising=False)
    return N


# ── 3-gram repetition helper (the acceptance check) ───────────────────────

def _has_3gram_repeat(text: str) -> bool:
    toks = text.split()
    seen = set()
    for i in range(len(toks) - 2):
        g = tuple(toks[i:i + 3])
        if g in seen:
            return True
        seen.add(g)
    return False


def test_3gram_helper_sanity():
    assert _has_3gram_repeat("a b c a b c") is True
    assert _has_3gram_repeat("the cat sat on the mat") is False


# ── feature detection ─────────────────────────────────────────────────────

def _install_fake_ct2(translate_batch):
    ct2 = types.ModuleType("ctranslate2")

    class _Translator:
        pass

    _Translator.translate_batch = translate_batch
    ct2.Translator = _Translator
    sys.modules["ctranslate2"] = ct2
    return ct2


def test_decode_kwargs_detected_from_signature(N, monkeypatch):
    # A plain Python function exposes its params to inspect.signature.
    def translate_batch(self, source, target_prefix=None, beam_size=2,
                        repetition_penalty=1.0, no_repeat_ngram_size=0):
        return None

    _install_fake_ct2(translate_batch)
    kw = N._ct2_decode_kwargs()
    assert kw == {"repetition_penalty": 1.1, "no_repeat_ngram_size": 3}


def test_decode_kwargs_detected_from_docstring_when_signature_unreadable(N, monkeypatch):
    # Emulate a pybind11 binding: inspect.signature raises, but the docstring
    # carries the call signature (this is the real CTranslate2 case).
    def translate_batch(*a, **k):
        return None
    translate_batch.__doc__ = (
        "translate_batch(self, source, target_prefix=None, beam_size=2, "
        "repetition_penalty=1, no_repeat_ngram_size=0) -> list")
    _install_fake_ct2(translate_batch)

    import inspect

    def _boom(_f):
        raise ValueError("no signature found for builtin")
    monkeypatch.setattr(inspect, "signature", _boom)

    kw = N._ct2_decode_kwargs()
    assert kw == {"repetition_penalty": 1.1, "no_repeat_ngram_size": 3}


def test_decode_kwargs_pass_anyway_when_unreadable(N, monkeypatch):
    # Neither signature nor docstring readable (e.g. python -OO): still pass the
    # kwargs (build is pinned >= 4.0); the call site self-heals if wrong.
    def translate_batch(*a, **k):
        return None
    translate_batch.__doc__ = None
    _install_fake_ct2(translate_batch)

    import inspect
    monkeypatch.setattr(inspect, "signature",
                        lambda _f: (_ for _ in ()).throw(ValueError()))

    kw = N._ct2_decode_kwargs()
    assert kw == {"repetition_penalty": 1.1, "no_repeat_ngram_size": 3}


# ── self-heal on an older build ───────────────────────────────────────────

class _Hyp:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


def test_translate_batch_self_heals_on_unexpected_kwarg(N):
    calls = []

    class _OldTranslator:
        def translate_batch(self, source_list, **kwargs):
            calls.append(kwargs)
            if "repetition_penalty" in kwargs or "no_repeat_ngram_size" in kwargs:
                raise TypeError(
                    "translate_batch(): got an unexpected keyword argument "
                    "'repetition_penalty'")
            return [_Hyp([["ok"]])]

    out = N._ct2_translate_batch(_OldTranslator(), [["x"]], beam_size=5)
    assert out[0].hypotheses[0] == ["ok"]
    # First call carried the kwargs (rejected), retry dropped them.
    assert len(calls) == 2
    assert "repetition_penalty" in calls[0] and "repetition_penalty" not in calls[1]
    # Cache demoted so subsequent calls skip the doomed kwargs.
    assert N._CT2_DECODE_KWARGS == {}


# ── end-to-end: decode sites pass the guard and output is clean ───────────

class _FakeTokenizer:
    """Whitespace SentencePiece stand-in."""

    def encode_as_pieces(self, text):
        return text.split()

    def decode(self, pieces):
        return " ".join(pieces)


class _AntiRepeatTranslator:
    """Fake CT2 translator that LOOPS unless the anti-repetition kwargs are
    supplied — so a clean result proves the guard was actually passed."""

    def __init__(self):
        self.last_kwargs = None

    def translate_batch(self, source_list, **kwargs):
        self.last_kwargs = kwargs
        src = source_list[0]
        # Drop the leading flores/lang token + trailing </s> if present.
        body = [t for t in src if not (t.endswith("_Latn") or t.endswith("_Jpan")
                                       or t == "</s>")]
        prefix = kwargs.get("target_prefix")
        tgt_tok = prefix[0][0] if prefix else None
        if "no_repeat_ngram_size" in kwargs or "repetition_penalty" in kwargs:
            hyp = (([tgt_tok] if tgt_tok else []) + body)        # clean echo
        else:
            hyp = (([tgt_tok] if tgt_tok else []) + body + body + body)  # loop
        return [_Hyp([hyp])]


def test_nllb_chunk_decode_passes_guard_and_is_loopfree(N, monkeypatch):
    monkeypatch.setattr(N, "_CT2_DECODE_KWARGS",
                        {"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
                        raising=False)
    tr = N.NMTTranslator(model_id="facebook/nllb-200-distilled-1.3B")
    tr._translator = _AntiRepeatTranslator()
    tr._tokenizer = _FakeTokenizer()
    tr._loaded = True

    run_on = "the runner kept running and running over the long winding road"
    out = tr._translate_chunk(run_on, "eng_Latn", "eng_Latn")

    assert tr._translator.last_kwargs.get("beam_size") == 5
    assert tr._translator.last_kwargs.get("repetition_penalty") == 1.1
    assert tr._translator.last_kwargs.get("no_repeat_ngram_size") == 3
    assert out and not _has_3gram_repeat(out), f"looped output: {out!r}"


def test_opus_decode_passes_guard_and_is_loopfree(N, monkeypatch):
    monkeypatch.setattr(N, "_CT2_DECODE_KWARGS",
                        {"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
                        raising=False)
    op = N.OpusMTTranslator("es", "en")
    op._translator = _AntiRepeatTranslator()
    op._tokenizer = _FakeTokenizer()
    op._loaded = True

    out = op.translate_batch(["el corredor siguio corriendo y corriendo sin parar"])
    assert op._translator.last_kwargs.get("beam_size") == 5
    assert op._translator.last_kwargs.get("no_repeat_ngram_size") == 3
    assert out and not _has_3gram_repeat(out[0]), f"looped output: {out[0]!r}"
