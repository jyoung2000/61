"""Tests for the offline NMT → MTPE post-edit pass (Task 4).

After the offline NMT engine produces a fluent draft, the dedicated local
translation model (OLLAMA_TRANSLATION_MODEL) post-edits it as a machine-
translation post-editor — fixing fluency / idioms / glossary WITHOUT altering
cue count or timing. The pass is fully fail-soft: a bad/short response, a count
mismatch, the feature disabled, or no Ollama host all keep the raw NMT draft.

The Ollama call is faked, so no model / network is needed.
"""

import asyncio
import json
import re
import sys
import types

import pytest


def _stub_provider_sdks():
    """Stub optional provider SDKs not installed here so ai_orchestrator — and
    thus translator — imports (mirrors test_translation_fail_loud.py)."""
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    if "groq" not in sys.modules:
        gr = types.ModuleType("groq")
        gr.AsyncGroq = object
        sys.modules["groq"] = gr


_stub_provider_sdks()

from backend.models import TranscriptSegment  # noqa: E402
from backend.services import translator as T  # noqa: E402


def _seg(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _draft():
    # NLLB draft (English) for a Japanese source — rough but complete.
    return [
        _seg(0.0, 1.4, "He to the store went."),
        _seg(1.4, 3.2, "She a book bought."),
        _seg(3.2, 5.0, "They together home returned."),
    ]


def _source():
    return [
        _seg(0.0, 1.4, "彼は店に行った。"),
        _seg(1.4, 3.2, "彼女は本を買った。"),
        _seg(3.2, 5.0, "彼らは一緒に家に帰った。"),
    ]


@pytest.fixture(autouse=True)
def _mtpe_settings(monkeypatch):
    """Enable MTPE with a configured (fake) Ollama host + translation model."""
    monkeypatch.setattr(T.settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True, raising=False)
    monkeypatch.setattr(T.settings, "OFFLINE_TRANSLATION_MTPE_NUM_CTX", 8192, raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_TRANSLATION_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(T.settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(T.settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)


def _install_fake_ollama(monkeypatch, *, response=None, capture=None):
    """Patch the direct Ollama call. By default returns a valid edited JSON
    array sized to the batch (parsed from the prompt)."""
    async def fake(prompt, model, timeout=180.0, num_ctx=4096):
        if capture is not None:
            capture["num_ctx"] = num_ctx
            capture["model"] = model
            capture.setdefault("calls", 0)
            capture["calls"] += 1
        if response is not None:
            return response
        m = (re.search(r"JSON array of (\d+) objects", prompt)
             or re.search(r"EXACTLY (\d+)", prompt))
        n = int(m.group(1)) if m else 1
        return json.dumps([f"edited line {i + 1}" for i in range(n)])

    monkeypatch.setattr(T, "_translate_batch_via_ollama", fake)


def test_mtpe_preserves_cue_count_and_timing(monkeypatch):
    _install_fake_ollama(monkeypatch)
    draft = _draft()
    out = asyncio.run(T.mtpe_postedit_offline(
        list(draft), _source(), "ja", "en", glossary=None))

    assert len(out) == len(draft)                      # cue count preserved
    for o, d in zip(out, draft):
        assert o.start == d.start and o.end == d.end    # timing preserved
        assert o.speaker == d.speaker
        assert o.text != d.text                         # text was post-edited
        assert o.text.startswith("edited line")


def test_mtpe_uses_raised_num_ctx_8192(monkeypatch):
    cap = {}
    _install_fake_ollama(monkeypatch, capture=cap)
    asyncio.run(T.mtpe_postedit_offline(_draft(), _source(), "ja", "en"))
    assert cap["num_ctx"] == 8192
    assert cap["model"] == "qwen2.5:3b"
    assert cap["calls"] >= 1


def test_mtpe_disabled_returns_raw_draft(monkeypatch):
    monkeypatch.setattr(T.settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", False)
    cap = {}
    _install_fake_ollama(monkeypatch, capture=cap)
    draft = _draft()
    out = asyncio.run(T.mtpe_postedit_offline(draft, _source(), "ja", "en"))
    assert out is draft                                 # untouched
    assert "calls" not in cap                           # model never called


def test_mtpe_no_ollama_host_returns_raw_draft(monkeypatch):
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "")
    cap = {}
    _install_fake_ollama(monkeypatch, capture=cap)
    draft = _draft()
    out = asyncio.run(T.mtpe_postedit_offline(draft, _source(), "ja", "en"))
    assert out is draft
    assert "calls" not in cap


def test_mtpe_bad_response_falls_back_to_draft(monkeypatch):
    # Non-JSON garbage → correct_transcript keeps the draft per-batch → the
    # returned cues equal the draft (count + timing intact, text unchanged).
    _install_fake_ollama(monkeypatch, response="totally not json")
    draft = _draft()
    out = asyncio.run(T.mtpe_postedit_offline(list(draft), _source(), "ja", "en"))
    assert len(out) == len(draft)
    for o, d in zip(out, draft):
        assert o.start == d.start and o.end == d.end
        assert o.text == d.text                         # raw draft kept


def test_mtpe_count_mismatch_keeps_draft(monkeypatch):
    # If the post-editor ever returns a different cue COUNT, fail-soft to draft.
    draft = _draft()

    async def _bad_correct(*a, **k):
        return draft[:-1]                               # drops a cue

    monkeypatch.setattr(
        "backend.services.transcript_polisher.correct_transcript", _bad_correct)
    out = asyncio.run(T.mtpe_postedit_offline(list(draft), _source(), "ja", "en"))
    assert len(out) == len(draft)
    assert [o.text for o in out] == [d.text for d in draft]


def test_mtpe_rejects_source_language_reintroduction(monkeypatch):
    # A post-edit that puts CJK back into an English track is rejected.
    draft = _draft()

    async def _regress(*a, **k):
        return [_seg(s.start, s.end, "彼は店に行った。") for s in draft]

    monkeypatch.setattr(
        "backend.services.transcript_polisher.correct_transcript", _regress)
    out = asyncio.run(T.mtpe_postedit_offline(list(draft), _source(), "ja", "en"))
    assert [o.text for o in out] == [d.text for d in draft]   # draft kept
