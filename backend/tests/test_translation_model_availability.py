"""Task 5 — missing local translation model falls back to NMT (no crash, no auto-pull)."""

import asyncio
import sys
import types


def _stub_provider_sdks():
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

from backend.config import settings  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402
from backend.services import translator as T  # noqa: E402
from backend.services import local_models as LM  # noqa: E402
from backend.services import transcript_polisher as TP  # noqa: E402


def _segs():
    return [
        TranscriptSegment(start=0.0, end=1.0, text="Hola mundo", speaker="Speaker 1"),
        TranscriptSegment(start=1.0, end=2.0, text="Adiós", speaker="Speaker 1"),
    ]


def test_missing_model_falls_back_to_nmt_draft(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

    # Host has other models but NOT the configured translation model.
    async def _fake_list(timeout=4.0):
        return ["moondream:1.8b", "qwen2.5:3b-instruct"]
    monkeypatch.setattr(LM, "list_ollama_models", _fake_list)

    # correct_transcript must NOT be called when the model is missing.
    called = {"n": 0}

    async def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("correct_transcript should not run when model missing")
    monkeypatch.setattr(TP, "correct_transcript", _boom)

    draft = _segs()
    out = asyncio.run(T.mtpe_postedit_offline(
        draft, draft, "es", "en"))
    # Fell back to the raw NMT draft unchanged; no MTPE attempted.
    assert out == draft
    assert called["n"] == 0


def test_present_model_proceeds_to_mtpe(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

    async def _fake_list(timeout=4.0):
        # ":latest" suffix must still match the configured tag.
        return ["qwen3:4b-instruct-2507-q4_K_M:latest"]
    monkeypatch.setattr(LM, "list_ollama_models", _fake_list)

    proceeded = {"n": 0}

    async def _passthrough(segs, *a, **k):
        proceeded["n"] += 1
        return segs  # pretend MTPE returned the same cues
    monkeypatch.setattr(TP, "correct_transcript", _passthrough)

    draft = _segs()
    out = asyncio.run(T.mtpe_postedit_offline(draft, draft, "es", "en"))
    assert proceeded["n"] == 1
    assert len(out) == len(draft)


def test_list_failure_is_non_fatal(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434")
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M")

    async def _raise(timeout=4.0):
        raise RuntimeError("ollama unreachable")
    monkeypatch.setattr(LM, "list_ollama_models", _raise)

    proceeded = {"n": 0}

    async def _passthrough(segs, *a, **k):
        proceeded["n"] += 1
        return segs
    monkeypatch.setattr(TP, "correct_transcript", _passthrough)

    draft = _segs()
    out = asyncio.run(T.mtpe_postedit_offline(draft, draft, "es", "en"))
    # A transient list failure must not block translation — proceed to MTPE.
    assert proceeded["n"] == 1
    assert len(out) == len(draft)
