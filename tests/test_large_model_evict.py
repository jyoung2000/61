"""Auto-fit the large translation model on the Companion GPU (2026-07-15).

The measured 4070 run left a ~2 GB editorial 3B resident, so the 12B q4 loaded
partially on the CPU → 45-65 s/batch (~24 min). Before the first large-model
batch, translate_via_llm now evicts every other model from the Companion Ollama
so the 12B reloads into the full budget (GPU-resident). Small models are
untouched.
"""
import asyncio
import json
import re
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
    for name, attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                       ("anthropic", "AsyncAnthropic")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, attr, object)
            sys.modules[name] = mod


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_stub_provider_sdks()

from backend.config import settings  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402
from backend.services import translator as T  # noqa: E402


class _Orch:
    """Fake orchestrator that IS its own ollama provider and records
    clear_vram() calls."""

    def __init__(self):
        self.calls = 0
        self.cleared = []
        self._providers = {"ollama": self}

    async def clear_vram(self, except_model=None):
        self.cleared.append(except_model)

    def _get_active_chain(self):
        class _P:
            provider_name = "ollama"
            text_model_name = "gemma3:12b-it-q4_K_M"
        return [_P()]

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self.calls += 1
        m = (re.search(r"array of exactly (\d+)", prompt)
             or re.search(r"EVERY one of the (\d+)", prompt)
             or re.search(r"JSON array of (\d+)", prompt))
        n = int(m.group(1)) if m else 1
        return json.dumps([f"line {i + 1}" for i in range(n)])


def _src(n):
    return [TranscriptSegment(start=float(i), end=float(i + 1),
                              text=f"ソース{i + 1}", speaker="S1") for i in range(n)]


def _run(orch, monkeypatch, model_override):
    monkeypatch.setattr(settings, "TRANSLATION_AUTO_GLOSSARY", False, raising=False)
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "TRANSLATION_LLM_REFINE_PASS", False, raising=False)
    return asyncio.run(T.translate_via_llm(
        _src(6), "ja", "en", orch, model_override=model_override))


def test_large_model_evicts_companion_vram(monkeypatch):
    orch = _Orch()
    out = _run(orch, monkeypatch, "gemma3:12b-it-q4_K_M")
    assert out is not None and len(out) == 6
    assert orch.cleared == [None]        # evicted ALL (except_model=None), once


def test_small_model_does_not_evict(monkeypatch):
    orch = _Orch()
    _run(orch, monkeypatch, "qwen3:4b-instruct-2507")
    assert orch.cleared == []            # small model never evicts


def test_evict_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_LARGE_EVICT_OTHERS", False, raising=False)
    orch = _Orch()
    _run(orch, monkeypatch, "gemma3:12b-it-q4_K_M")
    assert orch.cleared == []            # flag off → no eviction
