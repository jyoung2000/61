"""Opt-in local self-refinement pass (Task 3): default OFF = byte-identical, no
extra LLM call; ON = one post-edit pass over the model's own output, reverted if
it reintroduces the source language; cue count preserved 1:1."""

import asyncio
import json
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


_stub_provider_sdks()

from backend.config import settings  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402
from backend.services import translator as T  # noqa: E402


class _Orch:
    """Counts text_completion calls and returns a sized JSON array. The first
    call (translation) returns English; later calls (refine) echo `refine_out`."""

    def __init__(self, refine_out=None):
        self.calls = 0
        self.refine_out = refine_out

    def _get_active_chain(self):
        class _P:
            provider_name = "ollama"
        return [_P()]

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self.calls += 1
        # Count expected array length from either prompt style.
        import re
        m = re.search(r"JSON array of (\d+) objects", prompt) or re.search(r"return EXACTLY (\d+)", prompt) or re.search(r"(\d+) numbered", prompt)
        n = int(m.group(1)) if m else 1
        if self.calls == 1:
            return json.dumps([f"line {i+1}" for i in range(n)])
        return json.dumps((self.refine_out or [f"line {i+1}" for i in range(n)])[:n])


def _src():
    return [
        TranscriptSegment(start=0.0, end=1.0, text="ソース1", speaker="S1"),
        TranscriptSegment(start=1.0, end=2.0, text="ソース2", speaker="S1"),
    ]


def _run(orch, monkeypatch, **flags):
    for k, v in flags.items():
        monkeypatch.setattr(settings, k, v, raising=False)
    monkeypatch.setattr(settings, "TRANSLATION_AUTO_GLOSSARY", False, raising=False)
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)
    return asyncio.run(T.translate_via_llm(_src(), "ja", "en", orch))


def test_refine_off_no_extra_call(monkeypatch):
    orch = _Orch()
    out = _run(orch, monkeypatch, TRANSLATION_LLM_REFINE_PASS=False)
    assert orch.calls == 1                     # translation only — no refine call
    assert [s.text for s in out] == ["line 1", "line 2"]


def test_refine_on_runs_once_and_applies(monkeypatch):
    orch = _Orch(refine_out=["polished one", "polished two"])
    out = _run(orch, monkeypatch, TRANSLATION_LLM_REFINE_PASS=True)
    assert orch.calls >= 2                      # translation + at least one refine
    assert [s.text for s in out] == ["polished one", "polished two"]
    # Timing preserved, cue count 1:1.
    assert len(out) == 2
    assert out[0].start == 0.0 and out[1].end == 2.0


def test_refine_reverts_on_source_language_regression(monkeypatch):
    # Refine output puts text back in the source script → must be rejected.
    orch = _Orch(refine_out=["ソースに戻る", "また日本語"])
    out = _run(orch, monkeypatch, TRANSLATION_LLM_REFINE_PASS=True)
    # Kept the clean pre-refine English translation, not the regressed refine.
    assert [s.text for s in out] == ["line 1", "line 2"]
