"""Subtitle polishing must run on the dedicated TRANSLATION model, not the
editorial model. The editorial model (qwen2.5:3b) is reserved for SEO +
summaries; the translation model (qwen3:4b) owns the subtitles end-to-end
(translate + polish).

Two layers are checked:
  1. transcript_polisher.correct_transcript forwards ``model_override`` all the
     way down to ``text_completion`` (the actual model selector).
  2. pipeline._resolve_polish_model_override returns the translation model
     (gated by SUBTITLE_POLISH_USES_TRANSLATION_MODEL), falling back to the
     editorial model when no separate translation model is configured.
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


_stub_provider_sdks()

from backend.models import TranscriptSegment  # noqa: E402
from backend.services import transcript_polisher as P  # noqa: E402

XLATE = "qwen3:4b-instruct-2507-q4_K_M"
EDIT = "qwen2.5:3b-instruct"


def _seg(start, end, text):
    return TranscriptSegment(start=start, end=end, text=text, speaker="Speaker 1")


class _CapturingOrch:
    """Records the model_override passed to every text_completion call and
    returns a correctly-sized JSON array so the polish loop accepts it."""

    def __init__(self):
        self.overrides = []

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self.overrides.append(model_override)
        m = re.search(r"return EXACTLY (\d+)", prompt)
        n = int(m.group(1)) if m else 1
        return json.dumps([f"Polished line {i + 1}." for i in range(n)])


def _raw():
    return [
        _seg(0.0, 1.5, "um so he went to the store"),
        _seg(1.5, 3.0, "and uh she bought a book"),
    ]


def _enable(monkeypatch):
    monkeypatch.setattr(P.settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(P.settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(P.settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)


def test_correct_transcript_forwards_translation_model(monkeypatch):
    _enable(monkeypatch)
    orch = _CapturingOrch()
    asyncio.run(P.correct_transcript(
        _raw(), orch, language="en", model_override=XLATE))
    assert orch.overrides, "text_completion was never called"
    # Every polish batch ran on the translation model.
    assert all(o == XLATE for o in orch.overrides)


def test_correct_transcript_defaults_to_editorial_model(monkeypatch):
    _enable(monkeypatch)
    orch = _CapturingOrch()
    asyncio.run(P.correct_transcript(_raw(), orch, language="en"))
    # No override → editorial model (None tells the orchestrator to keep it).
    assert orch.overrides and all(o is None for o in orch.overrides)


# ── pipeline resolver ────────────────────────────────────────────────────────

class _OrchInfo:
    def __init__(self, provider, model):
        self._info = {"provider": provider, "model": model}

    def get_editorial_model_info(self):
        return self._info


def test_resolver_picks_translation_model(monkeypatch):
    from backend.services import pipeline as PL
    monkeypatch.setattr(PL.settings, "SUBTITLE_POLISH_USES_TRANSLATION_MODEL", True, raising=False)
    monkeypatch.setattr(PL.settings, "OLLAMA_TRANSLATION_MODEL", XLATE, raising=False)
    got = PL._resolve_polish_model_override(_OrchInfo("ollama", EDIT))
    assert got == XLATE


def test_resolver_off_uses_editorial(monkeypatch):
    from backend.services import pipeline as PL
    monkeypatch.setattr(PL.settings, "SUBTITLE_POLISH_USES_TRANSLATION_MODEL", False, raising=False)
    monkeypatch.setattr(PL.settings, "OLLAMA_TRANSLATION_MODEL", XLATE, raising=False)
    assert PL._resolve_polish_model_override(_OrchInfo("ollama", EDIT)) is None


def test_resolver_none_when_translation_equals_editorial(monkeypatch):
    from backend.services import pipeline as PL
    monkeypatch.setattr(PL.settings, "SUBTITLE_POLISH_USES_TRANSLATION_MODEL", True, raising=False)
    monkeypatch.setattr(PL.settings, "OLLAMA_TRANSLATION_MODEL", EDIT, raising=False)
    # Same model → no override needed (polishing already runs on it).
    assert PL._resolve_polish_model_override(_OrchInfo("ollama", EDIT)) is None
