"""Batch-0 split-and-retry recovery (2026-07-15).

The measured Gundam runs showed the gemma3:12b LLM translation completing but its
big first batch (20 lines) failing the strict count/parse gate — which used to
return None and silently drop the WHOLE job to the far weaker offline NMT
(FuguMT), producing the butchered names. Batch 0 now split-and-retries like every
other batch, with a usability sentinel so a genuinely dead model still bails.
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


def _expected(prompt: str) -> int:
    m = (re.search(r"array of exactly (\d+)", prompt)
         or re.search(r"EVERY one of the (\d+)", prompt)
         or re.search(r"JSON array of (\d+)", prompt)
         or re.search(r"return EXACTLY (\d+)", prompt))
    return int(m.group(1)) if m else 1


class _BigBatchMissOrch:
    """A WORKING model that fumbles only the big first array: returns a
    wrong-count reply for a batch larger than ``ok_at`` (→ _call None), but a
    correct-count reply once the batch is halved down to ``ok_at`` or fewer."""

    def __init__(self, ok_at=4):
        self.ok_at = ok_at
        self.calls = 0

    def _get_active_chain(self):
        class _P:
            provider_name = "ollama"
        return [_P()]

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self.calls += 1
        n = _expected(prompt)
        if n > self.ok_at:
            return json.dumps([f"line {i + 1}" for i in range(n - 1)])  # wrong count
        return json.dumps([f"line {i + 1}" for i in range(n)])          # good


class _DeadOrch:
    """A genuinely unusable model: every reply is a wrong-count array, at every
    granularity down to singletons."""

    def __init__(self):
        self.calls = 0

    def _get_active_chain(self):
        class _P:
            provider_name = "ollama"
        return [_P()]

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self.calls += 1
        n = _expected(prompt)
        return json.dumps([f"x{i}" for i in range(n + 1)])              # always wrong


def _src(n):
    return [TranscriptSegment(start=float(i), end=float(i + 1),
                              text=f"ソース{i + 1}", speaker="S1") for i in range(n)]


def _run(orch, monkeypatch):
    monkeypatch.setattr(settings, "TRANSLATION_AUTO_GLOSSARY", False, raising=False)
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "TRANSLATION_LLM_REFINE_PASS", False, raising=False)
    return asyncio.run(T.translate_via_llm(_src(8), "ja", "en", orch))


def test_batch0_recovers_via_split_and_retry(monkeypatch):
    # The first 8-line batch misses (returns 7) → recover by halving to 4-line
    # sub-calls that parse → the LLM path is KEPT (not None / not FuguMT).
    orch = _BigBatchMissOrch(ok_at=4)
    out = _run(orch, monkeypatch)
    assert out is not None                       # did NOT bail to offline NMT
    assert len(out) == 8                          # every cue translated 1:1
    assert all("ソース" not in s.text for s in out)   # nothing left in source
    assert orch.calls >= 3                        # 1 big miss + ≥2 half retries


def test_dead_model_still_bails_to_nmt(monkeypatch):
    # Every reply is wrong-count at every granularity → no sub-call ever
    # succeeds → return None so the caller falls back to offline NMT.
    orch = _DeadOrch()
    out = _run(orch, monkeypatch)
    assert out is None
