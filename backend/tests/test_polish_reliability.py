"""Polish reliability tests — the "polish must never silently die" change set.

Real-world failure this codifies (GTX 1650 run, 2026-07-03 log): Ollama-only
chain + partial-offload cold load blew the flat 90s polish timeout → circuit
breaker degraded Ollama → EVERY batch of BOTH jobs failed → raw unpolished
draft shipped (Zeks/Zecks/Zeck for Zechs, romaji lines in the English track).

Covers: cold-load timeout scaling, cloud fallback on local-chain failure,
cloud-direct routing for pinned OpenRouter models, the auto entity glossary,
and the romaji gate (detector + cleanup selection).
"""

import asyncio
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _name, _attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                     ("anthropic", "AsyncAnthropic")):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod
if "google.generativeai" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.generativeai")
    _gg.configure = lambda *a, **k: None
    _gg.GenerativeModel = object
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules["google.generativeai"] = _gg

import json

import pytest

from backend.config import settings  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402
from backend.services import transcript_polisher as P  # noqa: E402


def _seg(i, text):
    return TranscriptSegment(start=float(i * 3), end=float(i * 3 + 2.5),
                             text=text, speaker="Speaker 1")


class _FailingOrchestrator:
    """Local chain that always fails (the degraded-Ollama state)."""

    def __init__(self):
        self.calls = []

    async def text_completion(self, prompt, timeout=60, model_override=None,
                              **kw):
        self.calls.append({"timeout": timeout, "override": model_override})
        raise RuntimeError("All providers failed for text completion")


class _EchoOrchestrator:
    """Local chain that succeeds, echoing input lines back as JSON."""

    def __init__(self):
        self.calls = []

    async def text_completion(self, prompt, timeout=60, model_override=None,
                              **kw):
        self.calls.append({"timeout": timeout, "override": model_override})
        lines = []
        for raw in prompt.splitlines():
            raw = raw.strip()
            if raw.startswith('"') and raw.endswith('",') or (
                    raw.startswith('"') and raw.endswith('"')):
                continue
        # Parse the numbered input list the prompt carries
        import re
        items = re.findall(r'^\s*\d+\.\s+(.*)$', prompt, re.MULTILINE)
        return json.dumps(items or ["ok"])


# ── Cold-load timeout scaling ────────────────────────────────────────────

def test_first_batch_gets_scaled_timeout(monkeypatch):
    orch = _FailingOrchestrator()
    # No cloud fallback for this test — isolate the timeout behavior
    monkeypatch.setattr(P, "_cloud_polish_available", lambda: False)
    segs = [_seg(i, f"line number {i} spoken here") for i in range(40)]
    out = asyncio.run(P.correct_transcript(
        segs, orchestrator=orch, language="en", timeout_per_batch=90.0))
    assert len(out) == len(segs)  # fail-soft: originals kept
    assert orch.calls, "orchestrator was never called"
    # First batch: 3x scaled (capped 300); later batches: base timeout
    assert orch.calls[0]["timeout"] == pytest.approx(270.0)
    if len(orch.calls) > 1:
        assert orch.calls[1]["timeout"] == pytest.approx(90.0)


# ── Cloud fallback ───────────────────────────────────────────────────────

def test_cloud_fallback_rescues_failed_batch(monkeypatch):
    orch = _FailingOrchestrator()
    cloud_calls = []

    async def fake_cloud(prompt, timeout):
        cloud_calls.append(timeout)
        # Parse the JSON segment block after "SEGMENTS TO POLISH"
        block = prompt.split("SEGMENTS TO POLISH", 1)[1]
        start = block.index("[")
        items = json.JSONDecoder().raw_decode(block[start:])[0]
        texts = [it["text"] if isinstance(it, dict) else it for it in items]
        # Length-preserving edit (the polisher's word-budget guard rejects
        # rewrites outside ±15% — as it should)
        return json.dumps([t.replace("noisy", "POLISHED") for t in texts])

    monkeypatch.setattr(P, "_cloud_polish_completion", fake_cloud)
    segs = [_seg(i, f"noisy asr line {i} with mistakes") for i in range(5)]
    out = asyncio.run(P.correct_transcript(
        segs, orchestrator=orch, language="en", timeout_per_batch=90.0))
    assert cloud_calls, "cloud fallback never invoked"
    assert any("POLISHED" in s.text for s in out), \
        "cloud-polished text did not land in the output"


def test_cloud_direct_for_openrouter_pinned_model(monkeypatch):
    """An OpenRouter-style model_override must NOT be fed to the local
    chain — it routes straight to the cloud."""
    orch = _FailingOrchestrator()
    cloud_calls = []

    async def fake_cloud(prompt, timeout):
        cloud_calls.append(timeout)
        import re
        items = re.findall(r'^\s*\d+\.\s+(.*)$', prompt, re.MULTILINE)
        return json.dumps(["fixed"] * len(items))

    monkeypatch.setattr(P, "_cloud_polish_completion", fake_cloud)
    monkeypatch.setattr(P, "_cloud_polish_available", lambda: True)
    segs = [_seg(i, f"line {i} here now") for i in range(3)]
    asyncio.run(P.correct_transcript(
        segs, orchestrator=orch, language="en",
        model_override="anthropic/claude-haiku-4.5"))
    assert cloud_calls, "cloud-direct path not used"
    assert not orch.calls, \
        "local chain was called despite an OpenRouter-pinned polish model"


def test_cloud_model_resolution_order(monkeypatch):
    prev_pin = settings.SUBTITLE_POLISH_MODEL
    prev_cloud = settings.SUBTITLE_POLISH_CLOUD_MODEL
    try:
        settings.SUBTITLE_POLISH_MODEL = "deepseek/deepseek-chat-v3.1"
        assert P._resolve_cloud_polish_model() == "deepseek/deepseek-chat-v3.1"
        # Ollama-style pin is NOT usable on OpenRouter → falls through
        settings.SUBTITLE_POLISH_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
        settings.SUBTITLE_POLISH_CLOUD_MODEL = "google/gemini-2.5-flash"
        assert P._resolve_cloud_polish_model() == "google/gemini-2.5-flash"
        # Nothing set → first efficient shortlist entry
        settings.SUBTITLE_POLISH_MODEL = ""
        settings.SUBTITLE_POLISH_CLOUD_MODEL = ""
        from backend.services.providers.openrouter_provider import (
            SUBTITLE_POLISH_SHORTLIST)
        first_efficient = next(e["id"] for e in SUBTITLE_POLISH_SHORTLIST
                               if e["tier"] == "efficient")
        assert P._resolve_cloud_polish_model() == first_efficient
    finally:
        settings.SUBTITLE_POLISH_MODEL = prev_pin
        settings.SUBTITLE_POLISH_CLOUD_MODEL = prev_cloud


def test_cloud_fallback_respects_flag_and_key(monkeypatch):
    prev = settings.SUBTITLE_POLISH_CLOUD_FALLBACK
    prev_key = settings.OPENROUTER_API_KEY
    try:
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = False
        settings.OPENROUTER_API_KEY = "sk-or-xxx"
        assert P._cloud_polish_available() is False
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = True
        assert P._cloud_polish_available() is True
        settings.OPENROUTER_API_KEY = ""
        assert P._cloud_polish_available() is False
    finally:
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = prev
        settings.OPENROUTER_API_KEY = prev_key


# ── Auto entity glossary ─────────────────────────────────────────────────

def test_entity_glossary_clusters_name_variants():
    texts = [
        "Zechs piloted the white suit.",
        "I told Zechs about the plan.",
        "Understand, Zeck?",
        "Notify Zeks of the axial data.",
        "Then Zechs left the base.",
        "The colony fell yesterday.",
        "This is fine.",
    ]
    terms = P.derive_entity_glossary(texts, min_count=3)
    assert "Zechs" in terms          # most frequent variant wins
    assert "Zeks" not in terms       # clustered into Zechs
    assert "Zeck" not in terms
    assert "This" not in terms       # sentence-initial-only noise dropped
    assert "The" not in terms


def test_entity_glossary_ignores_nonrecurring():
    texts = ["Kenobi appears once.", "Nothing else here.", "More filler text."]
    assert "Kenobi" not in P.derive_entity_glossary(texts, min_count=3)


# ── Romaji gate ──────────────────────────────────────────────────────────

def test_romaji_lines_flagged_as_untranslated():
    from backend.services.translator import _is_untranslated
    shipped = [  # actual lines from the 2026-07-03 run's English track
        "Nametotte ageru kara.",
        "Zutto yuwara sakuda nee,",
        "Kawaii mou humeschiku naku natte kida dan hajikushika nakutte kida",
        "Irondo toko ni sakomarete mitakatta nda yo hontou?",
        # mixed romaji/English line — first half romaji
        "Zotten nan ka onegatō no ne puripuri shite nagaru ga kizuitaseru kara puru Repeating it all,",
    ]
    for line in shipped:
        assert _is_untranslated(line, "ja"), line


def test_english_lines_not_flagged():
    from backend.services.translator import _is_untranslated
    english = [
        "I want to see you no matter what happens today.",
        "We'll see you next time.",
        "It feels great, doesn't it?",
        "Tonight, I want to tell you from the sky.",
        "See how cute this one is, such a pa",
        "You can start from the edge—it's a bit unusual though, right?",
    ]
    for line in english:
        assert not _is_untranslated(line, "ja"), line
        assert not _is_untranslated(line, "auto"), line


def test_cleanup_selects_romaji_cues():
    """The per-cue LLM cleanup must select romaji leftovers, not just CJK."""
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline._llm_cleanup_untranslated)
    assert "_is_untranslated(_txt(s), source_lang)" in src
    assert "_cloud_polish_completion" in src


def test_polish_reliability_flag_defaults():
    assert settings.SUBTITLE_POLISH_CLOUD_FALLBACK is True
    assert settings.SUBTITLE_POLISH_CLOUD_MODEL == ""
    assert settings.SUBTITLE_POLISH_AUTO_GLOSSARY is True
