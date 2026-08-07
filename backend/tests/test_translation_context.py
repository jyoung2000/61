"""Episode context brief (translation item 6) — builder guards + cache."""
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
    for name, attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                       ("anthropic", "AsyncAnthropic")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, attr, object)
            sys.modules[name] = mod


_stub_provider_sdks()

from backend.services import translator as T  # noqa: E402


class _FakeOrch:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def text_completion(self, prompt, **kw):
        self.calls += 1
        return self.reply


def _segs(n=20):
    return [{"text": f"これは第{i}話のセリフです。登場人物が会話します。"}
            for i in range(n)]


def test_brief_built_cached_and_readable_by_job():
    T._EPISODE_BRIEFS.clear()
    orch = _FakeOrch(
        "Relena returns to Earth with her father, the Vice Foreign Minister. "
        "A mysterious boy named Heero crashes nearby and enrolls at her "
        "school. Officers Zechs and Treize discuss the Gundam threat in a "
        "formal military register. The tone is serious wartime drama.")
    brief = asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", orch, "job-1", None))
    assert brief and "Relena" in brief
    assert T.episode_brief_for_job("job-1") == brief
    assert T.episode_brief_for_job("other-job") == ""


def test_brief_rejects_fragments_and_leaked_structure():
    T._EPISODE_BRIEFS.clear()
    # A JSON-shaped or fragment reply must be rejected, never injected into
    # every batch prompt as noise.
    assert asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", _FakeOrch('{"summary": "..."}'), "j2", None)) == ""
    assert asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", _FakeOrch("Okay."), "j3", None)) == ""
    assert T.episode_brief_for_job("j2") == ""


def test_brief_skips_short_transcripts_without_an_llm_call():
    T._EPISODE_BRIEFS.clear()
    orch = _FakeOrch("irrelevant")
    out = asyncio.run(T._build_episode_brief(
        _segs(5), "Japanese", orch, "j4", None))
    assert out == "" and orch.calls == 0
