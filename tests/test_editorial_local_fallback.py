"""Option A (cloud key-limit → local editorial fallback) + Option B1 (translate
the clip hook, not just the caption)."""
import asyncio

import pytest

from backend.config import settings

# pipeline pulls in PIL/torch (frame_extractor) — available in CI, not always in
# the lean unit sandbox. The chain tests below need only config; the refresh
# tests need pipeline, so skip those gracefully when it can't import.
try:
    import backend.services.pipeline as P
except Exception:  # pragma: no cover - env-dependent
    P = None

requires_pipeline = pytest.mark.skipif(
    P is None, reason="backend.services.pipeline needs PIL/torch (CI only)")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Option A: editorial_provider_chain appends a local Ollama fallback ──

def _cloud_defaults():
    settings.SELF_HOSTED_MODE = False
    settings.EDITORIAL_AI_SOURCE = "auto"
    settings.CLIP_ENGINE_SOURCE = "auto"
    settings.AI_FALLBACK_CHAIN = "openrouter,gemini,groq"
    settings.OLLAMA_HOST = "http://localhost:11434"
    settings.EDITORIAL_LOCAL_FALLBACK = True


def test_cloud_chain_gets_local_fallback_appended():
    _cloud_defaults()
    # The displayed/active chain stays cloud-only…
    assert settings.active_provider_chain == ["openrouter", "gemini", "groq"]
    # …but the orchestrator's execution chain ends with a local Ollama fallback.
    assert settings.editorial_provider_chain == ["openrouter", "gemini", "groq", "ollama"]


def test_offline_chain_stays_ollama_only_no_cloud():
    _cloud_defaults()
    settings.SELF_HOSTED_MODE = True  # Offline Mode
    # Offline must make NO cloud calls — exec chain is Ollama only.
    assert settings.editorial_provider_chain == ["ollama"]


def test_fallback_disabled_keeps_cloud_only():
    _cloud_defaults()
    settings.EDITORIAL_LOCAL_FALLBACK = False
    assert settings.editorial_provider_chain == ["openrouter", "gemini", "groq"]


def test_no_ollama_host_means_no_fallback():
    _cloud_defaults()
    settings.OLLAMA_HOST = ""
    assert settings.editorial_provider_chain == ["openrouter", "gemini", "groq"]


def test_existing_pinned_ollama_not_duplicated():
    _cloud_defaults()
    settings.AI_FALLBACK_CHAIN = "openrouter,ollama"
    settings.EDITORIAL_AI_SOURCE = "cloud"  # demotes pinned ollama to the end
    assert settings.editorial_provider_chain == ["openrouter", "ollama"]


# ── Option B1: the post-translation refresh translates the hook too ──

class _Job:
    def __init__(self, clips):
        self.clips = clips


@pytest.fixture
def _capture(monkeypatch):
    captured = {}


    async def _load(job_id):
        return _Job([])  # 0 clips → use the in-process fallback list

    async def _update(job_id, **kw):
        captured.update(kw)
        return True

    monkeypatch.setattr(P.database, "load_job", _load)
    monkeypatch.setattr(P.database, "update_job_status", _update)
    return captured


@requires_pipeline
def test_refresh_translates_hook_not_just_caption(_capture):
    translated = [
        {"start": 0.0, "end": 3.0, "text": "I won't get rained on,"},
        {"start": 3.0, "end": 6.0, "text": "so I can't hold back."},
    ]
    clip = {
        "id": 1, "start_time": 0.0, "end_time": 6.0,
        "title": "古いタイトル",
        "hook_text": "日本語フック",
        "suggested_caption": "日本語キャプション",
        "vlm_hook": "雨には打たれない",  # Japanese VLM hook
        "vlm_reason": "", "judge_title": "The Opening Song",
    }
    n = _run(P._refresh_clips_with_translation("job1", translated, fallback_clips=[clip]))
    assert n >= 1
    out = _capture["clips"][0]
    # Hook is the English translated first cue — NOT the Japanese vlm_hook.
    assert "rained on" in out["hook_text"]
    assert out["hook_text"] != clip["vlm_hook"]
    # Caption is English too, and the title keeps the English judge title.
    assert "rained on" in out["suggested_caption"]
    assert out["title"] == "The Opening Song"


@requires_pipeline
def test_refresh_keeps_clip_when_no_translation_overlap(_capture):
    # A clip with no overlapping translated speech is left untouched (not blanked).
    translated = [{"start": 0.0, "end": 3.0, "text": "Hello there."}]
    clip = {
        "id": 2, "start_time": 100.0, "end_time": 106.0,
        "title": "keep", "hook_text": "keep-hook", "suggested_caption": "keep-cap",
        "vlm_hook": "", "vlm_reason": "", "judge_title": "",
    }
    _run(P._refresh_clips_with_translation("job2", translated, fallback_clips=[clip]))
    out = _capture["clips"][0]
    assert out["hook_text"] == "keep-hook"
    assert out["suggested_caption"] == "keep-cap"
