"""Tests for the Offline-Mode toggle (Settings → AI Provider).

The toggle drives the existing ``SELF_HOSTED_MODE`` machinery but speaks in
terms of the four user-facing stages it controls: clip detection,
transcription, translation and polishing. These tests pin the contract the
Settings UI relies on:

  * ``settings.resolve_stage_source(stage)`` maps each stage onto the right
    underlying engine — transcription is always local; clip detection follows
    the clip engine; translation + polishing follow the editorial engine.
  * ``GET/POST /api/self-hosted/settings`` report a ``stages`` block so the UI
    can show a Local/Cloud badge per stage, and flipping the master toggle on
    routes all four stages local with the editorial chain on Ollama.

The autouse ``_restore_settings`` fixture (tests/conftest.py) rolls the global
settings object back after each test, so these mutations don't leak.
"""
import asyncio

from backend.config import settings


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


STAGES = ("clip_detection", "transcription", "translation", "polishing")


def _set(self_hosted=False, clip="auto", editorial="auto"):
    settings.SELF_HOSTED_MODE = self_hosted
    settings.CLIP_ENGINE_SOURCE = clip
    settings.EDITORIAL_AI_SOURCE = editorial


def test_cloud_defaults_only_transcription_local():
    _set(self_hosted=False)
    resolved = {s: settings.resolve_stage_source(s) for s in STAGES}
    assert resolved == {
        "clip_detection": "cloud",
        "transcription": "local",   # Whisper is always local
        "translation": "cloud",
        "polishing": "cloud",
    }


def test_offline_mode_routes_every_stage_local():
    _set(self_hosted=True)
    for s in STAGES:
        assert settings.resolve_stage_source(s) == "local", s
    # Editorial LLM work (polish + LLM translation) must run on local Ollama.
    assert settings.active_provider_chain == ["ollama"]


def test_transcription_is_always_local_even_in_cloud_mode():
    _set(self_hosted=False, clip="cloud", editorial="cloud")
    assert settings.resolve_stage_source("transcription") == "local"


def test_per_engine_overrides_win_over_master():
    # Master off, but clip pinned local and editorial pinned cloud.
    _set(self_hosted=False, clip="local", editorial="cloud")
    assert settings.resolve_stage_source("clip_detection") == "local"
    assert settings.resolve_stage_source("translation") == "cloud"
    assert settings.resolve_stage_source("polishing") == "cloud"
    assert settings.resolve_stage_source("transcription") == "local"


def test_clip_alias_matches_clip_detection():
    _set(self_hosted=True)
    assert settings.resolve_stage_source("clip") == settings.resolve_stage_source("clip_detection")


def test_state_endpoint_reports_all_four_stages():
    import backend.routers.settings as S
    _set(self_hosted=False)
    state = S._self_hosted_state()
    assert set(state["stages"]) == set(STAGES)
    assert state["stages"]["transcription"] == "local"
    assert state["stages"]["clip_detection"] == "cloud"


def test_post_toggle_on_makes_all_stages_local():
    import backend.routers.settings as S
    _set(self_hosted=False)
    # The UI sends master on + overrides reset to auto so everything follows it.
    out = _run(S.save_self_hosted_settings(S.SaveSelfHostedRequest(
        self_hosted_mode=True, clip_engine_source="auto", editorial_ai_source="auto",
    )))
    assert out["self_hosted_mode"] is True
    assert all(out["stages"][s] == "local" for s in STAGES), out["stages"]


def test_post_toggle_off_returns_stages_to_cloud():
    import backend.routers.settings as S
    _set(self_hosted=True)
    out = _run(S.save_self_hosted_settings(S.SaveSelfHostedRequest(
        self_hosted_mode=False, clip_engine_source="auto", editorial_ai_source="auto",
    )))
    assert out["self_hosted_mode"] is False
    assert out["stages"]["clip_detection"] == "cloud"
    assert out["stages"]["translation"] == "cloud"
    assert out["stages"]["polishing"] == "cloud"
    assert out["stages"]["transcription"] == "local"
