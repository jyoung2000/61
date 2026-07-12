"""Regression tests: changing the Whisper model must actually take effect.

Two compounding bugs made a large-v3-turbo → large-v3 change bounce back:

  * Companion: ``whisper_rank`` scored full large-v3 and turbo as the SAME
    tier, and the tier table only emitted turbo — an explicit large-v3 pick
    was silently coerced (fixed in companion/src-tauri/src/state.rs, covered
    by its Rust unit tests).
  * Container: ``_sync_companion_whisper`` mirrored the Companion's
    currently-loaded model back into ``settings.WHISPER_MODEL`` and
    persisted it — actively REVERTING the user's pin right after they saved
    it. Covered here.
"""

import asyncio
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings

# Import the router at MODULE scope: its import runs _restore_user_settings(),
# which overwrites whisper fields from any user_settings.json lying around
# (e.g. written by an earlier test in this process). Importing it inside a
# test AFTER pinning the settings let that restore clobber the pin — a test
# artifact, not product behavior (in production the router imports once at
# startup, before any user interaction).
from backend.routers import settings as sr  # noqa: E402


class _FakeResp:
    status_code = 200

    @staticmethod
    def json():
        return {
            "whisper_model_effective": "large-v3-turbo",
            "whisper_beam_size": 5,
            "whisper_quality": "auto",
        }


class _FakeClient:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        return _FakeResp()


def _run_sync(monkeypatch):
    from backend.services import ollama_registry as oreg

    monkeypatch.setattr(sr, "httpx", types.SimpleNamespace(AsyncClient=_FakeClient))
    monkeypatch.setattr(oreg, "companion_base", lambda comp: "http://comp.test:11500")
    monkeypatch.setattr(oreg, "auth_headers", lambda comp: {})
    monkeypatch.setattr(sr, "_persist_user_settings",
                        lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(sr, "_invalidate_status_cache",
                        lambda *a, **kw: None, raising=False)
    return asyncio.run(sr._sync_companion_whisper(object()))


def test_sync_never_overwrites_a_user_pinned_model(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_MODEL", "large-v3", raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL_USER_SET", True, raising=False)
    out = _run_sync(monkeypatch)
    assert out["model"] == "large-v3-turbo"          # companion state reported…
    assert settings.WHISPER_MODEL == "large-v3"      # …but the pin survives


def test_sync_still_mirrors_when_not_pinned(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_MODEL", "small", raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL_USER_SET", False, raising=False)
    _run_sync(monkeypatch)
    assert settings.WHISPER_MODEL == "large-v3-turbo"


def test_pick_model_honors_the_pin(monkeypatch):
    from backend.services import reframer_audio as ra
    monkeypatch.setattr(settings, "WHISPER_REMOTE_MODEL", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "large-v3", raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL_USER_SET", True, raising=False)
    assert ra.remote_whisper_pick_model("ja") == "large-v3"
    assert ra.remote_whisper_pick_model("en") == "large-v3"
