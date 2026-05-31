"""The Editorial AI Fallback (judge fallback) spec must persist across a
no-cache container rebuild, exactly like the primary / editorial model picks.

Previously the fallback lived only in clipper_config.json + an async startup
restore; this test pins the fix that mirrors it into user_settings.json via
_PERSISTABLE_KEYS, so it rides the same _restore_user_settings() path the other
model dropdowns use (which runs synchronously at module import).
"""

import json
import os

import backend.routers.settings as s
from backend.config import settings


def _isolate_user_settings(tmp_path, monkeypatch):
    path = str(tmp_path / "user_settings.json")
    monkeypatch.setattr(s, "USER_SETTINGS_PATH", path)
    return path


def test_fallback_spec_in_persistable_keys():
    assert "EDITORIAL_AI_FALLBACK_SPEC" in s._PERSISTABLE_KEYS
    assert "EDITORIAL_AI_PRIMARY_SPEC" in s._PERSISTABLE_KEYS


def test_fallback_spec_persists_and_restores(tmp_path, monkeypatch):
    path = _isolate_user_settings(tmp_path, monkeypatch)
    spec = "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
    settings.EDITORIAL_AI_FALLBACK_SPEC = spec
    settings.EDITORIAL_AI_PRIMARY_SPEC = "openrouter:google/gemini-3.1-flash-lite"

    assert s._persist_user_settings() is True
    # It actually landed in the mount-backed file.
    data = json.loads(open(path).read())
    assert data["EDITORIAL_AI_FALLBACK_SPEC"] == spec

    # Simulate a fresh container: reset in-memory, restore from disk.
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    settings.EDITORIAL_AI_PRIMARY_SPEC = ""
    s._restore_user_settings()
    assert settings.EDITORIAL_AI_FALLBACK_SPEC == spec
    assert settings.EDITORIAL_AI_PRIMARY_SPEC == "openrouter:google/gemini-3.1-flash-lite"


def test_unrelated_save_preserves_persisted_fallback(tmp_path, monkeypatch):
    # _persist_user_settings now PRESERVES a real on-disk value when the
    # in-memory value is empty — so an unrelated settings save (fired while
    # the spec hasn't been restored into memory yet) can't silently wipe it.
    # This is the fix for the "fallback didn't save / vanished" reports.
    path = _isolate_user_settings(tmp_path, monkeypatch)
    settings.EDITORIAL_AI_FALLBACK_SPEC = "openrouter:some/model"
    s._persist_user_settings()

    # An unrelated save while the spec is empty in memory must NOT drop it.
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    s._persist_user_settings()
    data = json.loads(open(path).read())
    assert data.get("EDITORIAL_AI_FALLBACK_SPEC") == "openrouter:some/model"

    # And it restores on the next boot.
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    s._restore_user_settings()
    assert settings.EDITORIAL_AI_FALLBACK_SPEC == "openrouter:some/model"
