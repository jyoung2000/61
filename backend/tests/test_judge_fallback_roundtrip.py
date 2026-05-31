"""End-to-end persistence of the Editorial AI Fallback dropdown.

Exercises the REAL endpoints — put_judge_config → (simulated restart:
_restore_user_settings) → get_judge_config — proving the fallback spec is
saved from the first save and survives a no-cache container rebuild (where
clipper_config.json on an unmounted path is lost but user_settings.json on the
/data/logs mount persists).

The settings router lazily pulls the heavy provider/cv2/torch chain, so we
stub ONLY genuinely-missing modules (never shadowing installed packages →
no sibling-test pollution).
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

import pytest


class _AnyModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        return type(name, (), {})


def _stub_if_missing(name):
    if name in sys.modules:
        return
    base = name.split(".")[0]
    try:
        if importlib.util.find_spec(base) is not None:
            return
    except Exception:
        pass
    sys.modules[name] = _AnyModule(name)


for _n in ["google", "google.generativeai", "groq", "replicate", "cv2", "torch",
           "faster_whisper", "librosa", "soundfile", "sentencepiece",
           "ctranslate2", "transformers"]:
    _stub_if_missing(_n)
if isinstance(sys.modules.get("google"), _AnyModule):
    sys.modules["google"].generativeai = sys.modules.get(
        "google.generativeai", _AnyModule("google.generativeai"))

import backend.routers.settings as s  # noqa: E402
from backend.config import settings  # noqa: E402


def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point user_settings.json + clipper_config.json at temp files and reset
    the in-memory specs so each test starts clean."""
    us = str(tmp_path / "user_settings.json")
    cc = str(tmp_path / "clipper_config.json")
    monkeypatch.setattr(s, "USER_SETTINGS_PATH", us)
    monkeypatch.setattr(s, "_clipper_config_path", lambda: cc)
    # _write_clipper_config_fields writes to pipeline._canonical_clipper_config_path
    import backend.services.pipeline as pl
    monkeypatch.setattr(pl, "_canonical_clipper_config_path", lambda: cc)
    settings.EDITORIAL_AI_PRIMARY_SPEC = ""
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    return {"us": us, "cc": cc}


FALLBACK = "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
PRIMARY = "openrouter:google/gemini-3.1-flash-lite"


def test_fallback_saved_on_first_put_and_in_user_settings(isolated):
    _aiorun(s.put_judge_config(s.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    # Persisted to the mount-backed user_settings.json under the persistable key.
    data = json.loads(open(isolated["us"]).read())
    assert data["EDITORIAL_AI_FALLBACK_SPEC"] == FALLBACK
    assert data["EDITORIAL_AI_PRIMARY_SPEC"] == PRIMARY
    # GET returns it immediately.
    got = _aiorun(s.get_judge_config())
    assert got["fallback"] == FALLBACK
    assert got["primary"] == PRIMARY


def test_fallback_survives_no_cache_rebuild(isolated):
    # Save.
    _aiorun(s.put_judge_config(s.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    # Simulate a no-cache rebuild: clipper_config.json (unmounted path) is gone,
    # in-memory specs reset to defaults — only user_settings.json (mount) remains.
    os.remove(isolated["cc"])
    settings.EDITORIAL_AI_PRIMARY_SPEC = ""
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    # Startup restore runs at import; call it explicitly to mimic boot.
    s._restore_user_settings()
    assert settings.EDITORIAL_AI_FALLBACK_SPEC == FALLBACK
    got = _aiorun(s.get_judge_config())
    assert got["fallback"] == FALLBACK


def test_unrelated_save_does_not_wipe_fallback(isolated):
    # Save the fallback, then simulate an UNRELATED settings save while the
    # in-memory spec is empty (the latent bug: data={} dropped it).
    _aiorun(s.put_judge_config(s.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""   # e.g. before restore populated it
    s._persist_user_settings()                  # unrelated save
    data = json.loads(open(isolated["us"]).read())
    assert data.get("EDITORIAL_AI_FALLBACK_SPEC") == FALLBACK   # preserved, not wiped


def test_explicit_clear_removes_fallback(isolated):
    _aiorun(s.put_judge_config(s.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    # User clears the fallback (empty string).
    _aiorun(s.put_judge_config(s.JudgeConfigRequest(fallback="")))
    data = json.loads(open(isolated["us"]).read())
    assert "EDITORIAL_AI_FALLBACK_SPEC" not in data   # actually removed
    # And it stays cleared across a restart.
    settings.EDITORIAL_AI_FALLBACK_SPEC = ""
    s._restore_user_settings()
    assert settings.EDITORIAL_AI_FALLBACK_SPEC == ""
    # Primary is untouched by a fallback-only clear.
    assert settings.EDITORIAL_AI_PRIMARY_SPEC == PRIMARY
