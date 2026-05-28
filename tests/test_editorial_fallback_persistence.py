"""Regression tests for editorial-fallback (judge) model persistence.

The "Editorial AI Fallback" dropdown maps to the clipper judge fallback
spec. It must survive both a browser refresh and a full container restart.
Storage chain:

  PUT /clipper/judge-config
    -> writes judge_primary / judge_fallback to clipper_config.json
       (mount-backed /data/logs)
    -> mirrors a backup into user_settings.json under _judge_primary /
       _judge_fallback (also mount-backed)

  startup restore_judge_specs()
    -> rebuilds clipper_config.json from the user_settings.json backup if
       the config was reset by a rebuild / fresh volume

  GET /clipper/judge-config  (what the refresh reads back)

These tests pin: the save writes both files; a refresh reads the saved
value; an unrelated settings save doesn't drop the backup keys; and the
restore rebuilds the config after the canonical file is lost.
"""
import json
import os
import sys
import tempfile
import types

import pytest


@pytest.fixture()
def settings_env(monkeypatch):
    """Import the settings router with all on-disk paths redirected to a
    throwaway dir, and the heavy pipeline module stubbed to just the two
    path helpers the router imports lazily."""
    d = tempfile.mkdtemp()
    cfg_path = os.path.join(d, "clipper_config.json")
    us_path = os.path.join(d, "user_settings.json")

    pstub = types.ModuleType("backend.services.pipeline")
    pstub.clipper_config_path = lambda: cfg_path
    pstub._canonical_clipper_config_path = lambda: cfg_path
    monkeypatch.setitem(sys.modules, "backend.services.pipeline", pstub)

    import backend.routers.settings as S
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", us_path)
    return S, cfg_path, us_path


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


PRIMARY = "openrouter:google/gemini-3.1-flash-lite"
FALLBACK = "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"


def _restore_judge_specs(S, us_path):
    """Replica of backend.main.restore_judge_specs (avoids importing the
    full ASGI app, which pulls uvicorn). Kept faithful to the original."""
    if not os.path.exists(us_path):
        return
    user_data = json.load(open(us_path))
    jp = user_data.get("_judge_primary", "")
    jf = user_data.get("_judge_fallback", "")
    if not jp and not jf:
        return
    updates = {}
    if jp:
        updates["judge_primary"] = jp
    if jf:
        updates["judge_fallback"] = jf
    if updates:
        S._write_clipper_config_fields(updates)


def test_save_writes_config_and_backup(settings_env):
    S, cfg_path, us_path = settings_env
    out = _run(S.put_judge_config(
        S.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    assert out["fallback"] == FALLBACK
    cfg = json.load(open(cfg_path))
    assert cfg["judge_fallback"] == FALLBACK
    backup = json.load(open(us_path))
    assert backup["_judge_fallback"] == FALLBACK


def test_refresh_reads_back_fallback(settings_env):
    S, _, _ = settings_env
    _run(S.put_judge_config(S.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    got = _run(S.get_judge_config())
    assert got["fallback"] == FALLBACK
    assert got["primary"] == PRIMARY


def test_backup_survives_unrelated_settings_save(settings_env):
    S, _, us_path = settings_env
    _run(S.put_judge_config(S.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    # An unrelated settings persist must not drop the _judge_* backup keys.
    S._persist_user_settings()
    backup = json.load(open(us_path))
    assert backup.get("_judge_fallback") == FALLBACK
    assert backup.get("_judge_primary") == PRIMARY


def test_restart_restores_config_from_backup(settings_env):
    S, cfg_path, us_path = settings_env
    _run(S.put_judge_config(S.JudgeConfigRequest(primary=PRIMARY, fallback=FALLBACK)))
    # Simulate a rebuild that wiped the canonical config file.
    os.remove(cfg_path)
    _restore_judge_specs(S, us_path)
    assert os.path.exists(cfg_path)
    assert json.load(open(cfg_path))["judge_fallback"] == FALLBACK
    assert _run(S.get_judge_config())["fallback"] == FALLBACK
