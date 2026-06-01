"""Regression tests: every Settings model picker survives a process restart.

The Settings page populates each dropdown on load from the ``current`` block of
``GET /api/providers/models/available`` (plus ``GET /api/clipper/judge-config``
for the editorial fallback). A pick "doesn't persist" whenever it is saved to
disk correctly but the read-back endpoint resolves it to the wrong value — the
dropdown then reverts to blank on every reload even though the value is on disk.

That is exactly the bug these tests lock down for the Translation AI picker: an
Ollama translation pick is persisted to ``OLLAMA_TRANSLATION_MODEL`` but the
read-back used to be hardcoded to ``OPENROUTER_TRANSLATION_MODEL`` and returned
``""``. ``_current_translation_model()`` now resolves it provider-aware, exactly
like ``current_vision`` / ``current_text`` already did.

Each test simulates a fresh process restart: it writes ``user_settings.json``
(the mount-backed file restored at import) to a temp dir, builds a clean
``Settings`` object the way a new process would, re-runs the import-time restore
(``_restore_user_settings``), and then reads the value back through the real
endpoint/helper. The five pickers covered: Whisper, Primary AI, Editorial AI,
Editorial AI Fallback, and Translation AI — for both OpenRouter and Ollama picks
where applicable.

Style mirrors tests/test_editorial_fallback_persistence.py (same fixture shape,
same ``backend.services.pipeline`` stub, same ``_run`` helper).
"""
import json
import os
import sys
import tempfile
import types

import pytest


# ── Hermetic httpx stub ──────────────────────────────────────────────────
# available_models() probes Ollama's /api/tags when "ollama" is in the chain.
# Replace httpx with a stub so the read-back never touches the network; the
# ``current`` block is computed purely from settings regardless of the probe.
class _FakeResp:
    status_code = 200

    def json(self):
        return {"models": []}


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **k):
        return _FakeResp()


class _FakeHttpx:
    AsyncClient = _FakeClient


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


# Editorial-judge specs (stored "<backend>:<model>") for the fallback case.
JUDGE_PRIMARY = "openrouter:google/gemini-3.1-flash-lite"
JUDGE_FALLBACK = "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"


@pytest.fixture()
def env(monkeypatch):
    """Import the settings router with every on-disk path redirected to a
    throwaway dir and the network stubbed, then expose helpers that simulate a
    fresh-process restart and read each picker back."""
    d = tempfile.mkdtemp()
    cfg_path = os.path.join(d, "clipper_config.json")
    us_path = os.path.join(d, "user_settings.json")

    pstub = types.ModuleType("backend.services.pipeline")
    pstub.clipper_config_path = lambda: cfg_path
    pstub._canonical_clipper_config_path = lambda: cfg_path
    monkeypatch.setitem(sys.modules, "backend.services.pipeline", pstub)

    import backend.routers.settings as S
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", us_path)

    # Keep the read-back endpoint hermetic: no OpenRouter fetch, no Ollama probe.
    async def _no_openrouter():
        return None
    monkeypatch.setattr(S, "_fetch_openrouter_models", _no_openrouter)
    monkeypatch.setattr(S, "httpx", _FakeHttpx())

    def fresh_restart(persisted=None):
        """Simulate a new process: optionally write user_settings.json, build a
        clean Settings (the defaults a fresh container boots with), swap it in,
        and re-run the import-time restore that overlays the persisted file."""
        if persisted is not None:
            with open(us_path, "w") as f:
                json.dump(persisted, f)
        fresh = S.Settings(_env_file=None)  # ignore any ambient .env / env vars
        # Pin provider-source knobs so active_provider_chain == AI_FALLBACK_CHAIN
        # (a fresh boot's defaults — not "self-hosted/local" which would force
        # the chain to ["ollama"] and mask the read-back logic under test).
        fresh.SELF_HOSTED_MODE = False
        fresh.EDITORIAL_AI_SOURCE = "auto"
        fresh.CLIP_ENGINE_SOURCE = "auto"
        monkeypatch.setattr(S, "settings", fresh)
        S._restore_user_settings()
        return fresh

    def restart_current(persisted):
        """Persist → restart → return the ``current`` block the dropdowns read."""
        fresh_restart(persisted)
        return _run(S.available_models())["current"]

    def restore_judge_specs():
        """Replica of backend.main.restore_judge_specs (avoids importing the
        full ASGI app). Rebuilds clipper_config.json from the user_settings.json
        ``_judge_*`` backup, covering a rebuild that wiped the canonical file."""
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

    return types.SimpleNamespace(
        S=S, cfg_path=cfg_path, us_path=us_path,
        fresh_restart=fresh_restart, restart_current=restart_current,
        restore_judge_specs=restore_judge_specs,
    )


# ── 1. Whisper ────────────────────────────────────────────────────────────
def test_whisper_model_survives_restart(env):
    cur = env.restart_current({
        "WHISPER_MODEL": "large-v3-turbo",
        "WHISPER_MODEL_USER_SET": True,
    })
    assert cur["transcript_model"] == "large-v3-turbo"
    assert cur["whisper_model_selected"] == "large-v3-turbo"


# ── 2. Primary AI — OpenRouter ─────────────────────────────────────────────
def test_primary_openrouter_survives_restart(env):
    cur = env.restart_current({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "OPENROUTER_PRIMARY_MODEL": "anthropic/claude-3.5-sonnet",
        "AI_FALLBACK_CHAIN": "openrouter,gemini",
    })
    assert cur["vision_model"] == "anthropic/claude-3.5-sonnet"


# ── 3. Primary AI — Ollama (reads back as ollama/<model>) ──────────────────
def test_primary_ollama_survives_restart(env):
    cur = env.restart_current({
        "OLLAMA_PRIMARY_MODEL": "moondream:1.8b",
        "OLLAMA_EDITORIAL_MODEL": "qwen2.5:3b-instruct",
        "AI_FALLBACK_CHAIN": "ollama,openrouter",
    })
    assert cur["vision_model"] == "ollama/moondream:1.8b"


# ── 4. Editorial AI — OpenRouter + Ollama ──────────────────────────────────
def test_editorial_openrouter_survives_restart(env):
    cur = env.restart_current({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "OPENROUTER_EDITORIAL_MODEL": "google/gemini-2.5-pro",
        "AI_FALLBACK_CHAIN": "openrouter,gemini",
    })
    assert cur["text_model"] == "google/gemini-2.5-pro"


def test_editorial_ollama_survives_restart(env):
    cur = env.restart_current({
        "OLLAMA_PRIMARY_MODEL": "moondream:1.8b",
        "OLLAMA_EDITORIAL_MODEL": "qwen2.5:7b-instruct",
        "AI_FALLBACK_CHAIN": "ollama",
    })
    assert cur["text_model"] == "ollama/qwen2.5:7b-instruct"


# ── 5. Editorial AI Fallback (judge spec) ──────────────────────────────────
def test_editorial_fallback_spec_survives_restart(env):
    S = env.S
    # Save via the real endpoint — writes clipper_config.json AND the
    # user_settings.json backup (_judge_* + EDITORIAL_AI_FALLBACK_SPEC).
    _run(S.put_judge_config(
        S.JudgeConfigRequest(primary=JUDGE_PRIMARY, fallback=JUDGE_FALLBACK)))
    # Simulate a no-cache rebuild that wiped the canonical clipper_config.json.
    if os.path.exists(env.cfg_path):
        os.remove(env.cfg_path)
    # Fresh process: clean Settings + import-time restore from user_settings.json
    # (restores EDITORIAL_AI_FALLBACK_SPEC into settings) ...
    env.fresh_restart()
    # ... and the startup handler rebuilds clipper_config.json from the backup.
    env.restore_judge_specs()
    got = _run(S.get_judge_config())
    assert got["fallback"] == JUDGE_FALLBACK
    assert got["primary"] == JUDGE_PRIMARY


# ── 6. Translation AI — OpenRouter (was never broken; locked in) ───────────
def test_translation_openrouter_survives_restart(env):
    cur = env.restart_current({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "OPENROUTER_TRANSLATION_MODEL": "openai/gpt-4o-mini",
        "AI_FALLBACK_CHAIN": "openrouter,gemini",
    })
    assert cur["translation_model"] == "openai/gpt-4o-mini"


# ── 7. Translation AI — Ollama (THE regression: fails before the fix) ──────
def test_translation_ollama_survives_restart(env):
    # A distinctive, non-default Ollama model so this also proves the *persisted*
    # value is restored (not the OLLAMA_TRANSLATION_MODEL=qwen2.5:3b default).
    cur = env.restart_current({
        "OLLAMA_TRANSLATION_MODEL": "qwen2.5:7b",
        "AI_FALLBACK_CHAIN": "ollama,openrouter",
    })
    # Before the fix this read back "" (hardcoded to OPENROUTER_TRANSLATION_MODEL)
    # and the dropdown reverted to blank on every reload.
    assert cur["translation_model"] == "ollama/qwen2.5:7b"


# ── Constraint guard: blank Translation AI == reuse Editorial (OpenRouter) ─
def test_translation_blank_openrouter_does_not_leak_ollama_default(env):
    # OpenRouter-only user with NO translation pick. The OLLAMA_TRANSLATION_MODEL
    # default (qwen2.5:3b) lives on the fresh Settings but must NOT surface here —
    # blank means "reuse the Editorial AI model".
    cur = env.restart_current({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "OPENROUTER_PRIMARY_MODEL": "google/gemini-2.5-flash",
        "OPENROUTER_EDITORIAL_MODEL": "google/gemini-2.5-pro",
        "AI_FALLBACK_CHAIN": "openrouter,gemini",
    })
    assert cur["translation_model"] == ""


# ── Direct unit coverage of the read-back helper (all three call sites use it) ─
def test_current_translation_model_helper_matrix(env):
    S = env.S

    # OpenRouter path, explicit pick -> the pick.
    env.fresh_restart({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "OPENROUTER_TRANSLATION_MODEL": "openai/gpt-4o-mini",
        "AI_FALLBACK_CHAIN": "openrouter",
    })
    assert S._current_translation_model() == "openai/gpt-4o-mini"

    # OpenRouter path, no pick -> blank (Ollama default suppressed).
    env.fresh_restart({
        "OPENROUTER_API_KEY": "sk-or-test-123",
        "AI_FALLBACK_CHAIN": "openrouter",
    })
    assert S._current_translation_model() == ""

    # Ollama path, dedicated pick -> ollama/<model>.
    env.fresh_restart({
        "OLLAMA_TRANSLATION_MODEL": "aya:8b",
        "AI_FALLBACK_CHAIN": "ollama,openrouter",
    })
    assert S._current_translation_model() == "ollama/aya:8b"
