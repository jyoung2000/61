"""Companion-paired fixes (2026-07-15).

(1) The editorial-model param cap reads the LOCAL GPU (a 4 GB 1650), so it
dropped an explicitly-picked 12B editorial model from the ranking and silently
fell back to a 3B — even though editorial runs on the paired Companion 4070.
(2) The Companion's Update button 401'd because the installer manifest/download
endpoints sit behind the cookie-only AuthMiddleware, which a non-browser LAN peer
can't satisfy.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))


# ── Fix 1: editorial cap lifts when a Companion is paired ───────────────────

def test_editorial_cap_lifted_when_companion_paired(monkeypatch):
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG
    monkeypatch.setattr(REG, "companion_host", lambda: object())   # paired 4070
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.6)         # local 1650
    assert LM._effective_editorial_max_params_b() == float("inf")
    # …so an explicitly-picked 12B survives the ranking instead of being dropped.
    ranked = LM.rank_local_editorial_models(["qwen2.5:3b", "gemma3:12b-it-q4_K_M"])
    assert any("gemma3:12b" in n for n in ranked)


def test_editorial_cap_applies_locally_without_companion(monkeypatch):
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG
    monkeypatch.setattr(REG, "companion_host", lambda: None)       # local-only
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.6)
    assert LM._effective_editorial_max_params_b() == 3.0           # small-GPU downshift
    ranked = LM.rank_local_editorial_models(["qwen2.5:3b", "gemma3:12b-it-q4_K_M"])
    assert all("gemma3:12b" not in n for n in ranked)              # 12B correctly dropped


def test_editorial_cap_unknown_vram_keeps_configured(monkeypatch):
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG
    monkeypatch.setattr(REG, "companion_host", lambda: None)
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 0.0)         # CPU / headless
    assert LM._effective_editorial_max_params_b() == 4.0


def test_editorial_cap_failsoft_on_registry_error(monkeypatch):
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG

    def _boom():
        raise RuntimeError("registry not ready")
    monkeypatch.setattr(REG, "companion_host", _boom)
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.6)
    # Never raises → falls through to the local cap (previous behavior).
    assert LM._effective_editorial_max_params_b() == 3.0


# ── Fix 1b: an EXPLICIT editorial pick wins over the auto-selection cap ──────

def test_explicit_editorial_pick_honored_above_cap(monkeypatch):
    import asyncio
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG
    from backend.config import settings
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0, raising=False)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 5.5, raising=False)
    monkeypatch.setattr(REG, "companion_host", lambda: None)   # isolate: cap applies
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.7)     # small local card
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL",
                        "gemma3:12b-it-q4_K_M", raising=False)
    installed = ["gemma3:12b-it-q4_K_M", "qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b"]
    got = asyncio.run(LM.select_local_editorial_models(limit=2, model_names=installed))
    assert got[0] == "gemma3:12b-it-q4_K_M"      # explicit 12B pick wins despite the cap


def test_auto_editorial_still_capped_without_explicit_pick(monkeypatch):
    import asyncio
    import backend.services.local_models as LM
    import backend.services.ollama_registry as REG
    from backend.config import settings
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0, raising=False)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 5.5, raising=False)
    monkeypatch.setattr(REG, "companion_host", lambda: None)
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.7)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "", raising=False)  # auto
    installed = ["gemma3:12b-it-q4_K_M", "qwen2.5:3b-instruct"]
    got = asyncio.run(LM.select_local_editorial_models(limit=2, model_names=installed))
    assert "gemma3:12b-it-q4_K_M" not in got     # auto-select still respects the cap


# ── Fix 1c: the ACTIVE-MODELS banner uses the pick-honoring selector ────────

def test_active_models_display_uses_pick_honoring_selector():
    """The provider_status banner must resolve editorial via
    select_local_editorial_models (which honors an explicit OLLAMA_EDITORIAL_MODEL
    over the local-card cap), NOT the raw rank_local_editorial_models — otherwise
    the banner shows a 3B/4B while the pipeline actually runs the picked 12B
    (the "still doesnt show gemma 3" symptom)."""
    import inspect
    import backend.routers.settings as S
    src = inspect.getsource(S.provider_status)
    # The editorial-display block resolves through the pick-honoring selector…
    assert "select_local_editorial_models" in src
    # …and does NOT reach for the raw ranker there (which ignores the pick).
    assert "rank_local_editorial_models" not in src


# ── Fix 2: Companion installer endpoints are public (Update button 401) ─────

def test_companion_installer_reads_are_public():
    from backend.app.auth.middleware import _is_public_path
    assert _is_public_path("/api/downloads/companion/manifest") is True
    assert _is_public_path("/api/downloads/companion/windows") is True
    assert _is_public_path("/api/downloads/companion/windows_msi") is True
    assert _is_public_path("/api/downloads/companion/mac") is True


def test_companion_refresh_stays_authenticated():
    from backend.app.auth.middleware import _is_public_path
    # The admin "check for updates" refresh must NOT be public.
    assert _is_public_path("/api/downloads/companion/refresh") is False
    # A random API path is not public either.
    assert _is_public_path("/api/jobs") is False
