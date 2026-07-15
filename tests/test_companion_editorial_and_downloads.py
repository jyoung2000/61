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
