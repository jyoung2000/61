"""VRAM-aware editorial auto-selection: a ≤4 GB card must NOT auto-pick a 4B
editorial model (it OOMs to CPU and stalls per-clip SEO/summaries)."""

import asyncio

from backend.config import settings
from backend.services import local_models as LM


def test_small_gpu_excludes_4b_editorial(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 5.5)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_MAX_PARAMS_B", 3.0)
    # Clear the configured override so the assertion tests the VRAM cap itself.
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "")
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.7)  # GTX 1650

    installed = ["qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b-instruct", "llava:7b"]
    got = asyncio.run(LM.select_local_editorial_models(limit=2, model_names=installed))
    # 4B excluded on the small card → the 3B instruct wins.
    assert got[0] == "qwen2.5:3b-instruct"
    assert "qwen3:4b-instruct-2507-q4_K_M" not in got


def test_big_gpu_keeps_4b_editorial(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 5.5)
    # Clear the configured override so auto-rank (not the override) decides.
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "")
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 8.0)  # roomy card

    installed = ["qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b-instruct"]
    got = asyncio.run(LM.select_local_editorial_models(limit=2, model_names=installed))
    assert got[0] == "qwen3:4b-instruct-2507-q4_K_M"


def test_unknown_vram_keeps_configured_cap(monkeypatch):
    # No CUDA / headless → don't downscope (0.0 means "unknown").
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0)
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 0.0)
    assert LM._effective_editorial_max_params_b() == 4.0


def test_floor_zero_disables_downscope(monkeypatch):
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0)
    monkeypatch.setattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 0.0)
    monkeypatch.setattr(LM, "_total_vram_gb", lambda: 3.7)
    assert LM._effective_editorial_max_params_b() == 4.0


def test_rank_respects_explicit_cap_unchanged():
    # The pure ranker with an explicit cap is unchanged (used elsewhere/tests).
    ranked = LM.rank_local_editorial_models(
        ["qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b-instruct"], max_params_b=4.0)
    assert ranked[0] == "qwen3:4b-instruct-2507-q4_K_M"


def test_qwen35_outranks_qwen3_in_same_tier():
    # Task 6: a qwen3.5 refresh beats qwen3-2507 within the 4B tier.
    ranked = LM.rank_local_editorial_models(
        ["qwen3:4b-instruct-2507-q4_K_M", "qwen3.5:4b-instruct"], max_params_b=4.0)
    assert ranked[0] == "qwen3.5:4b-instruct"
