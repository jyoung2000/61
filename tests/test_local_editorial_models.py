"""Tests for Offline-Mode local editorial-model auto-selection.

In Offline Mode the editorial AI must run on a local Ollama model that fits the
GPU (the GTX 1650, 4 GB). ``rank_local_editorial_models`` filters vision models,
drops anything too large for the card, and ranks the rest best-first so the
pipeline can auto-pick the editorial primary (rank[0]) and fallback (rank[1]).
"""
from backend.services.local_models import (
    rank_local_editorial_models,
    _parse_params_b,
    _is_vision,
    select_local_editorial_models,
)
from backend.config import settings


def _run(coro):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(coro)


# The exact set of models from the user's GTX 1650 box.
BOX_MODELS = [
    "qwen2.5:3b-instruct", "qwen2.5:3b", "moondream:1.8b",
    "llama3.2:3b", "llava:latest",
]


def test_parse_params_b():
    assert _parse_params_b("qwen2.5:3b-instruct") == 3.0
    assert _parse_params_b("llama3.1:8b") == 8.0
    assert _parse_params_b("phi3:3.8b") == 3.8
    assert _parse_params_b("deepseek-r1:1.5b") == 1.5
    assert _parse_params_b("mistral:latest") is None


def test_vision_models_detected():
    assert _is_vision("llava:latest")
    assert _is_vision("moondream:1.8b")
    assert _is_vision("llama3.2-vision:11b")
    assert not _is_vision("qwen2.5:3b-instruct")
    assert not _is_vision("llama3.2:3b")


def test_ranks_instruct_first_then_family():
    ranked = rank_local_editorial_models(BOX_MODELS, max_params_b=4.0)
    # Vision models dropped entirely.
    assert "moondream:1.8b" not in ranked
    assert "llava:latest" not in ranked
    # Instruct-tuned wins; qwen family beats llama at the same size.
    assert ranked == ["qwen2.5:3b-instruct", "qwen2.5:3b", "llama3.2:3b"]


def test_best_and_second_best_are_local():
    ranked = rank_local_editorial_models(BOX_MODELS, max_params_b=4.0)
    assert ranked[0] == "qwen2.5:3b-instruct"   # editorial primary
    assert ranked[1] == "qwen2.5:3b"            # offline fallback — also local


def test_oversized_models_skipped_for_the_1650():
    # A 7B/70B would spill to CPU on a 4 GB card → never selected.
    models = ["qwen2.5:7b-instruct", "llama3.1:70b", "qwen2.5:3b-instruct"]
    assert rank_local_editorial_models(models, max_params_b=4.0) == ["qwen2.5:3b-instruct"]
    # A bigger card (higher budget) would prefer the larger instruct model.
    assert rank_local_editorial_models(models, max_params_b=8.0)[0] == "qwen2.5:7b-instruct"


def test_vision_only_box_yields_nothing_rankable():
    assert rank_local_editorial_models(["llava:latest", "moondream:1.8b"]) == []


def test_unknown_size_models_kept_below_sized_ones():
    ranked = rank_local_editorial_models(["mistral:latest", "qwen2.5:3b-instruct"], max_params_b=4.0)
    assert ranked[0] == "qwen2.5:3b-instruct"
    assert "mistral:latest" in ranked


def test_select_uses_provided_list_without_network():
    picks = _run(select_local_editorial_models(limit=2, model_names=BOX_MODELS))
    assert picks == ["qwen2.5:3b-instruct", "qwen2.5:3b"]


def test_select_falls_back_to_configured_when_none_rankable():
    settings.OLLAMA_EDITORIAL_MODEL = "qwen2.5:3b-instruct"
    picks = _run(select_local_editorial_models(limit=2, model_names=["llava:latest"]))
    assert picks == ["qwen2.5:3b-instruct"]
