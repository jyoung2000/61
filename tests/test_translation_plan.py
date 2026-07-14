"""Size-aware translation plan (large-model throughput fix, 2026-07-14).

A 12B translation model can't hold parallel KV caches in a modest budget and was
paying the fixed prompt prefix on 115 tiny batches. ``translation_plan`` switches
large models to fewer/bigger batches on a single slot; small models are unchanged.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.local_models import translation_plan  # noqa: E402


def test_small_model_unchanged():
    # 4B Ollama, big track, Companion advertising 3 slots → the old small plan.
    p = translation_plan("qwen3:4b-instruct-2507-q4_K_M", 918, is_ollama=True, companion_parallel=3)
    assert p == {"batch": 8, "num_ctx": 2048, "concurrency": 3}


def test_large_model_gets_big_batch_single_slot():
    p = translation_plan("gemma3:12b-it-q4_K_M", 918, is_ollama=True, companion_parallel=3)
    assert p["batch"] == 20          # 918 → ~46 batches, not 115
    assert p["num_ctx"] == 8192      # room for a 20-line batch + output
    assert p["concurrency"] == 1     # a 12B serves one slot — don't fan out


def test_threshold_boundary_10b():
    # qwen2.5:14b is large; qwen3:8b is small (boundary at 10B).
    assert translation_plan("qwen2.5:14b-instruct-q4_K_M", 100, companion_parallel=2)["batch"] == 20
    assert translation_plan("qwen3:8b-q4_K_M", 100, companion_parallel=3)["batch"] == 8


def test_batch_clamped_to_cue_count():
    # A short track never asks for more batch than it has cues.
    assert translation_plan("gemma3:12b-it-q4_K_M", 5, companion_parallel=3)["batch"] == 5
    assert translation_plan("qwen3:4b-q4_K_M", 3, companion_parallel=3)["batch"] == 3


def test_unknown_size_stays_small():
    # No parseable size tag → treated as small (safe default).
    p = translation_plan("mysterymodel:latest", 100, is_ollama=True, companion_parallel=2)
    assert p["batch"] == 8 and p["concurrency"] == 2


def test_cloud_small_model_batch_18():
    p = translation_plan("gpt-4o-mini", 100, is_ollama=False, companion_parallel=1)
    assert p["batch"] == 18


def test_explicit_batch_override_wins_on_small_path(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TRANSLATION_LLM_BATCH", 6, raising=False)
    assert translation_plan("qwen3:4b-q4_K_M", 100, is_ollama=True, companion_parallel=2)["batch"] == 6


def test_large_concurrency_never_exceeds_companion():
    # Even if the large-model concurrency knob were raised, it can't exceed what
    # the Companion advertises.
    p = translation_plan("gemma3:12b-it-q4_K_M", 100, companion_parallel=1)
    assert p["concurrency"] == 1
