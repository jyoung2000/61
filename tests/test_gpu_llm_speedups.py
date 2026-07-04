"""GPU-first model selection + LLM-stage speedups (post-perf-pass follow-up).

Pure-logic coverage (no GPU / Ollama needed):
  * select_gpu_fitting_quant — swap a too-big quant for a fitting one, keep it
    when it already fits or nothing smaller is installed.
  * quant / base-key parsing.
  * batched-translation response parsing.
  * summary chunk-count cap arithmetic.
"""

from __future__ import annotations

import pytest

from backend.services.local_models import (
    _parse_quant,
    _model_base_key,
    estimate_model_weights_gb,
    select_gpu_fitting_quant,
)


# ── quant / base parsing ─────────────────────────────────────────────


def test_parse_quant_longest_match():
    assert _parse_quant("qwen3:4b-instruct-2507-q4_K_M")[0] == "q4_k_m"
    assert _parse_quant("qwen3:4b-instruct-2507-q3_K_M")[0] == "q3_k_m"
    assert _parse_quant("qwen2.5:3b")[0] is None
    # q4_k_m must win over the shorter q4_0/q4 tokens.
    assert _parse_quant("model-q4_K_M")[1] > _parse_quant("model-q3_K_M")[1]


def test_base_key_ignores_quant():
    a = _model_base_key("qwen3:4b-instruct-2507-q4_K_M")
    b = _model_base_key("qwen3:4b-instruct-2507-q3_K_M")
    assert a == b == "qwen3:4b_instruct_2507"
    # A different size is a different base.
    assert _model_base_key("qwen2.5:3b") != _model_base_key("qwen2.5:7b")


def test_estimate_weights_scales_with_quant():
    q4 = estimate_model_weights_gb("qwen3:4b-instruct-2507-q4_K_M")
    q3 = estimate_model_weights_gb("qwen3:4b-instruct-2507-q3_K_M")
    assert q4 > q3 > 0
    # 4B q4 ≈ 2.4 GB, q3 ≈ 1.7 GB.
    assert 2.0 < q4 < 2.7
    assert 1.5 < q3 < 2.0


# ── select_gpu_fitting_quant ─────────────────────────────────────────

_GTX1650 = dict(total_vram_gb=3.7, baseline_reserve_gb=1.2, kv_headroom_gb=0.55)


def test_swaps_q4_to_installed_q3_on_small_card():
    installed = [
        "qwen3:4b-instruct-2507-q4_K_M",
        "qwen3:4b-instruct-2507-q3_K_M",
        "qwen2.5:3b",
    ]
    chosen, reason = select_gpu_fitting_quant(
        "qwen3:4b-instruct-2507-q4_K_M", installed, **_GTX1650)
    assert chosen == "qwen3:4b-instruct-2507-q3_K_M"
    assert reason and "won't fit" in reason


def test_keeps_model_when_no_smaller_quant_installed():
    installed = ["qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b"]
    chosen, reason = select_gpu_fitting_quant(
        "qwen3:4b-instruct-2507-q4_K_M", installed, **_GTX1650)
    # No q3 build of the same base is installed → keep the configured tag.
    assert chosen == "qwen3:4b-instruct-2507-q4_K_M"
    assert reason is None


def test_keeps_model_that_already_fits():
    installed = ["qwen2.5:3b", "qwen2.5:3b-q3_K_M"]
    chosen, reason = select_gpu_fitting_quant("qwen2.5:3b", installed, **_GTX1650)
    assert chosen == "qwen2.5:3b" and reason is None


def test_no_swap_on_ample_vram():
    installed = ["qwen3:4b-instruct-2507-q4_K_M", "qwen3:4b-instruct-2507-q3_K_M"]
    chosen, reason = select_gpu_fitting_quant(
        "qwen3:4b-instruct-2507-q4_K_M", installed,
        total_vram_gb=24.0, baseline_reserve_gb=1.2, kv_headroom_gb=0.55)
    assert chosen == "qwen3:4b-instruct-2507-q4_K_M" and reason is None


def test_no_swap_when_vram_undetectable():
    installed = ["qwen3:4b-instruct-2507-q3_K_M"]
    chosen, reason = select_gpu_fitting_quant(
        "qwen3:4b-instruct-2507-q4_K_M", installed,
        total_vram_gb=0.0, baseline_reserve_gb=1.2, kv_headroom_gb=0.55)
    assert chosen == "qwen3:4b-instruct-2507-q4_K_M" and reason is None


def test_picks_largest_fitting_quant():
    installed = [
        "qwen3:4b-instruct-2507-q4_K_M",
        "qwen3:4b-instruct-2507-q3_K_M",  # ~1.7 GB — fits
        "qwen3:4b-instruct-2507-q2_K",    # ~1.3 GB — also fits, lower fidelity
    ]
    chosen, _ = select_gpu_fitting_quant(
        "qwen3:4b-instruct-2507-q4_K_M", installed, **_GTX1650)
    # Largest quant that fits = best fidelity on GPU.
    assert chosen == "qwen3:4b-instruct-2507-q3_K_M"


# ── batched-translation response parsing ─────────────────────────────


def test_parse_batch_json_object():
    from backend.services.pipeline import _parse_batch_translation_response
    raw = '{"1": "Hello", "2": "World", "3": "Bye"}'
    assert _parse_batch_translation_response(raw, 3) == ["Hello", "World", "Bye"]


def test_parse_batch_json_with_fence_and_preamble():
    from backend.services.pipeline import _parse_batch_translation_response
    raw = 'Sure!\n```json\n{"1": "A", "2": "B"}\n```'
    assert _parse_batch_translation_response(raw, 2) == ["A", "B"]


def test_parse_batch_missing_lines_become_none():
    from backend.services.pipeline import _parse_batch_translation_response
    raw = '{"1": "A", "3": "C"}'
    assert _parse_batch_translation_response(raw, 3) == ["A", None, "C"]


def test_parse_batch_array_form():
    from backend.services.pipeline import _parse_batch_translation_response
    assert _parse_batch_translation_response('["x", "y"]', 2) == ["x", "y"]


def test_parse_batch_garbage_returns_all_none():
    from backend.services.pipeline import _parse_batch_translation_response
    assert _parse_batch_translation_response("not json at all", 3) == [None, None, None]


# ── summary chunk cap arithmetic ─────────────────────────────────────


def test_summary_chunk_cap_reduces_count_on_long_video():
    import math
    video_end = 128 * 60  # 7680s
    chunk_seconds = 5 * 60  # tier default → 26 chunks
    max_chunks = 12
    would = math.ceil(video_end / chunk_seconds)
    assert would > max_chunks
    enlarged = math.ceil(video_end / max_chunks)
    assert math.ceil(video_end / enlarged) <= max_chunks
    # Enlarged span is strictly bigger (fewer, coarser chunks) — never smaller.
    assert enlarged > chunk_seconds


def test_summary_chunk_cap_noop_on_short_video():
    import math
    video_end = 20 * 60
    chunk_seconds = 5 * 60  # 4 chunks, already under the cap
    assert math.ceil(video_end / chunk_seconds) <= 12
