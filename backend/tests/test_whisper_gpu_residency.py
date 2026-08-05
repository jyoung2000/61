"""Tests for AudioIntelligence.gpu_residency_estimate — the Whisper VRAM
estimate the live gauge falls back to when nvidia-smi can't be reached
(faster-whisper is CTranslate2, whose VRAM torch's allocator can't see)."""

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframer_audio import AudioIntelligence

GB = 1024 ** 3


def _reset():
    AudioIntelligence._cached_engine = None
    AudioIntelligence._cached_model_name = None
    AudioIntelligence._cached_device = None
    AudioIntelligence._last_loaded_model_name = None
    AudioIntelligence._last_loaded_device = None


def test_estimate_zero_when_no_engine_loaded():
    _reset()
    est = AudioIntelligence.gpu_residency_estimate()
    assert est["loaded"] is False
    assert est["on_gpu"] is False
    assert est["est_bytes"] == 0


def test_estimate_for_large_v3_float16_on_gpu():
    _reset()
    AudioIntelligence._cached_engine = object()  # a loaded engine
    AudioIntelligence._cached_model_name = "large-v3"
    AudioIntelligence._cached_device = "cuda_float16"
    est = AudioIntelligence.gpu_residency_estimate()
    assert est["loaded"] is True
    assert est["on_gpu"] is True
    assert est["est_bytes"] == int(3.0 * GB)
    assert est["model"] == "large-v3"


def test_estimate_int8_footprint_is_smaller():
    _reset()
    AudioIntelligence._cached_engine = object()
    AudioIntelligence._cached_model_name = "large-v3"
    AudioIntelligence._cached_device = "cuda_int8_float16"
    est = AudioIntelligence.gpu_residency_estimate()
    assert est["est_bytes"] == int(1.6 * GB)


def test_estimate_zero_on_cpu():
    _reset()
    AudioIntelligence._cached_engine = object()
    AudioIntelligence._cached_model_name = "large-v3"
    AudioIntelligence._cached_device = "cpu_int8"
    est = AudioIntelligence.gpu_residency_estimate()
    assert est["on_gpu"] is False
    assert est["est_bytes"] == 0


def test_estimate_unknown_model_uses_precision_default():
    _reset()
    AudioIntelligence._cached_engine = object()
    AudioIntelligence._cached_model_name = "some-custom-model"
    AudioIntelligence._cached_device = "cuda_float16"
    est = AudioIntelligence.gpu_residency_estimate()
    assert est["est_bytes"] == int(3.0 * GB)  # non-int8 default


def test_whisper_rank_orders_model_families_for_downgrade_detection():
    """A serve-DOWNGRADE (large-v3-turbo request answered with `small`) must
    be distinguishable from a cosmetic name difference — the measured failure
    decoded a whole episode on `small`/beam-1 because a stale VRAM budget
    capped the Companion's tier, and the backend accepted it silently."""
    from backend.services.reframer_audio import _whisper_rank
    assert _whisper_rank("large-v3") == 4
    assert _whisper_rank("ggml-large-v3-turbo.bin") == 3
    assert _whisper_rank("medium") == 2
    assert _whisper_rank("small") == 1
    assert _whisper_rank("base") == 1
    assert _whisper_rank("") == 0
    # The downgrade predicate used by the one-shot retry:
    assert 0 < _whisper_rank("small") < _whisper_rank("large-v3-turbo")
    # Same family spelled differently is NOT a downgrade.
    assert not (_whisper_rank("whisper-large-v3-turbo") < _whisper_rank("large-v3-turbo"))
