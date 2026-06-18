"""Tests for the live GPU-usage computation behind the Analysis-page panel.

``_compute_gpu_usage`` must reflect the WHOLE pipeline's GPU use — Ollama
models, the app container's torch VRAM (YOLO etc.), AND real device residency
from nvidia-smi (CTranslate2 / Whisper, which torch can't see) — so the panel
truthfully answers "is the GPU being used" during analysis, not just for Ollama.
"""

from backend.routers.diagnostics import _compute_gpu_usage

GB = 1024 ** 3
MB = 1024 ** 2


def test_idle_gpu_not_in_use():
    used, in_use, poisoned = _compute_gpu_usage(
        vram_used_ollama=0, ollama_in_use=False, torch_reserved=0,
        device_used=120 * MB, has_loaded_models=False, gpu_available=True)
    assert in_use is False
    assert poisoned is False
    assert used == 120 * MB                 # prefers the real device figure


def test_ollama_model_on_gpu_in_use():
    used, in_use, poisoned = _compute_gpu_usage(
        vram_used_ollama=2 * GB, ollama_in_use=True, torch_reserved=0,
        device_used=2 * GB, has_loaded_models=True, gpu_available=True)
    assert in_use is True
    assert poisoned is False


def test_whisper_on_gpu_lights_dot_even_without_ollama_or_torch():
    # CTranslate2/Whisper holds ~1.8 GB that torch's allocator can't see; the
    # real device figure is what proves the GPU is in use.
    used, in_use, _ = _compute_gpu_usage(
        vram_used_ollama=0, ollama_in_use=False, torch_reserved=0,
        device_used=int(1.8 * GB), has_loaded_models=False, gpu_available=True)
    assert in_use is True
    assert used == int(1.8 * GB)


def test_torch_based_stage_lights_dot():
    # YOLO subject detection holds VRAM via torch's allocator.
    _used, in_use, _ = _compute_gpu_usage(
        vram_used_ollama=0, ollama_in_use=False, torch_reserved=512 * MB,
        device_used=None, has_loaded_models=False, gpu_available=True)
    assert in_use is True


def test_vram_used_prefers_real_device_over_estimate():
    # device (real) > ollama+torch estimate → report the real figure.
    used, _in_use, _ = _compute_gpu_usage(
        vram_used_ollama=500 * MB, ollama_in_use=True, torch_reserved=200 * MB,
        device_used=3 * GB, has_loaded_models=True, gpu_available=True)
    assert used == 3 * GB
    # estimate wins when nvidia-smi is unavailable.
    used2, _i, _p = _compute_gpu_usage(
        vram_used_ollama=500 * MB, ollama_in_use=True, torch_reserved=200 * MB,
        device_used=None, has_loaded_models=True, gpu_available=True)
    assert used2 == 700 * MB


def test_poisoned_when_models_loaded_but_on_cpu():
    _used, in_use, poisoned = _compute_gpu_usage(
        vram_used_ollama=0, ollama_in_use=False, torch_reserved=0,
        device_used=50 * MB, has_loaded_models=True, gpu_available=True)
    assert poisoned is True
    assert in_use is False


def test_not_poisoned_without_gpu_hardware():
    _used, _in_use, poisoned = _compute_gpu_usage(
        vram_used_ollama=0, ollama_in_use=False, torch_reserved=0,
        device_used=None, has_loaded_models=True, gpu_available=False)
    assert poisoned is False
