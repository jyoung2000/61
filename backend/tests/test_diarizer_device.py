"""Tests: the local ECAPA diarizer prefers the GPU under "auto" (so it embeds
cues in seconds instead of the minutes CPU takes), while explicit overrides and
the no-GPU / low-VRAM cases still resolve to CPU. _load_model keeps the
risk-free CPU fallback, so "auto" can never lose diarization.
"""

import sys
import types

from backend.services.local_diarizer import LocalEmbeddingDiarizer as D


def _fake_torch(*, available: bool, free_mb: float):
    m = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        # mem_get_info returns (free_bytes, total_bytes)
        mem_get_info=lambda: (int(free_mb * 1024 * 1024), int(4096 * 1024 * 1024)),
    )
    m.cuda = cuda
    return m


def test_explicit_cpu_and_cuda_pass_through(monkeypatch):
    # Explicit choices are honored regardless of hardware.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, free_mb=0))
    assert D._resolve_device("cpu") == "cpu"
    assert D._resolve_device("cuda") == "cuda"


def test_auto_prefers_gpu_when_free_vram(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, free_mb=2648))
    assert D._resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, free_mb=0))
    assert D._resolve_device("auto") == "cpu"


def test_auto_stays_cpu_when_vram_too_low(monkeypatch):
    # GPU present but nearly full (< 200 MB free) → don't risk OOM, use CPU.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, free_mb=120))
    assert D._resolve_device("auto") == "cpu"


def test_auto_is_safe_when_torch_missing(monkeypatch):
    # No torch at all → CPU, never raises.
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch → ImportError
    assert D._resolve_device("auto") == "cpu"


def test_default_config_is_auto():
    from backend.config import settings
    assert settings.LOCAL_DIARIZER_DEVICE == "auto"
