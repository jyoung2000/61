"""Tests for the NLLB-1.3B default + VRAM/disk gating (Task 2).

The default offline model is now ``facebook/nllb-200-distilled-1.3B``; its
larger footprint must scale the free-VRAM threshold (GPU-vs-CPU ``auto``
decision) and the pre-download disk-headroom floor, while a user can pin back
to the 600M via the ``NMT_NLLB_MODEL`` env override.
"""

import sys
import types

import pytest


def _install_config_stub(**overrides):
    mod = types.ModuleType("backend.config")
    defaults = dict(
        NMT_NLLB_MODEL="facebook/nllb-200-distilled-1.3B",
        NMT_OPUS_MT_TEMPLATE="Helsinki-NLP/opus-mt-{src}-{tgt}",
        NMT_MAX_OPUS_PAIRS=5,
        NMT_AUTODOWNLOAD=True,
        NMT_DEVICE="auto",
    )
    defaults.update(overrides)
    mod.settings = types.SimpleNamespace(**defaults)
    sys.modules["backend.config"] = mod
    return mod


@pytest.fixture()
def N():
    _install_config_stub()
    from backend.services import nmt_translator as N
    return N


def test_is_1p3b_detection(N):
    assert N._is_nllb_1p3b("facebook/nllb-200-distilled-1.3B")
    assert N._is_nllb_1p3b("facebook/nllb-200-distilled-1.3b")  # case-insensitive
    assert not N._is_nllb_1p3b("facebook/nllb-200-distilled-600M")
    assert not N._is_nllb_1p3b(None)


def test_vram_floor_scales_with_model(N):
    # 1.3B needs a higher free-VRAM floor than the 600M.
    big = N._nllb_cuda_min_free_gb("facebook/nllb-200-distilled-1.3B")
    small = N._nllb_cuda_min_free_gb("facebook/nllb-200-distilled-600M")
    assert big > small
    assert big == N._NLLB_1P3B_CUDA_MIN_FREE_GB == 2.4
    assert small == N._NLLB_CUDA_MIN_FREE_GB == 1.8


def test_disk_floor_scales_with_model(N):
    big = N._nllb_min_free_bytes("facebook/nllb-200-distilled-1.3B")
    small = N._nllb_min_free_bytes("facebook/nllb-200-distilled-600M")
    assert big > small
    assert big == N._NLLB_MIN_FREE_BYTES == 8 * 1024 ** 3
    assert small == N._NLLB_600M_MIN_FREE_BYTES == 6 * 1024 ** 3


def test_default_model_is_1p3b(N):
    tr = N.NMTTranslator()
    assert tr.model_id == "facebook/nllb-200-distilled-1.3B"


def test_env_override_pins_back_to_600m():
    # A user on a smaller card can pin back via NMT_NLLB_MODEL.
    _install_config_stub(NMT_NLLB_MODEL="facebook/nllb-200-distilled-600M")
    from backend.services import nmt_translator as N
    tr = N.NMTTranslator()
    assert tr.model_id == "facebook/nllb-200-distilled-600M"
    assert N._nllb_cuda_min_free_gb(tr.model_id) == 1.8
    assert N._nllb_min_free_bytes(tr.model_id) == 6 * 1024 ** 3


def test_ensure_download_uses_model_aware_disk_floor(N, monkeypatch, tmp_path):
    """ensure_nllb_downloaded must check the 1.3B floor (8 GB), not the 600M's."""
    monkeypatch.setattr(N, "_models_dir", lambda: str(tmp_path))
    seen = {}

    def _fake_require(target_dir, min_free, label):
        seen["min_free"] = min_free
        raise OSError("stop here — only checking the floor passed")

    monkeypatch.setattr(N, "_require_free_space", _fake_require)
    monkeypatch.setattr(N.NMTTranslator, "_model_files_present",
                        staticmethod(lambda *_a: False))

    with pytest.raises(OSError):
        N.ensure_nllb_downloaded("facebook/nllb-200-distilled-1.3B")
    assert seen["min_free"] == 8 * 1024 ** 3
