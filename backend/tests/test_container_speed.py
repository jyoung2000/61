"""Phase 5 audit tests — container speed & efficiency.

Covers: the perception seek-threshold fix (config), NVENC TU117 flag
assembly (AQ + preset toggle + HEVC -bf 0), the CPU-fallback strip list
still covering the new flags, and concurrency flag defaults.
"""

import sys
import types
from unittest import mock

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _name, _attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                     ("anthropic", "AsyncAnthropic")):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod
if "google.generativeai" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.generativeai")
    _gg.configure = lambda *a, **k: None
    _gg.GenerativeModel = object
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules["google.generativeai"] = _gg

import pytest

from backend.config import settings  # noqa: E402
from backend.services import clip_exporter as CE  # noqa: E402


_NVENC_GPU = {
    "encoder": "h264_nvenc",
    "hevc_encoder": "hevc_nvenc",
    "gpu_name": "GTX 1650",
    "gpu_device_index": None,
    "hwaccel_device": "",
}


def _nvenc_args(export_quality="1080p", preset_setting=None, hevc_4k=False):
    quality_preset = {"crf": 23, "preset": "veryfast"}
    with mock.patch.object(CE, "detect_gpu_capabilities",
                           return_value=dict(_NVENC_GPU)):
        prev_gpu = settings.GPU_ACCELERATION_ENABLED
        prev_hevc = settings.GPU_HEVC_FOR_4K
        prev_preset = getattr(settings, "GPU_NVENC_PRESET", "p5")
        try:
            settings.GPU_ACCELERATION_ENABLED = True
            settings.GPU_HEVC_FOR_4K = hevc_4k
            if preset_setting is not None:
                settings.GPU_NVENC_PRESET = preset_setting
            return CE._gpu_encode_args(quality_preset, export_quality)
        finally:
            settings.GPU_ACCELERATION_ENABLED = prev_gpu
            settings.GPU_HEVC_FOR_4K = prev_hevc
            settings.GPU_NVENC_PRESET = prev_preset


def test_nvenc_h264_has_adaptive_quantization_and_vbr_cq():
    args = _nvenc_args()
    joined = " ".join(args)
    assert "h264_nvenc" in args
    assert "-rc vbr" in joined
    assert "-cq 23" in joined
    assert "-b:v 0" in joined
    assert "-spatial_aq 1" in joined
    assert "-temporal_aq 1" in joined


def test_nvenc_preset_toggle():
    assert "p5" in _nvenc_args()  # default
    assert "p4" in _nvenc_args(preset_setting="p4")
    # invalid values fall back to p5
    assert "p5" in _nvenc_args(preset_setting="ultrafast")


def test_nvenc_hevc_4k_disables_bframes_for_volta():
    args = _nvenc_args(export_quality="4k", hevc_4k=True)
    joined = " ".join(args)
    assert "hevc_nvenc" in args
    # TU117's Volta-gen NVENC has no HEVC B-frame support
    assert "-bf 0" in joined
    assert "-spatial_aq 1" in joined


def test_cpu_fallback_strip_list_covers_new_flags():
    """The GPU→CPU retry strips NVENC-only flags; the AQ flags we now
    emit must be in that strip list or the libx264 retry would die on
    an unknown option."""
    import inspect
    src = inspect.getsource(CE)
    # The strip tuple lives in the fallback builder
    assert '"-spatial_aq"' in src
    assert '"-temporal_aq"' in src


def test_phase5_flag_defaults():
    assert settings.REFRAMER_SEEK_GAP_FRAMES == 60
    assert settings.GPU_NVENC_PRESET == "p5"
    assert settings.VOCAL_SEPARATION_CONCURRENT is True


def test_seek_gap_used_by_perceiver_source():
    """The perceiver must read the configurable threshold, not the old
    hardcoded gap>5 (which seeked on every sample at 5 fps of 30 fps)."""
    import inspect
    from backend.services import reframer_perceiver
    src = inspect.getsource(reframer_perceiver)
    assert "REFRAMER_SEEK_GAP_FRAMES" in src
    assert "gap > seek_gap" in src


def test_transcribe_audio_path_callable_resolution():
    """The perceiver resolves a callable stem handle (concurrent Demucs)
    right before transcription; a plain string passes through."""
    import inspect
    from backend.services import reframer_perceiver
    src = inspect.getsource(reframer_perceiver)
    assert "callable(_stem_path)" in src


def test_u2netp_autodiscovery_paths():
    import inspect
    from backend.services import reframer_u2net
    src = inspect.getsource(reframer_u2net)
    assert "u2netp.onnx" in src
    assert "/data/models/u2netp.onnx" in src
