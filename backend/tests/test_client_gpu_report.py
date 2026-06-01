"""The browser/phone client GPU report must NEVER reconfigure the SERVER's GPU
targeting.

A phone reporting "Adreno 740, gpu_index=0, vendor=qualcomm" previously:
  * overwrote settings.GPU_DEVICE_INDEX (mis-pointing the server's FFmpeg), and
  * could flip server GPU acceleration on the strength of the client's claim.
Both corrupted "use the 1650, not the CPU or the device GPU". This pins the
corrected contract: the client report leaves server GPU device + model config
untouched, and only auto-enables acceleration when the SERVER itself has NVIDIA.

The settings router pulls the heavy provider/cv2/torch chain, so stub ONLY
genuinely-missing modules (never shadow installed ones → no sibling pollution).
"""

import asyncio
import importlib.util
import sys
import types

import pytest


class _AnyModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        return type(name, (), {})


def _stub_if_missing(name):
    if name in sys.modules:
        return
    base = name.split(".")[0]
    try:
        if importlib.util.find_spec(base) is not None:
            return
    except Exception:
        pass
    sys.modules[name] = _AnyModule(name)


for _n in ["google", "google.generativeai", "groq", "replicate", "cv2", "torch",
           "faster_whisper", "librosa", "soundfile", "sentencepiece",
           "ctranslate2", "transformers"]:
    _stub_if_missing(_n)
if isinstance(sys.modules.get("google"), _AnyModule):
    sys.modules["google"].generativeai = sys.modules.get(
        "google.generativeai", _AnyModule("google.generativeai"))

import backend.routers.settings as s  # noqa: E402
from backend.config import settings  # noqa: E402


def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _no_persist(monkeypatch):
    # Don't touch the real user_settings.json during the test.
    monkeypatch.setattr(s, "_persist_user_settings", lambda: True)
    # Pretend the SERVER has NO nvidia GPU unless a test says otherwise.
    monkeypatch.setattr(s, "subprocess",
                        types.SimpleNamespace(run=lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="")),
                        raising=False)
    yield


def test_phone_report_does_not_change_server_device_index(monkeypatch):
    settings.GPU_DEVICE_INDEX = "7"   # the server's real, correct device
    settings.GPU_ACCELERATION_ENABLED = True
    req = s.ClientGpuReport(
        gpu_name="Adreno 740", gpu_vendor="qualcomm", gpu_index="0",
        webgpu_supported=True, whisper_capable=True,
    )
    _aiorun(s.report_client_gpu(req))
    # The phone's index 0 must NOT overwrite the server's device.
    assert settings.GPU_DEVICE_INDEX == "7"


def test_phone_report_does_not_enable_accel_without_server_nvidia(monkeypatch):
    settings.GPU_ACCELERATION_ENABLED = False
    # No nvidia-smi (returncode 1 via the autouse stub) and no /dev/nvidia*.
    monkeypatch.setattr(s, "glob",
                        types.SimpleNamespace(glob=lambda *a, **k: []), raising=False)
    req = s.ClientGpuReport(gpu_name="Adreno 740", gpu_vendor="qualcomm",
                            whisper_capable=True)
    _aiorun(s.report_client_gpu(req))
    # A phone claiming a GPU must NOT flip the server into GPU mode.
    assert settings.GPU_ACCELERATION_ENABLED is False


def test_does_not_persist_whisper_model_via_client_report(monkeypatch):
    # The client report path must not persist server model settings (the
    # WHISPER_MODEL=medium-overwrite regression). We assert _persist_user_settings
    # is NOT called when there's nothing server-side to change.
    settings.GPU_DEVICE_INDEX = "0"
    settings.GPU_ACCELERATION_ENABLED = True   # already on → no auto-enable
    calls = {"n": 0}
    monkeypatch.setattr(s, "_persist_user_settings",
                        lambda: calls.__setitem__("n", calls["n"] + 1) or True)
    req = s.ClientGpuReport(gpu_name="Adreno 740", gpu_vendor="qualcomm", gpu_index="0")
    _aiorun(s.report_client_gpu(req))
    assert calls["n"] == 0   # nothing server-side changed → no persist
