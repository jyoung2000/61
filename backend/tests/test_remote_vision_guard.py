"""The vision offload must not fight remote Whisper for the same Companion GPU.

Regression guard for the observed failure: with the sidecar installed, ClipAI
offloaded face detection to the same Companion that was decoding Whisper, which
starved the transcription (18-min hang) and made the Companion unresponsive.
Default behavior is now: stay LOCAL when the offload would share the GPU with
remote Whisper; only offload when the operator explicitly opts in.
"""
import time

from backend.config import settings
from backend.services.remote_vision import RemoteVisionDetector


def test_offload_stays_local_by_default_when_whisper_is_remote(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", False, raising=False)
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "http://gpu:11500")
    assert d.available() is False


def test_offload_allowed_when_operator_opts_in(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", True, raising=False)
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "http://gpu:11500")
    # Prime the health cache so available() doesn't touch the network.
    d._healthy = True
    d._healthy_at = time.monotonic()
    assert d.available() is True


def test_offload_stays_local_when_no_whisper_host(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "")
    assert d.available() is False
