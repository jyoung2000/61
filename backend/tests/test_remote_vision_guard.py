"""The vision offload shares the Companion GPU with remote Whisper — the policy
is decided by the Companion's REAL free VRAM, not a blanket block.

Background: with the sidecar installed, offloading faces to the same Companion
that decodes Whisper starved transcription on a SMALL card (18-min hang). But on
a big card (a 4070's 12 GB) both fit, and keeping faces on the small server GPU
is the real bottleneck the user installed the offload to avoid. So overlap now
engages when the Companion reports enough headroom, stays local when it doesn't,
and the operator can still force either way.
"""
import time

from backend.config import settings
from backend.services.remote_vision import RemoteVisionDetector


def _detector(monkeypatch, free_mb):
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "http://gpu:11500")
    monkeypatch.setattr(d, "_companion_free_mb", lambda base: free_mb)
    # Prime the sidecar-health cache so available() doesn't touch the network
    # once the overlap policy says yes.
    d._healthy = True
    d._healthy_at = time.monotonic()
    return d


def test_offload_engages_when_companion_has_headroom(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", False, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_AUTO_WHEN_HEADROOM", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_MIN_FREE_MB", 3500, raising=False)
    d = _detector(monkeypatch, free_mb=8100)   # a 4070 with room to spare
    assert d.available() is True


def test_offload_stays_local_when_headroom_low(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", False, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_AUTO_WHEN_HEADROOM", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_MIN_FREE_MB", 3500, raising=False)
    d = _detector(monkeypatch, free_mb=1500)   # a small, contended card
    assert d.available() is False


def test_offload_stays_local_when_companion_vram_unreadable(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", False, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_AUTO_WHEN_HEADROOM", True, raising=False)
    d = _detector(monkeypatch, free_mb=None)   # probe failed → be conservative
    assert d.available() is False


def test_offload_forced_on_ignores_vram_reading(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", True, raising=False)
    d = _detector(monkeypatch, free_mb=500)    # low VRAM, but forced
    assert d.available() is True


def test_offload_stays_local_when_auto_disabled(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER", False, raising=False)
    monkeypatch.setattr(settings, "REMOTE_VISION_AUTO_WHEN_HEADROOM", False, raising=False)
    d = _detector(monkeypatch, free_mb=8100)   # ample, but auto turned off
    assert d.available() is False


def test_offload_stays_local_when_no_whisper_host(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_VISION_ENABLED", True, raising=False)
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "")
    assert d.available() is False
