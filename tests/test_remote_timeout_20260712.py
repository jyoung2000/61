"""Regression tests: long videos must not lose their transcript to transport.

The observed failure: a fixed 600 s request timeout abandoned the whole-file
decode of a 128-min video mid-flight (turbo needs ~25-30 min; full large-v3
more), the retries then hit the Companion's 503-busy instantly (its GPU was
still chewing the abandoned decode) and burned out in under a minute — the
pipeline continued with NO transcript. The timeout now scales with audio
duration and 503-busy waits on its own budget without consuming retries.
"""

import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings
from backend.services.reframer_audio import RemoteWhisperEngine


def _wav_bytes(minutes: float) -> bytes:
    # 16 kHz mono s16 PCM ⇒ 32,000 bytes per second (headers negligible).
    return b"\x00" * int(minutes * 60 * 32000)


def test_timeout_scales_with_audio_duration():
    t42 = RemoteWhisperEngine._timeout_for(_wav_bytes(42))
    t128 = RemoteWhisperEngine._timeout_for(_wav_bytes(128))
    assert t42 == pytest.approx(600 + 42 * 60, rel=0.02)
    assert t128 == pytest.approx(600 + 128 * 60, rel=0.02)
    assert t128 > 25 * 60, "a 128-min decode must be allowed to run to completion"


def test_timeout_has_floor_and_cap(monkeypatch):
    # A tiny gap-recovery slice stays at ~base (base + a few seconds of audio).
    assert RemoteWhisperEngine._timeout_for(_wav_bytes(0.5)) == pytest.approx(630.0)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_TIMEOUT_MAX_S", 4000.0,
                        raising=False)
    assert RemoteWhisperEngine._timeout_for(_wav_bytes(300)) == 4000.0


def _engine(monkeypatch):
    monkeypatch.setattr(
        "backend.services.reframer_audio._remote_whisper_base",
        lambda: "http://companion.test:11500")
    monkeypatch.setattr(
        "backend.services.reframer_audio._remote_whisper_token", lambda: "")
    return RemoteWhisperEngine()


def test_busy_503_does_not_consume_upload_retries(monkeypatch):
    eng = _engine(monkeypatch)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_BUSY_WAIT_S", 600.0,
                        raising=False)
    calls = {"n": 0}

    def _fake_post(wav_bytes, filename, data, headers, timeout_s=None):
        calls["n"] += 1
        if calls["n"] <= 4:   # more busy answers than UPLOAD_RETRIES
            return (False, None, True, "busy/paused (503)", 0.01)
        return (True, {"segments": [{"text": "ok", "start": 0, "end": 1}],
                       "language": "ja"}, False, "", 0.0)

    monkeypatch.setattr(eng, "_post_wav", _fake_post)
    monkeypatch.setattr(
        "backend.services.cloud_transcription._map_verbose_json",
        lambda p: p.get("segments", []))
    out = eng._transcribe_payload(b"\x00" * 32000, "a.wav", {}, {}, "ja", "m")
    assert out is not None and out["segments"], \
        "busy answers must be waited out, not treated as failed attempts"
    assert calls["n"] == 5


def test_busy_budget_exhaustion_still_fails(monkeypatch):
    eng = _engine(monkeypatch)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_BUSY_WAIT_S", 0.0,
                        raising=False)   # no patience → busy consumes retries
    calls = {"n": 0}

    def _always_busy(wav_bytes, filename, data, headers, timeout_s=None):
        calls["n"] += 1
        return (False, None, True, "busy/paused (503)", 0.01)

    monkeypatch.setattr(eng, "_post_wav", _fake := _always_busy)
    out = eng._transcribe_payload(b"\x00" * 32000, "a.wav", {}, {}, "ja", "m")
    assert out is None
    assert calls["n"] == RemoteWhisperEngine.UPLOAD_RETRIES


def test_non_busy_failures_still_bounded(monkeypatch):
    eng = _engine(monkeypatch)
    calls = {"n": 0}

    def _network_error(wav_bytes, filename, data, headers, timeout_s=None):
        calls["n"] += 1
        return (False, None, True, "ConnectError: boom", 0.01)

    monkeypatch.setattr(eng, "_post_wav", _network_error)
    out = eng._transcribe_payload(b"\x00" * 32000, "a.wav", {}, {}, "ja", "m")
    assert out is None
    assert calls["n"] == RemoteWhisperEngine.UPLOAD_RETRIES
