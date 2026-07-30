"""Remote Whisper (OpenAI-compatible /v1/audio/transcriptions) contract tests.

The remote path must produce the exact segment schema the local
faster-whisper path produces (so SRT/ASS generation and TACT correction
inputs are unchanged), select itself only when configured + healthy, fall
back to the local ladder on any mid-stage failure, and let the pipeline
skip the VRAM-release dance when the local GPU was never touched."""

import asyncio
import json
import os
import socket
import sys
import threading
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings
from backend.services import reframer_audio as RA
from backend.services.reframer_audio import (
    AudioIntelligence, RemoteWhisperEngine,
    remote_whisper_configured, remote_whisper_pick_model,
)


# The canonical local-path segment schema (what transcribe()'s local loop
# emits). Downstream consumers key on exactly these fields.
LOCAL_SEGMENT_KEYS = {
    "start_sec", "end_sec", "text", "words",
    "is_hallucination", "no_speech_prob", "avg_logprob",
}
LOCAL_WORD_KEYS = {"word", "start", "end", "confidence"}


VERBOSE_JSON = {
    "task": "transcribe",
    "language": "en",
    "duration": 3.4,
    "text": "Hello world. This is a test.",
    "segments": [
        {"id": 0, "start": 0.0, "end": 1.6, "text": " Hello world.",
         "no_speech_prob": 0.02, "avg_logprob": -0.21},
        {"id": 1, "start": 1.7, "end": 3.4, "text": " This is a test.",
         "no_speech_prob": 0.03, "avg_logprob": -0.18},
    ],
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.6},
        {"word": "world.", "start": 0.7, "end": 1.5},
        {"word": "This", "start": 1.7, "end": 2.0},
        {"word": "is", "start": 2.1, "end": 2.3},
        {"word": "a", "start": 2.4, "end": 2.5},
        {"word": "test.", "start": 2.6, "end": 3.3},
    ],
}


class _FixtureServer(threading.Thread):
    """Minimal OpenAI-compatible transcription server on a real socket."""

    def __init__(self, status_sequence=(200,), payload=None):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.requests = []       # raw header blocks (decoded)
        self.paths = []
        self._statuses = list(status_sequence)
        self._payload = payload if payload is not None else VERBOSE_JSON
        self._stop = threading.Event()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def run(self):
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed by stop()
            with conn:
                conn.settimeout(3)
                data = b""
                try:
                    while b"\r\n\r\n" not in data:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    head, rest = data.split(b"\r\n\r\n", 1)
                    clen = 0
                    for line in head.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            clen = int(line.split(b":", 1)[1].strip())
                    while len(rest) < clen:
                        more = conn.recv(65536)
                        if not more:
                            break
                        rest += more
                except Exception:
                    head = data
                self.requests.append(head.decode("utf-8", "replace"))
                first_line = self.requests[-1].split("\r\n", 1)[0]
                path = first_line.split(" ")[1] if " " in first_line else "/"
                self.paths.append(path)
                if path.endswith("/v1/audio/transcriptions"):
                    status = self._statuses.pop(0) if self._statuses else 200
                else:
                    status = 200  # health probes
                if status == 200 and path.endswith("/v1/audio/transcriptions"):
                    body = json.dumps(self._payload)
                elif status == 503:
                    body = json.dumps({"error": "busy"})
                else:
                    body = json.dumps({"status": "ok", "gpu_name": "RTX 4070"})
                extra = "Retry-After: 0\r\n" if status == 503 else ""
                resp = (f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
                        f"{extra}Content-Length: {len(body)}\r\n"
                        f"Connection: close\r\n\r\n{body}")
                try:
                    conn.sendall(resp.encode())
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_MODEL", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL_USER_SET", False, raising=False)
    RA._REMOTE_HEALTH_CACHE.update({"checked_at": 0.0, "healthy": False, "url": ""})
    AudioIntelligence._cached_engine = None
    AudioIntelligence._cached_model_name = None
    AudioIntelligence._cached_device = None
    AudioIntelligence._last_loaded_model_name = None
    AudioIntelligence._last_loaded_device = None
    yield


@pytest.fixture
def wav_file(tmp_path):
    # Content is irrelevant — the fixture server never decodes it.
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF" + b"\x00" * 64)
    return str(p)


# ── Model auto-ladder ────────────────────────────────────────────────


def test_pick_model_auto_ladder(monkeypatch):
    # Default WHISPER_REMOTE_PREFER_ACCURACY=True → full large-v3 even for
    # English/auto (the Companion GPU can afford the accuracy win); pinned
    # non-English always uses large-v3.
    assert remote_whisper_pick_model(None) == "large-v3"
    assert remote_whisper_pick_model("en") == "large-v3"
    assert remote_whisper_pick_model("auto") == "large-v3"
    assert remote_whisper_pick_model("ja") == "large-v3"


def test_pick_model_speed_mode(monkeypatch):
    # Opting into speed (PREFER_ACCURACY=False) drops English/auto to the
    # pruned turbo decoder; pinned non-English still uses full large-v3.
    monkeypatch.setattr(settings, "WHISPER_REMOTE_PREFER_ACCURACY", False, raising=False)
    assert remote_whisper_pick_model(None) == "large-v3-turbo"
    assert remote_whisper_pick_model("en") == "large-v3-turbo"
    assert remote_whisper_pick_model("auto") == "large-v3-turbo"
    assert remote_whisper_pick_model("ja") == "large-v3"


def test_pick_model_configured_wins(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_MODEL", "distil-large-v3", raising=False)
    assert remote_whisper_pick_model("ja") == "distil-large-v3"


def test_pick_model_honors_user_pin(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_MODEL_USER_SET", True, raising=False)
    monkeypatch.setattr(settings, "WHISPER_MODEL", "medium", raising=False)
    assert remote_whisper_pick_model("en") == "medium"


# ── Engine contract: schema parity with the local path ───────────────


def test_remote_segments_match_local_schema(monkeypatch, wav_file):
    server = _FixtureServer()
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "sek-ret", raising=False)
        from backend.services.request_context import set_job, set_stage, clear
        set_job("job-7", "Podcast Ep 12")
        set_stage("transcription")
        try:
            engine = RemoteWhisperEngine(model="large-v3-turbo")
            out = engine.transcribe_wav(wav_file, language="en")
        finally:
            clear()
        assert out is not None
        assert out["provider"] == "remote"
        assert out["language"] == "en"
        segs = out["segments"]
        assert len(segs) == 2
        for seg in segs:
            assert LOCAL_SEGMENT_KEYS.issubset(seg.keys())
            for w in seg["words"]:
                assert set(w.keys()) == LOCAL_WORD_KEYS
        assert segs[0]["text"] == "Hello world."
        assert segs[0]["words"][0]["word"] == "Hello"
        # Auth + live-feed headers reached the server; token never logged.
        joined = "\n".join(server.requests).lower()
        assert "authorization: bearer sek-ret" in joined
        assert "x-clipai-job-id: job-7" in joined
        assert "x-clipai-stage: transcription" in joined
        assert "x-clipai-job-title: podcast ep 12" in joined
        # Selected model is synced so the Companion loads the same GPU model.
        assert "x-clipai-whisper-model: large-v3-turbo" in joined
    finally:
        server.stop()


def test_remote_busy_503_retries_once(monkeypatch, wav_file):
    server = _FixtureServer(status_sequence=(503, 200))
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        out = RemoteWhisperEngine(model="large-v3-turbo").transcribe_wav(wav_file, "en")
        assert out is not None
        assert len([p for p in server.paths
                    if p.endswith("/v1/audio/transcriptions")]) == 2
    finally:
        server.stop()


def test_remote_failure_returns_none(monkeypatch, wav_file):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL",
                        "http://127.0.0.1:1", raising=False)
    out = RemoteWhisperEngine(model="large-v3-turbo").transcribe_wav(wav_file, "en")
    assert out is None


# ── Translate pass: model must actually HAVE the translate task ──────


def test_model_lacks_translate_predicate():
    # Distilled checkpoints kept only transcription — task=translate silently
    # transcribes on them (observed: turbo returned 3495 CJK vs 16 Latin).
    assert RA._model_lacks_translate("large-v3-turbo")
    assert RA._model_lacks_translate("distil-large-v3")
    assert RA._model_lacks_translate("kotoba-tech/kotoba-whisper-v2.0-faster")
    # The multitask family translates fine.
    assert not RA._model_lacks_translate("large-v3")
    assert not RA._model_lacks_translate("medium")
    assert not RA._model_lacks_translate("small")
    assert not RA._model_lacks_translate("")


def test_translate_swaps_turbo_for_multitask_model(monkeypatch, wav_file):
    # translate=True on a turbo engine must request the multitask checkpoint
    # (WHISPER_TRANSLATE_MODEL) instead — header AND form model switch so the
    # Companion's ensure_running loads a model that can actually translate.
    server = _FixtureServer()
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        out = RemoteWhisperEngine(model="large-v3-turbo").transcribe_wav(
            wav_file, language="ja", translate=True)
        assert out is not None
        joined = "\n".join(server.requests).lower()
        assert "x-clipai-whisper-model: medium" in joined
        assert "x-clipai-whisper-model: large-v3-turbo" not in joined
    finally:
        server.stop()


def test_translate_keeps_multitask_model(monkeypatch, wav_file):
    # A model that CAN translate is left alone — no silent downgrade.
    server = _FixtureServer()
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        out = RemoteWhisperEngine(model="large-v3").transcribe_wav(
            wav_file, language="ja", translate=True)
        assert out is not None
        joined = "\n".join(server.requests).lower()
        assert "x-clipai-whisper-model: large-v3" in joined
    finally:
        server.stop()


def test_transcribe_never_swaps_model(monkeypatch, wav_file):
    # The swap is translate-pass-only: plain transcription keeps the turbo
    # pick (its transcription quality is the whole point of the family).
    server = _FixtureServer()
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        out = RemoteWhisperEngine(model="large-v3-turbo").transcribe_wav(
            wav_file, language="ja")
        assert out is not None
        joined = "\n".join(server.requests).lower()
        assert "x-clipai-whisper-model: large-v3-turbo" in joined
    finally:
        server.stop()


# ── Selection order in try_load ──────────────────────────────────────


def test_try_load_prefers_healthy_remote(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://gpu:11500", raising=False)
    monkeypatch.setattr(RA, "remote_whisper_healthy", lambda force=False: True)
    ai = AudioIntelligence(model_name="small")
    assert ai.try_load() is True
    assert ai.device_used == "remote"
    assert isinstance(ai.engine, RemoteWhisperEngine)
    # The local GPU/class cache was never touched.
    assert AudioIntelligence._cached_engine is None
    assert AudioIntelligence._last_loaded_device == "remote"


def test_try_load_force_local_skips_remote(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://gpu:11500", raising=False)
    monkeypatch.setattr(RA, "remote_whisper_healthy", lambda force=False: True)
    ai = AudioIntelligence(model_name="small")
    # No faster-whisper in the test env → the local ladder reports False,
    # which proves the remote path was skipped.
    assert ai.try_load(force_local=True) is False
    assert ai.device_used != "remote"


def test_try_load_unhealthy_remote_falls_to_local(monkeypatch):
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://gpu:11500", raising=False)
    monkeypatch.setattr(RA, "remote_whisper_healthy", lambda force=False: False)
    ai = AudioIntelligence(model_name="small")
    assert ai.try_load() is False  # local ladder (no faster-whisper here)
    assert ai.device_used != "remote"


def test_try_load_unconfigured_never_probes(monkeypatch):
    called = {"n": 0}

    def probe(force=False):
        called["n"] += 1
        return True

    monkeypatch.setattr(RA, "remote_whisper_healthy", probe)
    ai = AudioIntelligence(model_name="small")
    ai.try_load()
    assert called["n"] == 0
    assert not remote_whisper_configured()


# ── Full transcribe() through the remote branch ──────────────────────


def test_transcribe_remote_end_to_end(monkeypatch, wav_file, tmp_path):
    server = _FixtureServer()
    server.start()
    try:
        monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", server.url, raising=False)
        ai = AudioIntelligence(model_name="small")
        assert ai.try_load() is True

        video = str(tmp_path / "video.mp4")
        with open(video, "wb") as f:
            f.write(b"\x00" * 128)
        result = ai.transcribe(video, duration_ms=3400, language="en",
                               audio_path_override=wav_file)
        assert result["language"] == "en"
        assert result["transcription_provider"] == "remote"
        assert result["transcription_location"] == "remote"
        assert len(result["segments"]) == 2
        for seg in result["segments"]:
            assert LOCAL_SEGMENT_KEYS.issubset(seg.keys())
        assert result["speech_active"]  # 100ms grid populated
        assert result.get("coverage_ledger") is not None
        # Effective model recorded for the compute summary / Settings page.
        # Auto ladder with the accuracy-first default → full large-v3.
        assert ai.model_name == "large-v3"
        assert AudioIntelligence._last_loaded_device == "remote"
    finally:
        server.stop()


def test_transcribe_remote_failure_falls_back_to_local(monkeypatch, wav_file, tmp_path):
    # Remote passes the health probe but dies on the actual request; the
    # local ladder is unavailable in this env, so transcribe() must return
    # the empty-schema result INSTEAD of raising.
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://127.0.0.1:1", raising=False)
    monkeypatch.setattr(RA, "remote_whisper_healthy", lambda force=False: True)
    ai = AudioIntelligence(model_name="small")
    assert ai.try_load() is True
    video = str(tmp_path / "video.mp4")
    with open(video, "wb") as f:
        f.write(b"\x00" * 128)
    result = ai.transcribe(video, duration_ms=1000, language="en",
                           audio_path_override=wav_file)
    assert result == {"speech_active": {}, "segments": [], "language": ""}


# ── Pipeline: VRAM release skipped for remote transcription ──────────


def test_release_whisper_vram_skipped_for_remote(monkeypatch, caplog):
    from backend.services.pipeline import _release_whisper_vram
    AudioIntelligence._cached_engine = None
    AudioIntelligence._last_loaded_device = "remote"
    with caplog.at_level("INFO"):
        asyncio.run(_release_whisper_vram("job-x"))
    assert any("skipped" in r.message and "remote" in r.message
               for r in caplog.records)


def test_release_whisper_vram_runs_for_local(monkeypatch, caplog):
    from backend.services.pipeline import _release_whisper_vram
    AudioIntelligence._cached_engine = object()
    AudioIntelligence._cached_model_name = "small"
    AudioIntelligence._cached_device = "cuda_float16"
    AudioIntelligence._last_loaded_device = "cuda_float16"
    with caplog.at_level("INFO"):
        asyncio.run(_release_whisper_vram("job-y"))
    assert AudioIntelligence._cached_engine is None  # actually released


# ── Resolution from the Ollama host registry (no separate field) ─────────


def test_remote_whisper_resolves_from_companion_host(monkeypatch):
    """A paired Companion in the Ollama registry IS the Whisper endpoint —
    base URL (minus /ollama) + its token — with no WHISPER_REMOTE_URL set."""
    from backend.services import ollama_registry as R
    R.reset_state()
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", json.dumps([
        {"id": "c", "name": "4070", "url": "http://192.168.8.10:11500/ollama",
         "token": "tok-xyz", "is_companion": True},
        {"id": "l", "name": "local", "url": "http://ollama:11434"},
    ]), raising=False)
    try:
        assert RA.remote_whisper_configured() is True
        assert RA._remote_whisper_base() == "http://192.168.8.10:11500"
        assert RA._remote_whisper_token() == "tok-xyz"
    finally:
        R.reset_state()


def test_remote_whisper_not_configured_without_companion(monkeypatch):
    """A local-only registry (no remote host) and no WHISPER_REMOTE_URL means
    remote Whisper stays off — transcription runs locally."""
    from backend.services import ollama_registry as R
    R.reset_state()
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", json.dumps([
        {"id": "l", "name": "local", "url": "http://ollama:11434"},
    ]), raising=False)
    try:
        assert RA.remote_whisper_configured() is False
        assert RA._remote_whisper_base() == ""
    finally:
        R.reset_state()


def test_remote_whisper_falls_back_to_env_url(monkeypatch):
    """With no Companion but an explicit third-party WHISPER_REMOTE_URL set,
    that URL is still honored (env fallback path)."""
    from backend.services import ollama_registry as R
    R.reset_state()
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://speaches.lan:8000", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "envkey", raising=False)
    try:
        assert RA.remote_whisper_configured() is True
        assert RA._remote_whisper_base() == "http://speaches.lan:8000"
        assert RA._remote_whisper_token() == "envkey"
    finally:
        R.reset_state()


# ── Sidecar release must WAIT for a decode in flight ──────────────────────
# A 409 means the sidecar is mid-decode, not that releasing is impossible.
# Giving up on the first 409 and loading a 14B model beside the resident
# sidecar cost a measured run 447 s for its FIRST batch of 20 cues; the other
# 285 cues took 86 s in total once the card was clear.

def _release_probe(monkeypatch, codes, wait_s):
    """Drive remote_whisper_release against a scripted sequence of statuses."""
    calls = {"n": 0, "slept": 0.0}

    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _FakeHttpx:
        @staticmethod
        def post(url, headers=None, timeout=None):
            i = min(calls["n"], len(codes) - 1)
            calls["n"] += 1
            return _Resp(codes[i])

    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "http://c:11500", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "k", raising=False)
    monkeypatch.setattr(RA, "remote_whisper_configured", lambda: True)
    monkeypatch.setattr(RA, "_remote_whisper_base", lambda: "http://c:11500")
    monkeypatch.setattr(RA, "_remote_whisper_token", lambda: "k")
    monkeypatch.setitem(sys.modules, "httpx", _FakeHttpx)

    import time as _t
    real_sleep, real_mono = _t.sleep, _t.monotonic
    clock = {"t": 0.0}
    monkeypatch.setattr(_t, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(_t, "sleep", lambda s: (clock.__setitem__("t", clock["t"] + s),
                                                calls.__setitem__("slept", calls["slept"] + s)))
    try:
        return RA.remote_whisper_release(wait_s), calls
    finally:
        _t.sleep, _t.monotonic = real_sleep, real_mono


def test_release_retries_while_the_sidecar_is_mid_decode(monkeypatch):
    ok, calls = _release_probe(monkeypatch, [409, 409, 409, 200], wait_s=60.0)
    assert ok is True
    assert calls["n"] == 4, "should have re-polled until the sidecar freed up"
    assert calls["slept"] > 0


def test_release_gives_up_at_the_deadline_rather_than_hanging(monkeypatch):
    ok, calls = _release_probe(monkeypatch, [409], wait_s=6.0)
    assert ok is False
    # 2 s between polls, so a 6 s budget is a handful of attempts, not a hang.
    assert 2 <= calls["n"] <= 6, calls["n"]


def test_release_does_not_wait_when_the_route_is_missing(monkeypatch):
    # 404 = an older Companion with no release route. Retrying cannot help.
    ok, calls = _release_probe(monkeypatch, [404], wait_s=600.0)
    assert ok is False
    assert calls["n"] == 1 and calls["slept"] == 0.0


def test_release_default_is_a_single_attempt(monkeypatch):
    ok, calls = _release_probe(monkeypatch, [409], wait_s=0.0)
    assert ok is False
    assert calls["n"] == 1, "no wait budget → one attempt, same as before"


# ---------------------------------------------------------------------------
# _absolutize_slice_segment — the 0:00 phantom-subtitle fix. Gap recovery
# decodes a SLICE of the video; a measured run shifted start_sec/end_sec
# (keys transcribe_wav output doesn't carry) so every recovered cue shipped
# zero-width at the slice offset while its real start/end stayed
# slice-relative — press-scrum lines rendered at 0:00-0:13 over silence.
# ---------------------------------------------------------------------------

def test_absolutize_shifts_slice_relative_times_onto_both_schemas():
    seg = {"start": 1.2, "end": 3.4, "text": "了解",
           "words": [{"word": "了解", "start": 1.2, "end": 3.4}]}
    assert RA._absolutize_slice_segment(seg, ss=705.0, dur=17.0) is True
    # BOTH schemas carry the same absolute times: every downstream consumer
    # (start-preferring and start_sec-preferring alike) agrees on placement.
    assert seg["start"] == seg["start_sec"] == 706.2
    assert seg["end"] == seg["end_sec"] == 708.4
    assert seg["words"][0]["start"] == 706.2
    assert seg["words"][0]["end"] == 708.4


def test_absolutize_does_not_double_shift_absolute_decodes():
    # Some engines return absolute times already. A value far outside
    # [0, dur] is absolute — shifting it again would land past the slice
    # and (correctly) fail the containment attestation.
    seg = {"start": 706.2, "end": 708.4, "text": "了解"}
    assert RA._absolutize_slice_segment(seg, ss=705.0, dur=17.0) is True
    assert seg["start"] == seg["start_sec"] == 706.2
    assert seg["end"] == seg["end_sec"] == 708.4


def test_absolutize_drops_a_decode_outside_its_own_slice():
    # The attestation: a recovered cue must lie inside the slice it was
    # decoded from (±1.5 s). A mistimed decode is dropped, never shipped
    # somewhere else on the timeline.
    seg = {"start": 40.0, "end": 44.0, "text": "phantom"}
    assert RA._absolutize_slice_segment(seg, ss=705.0, dur=17.0) is False


def test_absolutize_drops_zero_width_and_inverted_decodes():
    assert RA._absolutize_slice_segment(
        {"start": 2.0, "end": 2.0, "text": "x"}, ss=100.0, dur=10.0) is False
    assert RA._absolutize_slice_segment(
        {"start": 5.0, "end": 2.0, "text": "x"}, ss=100.0, dur=10.0) is False


def test_absolutize_clamps_word_rows_into_the_cue():
    # Word rows drive active-word highlighting: they must land inside their
    # own cue after the shift, or the highlight disagrees with the subtitle.
    seg = {"start": 1.0, "end": 4.0, "text": "a b",
           "words": [{"word": "a", "start": 0.2, "end": 1.5},
                     {"word": "b", "start": 3.9, "end": 6.0}]}
    assert RA._absolutize_slice_segment(seg, ss=200.0, dur=10.0) is True
    for w in seg["words"]:
        assert seg["start"] <= w["start"] <= w["end"] <= seg["end"]


def test_absolutize_legacy_start_sec_only_schema_still_works():
    # Defensive: a caller feeding the OLD schema (start_sec/end_sec only)
    # gets the same absolute result instead of a silent zero-width phantom.
    seg = {"start_sec": 1.0, "end_sec": 2.5, "text": "x"}
    assert RA._absolutize_slice_segment(seg, ss=50.0, dur=8.0) is True
    assert seg["start"] == seg["start_sec"] == 51.0
    assert seg["end"] == seg["end_sec"] == 52.5
