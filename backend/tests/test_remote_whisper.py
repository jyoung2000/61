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
