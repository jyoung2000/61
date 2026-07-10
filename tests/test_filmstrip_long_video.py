"""Filmstrip + peaks upgrades for LONG videos.

The editor timeline on a long source depended on a single fine-grained
sprite pass (a whole-file keyframe scan — minutes on a 2-hour video) and,
until it finished, the frontend fell back to seeking a hidden <video> per
thumbnail. These tests pin the new behavior:

  * ``generate_sprite_coarse`` — seek-sampled sheet in seconds, marked
    ``"coarse": true`` with a ``"v"`` stamp; skipped for short sources and
    when a FINE sprite already exists.
  * ``generate_sprite`` manifests carry ``"v"`` so the frontend can
    cache-bust ``sprite.jpg`` when the sheet upgrades.
  * ``generate_peaks`` bins the pipeline's 16 kHz mono ``audio.wav`` via a
    zero-copy memmap (no decode subprocess) and streams FFmpeg output for
    everything else — never holding the whole PCM in RAM.
  * The manifest endpoint serves ``Cache-Control: no-cache`` (the 30-day
    max-age froze the first sheet a client ever saw); the sprite image
    keeps the long max-age and is cache-busted by ``?v=``.
"""

import json
import os
import shutil
import subprocess
import sys
import types
import wave

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_video(path: str, seconds: float, size: str = "320x180"):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={size}:d={seconds}:rate=10", str(path)],
        capture_output=True, check=True)


def _make_wav(path: str, seconds: float, rate: int = 16000, channels: int = 1):
    import numpy as np
    t = np.arange(int(seconds * rate), dtype=np.float32)
    pcm = (np.sin(2 * 3.14159 * 220.0 * t / rate) * 20000).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# ─────────────────────────────────────────────────────────────────────────────
# Coarse sprite
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_coarse_sprite_generates_for_long_source(tmp_path, monkeypatch):
    from backend.services import filmstrip_generator as fg
    # Pretend 20s is "long" so the fixture video stays small.
    monkeypatch.setattr(fg, "_COARSE_MIN_DURATION", 5.0)
    src = tmp_path / "src.mp4"
    _make_video(src, 20)
    out = fg.generate_sprite_coarse(str(src), str(tmp_path / "job"), tiles=12)
    assert out is not None
    assert out["coarse"] is True
    assert out["v"] > 0
    assert out["count"] == 12
    assert os.path.getsize(tmp_path / "job" / "sprite.jpg") > 0
    manifest = json.loads((tmp_path / "job" / "sprite.json").read_text())
    assert manifest["coarse"] is True
    # No temp tile dir left behind.
    assert not (tmp_path / "job" / ".sprite_tiles").exists()


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_coarse_sprite_skips_short_sources(tmp_path):
    from backend.services import filmstrip_generator as fg
    src = tmp_path / "src.mp4"
    _make_video(src, 4)   # << _COARSE_MIN_DURATION (300s)
    assert fg.generate_sprite_coarse(str(src), str(tmp_path / "job")) is None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_coarse_sprite_never_clobbers_a_fine_sprite(tmp_path, monkeypatch):
    from backend.services import filmstrip_generator as fg
    monkeypatch.setattr(fg, "_COARSE_MIN_DURATION", 5.0)
    src = tmp_path / "src.mp4"
    _make_video(src, 20)
    job = tmp_path / "job"
    fine = fg.generate_sprite(str(src), str(job))
    assert fine is not None and "coarse" not in fine
    fine_bytes = os.path.getsize(job / "sprite.jpg")
    assert fg.generate_sprite_coarse(str(src), str(job)) is None
    assert os.path.getsize(job / "sprite.jpg") == fine_bytes
    assert "coarse" not in json.loads((job / "sprite.json").read_text())


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_fine_sprite_replaces_coarse_and_bumps_v(tmp_path, monkeypatch):
    from backend.services import filmstrip_generator as fg
    monkeypatch.setattr(fg, "_COARSE_MIN_DURATION", 5.0)
    src = tmp_path / "src.mp4"
    _make_video(src, 20)
    job = tmp_path / "job"
    coarse = fg.generate_sprite_coarse(str(src), str(job), tiles=12)
    assert coarse is not None
    # v stamps are second-granular; force a visible difference.
    monkeypatch.setattr(fg.time, "time", lambda: coarse["v"] + 5)
    fine = fg.generate_sprite(str(src), str(job))
    assert fine is not None
    manifest = json.loads((job / "sprite.json").read_text())
    assert "coarse" not in manifest
    assert manifest["v"] > coarse["v"]


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_fine_sprite_manifest_carries_v(tmp_path):
    from backend.services import filmstrip_generator as fg
    src = tmp_path / "src.mp4"
    _make_video(src, 4)
    out = fg.generate_sprite(str(src), str(tmp_path / "job"))
    assert out is not None
    assert out["v"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Peaks: memmap WAV fast path + streaming ffmpeg fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_peaks_wav_memmap_path_no_subprocess(tmp_path, monkeypatch):
    from backend.services import filmstrip_generator as fg
    wav = tmp_path / "audio.wav"
    _make_wav(wav, 3.0)

    def _boom(*a, **k):
        raise AssertionError("mono 16-bit WAV must not spawn ffmpeg/ffprobe")
    monkeypatch.setattr(fg, "_run", _boom)
    monkeypatch.setattr(fg.subprocess, "Popen", _boom)

    out = fg.generate_peaks(str(wav), str(tmp_path / "job"), n_peaks=100)
    assert out is not None
    assert out["length"] == 100
    assert abs(out["duration"] - 3.0) < 0.05
    data = out["data"]
    assert len(data) == 200                      # interleaved min/max
    assert min(data) < -0.3 and max(data) > 0.3  # the sine actually registered
    assert os.path.isfile(tmp_path / "job" / "peaks.json")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_peaks_streaming_path_for_stereo_wav(tmp_path):
    # Stereo WAVs skip the memmap (a strided view would copy on reshape) and
    # go through the streaming ffmpeg decode instead.
    from backend.services import filmstrip_generator as fg
    wav = tmp_path / "stereo.wav"
    _make_wav(wav, 2.0, channels=2)
    out = fg.generate_peaks(str(wav), str(tmp_path / "job"), n_peaks=50)
    assert out is not None
    assert out["length"] >= 50
    assert abs(out["duration"] - 2.0) < 0.1


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_peaks_streaming_path_for_video_source(tmp_path):
    from backend.services import filmstrip_generator as fg
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=size=160x90:d=2:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-shortest", str(src)],
        capture_output=True, check=True)
    out = fg.generate_peaks(str(src), str(tmp_path / "job"), n_peaks=50)
    assert out is not None
    assert max(out["data"]) > 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Router: cache semantics + coarse re-kick
# ─────────────────────────────────────────────────────────────────────────────

def _client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.routers import filmstrip as fs
    monkeypatch.setattr(fs, "_JOB_ROOT", str(tmp_path))
    app = FastAPI()
    app.include_router(fs.router)
    return TestClient(app), fs


def test_manifest_served_no_cache_sprite_long_cache(tmp_path, monkeypatch):
    client, fs = _client(tmp_path, monkeypatch)
    job = tmp_path / "job1"
    job.mkdir()
    (job / "sprite.json").write_text(json.dumps({"cols": 12, "v": 1}))
    (job / "sprite.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    r = client.get("/api/jobs/job1/filmstrip.json")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"

    r = client.get("/api/jobs/job1/filmstrip.jpg?v=1")
    assert r.status_code == 200
    assert "max-age=2592000" in r.headers["cache-control"]


def test_coarse_manifest_rekicks_fine_prep(tmp_path, monkeypatch):
    client, fs = _client(tmp_path, monkeypatch)
    job = tmp_path / "job2"
    job.mkdir()
    (job / "sprite.json").write_text(json.dumps({"cols": 12, "coarse": True, "v": 1}))
    (job / "sprite.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    kicked = []
    monkeypatch.setattr(fs, "_kick_prep", lambda job_id: kicked.append(job_id))
    r = client.get("/api/jobs/job2/filmstrip.json")
    assert r.status_code == 200
    assert kicked == ["job2"]


def test_fine_manifest_does_not_rekick(tmp_path, monkeypatch):
    client, fs = _client(tmp_path, monkeypatch)
    job = tmp_path / "job3"
    job.mkdir()
    (job / "sprite.json").write_text(json.dumps({"cols": 12, "v": 2}))
    (job / "sprite.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    kicked = []
    monkeypatch.setattr(fs, "_kick_prep", lambda job_id: kicked.append(job_id))
    r = client.get("/api/jobs/job3/filmstrip.json")
    assert r.status_code == 200
    assert kicked == []
