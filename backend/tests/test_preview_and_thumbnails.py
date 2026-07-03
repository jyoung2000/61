"""Preview-proxy triggers + single-decode scene thumbnails.

Real-world failure this codifies: 640x360 H.264 MP4s passed every
_needs_preview check, so the RAW long-GOP web rip was served to the
editor — every scrub decoded from a distant keyframe and the player
looked frozen. And bridge_conversion spent 167s running one ffmpeg
seek per scene (224 scenes) for thumbnails.
"""

import os
import struct
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import pytest

from backend.services import browser_preview as BP  # noqa: E402


def _probe(**kw):
    defaults = dict(video_codec="h264", audio_codec="aac", has_audio=True,
                    width=640, height=360, fps=23.98, bitrate_kbps=400,
                    keyframe_interval_s=2.0, faststart=True)
    defaults.update(kw)
    return BP._ProbeResult(**defaults)


# ── _needs_preview triggers ─────────────────────────────────────────────

def test_compatible_dense_gop_needs_no_preview():
    assert BP._needs_preview("/x/video.mp4", _probe()) is False


def test_sparse_keyframes_trigger_preview():
    # The exact profile of the two logged runs: small h264 mp4, long GOP
    assert BP._needs_preview("/x/video.mp4",
                             _probe(keyframe_interval_s=8.0)) is True


def test_missing_faststart_triggers_preview():
    assert BP._needs_preview("/x/video.mp4", _probe(faststart=False)) is True


def test_unknown_keyint_does_not_trigger():
    assert BP._needs_preview("/x/video.mp4",
                             _probe(keyframe_interval_s=0.0)) is False


def test_sparse_gop_forbids_stream_copy():
    # Copying would carry the sparse GOP into the "preview" — pointless
    assert BP._should_copy_video(_probe(keyframe_interval_s=8.0)) is False
    assert BP._should_copy_video(_probe(keyframe_interval_s=2.0)) is True


def test_preview_path_version_bumped():
    """v4 bump invalidates the v3 '.none' sentinels written for exactly the
    files the new keyint trigger must now rebuild."""
    p = BP._preview_path_for("/data/uploads/j1/video.mp4")
    assert p.endswith("browser_preview.v4.mp4")
    assert BP._none_marker_for("/data/uploads/j1/video.mp4").endswith(".v4.mp4.none")


# ── faststart sniffing on synthetic MP4 atom layouts ────────────────────

def _atom(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def test_probe_faststart_moov_first(tmp_path):
    f = tmp_path / "fast.mp4"
    f.write_bytes(_atom(b"ftyp", b"isom") + _atom(b"moov", b"\0" * 32)
                  + _atom(b"mdat", b"\0" * 64))
    assert BP._probe_faststart(str(f)) is True


def test_probe_faststart_mdat_first(tmp_path):
    f = tmp_path / "slow.mp4"
    f.write_bytes(_atom(b"ftyp", b"isom") + _atom(b"mdat", b"\0" * 64)
                  + _atom(b"moov", b"\0" * 32))
    assert BP._probe_faststart(str(f)) is False


def test_probe_faststart_non_mp4_is_true(tmp_path):
    f = tmp_path / "x.mkv"
    f.write_bytes(b"\x1a\x45\xdf\xa3" + b"\0" * 100)
    assert BP._probe_faststart(str(f)) is True


# ── encoder command variants ────────────────────────────────────────────

def test_build_cmd_nvenc_variant():
    cmd = BP._build_ffmpeg_cmd("/in.mp4", "/out.mp4",
                               _probe(keyframe_interval_s=8.0),
                               encoder="h264_nvenc")
    joined = " ".join(cmd)
    assert "h264_nvenc" in joined
    assert "-g" in cmd  # dense-GOP contract holds on the GPU path too
    assert "+faststart" in joined


def test_build_cmd_default_is_x264():
    cmd = BP._build_ffmpeg_cmd("/in.mp4", "/out.mp4",
                               _probe(keyframe_interval_s=8.0))
    assert "libx264" in cmd
    assert "-sc_threshold" in cmd


def test_early_preview_scheduled_in_pipeline():
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    assert "Early browser-preview build" in src
    assert "ensure_browser_preview" in src


# ── single-decode scene thumbnails ──────────────────────────────────────

def test_batch_thumbnails_single_ffmpeg_run(tmp_path, monkeypatch):
    from backend.services import reframer_bridge as RB

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # Emulate the select filter writing one jpg per between() clause
        vf = cmd[cmd.index("-vf") + 1]
        n = vf.count("between(")
        pattern = cmd[-1]
        d = os.path.dirname(pattern)
        for i in range(1, n + 1):
            with open(os.path.join(d, f"t_{i:06d}.jpg"), "wb") as fh:
                fh.write(b"\xff\xd8fakejpg")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(RB.subprocess, "run", fake_run)
    ts = [float(i * 10) for i in range(12)]
    outs = [str(tmp_path / f"scene_{i:04d}.jpg") for i in range(12)]
    written = RB._extract_thumbnails_batch("/video.mp4", ts, outs)
    assert written == 12
    assert len(calls) == 1, "12 thumbnails must be ONE ffmpeg invocation"
    for o in outs:
        assert os.path.getsize(o) > 0


def test_batch_thumbnails_chunking(monkeypatch, tmp_path):
    from backend.services import reframer_bridge as RB
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        vf = cmd[cmd.index("-vf") + 1]
        n = vf.count("between(")
        d = os.path.dirname(cmd[-1])
        for i in range(1, n + 1):
            open(os.path.join(d, f"t_{i:06d}.jpg"), "wb").write(b"x")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(RB.subprocess, "run", fake_run)
    n = 450  # > 2 chunks of 200
    ts = [float(i) for i in range(n)]
    outs = [str(tmp_path / f"s_{i:04d}.jpg") for i in range(n)]
    assert RB._extract_thumbnails_batch("/v.mp4", ts, outs) == n
    assert len(calls) == 3


def test_per_scene_extraction_is_fallback_only():
    import inspect
    from backend.services import reframer_bridge as RB
    src = inspect.getsource(RB.to_fez_scenes)
    assert "_extract_thumbnails_batch" in src
    # per-scene seek only runs when the batch missed the file
    assert "os.path.exists(thumb)" in src
