"""Regression tests for defects found in the 2026-07-10 live-run logs.

  * Engine checkpoint saved with an EMPTY source SHA → verify-after-save
    always failed (a signature without a SHA can never be loaded). The save
    now refuses upfront, and the pipeline awaits the background hash before
    building the signature.
  * Filmstrip sprite ffmpeg wrote to "sprite.jpg.tmp" — ffmpeg cannot infer
    an output muxer from ".tmp", so EVERY sprite attempt failed (rc=234) and
    long videos got no scrub filmstrip. The tmp name now keeps the .jpg
    extension.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint: empty source SHA is refused loudly, not written-then-rejected
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_save_refuses_empty_source_sha(tmp_path, monkeypatch):
    from backend.services import pipeline_checkpoint as pc
    from types import SimpleNamespace

    monkeypatch.setattr(pc, "checkpoint_dir", lambda job_id: str(tmp_path / "ckpt"))
    sig = pc.checkpoint_signature(
        source_sha="", source_language="ja", sample_fps=1.25,
        aspect_ratio="9:16")
    ok = asyncio.run(pc.save_engine_checkpoint(
        "job-x", SimpleNamespace(), SimpleNamespace(), signature=sig))
    assert ok is False
    assert not (tmp_path / "ckpt").exists()   # nothing written at all


def test_checkpoint_save_verifies_with_real_sha(tmp_path, monkeypatch):
    from backend.services import pipeline_checkpoint as pc
    from types import SimpleNamespace
    import dataclasses

    monkeypatch.setattr(pc, "checkpoint_dir", lambda job_id: str(tmp_path / "ckpt"))
    monkeypatch.setattr(pc, "shared_cache_dir", lambda sig: str(tmp_path / "shared"))

    perception = SimpleNamespace(
        src_w=1920, src_h=1080, fps=30.0, duration_ms=10_000, total_frames=300,
        scene_cuts=[0], transcript_segments=[{"start": 0, "end": 1, "text": "hi"}],
        detected_language="ja",
        face_timeline={0: []}, motion_timeline={}, motion_hotspot={},
        speaker_timeline={}, track_speaker_map={}, speech_active={},
        audio_rms={}, audio_events={}, person_timeline={}, saliency_hotspot={},
    )

    @dataclasses.dataclass
    class _Plan:
        ops: list

    sig = pc.checkpoint_signature(source_sha="ab" * 32, source_language="ja",
                                  sample_fps=1.25, aspect_ratio="9:16",
                                  asr_model="large-v3-turbo")
    ok = asyncio.run(pc.save_engine_checkpoint(
        "job-x", perception, _Plan(ops=[]), signature=sig))
    assert ok is True                          # save + verify round-trip


# ─────────────────────────────────────────────────────────────────────────────
# Filmstrip sprite: ffmpeg-compatible tmp naming, atomic publish
# ─────────────────────────────────────────────────────────────────────────────

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_sprite_generates_with_jpg_tmp_naming(tmp_path):
    from backend.services.filmstrip_generator import generate_sprite
    src = str(tmp_path / "src.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:d=4:rate=10",
         src], capture_output=True, check=True)
    out_dir = str(tmp_path / "job")
    manifest = generate_sprite(src, out_dir)
    assert manifest is not None
    assert os.path.isfile(os.path.join(out_dir, "sprite.jpg"))
    assert os.path.getsize(os.path.join(out_dir, "sprite.jpg")) > 0
    # No temp artifacts left behind (either naming generation).
    leftovers = [f for f in os.listdir(out_dir) if ".tmp" in f]
    assert leftovers == []
