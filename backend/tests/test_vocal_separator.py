"""Tests for the Demucs vocal-separation helper. Pure-logic only — the actual
Demucs/ffmpeg subprocesses are never invoked here (no model/GPU in CI), so we
cover command construction, output-path resolution, device ordering, and the
self-healing early returns."""

import os
import sys

from backend.services import vocal_separator as vs


# ── _demucs_cmd ─────────────────────────────────────────────────────────────

def test_demucs_cmd_basic_shape():
    cmd = vs._demucs_cmd("/in.wav", "/out", "htdemucs", "cpu", segment=None)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "demucs"]
    assert "--two-stems" in cmd and cmd[cmd.index("--two-stems") + 1] == "vocals"
    assert cmd[cmd.index("-n") + 1] == "htdemucs"
    assert cmd[cmd.index("-o") + 1] == "/out"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[-1] == "/in.wav"


def test_demucs_cmd_segment_only_on_cuda():
    # --segment caps CUDA VRAM; it's pointless (and we omit it) on CPU.
    cpu = vs._demucs_cmd("/in.wav", "/out", "htdemucs", "cpu", segment=7)
    assert "--segment" not in cpu
    cuda = vs._demucs_cmd("/in.wav", "/out", "htdemucs", "cuda", segment=7)
    assert cuda[cuda.index("--segment") + 1] == "7"


# ── _vocals_output_path ─────────────────────────────────────────────────────

def test_vocals_output_path():
    p = vs._vocals_output_path("/work", "htdemucs", "/work/demucs_input.wav")
    assert p == os.path.join("/work", "htdemucs", "demucs_input", "vocals.wav")


# ── _device_order ───────────────────────────────────────────────────────────

def test_device_order_explicit(monkeypatch):
    assert vs._device_order("cpu") == ["cpu"]
    assert vs._device_order("cuda") == ["cuda", "cpu"]


def test_device_order_auto_follows_cuda(monkeypatch):
    monkeypatch.setattr(vs, "_cuda_available", lambda: True)
    assert vs._device_order("auto") == ["cuda", "cpu"]
    monkeypatch.setattr(vs, "_cuda_available", lambda: False)
    assert vs._device_order("auto") == ["cpu"]


# ── separate_vocals: self-healing early returns ─────────────────────────────

def test_separate_vocals_none_when_demucs_absent(monkeypatch):
    monkeypatch.setattr(vs, "is_available", lambda: False)
    # Must short-circuit BEFORE touching ffmpeg/subprocess.
    monkeypatch.setattr(vs, "_run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("_run must not be called when demucs is absent")))
    assert vs.separate_vocals("/some/video.mp4", "/tmp/work_x") is None


def test_separate_vocals_none_when_source_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "is_available", lambda: True)
    missing = str(tmp_path / "does_not_exist.mp4")
    assert vs.separate_vocals(missing, str(tmp_path / "work")) is None


def test_separate_vocals_none_when_extract_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(vs, "is_available", lambda: True)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00\x00")  # exists, but ffmpeg (mocked) "fails"
    monkeypatch.setattr(vs, "_run", lambda cmd, timeout: (1, b"boom"))
    assert vs.separate_vocals(str(src), str(tmp_path / "work")) is None
