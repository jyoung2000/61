"""Round-3 pipeline speedups — background work must never slow analysis.

  * Editor asset prep: coarse sprite immediately; the disk-heavy passes
    (faststart remux, fine sprite scan, peaks) WAIT for extraction's
    ``audio.wav`` so they stop fighting frame/audio extraction for the same
    multi-GB source — and peaks then hit the zero-copy WAV memmap instead of
    re-decoding the video's audio track.
  * Background transcodes (browser preview, faststart, sprite/peaks ffmpeg)
    run below normal CPU priority; the CPU x264 preview also caps threads.
  * Companion whisper sidecar is released the moment transcription completes
    so Ollama gets the whole GPU for the LLM phases (translation/polish/SEO).
"""

import asyncio
import os
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# editor_assets: coarse now, heavy passes after extraction
# ─────────────────────────────────────────────────────────────────────────────

def _run_prep(tmp_path, monkeypatch, audio_appears_after_polls):
    """Run prepare_editor_assets with stubbed generators + injected sleep.
    Returns the recorded call order."""
    from backend.services import editor_assets as EA
    from backend.services import filmstrip_generator as FG
    from backend.services import faststart as FS

    calls = []
    audio_path = str(tmp_path / "audio.wav")

    monkeypatch.setattr(FG, "generate_sprite_coarse",
                        lambda v, d: calls.append("coarse"))
    monkeypatch.setattr(FG, "generate_sprite",
                        lambda v, d: calls.append("fine"))
    monkeypatch.setattr(FG, "generate_peaks",
                        lambda src, d: calls.append(f"peaks:{os.path.basename(src)}"))
    monkeypatch.setattr(FS, "ensure_faststart",
                        lambda p: calls.append("faststart"))

    polls = [0]

    async def fake_sleep(_s):
        polls[0] += 1
        calls.append(f"sleep{polls[0]}")
        if audio_appears_after_polls is not None and polls[0] >= audio_appears_after_polls:
            (tmp_path / "audio.wav").write_bytes(b"RIFF")

    asyncio.run(EA.prepare_editor_assets(
        str(tmp_path / "video.mp4"), str(tmp_path), audio_path,
        poll_s=0.01, max_wait_s=0.05 if audio_appears_after_polls is None else 999,
        sleep=fake_sleep))
    return calls, audio_path


def test_prep_waits_for_audio_then_runs_heavy_passes(tmp_path, monkeypatch):
    calls, _ = _run_prep(tmp_path, monkeypatch, audio_appears_after_polls=2)
    # Coarse first, before any waiting; heavy passes strictly after the wait.
    assert calls[0] == "coarse"
    assert "sleep1" in calls and "sleep2" in calls
    tail = calls[calls.index("sleep2") + 1:]
    assert tail == ["faststart", "fine", "peaks:audio.wav"]


def test_prep_peaks_prefer_extracted_audio_wav(tmp_path, monkeypatch):
    calls, _ = _run_prep(tmp_path, monkeypatch, audio_appears_after_polls=1)
    assert "peaks:audio.wav" in calls        # memmap fast path source


def test_prep_gives_up_waiting_and_still_builds(tmp_path, monkeypatch):
    # Extraction never lands audio.wav (crashed run) — the cap expires and
    # assets still build from the source video.
    calls, _ = _run_prep(tmp_path, monkeypatch, audio_appears_after_polls=None)
    assert calls[0] == "coarse"
    assert "faststart" in calls and "fine" in calls
    assert "peaks:video.mp4" in calls        # falls back to the source


def test_prep_skips_nothing_when_audio_already_exists(tmp_path, monkeypatch):
    (tmp_path / "audio.wav").write_bytes(b"RIFF")
    calls, _ = _run_prep(tmp_path, monkeypatch, audio_appears_after_polls=99)
    # No sleeps at all — straight through.
    assert [c for c in calls if c.startswith("sleep")] == []
    assert calls == ["coarse", "faststart", "fine", "peaks:audio.wav"]


# ─────────────────────────────────────────────────────────────────────────────
# Low-priority background transcodes
# ─────────────────────────────────────────────────────────────────────────────

def test_low_priority_kwargs_posix():
    from backend.services.proc_priority import low_priority_popen_kwargs
    kw = low_priority_popen_kwargs()
    if os.name == "posix":
        assert callable(kw.get("preexec_fn"))
    else:  # pragma: no cover
        assert kw == {} or "creationflags" in kw


def test_preview_x264_caps_threads_nvenc_does_not():
    from backend.services import browser_preview as bp
    probe = bp._ProbeResult(
        video_codec="hevc", audio_codec="aac", has_audio=True,
        width=3840, height=2160, fps=30.0, bitrate_kbps=20000,
        keyframe_interval_s=2.0)
    cpu_cmd = bp._build_ffmpeg_cmd("in.mkv", "out.mp4", probe, encoder="libx264")
    assert "-threads" in cpu_cmd
    cap = int(cpu_cmd[cpu_cmd.index("-threads") + 1])
    assert 2 <= cap <= (os.cpu_count() or 8)
    gpu_cmd = bp._build_ffmpeg_cmd("in.mkv", "out.mp4", probe, encoder="h264_nvenc")
    assert "-threads" not in gpu_cmd                 # NVENC needs no cap
    # The 2s-GOP seek contract is untouched.
    assert "-g" in cpu_cmd and "-g" in gpu_cmd


def test_filmstrip_run_uses_low_priority(monkeypatch):
    from backend.services import filmstrip_generator as fg
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        class R:  # noqa: N801 — minimal stand-in
            returncode = 1
            stdout = b""
            stderr = b""
        return R()
    monkeypatch.setattr(fg.subprocess, "run", fake_run)
    fg._run(["ffprobe", "-v", "error"], timeout=5)
    if os.name == "posix":
        assert callable(seen.get("preexec_fn"))


# ─────────────────────────────────────────────────────────────────────────────
# Companion whisper release after transcription
# ─────────────────────────────────────────────────────────────────────────────

def test_release_flag_defaults_on():
    assert settings.WHISPER_REMOTE_RELEASE_AFTER_TRANSCRIBE is True


def _patch_remote(monkeypatch, RA, status_code=200, raise_exc=False):
    import httpx
    posted = {}
    monkeypatch.setattr(RA, "remote_whisper_configured", lambda: True)
    monkeypatch.setattr(RA, "_remote_whisper_base", lambda: "http://companion:11500")
    monkeypatch.setattr(RA, "_remote_whisper_token", lambda: "tok123")

    def fake_post(url, headers=None, timeout=None):
        if raise_exc:
            raise ConnectionError("down")
        posted["url"] = url
        posted["headers"] = headers or {}

        class R:  # noqa: N801
            pass
        R.status_code = status_code
        return R()
    monkeypatch.setattr(httpx, "post", fake_post)
    return posted


def test_release_posts_bearer_to_release_route(monkeypatch):
    from backend.services import reframer_audio as RA
    posted = _patch_remote(monkeypatch, RA, status_code=200)
    assert RA.remote_whisper_release() is True
    assert posted["url"] == "http://companion:11500/v1/sidecar/release"
    assert posted["headers"]["Authorization"] == "Bearer tok123"


def test_release_is_false_on_conflict_or_old_companion(monkeypatch):
    from backend.services import reframer_audio as RA
    _patch_remote(monkeypatch, RA, status_code=409)
    assert RA.remote_whisper_release() is False
    _patch_remote(monkeypatch, RA, status_code=404)   # older Companion build
    assert RA.remote_whisper_release() is False


def test_release_never_raises(monkeypatch):
    from backend.services import reframer_audio as RA
    _patch_remote(monkeypatch, RA, raise_exc=True)
    assert RA.remote_whisper_release() is False


def test_release_skipped_when_unconfigured(monkeypatch):
    from backend.services import reframer_audio as RA
    monkeypatch.setattr(RA, "remote_whisper_configured", lambda: False)
    assert RA.remote_whisper_release() is False
