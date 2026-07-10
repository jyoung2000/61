"""Pipeline speed round 2 — quality-identical scheduling wins.

Three changes are pinned here:

1. ATOMIC shared audio: ``extract_audio`` streams into ``audio.wav.part.wav``
   and publishes the complete file via os.replace(), and the transcription
   thread's sibling-reuse WAITS on the in-flight marker instead of paying a
   SECOND full preconditioning pass (highpass+afftdn+loudnorm — minutes of
   duplicated CPU on long videos). This is what makes the extraction↔perceive
   overlap (PIPELINE_OVERLAP_EXTRACTION, now default ON) race-free.

2. Auto-SEO fan-out: the per-clip SEO loop runs a bounded-concurrency gather
   (SEO_PARALLEL_MAX) instead of N serial LLM round-trips — same prompts,
   same caps, same skip rules, same clip order.

3. Translation-cleanup waves: the batched prefill runs a small wave of
   independent batches concurrently (TRANSLATION_LLM_CLEANUP_CONCURRENCY)
   with the wall-clock budget re-checked between waves.
"""

import asyncio
import os
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# 1. Shared-audio readiness wait (pure logic, injected sleep)
# ─────────────────────────────────────────────────────────────────────────────

from backend.services.reframer_audio import _wait_for_sibling_audio


def _write(path, size=2048):
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


def test_wait_ready_immediately(tmp_path):
    final = str(tmp_path / "audio.wav")
    _write(final)
    calls = []
    assert _wait_for_sibling_audio(final, final + ".part.wav",
                                   sleep=calls.append) is True
    assert not calls  # no waiting when the file is already there


def test_wait_polls_while_part_exists_then_succeeds(tmp_path):
    final = str(tmp_path / "audio.wav")
    part = final + ".part.wav"
    _write(part)
    ticks = {"n": 0}

    def _sleep(_s):
        # After two polls the extractor "finishes": part → final (atomic).
        ticks["n"] += 1
        if ticks["n"] == 2:
            _write(final)
            os.remove(part)

    assert _wait_for_sibling_audio(final, part, duration_ms=60_000,
                                   sleep=_sleep) is True
    assert ticks["n"] >= 2


def test_wait_part_vanishes_without_final_falls_back(tmp_path):
    final = str(tmp_path / "audio.wav")
    part = final + ".part.wav"
    _write(part)

    def _sleep(_s):
        if os.path.exists(part):
            os.remove(part)   # extraction died; no final ever appears

    assert _wait_for_sibling_audio(final, part, sleep=_sleep) is False


def test_wait_nothing_appears_gives_up_after_grace(tmp_path):
    final = str(tmp_path / "audio.wav")
    slept = []
    out = _wait_for_sibling_audio(final, final + ".part.wav",
                                  grace_s=6.0, poll_s=2.0,
                                  sleep=slept.append)
    assert out is False
    # Gave up shortly after the grace window — not the duration-scaled cap.
    assert 3 <= len(slept) <= 5


def test_wait_tiny_final_file_not_treated_as_ready(tmp_path):
    final = str(tmp_path / "audio.wav")
    _write(final, size=100)   # under the 1024-byte sanity floor
    assert _wait_for_sibling_audio(final, final + ".part.wav",
                                   grace_s=2.0, poll_s=1.0,
                                   sleep=lambda s: None) is False


def test_overlap_extraction_default_on():
    assert settings.PIPELINE_OVERLAP_EXTRACTION is True


# ─────────────────────────────────────────────────────────────────────────────
# 1b. extract_audio writes atomically (real ffmpeg, tiny synthetic clip)
# ─────────────────────────────────────────────────────────────────────────────

import shutil
import subprocess

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_extract_audio_atomic_no_part_left(tmp_path, monkeypatch):
    from backend.services.frame_extractor import extract_audio
    src = str(tmp_path / "src.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
         "-shortest", src],
        capture_output=True, check=True)
    out = str(tmp_path / "audio.wav")
    result = asyncio.run(extract_audio(src, out, precondition=True,
                                       video_duration=2.0))
    assert result == out
    assert os.path.isfile(out) and os.path.getsize(out) > 1024
    assert not os.path.exists(out + ".part.wav")   # marker cleaned up


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
def test_extract_audio_failure_cleans_marker(tmp_path):
    from backend.services.frame_extractor import extract_audio
    src = str(tmp_path / "not_a_video.mp4")
    _write(src)   # garbage input → ffmpeg fails on both attempts
    out = str(tmp_path / "audio.wav")
    with pytest.raises(RuntimeError):
        asyncio.run(extract_audio(src, out, precondition=True,
                                  video_duration=2.0))
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".part.wav")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Auto-SEO bounded fan-out
# ─────────────────────────────────────────────────────────────────────────────

def _stub_clip(i, platform="tiktok", with_seo=False):
    d = {
        "id": i, "title": f"Clip {i}", "platform": platform,
        "start_time": float(i), "end_time": float(i) + 5.0,
        "suggested_caption": f"caption {i}",
        "seo_by_platform": {}, "seo_title": "",
    }
    if with_seo:
        d["seo_by_platform"] = {platform: {"title": "already", "description": "d",
                                           "tags": [], "platform_tips": ""}}
    return d


def test_auto_seo_runs_concurrently_and_preserves_order(monkeypatch):
    from backend.services import pipeline as P
    from backend.models import ClipSEO

    clips = [_stub_clip(i) for i in range(8)]
    # Clip 3 already has SEO → must be skipped, not regenerated.
    clips[3] = _stub_clip(3, with_seo=True)

    class _Job:
        summary = None
        def __init__(self):
            self.clips = clips
    async def _load_job(job_id):
        return _Job()
    saved = {}
    async def _update(job_id, **kw):
        saved.update(kw)
    monkeypatch.setattr(P.database, "load_job", _load_job)
    monkeypatch.setattr(P.database, "update_job_status", _update)
    async def _no_ws(*a, **k):
        pass
    monkeypatch.setattr(P, "broadcast_ws", _no_ws)
    monkeypatch.setattr(P, "set_heartbeat_stage", lambda *a, **k: None)
    monkeypatch.setattr(settings, "SEO_PARALLEL_MAX", 3)
    monkeypatch.setattr(settings, "LIVE_TRENDS_ENABLED", False)

    state = {"active": 0, "peak": 0, "calls": 0}

    class _FakeOrch:
        def __init__(self, *a, **k):
            pass
        async def generate_seo(self, clip_title="", clip_transcript="",
                               video_summary="", platform="", job_id=""):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"] += 1
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return (ClipSEO(title=f"SEO {clip_title}", description="d",
                            tags=["#x"]), "fake")

    import backend.services.ai_orchestrator as AO
    monkeypatch.setattr(AO, "AIOrchestrator", _FakeOrch)

    generated, failed, updated = asyncio.run(
        P._auto_generate_clip_seo("job-x", [], None, fallback_clips=None,
                                  output_language="en"))

    assert failed == 0
    assert generated == 7                       # 8 clips minus 1 pre-SEO'd
    assert state["calls"] == 7
    assert 2 <= state["peak"] <= 3              # genuinely concurrent, bounded
    # Order preserved and the pre-SEO'd clip untouched.
    assert [c["id"] for c in updated] == list(range(8))
    assert updated[3]["seo_by_platform"]["tiktok"]["title"] == "already"
    assert updated[0]["seo_title"] == "SEO Clip 0"


def test_auto_seo_per_clip_failure_is_isolated(monkeypatch):
    from backend.services import pipeline as P
    from backend.models import ClipSEO

    clips = [_stub_clip(i) for i in range(4)]

    class _Job:
        summary = None
        def __init__(self):
            self.clips = clips
    async def _load_job(job_id):
        return _Job()
    async def _update(job_id, **kw):
        pass
    monkeypatch.setattr(P.database, "load_job", _load_job)
    monkeypatch.setattr(P.database, "update_job_status", _update)
    async def _no_ws(*a, **k):
        pass
    monkeypatch.setattr(P, "broadcast_ws", _no_ws)
    monkeypatch.setattr(P, "set_heartbeat_stage", lambda *a, **k: None)
    monkeypatch.setattr(settings, "LIVE_TRENDS_ENABLED", False)

    class _FlakyOrch:
        def __init__(self, *a, **k):
            pass
        async def generate_seo(self, clip_title="", **k):
            if clip_title == "Clip 2":
                raise RuntimeError("provider exploded")
            return (ClipSEO(title="ok", description="d"), "fake")

    import backend.services.ai_orchestrator as AO
    monkeypatch.setattr(AO, "AIOrchestrator", _FlakyOrch)

    generated, failed, updated = asyncio.run(
        P._auto_generate_clip_seo("job-x", [], None))
    assert generated == 3 and failed == 1
    assert updated[2]["seo_title"] == ""        # failed clip kept, un-SEO'd


# ─────────────────────────────────────────────────────────────────────────────
# 3. Translation-cleanup prefill waves
# ─────────────────────────────────────────────────────────────────────────────

def test_prefill_waves_run_concurrently_and_merge(monkeypatch):
    from backend.services import pipeline as P

    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_BATCH_CUES", 2)
    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_CONCURRENCY", 3)

    state = {"active": 0, "peak": 0}

    class _Orch:
        async def text_completion(self, prompt, **kw):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            # Echo back a valid JSON map for the numbered lines in the prompt.
            n = prompt.count("\n") - prompt.count("\n\n")  # crude but stable
            import json as _json
            lines = [l for l in prompt.splitlines() if l[:2].rstrip(".").isdigit()]
            return _json.dumps({str(i + 1): f"EN {i}" for i in range(len(lines))})

    texts = [f"cue {i}" for i in range(8)]     # 4 batches of 2 → 2 waves of 3+1
    out = asyncio.run(P._batch_prefill_translations(
        _Orch(), texts, "ja", "English", "", "job-x", None))
    assert len(out) == 8
    assert state["peak"] >= 2                   # batches genuinely overlapped


def test_prefill_budget_checked_between_waves(monkeypatch):
    from backend.services import pipeline as P
    import time as _t

    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_BATCH_CUES", 1)
    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_CONCURRENCY", 2)

    calls = {"n": 0}

    class _Orch:
        async def text_completion(self, prompt, **kw):
            calls["n"] += 1
            return '{"1": "EN"}'

    deadline = _t.monotonic() - 1               # already over budget
    out = asyncio.run(P._batch_prefill_translations(
        _Orch(), [f"c{i}" for i in range(6)], "ja", "English", "",
        "job-x", deadline))
    assert out == {} and calls["n"] == 0        # stopped before any wave


def test_prefill_batch_exception_does_not_kill_other_batches(monkeypatch):
    from backend.services import pipeline as P

    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_BATCH_CUES", 1)
    monkeypatch.setattr(settings, "TRANSLATION_LLM_CLEANUP_CONCURRENCY", 4)

    class _Orch:
        async def text_completion(self, prompt, **kw):
            if "cue 1" in prompt:
                raise RuntimeError("boom")
            return '{"1": "EN ok"}'

    out = asyncio.run(P._batch_prefill_translations(
        _Orch(), ["cue 0", "cue 1", "cue 2"], "ja", "English", "",
        "job-x", None))
    assert "cue 0" in out and "cue 2" in out and "cue 1" not in out
