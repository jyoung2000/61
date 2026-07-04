"""Tests for the speed & efficiency pass (quality-neutral).

Covers:
  * Task 1 — job.json write-amplification: in-memory cache, coalesced
    progress-only persists (JOB_PROGRESS_FLUSH_INTERVAL), compact JSON
    (JOB_JSON_PRETTY).
  * Task 2 — source hashed exactly once per run, off the critical path.
  * Task 4 — sidecar serialization content-equality (fastjson vs stdlib).
  * Task 8 — fastjson round-trip equality + stdlib fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

import backend.database as db
from backend.config import settings
from backend.models import JobResult, JobStatus, TranscriptSegment


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mk_seg(text="hello") -> TranscriptSegment:
    return TranscriptSegment(start=0, end=1, text=text, speaker="Speaker 1")


@pytest.fixture()
def tmp_job_store(monkeypatch):
    """Throwaway on-disk job store + clean cache/debounce state."""
    d = tempfile.mkdtemp()
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    db._reset_job_cache_for_tests()
    yield d
    db._reset_job_cache_for_tests()


def _read_raw(job_id: str) -> dict:
    with open(db._job_path(job_id), "rb") as f:
        return json.loads(f.read())


# ── Task 1a: in-memory cache ─────────────────────────────────────────


def test_load_serves_cache_and_mtime_change_invalidates(tmp_job_store):
    async def scenario():
        job = JobResult(job_id="a", filename="f", file_path="p",
                        status=JobStatus.QUEUED.value, progress=1)
        await db.save_job(job)
        first = await db.load_job("a")
        again = await db.load_job("a")
        # Unchanged file → the SAME parsed object is served (no re-parse).
        assert again is first

        # Out-of-band edit (different mtime) must invalidate the entry.
        path = db._job_path("a")
        raw = _read_raw("a")
        raw["progress"] = 77
        with open(path, "w") as f:
            json.dump(raw, f)
        os.utime(path, (os.path.getmtime(path) + 5, os.path.getmtime(path) + 5))
        fresh = await db.load_job("a")
        assert fresh is not first
        assert fresh.progress == 77
    _run(scenario())


def test_cache_is_bounded(tmp_job_store):
    async def scenario():
        for i in range(db._JOB_CACHE_MAX + 4):
            await db.save_job(JobResult(job_id=f"j{i}", filename="f",
                                        file_path="p"))
        assert len(db._job_cache) <= db._JOB_CACHE_MAX
    _run(scenario())


def test_delete_job_invalidates_cache(tmp_job_store):
    async def scenario():
        await db.save_job(JobResult(job_id="del1", filename="f", file_path="p"))
        await db.load_job("del1")
        path = db._job_path("del1")
        assert path in db._job_cache
        await db.delete_job("del1")
        assert path not in db._job_cache
        assert await db.load_job("del1") is None
    _run(scenario())


# ── Task 1b: coalesced progress-only persists ────────────────────────


def test_progress_only_update_defers_disk_write(tmp_job_store, monkeypatch):
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 30.0)

    async def scenario():
        job = JobResult(job_id="b", filename="f", file_path="p",
                        status=JobStatus.TRANSCRIBING.value, progress=10)
        await db.save_job(job)
        await db.update_job_status(
            "b", status=JobStatus.TRANSCRIBING.value, progress=42,
            progress_message="Transcribing...")
        # Readers see the fresh progress immediately (via the cache)…
        live = await db.load_job("b")
        assert live.progress == 42
        # …but the disk write was debounced.
        assert _read_raw("b")["progress"] == 10
    _run(scenario())


def test_field_carrying_update_writes_through(tmp_job_store, monkeypatch):
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 30.0)

    async def scenario():
        await db.save_job(JobResult(job_id="c", filename="f", file_path="p",
                                    status=JobStatus.TRANSCRIBING.value, progress=10))
        await db.update_job_status("c", progress=55, transcript=[_mk_seg()])
        raw = _read_raw("c")
        assert raw["progress"] == 55
        assert len(raw["transcript"]) == 1
    _run(scenario())


def test_status_change_and_terminal_write_through(tmp_job_store, monkeypatch):
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 30.0)

    async def scenario():
        await db.save_job(JobResult(job_id="d", filename="f", file_path="p",
                                    status=JobStatus.TRANSCRIBING.value, progress=10))
        # A status CHANGE is not progress-only.
        await db.update_job_status("d", status=JobStatus.ANALYZING_SCENES.value,
                                   progress=20)
        assert _read_raw("d")["status"] == "analyzing_scenes"
        # Debounce some progress, then a terminal write must land (carrying
        # the debounced progress with it — same object).
        await db.update_job_status("d", status=JobStatus.ANALYZING_SCENES.value,
                                   progress=33, progress_message="x")
        assert _read_raw("d")["progress"] == 20  # deferred
        await db.update_job_status("d", status=JobStatus.COMPLETE.value,
                                   progress=100)
        raw = _read_raw("d")
        assert raw["status"] == "complete" and raw["progress"] == 100
    _run(scenario())


def test_trailing_flush_lands(tmp_job_store, monkeypatch):
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 0.15)

    async def scenario():
        await db.save_job(JobResult(job_id="e", filename="f", file_path="p",
                                    status=JobStatus.TRANSCRIBING.value, progress=10))
        await db.update_job_status("e", progress=64, progress_message="working")
        if _read_raw("e")["progress"] == 64:
            # The interval elapsed before the update (slow CI) — the write
            # went through directly, which is also correct.
            return
        await asyncio.sleep(0.6)
        assert _read_raw("e")["progress"] == 64
    _run(scenario())


def test_zero_interval_disables_debounce(tmp_job_store, monkeypatch):
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 0.0)

    async def scenario():
        await db.save_job(JobResult(job_id="g", filename="f", file_path="p",
                                    status=JobStatus.TRANSCRIBING.value, progress=10))
        await db.update_job_status("g", progress=90)
        assert _read_raw("g")["progress"] == 90
    _run(scenario())


def test_stale_snapshot_guard_survives_cache_and_debounce(tmp_job_store, monkeypatch):
    """The test_job_persistence_race pattern with the cache + debounce live:
    a guarded stale progress relay must not revert COMPLETE, and readers
    (cache included) must see the terminal state afterwards."""
    monkeypatch.setattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 30.0)

    async def scenario():
        await db.save_job(JobResult(job_id="h", filename="f", file_path="p",
                                    status=JobStatus.DETECTING_CLIPS.value))
        await db.update_job_status(
            "h", status=JobStatus.DETECTING_CLIPS.value, progress=96,
            progress_message="Finalizing clip detection...",
            protect_terminal=True)  # progress-only → debounced
        await db.update_job_status(
            "h", status=JobStatus.COMPLETE.value, progress=100,
            translated_transcript=[_mk_seg()])
        await db.update_job_status(
            "h", status=JobStatus.DETECTING_CLIPS.value, progress=97,
            progress_message="stale relay", protect_terminal=True)
        cached = await db.load_job("h")
        raw = _read_raw("h")
        assert str(cached.status) == str(JobStatus.COMPLETE)
        assert raw["status"] == "complete" and raw["progress"] == 100
        assert len(raw["translated_transcript"]) == 1
    _run(scenario())


# ── Task 1c: compact vs pretty JSON ──────────────────────────────────


def test_job_json_compact_by_default_pretty_when_enabled(tmp_job_store, monkeypatch):
    async def scenario():
        await db.save_job(JobResult(job_id="k", filename="f", file_path="p"))
        with open(db._job_path("k"), "rb") as f:
            compact = f.read()
        assert b"\n  " not in compact  # no indent
        monkeypatch.setattr(settings, "JOB_JSON_PRETTY", True)
        await db.save_job(JobResult(job_id="k2", filename="f", file_path="p"))
        with open(db._job_path("k2"), "rb") as f:
            pretty = f.read()
        assert pretty.startswith(b"{\n")
        # Both parse to equivalent records.
        a, b = json.loads(compact), json.loads(pretty)
        for key in ("filename", "file_path", "status"):
            assert a[key] == b[key]
    _run(scenario())


# ── Task 2: one source hash per run, off the critical path ───────────


def test_cache_probe_uses_precomputed_sha_without_hashing(tmp_path, monkeypatch):
    from backend.services import pipeline_helpers as ph
    from backend.models import FrameData

    src = tmp_path / "video.mp4"
    src.write_bytes(b"source bytes" * 64)
    real_sha = ph._hash_file_sha256(str(src))
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(6):
        (frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"jpg")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"riff" * 16)
    ph._write_extraction_manifest(
        str(frames_dir),
        [FrameData(timestamp=i * 1.0, path=f"frame_{i:04d}.jpg") for i in range(6)],
        [2.0])

    calls = {"n": 0}
    def _counting_hash(path, **kw):
        calls["n"] += 1
        return real_sha
    monkeypatch.setattr(ph, "_hash_file_sha256", _counting_hash)

    # Precomputed digest → hit with ZERO hash invocations.
    result = asyncio.run(ph._maybe_use_cached_extraction(
        job_id="j", video_path=str(src), frames_dir=str(frames_dir),
        audio_path=str(audio), expected_sha=real_sha, precomputed_sha=real_sha))
    assert result is not None and calls["n"] == 0

    # Legacy path (no precomputed) still hashes exactly once.
    result = asyncio.run(ph._maybe_use_cached_extraction(
        job_id="j", video_path=str(src), frames_dir=str(frames_dir),
        audio_path=str(audio), expected_sha=real_sha))
    assert result is not None and calls["n"] == 1


def test_background_hash_task_hashes_once(monkeypatch, tmp_path):
    import backend.services.pipeline as pl

    src = tmp_path / "v.mp4"
    src.write_bytes(b"x" * 128)
    calls = {"n": 0}

    def _counting_hash(path, **kw):
        calls["n"] += 1
        return "deadbeef" * 8

    monkeypatch.setattr(pl, "_hash_file_sha256", _counting_hash)

    async def scenario():
        task = pl._start_source_hash_task("jobH", str(src))
        assert task is not None
        # Awaited by the cache probe AND by the post-extraction persist —
        # both consumers get the single result.
        sha1 = await pl._await_source_hash("jobH", task, str(src))
        sha2 = await pl._await_source_hash("jobH", task, str(src))
        assert sha1 == sha2 == "deadbeef" * 8
        pl._bg_hash_tasks.pop("jobH", None)
        return calls["n"]

    assert asyncio.run(scenario()) == 1


def test_await_source_hash_falls_back_inline(monkeypatch, tmp_path):
    import backend.services.pipeline as pl
    src = tmp_path / "v.mp4"
    src.write_bytes(b"y" * 64)
    calls = {"n": 0}

    def _counting_hash(path, **kw):
        calls["n"] += 1
        return "cafe" * 16

    monkeypatch.setattr(pl, "_hash_file_sha256", _counting_hash)

    async def scenario():
        # No task (start failed) → inline hash, exactly once.
        return await pl._await_source_hash("jobF", None, str(src))

    assert asyncio.run(scenario()) == "cafe" * 16
    assert calls["n"] == 1


def test_checkpoint_signature_shape_unchanged():
    from backend.services.pipeline_checkpoint import checkpoint_signature, CHECKPOINT_VERSION
    sig = checkpoint_signature(
        source_sha="ab" * 32, source_language="ja", sample_fps=1.25,
        aspect_ratio="9:16", vocal_sep_key="False:htdemucs",
        planner_fingerprint="fp")
    assert sig == {
        "version": CHECKPOINT_VERSION,
        "source_sha": "ab" * 32,
        "source_language": "ja",
        "sample_fps": 1.25,
        "aspect_ratio": "9:16",
        "vocal_sep_key": "False:htdemucs",
        "planner_fingerprint": "fp",
    }


# ── Tasks 4 + 8: fastjson round-trips (sidecars, job.json, checkpoint) ─


def _representative_job_dump() -> dict:
    job = JobResult(
        job_id="rt", filename="f.mp4", file_path="/p/f.mp4",
        status=JobStatus.COMPLETE.value, progress=100,
        transcript=[_mk_seg("hello"), _mk_seg("world")],
        translated_transcript=[_mk_seg("bonjour")],
    )
    return job.model_dump(mode="json")


def test_fastjson_roundtrip_matches_stdlib_on_job_dump():
    from backend.services import fastjson
    data = _representative_job_dump()
    via_fast = fastjson.loads(fastjson.dumps_bytes(data, default=db._numpy_safe_default))
    via_std = json.loads(json.dumps(data, default=db._numpy_safe_default))
    assert via_fast == via_std == data


def test_fastjson_roundtrip_int_keyed_perception_dict():
    from backend.services import fastjson
    from backend.services.pipeline_checkpoint import _json_safe
    perception_like = _json_safe({
        "face_timeline": {0: [{"cx": 1, "cy": 2}], 1000: []},
        "motion_timeline": {0: 0.5, 1000: 0.25},
        "scene_cuts": [0, 1000],
    })
    via_fast = fastjson.loads(fastjson.dumps_bytes(perception_like))
    via_std = json.loads(json.dumps(perception_like))
    assert via_fast == via_std
    assert "1000" in via_fast["face_timeline"]  # int keys stringified, both paths


def test_fastjson_numpy_payloads():
    np = pytest.importorskip("numpy")
    from backend.services import fastjson
    data = {"arr": np.arange(3), "f": np.float32(1.5), "i": np.int64(7)}
    parsed = fastjson.loads(fastjson.dumps_bytes(data, default=db._numpy_safe_default))
    assert parsed["arr"] == [0, 1, 2]
    assert parsed["f"] == 1.5
    assert parsed["i"] == 7


def test_fastjson_stdlib_fallback(monkeypatch):
    from backend.services import fastjson
    monkeypatch.setattr(fastjson, "_orjson", None)
    data = _representative_job_dump()
    payload = fastjson.dumps_bytes(data, default=db._numpy_safe_default)
    assert isinstance(payload, bytes)
    assert fastjson.loads(payload) == data
    # Compact separators — identical to the stdlib arm's contract.
    assert b", " not in payload[:200] or b": " not in payload[:200]


def test_checkpoint_roundtrip_with_fastjson(monkeypatch, tmp_path):
    """Engine checkpoint save→load round-trips through fastjson with int
    timeline keys restored (the _int_keyed contract)."""
    from backend.services import pipeline_checkpoint as pc
    from types import SimpleNamespace

    monkeypatch.setattr(pc, "checkpoint_dir",
                        lambda job_id: str(tmp_path / "ckpt"))

    class _FakePlan:
        pass

    perception = SimpleNamespace(
        src_w=1920, src_h=1080, fps=30.0, duration_ms=10_000, total_frames=300,
        scene_cuts=[0, 4000], transcript_segments=[{"start": 0, "end": 1, "text": "hi"}],
        detected_language="en",
        face_timeline={0: [{"cx": 5}], 2000: []}, motion_timeline={0: 0.1},
        motion_hotspot={}, speaker_timeline={}, track_speaker_map={},
        speech_active={}, audio_rms={}, audio_events={}, person_timeline={},
        saliency_hotspot={},
    )
    sig = pc.checkpoint_signature(source_sha="cd" * 32, source_language="en",
                                  sample_fps=2.0, aspect_ratio="9:16")

    import dataclasses

    @dataclasses.dataclass
    class _Plan:
        ops: list

    plan = _Plan(ops=[{"t": 0, "x": 1}])

    # RenderPlan.load is exercised elsewhere; stub it to isolate the JSON layer.
    class _RP:
        @staticmethod
        def load(path):
            with open(path) as f:
                return json.load(f)
    import backend.services.reframer_models as rm
    monkeypatch.setattr(rm, "RenderPlan", _RP)

    assert pc._save_sync("jobC", perception, plan, sig, {"device": "cpu"})
    out = pc._load_sync("jobC", sig)
    assert out is not None
    restored, plan_loaded, stub = out
    assert restored.face_timeline == {0: [{"cx": 5}], 2000: []}
    assert restored.motion_timeline == {0: 0.1}
    assert plan_loaded == {"ops": [{"t": 0, "x": 1}]}
    assert stub._resumed_from_checkpoint is True
