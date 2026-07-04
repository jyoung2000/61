"""Stage checkpoints (translation) + judge verdict cache — the post-engine
resume layer. Same contract as the engine checkpoint: exact signature match
or re-run, atomic writes, best-effort failure behavior."""

import asyncio
import json
import os

import pytest

from backend.services import stage_checkpoints as sc
from backend.services.reframer_clipper import _JudgeVerdictCache


def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture()
def job_dir(tmp_path, monkeypatch):
    # checkpoint_dir() hardcodes /data/uploads/<job>; point it at tmp.
    monkeypatch.setattr(
        "backend.services.pipeline_checkpoint.checkpoint_dir",
        lambda job_id: str(tmp_path / job_id / "checkpoint"))
    monkeypatch.setattr(
        "backend.services.stage_checkpoints.checkpoint_dir",
        lambda job_id: str(tmp_path / job_id / "checkpoint"))
    return tmp_path


SIG = {"src_hash": "abc123", "source_lang": "ja", "target_lang": "en",
       "models": "ollama=qwen3"}


def test_stage_checkpoint_roundtrip(job_dir):
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
               "readability": {"grade": "A"}}
    assert _aiorun(sc.save_stage_checkpoint("j1", "translation", payload,
                                            signature=SIG)) is True
    loaded = _aiorun(sc.load_stage_checkpoint("j1", "translation", signature=SIG))
    assert loaded == payload


def test_stage_checkpoint_signature_mismatch_misses(job_dir):
    _aiorun(sc.save_stage_checkpoint("j2", "translation", {"segments": [1]},
                                     signature=SIG))
    other = dict(SIG, target_lang="de")
    assert _aiorun(sc.load_stage_checkpoint("j2", "translation",
                                            signature=other)) is None
    # An empty identity value on the expected side is never trusted.
    empty = dict(SIG, src_hash="")
    assert _aiorun(sc.load_stage_checkpoint("j2", "translation",
                                            signature=empty)) is None


def test_stage_checkpoint_absent_and_corrupt(job_dir):
    assert _aiorun(sc.load_stage_checkpoint("nope", "translation",
                                            signature=SIG)) is None
    # Corrupt file falls back to re-running the stage, never raises.
    d = job_dir / "j3" / "checkpoint"
    d.mkdir(parents=True)
    (d / "translation.json").write_text("{not json")
    assert _aiorun(sc.load_stage_checkpoint("j3", "translation",
                                            signature=SIG)) is None


def test_transcript_hash_tracks_content():
    a = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    b = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    c = [{"start": 0.0, "end": 1.0, "text": "goodbye"}]
    d = [{"start": 0.5, "end": 1.0, "text": "hello"}]
    assert sc.transcript_hash(a) == sc.transcript_hash(b)
    assert sc.transcript_hash(a) != sc.transcript_hash(c)
    assert sc.transcript_hash(a) != sc.transcript_hash(d)
    assert sc.transcript_hash([]) == sc.transcript_hash(None)


# ── Judge verdict cache ──────────────────────────────────────────────


class _Cand:
    def __init__(self, start_s, end_s, transcript_slice="talk"):
        self.start_s = start_s
        self.end_s = end_s
        self.transcript_slice = transcript_slice


def test_verdict_cache_roundtrip(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    cache = _JudgeVerdictCache(str(video), "gpt-x → fallback")
    c = _Cand(10.0, 25.0)
    assert cache.get(c) is None
    verdict = {"verdict": "keep", "title": "Great bit",
               "hook": 8, "payoff": 7}
    cache.put(c, verdict)
    cache.flush()

    # A fresh instance (new run) with the SAME judge spec replays it…
    again = _JudgeVerdictCache(str(video), "gpt-x → fallback")
    assert again.get(_Cand(10.0, 25.0)) == verdict
    # …and a different window or transcript misses.
    assert again.get(_Cand(10.0, 30.0)) is None
    assert again.get(_Cand(10.0, 25.0, "other words")) is None


def test_verdict_cache_invalidated_by_judge_spec(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    cache = _JudgeVerdictCache(str(video), "judge-a")
    cache.put(_Cand(1.0, 5.0), {"verdict": "keep"})
    cache.flush()
    other = _JudgeVerdictCache(str(video), "judge-b")
    assert other.get(_Cand(1.0, 5.0)) is None


def test_verdict_cache_never_stores_errors(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    cache = _JudgeVerdictCache(str(video), "j")
    cache.put(_Cand(1.0, 5.0), {"error": "429"})
    cache.put(_Cand(1.0, 5.0), None)
    assert cache.get(_Cand(1.0, 5.0)) is None


def test_verdict_cache_corrupt_file_starts_empty(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"x")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "judge_verdicts.json").write_text("{broken")
    cache = _JudgeVerdictCache(str(video), "j")
    assert cache.get(_Cand(1.0, 5.0)) is None
    # And it can still write afterwards.
    cache.put(_Cand(1.0, 5.0), {"verdict": "keep"})
    cache.flush()
    data = json.loads((ckpt / "judge_verdicts.json").read_text())
    assert data["judge"] == "j" and len(data["verdicts"]) == 1
