"""Multi-select transcript download: one ZIP with a .srt/.txt per selected
video (translated track preferred), skipped videos listed in _skipped.txt,
404 only when nothing at all has a transcript."""

import io
import zipfile
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.jobs as J


def _client():
    app = FastAPI()
    app.include_router(J.router)
    return TestClient(app)


def _job(job_id, filename, transcript=None, translated=None):
    return SimpleNamespace(
        job_id=job_id, filename=filename,
        transcript=transcript or [],
        translated_transcript=translated or [],
    )


_SEGS = [
    {"start": 0.0, "end": 2.0, "text": "Hello there.", "speaker": "Speaker 1"},
    {"start": 2.0, "end": 4.0, "text": "General Kenobi!", "speaker": "Speaker 2"},
]


@pytest.fixture()
def _jobs(monkeypatch):
    jobs = {}

    async def load_job(job_id):
        return jobs.get(job_id)
    monkeypatch.setattr(J.database, "load_job", load_job)

    async def fps(job):
        return None
    monkeypatch.setattr(J, "_effective_fps", fps)

    async def clean(job):
        return job.translated_transcript
    monkeypatch.setattr(J, "_clean_translated_rows", clean)
    return jobs


def _zip_names(resp):
    return sorted(zipfile.ZipFile(io.BytesIO(resp.content)).namelist())


def test_archive_zips_srt_per_selected_video(_jobs):
    _jobs["a"] = _job("a", "first clip.mp4", transcript=_SEGS)
    _jobs["b"] = _job("b", "second.mkv", transcript=_SEGS)
    r = _client().post("/api/jobs/transcripts/archive",
                       json={"job_ids": ["a", "b"], "format": "srt"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert 'filename="clipai-transcripts-srt.zip"' in r.headers["content-disposition"]
    names = _zip_names(r)
    assert names == ["first clip.srt", "second.srt"]
    body = zipfile.ZipFile(io.BytesIO(r.content)).read("second.srt").decode()
    assert "General Kenobi!" in body and "-->" in body


def test_archive_txt_includes_speakers_and_prefers_translation(_jobs):
    translated = [
        {"start": 0.0, "end": 2.0, "text": "Bonjour.", "speaker": "Speaker 1"},
    ]
    _jobs["a"] = _job("a", "clip.mp4", transcript=_SEGS, translated=translated)
    r = _client().post("/api/jobs/transcripts/archive",
                       json={"job_ids": ["a"], "format": "txt"})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.namelist() == ["clip_translated.txt"]
    assert z.read("clip_translated.txt").decode() == "Speaker 1: Bonjour.\n"


def test_archive_skips_transcriptless_videos_with_manifest(_jobs):
    _jobs["a"] = _job("a", "good.mp4", transcript=_SEGS)
    _jobs["b"] = _job("b", "still-analyzing.mp4")          # no transcript yet
    r = _client().post("/api/jobs/transcripts/archive",
                       json={"job_ids": ["a", "b", "ghost"], "format": "txt"})
    assert r.status_code == 200
    names = _zip_names(r)
    assert names == ["_skipped.txt", "good.txt"]
    manifest = zipfile.ZipFile(io.BytesIO(r.content)).read("_skipped.txt").decode()
    assert "still-analyzing.mp4" in manifest and "ghost" in manifest


def test_archive_dedupes_same_filename(_jobs):
    _jobs["a"] = _job("a", "clip.mp4", transcript=_SEGS)
    _jobs["b"] = _job("b", "clip.mp4", transcript=_SEGS)   # same upload name twice
    r = _client().post("/api/jobs/transcripts/archive",
                       json={"job_ids": ["a", "b"], "format": "srt"})
    assert _zip_names(r) == ["clip (2).srt", "clip.srt"]


def test_archive_404_when_nothing_has_a_transcript(_jobs):
    _jobs["a"] = _job("a", "empty.mp4")
    r = _client().post("/api/jobs/transcripts/archive",
                       json={"job_ids": ["a"], "format": "srt"})
    assert r.status_code == 404


def test_archive_validates_inputs(_jobs):
    c = _client()
    assert c.post("/api/jobs/transcripts/archive",
                  json={"job_ids": ["a"], "format": "docx"}).status_code == 400
    assert c.post("/api/jobs/transcripts/archive",
                  json={"job_ids": [], "format": "srt"}).status_code == 400
