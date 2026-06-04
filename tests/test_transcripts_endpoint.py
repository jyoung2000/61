"""Test the lightweight /jobs/{id}/transcripts endpoint.

It exists so the UI can load the translated transcript reliably even when the
full job payload is too large/slow to fetch over a tunnel — the failure that
left the on-screen transcript stuck on the source language across every client.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

jobs_router = pytest.importorskip("backend.routers.jobs")


def _job():
    return SimpleNamespace(
        status="complete",
        translation_status="translated",
        subtitle_language="en",
        language="ja",
        transcript=[{"text": "こんにちは", "start": 0.0, "end": 1.0, "speaker": "Speaker 1"}],
        translated_transcript=[
            {"text": "Hello", "start": 0.0, "end": 1.0, "speaker": "Speaker 1"},
            {"text": "World", "start": 1.0, "end": 2.0, "speaker": "Speaker 1"},
        ],
        speaker_names={"Speaker 1": "Heero"},
        speaker_colors={},
    )


def test_transcripts_endpoint_returns_translated(monkeypatch):
    job = _job()

    async def _access(job_id, user):
        return job

    monkeypatch.setattr(jobs_router, "_require_job_access", _access)
    out = asyncio.run(jobs_router.get_transcripts("job1", user=None))

    assert out["status"] == "complete"
    assert out["translation_status"] == "translated"
    assert len(out["translated_transcript"]) == 2
    assert out["translated_transcript"][0]["text"] == "Hello"   # English, not the source
    assert len(out["transcript"]) == 1
    assert out["speaker_names"] == {"Speaker 1": "Heero"}


def test_transcripts_endpoint_empty_translation(monkeypatch):
    job = _job()
    job.translated_transcript = []

    async def _access(job_id, user):
        return job

    monkeypatch.setattr(jobs_router, "_require_job_access", _access)
    out = asyncio.run(jobs_router.get_transcripts("job1", user=None))
    assert out["translated_transcript"] == []
    assert len(out["transcript"]) == 1
