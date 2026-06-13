"""Regression tests for the translate-step-reliability fix.

These cover the decoupling of subtitle translation from clip extraction and
the fail-loud behaviour, exercising the units that don't need the heavy ML
stack:

  * ``_background_post_processing`` returns an outcome contract, sets a visible
    ``translation_status``, and NEVER pins ``status=COMPLETE`` (the job is no
    longer terminal at that point — clips run after it).
  * A translation FAILURE falls back to the source transcript, surfaces a
    ``translation_failed`` status + reason, and never relabels source as
    translated.
  * The no-translation path returns the (deduped) source for the COMPLETE save.
  * ``_run_post_clip_followups`` only refreshes captions when a translation
    actually happened, and always seeds Auto-SEO from the right transcript.
  * ``_set_translation_status`` persists the job field + broadcasts.

Each test mocks the DB / websocket / translator so the suite runs without
touching ``/data`` or any model weights.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Auth/store modules resolve their data dir from HOME at import time.
_TMP = Path(tempfile.mkdtemp(prefix="clipai_translate_test_"))
os.environ["HOME"] = str(_TMP)

from backend.models import JobStatus, TranscriptSegment  # noqa: E402
import backend.services.pipeline as pipeline  # noqa: E402
import backend.services.translator as translator  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────


def _run(coro):
    """Run a coroutine on a fresh loop WITHOUT clobbering the process-global
    event loop. ``asyncio.run`` closes the loop and unsets the current loop,
    which breaks sibling test modules that use the legacy
    ``asyncio.get_event_loop()`` pattern when co-run. Setting a fresh, open loop
    as current here keeps those modules working regardless of collection order.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _seg(text, start, end, speaker="Speaker 1"):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


class _FakeDB:
    """Minimal stand-in for ``backend.database`` capturing every write."""

    def __init__(self, initial=None):
        self.fields = dict(initial or {})
        self.update_calls = []

    async def update_job_status(self, job_id, **kwargs):
        self.update_calls.append(kwargs)
        self.fields.update(kwargs)

    async def load_job(self, job_id):
        return SimpleNamespace(**self.fields)


def _install_common(monkeypatch, db, *, engine="llm"):
    """Patch DB, websocket and the translator engine resolver."""
    broadcasts = []

    async def _bcast(job_id, message):
        broadcasts.append(message)

    monkeypatch.setattr(pipeline.database, "update_job_status", db.update_job_status)
    monkeypatch.setattr(pipeline.database, "load_job", db.load_job)
    monkeypatch.setattr(pipeline, "broadcast_ws", _bcast)
    monkeypatch.setattr(translator, "_resolve_translation_engine", lambda s, t: engine)
    # The editorial-LLM and Whisper-native →English paths now run BEFORE offline
    # NMT for non-English → English jobs; disable both here so these tests
    # deterministically exercise the text/NMT path they stub via
    # ``translate_segments_with_fallback``.
    monkeypatch.setattr(pipeline.settings, "TRANSLATION_PREFER_LLM", False, raising=False)
    monkeypatch.setattr(pipeline.settings, "WHISPER_TRANSLATE_TO_EN", False, raising=False)
    # Keep optional LLM/segmentation passes out of the unit under test.
    monkeypatch.setattr(pipeline.settings, "AI_TRANSCRIPT_CORRECTION", False, raising=False)
    monkeypatch.setattr(pipeline.settings, "SENTENCE_SEGMENTATION_ENABLED", False, raising=False)
    monkeypatch.setattr(pipeline.settings, "SUBTITLE_CPS_ENFORCEMENT", False, raising=False)
    monkeypatch.setattr(pipeline.settings, "TRANSLATION_GLOSSARY_ENABLED", False, raising=False)
    return broadcasts


def _fake_orchestrator():
    return SimpleNamespace(
        reset_circuit_breaker=lambda: None,
        get_editorial_model_info=lambda: {"is_thinking": False},
    )


# ── _background_post_processing: success ─────────────────────────────


def test_translation_success_sets_status_and_no_complete_pin(monkeypatch):
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    broadcasts = _install_common(monkeypatch, db)

    async def _fake_translate(segments, **kwargs):
        # Return genuine target-language (non-CJK) text so changed>0, translation
        # "succeeds", AND the final purity gate (which rejects source-script
        # output) passes. Prefixing the Japanese source would leave CJK behind.
        _EN = {"こんにちは": "Hello", "世界": "World"}
        return [
            TranscriptSegment(text=_EN.get(s.text, "Text"), start=s.start, end=s.end, speaker=s.speaker)
            for s in segments
        ]

    monkeypatch.setattr(translator, "translate_segments_with_fallback", _fake_translate)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[], summary=None)
    transcript = [_seg("こんにちは", 0.0, 1.0), _seg("世界", 1.0, 2.0)]

    result = _run(
        pipeline._background_post_processing(
            "job1", transcript, _fake_orchestrator(), job, polished_already=False,
        )
    )

    assert result["will_translate"] is True
    assert result["translated"] is True
    assert result["target_transcript"] and result["target_transcript"][0]["text"] == "Hello"
    assert result["seo_transcript"] == result["target_transcript"]
    # The source transcript is preserved (untouched) for the `transcript` field.
    assert result["source_transcript"][0]["text"] == "こんにちは"

    # translated_transcript was persisted...
    assert any("translated_transcript" in c for c in db.update_calls)
    # ...and translation_status was set to "translated"...
    assert db.fields.get("translation_status") == "translated"
    # ...and NO update ever pinned status=COMPLETE (clips still run after this).
    assert all(c.get("status") != JobStatus.COMPLETE for c in db.update_calls)
    assert all("status" not in c for c in db.update_calls)


# ── _background_post_processing: failure → source fallback ───────────


def test_translation_failure_falls_back_and_flags(monkeypatch):
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    broadcasts = _install_common(monkeypatch, db)

    async def _boom(segments, **kwargs):
        raise translator.TranslationRateLimitedError("429 less than $5.0 in credit")

    monkeypatch.setattr(translator, "translate_segments_with_fallback", _boom)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[], summary=None)
    transcript = [_seg("こんにちは", 0.0, 1.0), _seg("世界", 1.0, 2.0)]

    result = _run(
        pipeline._background_post_processing(
            "job2", transcript, _fake_orchestrator(), job, polished_already=False,
        )
    )

    assert result["will_translate"] is True
    assert result["translated"] is False
    assert result["failed_reason"]  # a concrete reason was recorded
    # Falls back to the SOURCE transcript for both the `transcript` field + SEO.
    assert result["source_transcript"][0]["text"] == "こんにちは"
    assert result["seo_transcript"] == result["source_transcript"]
    assert result["target_transcript"] is None
    # Never relabelled source as translated.
    assert not any("translated_transcript" in c for c in db.update_calls)
    # Visible, persisted failure state.
    assert db.fields.get("translation_status") == "translation_failed"
    assert db.fields.get("translation_error")
    # A websocket failure event with the translation_failed state was emitted.
    assert any(m.get("state") == "translation_failed" for m in broadcasts)
    # No premature COMPLETE pin.
    assert all(c.get("status") != JobStatus.COMPLETE for c in db.update_calls)


# ── _background_post_processing: no translation planned ──────────────


def test_no_translation_returns_source(monkeypatch):
    db = _FakeDB({"subtitle_language": "", "language": "en", "clips": [], "summary": None})
    _install_common(monkeypatch, db)

    # translate_segments_with_fallback must never be called on this path.
    async def _must_not_call(*a, **k):
        raise AssertionError("translation ran for a non-translating job")

    monkeypatch.setattr(translator, "translate_segments_with_fallback", _must_not_call)

    job = SimpleNamespace(subtitle_language="", language="en", clips=[], summary=None)
    transcript = [_seg("hello", 0.0, 1.0), _seg("world", 1.0, 2.0)]

    result = _run(
        pipeline._background_post_processing(
            "job3", transcript, _fake_orchestrator(), job, polished_already=True,
        )
    )
    assert result["will_translate"] is False
    assert result["translated"] is False
    assert result["source_transcript"][0]["text"] == "hello"
    assert result["seo_transcript"] == result["source_transcript"]
    # No translation planned → translation_status stays unset.
    assert db.fields.get("translation_status") is None


# ── _run_post_clip_followups ─────────────────────────────────────────


def test_followups_refresh_only_when_translated(monkeypatch):
    refresh_calls = []
    seo_calls = []

    async def _fake_refresh(job_id, translated, fallback_clips=None):
        refresh_calls.append(list(translated))
        return len(translated)

    async def _fake_seo(job_id, transcript, orchestrator, fallback_clips=None):
        seo_calls.append(list(transcript))
        return (len(transcript), 0)

    async def _bcast(job_id, message):
        pass

    monkeypatch.setattr(pipeline, "_refresh_clips_with_translation", _fake_refresh)
    monkeypatch.setattr(pipeline, "_auto_generate_clip_seo", _fake_seo)
    monkeypatch.setattr(pipeline, "broadcast_ws", _bcast)

    target = [_seg("EN: hi", 0.0, 1.0)]
    source = [_seg("こんにちは", 0.0, 1.0)]

    # Translated job → refresh runs against target, SEO against seo_transcript.
    pp_translated = {
        "translated": True, "target_transcript": target,
        "seo_transcript": target, "target_name": "English",
    }
    _run(pipeline._run_post_clip_followups("j", _fake_orchestrator(), pp_translated, ["c1"]))
    assert refresh_calls == [target]
    assert seo_calls == [target]

    refresh_calls.clear()
    seo_calls.clear()

    # Non-translated job → NO refresh, SEO seeds from source.
    pp_source = {
        "translated": False, "target_transcript": None,
        "seo_transcript": source, "target_name": "",
    }
    _run(pipeline._run_post_clip_followups("j", _fake_orchestrator(), pp_source, ["c1"]))
    assert refresh_calls == []
    assert seo_calls == [source]


# ── _set_translation_status ──────────────────────────────────────────


def test_set_translation_status_persists_and_broadcasts(monkeypatch):
    db = _FakeDB()
    broadcasts = []

    async def _bcast(job_id, message):
        broadcasts.append(message)

    monkeypatch.setattr(pipeline.database, "update_job_status", db.update_job_status)
    monkeypatch.setattr(pipeline, "broadcast_ws", _bcast)

    _run(pipeline._set_translation_status("j", "translation_failed", "rate-limited"))
    assert db.fields.get("translation_status") == "translation_failed"
    assert db.fields.get("translation_error") == "rate-limited"
    assert any(m.get("type") == "translation_status" for m in broadcasts)
    assert any(m.get("state") == "translation_failed" for m in broadcasts)
