"""Tests for durable per-job PROCESSING LOG persistence (services.job_events).

The Analysis page's log was lost on tab reload / new device / container restart
because it was built only from live WebSocket messages. These pin that broadcast
events are now persisted append-only (and re-fetchable), that keepalive
heartbeats are skipped, and that rapid same-stage progress ticks are throttled so
the durable log stays meaningful and bounded.
"""

import os
import tempfile

from backend.services import job_events


def _fresh(tmp):
    """Point the module at a clean temp uploads dir and reset throttle state."""
    job_events.EVENTS_BASE_DIR = tmp
    job_events._last_status.clear()
    job_events._line_counts.clear()


def test_append_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 0  # disable throttle for this test
        job_events.append_job_event("job1", {"type": "status", "message": "A", "stage_id": "metadata"})
        job_events.append_job_event("job1", {"type": "status", "message": "B", "stage_id": "metadata"})
        evs = job_events.load_job_events("job1")
        assert [e["message"] for e in evs] == ["A", "B"]
        # A wall-clock ts is stamped on every event.
        assert all(isinstance(e.get("ts"), (int, float)) for e in evs)
        # Stored on disk in the job's own directory.
        assert os.path.isfile(os.path.join(tmp, "job1", "events.jsonl"))


def test_heartbeats_are_not_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 0
        job_events.append_job_event("j", {"type": "heartbeat", "message": "Still processing…"})
        job_events.append_job_event("j", {"type": "status", "message": "real", "stage_id": "x"})
        job_events.append_job_event("j", {"type": "pong"})
        evs = job_events.load_job_events("j")
        assert [e["message"] for e in evs] == ["real"]


def test_same_stage_status_is_throttled_but_stage_change_and_complete_pass():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 9999  # nothing "ages out" within the test
        job_events.append_job_event("j", {"type": "status", "message": "frames 1", "stage_id": "extraction"})
        job_events.append_job_event("j", {"type": "status", "message": "frames 2", "stage_id": "extraction"})  # throttled
        job_events.append_job_event("j", {"type": "status", "message": "whisper", "stage_id": "transcription"})  # stage change
        job_events.append_job_event("j", {"type": "complete", "message": "done", "stage_id": "transcription"})   # terminal
        msgs = [e["message"] for e in job_events.load_job_events("j")]
        assert msgs == ["frames 1", "whisper", "done"]


def test_non_status_events_are_never_throttled():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 9999
        for i in range(3):
            job_events.append_job_event("j", {"type": "background_task", "task": "subtitle_translation",
                                              "status": "running", "message": f"batch {i}"})
        msgs = [e["message"] for e in job_events.load_job_events("j")]
        assert msgs == ["batch 0", "batch 1", "batch 2"]


def test_load_returns_only_the_most_recent_limit():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 0
        for i in range(50):
            job_events.append_job_event("j", {"type": "status", "message": f"m{i}", "stage_id": "s"})
        evs = job_events.load_job_events("j", limit=5)
        assert [e["message"] for e in evs] == ["m45", "m46", "m47", "m48", "m49"]


def test_missing_job_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        assert job_events.load_job_events("nope") == []


def test_compaction_caps_file_growth():
    with tempfile.TemporaryDirectory() as tmp:
        _fresh(tmp)
        job_events._STATUS_THROTTLE_S = 0
        job_events._COMPACT_AT_LINES = 100
        job_events._COMPACT_KEEP = 40
        for i in range(250):
            job_events.append_job_event("j", {"type": "status", "message": f"m{i}", "stage_id": "s"})
        with open(os.path.join(tmp, "j", "events.jsonl"), encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        # Never far above the compaction ceiling.
        assert len(lines) <= job_events._COMPACT_AT_LINES
        # The most-recent event always survived.
        assert "m249" in lines[-1]
        # restore defaults for any later import users
        job_events._COMPACT_AT_LINES = 6000
        job_events._COMPACT_KEEP = 3000
