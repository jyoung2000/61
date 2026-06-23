"""Durable per-job processing-log events.

The Analysis page's PROCESSING LOG was built purely from live WebSocket
messages, so a tab reload, a new device, or a container restart lost the whole
history (React state reset to ``[]`` and only future events were received).

This module persists each broadcast event as one JSON line in the job's own
directory (``/data/uploads/{job_id}/events.jsonl``) so the log can be re-fetched
in full and survives restarts. It is deliberately:

  * **Append-only** — a one-line append is cheap and never rewrites the (often
    multi-MB) ``job.json``; the heavy-serialization stalls we hit elsewhere came
    from rewriting big payloads, not from tiny appends.
  * **Filtered + throttled** — pure keepalive ``heartbeat`` pings are skipped and
    rapid same-stage progress ticks are throttled, so the durable log carries
    meaningful events (stage changes, warnings, completions) rather than noise,
    and the read window isn't flooded by keepalives.
  * **Best-effort** — any failure is swallowed; logging must never break a job.

Kept dependency-light (stdlib only) so it imports without the heavy pipeline /
ASR stack and is unit-testable.
"""

from __future__ import annotations

import json
import os
import time as _time

# Where job directories live (mirrors ``backend.database._job_dir``). Overridable
# via env so tests can point it at a temp dir without importing the DB layer.
EVENTS_BASE_DIR = os.environ.get("CLIPAI_UPLOADS_DIR", "/data/uploads")
EVENTS_FILENAME = "events.jsonl"

# Pure keepalive / transport message types — never worth persisting.
_SKIP_TYPES = frozenset({"heartbeat", "pong", "ping"})

# Minimum seconds between persisted ``status`` events that share the SAME stage —
# collapses the every-few-seconds progress ticks ("Extracted N frames",
# "Preconditioning audio N%") into a bounded trail. A stage CHANGE or a terminal
# ``complete`` is never throttled.
_STATUS_THROTTLE_S = 8.0

# Most-recent events returned to a client on hydration.
DEFAULT_READ_LIMIT = 2000

# One-time compaction so a very long run can't grow the file without bound.
_COMPACT_AT_LINES = 6000
_COMPACT_KEEP = 3000

# Per-job in-memory state (lost on restart — that's fine; the file persists and
# throttling simply restarts fresh).
_last_status: dict = {}     # job_id -> (stage_id, monotonic_ts)
_line_counts: dict = {}     # job_id -> approx appended line count this process


def _events_path(job_id: str) -> str:
    return os.path.join(EVENTS_BASE_DIR, job_id, EVENTS_FILENAME)


def should_persist(job_id: str, message: dict) -> bool:
    """Decide whether ``message`` is worth writing to the durable log.

    Skips keepalive types and throttles repeated same-stage ``status`` ticks.
    Has the side effect of recording the last-seen stage/time for throttling.
    """
    mtype = str(message.get("type", "") or "")
    if mtype in _SKIP_TYPES:
        return False
    if mtype in ("status", "complete"):
        stage = str(message.get("stage_id", "") or "")
        now = _time.monotonic()
        prev = _last_status.get(job_id)
        if mtype == "status" and prev is not None:
            prev_stage, prev_t = prev
            if prev_stage == stage and (now - prev_t) < _STATUS_THROTTLE_S:
                return False
        _last_status[job_id] = (stage, now)
    return True


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def append_job_event(job_id: str, message: dict) -> None:
    """Append one event to the job's durable log (best-effort, synchronous).

    MUST contain no ``await`` so it runs atomically within the asyncio event
    loop — no other coroutine can interleave a half-written line.
    """
    try:
        if not job_id or not isinstance(message, dict):
            return
        if not should_persist(job_id, message):
            return
        ev = dict(message)
        ev.setdefault("ts", _time.time())   # wall-clock epoch seconds
        path = _events_path(job_id)
        directory = os.path.dirname(path)
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        line = json.dumps(ev, ensure_ascii=False, default=str)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        # Track size so a long run gets compacted once, not re-counted per write.
        n = _line_counts.get(job_id)
        if n is None:
            n = _count_lines(path)
        n += 1
        if n >= _COMPACT_AT_LINES:
            _compact(path)
            n = _COMPACT_KEEP
        _line_counts[job_id] = n
    except Exception:
        # Never let logging break the pipeline.
        pass


def _compact(path: str) -> None:
    """Rewrite ``path`` keeping only the most-recent ``_COMPACT_KEEP`` lines."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) <= _COMPACT_KEEP:
            return
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(lines[-_COMPACT_KEEP:])
        os.replace(tmp, path)
    except Exception:
        pass


def load_job_events(job_id: str, limit: int = DEFAULT_READ_LIMIT) -> list:
    """Return up to ``limit`` most-recent persisted events (oldest → newest)."""
    path = _events_path(job_id)
    out: list = []
    try:
        if not os.path.isfile(path):
            return out
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        for ln in lines[-max(1, int(limit)):]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return out
    return out
