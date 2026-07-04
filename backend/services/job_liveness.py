"""Job liveness — the pure staleness math behind revive/reconcile.

A live analysis run leaves two freshness signals on the job record:

  * ``updated_at``    — stamped by every real progress write.
  * ``heartbeat_at``  — stamped ~once a minute by the pipeline heartbeat,
    even during long silent stages (a 40-minute Whisper pass on CPU can
    legitimately go that long between progress writes).

``job_age_seconds`` returns the age of the FRESHEST signal, so revive
thresholds can be minutes instead of the old 30/120-minute guesses
without ever touching an in-flight run. Kept dependency-free (no
fastapi/uvicorn) so the reconcile contract is unit-testable in the same
image as the database tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _parse_age_seconds(ts: str) -> Optional[float]:
    ts = (ts or "").strip()
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def job_age_seconds(job) -> Optional[float]:
    """Seconds since the job's freshest liveness signal (``updated_at`` or
    ``heartbeat_at``); ``None`` when neither is parseable.

    A small age means SOMETHING is still advancing or heart-beating this
    job; a large age means no worker owns it (the run died)."""
    ages = [
        a for a in (
            _parse_age_seconds(getattr(job, "updated_at", "") or ""),
            _parse_age_seconds(getattr(job, "heartbeat_at", "") or ""),
        ) if a is not None
    ]
    return min(ages) if ages else None


def is_stale(job, stale_after_s: float) -> bool:
    """True when the job has had no liveness signal for ``stale_after_s``
    (or has none at all). ``stale_after_s <= 0`` means 'no staleness
    requirement' — used at startup, where every worker thread is already
    dead."""
    if stale_after_s <= 0:
        return True
    age = job_age_seconds(job)
    return age is None or age >= stale_after_s
