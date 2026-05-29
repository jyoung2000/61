"""Transcript ↔ NLE-timeline sync helpers.

The transcript is the single source of truth for subtitle timing + text. The
transcript editor and the NLE timeline both read from it and write back to it,
so an edit in either surface (element timing/words on the timeline, or text in
the transcript panel) shows up everywhere — the panel, the timeline, and the
SRT / VTT / TXT downloads — and persists to ``job.json`` across restarts.

Kept dependency-light (only ``backend.models``) so the endpoint logic is unit
-testable without importing the heavy router/provider import chain.
"""

from __future__ import annotations

from backend.models import TranscriptSegment


def clean_and_sort_segments(segments: list) -> list[dict]:
    """Coerce timeline/editor segments into a clean, chronological transcript.

    - Coerces dicts → TranscriptSegment (dropping anything unparseable).
    - Drops blank-text and backwards (end < start) cues so a corrupt timeline
      edit can't poison the transcript.
    - Sorts chronologically (SRT/VTT require it; the readability + gap passes
      assume it).

    Returns a list of plain dicts ready to persist on the job.
    """
    coerced: list[TranscriptSegment] = []
    for s in (segments or []):
        try:
            seg = TranscriptSegment(**s) if isinstance(s, dict) else s
        except Exception:
            continue
        if not (seg.text or "").strip():
            continue
        if seg.end < seg.start:
            continue
        coerced.append(seg)
    coerced.sort(key=lambda x: (x.start, x.end))
    return [s.model_dump() for s in coerced]
