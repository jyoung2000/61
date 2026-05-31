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


def is_editor_state_corrupt(state: dict) -> tuple[bool, str]:
    """Detect a corrupt NLE editor-state (the server-side mirror of the
    frontend validator).

    Corruption from the reverse-sync bug shows as subtitle ``items`` that are
    out of chronological order, duplicated en masse, or backwards
    (end < start). Such a cached state must never be served back — it would
    re-poison the transcript. Returns ``(corrupt, reason)``.
    """
    if not isinstance(state, dict):
        return False, ""
    items = state.get("items")
    if not isinstance(items, list):
        return False, ""
    subs = [
        it for it in items
        if isinstance(it, dict)
        and it.get("type") == "subtitle"
        and str(it.get("subtitleText") or "").strip()
    ]
    if not subs:
        return False, ""

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # 1) Backwards / invalid cues.
    for s in subs:
        st, en = _num(s.get("start")), _num(s.get("end"))
        if st is None or en is None or en < st - 0.001:
            return True, f"backwards/invalid cue (start={s.get('start')} end={s.get('end')})"

    # 2) Out-of-order cues.
    for i in range(1, len(subs)):
        prev, cur = _num(subs[i - 1].get("start")), _num(subs[i].get("start"))
        if prev is not None and cur is not None and cur < prev - 0.05:
            return True, f"out-of-order cue at index {i}"

    # 3) Mass-duplicated (start, text) cues.
    seen: dict = {}
    dupes = 0
    for s in subs:
        st = _num(s.get("start")) or 0.0
        key = (round(st * 10), str(s.get("subtitleText") or "").strip())
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            dupes += 1
    if dupes >= 5 and dupes > len(subs) * 0.1:
        return True, f"{dupes}/{len(subs)} duplicate cues"

    return False, ""
