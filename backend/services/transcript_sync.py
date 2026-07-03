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
    # Drop verbatim duplicates at the SAME position (same normalized text and
    # ~same start) — the signature of a subtitle track that was backfilled
    # twice into the same timeline. Duplicates at different times are handled
    # by detect_union_write at the endpoint (they need the stored track to
    # judge), never here — real dialogue genuinely repeats lines.
    deduped: list[TranscriptSegment] = []
    seen: set = set()
    for seg in coerced:
        key = (round(seg.start, 1), _norm_cue_text(seg.text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(seg)
    return [s.model_dump() for s in deduped]


def _norm_cue_text(text: str) -> str:
    """Lowercased, whitespace-collapsed cue text for duplicate comparison."""
    return " ".join((text or "").lower().split())


def _duplicate_text_share(rows: list, min_chars: int = 8) -> float:
    """Fraction of cues whose normalized text also appears on ANOTHER cue.

    Short interjections ("Yeah.", "Okay.") legitimately recur in real
    dialogue, so only lines of ``min_chars``+ characters count. Accepts
    dicts or models; timing is ignored — this measures pure text-level
    duplication across the track.
    """
    texts = []
    for s in rows or []:
        t = s.get("text") if isinstance(s, dict) else getattr(s, "text", "")
        norm = _norm_cue_text(t)
        if len(norm) >= min_chars and not norm.startswith("["):
            texts.append(norm)
    if not texts:
        return 0.0
    from collections import Counter
    counts = Counter(texts)
    dup = sum(n for n in counts.values() if n > 1)
    return dup / len(texts)


def detect_union_write(incoming: list, stored: list) -> tuple[bool, str]:
    """Detect a "stale-union" transcript replace before it poisons storage.

    The observed corruption (job 77198bd0, 2026-07-03 run): the NLE
    reverse-sync PUT a timeline holding the fresh subtitle track PLUS stale
    generations of the same cues at shifted times — the stored track went
    406 clean cues → 625 with 117 texts repeated ~3x each at timestamps
    minutes apart (one phantom family at a constant +19min offset, another
    splayed by overlap resolution). A genuine user edit never looks like
    that: splits/merges change the count modestly and produce NEW text
    fragments, not hundreds of verbatim copies of existing lines.

    Signature required (BOTH must hold):
      * the incoming track is much bigger than the stored one
        (>25% AND >15 cues more), and
      * the incoming track's duplicate-text share is well above the stored
        track's (+10 points) — i.e. the extra cues are copies, not content.

    Returns ``(is_union, reason)``. Fail-open: an empty/absent stored track
    can't be poisoned, so anything is allowed then.
    """
    inc_n = len(incoming or [])
    st_n = len(stored or [])
    if st_n == 0 or inc_n <= max(int(st_n * 1.25), st_n + 15):
        return False, ""
    inc_share = _duplicate_text_share(incoming)
    st_share = _duplicate_text_share(stored)
    if inc_share >= st_share + 0.10:
        return True, (
            f"incoming {inc_n} cues vs stored {st_n} with duplicate-text "
            f"share {inc_share:.0%} vs {st_share:.0%} — looks like a stale "
            "timeline union, not an edit")
    return False, ""


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
