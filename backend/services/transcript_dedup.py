"""Dependency-light transcript de-duplication helpers.

Extracted from reframer_audio (which imports cv2/torch) so the logic is unit
-testable without the heavy ASR import chain.
"""

from __future__ import annotations

import re


def drop_repetition_loops(segments: list, text_key: str = "text") -> tuple[list, int]:
    """Remove Whisper repetition-loop hallucinations.

    When the gap-fill pass re-transcribes music / quiet regions with VAD off,
    Whisper loops and emits the SAME text repeatedly, scattered across the
    timeline (so an adjacent-only dedup misses them). Real dialogue almost
    never repeats verbatim many times across an episode, so an exact-text
    segment that recurs beyond a small cap is a hallucination.

    Keeps the earliest occurrences — 1 for long lines (a repeating sentence is
    unambiguous hallucination), up to 3 for short interjections (e.g. "了解",
    "Roger") that can legitimately recur — and drops the rest. Order-preserving.

    Returns ``(kept_segments, dropped_count)``.
    """
    seen: dict = {}
    out = []
    dropped = 0
    for seg in segments or []:
        raw = (seg.get(text_key, "") or "").strip() if isinstance(seg, dict) else \
            (getattr(seg, text_key, "") or "").strip()
        if not raw:
            out.append(seg)
            continue
        key = re.sub(r"\s+", "", raw)
        cap = 1 if len(key) > 24 else 3
        n = seen.get(key, 0)
        if n >= cap:
            dropped += 1
            continue
        seen[key] = n + 1
        out.append(seg)
    return out, dropped


def _seg_get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _seg_set(seg, key, value):
    if isinstance(seg, dict):
        seg[key] = value
    else:
        setattr(seg, key, value)


def collapse_adjacent_duplicates(
    segments: list,
    text_key: str = "text",
    start_key: str = "start",
    end_key: str = "end",
) -> tuple[list, int]:
    """Collapse runs of consecutive segments that carry the SAME text.

    Whisper (and the downstream speaker-fusion / resegmentation passes) can emit
    the identical cue twice in a row at adjacent timestamps — e.g. the exported
    transcript showed ``[11:25] アフターコロニー…`` immediately followed by another
    ``[11:25] アフターコロニー…``. ``drop_repetition_loops`` only caps *global*
    repeats (and legitimately keeps a few short interjections), so a back-to-back
    duplicate of a long line survives it. This pass removes the later copy and
    stretches the kept segment's end to cover the dropped one, so the on-screen
    subtitle and the transcript panel show one clean cue.

    Comparison is whitespace-insensitive. Order-preserving. Returns
    ``(kept_segments, dropped_count)``.
    """
    out: list = []
    dropped = 0
    prev_key = None
    for seg in segments or []:
        raw = (_seg_get(seg, text_key, "") or "").strip()
        norm = re.sub(r"\s+", "", raw)
        if norm and norm == prev_key and out:
            # Same text as the previous KEPT cue — fold this one into it.
            prev = out[-1]
            try:
                prev_end = float(_seg_get(prev, end_key, 0) or 0)
                this_end = float(_seg_get(seg, end_key, 0) or 0)
                if this_end > prev_end:
                    _seg_set(prev, end_key, _seg_get(seg, end_key))
            except (TypeError, ValueError):
                pass
            dropped += 1
            continue
        out.append(seg)
        prev_key = norm or None
    return out, dropped
