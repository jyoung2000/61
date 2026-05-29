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
