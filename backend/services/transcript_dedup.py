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


def _norm_for_compare(text: str) -> str:
    """Whitespace + punctuation-stripped lowercase key for similarity tests.

    Strips both ASCII punctuation (string.punctuation) and a handful of
    CJK punctuation marks (full-width middle dot, full-width comma /
    period, ideographic comma, etc.) so the Japanese gap-fill near-
    duplicates that differ only in '・' / '、' collapse correctly.
    """
    import string
    if not text:
        return ""
    t = text.strip().lower()
    # ASCII punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # CJK + full-width punctuation observed on real ja/zh/ko transcripts
    cjk_punct = "・、。「」『』【】〈〉《》（）〔〕［］｛｝!?,.；："
    t = t.translate(str.maketrans("", "", cjk_punct))
    return re.sub(r"\s+", "", t)


def collapse_overlapping_duplicates(
    segments: list,
    text_key: str = "text",
    start_key: str = "start",
    end_key: str = "end",
    text_similarity_threshold: float = 0.8,
    min_overlap_sec: float = 0.1,
) -> tuple[list, int]:
    """Drop non-adjacent near-duplicates that overlap in time.

    The gap-fill pass + speaker fusion + readability reflow can produce
    segment pairs that are NOT adjacent in the list (other cues sit
    between them) but DO overlap in the timeline AND carry near-identical
    text. The adjacent-only ``collapse_adjacent_duplicates`` misses
    these; a global Jaccard-on-tokens + timestamp-overlap pass catches
    them. Compares each segment against the K most-recently-emitted ones
    (K=8 — wide enough to find any realistic temporal overlap, narrow
    enough to keep this O(n)).

    Order-preserving — emits the first occurrence and drops later
    duplicates. Returns ``(kept_segments, dropped_count)``.
    """
    if not segments:
        return segments, 0

    kept: list = []
    kept_meta: list[tuple[float, float, set[str]]] = []  # (start, end, tokens)
    dropped = 0
    WINDOW = 8

    for seg in segments:
        text = (_seg_get(seg, text_key, "") or "").strip()
        if not text:
            kept.append(seg)
            kept_meta.append((0.0, 0.0, set()))
            continue
        try:
            s = float(_seg_get(seg, start_key, 0) or 0)
            e = float(_seg_get(seg, end_key, 0) or 0)
        except (TypeError, ValueError):
            kept.append(seg)
            kept_meta.append((0.0, 0.0, set()))
            continue

        # Use normalized characters (no whitespace) as the matching unit so
        # CJK text (no inter-word spaces) compares correctly too. For Latin
        # text the token version is more discriminating, so we keep both.
        norm = _norm_for_compare(text)
        tokens = set(t for t in re.split(r"\s+", text.lower()) if t)

        is_dup = False
        # Look back at the WINDOW most-recent kept segments only.
        for (ps, pe, ptoks), pseg in zip(
            reversed(kept_meta[-WINDOW:]),
            reversed(kept[-WINDOW:]),
        ):
            # Time overlap required — non-overlapping repeats stay (a recap
            # at minute 23 is not the same instance as the original at 04).
            overlap = max(0.0, min(e, pe) - max(s, ps))
            if overlap < min_overlap_sec:
                continue
            ptext = (_seg_get(pseg, text_key, "") or "").strip()
            if not ptext:
                continue
            # Two-axis similarity: byte-exact normalized match wins
            # outright; otherwise compute Jaccard over whitespace tokens
            # (Latin) so paraphrased near-duplicates still collapse.
            if norm and norm == _norm_for_compare(ptext):
                is_dup = True
                break
            if tokens and ptoks:
                inter = len(tokens & ptoks)
                union = len(tokens | ptoks)
                if union and (inter / union) >= text_similarity_threshold:
                    is_dup = True
                    break
        if is_dup:
            dropped += 1
            continue
        kept.append(seg)
        kept_meta.append((s, e, tokens))
    return kept, dropped


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
