"""Dependency-light transcript de-duplication helpers.

Extracted from reframer_audio (which imports cv2/torch) so the logic is unit
-testable without the heavy ASR import chain.
"""

from __future__ import annotations

import re


def _is_marker_text(s: str) -> bool:
    """True for a bracketed non-speech caption marker (``[♪ music ♪]``,
    ``[applause]`` …). Mirrors ``audio_analyzer.is_subtitle_marker`` without
    importing it, so this module stays dependency-light. Such cues are
    intentional and language-neutral, so they must NEVER be treated as
    repetition-loop hallucinations even when several identical ones appear
    (an episode can have OP + ED + insert songs, i.e. >3 ``[♪ music ♪]`` cues)."""
    t = (s or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        return False
    inner = t[1:-1]
    return ("♪" in t) or (len(inner.split()) <= 2 and inner.replace(" ", "").isalpha())


def is_low_confidence_phantom(
    words: list,
    no_speech_prob: float,
    *,
    max_avg_conf: float = 0.40,
    min_lowconf_frac: float = 0.80,
    min_no_speech: float = 0.50,
    conf_key: str = "confidence",
) -> bool:
    """True when a cue is a low-confidence phantom Whisper hallucination.

    Whisper invents short cues over silence / music — the "Don't let",
    "So nice", "Hmm." fragments that flood a mostly-silent video — which
    survive the exact-match boilerplate blocklist and the ``no_speech_prob``
    > 0.7 clamp (they sit just under it). The Temporal Audio Coverage (TACT)
    ledger already classifies a *word* as ``low_confidence`` below 0.4; this
    feeds that SAME signal back to DROP the *cue* when its words are
    OVERWHELMINGLY low-confidence (average below ``max_avg_conf`` AND at least
    ``min_lowconf_frac`` of words under 0.4) AND Whisper itself doubted the
    chunk was speech (``no_speech_prob`` at/above ``min_no_speech``).

    ALL conditions must hold, so genuine quiet speech — which Whisper
    transcribes with confident words even when ``no_speech_prob`` is moderate
    — is preserved. Cues with no word-level confidences return ``False``
    (other filters handle those).

    Schema-flexible: ``words`` may be dicts or objects carrying ``conf_key``.
    """
    if not words or no_speech_prob < min_no_speech:
        return False
    confs = []
    for w in words:
        c = w.get(conf_key) if isinstance(w, dict) else getattr(w, conf_key, None)
        if c is not None:
            try:
                confs.append(float(c))
            except (TypeError, ValueError):
                continue
    if not confs:
        return False
    avg_conf = sum(confs) / len(confs)
    lowconf_frac = sum(1 for c in confs if c < 0.4) / len(confs)
    return avg_conf < max_avg_conf and lowconf_frac >= min_lowconf_frac


def drop_repetition_loops(
    segments: list,
    text_key: str = "text",
    *,
    similarity_threshold: float = 0.9,
    long_block_chars: int = 24,
    fuzzy_window: int = 64,
) -> tuple[list, int]:
    """Remove Whisper repetition-loop hallucinations.

    When the main/gap-fill passes run over music / quiet regions, Whisper loops
    and emits the same content repeatedly, scattered across the timeline (so an
    adjacent-only dedup misses them). Real dialogue almost never repeats across
    an episode, so a recurring long block is a hallucination.

    Long blocks (> ``long_block_chars`` normalised chars) are caught two ways,
    order-preserving, keeping the earliest occurrence:

      * **Exact (normalised) recurrence — anywhere on the timeline.** Text is
        normalised by stripping punctuation + whitespace, so the opening
        narration re-emitted at six separated timestamps with only punctuation /
        spacing differences collapses to one. This is an O(1) dict lookup, so it
        is unbounded in reach yet cheap (a 1400-cue episode is near-instant), and
        it is what real Whisper loops (byte-identical re-emissions) trip.
      * **Fuzzy near-identical recurrence — within the last ``fuzzy_window``
        long cues.** Catches copies that differ by genuine ASR character drift.
        A length gate (``min/max ≥ similarity_threshold``) is applied first, so a
        DISTINCT longer line that merely CONTAINS a shorter kept line (e.g.
        "…protect this colony" then "…protect this colony until my dying breath")
        is never matched and the fuller line is kept. The window bounds the cost
        to O(n·window) instead of an O(n²) full-history scan.

    Short interjections (e.g. "了解", "Roger") can legitimately recur, so exact
    repeats are capped at 3.

    Returns ``(kept_segments, dropped_count)``.
    """
    from collections import deque
    seen: dict = {}                    # exact-key counts for short interjections
    long_exact: set = set()            # normalised text of long blocks kept
    recent_long = deque(maxlen=max(1, int(fuzzy_window)))  # (raw, norm_len) recents
    out = []
    dropped = 0
    for seg in segments or []:
        raw = (seg.get(text_key, "") or "").strip() if isinstance(seg, dict) else \
            (getattr(seg, text_key, "") or "").strip()
        if not raw:
            out.append(seg)
            continue
        if _is_marker_text(raw):
            # Intentional non-speech marker — keep every occurrence.
            out.append(seg)
            continue
        key = re.sub(r"\s+", "", raw)
        if len(key) > long_block_chars:
            nrm = _normalize_text(raw)
            # 1. Exact (punctuation/space-insensitive) recurrence, unbounded reach.
            if nrm in long_exact:
                dropped += 1
                continue
            # 2. Fuzzy near-identical recurrence among recent long cues, length-
            #    gated so a longer line that merely contains a shorter one is not
            #    treated as a loop.
            L = len(nrm)
            is_loop = False
            for prev_raw, prev_len in recent_long:
                ratio = (min(L, prev_len) / max(L, prev_len)) if max(L, prev_len) else 1.0
                if ratio >= similarity_threshold and \
                        _text_similarity(raw, prev_raw) >= similarity_threshold:
                    is_loop = True
                    break
            if is_loop:
                dropped += 1
                continue
            long_exact.add(nrm)
            recent_long.append((raw, L))
            out.append(seg)
        else:
            n = seen.get(key, 0)
            if n >= 3:
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


def _normalize_text(s: str) -> str:
    """Punctuation- and whitespace-stripped text for similarity comparison.

    Strips ALL Unicode punctuation (so "アフターコロニー195。" and
    "アフターコロニー195、" — the same narration with different trailing CJK
    punctuation from two ASR passes — normalise equal) and all whitespace."""
    import unicodedata
    return "".join(
        ch for ch in (s or "")
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _text_similarity(a: str, b: str) -> float:
    """Rough text similarity in [0, 1].

    Whitespace-insensitive exact match → 1.0; full containment of the
    shorter inside the longer → 0.95; otherwise word-set Jaccard for
    space-delimited text, falling back to a character-bigram Jaccard for
    CJK / no-space scripts (where the gap-fill duplicates show up).
    """
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) >= 8 and len(nb) >= 8 and (na in nb or nb in na):
        return 0.95
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if wa and wb and (len(wa) > 1 or len(wb) > 1):
        union = len(wa | wb)
        if union:
            return len(wa & wb) / union
    # CJK / no-space fallback: character-bigram Jaccard.
    ba = {na[i:i + 2] for i in range(len(na) - 1)}
    bb = {nb[i:i + 2] for i in range(len(nb) - 1)}
    if ba and bb:
        return len(ba & bb) / len(ba | bb)
    return 0.0


def collapse_overlapping_duplicates(
    segments: list,
    text_key: str = "text",
    start_key: str = "start",
    end_key: str = "end",
    similarity: float = 0.8,
    time_tolerance: float = 0.3,
) -> tuple[list, int]:
    """Collapse near-duplicate segments by timestamp overlap + text similarity.

    Unlike :func:`collapse_adjacent_duplicates` (exact text, strictly
    adjacent), this removes re-transcription artefacts that land at
    OVERLAPPING or near-overlapping times with merely SIMILAR text — e.g. the
    gap-fill pass re-emitting the same line a few hundred ms off the primary
    cue, or the repeated ``作戦名オペレーション・メテオ`` observed near the OP. A
    later segment is dropped when, against any already-kept segment, it BOTH
    (a) overlaps in time (or sits within ``time_tolerance`` seconds) AND
    (b) has text similarity ≥ ``similarity``. The kept segment's end is
    stretched to cover the dropped one. Input order is preserved among the
    survivors.

    Returns ``(kept_segments, dropped_count)``.
    """
    if not segments:
        return segments, 0

    def _start(seg):
        return float(_seg_get(seg, start_key, 0) or 0)

    def _end(seg):
        return float(_seg_get(seg, end_key, _start(seg)) or _start(seg))

    # Evaluate in time order so the earliest occurrence wins, but remember
    # original positions so the survivors come back in input order.
    indexed = sorted(enumerate(segments), key=lambda p: (_start(p[1]), _end(p[1])))
    kept: list = []
    dropped_orig: set = set()
    for orig_i, seg in indexed:
        s, e, txt = _start(seg), _end(seg), _seg_get(seg, text_key, "") or ""
        is_dup = False
        for kseg in reversed(kept):
            ks, ke = _start(kseg), _end(kseg)
            if ke < s - time_tolerance:
                break  # nothing earlier can still collide in time
            overlaps = (min(e, ke) - max(s, ks)) > -time_tolerance
            if not overlaps:
                continue
            if _text_similarity(txt, _seg_get(kseg, text_key, "") or "") >= similarity:
                try:
                    if e > ke:
                        _seg_set(kseg, end_key, _seg_get(seg, end_key))
                except (TypeError, ValueError):
                    pass
                is_dup = True
                break
        if is_dup:
            dropped_orig.add(orig_i)
        else:
            kept.append(seg)

    if not dropped_orig:
        return segments, 0
    out = [seg for i, seg in enumerate(segments) if i not in dropped_orig]
    return out, len(dropped_orig)


def clamp_segments_to_duration(
    segments: list,
    duration_sec: float,
    start_key: str = "start",
    end_key: str = "end",
    words_key: str = "words",
) -> tuple[list, int]:
    """Drop cues stamped past the end of the audio; clamp cues that overrun it.

    faster-whisper can DRIFT/LOOP on repetitive music and emit segment
    timestamps BEYOND the real audio duration — e.g. cues at 35:51 on a 24:27
    video — which then mis-times the back third of the subtitle track. There is
    no audio past ``duration_sec``, so a cue that STARTS at/after it is a
    hallucination / timestamp-drift artefact and is dropped; a cue that merely
    RUNS PAST the end is clamped back to it (with its word timings). On this AMV
    the past-the-end cues were verified to be loop-repeats of earlier lines, not
    unique tail dialogue, so dropping them loses no real content.

    Order-preserving. Schema-flexible like the other helpers here (pass
    ``start_key="start_sec"`` / ``end_key="end_sec"`` for the reframer's dicts).
    Returns ``(kept_segments, dropped_or_clamped_count)``.
    """
    if not segments or not duration_sec or duration_sec <= 0:
        return segments, 0
    limit = float(duration_sec)
    out: list = []
    changed = 0
    for seg in segments:
        st = float(_seg_get(seg, start_key, 0.0) or 0.0)
        en = float(_seg_get(seg, end_key, st) or st)
        if st >= limit:
            # Starts at/after the audio end → nothing real here; drop it.
            changed += 1
            continue
        if en > limit:
            # Straddles the end → clamp the cue (and any words) back to it.
            _seg_set(seg, end_key, round(limit, 3))
            words = _seg_get(seg, words_key, None)
            if words:
                kept_w = []
                for w in words:
                    if float(_seg_get(w, "start", st) or st) >= limit:
                        continue
                    if float(_seg_get(w, "end", st) or st) > limit:
                        _seg_set(w, "end", round(limit, 3))
                    kept_w.append(w)
                _seg_set(seg, words_key, kept_w)
            changed += 1
        out.append(seg)
    return out, changed
