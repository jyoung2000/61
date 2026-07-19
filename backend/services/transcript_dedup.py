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


def drop_scattered_duplicates(
    segments: list,
    text_key: str = "text",
    *,
    threshold: int = 4,
    keep: int = 1,
) -> tuple[list, int]:
    """Collapse a phrase that recurs VERBATIM ``threshold``+ times across the
    whole timeline down to its first ``keep`` occurrence(s).

    ``drop_repetition_loops`` only fully collapses LONG (> ``long_block_chars``)
    blocks; a SHORT phrase is capped at 3, and a CJK utterance is frequently
    under that length so it is treated as short too. That left transcripts where
    a small set of short fragments ("real breasts and you're", "Going deeper,
    uh…") each recurred 5–10× — the duplicated-transcript report: a Whisper loop
    on the source, or a small editorial model echoing the same safe phrase for
    several distinct source lines. Real dialogue does not repeat the SAME
    fragment 4+ times verbatim across a video (a genuine interjection stays at
    2–3×), so a high recurrence count is an unambiguous artifact regardless of
    length.

    Markers ("[♪ music ♪]") are exempt — every occurrence is kept. Phrases that
    recur fewer than ``threshold`` times are left untouched, preserving genuine
    short interjections. Order-preserving; keeps the earliest occurrence(s).
    Returns ``(kept_segments, dropped_count)``.
    """
    from collections import Counter
    counts: Counter = Counter()
    for seg in segments or []:
        raw = (_seg_get(seg, text_key, "") or "").strip()
        if not raw or _is_marker_text(raw):
            continue
        counts[_normalize_text(raw)] += 1
    seen: dict = {}
    out: list = []
    dropped = 0
    for seg in segments or []:
        raw = (_seg_get(seg, text_key, "") or "").strip()
        if not raw or _is_marker_text(raw):
            out.append(seg)
            continue
        k = _normalize_text(raw)
        if counts.get(k, 0) >= threshold:
            n = seen.get(k, 0)
            if n >= keep:
                dropped += 1
                continue
            seen[k] = n + 1
        out.append(seg)
    return out, dropped


def collapse_repeated_runs(
    segments: list,
    text_key: str = "text",
    *,
    min_run: int = 3,
) -> tuple[list, int]:
    """Drop a CONTIGUOUS run of cues whose normalized-text sequence already
    appeared earlier — the Whisper double-pass / gap-fill signature where a whole
    span (e.g. a 2-minute opening) is re-transcribed at later timestamps, so the
    SAME ordered block of cues shows up twice.

    Per-phrase dedup can't catch this: ``drop_repetition_loops`` caps a short
    line at 3 and ``drop_scattered_duplicates`` only fires at 4+ occurrences, so
    a block of short cues repeated 2-3× survives — each individual line is within
    tolerance. This matches the block as a UNIT: a run of ``min_run``+ consecutive
    cues equal (normalised) to an earlier run is the later (duplicate) copy and is
    dropped. Keeps the EARLIEST occurrence; markers break a run (never matched).
    Order-preserving. Returns ``(kept_segments, dropped_count)``.
    """
    segs = list(segments or [])
    n = len(segs)
    if n < min_run * 2:
        return segs, 0
    keys: list = []
    for s in segs:
        raw = (_seg_get(s, text_key, "") or "").strip()
        keys.append(None if (not raw or _is_marker_text(raw)) else _normalize_text(raw))
    drop = [False] * n
    first: dict = {}     # normalised key -> earliest (kept) index
    i = 0
    while i < n:
        k = keys[i]
        if k is None:
            i += 1
            continue
        p = first.get(k)
        if p is not None and p < i:
            # Extend the match between the earlier block (from p) and here (i).
            run = 0
            while (i + run < n and p + run < i
                   and keys[i + run] is not None
                   and keys[i + run] == keys[p + run]):
                run += 1
            if run >= min_run:
                for j in range(i, i + run):
                    drop[j] = True
                i += run
                continue
        if k not in first:
            first[k] = i
        i += 1
    if not any(drop):
        return segs, 0
    return [s for j, s in enumerate(segs) if not drop[j]], sum(drop)


def _norm_token(t: str) -> str:
    """Lowercased, punctuation-stripped token for repetition comparison."""
    import unicodedata
    return "".join(
        ch for ch in (t or "").lower()
        if not unicodedata.category(ch).startswith("P")
    )


_SENT_TERMINATORS = (".", "!", "?", "。", "！", "？", "…")


def _has_sentence_terminator(surface_words: list) -> bool:
    """True when any word in the phrase ends a sentence — so a cross-sentence
    restatement ('Go home. Go home now.') is NOT collapsed as a loop."""
    for w in surface_words:
        s = (w or "").rstrip("\"')]}»”’")
        if s.endswith(_SENT_TERMINATORS):
            return True
    return False


def _collapse_consecutive_phrase_repeats(words, norm, *, max_phrase=6, min_phrase=2):
    """Collapse any contiguous phrase (``min_phrase``..``max_phrase`` words) that
    repeats ≥2× back-to-back ANYWHERE in the line, keeping ONE copy. Returns
    ``(words, changed)``. Order-preserving; operates on the ORIGINAL surfaces,
    compares on ``norm``. The repeated unit must be ≥2 words and must not end a
    sentence (a cross-sentence restatement is left to editorial judgement)."""
    n = len(words)
    changed = False
    i = 0
    out_w, out_n = [], []
    while i < n:
        collapsed = False
        # Prefer the LONGEST repeating unit at this position.
        for p in range(min(max_phrase, (n - i) // 2), min_phrase - 1, -1):
            unit = norm[i:i + p]
            if not any(unit):           # skip empty / punct-only units
                continue
            reps = 1
            while norm[i + reps * p: i + (reps + 1) * p] == unit:
                reps += 1
            if reps >= 2 and not _has_sentence_terminator(words[i:i + p]):
                out_w.extend(words[i:i + p])   # keep ONE copy (original surfaces)
                out_n.extend(norm[i:i + p])
                i += reps * p
                changed = True
                collapsed = True
                break
        if not collapsed:
            out_w.append(words[i])
            out_n.append(norm[i])
            i += 1
    return out_w, changed


def _collapse_text_repetition(text: str, *, min_word_run: int = 3, keep_run: int = 2) -> str:
    """Collapse repetition WITHIN a single line of text. Conservative.

    1. Whole-line phrase loop: when the line is the SAME phrase repeated k≥2
       times back-to-back, keep ONE copy — "I will protect you I will protect
       you" → "I will protect you". The repeating unit must be ≥2 words (a
       single repeated word is handled by rule 2) so genuine short emphasis
       ("Bye bye") is untouched.
    2. Immediate identical-word run of ≥``min_word_run`` copies → keep
       ``keep_run`` — "no no no no no" → "no no" — protecting legitimate
       doubling ("No, no.")."""
    raw = (text or "").strip()
    if not raw:
        return text
    words = raw.split()
    n = len(words)
    if n < 2:
        return text

    norm = [_norm_token(w) for w in words]

    # ── Rule 0: consecutive repeated PHRASE anywhere in the line ──
    # Catches the dominant 4B failure mode the tiling/single-word rules miss: a
    # repeated phrase with a trailing tail ("I will protect you I will protect
    # you no matter what" → "I will protect you no matter what").
    words2, ph_changed = _collapse_consecutive_phrase_repeats(words, norm)
    if ph_changed:
        words = words2
        norm = [_norm_token(w) for w in words]
        n = len(words)

    # ── Rule 1: whole-line phrase periodicity ──
    for p in range(2, n // 2 + 1):
        if n % p != 0:
            continue
        unit = norm[:p]
        if all(norm[i:i + p] == unit for i in range(0, n, p)) and unit != [""] * p:
            # Keep the first occurrence's ORIGINAL words (with its punctuation).
            return " ".join(words[:p])

    # ── Rule 2: immediate identical-word runs ──
    out: list[str] = []
    i = 0
    changed = ph_changed
    while i < n:
        j = i + 1
        while j < n and norm[j] == norm[i] and norm[i] != "":
            j += 1
        run = j - i
        if run >= min_word_run:
            out.extend(words[i:i + keep_run])
            changed = True
        else:
            out.extend(words[i:j])
        i = j
    return " ".join(out) if changed else raw


# A letter repeated 5+ times in a row (case-insensitive: "Uuuuuu",
# "AAAAAAAAAAAA", "Eeeeeee") -- non-verbal vocalization the ASR stretched into
# a wall of glyphs. Netflix-style subs cap these; keep 3 so the cue still
# reads as a sound, not a scream of characters.
_CHAR_RUN_RE = re.compile(r"([A-Za-z])(\1{4,})", re.IGNORECASE)


def collapse_char_runs(
    segments: list,
    text_key: str = "text",
    *,
    keep: int = 3,
) -> tuple[list, int]:
    """Collapse runs of one repeated letter to ``keep`` glyphs per cue.

    Targets stretched vocalizations on the TRANSLATED track ("Uuuuuuuuuu.",
    "AAAAAAAAAA AIBON") that read as noise walls in the export. Real English
    words never contain 5 identical consecutive letters, so text is safe.
    Order/timing untouched; markers skipped. Returns ``(segments, changed)``."""
    changed = 0
    for seg in segments or []:
        raw = (_seg_get(seg, text_key, "") or "")
        if not raw or _is_marker_text(raw):
            continue
        new = _CHAR_RUN_RE.sub(lambda m: m.group(1) + m.group(2)[: keep - 1], raw)
        if new != raw:
            _seg_set(seg, text_key, new)
            changed += 1
    return segments, changed


def collapse_intra_cue_repetition(
    segments: list,
    text_key: str = "text",
    *,
    min_word_run: int = 3,
    keep_run: int = 2,
) -> tuple[list, int]:
    """Clean repetition INSIDE individual cues — the artifact the cross-cue
    dedup passes can't see: one translated line that reads "I will protect you
    I will protect you" or "no no no no no no" (a small LLM duplicating its own
    output). Markers ("[♪ music ♪]") are left untouched. Order-preserving;
    timing/word fields are untouched. Returns ``(segments, changed_count)``."""
    out: list = []
    changed = 0
    for seg in segments or []:
        raw = (_seg_get(seg, text_key, "") or "")
        if not raw.strip() or _is_marker_text(raw):
            out.append(seg)
            continue
        cleaned = _collapse_text_repetition(
            raw, min_word_run=min_word_run, keep_run=keep_run)
        if cleaned != raw.strip() and cleaned != raw:
            _seg_set(seg, text_key, cleaned)
            changed += 1
        out.append(seg)
    return out, changed


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


# ── Sentence-level near-duplicate removal (translated track) ────────────
# The overlap dedup above works on WHOLE cues at OVERLAPPING times. Two real
# duplication patterns slip through it: (a) a re-decoded span landing at
# SHIFTED (non-overlapping) times whose translation is a paraphrase of a
# nearby cue ("Grown from Earth, humanity sought new hope…" 10 s before
# "Seeking new hope for a better life…"), and (b) a duplicated SENTENCE
# embedded inside a longer neighboring cue ("…trajectory? The surveillance
# satellite's vision is lacking." repeating the previous cue verbatim).
# This pass splits cues into sentences and drops any sentence that
# near-repeats a sentence from a nearby EARLIER cue; a cue whose every
# sentence is a repeat is dropped whole. Conservative by construction:
# only sentences with enough content words are eligible, so legitimate
# short repeats ("Fire! Fire!", a name called twice) are never touched.

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with is are was were be been it "
    "its this that these those i you he she we they them his her their my "
    "your our as by from not no so do does did have has had will would can "
    "could there here".split())


def _content_words(s: str) -> set:
    """Content words normalized for matching: curly apostrophes unified with
    straight ones and possessive 's stripped, so "satellite’s" ≡ "satellite's"
    ≡ "satellite" (a real duplicate escaped on exactly that difference)."""
    out = set()
    for w in re.findall(r"[\w']+", (s or "").lower().replace("’", "'")):
        if w.endswith("'s"):
            w = w[:-2]
        w = w.strip("'")
        if w and w not in _STOPWORDS and len(w) >= 2:
            out.add(w)
    return out


def drop_repeated_sentences(
    segments: list,
    text_key: str = "text",
    start_key: str = "start",
    window_s: float = 25.0,
    similarity: float = 0.70,
    min_content_words: int = 4,
) -> tuple[list, int]:
    """Remove sentences that near-repeat a sentence from a nearby earlier cue.

    A sentence is dropped when (a) it carries ≥ ``min_content_words`` content
    words (long enough that a coincidental repeat is implausible), and (b) an
    earlier sentence within ``window_s`` seconds has ≥ ``similarity``
    content-word overlap (relative to the SMALLER set — catches paraphrases
    that reorder or trim words) or ≥ 0.8 :func:`_text_similarity`. Cues left
    empty are removed. Returns ``(segments, sentences_dropped)``."""
    if not segments:
        return segments, 0
    kept_sents: list = []  # (start_s, cue_index, content_word_set, raw_sentence)
    out: list = []
    dropped = 0
    for cue_idx, seg in enumerate(segments):
        txt = (_seg_get(seg, text_key, "") or "").strip()
        start = float(_seg_get(seg, start_key, 0) or 0)
        if not txt:
            out.append(seg)
            continue
        sents = [s for s in _SENT_SPLIT_RE.split(txt) if s.strip()]
        keep: list = []
        for sent in sents:
            cw = _content_words(sent)
            is_dup = False
            if len(cw) >= min_content_words:
                for (ps, pidx, pcw, praw) in reversed(kept_sents):
                    if start - ps > window_s:
                        break
                    # Same-cue echoes ("Five? Five Gundams?") are deliberate
                    # dramatic repeats — leave them to the intra-cue pass.
                    if pidx == cue_idx or not pcw:
                        continue
                    denom = min(len(cw), len(pcw))
                    if denom and (len(cw & pcw) / denom) >= similarity:
                        is_dup = True
                        break
                    if _text_similarity(sent, praw) >= 0.8:
                        is_dup = True
                        break
            if is_dup:
                dropped += 1
            else:
                keep.append(sent)
                kept_sents.append((start, cue_idx, cw, sent))
        if not keep:
            continue  # every sentence was a repeat — drop the cue
        if len(keep) != len(sents):
            _seg_set(seg, text_key, " ".join(keep))
        out.append(seg)
    if dropped:
        return out, dropped
    return segments, 0


def drop_bare_glossary_runs(
    segments: list,
    glossary_terms,
    text_key: str = "text",
    start_key: str = "start",
    min_run: int = 2,
    max_gap_s: float = 30.0,
) -> tuple[list, int]:
    """Drop RUNS of consecutive cues that are each a bare glossary term.

    The observed failure: over ED music, Whisper hallucinates romaji lyric
    fragments and the translator maps each to the nearest pinned name —
    shipping "Justlove." / "Space Port." / "Gundanium." as three consecutive
    subtitles. A single bare name can be legitimate dialogue (someone called
    by name), so only a RUN of ≥ ``min_run`` such cues (each within
    ``max_gap_s`` of the previous) is treated as music-section noise and
    removed. Returns ``(segments, dropped_count)``."""
    terms = {str(t or "").strip().lower() for t in (glossary_terms or []) if t}
    if not segments or len(terms) < 2:
        return segments, 0

    def _is_bare(seg) -> bool:
        txt = (_seg_get(seg, text_key, "") or "").strip().replace("’", "'")
        core = re.sub(r"[^\w\s']+", "", txt).strip().lower()
        if not core:
            return False
        words = core.split()
        if len(words) > 3:
            return False
        # The whole cue is one glossary term, or every word of it is one
        # (the miner splits "Just Love" into two single-word terms).
        return core in terms or all(w in terms for w in words)

    flags = [_is_bare(s) for s in segments]
    drop: set = set()
    i = 0
    while i < len(segments):
        if not flags[i]:
            i += 1
            continue
        j = i
        run = [i]
        while j + 1 < len(segments) and flags[j + 1]:
            _gap = (float(_seg_get(segments[j + 1], start_key, 0) or 0)
                    - float(_seg_get(segments[j], start_key, 0) or 0))
            if _gap > max_gap_s:
                break
            run.append(j + 1)
            j += 1
        if len(run) >= min_run:
            drop.update(run)
        i = j + 1
    if not drop:
        return segments, 0
    return [s for k, s in enumerate(segments) if k not in drop], len(drop)
