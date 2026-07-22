"""Defensive cleanup for a translated transcript corrupted by an interrupted run.

When the container restarts mid-pipeline (GPU drops, OOM) and the job resumes —
or a flaky websocket drives repeated relay saves — the stored
``translated_transcript`` can end up a UNION of the source + translated tracks
with cues duplicated many times over (observed: 584 cues, 37% still Japanese,
hot lines repeated 10×). The pipeline itself persists a clean track; this is
damage that happens AFTER, in the resume/relay path.

``sanitize_translated_transcript`` repairs that on the way out (and is applied
before persist as hygiene): it drops cues still in the source script (CJK when
the target is non-CJK — an English "translated" track must not contain Japanese)
and collapses duplication, while leaving a clean track untouched. Pure stdlib so
it stays unit-testable and cheap.

Duplication has two sources, handled together: the resume/relay UNION above
(same blob many times), AND Whisper repetition/hallucination on non-speech audio
(music, moans, silence) that emits the same line at several timestamps and is
carried verbatim through the 1:1 translation. A SUBSTANTIAL line collapses to a
single occurrence; SHORT lines (which can legitimately recur) keep a couple;
markers ("[♪ music ♪]") are never deduped.

``merge_transcript_fragments`` is a SEPARATE, opt-in pass (the sanitizer never
calls it) that folds Whisper's mid-sentence splits back into whole utterances.
Whisper segments on acoustic pauses, so one spoken sentence arrives as 2-4 cues
of a few words ("It's just the" / "number 21."). It merges a cue into the next
ONLY when the current text looks unfinished (no sentence-final punctuation) and
the two are clearly one utterance: same speaker, a small time gap, result under
a readable length/duration. It is a strict FIXED POINT (merging twice == once),
which is what makes it safe to apply once on read for a terminal job without the
"lines move around while I'm reading" churn an earlier non-idempotent re-flow
caused. Space-joining is wrong for CJK, so CJK targets are returned untouched.
"""
from __future__ import annotations

import re

_CJK_TARGETS = {"ja", "ko", "zh", "zh-cn", "zh-tw", "yue"}
# A whole cue that is only a repeated grunt letter + trailing dots ("Nn...",
# "Nnn...", "Mmm") is non-lexical mumble filler Whisper emits and the 1:1
# translation carries through — YouTube omits these. ≥2 of the same letter so
# a legitimate lone "n" survives and no real word can match.
_GRUNT_CUE_RE = re.compile(r"(?:n{2,}|m{2,})[.…!?\s]*$", re.IGNORECASE)
# Duplicate-cue policy. A SUBSTANTIAL line (a real sentence/phrase) should appear
# once: a verbatim repeat far apart is almost always Whisper repetition /
# hallucination on non-speech audio (music, moans, silence), faithfully carried
# through the 1:1 translation — the "same English line shows up at 0:14 and 3:16"
# symptom. SHORT lines ("Yes.", "Okay?", "No no") can legitimately recur, so a
# couple of copies are allowed there. Markers ("[♪ music ♪]") are never deduped —
# music genuinely plays at several points.
_MAX_DUPLICATES = 2
# A normalized cue this long (chars) is treated as substantial → collapsed to a
# single occurrence rather than the 2-copy allowance.
_SUBSTANTIAL_DEDUP_CHARS = 16
# When a SUBSTANTIAL line recurs at least this many times it isn't "a duplicate
# to trim to one" — it's a Whisper hallucination LOOP (the same full sentence
# emitted at a dozen-plus timestamps over non-speech audio: moans, music,
# silence), carried 1:1 through translation. Keeping even one places a wrong line
# at a wrong time, so drop EVERY occurrence. 6+ verbatim repeats of a real
# sentence is vanishingly rare in genuine dialogue. Observed: a 4-line block
# repeated 16× across 64:00–114:00 of a mostly-non-speech video.
_GROSS_REPEAT_DROP = 6


def _cjk_ratio(text: str) -> float:
    t = text or ""
    cjk = base = 0
    for c in t:
        if ("぀" <= c <= "ヿ") or ("㐀" <= c <= "鿿") or ("가" <= c <= "힣") or ("ｦ" <= c <= "ﾟ"):
            cjk += 1
            base += 1
        elif c.isalpha():
            base += 1
    return (cjk / base) if base else 0.0


def _get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _is_marker(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("[") and t.endswith("]")


# ── Fragment-merge tuning ────────────────────────────────────────────────────
# A cue that does NOT end with sentence-final punctuation is treated as an
# unfinished fragment and folded into the following cue when they're clearly one
# utterance. The caps keep a merged cue subtitle-readable and bound any
# over-merge of a run of unpunctuated lines.
_FRAG_GAP_MAX_S = 2.5     # max silence between two fragments of one utterance
_FRAG_LEN_MAX = 100       # max merged characters (≈ 2 subtitle lines)
_FRAG_DUR_MAX_S = 8.0     # max merged on-screen duration
_SENTENCE_FINAL = ".!?…。！？"      # a cue ending here is a complete thought
_TRAILING_CLOSERS = "\"'”’`)]》」』"  # peel these before checking the last char


def _ends_complete(text: str) -> bool:
    """True if ``text`` ends a sentence (so it should NOT absorb the next cue).

    Trailing quotes/brackets are peeled first so ``He left."`` still reads as
    complete. A comma/semicolon/word ending is NOT complete → a continuation."""
    t = (text or "").rstrip()
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()
    return (not t) or (t[-1] in _SENTENCE_FINAL)


def _as_rows(segments):
    rows = []
    for r in (segments or []):
        if hasattr(r, "model_dump"):
            rows.append(r.model_dump(mode="json"))
        elif isinstance(r, dict):
            rows.append(r)
        else:
            rows.append(dict(r))
    return rows


def merge_transcript_fragments(segments, target_lang: str = "en"):
    """Fold Whisper's mid-sentence fragment splits into whole utterances.

    Returns ``(rows, changed)`` (plain dicts). A cue is extended with the
    following cue(s) only while ALL hold: the running text is unfinished (no
    sentence-final punctuation), same speaker, gap ≤ ``_FRAG_GAP_MAX_S``,
    neither side a ``[marker]``, and the result stays within the length /
    duration caps. Complete lines ("How old are you now?") never absorb the
    next line. Strict fixed point. Fail-soft: returns the input on any error."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        # Space-joining is wrong for CJK (no inter-word spaces) — leave as-is.
        if tgt in _CJK_TARGETS or len(rows) < 2:
            return rows, False

        out = []
        i = 0
        n = len(rows)
        while i < n:
            base = rows[i]
            raw = base.get("text") or ""
            # A complete or marker cue never starts a merge — pass through byte
            # for byte so unchanged cues stay identical (no spurious churn).
            if _is_marker(raw.strip()) or _ends_complete(raw):
                out.append(dict(base))
                i += 1
                continue
            norm = " ".join(raw.split())
            start = float(base.get("start") or 0.0)
            end = float(base.get("end") or start)
            spk = base.get("speaker")
            words = list(base.get("words") or [])  # keep per-word timing aligned
            j = i + 1
            absorbed = False
            while j < n and not _ends_complete(norm) and not _is_marker(norm):
                nxt = rows[j]
                nxt_text = " ".join((nxt.get("text") or "").split())
                if not nxt_text or _is_marker(nxt_text):
                    break
                if nxt.get("speaker") != spk:
                    break
                ns = float(nxt.get("start") or 0.0)
                ne = float(nxt.get("end") or ns)
                if ns - end > _FRAG_GAP_MAX_S:
                    break
                joiner = "" if norm.endswith("-") else " "
                cand = norm + joiner + nxt_text
                if len(cand) > _FRAG_LEN_MAX:
                    break
                if ne - start > _FRAG_DUR_MAX_S:
                    break
                norm, end, j, absorbed = cand, ne, j + 1, True
                words += list(nxt.get("words") or [])
            if absorbed:
                merged = dict(base)
                merged["text"], merged["start"], merged["end"] = norm, start, end
                # Span the per-word timestamps across all folded fragments so a
                # merged cue's karaoke highlight stays correct (stale partial
                # ``words`` from only the first fragment would mis-highlight).
                merged["words"] = words or None
                out.append(merged)
                i = j
            else:
                out.append(dict(base))
                i += 1
        return out, (len(out) != len(rows))
    except Exception:
        return _as_rows(segments), False


# ── Run-on cue split (YouTube-style one-thought-per-cue) ────────────────────
# Official subs put ONE sentence/thought per cue; Whisper+1:1 translation can
# cram 3-4 sentences into an 11-second cue ("The capsule has altered its
# course. Does it have suicidal tendencies? If it burns out… I suppose that's
# about right."), which reads far worse than the reference. Split such cues at
# sentence boundaries, allocating time by character share.
_RUNON_MAX_CHARS = 50          # a cue past ~one 42-char line is a run-on (fallback)
_RUNON_MIN_PIECE_S = 1.0       # never create a cue shorter than this
_RUNON_MAX_PIECES = 6          # a cue never explodes into confetti (fallback)
_RUNON_MIN_PIECE_CHARS = 16    # never orphan a sub-readable fragment onto its own cue
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'‘“(\[]?[A-Z0-9])")
# Strong intra-sentence clause boundaries — split AFTER the punctuation, at the
# following whitespace, so tokens are never cut in half (the word-partition
# invariant stays exact). Deliberately NOT conjunction WORDS ("and"/"that"/…):
# a punctuation-only rule keeps 1:1 word partitions intact and avoids
# over-fragmenting mid-phrase.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:—–])\s+")


def _clause_units(sentence: str) -> list[str]:
    """Break one sentence into clause units at strong punctuation boundaries.
    Concatenates (with single spaces) back to the input, so token counts are
    preserved. Returns ``[sentence]`` when there is no clause boundary."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
    return parts or [sentence]


def _pack_units(units: list[str], budget: int, min_chars: int) -> list[str]:
    """Greedily pack clause units into ≤ ``budget``-char pieces, never closing a
    piece while it is still shorter than ``min_chars`` (so a 7-char clause like
    "M Plan," is absorbed into its neighbour instead of orphaned onto its own
    cue). A single over-budget unit becomes its own piece."""
    pieces: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) >= min_chars and (len(cur) + 1 + len(u) > budget):
            pieces.append(cur)
            cur = u
        else:
            cur = (cur + " " + u).strip()
    if cur:
        if pieces and len(cur) < min_chars:
            pieces[-1] = (pieces[-1] + " " + cur).strip()
        else:
            pieces.append(cur)
    return pieces


def split_run_on_cues(segments, target_lang: str = "en"):
    """Split run-on cues into YouTube-style one-thought-per-cue pieces.

    Returns ``(rows, changed)`` (plain dicts). A cue is split when it is a
    non-CJK, non-``[marker]`` cue at least ``2 × _RUNON_MIN_PIECE_S`` long AND
    it either runs past one subtitle line (``TRANSCRIPT_RUNON_MAX_CHARS``) or
    carries ≥ 2 finished sentences. Splitting rules:

      * **One sentence per piece** — two finished thoughts are never welded into
        one cue (this is also what keeps a second pass a no-op: each emitted
        piece is a single ≤-line sentence and re-fires nothing).
      * **Clause splitting** — a single sentence longer than one line is broken
        at strong clause punctuation (``, ; : — –``), packed back up to the
        line budget. Only done when the cue's words are 1:1 with its text (tier
        A/B) so each piece can carry exact word timings; word-less (tier C)
        cues stay sentence-only, timed char-proportionally.
      * **Timing** — piece boundaries land on the real word start when word
        timings are present (no char-proportional drift at clause cuts),
        otherwise by character share. Word timestamps are partitioned into
        their piece by token count. Every piece keeps ≥ ``_RUNON_MIN_PIECE_S``.

    Fail-soft: returns the input unchanged on any error."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or not rows:
            return rows, False
        try:
            from backend.config import settings as _s
            max_chars = int(getattr(_s, "TRANSCRIPT_RUNON_MAX_CHARS", _RUNON_MAX_CHARS))
            max_pieces = int(getattr(_s, "TRANSCRIPT_RUNON_MAX_PIECES", _RUNON_MAX_PIECES))
            clause_split = bool(getattr(_s, "TRANSCRIPT_RUNON_CLAUSE_SPLIT", True))
        except Exception:
            max_chars, max_pieces, clause_split = _RUNON_MAX_CHARS, _RUNON_MAX_PIECES, True
        out = []
        changed = False
        for seg in rows:
            text = (seg.get("text") or "").strip()
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            dur = end - start
            if _is_marker(text) or dur < 2 * _RUNON_MIN_PIECE_S or not text:
                out.append(seg)
                continue
            sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
            # A run-on is a cue that spills past one line OR carries ≥2 finished
            # thoughts (official subs give each its own cue regardless of length).
            if not sentences or not (len(text) > max_chars or len(sentences) >= 2):
                out.append(seg)
                continue

            words = list(seg.get("words") or [])
            token_partition = bool(words) and len(text.split()) == len(words)

            # Build pieces: one sentence per piece; a long single sentence is
            # clause-split (only when word-timed, to avoid scrambling tier-C
            # karaoke). Never weld two sentences together.
            pieces: list[str] = []
            for s in sentences:
                if clause_split and token_partition and len(s) > max_chars:
                    units = _clause_units(s)
                else:
                    units = [s]
                pieces.extend(_pack_units(units, max_chars, _RUNON_MIN_PIECE_CHARS))
            if len(pieces) < 2:
                out.append(seg)
                continue

            # Confetti cap: rein in clause-fragment explosion, but NEVER weld two
            # COMPLETE sentences — welding whole thoughts would (a) re-manufacture
            # a run-on and (b) re-split on the next pass (the welded piece carries
            # ≥2 sentence marks), breaking idempotency. So only merge a piece whose
            # LEFT is an unfinished clause fragment; when every piece is already a
            # whole sentence we stop and let each keep its own cue (one thought per
            # cue, regardless of count). The min-duration floor is handled below in
            # time allocation, so it no longer forces a sentence weld here.
            while len(pieces) > max_pieces:
                cand = [i for i in range(len(pieces) - 1) if not _ends_complete(pieces[i])]
                if not cand:
                    break
                j = min(cand, key=lambda i: len(pieces[i]) + len(pieces[i + 1]))
                pieces[j] = (pieces[j] + " " + pieces[j + 1]).strip()
                del pieces[j + 1]

            piece_tokens = [len(p.split()) for p in pieces]
            token_partition = bool(words) and sum(piece_tokens) == len(words)

            # Word-timed contiguous boundaries when 1:1 words exist: each cut
            # lands on the real start of the next piece's first word.
            bounds = None
            if token_partition:
                cand = [start]
                cum = 0
                ok = True
                for k in range(len(pieces) - 1):
                    cum += piece_tokens[k]
                    w = words[cum]
                    ws = w.get("start") if isinstance(w, dict) else getattr(w, "start", None)
                    if ws is None or ws != ws:   # `ws != ws` rejects NaN
                        ok = False
                        break
                    cand.append(min(end, max(cand[-1], float(ws))))
                if ok:
                    cand.append(end)
                    bounds = cand

            total_chars = sum(len(p) for p in pieces) or 1
            n = len(pieces)
            # Feasible per-piece floor: honor _RUNON_MIN_PIECE_S when the cue is
            # long enough to give every piece that much, else fall back to an
            # equal share (dur / n) so a dense multi-thought cue still yields
            # short-but-nonzero cues instead of a zero-length one. n·floor ≤ dur
            # by construction, so every window below is non-empty.
            floor = min(_RUNON_MIN_PIECE_S, dur / n) if n else 0.0
            w_off = 0
            t = start
            for k, p in enumerate(pieces):
                tail = n - 1 - k
                if k == n - 1:
                    p_end = end
                else:
                    if bounds is not None:
                        p_end = bounds[k + 1]
                    else:
                        p_end = t + dur * (len(p) / total_chars)
                    # Reserve the floor for every remaining piece, then honor
                    # this piece's own floor — so no cue is starved and the last
                    # piece (fixed to ``end``) still clears the floor.
                    p_end = min(p_end, end - tail * floor)
                    p_end = max(p_end, t + floor)
                p_end = min(end, max(p_end, t))
                piece_row = dict(seg)
                piece_row["text"] = p
                piece_row["start"], piece_row["end"] = round(t, 3), round(p_end, 3)
                if token_partition:
                    piece_row["words"] = words[w_off:w_off + piece_tokens[k]] or None
                    w_off += piece_tokens[k]
                else:
                    piece_row["words"] = None
                out.append(piece_row)
                t = p_end
            changed = True
        return out, changed
    except Exception:
        return _as_rows(segments), False


_THEME_OPEN_LABEL = "[♪ Opening theme ♪]"
_THEME_END_LABEL = "[♪ Ending theme ♪]"
_PREVIEW_RE = re.compile(
    r"\b(next (?:episode|time)|next,?\s+on|to be continued|preview|deathscythe)\b", re.I)
_CHORUS_MIN_CHARS = 12
_LYRIC_MAX_CHARS = 48
_THEME_MIN_SPAN_S = 12.0


def _song_norm(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _sent_split(text: str) -> list:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", (text or "").strip()) if p.strip()]


def _proper_noun_count(text: str) -> int:
    """Capitalized, mid-sentence tokens — a preview/dialogue signal (a sung
    lyric line rarely names characters). Sentence-initial caps don't count."""
    n, sent_start = 0, True
    for tok in (text or "").split():
        w = tok.strip(".,!?;:\"'()[]…—–“”’")
        if not sent_start and len(w) > 1 and w[:1].isupper() and not w.isupper():
            n += 1
        sent_start = bool(tok) and tok[-1] in ".!?…"
    return n


def collapse_song_choruses(segments, target_lang: str = "en"):
    """Collapse a sung OPENING/ENDING theme — mis-transcribed as duplicated,
    garbled dialogue — into a single ``[♪ … theme ♪]`` marker, the way official
    subtitles do.

    The signal is chorus REPETITION inside the head/tail windows: a normalized
    sentence that recurs ≥2× is a chorus line (spoken dialogue does not repeat
    whole sentences within a couple of minutes). The collapsed run is anchored
    to the actual chorus cues, so it can never eat surrounding dialogue, and a
    next-episode preview narrated over the ending theme (flagged by preview
    keywords or ≥2 proper nouns) is preserved. Returns ``(rows, changed)``;
    CJK targets, short transcripts, and non-repeating windows are passed
    through unchanged. Fail-soft."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or len(rows) < 6:
            return rows, False
        try:
            from backend.config import settings as _s
            if not getattr(_s, "TRANSCRIPT_MARK_THEME_SONGS", True):
                return rows, False
            head_s = float(getattr(_s, "TRANSCRIPT_THEME_HEAD_S", 150.0))
            tail_s = float(getattr(_s, "TRANSCRIPT_THEME_TAIL_S", 210.0))
            min_rep = int(getattr(_s, "TRANSCRIPT_THEME_MIN_REPEATS", 2))
        except Exception:
            head_s, tail_s, min_rep = 150.0, 210.0, 2

        def _st(r):
            return float(r.get("start") or 0.0)

        def _en(r):
            return float(r.get("end") or _st(r))

        first_t = _st(rows[0])
        last_t = max(_en(r) for r in rows)
        windows = [(first_t, first_t + head_s, _THEME_OPEN_LABEL),
                   (last_t - tail_s, last_t, _THEME_END_LABEL)]

        def _is_preview(txt):
            return bool(_PREVIEW_RE.search(txt)) or _proper_noun_count(txt) >= 2

        drop = set()
        marker_at = {}
        for w0, w1, label in windows:
            idxs = [i for i, r in enumerate(rows)
                    if (r.get("text") or "").strip()
                    and not _is_marker((r.get("text") or "").strip())
                    and w0 - 0.01 <= _st(r) <= w1 + 0.01]
            if len(idxs) < 4:
                continue
            counts = {}
            for i in idxs:
                for s in _sent_split(rows[i].get("text") or ""):
                    ns = _song_norm(s)
                    if len(ns) >= _CHORUS_MIN_CHARS:
                        counts[ns] = counts.get(ns, 0) + 1
            chorus = {ns for ns, c in counts.items() if c >= 2}
            if len(chorus) < min_rep:
                continue   # no repeated chorus → not a song window; leave alone
            chorus_idxs = [i for i in idxs
                           if not _is_preview((rows[i].get("text") or "").strip())
                           and any(_song_norm(s) in chorus
                                   for s in _sent_split(rows[i].get("text") or ""))]
            if len(chorus_idxs) < 3:
                continue
            lo, hi = min(chorus_idxs), max(chorus_idxs)
            run = []
            for i in idxs:
                if not (lo <= i <= hi):
                    continue
                txt = (rows[i].get("text") or "").strip()
                if _is_preview(txt):
                    continue   # preserve preview/dialogue interleaved in the run
                is_chorus = any(_song_norm(s) in chorus for s in _sent_split(txt))
                lyric_like = _proper_noun_count(txt) == 0 and len(txt) <= _LYRIC_MAX_CHARS
                if is_chorus or lyric_like:
                    run.append(i)
            if len(run) < 3:
                continue
            m_start = min(_st(rows[i]) for i in run)
            m_end = max(_en(rows[i]) for i in run)
            if m_end - m_start < _THEME_MIN_SPAN_S:
                continue
            drop.update(run)
            marker_at[lo] = (m_start, m_end, label)

        if not drop:
            return rows, False
        out = []
        for i, r in enumerate(rows):
            if i in marker_at:
                s, e, lab = marker_at[i]
                base = dict(r)
                base.update({"text": lab, "start": round(s, 3), "end": round(e, 3),
                             "words": None})
                out.append(base)
            if i in drop:
                continue
            out.append(dict(r))
        return out, True
    except Exception:
        return _as_rows(segments), False


def sanitize_translated_transcript(segments, target_lang: str = "en"):
    """Return ``(cleaned_rows, changed)``.

    ``cleaned_rows`` are plain dicts. ``changed`` is True when anything was
    dropped/reordered, so callers can persist the repair only when needed.
    Fail-soft: on any error the original rows are returned unchanged.
    """
    try:
        rows = list(segments or [])
        if not rows:
            return rows, False
        tgt = (target_lang or "").strip().lower().split("-")[0]
        # A CJK target legitimately contains CJK — only de-dup there, never drop.
        drop_source_script = tgt not in _CJK_TARGETS

        # Pre-count normalized keys so a SUBSTANTIAL line that recurs ≥
        # _GROSS_REPEAT_DROP times can be recognized as a hallucination LOOP and
        # dropped entirely (not just trimmed to one). Markers + source-script
        # relapses are excluded from the count exactly as from the keep logic.
        totals: dict[str, int] = {}
        for seg in rows:
            text = (_get(seg, "text", "") or "").strip()
            if not text or _is_marker(text):
                continue
            if drop_source_script and _cjk_ratio(text) > 0.30:
                continue
            key = " ".join(text.lower().split())
            if len(key) >= _SUBSTANTIAL_DEDUP_CHARS:
                totals[key] = totals.get(key, 0) + 1

        kept = []
        counts: dict[str, int] = {}
        for seg in rows:
            text = (_get(seg, "text", "") or "").strip()
            if not text:
                continue
            # Standalone non-lexical grunt cue ("Nn...", "Mmm") — always-on
            # drop (the polisher's filler strip is off by default). The
            # ``fullmatch`` keeps it to a WHOLE-cue grunt, never a real cue
            # that merely ends in "…mm".
            if not _is_marker(text) and _GRUNT_CUE_RE.fullmatch(text):
                continue
            if drop_source_script and not _is_marker(text) and _cjk_ratio(text) > 0.30:
                continue  # source-language relapse — not part of a translation
            if not _is_marker(text):
                key = " ".join(text.lower().split())
                # A substantial line repeated many times over is a Whisper
                # hallucination loop — drop every copy, not just the excess.
                if len(key) >= _SUBSTANTIAL_DEDUP_CHARS and totals.get(key, 0) >= _GROSS_REPEAT_DROP:
                    continue
                n = counts.get(key, 0)
                # Substantial lines collapse to one; short lines keep up to 2.
                cap = 1 if len(key) >= _SUBSTANTIAL_DEDUP_CHARS else _MAX_DUPLICATES
                if n >= cap:
                    continue  # duplicate artifact (Whisper repetition / corruption)
                counts[key] = n + 1
            kept.append(seg.model_dump(mode="json") if hasattr(seg, "model_dump")
                       else dict(seg))

        kept.sort(key=lambda s: (float(s.get("start") or 0), float(s.get("end") or 0)))
        changed = (len(kept) != len(rows))
        if not changed:
            # Length unchanged but order might have been fixed — detect a reorder.
            for a, b in zip(kept, rows):
                bt = (_get(b, "text", "") or "")
                if (a.get("text") or "") != bt:
                    changed = True
                    break
        return kept, changed
    except Exception:
        return list(segments or []), False
