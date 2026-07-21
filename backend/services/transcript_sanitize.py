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
