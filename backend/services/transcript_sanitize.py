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
"""
from __future__ import annotations

_CJK_TARGETS = {"ja", "ko", "zh", "zh-cn", "zh-tw", "yue"}
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

        kept = []
        counts: dict[str, int] = {}
        for seg in rows:
            text = (_get(seg, "text", "") or "").strip()
            if not text:
                continue
            if drop_source_script and not _is_marker(text) and _cjk_ratio(text) > 0.30:
                continue  # source-language relapse — not part of a translation
            if not _is_marker(text):
                key = " ".join(text.lower().split())
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
