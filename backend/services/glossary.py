"""Auto-derive a per-video glossary of recurring proper nouns for translation.

Small local models (and, under load, cloud ones) garble recurring names and
invent alternates — the same character came out "Relena", "Lillian", "Liliana"
and "Lily" in one pass, and coined nouns got translated into ordinary words
("Leo"→"X-rays", "Deathscythe"→"god of death"). Feeding the translator the list
of terms that recur in THIS video — and telling it to render each one the same
way every time and to transliterate rather than translate names — fixes both.

This is content-agnostic: it works for an anime, a podcast, a lecture, or a news
clip, in any source language, because it keys off generic proper-noun signals
(recurring capitalized words / acronyms in cased scripts; recurring katakana runs
in Japanese) rather than any hardcoded vocabulary.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

# Uppercase / lowercase letter ranges across the common cased scripts (Latin
# incl. accents, Cyrillic, Greek). Kept as character-class fragments so a
# "capitalized word" matches names in most languages, not just ASCII.
_UPPER = "A-ZÀ-ÖØ-ÞĀ-ſЀ-ЯΑ-Ω"
_LOWER = "a-zà-öø-ÿĀ-ſа-яα-ω"
_CAP_WORD = re.compile(rf"[{_UPPER}][{_LOWER}][{_LOWER}\d'’\-]*")
_ACRONYM = re.compile(r"[A-Z]{2,6}")
# Katakana runs (incl. the prolonged-sound mark + halfwidth) — the strongest
# proper-noun / loanword signal in Japanese source text.
_KATAKANA = re.compile(r"[ァ-ヺーｦ-ﾟ]{2,}")

# High-frequency cased words that start sentences but aren't proper nouns —
# excluded so the list stays names/places/orgs, not "The"/"And"/"Mr".
_STOPCAPS = {
    "the", "a", "an", "and", "but", "so", "if", "or", "as", "of", "to", "in",
    "on", "at", "it", "is", "be", "we", "you", "he", "she", "they", "this",
    "that", "these", "those", "there", "here", "now", "then", "when", "what",
    "why", "how", "who", "yes", "no", "not", "well", "oh", "okay", "ok", "yeah",
    "i", "i'm", "i'll", "mr", "ms", "mrs", "dr", "sir", "ma'am",
}


def _seg_text(seg) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", "") or "")
    return str(getattr(seg, "text", "") or "")


def extract_recurring_terms(
    segments, source_lang: str = "", max_terms: int = 40, min_count: int = 2
) -> list[str]:
    """Return proper-noun candidates that recur in the transcript, best-first.

    A term must appear at least ``min_count`` times to count (recurrence is what
    creates the consistency problem a glossary fixes; one-off mentions don't).
    Returns at most ``max_terms``.
    """
    blob = "\n".join(t for t in (_seg_text(s) for s in (segments or [])) if t)
    if not blob.strip():
        return []

    counts: Counter[str] = Counter()
    # Cased-script proper nouns (any source/target language).
    for m in _CAP_WORD.findall(blob):
        if m.lower() not in _STOPCAPS and len(m) >= 2:
            counts[m] += 1
    for m in _ACRONYM.findall(blob):
        if m.lower() not in _STOPCAPS:
            counts[m] += 1
    # Japanese names / coined terms via katakana.
    if (source_lang or "").lower().startswith("ja") or _KATAKANA.search(blob):
        for m in _KATAKANA.findall(blob):
            if 2 <= len(m) <= 20:
                counts[m] += 1

    # Case-insensitive dedupe, keeping the most common surface form, ranked by
    # total frequency then length (longer = more specific) for stable output.
    best: dict[str, tuple[int, str]] = {}
    for term, c in counts.items():
        key = term.lower()
        prev = best.get(key)
        if prev is None or c > prev[0]:
            best[key] = (counts[term], term)
    ranked = sorted(best.values(), key=lambda t: (t[0], len(t[1])), reverse=True)
    return [term for c, term in ranked if c >= min_count][:max_terms]


def build_recurring_terms_block(terms: list, target_lang: str = "the target language") -> str:
    """Prompt block that pins recurring proper nouns to one consistent rendering.

    Empty string when there's nothing recurring, so it adds nothing to the
    prompt for content without recurring names.
    """
    terms = [str(t).strip() for t in (terms or []) if str(t).strip()]
    if not terms:
        return ""
    joined = " · ".join(terms)
    return (
        "RECURRING NAMES & TERMS — keep these consistent:\n"
        f"  {joined}\n"
        "Each of these appears several times in this video. Translate each one "
        f"the SAME way every time. Treat them as proper nouns: transliterate names "
        f"and coined terms into {target_lang} rather than translating them into "
        "ordinary words, and never swap one for a different name or invent a new "
        "one.\n\n"
    )


def build_translation_glossary_block(
    segments, source_lang: str = "", target_lang: str = "the target language",
    max_terms: int = 40,
) -> str:
    """Convenience: extract + format in one call (empty string when nothing recurs)."""
    return build_recurring_terms_block(
        extract_recurring_terms(segments, source_lang, max_terms=max_terms),
        target_lang,
    )
