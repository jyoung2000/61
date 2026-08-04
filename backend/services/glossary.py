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


def build_recurring_terms_block(
    terms: list, target_lang: str = "the target language",
    canonical_map: Optional[dict] = None, protected_terms: Optional[list] = None,
) -> str:
    """Prompt block that pins recurring proper nouns to one consistent rendering.

    Empty string when there's nothing recurring, so it adds nothing to the
    prompt for content without recurring names.

    ``canonical_map`` (``{detected_term: canonical_name}``, e.g. from
    ``backend.services.canonical_names.resolve_canonical_names``) upgrades a
    mis-heard auto term to a ``"Ririna → Relena Darlian"`` entry — same comma
    list, same slot, so the translation LLM converges on the canonical
    spelling instead of Whisper's romaji. ``protected_terms`` (the user's
    custom-vocabulary spellings) are NEVER rewritten: user terms always win
    over canonical rewrites. Lookups are case-insensitive; identity mappings
    are ignored.
    """
    terms = [str(t).strip() for t in (terms or []) if str(t).strip()]
    if not terms:
        return ""
    cmap = {
        str(k).strip().lower(): str(v).strip()
        for k, v in (canonical_map or {}).items()
        if str(k or "").strip() and str(v or "").strip()
    }
    protected = {str(t or "").strip().lower() for t in (protected_terms or [])}
    entries: list[str] = []
    mapped_any = False
    for t in terms:
        canon = cmap.get(t.lower())
        if canon and t.lower() not in protected and canon.lower() != t.lower():
            entries.append(f"{t} → {canon}")
            mapped_any = True
        else:
            entries.append(t)
    # Join with ", " NOT " · ": a small model that fails a cue sometimes echoes
    # this very template back as its "translation", and a middot-joined list is
    # the exact word-salad garble we then have to detect + repair downstream.
    # A comma list carries the same meaning with no salad template to mimic.
    joined = ", ".join(entries)
    canon_rule = (
        "Entries written as \"detected → canonical\" mean the left-hand "
        "spelling is a mis-transcription: ALWAYS write that name using the "
        "canonical right-hand spelling, never the left-hand one.\n"
    ) if mapped_any else ""
    return (
        "RECURRING NAMES & TERMS — keep these consistent:\n"
        f"  {joined}\n"
        + canon_rule +
        "Each of these appears several times in this video. Translate each one "
        f"the SAME way every time. Treat them as proper nouns: transliterate names "
        f"and coined terms into {target_lang} rather than translating them into "
        "ordinary words, and never swap one for a different name or invent a new "
        "one.\n\n"
    )


def load_custom_vocabulary_terms() -> list[str]:
    """The user's global custom-vocabulary glossary (authoritative proper-noun
    spellings), or ``[]``. Lazy + fail-soft + gated by ``CUSTOM_VOCABULARY_ENABLED``
    so this module stays importable without the settings / vocabulary deps and
    degrades to the auto-derived list when the feature is off or unreadable."""
    try:
        from backend.config import settings
        if not bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)):
            return []
        from backend.services.custom_vocabulary import load_vocabulary
        return list(load_vocabulary() or [])
    except Exception:
        return []


def merge_glossary_terms(user_terms, auto_terms, cap: int = 40) -> list[str]:
    """User terms first (authoritative), then auto-derived terms not already
    present — case-insensitive dedupe, order-preserving, capped at ``cap``."""
    out: list[str] = []
    seen: set[str] = set()
    for t in list(user_terms or []) + list(auto_terms or []):
        term = str(t or "").strip()
        key = term.lower()
        if not term or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= cap:
            break
    return out


def build_translation_glossary_block(
    segments, source_lang: str = "", target_lang: str = "the target language",
    max_terms: int = 40, user_terms: Optional[list] = None,
    canonical_map: Optional[dict] = None,
    extra_terms: Optional[list] = None,
) -> str:
    """Extract + format the recurring-terms glossary in one call (empty string
    when there is nothing to pin).

    ``user_terms`` supplies the user's authoritative canonical spellings (the
    global custom vocabulary). When ``None`` they are auto-loaded, so EVERY
    translation path gets the user's names without threading them through — a
    name that Whisper mis-heard, that appears only once, or that is katakana in
    the source (so the model would guess the romanization) is still pinned to
    the exact spelling the user typed. User terms rank first and the merged list
    is capped at ``max_terms`` to keep the per-batch prompt bounded.

    ``canonical_map`` (``{detected_term: canonical}``, typically from
    ``backend.services.canonical_names.resolve_canonical_names`` fed with the
    video title) rewrites mis-heard AUTO terms as ``"detected → canonical"``
    entries in the block. User terms are passed through as the protected set,
    so a spelling the user typed is never overridden by the LLM's canonical
    guess. The cap is unchanged: the map only re-renders entries already in
    the merged, capped list — it never adds terms.

    ``extra_terms`` are the exception to "never adds": terms with AUTHORITATIVE
    canonical mappings (wiki-mined katakana readings) that the recurrence miner
    can't see because they appear only once — which is exactly the case that
    needs pinning most (a name said once has no in-transcript consistency to
    fall back on; the model just guesses a romanization). They rank after the
    auto terms and stay under the same cap."""
    if user_terms is None:
        user_terms = load_custom_vocabulary_terms()
    auto = extract_recurring_terms(segments, source_lang, max_terms=max_terms)
    if extra_terms:
        auto = list(auto) + [t for t in extra_terms if t]
    merged = merge_glossary_terms(user_terms, auto, cap=max_terms)
    return build_recurring_terms_block(
        merged, target_lang, canonical_map=canonical_map,
        protected_terms=user_terms)
