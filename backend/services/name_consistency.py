"""Auto name-consistency for offline translations.

Small offline MT models (NLLB / FuguMT) spell the same recurring proper noun
several different ways across one video — the Gundam Wing pass produced
"Doria" / "Dorian" / "Lilyna Doria" for one character, "Zechs" / "Zex" for
another, "Aries" / "Airys" / "Airy" for a unit. The user can't pre-list these
(they don't know what's in the video until it's transcribed), so this derives
the fixes automatically from the translated output: cluster near-identical
recurring proper-noun spellings and rewrite the rare variants to the dominant
one.

It's deliberately conservative — a variant is only rewritten when a clearly
*dominant* and *highly similar* form exists — so two genuinely distinct names
("Leo" vs "Leon") are left alone. Pure / deterministic, output-only: it can
never break the transcript; worst case it's a no-op. It makes names
CONSISTENT, not necessarily *official* (offline models don't know the
canonical spelling) — which is the best automatic result possible offline.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

# A single capitalised proper-noun token: Initial cap + ≥2 more letters, so
# "Doria"/"Zechs"/"Aries" match but "I"/"OK"/"Mr" don't. Apostrophes/hyphens
# allowed inside (O'Brien, Anne-Marie).
_PROPER = re.compile(r"\b[A-Z][a-zA-Z][a-zA-Z'’\-]+\b")

# Capitalised words that open sentences but aren't names — never cluster these.
_STOP = {
    "the", "this", "that", "these", "those", "there", "then", "they", "their",
    "them", "what", "when", "where", "which", "who", "why", "how", "and", "but",
    "for", "are", "you", "your", "yes", "not", "now", "well", "okay", "his",
    "her", "she", "him", "our", "out", "all", "any", "can", "got", "had", "has",
    "have", "was", "were", "with", "will", "would", "could", "should", "from",
    "just", "like", "into", "over", "only", "they're", "i'm", "i'll", "don't",
}


def _seg_text(seg) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", "") or "")
    return str(getattr(seg, "text", "") or "")


def _set_seg_text(seg, text):
    if isinstance(seg, dict):
        seg = dict(seg)
        seg["text"] = text
        return seg
    try:
        seg = seg.model_copy(update={"text": text})  # pydantic v2
        return seg
    except Exception:
        try:
            setattr(seg, "text", text)
        except Exception:
            pass
        return seg


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_consistency_map(
    segments,
    min_total: int = 3,
    sim_threshold: float = 0.82,
    dominance: float = 2.0,
) -> dict[str, str]:
    """Derive ``{rare_variant: dominant_spelling}`` rewrites from the output.

    A variant V is mapped to a dominant form D only when, within a cluster of
    highly-similar (``sim_threshold``) recurring tokens sharing a first letter:
      * the cluster recurs (total count ≥ ``min_total``), and
      * D clearly dominates V (count[D] ≥ ``dominance`` × count[V]).
    This rewrites misspellings toward the majority spelling without collapsing
    two co-dominant, genuinely different names. Pure / deterministic.
    """
    counts: Counter[str] = Counter()
    for seg in segments or []:
        for m in _PROPER.findall(_seg_text(seg)):
            if m.lower() in _STOP:
                continue
            counts[m] += 1
    if not counts:
        return {}

    forms = sorted(counts, key=lambda f: (-counts[f], f))
    # Greedy clustering: attach each form to the first earlier (more frequent)
    # form it's similar enough to and shares a first letter with.
    parent: dict[str, str] = {}
    for i, f in enumerate(forms):
        for g in forms[:i]:
            if f[:1].lower() == g[:1].lower() and _similar(f, g) >= sim_threshold:
                parent[f] = parent.get(g, g)
                break

    clusters: dict[str, list[str]] = {}
    for f in forms:
        root = parent.get(f, f)
        clusters.setdefault(root, []).append(f)

    mapping: dict[str, str] = {}
    for root, members in clusters.items():
        if len(members) < 2:
            continue
        total = sum(counts[m] for m in members)
        if total < min_total:
            continue
        dominant = max(members, key=lambda m: counts[m])
        d_count = counts[dominant]
        for m in members:
            if m == dominant:
                continue
            # Only rewrite a clearly-minority variant toward the dominant.
            if d_count >= dominance * counts[m]:
                mapping[m] = dominant
    return mapping


def unify_proper_noun_variants(segments, **kwargs) -> tuple[list, int]:
    """Rewrite rare proper-noun variants to their dominant spelling across the
    transcript. Returns ``(new_segments, n_rewrites)``. No-op (original list,
    0) when nothing qualifies."""
    seg_list = list(segments or [])
    if not seg_list:
        return seg_list, 0
    mapping = build_consistency_map(seg_list, **kwargs)
    if not mapping:
        return seg_list, 0
    # Longest source first so a full form is replaced before a substring of it.
    patterns = [
        (re.compile(rf"\b{re.escape(src)}\b"), dst)
        for src, dst in sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    ]
    out, n = [], 0
    for seg in seg_list:
        text = _seg_text(seg)
        new = text
        for pat, dst in patterns:
            new, k = pat.subn(dst, new)
            n += k
        out.append(_set_seg_text(seg, new) if new != text else seg)
    return out, n
