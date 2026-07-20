"""Canonical-name resolution for the auto translation glossary.

The auto glossary (``backend/services/glossary.py``) mines recurring proper
nouns straight out of the Whisper transcript, so it faithfully locks in
Whisper's *mis-heard romaji* — a real Gundam Wing run shipped "Ririna / Leena
Dorian" (official: Relena Darlian), "Zex/Zekus" (Zechs), "Hero Yuu" (Heero
Yuy), "Katō" (Quatre) and so on, because the glossary told the translator to
keep those wrong spellings "consistent".

The video title ("MOBILE SUIT GUNDAM WING Episode 1") is available in job
metadata, and even a small local model (gemma3:12b on the Companion) knows the
series' canonical English names. This module makes ONE fail-soft LLM call that
maps each mined term to its canonical official English romanization, producing
a ``{detected: canonical}`` dict the glossary builder renders as
``"Ririna → Relena Darlian"`` entries — so the translation LLM converges on
the right spelling instead of the mis-heard one.

Design constraints:
  * ONE call per job (module-level cache keyed by job_id), ~45 s timeout,
    ``skip_circuit_breaker=True`` so a failure never degrades the provider
    chain for the actual translation.
  * Fail-soft everywhere: any error → ``{}`` and the glossary ships as-is.
  * Defensive parsing: only accept a JSON object; drop non-string values,
    identity mappings, sentence-like values, invented keys (keys must come
    from the input term list), profane values, and duplicate canonical values
    whose sources are not obvious variants of each other (e.g. "Zex" and
    "Zekus" may both → "Zechs", but "Shuttle" → "Zechs" is dropped).
  * Non-name terms ("shuttle", "colony", "capsule") pass through unmapped
    unless the LLM explicitly (and plausibly) maps them.

Config knobs (all optional, read via ``getattr(settings, ..., default)``):
  * ``TRANSLATION_CANONICAL_NAMES: bool = True`` — master switch.
  * ``TRANSLATION_CANONICAL_NAMES_TIMEOUT: float = 45.0`` — LLM call timeout.
  * ``TRANSLATION_CANONICAL_NAMES_MAX_TERMS: int = 40`` — cap on terms sent.
  * ``TRANSLATION_CANONICAL_NAMES_MODEL: str = ""`` — optional
    ``model_override`` for the orchestrator call.
"""
from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re

logger = logging.getLogger("clipai.canonical_names")

# ── Defaults (overridable via settings, see module docstring) ──
_DEF_TIMEOUT = 45.0
_DEF_MAX_TERMS = 40

# Canonical values longer than this are "the model wrote a sentence", not a
# name — official romanizations are 1-4 words ("Relena Darlian", "Mobile Suit
# Gundam Wing" edge-cases included).
_MAX_VALUE_WORDS = 4
_MAX_VALUE_CHARS = 60

# Deny-heuristic: never emit a canonical value containing profanity (a
# hallucinating model must not be able to inject slurs into the translation
# prompt as an "official name"). Word-boundary match, case-insensitive.
_PROFANITY = {
    "fuck", "fucking", "shit", "bitch", "cunt", "asshole", "bastard",
    "dick", "cock", "pussy", "whore", "slut", "nigger", "faggot", "retard",
}
_PROFANITY_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(_PROFANITY)) + r")\b",
    re.IGNORECASE,
)

# ── Per-job in-module cache: repeat calls within a job are free. Failures are
# cached too ({}) — this is a single-shot pre-translation step and retrying a
# dead/slow provider mid-job would just stall the pipeline again. ──
_CACHE: dict[str, dict[str, str]] = {}
_CACHE_MAX = 64


def clear_cache() -> None:
    """Test hook / memory hygiene."""
    _CACHE.clear()


def _cache_put(key: str, value: dict[str, str]) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        try:
            _CACHE.pop(next(iter(_CACHE)))
        except Exception:
            _CACHE.clear()
    _CACHE[key] = value


def _settings():
    try:
        from backend.config import settings
        return settings
    except Exception:
        return None


def _clean_terms(auto_terms, max_terms: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in auto_terms or []:
        term = str(raw or "").strip()
        key = term.lower()
        if not term or len(term) > 80 or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _similar(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are plausibly the same name mis-heard —
    "Zex"/"Zechs", "Zekus"/"Zechs". Used only for the duplicate-value guard."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 3 and len(nb) >= 3 and (na.startswith(nb) or nb.startswith(na)):
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.5


def _parse_json_object(raw: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not raw or not isinstance(raw, str):
        return None
    data = None
    try:
        from backend.services.providers.base import extract_json
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = None
    if data is None:
        try:
            start, end = raw.index("{"), raw.rindex("}")
            parsed = json.loads(raw[start:end + 1])
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            return None
    if data is None:
        return None
    # Unwrap a single wrapper key ({"mappings": {...}}) when the top level
    # carries no string values itself.
    if data and not any(isinstance(v, str) for v in data.values()):
        for v in data.values():
            if isinstance(v, dict):
                return v
    return data


def _sanitize_mapping(raw: dict, terms: list[str]) -> dict[str, str]:
    """Reduce a raw LLM dict to safe ``{detected_term: canonical}`` entries.

    Drops: keys not in ``terms`` (never invent names), non-string values,
    identity mappings, sentence-like / over-long values, profane values, and
    colliding canonical values whose sources are not obvious variants.
    """
    by_lower = {t.lower(): t for t in terms}
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        src = by_lower.get(k.strip().lower())
        if not src:
            continue  # invented / hallucinated key — not one of our terms
        val = " ".join(v.strip().split())
        if not val or len(val) > _MAX_VALUE_CHARS:
            continue
        if len(val.split()) > _MAX_VALUE_WORDS:
            continue  # looks like a sentence/explanation, not a name
        if val.lower() == src.lower():
            continue  # identity mapping adds nothing
        if _PROFANITY_RE.search(val):
            continue
        out[src] = val

    # Duplicate-canonical guard: two sources may share a canonical value only
    # when they are obvious variants (of the value or of each other).
    groups: dict[str, list[str]] = {}
    for src, val in out.items():
        groups.setdefault(val.lower(), []).append(src)
    for _val_key, srcs in groups.items():
        if len(srcs) < 2:
            continue
        keep = [s for s in srcs if _similar(s, out[s])]
        changed = True
        while changed:
            changed = False
            for s in srcs:
                if s not in keep and any(_similar(s, k) for k in keep):
                    keep.append(s)
                    changed = True
        for s in srcs:
            if s not in keep:
                out.pop(s, None)
    return out


# Filenames that identify nothing ("videoplayback.mp4", "download (3).mkv",
# "movie.mp4"…). A real production run passed exactly this — the model had no
# series anchor and phonetically romanized every name (リリーナ→"Reirina"
# instead of the official "Relena"). Treat these as NO title so the prompt
# leans on the terms themselves to identify the work.
_GENERIC_TITLE_RE = re.compile(
    r"^(?:videoplayback|video|movie|film|clip|download|untitled|output|"
    r"recording|export|final|sample|media|track|episode|ep)?"
    r"[\s\-_\.\(\)\[\]#\d]*$",
    re.IGNORECASE,
)


def _informative_title(video_title: str) -> str:
    """The title stripped of its extension, or '' when it identifies nothing."""
    t = (video_title or "").strip()
    t = re.sub(r"\.(mp4|mkv|avi|mov|webm|m4v|ts|flv|wmv)$", "", t,
               flags=re.IGNORECASE).strip()
    if not t or _GENERIC_TITLE_RE.match(t):
        return ""
    return t


def _build_prompt(terms: list[str], video_title: str, series_hint: str) -> str:
    title = _informative_title(video_title)
    hint = (series_hint or "").strip()
    context_bits = []
    if title:
        context_bits.append(f'This is a video titled "{title}".')
    if hint:
        context_bits.append(f"Series/context hint: {hint}.")
    context_bits.append(
        "First, silently identify which published work (anime, film, series, "
        "or game) this transcript is from — use the title/hint if given, and "
        "the distinctive detected terms below either way (recurring character "
        "and mecha names usually identify a work unambiguously).")
    context = " ".join(context_bits)
    terms_json = json.dumps(terms, ensure_ascii=False)
    return (
        f"{context}\n"
        "The following terms were automatically detected in its speech "
        "transcript. Some are names of characters, people, places, machines, "
        "ships, or organizations from this work that the speech recognizer "
        "romanized incorrectly; others are ordinary words.\n\n"
        f"DETECTED TERMS: {terms_json}\n\n"
        "For each detected term that you recognize as a name from this work, "
        "map it to its canonical official English romanization (the spelling "
        "used in the official English release).\n"
        "Return ONLY a JSON object mapping detected term to canonical "
        'spelling, e.g. {"Ririna": "Relena", "Zekus": "Zechs"}.\n'
        "Rules:\n"
        "- Only map names when you have CONFIDENTLY identified the specific "
        "work. If you cannot identify it, return {} — never guess.\n"
        "- Keys MUST be terms copied exactly from DETECTED TERMS. NEVER "
        "invent or add names that are not in the list.\n"
        "- If you do not recognize a term, or it is an ordinary word (e.g. "
        '"shuttle", "colony"), return it unchanged or omit it.\n'
        "- Values must be short proper names (1-4 words), never sentences or "
        "explanations.\n"
    )


async def resolve_canonical_names(
    auto_terms: list[str],
    video_title: str,
    orchestrator,
    job_id: str = "",
    series_hint: str = "",
    model_override: str | None = None,
) -> dict[str, str]:
    """Map mined transcript terms to canonical official English names.

    ONE LLM call via ``orchestrator.text_completion`` (json_mode when
    supported, ``skip_circuit_breaker=True``, ~45 s timeout). Returns
    ``{detected_term: canonical_name}`` for the terms the model recognized;
    ``{}`` on any error, when disabled, or when there is nothing to anchor on
    (no terms, or no title AND no series hint). Cached in-module per job so
    repeat calls are free.
    """
    key = None
    try:
        s = _settings()
        if s is not None and not bool(getattr(s, "TRANSLATION_CANONICAL_NAMES", True)):
            return {}
        max_terms = int(getattr(s, "TRANSLATION_CANONICAL_NAMES_MAX_TERMS", _DEF_MAX_TERMS) or _DEF_MAX_TERMS) if s else _DEF_MAX_TERMS
        terms = _clean_terms(auto_terms, max_terms=max(1, max_terms))
        if not terms or orchestrator is None:
            return {}
        title = (video_title or "").strip()
        hint = (series_hint or "").strip()
        # A generic filename ("videoplayback.mp4") anchors nothing, but the
        # TERMS themselves usually identify the work — recurring character +
        # mecha names are close to a fingerprint, and the prompt's
        # confidence rule ("return {} unless you identified the work")
        # carries the no-guessing guarantee. Only bail when there's neither
        # an informative title nor enough distinctive terms to fingerprint.
        if not _informative_title(title) and not hint and len(terms) < 4:
            return {}

        key = f"job:{job_id}" if job_id else (
            "anon:" + title.lower() + "|" + "|".join(sorted(t.lower() for t in terms)))
        cached = _CACHE.get(key)
        if cached is not None:
            return dict(cached)

        timeout = float(getattr(s, "TRANSLATION_CANONICAL_NAMES_TIMEOUT", _DEF_TIMEOUT) or _DEF_TIMEOUT) if s else _DEF_TIMEOUT
        # Model priority: explicit config pin > the caller's translation model
        # (the biggest model in the rig — a 12B recognizes a series where the
        # small editorial default just romanizes phonetically) > chain default.
        model = str(getattr(s, "TRANSLATION_CANONICAL_NAMES_MODEL", "") or "").strip() if s else ""
        if not model and model_override:
            model = str(model_override).strip()

        prompt = _build_prompt(terms, title, hint)
        kwargs: dict = {
            "max_tokens": min(2048, 200 + 24 * len(terms)),
            "timeout": timeout,
            "job_id": job_id or "",
            "skip_circuit_breaker": True,
            "json_mode": True,
        }
        if model:
            kwargs["model_override"] = model
        try:
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt, **kwargs), timeout + 15)
        except TypeError:
            # Orchestrator variant without json_mode/model_override kwargs —
            # retry with the bare positional call (wait_for still bounds it).
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt), timeout + 15)

        data = _parse_json_object(raw or "")
        result = _sanitize_mapping(data, terms) if data else {}
        if result:
            logger.info(
                "[%s] canonical names resolved for %d/%d term(s): %s",
                job_id or "-", len(result), len(terms),
                "; ".join(f"{k}→{v}" for k, v in list(result.items())[:8]))
        _cache_put(key, result)
        return dict(result)
    except Exception as e:
        logger.debug("[%s] canonical name resolution skipped: %s", job_id or "-", e)
        if key is not None:
            _cache_put(key, {})
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Roster-based garble correction (translated track)
# ═══════════════════════════════════════════════════════════════════════════
# The canonical-names pass above fixes the terms the glossary MINED — but
# mining needs recurrence, so names that appear a handful of times in
# music-heavy scenes never get anchored. Whisper mis-hears them, the
# translator spells them phonetically, and the shipped subs read
# "Ail Reese" (Aries), "Gundarium" (Gundanium), "Hero Yui" (Heero Yuy),
# "Trowat" (Trowa), "Katru" (Quatre) — all observed on a real run whose
# glossary was otherwise correct.
#
# This second, fail-soft pass runs AFTER translation: mine name-like
# TitleCase tokens from the TRANSLATED text, hand them (with the already-
# resolved canonical mapping as series evidence) to the same model, and ask
# ONLY for confident corrections. The apply step is deterministic
# (word-boundary replaces) behind strict vetting, so a hallucinating model
# can rewrite nothing outside the candidate list.

# Words that are never garbled names no matter how they are capitalized.
_ROSTER_SAFE_WORDS = frozenset("""
the this that these those what where when why how who which i we you they he
she it but and then now well yes no not just so oh ah huh hey mm even if do
don is are was were will would can could should please speaker sir madam
miss mister mrs ms dr father mother captain general colonel lieutenant major
sergeant left right up down here there earth space okay ok
""".split())

# TitleCase run: 1-3 capitalized words (Latin incl. extended chars like ō),
# optional possessive stripped by the miner.
_TITLECASE_RUN_RE = re.compile(
    r"\b([A-Z][a-zÀ-ſ]+"
    r"(?:[-'’][A-Za-zÀ-ſ]+)*"
    r"(?:\s+[A-Z][a-zÀ-ſ]+(?:[-'’][A-Za-zÀ-ſ]+)*){0,2})")

_SENT_START_RE = re.compile(r"(?:^|[.!?…\"'“‘]\s*)$")


def _strip_possessive(tok: str) -> str:
    return re.sub(r"[’']s?$", "", tok)


def _mine_name_candidates(texts: list, max_candidates: int = 60) -> list[tuple[str, str]]:
    """Mine likely proper-noun tokens from translated cue texts.

    Returns ``[(token, example_line), …]`` most-frequent-first. Single
    TitleCase words count only when they appear MID-sentence at least once
    (sentence-initial capitalization proves nothing); multi-word runs count
    anywhere — and each constituent word is ALSO offered on its own, so
    "Recorder Trowat" surfaces "Trowat" too. Tokens that also appear as an
    ordinary lowercase word anywhere in the CASE-PRESERVED corpus are
    skipped (real words, not names), as are safelisted structural words.
    """
    # Case preserved: a token trivially matches itself in a lowercased
    # corpus, which silently disabled this filter's purpose.
    corpus = " " + " ".join(str(t) for t in texts) + " "

    def _lowercase_attested(word: str) -> bool:
        return re.search(
            r"(?<![A-Za-z])" + re.escape(word.lower()) + r"(?![A-Za-z])",
            corpus) is not None

    counts: dict[str, int] = {}
    examples: dict[str, str] = {}

    def _offer(tok: str, line: str) -> None:
        tok = _strip_possessive(tok.strip())
        if not tok or len(tok) < 3:
            return
        words = tok.split()
        if all(w.lower() in _ROSTER_SAFE_WORDS for w in words):
            return
        if len(words) == 1 and _lowercase_attested(tok):
            return
        counts[tok] = counts.get(tok, 0) + 1
        examples.setdefault(tok, line.strip()[:110])

    for raw in texts:
        line = str(raw or "")
        for m in _TITLECASE_RUN_RE.finditer(line):
            tok = _strip_possessive(m.group(1).strip())
            if not tok:
                continue
            words = tok.split()
            if len(words) == 1:
                # Sentence-initial single words are ambiguous — skip this
                # occurrence (a mid-sentence sighting elsewhere still counts).
                if _SENT_START_RE.search(line[: m.start(1)]):
                    continue
                _offer(tok, line)
            else:
                _offer(tok, line)
                # Constituents too — the phrase may be one real word plus one
                # garbled name ("Recorder Trowat"), and the model corrects
                # single tokens more reliably than mixed phrases.
                for w in words:
                    _offer(w, line)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(tok, examples.get(tok, "")) for tok, _n in ranked[:max_candidates]]


def _vet_roster_pairs(pairs, candidates: set) -> dict[str, str]:
    """Keep only corrections that are safe to auto-apply: the wrong form must
    be one of OUR mined candidates verbatim, the right form must look like a
    name (letters/spaces/apostrophes, ≤40 chars, ≤4 words), differ from the
    wrong form beyond case, and not be a safelisted ordinary word."""
    out: dict[str, str] = {}
    for p in (pairs or []):
        if not isinstance(p, dict):
            continue
        wrong = str(p.get("wrong") or "").strip()
        right = str(p.get("right") or "").strip()
        if not wrong or not right or wrong not in candidates:
            continue
        if len(right) > 40 or len(right.split()) > _MAX_VALUE_WORDS:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'’-]*", right):
            continue
        if right.lower() == wrong.lower():
            continue
        if right.lower() in _ROSTER_SAFE_WORDS:
            continue
        if _PROFANITY_RE.search(right):
            continue
        out[wrong] = right
    return out


def roster_corrections_for_job(job_id: str) -> dict[str, str]:
    """The roster mapping resolved for this job (empty when none ran)."""
    return dict(_CACHE.get(f"roster:{job_id}") or {}) if job_id else {}


def apply_roster_corrections(texts: list, mapping: dict[str, str]) -> tuple[list, int]:
    """Word-boundary replace each vetted wrong→right pair in each text.
    Longest wrong-forms first so "Ail Reese" wins over a hypothetical "Ail".
    Case-sensitive: only the exact mined surface is touched. Returns
    ``(new_texts, replacements)``."""
    if not mapping:
        return list(texts), 0
    ordered = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    total = 0
    out: list = []
    for raw in texts:
        line = str(raw or "")
        for wrong, right in ordered:
            line, n = re.subn(r"\b" + re.escape(wrong) + r"\b", right, line)
            total += n
        out.append(line)
    return out, total


def _parse_json_pairs(raw: str) -> list:
    """First JSON array of objects in ``raw`` (code fences tolerated)."""
    s = (raw or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", s)
    start = s.find("[")
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(s[start:i + 1])
                    return data if isinstance(data, list) else []
                except Exception:
                    return []
    return []


async def resolve_roster_corrections(
    texts: list,
    orchestrator,
    job_id: str = "",
    model_override: str | None = None,
) -> dict[str, str]:
    """ONE fail-soft LLM call mapping mined garbled tokens → canonical names.

    Requires the per-job canonical mapping (already resolved during
    translation, cached in this module) as series evidence — with no series
    identified the pass is too risky and returns ``{}``. Fail-soft: any
    error → ``{}`` and the transcript ships as-is."""
    try:
        s = _settings()
        if s is not None and not bool(getattr(s, "TRANSLATION_ROSTER_CORRECTIONS", True)):
            return {}
        if orchestrator is None or not texts:
            return {}
        series_map = dict(_CACHE.get(f"job:{job_id}") or {}) if job_id else {}
        if len(series_map) < 3:
            return {}
        candidates = _mine_name_candidates(texts)
        if not candidates:
            return {}
        # Terms already canonical need no second look — but keep them in the
        # prompt context; only ASK about the rest.
        known = {v.lower() for v in series_map.values()}
        ask = [(t, ex) for (t, ex) in candidates if t.lower() not in known]
        if not ask:
            return {}
        evidence = "; ".join(f"{k} = {v}" for k, v in list(series_map.items())[:20])
        cand_block = "\n".join(f'- "{t}"  (e.g. “{ex}”)' for t, ex in ask)
        prompt = (
            "You are repairing machine subtitles for one specific episode. "
            "Speech recognition mis-heard some Japanese proper nouns and the "
            "translator spelled them phonetically.\n"
            f"Canonical terms already verified for this episode: {evidence}. "
            "These identify the series precisely.\n\n"
            "For each candidate token below, decide whether it is a GARBLED "
            "rendering of a character, mecha, faction, place or term from "
            "this series. Return ONLY the corrections you are confident "
            "about, as a JSON array of objects "
            '[{"wrong": "<token exactly as listed>", "right": "<official '
            'English spelling>"}]. Omit tokens that are already correct, '
            "are ordinary words, or that you are unsure about. Output only "
            "the JSON array.\n\n"
            f"Candidates:\n{cand_block}"
        )
        # 150s default: the call lands right after translation + condensation
        # on the SAME busy 12B — the original 75s ceiling timed out on a real
        # run, cascaded through a dead cloud provider (402), and the names
        # shipped garbled with a scary provider-fallback warning in the UI.
        timeout = float(getattr(s, "TRANSLATION_ROSTER_TIMEOUT", 150.0) or 150.0) if s else 150.0
        # Grammar-level schema, not plain json_mode: format=json biases the
        # model toward a bare OBJECT while this prompt needs an ARRAY of
        # pairs — the run-45 pass silently produced zero corrections on
        # exactly that mismatch. The schema forces the array shape on both
        # Ollama and OpenRouter; providers without schema support fall back
        # to the parser guard as before.
        kwargs: dict = {
            "max_tokens": min(2048, 200 + 24 * len(ask)),
            "timeout": timeout,
            "job_id": job_id or "",
            "skip_circuit_breaker": True,
            "json_schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["wrong", "right"],
                    "properties": {"wrong": {"type": "string"},
                                   "right": {"type": "string"}},
                },
                "maxItems": max(1, len(ask)),
            },
        }
        if model_override:
            kwargs["model_override"] = str(model_override).strip()
        # When the editorial provider is local Ollama, keep this garnish call
        # local: a timeout must degrade to "no corrections", not cascade
        # through cloud fallbacks (observed: a 75s timeout marched into a
        # 402-dead OpenRouter and surfaced a provider-fallback warning).
        try:
            _prov = (orchestrator.get_editorial_model_info() or {}).get("provider")
            if str(_prov or "").lower() == "ollama":
                kwargs["local_only"] = True
        except Exception:
            pass
        try:
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt, **kwargs), timeout + 15)
        except TypeError:
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt), timeout + 15)
        mapping = _vet_roster_pairs(
            _parse_json_pairs(raw or ""), {t for t, _ in ask})
        if not mapping:
            # Visible at INFO: a silent no-op here shipped "Gundarium" after
            # the infrastructure was in place — diagnosability matters more
            # than log quiet.
            logger.info(
                "[%s] roster corrections: %d candidate(s) offered, none "
                "confidently corrected (raw reply %d chars)",
                job_id or "-", len(ask), len(raw or ""))
        if mapping:
            logger.info(
                "[%s] roster corrections resolved for %d token(s): %s",
                job_id or "-", len(mapping),
                "; ".join(f"{k}→{v}" for k, v in list(mapping.items())[:10]))
        # Stash per job: downstream copy generators (clip SEO titles /
        # descriptions / tags, summaries) re-apply the same deterministic
        # corrections so a model can't re-introduce the garbled spellings
        # the transcript pass just fixed.
        if job_id:
            _cache_put(f"roster:{job_id}", dict(mapping))
        return mapping
    except Exception as e:
        logger.debug("[%s] roster correction skipped: %s", job_id or "-", e)
        return {}
