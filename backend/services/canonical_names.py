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
import os
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
# Consonant-skeleton agreement required of any katakana → English mapping.
# 0.6 sits in the empty band between the two measured populations: correct
# romanizations score 0.86-1.00, the junk mapping that motivated the check
# scores 0.33.
_SKELETON_FLOOR = 0.6

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


# ── Durable store: keep a RESOLVED name map across runs ──────────────────
# The in-memory cache is keyed per job, so re-running the same video asks the
# model again — and the answer varies. Two consecutive runs of one episode
# resolved 6 names and then 0: the first produced "Relena"/"Zechs"/"Gundanium",
# the second "Lilyana"/"Sixes"/"Eries". Persisting a SUCCESSFUL map under a
# content fingerprint makes names deterministic run-over-run and lets a title
# improve once and stay improved. Only non-empty results are stored, so a
# transient provider failure can't be cached forever.
_PERSIST_NAME = "canonical_names.json"
_PERSIST_MAX = 200
# Bump to retire every stored map. v1 could hold guesses made with no
# informative title and no series hint. v2 dropped those entirely; v3 keeps
# them again but marked ``provisional`` so they act as a fallback rather than
# an authority. v2 entries are read forward as authoritative.
_PERSIST_VERSION = 3


def _persist_path() -> str:
    """Path to the durable store, or "" when there is nowhere durable to write.

    Deliberately ONLY the deployment volume — no home-directory fallback. A
    library that writes to a global path outside a real deployment makes runs
    order-dependent on whatever a previous run left behind: an earlier version of
    this store fell back to ~/.clipai, and a test that resolved names then wrote
    an entry that the NEXT run read back, so the model was never called and the
    test failed depending on execution order. Restricting the store to the mounted
    volume makes persistence a property of deployments, not of the library."""
    docker_dir = "/data/logs"
    if not os.path.isdir(docker_dir):
        return ""
    return os.path.join(docker_dir, _PERSIST_NAME)


def _persist_load() -> dict[str, dict]:
    """Fail-soft read of the durable store ({} on any problem).

    Returns ``{key: {"map": {...}, "provisional": bool}}``. A v2 entry was a
    bare mapping and was only ever written for an ANCHORED resolution, so it
    reads back as authoritative.
    """
    try:
        path = _persist_path()
        if not path or not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # A schema version, so a store written under older rules can be
        # discarded wholesale. Needed because entries were previously persisted
        # even when the resolution was a terms-only GUESS: a real deployment has
        # wrong names ("Aires" for Aries, "Dorian" for Darlian) pinned on disk
        # and reused on every run. Bumping the version retires them; without
        # this, the only cure is deleting the file by hand.
        ver = int(data.get("v") or 0) if isinstance(data, dict) else 0
        if ver not in (2, _PERSIST_VERSION):
            return {}
        entries = data.get("entries")
        if not isinstance(entries, dict):
            return {}
        out: dict[str, dict] = {}
        for k, v in entries.items():
            if not isinstance(v, dict):
                continue
            if ver == 2:
                out[k] = {"map": {str(a): str(b) for a, b in v.items()},
                          "provisional": False}
                continue
            m = v.get("map")
            if isinstance(m, dict) and m:
                out[k] = {"map": {str(a): str(b) for a, b in m.items()},
                          "provisional": bool(v.get("provisional"))}
        return out
    except Exception:
        return {}


def _persist_put(key: str, value: dict[str, str], provisional: bool = False) -> None:
    """Store a non-empty resolved map. Never raises.

    ``provisional`` marks a map resolved from the mined terms alone (no
    informative filename, no operator hint). Those are not authoritative — the
    model is guessing which work this is — so they never short-circuit a later
    run's own resolution. They are still worth keeping: without them a run
    whose model call comes back empty ships raw phonetic romanization
    ("Zekus", "Lilyana", "Hero") where the previous run shipped the official
    spellings, and the track's names change every time it is re-processed.
    Storing them turns that into a floor: this run's answer wins when it has
    one, and last run's answer covers it when it does not.
    """
    if not key or not value:
        return
    try:
        path = _persist_path()
        if not path:
            return                      # no deployment volume -> no persistence
        entries = _persist_load()
        entries[key] = {"map": dict(value), "provisional": bool(provisional)}
        if len(entries) > _PERSIST_MAX:            # drop oldest insertions
            for k in list(entries)[:len(entries) - _PERSIST_MAX]:
                entries.pop(k, None)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"v": _PERSIST_VERSION, "entries": entries}, f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _content_key(title: str, terms: list[str]) -> str:
    """Fingerprint a resolution request by what it actually depends on — the
    title and the mined terms — so the same video reuses the same answer
    instead of re-asking under a fresh job id."""
    return "sig:" + (title or "").strip().lower() + "|" + "|".join(
        sorted(t.strip().lower() for t in (terms or []) if t and t.strip()))


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
        # A name is not an initial and not a label. A measured run's
        # second-chance resolver answered "Just" → "J" and "Love" → "L2",
        # then enforced both across the track. Neither the phonetic gate nor
        # the ordinary-word gate can see these: "J" is not an English word and
        # its lead class matches almost anything short.
        _alpha = [c for c in val if c.isalpha()]
        if len(_alpha) < 2 or any(c.isdigit() for c in val):
            logger.info(
                "canonical names: dropped %r → %r — a canonical name is not "
                "an initial or a label", src, val)
            continue
        if not _canonical_sounds_plausible(src, val):
            # A canonical mapping is a SPELLING of the same name, so it has to
            # sound like the katakana. Without this the model could answer with
            # any name from the series it thought it recognised: a real run
            # returned エアリーズ (eariizu — "Aries", a mobile suit) as
            # "Peacecraft" and レン as "Heero", and both shipped — the transcript
            # called Relena Darlian "Relena Peacecraft" and misattributed lines.
            logger.info(
                "canonical names: dropped %r → %r — the English name does not "
                "sound like the katakana (romaji %r)",
                src, val, _kana_to_romaji(src))
            continue
        # The phonetic gate above only means anything for KATAKANA sources; a
        # term already in Latin script sails through it, and the model is then
        # free to "canonicalize" a garbled name into plain English vocabulary —
        # a real run mapped "Justlove" → "Justice", turning a mishearing into an
        # ordinary word that reads as dialogue, not a name. A Latin-script
        # source may only be RESPELLED (a near-variant); a value made entirely
        # of ordinary English words that isn't one is a guess, not a spelling.
        if _normalize(_kana_to_romaji(src)) == _normalize(src):
            _vwords = [w.strip(".,!?'\"’“”").lower() for w in val.split()]
            _vwords = [w for w in _vwords if w]
            # Stricter than ``_similar`` (0.5): "Justlove"/"Justice" share
            # enough letters to score 0.67, but a genuine RESPELLING keeps
            # nearly the whole word ("Uing"→"Wing" 0.75, "Dorian"→"Dorlian"
            # 0.92). Ordinary-word values must clear the respelling bar.
            _respelling = difflib.SequenceMatcher(
                None, _normalize(src), _normalize(val)).ratio() >= 0.75
            if (_vwords and all(w in _ROSTER_SAFE_WORDS
                                or w in _ROSTER_COMMON_WORDS for w in _vwords)
                    and not _respelling):
                logger.info(
                    "canonical names: dropped %r → %r — ordinary English "
                    "word(s) offered as the canonical form of a Latin-script "
                    "term (a respelling must resemble the source)", src, val)
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

        # Key on CONTENT (title + mined terms), not the job id. Keying per job
        # meant re-running the same video asked the model again, and the answer
        # varies: consecutive runs of one episode resolved 6 names and then 0,
        # shipping "Relena"/"Zechs"/"Gundanium" one time and
        # "Lilyana"/"Sixes"/"Eries" the next. A content key makes the same video
        # reuse the same answer, and the durable store carries it across restarts.
        key = _content_key(title, terms)
        cached = _CACHE.get(key)
        if cached is not None:
            return dict(cached)
        # The durable store is a PIPELINE concern, so it is only consulted for a
        # real run (job_id set). A direct call with no job id — a unit test or an
        # ad-hoc probe — always asks the model, so behaviour can't depend on
        # whatever a previous run happened to leave on disk.
        anchored = bool(_informative_title(title) or hint)
        fallback: dict[str, str] = {}
        if job_id:
            entry = _persist_load().get(key) or {}
            stored = entry.get("map") or {}
            if stored and not entry.get("provisional"):
                _cache_put(key, stored)
                _publish_series_evidence(job_id, stored)
                logger.info(
                    "[%s] canonical names reused from the durable store: "
                    "%d mapping(s) — same title+terms as an earlier run, so the "
                    "names stay identical instead of being re-guessed",
                    job_id, len(stored))
                return dict(stored)
            # Provisional: keep it as a floor, but still ask this run's model.
            # A guess must not outrank a fresh answer; it must also not be
            # thrown away, or a run whose model call comes back empty ships
            # raw romanization where the last run shipped official spellings.
            fallback = dict(stored)

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
        # Second chance for the LEFTOVERS. A partial first answer is the
        # common real outcome (a measured run resolved 9/17 — Relena, Zechs,
        # OZ, Gundam — and left ヒイロ, ドーリアン, トロワ unresolved), and
        # the terms that stay unresolved are precisely the ones the
        # downstream garble corrector needs as ground truth: with no
        # "Heero Yuy" in the known-names set, the correction "Hero Yu" →
        # "Heero Yuy" is uncorroborated and dies in vetting, so the episode
        # ships phonetic romanizations of half its cast. Once ≥2 names ARE
        # resolved, the work is identified — re-asking about the remainder
        # with those anchors in-context turns a guess into retrieval. Every
        # answer still passes the katakana phonetic gate.
        if result and len(result) >= 2:
            # "Remaining" means UNANSWERED — a term the model mapped to itself
            # was recognized and dropped as an identity, and re-asking about it
            # ("Shuttle", "Colony") is pure waste. Only terms the raw reply
            # never mentioned get the second ask.
            _term_lower = {t.lower(): t for t in terms}
            _answered = set(result)

            def _mark_answered(d, depth=0):
                if not isinstance(d, dict) or depth > 2:
                    return
                for k, v in d.items():
                    if isinstance(v, dict):
                        _mark_answered(v, depth + 1)
                    elif isinstance(k, str) and k.strip().lower() in _term_lower:
                        _answered.add(_term_lower[k.strip().lower()])

            _mark_answered(data)
            _remaining = [t for t in terms if t not in _answered]
            if _remaining:
                try:
                    _extra = await _resolve_remaining_terms(
                        job_id, _remaining, result, orchestrator, timeout, model)
                    if _extra:
                        result.update(_extra)
                except Exception as _rr_e:
                    logger.debug("[%s] second-chance term resolution skipped: %s",
                                 job_id or "-", _rr_e)
        if not result:
            # Say so. A silent empty result is how a whole episode shipped
            # "Zekus"/"Lilyana"/"Hero" with nothing in the log to show that the
            # name pass had run at all, let alone why it produced nothing.
            logger.info(
                "[%s] canonical names: no usable mapping from %d term(s) "
                "(model returned %d raw key(s), %d survived vetting)%s",
                job_id or "-", len(terms), len(data or {}), 0,
                "" if not fallback else
                f" — falling back to {len(fallback)} mapping(s) from the "
                "durable store so the names match the previous run")
            if fallback:
                _cache_put(key, fallback)
                _publish_series_evidence(job_id, fallback)
                return dict(fallback)
        if result:
            logger.info(
                "[%s] canonical names resolved for %d/%d term(s): %s",
                job_id or "-", len(result), len(terms),
                "; ".join(f"{k}→{v}" for k, v in list(result.items())[:8]))
            # Persist only a SUCCESSFUL map, so a transient provider failure is
            # never cached forever and a later run can still resolve the names.
            # Pipeline runs only, matching the read gate above.
            #
            # AND only when the resolution had a real anchor. With an
            # uninformative filename and no operator hint we fall through above
            # on the mined terms alone, which is a GUESS — and a guess must not
            # become permanent. A real run proves the cost: from the title
            # "videoplayback.mp4" the model returned six names, two of them
            # wrong (Aries as "Aires", Darlian as "Dorian"). Persisted, those
            # came back on every later run as "reused from the durable store",
            # and once the roster pass started working it began ENFORCING them
            # (correcting "Dorien" to the wrong "Dorian"). A determinism store
            # that pins a wrong answer is worse than re-asking.
            #
            # A terms-only resolution is stored PROVISIONALLY (see
            # ``_persist_put``): it never short-circuits a later run's own
            # resolution, so a wrong guess can't outlive the next model call,
            # but it does cover a run whose call comes back empty. Dropping it
            # entirely — which is what v2 did — cost real quality: consecutive
            # runs of the same episode shipped "Relena"/"Zechs"/"Heero" and
            # then "Lilyana"/"Zekus"/"Hero", because the second had nothing to
            # fall back on.
            if job_id:
                _persist_put(key, result, provisional=not anchored)
            if job_id and not anchored:
                logger.info(
                    "[%s] canonical names stored PROVISIONALLY — resolved from "
                    "mined terms with no informative title and no series hint, "
                    "so the map is a guess and every later run re-asks. Set "
                    "TRANSLATION_SERIES_HINT (or give the file a descriptive "
                    "name) to make it authoritative.",
                    job_id)
        _cache_put(key, result)
        _publish_series_evidence(job_id, result)
        return dict(result)
    except Exception as e:
        logger.debug("[%s] canonical name resolution skipped: %s", job_id or "-", e)
        if key is not None:
            _cache_put(key, {})
        return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Series-hint → ASR bias roster (source-side name fix)
# ═══════════════════════════════════════════════════════════════════════════
# The name-repair passes above and below are ALL downstream of Whisper and gated
# on recurrence/consistency — so mis-hearings that romanize DIFFERENTLY every
# time (Zechs → Zecks/Zexes/"Sex Unique") slip through every one of them. The
# only place to break that is the source: expand the operator's series hint into
# the work's real proper nouns ONCE and bias the ASR with them, so Whisper emits
# the names correctly + identically in the first place (hotwords only nudge the
# decoder — they cannot fabricate a subtitle). Then the downstream chain
# converges for free. Fail-soft: any error → [] and transcription is unchanged.

# Roster cache keyed by the normalized hint (a series roster is stable across
# every episode, so this is reused for a whole show).
_ROSTER_CACHE: dict[str, list[str]] = {}


def series_roster_terms() -> list[str]:
    """The ASR-bias roster for the currently configured series hint, or [].

    Synchronous + side-effect-free: reads the cache populated by
    ``expand_series_hint_to_names`` (called once at job start). Returns [] when
    no hint is set or the expansion hasn't run/succeeded — i.e. today's exact
    behaviour."""
    try:
        s = _settings()
        hint = str(getattr(s, "TRANSLATION_SERIES_HINT", "") or "").strip() if s else ""
        if not hint:
            return []
        return list(_ROSTER_CACHE.get(hint.lower(), []))
    except Exception:
        return []


async def expand_series_hint_to_names(
    series_hint: str, orchestrator, job_id: str = "",
    model_override: str | None = None,
) -> list[str]:
    """Expand a series hint into ≤~30 canonical proper nouns for ASR biasing.

    ONE fail-soft, per-hint-cached LLM call (``skip_circuit_breaker=True``).
    A KNOWLEDGE call — safe here BECAUSE the output only biases the decoder
    (hotwords can't invent a subtitle), unlike the deterministic replacers
    where knowledge mode is banned. Returns [] on any error/empty/disabled."""
    hint = str(series_hint or "").strip()
    if not hint:
        return []
    ckey = hint.lower()
    if ckey in _ROSTER_CACHE:
        return list(_ROSTER_CACHE[ckey])
    try:
        s = _settings()
        if orchestrator is None:
            return []
        if s is not None and not bool(getattr(s, "CUSTOM_VOCABULARY_ENABLED", True)):
            return []
        timeout = float(getattr(s, "TRANSLATION_CANONICAL_NAMES_TIMEOUT", _DEF_TIMEOUT) or _DEF_TIMEOUT) if s else _DEF_TIMEOUT
        model = str(getattr(s, "TRANSLATION_CANONICAL_NAMES_MODEL", "") or "").strip() if s else ""
        if not model and model_override:
            model = str(model_override).strip()
        prompt = (
            f'List the principal proper nouns from "{hint}" — main character '
            "names, mecha/vehicle/ship names, organizations/factions, and place "
            "names — using their official English spellings.\n"
            "Return ONLY a JSON array of short strings (1-3 words each), at most "
            '30 entries, e.g. ["Heero Yuy", "Relena Darlian", "Zechs Merquise", '
            '"Gundam", "OZ"]. No commentary. If you do not confidently know the '
            "work, return []."
        )
        kwargs: dict = {
            "max_tokens": 700, "timeout": timeout, "job_id": job_id or "",
            "skip_circuit_breaker": True, "json_mode": True,
        }
        if model:
            kwargs["model_override"] = model
        try:
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt, **kwargs), timeout + 15)
        except TypeError:
            raw = await asyncio.wait_for(
                orchestrator.text_completion(prompt), timeout + 15)
        names = _parse_json_string_list(raw or "")
        roster = _clean_terms(names, max_terms=30)
        _ROSTER_CACHE[ckey] = roster
        if roster:
            logger.info(
                "[%s] series-hint roster for %r: %d name(s): %s",
                job_id or "-", hint, len(roster), ", ".join(roster[:12]))
        return list(roster)
    except Exception as e:
        logger.debug("[%s] series-hint expansion skipped: %s", job_id or "-", e)
        _ROSTER_CACHE[ckey] = []
        return []


def _parse_json_string_list(raw: str) -> list[str]:
    """Parse a JSON array of strings from a possibly fenced/prose-wrapped reply."""
    try:
        txt = (raw or "").strip()
        if "```" in txt:
            import re as _re
            m = _re.search(r"```(?:json)?\s*(.+?)```", txt, _re.S)
            if m:
                txt = m.group(1).strip()
        start = txt.find("[")
        end = txt.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        data = json.loads(txt[start:end + 1])
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data if str(x or "").strip()]
    except Exception:
        return []


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

# Ordinary English vocabulary that surfaces as TitleCase candidates — a word
# opening a sentence, or a plain noun inside a "Colony Summit"-style phrase —
# but is NEVER a garbled proper noun worth correcting. When EVERY word of a
# candidate is structural (above) or common (here), the candidate is noise and
# is withheld from the model, so the handful of real garbles (Gundarium,
# Katul, Hero Yuu) aren't drowned in a 50-token list a busy local model just
# skims and abandons. Deliberately generic — anything series-specific (a
# character, mecha, faction, place) must NOT appear here, since a listed word
# is never offered for correction. Worst case of a wrong entry is fail-safe:
# the token is simply not corrected and ships as-is, never mis-corrected.
_ROSTER_COMMON_WORDS = frozenset("""
especially humanity human inform information report reports eastern western
northern southern academy school federation republic empire kingdom nation
alliance colony colonies headquarters base station port harbor summit council
meeting conference civilian civilians combat battle war peace justice power
control victory defeat mission operation weapon weapons target enemy enemies
pilot pilots soldier soldiers officer commander force forces army navy fleet
squadron division battalion troop troops unit units carrier ship shuttle
capsule fighter machine gun radar surveillance data analysis evidence salvage
search rescue attack defense assault backup reinforcement satellite meteor
meteorite atmosphere spaceport airport spaceship area zone sector region part
parts log logs wing area everyone everything something anything nothing
someone anyone perhaps maybe suddenly finally probably already almost
morning evening tonight today tomorrow yesterday minister government official
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
    return [(tok, examples.get(tok, ""), n) for tok, n in ranked[:max_candidates]]


def _roster_worth_asking(token: str, known_words: set) -> bool:
    """True when ``token`` is worth showing the correction model.

    Withholds two kinds of noise so the real garbles aren't buried:
    (1) any candidate whose words are ALL structural/common English — a
    sentence-opening ordinary word ("Especially", "Humanity") or a plain
    descriptive phrase ("Colony Summit", "Combat Log") is never a mis-heard
    name; (2) any candidate containing an already-correct canonical word
    ("Miss Relena", "Oz's Zechs", "New Mobile Report") — the canonical term is
    right, the phrase is just it in context. Single unusual tokens
    ("Gundarium", "Katul", "Deathbringer") and mixed phrases that carry a
    genuinely foreign word ("Hero Yuu", "Am Wufe") pass through."""
    words = [_strip_possessive(w).lower() for w in str(token).split()]
    words = [w for w in words if w]
    if not words:
        return False
    if any(w in known_words for w in words):
        return False
    if all(w in _ROSTER_SAFE_WORDS or w in _ROSTER_COMMON_WORDS for w in words):
        return False
    return True


# Katakana → rough romaji. Only needs to be good enough to compare SOUNDS: a
# canonical English name is a romanization of the same name the katakana spells,
# so the two must be phonetically related. Digraphs first (longest match wins).
_KANA_ROMAJI = [
    ("キャ", "kya"), ("キュ", "kyu"), ("キョ", "kyo"), ("シャ", "sha"),
    ("シュ", "shu"), ("ショ", "sho"), ("チャ", "cha"), ("チュ", "chu"),
    ("チョ", "cho"), ("ニャ", "nya"), ("ニュ", "nyu"), ("ニョ", "nyo"),
    ("ヒャ", "hya"), ("ヒュ", "hyu"), ("ヒョ", "hyo"), ("ミャ", "mya"),
    ("ミュ", "myu"), ("ミョ", "myo"), ("リャ", "rya"), ("リュ", "ryu"),
    ("リョ", "ryo"), ("ギャ", "gya"), ("ギュ", "gyu"), ("ギョ", "gyo"),
    ("ジャ", "ja"), ("ジュ", "ju"), ("ジョ", "jo"), ("ビャ", "bya"),
    ("ビュ", "byu"), ("ビョ", "byo"), ("ピャ", "pya"), ("ピュ", "pyu"),
    ("ピョ", "pyo"), ("ティ", "ti"), ("ディ", "di"), ("トゥ", "tu"),
    ("ドゥ", "du"), ("ファ", "fa"), ("フィ", "fi"), ("フェ", "fe"),
    ("フォ", "fo"), ("ヴァ", "va"), ("ヴィ", "vi"), ("ヴェ", "ve"),
    ("ヴォ", "vo"), ("ウィ", "wi"), ("ウェ", "we"), ("ウォ", "wo"),
    ("ア", "a"), ("イ", "i"), ("ウ", "u"), ("エ", "e"), ("オ", "o"),
    ("カ", "ka"), ("キ", "ki"), ("ク", "ku"), ("ケ", "ke"), ("コ", "ko"),
    ("サ", "sa"), ("シ", "shi"), ("ス", "su"), ("セ", "se"), ("ソ", "so"),
    ("タ", "ta"), ("チ", "chi"), ("ツ", "tsu"), ("テ", "te"), ("ト", "to"),
    ("ナ", "na"), ("ニ", "ni"), ("ヌ", "nu"), ("ネ", "ne"), ("ノ", "no"),
    ("ハ", "ha"), ("ヒ", "hi"), ("フ", "fu"), ("ヘ", "he"), ("ホ", "ho"),
    ("マ", "ma"), ("ミ", "mi"), ("ム", "mu"), ("メ", "me"), ("モ", "mo"),
    ("ヤ", "ya"), ("ユ", "yu"), ("ヨ", "yo"),
    ("ラ", "ra"), ("リ", "ri"), ("ル", "ru"), ("レ", "re"), ("ロ", "ro"),
    ("ワ", "wa"), ("ヲ", "o"), ("ン", "n"),
    ("ガ", "ga"), ("ギ", "gi"), ("グ", "gu"), ("ゲ", "ge"), ("ゴ", "go"),
    ("ザ", "za"), ("ジ", "ji"), ("ズ", "zu"), ("ゼ", "ze"), ("ゾ", "zo"),
    ("ダ", "da"), ("ヂ", "ji"), ("ヅ", "zu"), ("デ", "de"), ("ド", "do"),
    ("バ", "ba"), ("ビ", "bi"), ("ブ", "bu"), ("ベ", "be"), ("ボ", "bo"),
    ("パ", "pa"), ("ピ", "pi"), ("プ", "pu"), ("ペ", "pe"), ("ポ", "po"),
    ("ヴ", "vu"), ("ァ", "a"), ("ィ", "i"), ("ゥ", "u"), ("ェ", "e"),
    ("ォ", "o"), ("ャ", "ya"), ("ュ", "yu"), ("ョ", "yo"), ("ッ", ""),
    ("ー", ""), ("・", " "),
]


def _kana_to_romaji(text: str) -> str:
    """Rough romaji for katakana input; non-kana characters pass through."""
    out, i, n = [], 0, len(text or "")
    while i < n:
        for kana, roman in _KANA_ROMAJI:
            if text.startswith(kana, i):
                out.append(roman)
                i += len(kana)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _canonical_sounds_plausible(term: str, value: str) -> bool:
    """True when ``value`` could be the official romanization of ``term``.

    A canonical mapping is a SPELLING change, not a translation: the English
    name and the katakana are the same name, so they must sound alike. Without
    this the model was free to answer with any name from the series it thought
    it recognised, and a real run returned エアリーズ (eariizu — "Aries", a
    mobile suit) as "Peacecraft" (a person's surname) and レン as "Heero". Both
    shipped: the transcript called Relena Darlian "Relena Peacecraft" and gave
    Zechs's lines to other characters.

    Only applied to katakana terms — a term already in Latin script is handled
    by the existing duplicate/ordinary-word rules — and deliberately lenient,
    since romanizations differ a lot ("Zechs" for ゼクス, "Heero" for ヒイロ).
    It only has to reject pairs that share almost no sound at all."""
    romaji = _normalize(_kana_to_romaji(term))
    if not romaji or romaji == _normalize(term):
        return True                      # not katakana → not our business
    target = _normalize(value)
    if not target:
        return True
    # The LEADING SOUND is what a romanization preserves. Compared by sound
    # class, not letter, because the same sound is spelled differently across
    # romanizations: earizu/Aries (both vowel-initial), katoru/Quatre (k/q),
    # koroni/Colony (k/c). A ratio alone is not enough — "marina" scores 0.5
    # against "Relena" and 0.36 against "Peacecraft", which is exactly how a
    # ship became a character in a shipped transcript.
    if _lead_class(romaji) != _lead_class(target):
        return False
    # The CONSONANT SKELETON must agree, whatever the full-string ratio says.
    # It used to be a fallback reached only when the ratio was low, which left
    # a hole: romaji and an unrelated English word can share plenty of letters
    # in a different order — オラゴン (oragon) scored 0.46 against "Emotion" and
    # sailed through, after which a junk mapping put a literal "(Emotion)" tag
    # into shipped subtitles. Measured over the real roster pairs the two
    # populations do not overlap: every correct romanization scores ≥0.86 on
    # the skeleton (hiiro/Heero, katoru/Quatre, earizu/Aries, sekusu/Zechs all
    # 0.86-1.00) while oragon/Emotion scores 0.33. Vowels drift between
    # romanization systems; consonants do not.
    sk_r = _consonant_skeleton(romaji)
    sk_t = _consonant_skeleton(target)
    if sk_r and sk_t and difflib.SequenceMatcher(
            None, sk_r, sk_t).ratio() < _SKELETON_FLOOR:
        logger.info(
            "canonical names: dropped %r → %r — consonants disagree "
            "(%r vs %r)", term, value, sk_r, sk_t)
        return False
    if difflib.SequenceMatcher(None, romaji, target).ratio() >= 0.22:
        return True
    # Vowel-heavy names can score under the ratio floor even when nearly
    # right: earizu vs "Aires" is 0.18, so the gate dropped a correct-but-
    # misspelled Aries and the mobile suit stayed garbled. Romanization
    # preserves the CONSONANT skeleton far more faithfully than the vowels
    # (which shift freely between systems), so compare that as a fallback —
    # class-normalized, same groups as the lead check (r/l, s/z, k/c/q…).
    # The skeleton is a FALLBACK for near-misses, not a bypass: consonant-poor
    # names reduce to 1-2 classes and trivially "match" (rei/"Law" both shrink
    # to nothing alike yet scored), so the full-string ratio must still clear
    # a low floor — 0.15 admits the motivating earizu/"Aires" (0.18) while a
    # zero-letter-overlap cross-name pair stays dead.
    if difflib.SequenceMatcher(None, romaji, target).ratio() < 0.15:
        return False
    sk_r = _consonant_skeleton(romaji)
    sk_t = _consonant_skeleton(target)
    return bool(sk_r and sk_t
                and difflib.SequenceMatcher(None, sk_r, sk_t).ratio() >= 0.6)


def _consonant_skeleton(s: str) -> str:
    """The consonants of ``s`` reduced to sound classes ("earizu" → "rs",
    "aires" → "rs"), so romanization vowel drift can't hide a match."""
    return "".join(_lead_class(c) for c in (s or "")
                   if c.isalpha() and c.lower() not in "aeiouwy")


def _lead_class(s: str) -> str:
    """The first sound of ``s`` reduced to a class, so equivalent spellings of
    one sound compare equal (k/c/q, r/l, b/v, g/j, and all vowels)."""
    c = (s or "")[:1].lower()
    if not c:
        return ""
    # w/y are glides: Japanese ウ/イ regularly romanize into them
    # (ウーフェイ -> "ufei" but written "Wufei"), so they group with the vowels.
    if c in "aeiouwy":
        return "V"
    for group in ("kcq", "rl", "bv", "gj", "sz", "fh"):
        if c in group:
            return group[0]
    return c


def _roster_phonetic_ok(wrong: str, right: str) -> tuple[bool, float]:
    """A correction is only credible when the two forms SOUND alike — an ASR
    mishearing, not a different name that fits the scene. A real run mapped
    "Hero Yuu"→"Trowa Barton" (wrong character at Heero's introduction),
    "Earth Sphere Alliance"→"Zeon" (different franchise) and
    "Operation Meteor"→"Operation Endgame" (broke a correct term). Shared
    leading/trailing words are stripped first so a common prefix
    ("Operation …") can't carry an unrelated core past the ratio."""
    ws, rs = wrong.split(), right.split()
    while ws and rs and ws[0].lower() == rs[0].lower():
        ws.pop(0)
        rs.pop(0)
    while ws and rs and ws[-1].lower() == rs[-1].lower():
        ws.pop()
        rs.pop()
    a = _normalize(" ".join(ws)) or _normalize(wrong)
    b = _normalize(" ".join(rs)) or _normalize(right)
    if not a or not b:
        return False, 0.0
    # An ASR mishearing keeps the LEADING sound. Whisper garbles interior vowels
    # and consonants ("Zecks"/"Zechs", "Airies"/"Aries", "Dorian"/"Darlian") but a
    # changed initial means a DIFFERENT name, not a misrecognition. A real run
    # rewrote "Marina" — the Alliance's salvage ship, used consistently three
    # times — into the character "Relena" because they rhyme and Relena was more
    # frequent, corrupting three cues into nonsense ("Alliance's Relena is trying
    # to recover that machine"). This one check also rejects three other recorded
    # failures: "Hero Yuu"→"Trowa Barton", "Earth Sphere Alliance"→"Zeon", and
    # "Operation Meteor"→"Operation Endgame" (whose shared first word is stripped
    # above, leaving Meteor vs Endgame).
    if a[:1] != b[:1]:
        return False, 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    ok = ratio >= 0.5 or (len(a) >= 4 and len(b) >= 4
                          and (a.startswith(b) or b.startswith(a)))
    return ok, ratio


def _word_is_near_typo(word: str) -> bool:
    """True when ``word`` is NOT a structural/common English word but sits
    within edit distance 1 of one — the signature of a model typo, not a
    proper noun. A real run "corrected" the CORRECT name "General Septem"
    into "Gneral Septem": 'gneral' is one transposition from 'general', and
    no real character name lives that close to an ordinary word."""
    w = word.lower()
    if not w or w in _ROSTER_SAFE_WORDS or w in _ROSTER_COMMON_WORDS:
        return False
    for vocab in (_ROSTER_SAFE_WORDS, _ROSTER_COMMON_WORDS):
        for known in vocab:
            if abs(len(known) - len(w)) > 1 or len(known) < 4:
                continue
            # Cheap edit-distance ≤1 check (incl. adjacent transposition).
            if len(known) == len(w):
                diffs = [i for i in range(len(w)) if w[i] != known[i]]
                if len(diffs) == 1:
                    return True
                if (len(diffs) == 2 and diffs[1] == diffs[0] + 1
                        and w[diffs[0]] == known[diffs[1]]
                        and w[diffs[1]] == known[diffs[0]]):
                    return True
            else:
                longer, shorter = (w, known) if len(w) > len(known) else (known, w)
                for i in range(len(longer)):
                    if longer[:i] + longer[i + 1:] == shorter:
                        return True
    return False


def _vet_roster_pairs(pairs, candidates: set,
                      frequent: set | None = None,
                      max_pairs: int = 8,
                      corpus: str | None = None,
                      known_names: set | None = None) -> dict[str, str]:
    """Keep only corrections that are safe to auto-apply: the wrong form must
    be one of OUR mined candidates verbatim, the right form must look like a
    name (letters/spaces/apostrophes, ≤40 chars, ≤4 words), differ from the
    wrong form beyond case, not be a safelisted/common ordinary word (a real
    run rewrote the song title "Justlove" into "Justice", corrupting every
    lyric line), not contain a near-typo of an ordinary word (the same run
    "corrected" the correct "General Septem" into "Gneral Septem"), SOUND
    like the wrong form (see ``_roster_phonetic_ok``), and not target a
    FREQUENT consistent term (a spelling used 4+ times is the glossary
    working, not a garble). Capped at the ``max_pairs`` phonetically-
    strongest pairs — a model that "corrects" everything is hallucinating.

    When ``corpus`` (the joined cue texts) is provided, the right form must
    ALSO be attested in it strictly more often than the wrong form. This
    makes the pass a CONSISTENCY tool — consolidate variant spellings toward
    the form the transcript itself uses most — instead of a knowledge tool.
    The first fast-model run proved the knowledge mode untrustworthy: it
    "corrected" the CORRECT "Duo" into the invented "Dewo" and
    "General Septem" into "General Septain", while consolidations like
    "Hero-kun" → the transcript's own dominant "Heero" are exactly what it
    gets right. Series knowledge belongs to the canonical glossary and the
    user's custom vocabulary, which don't hallucinate."""
    _corpus_l = (corpus or "").lower()

    def _attested(term: str) -> int:
        if not _corpus_l:
            return 0
        return len(re.findall(
            r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])", _corpus_l))

    scored: list[tuple[float, str, str]] = []
    for p in (pairs or []):
        if not isinstance(p, dict):
            continue
        wrong = str(p.get("wrong") or "").strip()
        right = str(p.get("right") or "").strip()
        if not wrong or not right or wrong not in candidates:
            continue
        if frequent and wrong in frequent:
            continue
        if len(right) > 40 or len(right.split()) > _MAX_VALUE_WORDS:
            continue
        if not re.fullmatch(r"[A-Za-z][A-Za-z .'’-]*", right):
            continue
        if right.lower() == wrong.lower():
            continue
        _right_words = [w for w in re.split(r"[ .'’-]+", right) if w]
        # A "correction" whose entire right side is ordinary English is a
        # rewrite, not a name fix — names get REPLACED by dictionary words
        # only when the model hallucinates (Justlove → Justice).
        if _right_words and all(
                w.lower() in _ROSTER_SAFE_WORDS or w.lower() in _ROSTER_COMMON_WORDS
                for w in _right_words):
            continue
        if any(_word_is_near_typo(w) for w in _right_words):
            continue
        if _PROFANITY_RE.search(right):
            continue
        ok, ratio = _roster_phonetic_ok(wrong, right)
        if not ok:
            continue
        # Attestation, and the one thing that can safely override it.
        #
        # Requiring the RIGHT spelling to already out-appear the wrong one makes
        # a genuine knowledge correction mathematically impossible: a name the
        # ASR never once got right is attested zero times. Traced on a real run,
        # Dorian→Darlian, Aires→Aries and Hero Yu→Heero Yuy each cleared mining,
        # the ask-list, the ordinary-word and near-typo guards and the phonetic
        # check (ratios 0.67-0.86), then all died here on 0-vs-N.
        #
        # But attestation cannot simply be dropped. It is also what caught the
        # inverse failure, recorded from a fast-model run: the pass UN-corrected
        # names that were already right — Duo→Dewo, General Septem→General
        # Septain, Katul→Kattul. Those are invented homophones, so they clear the
        # phonetic gate by construction, and they have zero attestation too.
        # Attestation alone cannot tell "never heard right" from "already right".
        #
        # What separates them is external ground truth about which spelling is a
        # real name: the series roster, the canonical map, the operator's custom
        # vocabulary. Darlian is in the roster; Dewo is in nothing. So an
        # unattested correction is allowed ONLY when its right-hand side is
        # corroborated by ``known_names``; otherwise attestation still rules.
        _att_r, _att_w = _attested(right), _attested(wrong)
        _corroborated = bool(known_names) and right.lower() in {
            str(k).lower() for k in known_names}
        if corpus is not None and _att_r <= _att_w and not _corroborated:
            continue
        # Keep the signal as ranking too: a consolidation toward the
        # transcript's own dominant spelling outranks an unattested correction
        # of equal phonetic closeness when ``max_pairs`` truncates.
        if corpus is not None and _att_r > _att_w:
            ratio += 0.25
        scored.append((ratio, wrong, right))
    scored.sort(key=lambda t: -t[0])
    return {w: r for _s, w, r in scored[:max_pairs]}


async def _resolve_remaining_terms(
    job_id: str,
    remaining: list[str],
    resolved: dict[str, str],
    orchestrator,
    timeout: float,
    model: str,
) -> dict[str, str]:
    """One follow-up call for the terms the first resolution left behind.

    The anchors make this retrieval, not guessing: the model is shown the
    names it already identified and asked to (1) name the specific work and
    (2) spell the remaining names the way the official English release does.
    Romaji is included per term so the model has the phonetic ground truth in
    Latin script. Results pass ``_sanitize_mapping`` — including the katakana
    phonetic gate, so a cross-name swap (エアリーズ → "Peacecraft") still
    dies here no matter how confident the model sounds."""
    if not remaining or orchestrator is None:
        return {}
    anchors = "; ".join(f"{k} = {v}" for k, v in list(resolved.items())[:10])
    lines = "\n".join(
        f"- {t}  (romaji: {_kana_to_romaji(t) or '?'})" for t in remaining[:12])
    prompt = (
        "All of these terms come from ONE Japanese anime episode. Names "
        f"already identified from it: {anchors}.\n\n"
        "First, identify the specific series these belong to. Then give the "
        "OFFICIAL English spelling — exactly as the official English release "
        "spells it — for each remaining name below. A term may be a "
        "character, mecha, faction, place or technology name. Omit any term "
        "you do not recognize, and never map a term to a DIFFERENT "
        "character's name.\n\n"
        f"Remaining terms:\n{lines}\n\n"
        'Reply with ONLY JSON: {"series": "<series name>", '
        '"mappings": {"<term>": "<official spelling>"}}'
    )
    kwargs: dict = {
        "max_tokens": 300 + 24 * len(remaining),
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
        raw = await asyncio.wait_for(
            orchestrator.text_completion(prompt), timeout + 15)
    data = _parse_json_object(raw or "") or {}
    series = str(data.get("series") or "").strip()
    mappings = data.get("mappings")
    extra = _sanitize_mapping(mappings, remaining) if isinstance(mappings, dict) else {}
    # Series identified → pull the OFFICIAL spellings and snap everything to
    # them. The model names the show correctly but misspells its cast (a
    # measured knowledge ceiling: "Hero", "Dorian", "Aires" four runs in a
    # row); spelling is retrieval, so retrieve.
    if series:
        try:
            _gnames = await _get_series_glossary(series, job_id)
        except Exception:
            _gnames = []
        if _gnames:
            _cache_put(f"glossary:{job_id}",
                       {n: n for n in _gnames[:60]})
            extra, _n_fix1 = _apply_glossary_spellings(extra, _gnames)
            _res_fix, _n_fix2 = _apply_glossary_spellings(resolved, _gnames)
            if _n_fix2:
                resolved.clear()
                resolved.update(_res_fix)
            _still = [t for t in remaining if t not in extra]
            _direct = _glossary_resolve_terms(_still, _gnames)
            if _direct:
                extra.update(_direct)
            if _n_fix1 or _n_fix2 or _direct:
                logger.info(
                    "[%s] canonical names: glossary enforced official "
                    "spellings — %d respelled, %d resolved directly from "
                    "the glossary", job_id or "-", _n_fix1 + _n_fix2,
                    len(_direct))
    if extra:
        logger.info(
            "[%s] canonical names: second chance resolved %d/%d leftover "
            "term(s)%s: %s",
            job_id or "-", len(extra), len(remaining),
            f" (series identified as {series!r})" if series else "",
            "; ".join(f"{k}→{v}" for k, v in list(extra.items())[:8]))
    else:
        logger.info(
            "[%s] canonical names: second chance resolved none of %d leftover "
            "term(s)%s", job_id or "-", len(remaining),
            f" (series identified as {series!r})" if series else "")
    return extra


# ── Authoritative series glossary ─────────────────────────────────────────
# The local models reliably IDENTIFY the series ("Gundam Wing") but cannot
# SPELL its names — qwen2.5:14b shipped "Hero Yu", "Dorian", "Kato" and
# "Aires" across four consecutive runs while naming the show correctly every
# time. Spelling is retrieval, not generation: once the series is known, one
# small Wikipedia fetch yields the official English spellings, cached in a
# mount-backed store so it happens at most once per series ever. Fail-soft
# throughout — no network, no glossary, names stay model-resolved.

_GLOSSARY_STORE_PATH = os.path.join(
    "/data/logs" if os.path.isdir("/data/logs") else
    os.path.join(os.path.expanduser("~"), ".clipai"),
    "series_glossaries.json")

_GLOSSARY_NAME_RE = re.compile(
    r"\b([A-Z][a-zA-Zà-ÿ'\-]{2,}(?:\s+[A-Z][a-zA-Zà-ÿ'\-]{2,}){0,2})\b")

# Wiki-prose capitalized words that are never character/mecha names. The
# roster word lists cover most ordinary vocabulary; these are the leftovers
# specific to encyclopedia text.
_WIKI_STOP = frozenset("""
january february march april may june july august september october november
december wikipedia english japanese japan america american north south west
east press media network animation studio director producer writer episode
episodes series season seasons volume volumes chapter chapters manga anime
television film movie ova soundtrack theme opening ending release released
adaptation adapted broadcast aired  original story plot character characters
list history production reception development staff cast voice actor actress
""".split())


def _glossary_store_load() -> dict:
    try:
        with open(_GLOSSARY_STORE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _glossary_store_save(store: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_GLOSSARY_STORE_PATH), exist_ok=True)
        with open(_GLOSSARY_STORE_PATH, "w", encoding="utf-8") as fh:
            json.dump(store, fh, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.debug("glossary store save failed: %s", e)


def _remember_series(store: dict, key: str) -> None:
    """Persist ``key`` as the series this machine most recently resolved.

    Only ever called with a key the store already holds names for, so the
    pointer can never dangle. This is what lets the NEXT run bias Whisper's
    decoder with the right cast list: identification runs AFTER transcription,
    so within a single run the names are already mangled by the time we learn
    whose they are. Written only on change — one disk write per new series,
    not one per job.
    """
    try:
        if store.get(_LAST_SERIES_STORE_KEY) == key:
            return
        store[_LAST_SERIES_STORE_KEY] = key
        _glossary_store_save(store)
    except Exception as e:
        logger.debug("glossary store series pointer failed: %s", e)


# Reserved store-key prefix: per-series katakana→official-name pairs mined
# from the same wiki text as the names list. Prefixed so it can never collide
# with a real series key.
_KANA_STORE_PREFIX = "__kana__:"

# "Heero Yuy (ヒイロ・ユイ, Hīro Yui)" — the character-list convention on the
# English wiki. Captures the Latin name and the katakana run that opens the
# parenthetical (the romaji after the comma is not captured).
_KANA_PAIR_RE = re.compile(
    r"([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3})\s*"
    r"[（(]\s*([゠-ヿ＝\s]{2,}?)\s*[,、，)）]")


def _mine_kana_pairs(text: str) -> dict:
    """``{katakana: official-English-name}`` pairs from wiki plaintext.

    This is the series knowledge the English-side respeller can never have:
    on the JAPANESE side a mangled name is an exact dictionary hit (カトル →
    Quatre), while on the English side the same repair is a 0.40-similarity
    gamble that maps one character's name onto another's. Component pairs are
    mined too (ヒイロ・ユイ = Heero Yuy → ヒイロ→Heero, ユイ→Yuy) because
    dialogue uses given names alone far more often than full names.
    First mention wins; a kana form claimed by two different names is dropped
    as ambiguous rather than guessed."""
    pairs: dict = {}
    contested: set = set()

    def _claim(kana: str, en: str) -> None:
        prev = pairs.get(kana)
        if prev is not None and prev != en:
            contested.add(kana)
            return
        pairs[kana] = en

    for m in _KANA_PAIR_RE.finditer(text or ""):
        en = " ".join(m.group(1).split())
        # A sentence-initial "The Aries (エアリーズ)" captures the article —
        # the mapping target is the name, not the noun phrase around it.
        en = re.sub(r"^(?:The|A|An)\s+", "", en)
        kana = m.group(2).strip("・＝ 　")
        if len(kana) < 2 or len(en) < 3:
            continue
        _claim(kana, en)
        kparts = [p for p in re.split(r"[・＝\s]+", kana) if p]
        eparts = en.split()
        if len(kparts) == len(eparts) and len(kparts) > 1:
            for kp, ep in zip(kparts, eparts):
                if len(kp) >= 2 and len(ep) >= 3:
                    _claim(kp, ep)
    return {k: v for k, v in pairs.items() if k not in contested}


# Katakana → Hepburn-ish romaji. Digraphs (two-codepoint sequences) first so
# キャ reads "kya" and not "kiya"; the small-vowel forms cover loanword
# spellings (ティ, ファ, ヴァ) common in anime names.
_KANA_DIGRAPHS = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho", "シェ": "she",
    "チャ": "cha", "チュ": "chu", "チョ": "cho", "チェ": "che",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo", "ジェ": "je",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
    "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo",
    "ヴァ": "va", "ヴィ": "vi", "ヴェ": "ve", "ヴォ": "vo",
    "ティ": "ti", "トゥ": "tu", "ディ": "di", "ドゥ": "du",
    "デュ": "dyu", "テュ": "tyu", "フュ": "fyu", "ヴュ": "vyu",
    "ウィ": "wi", "ウェ": "we", "ウォ": "wo",
    "ツァ": "tsa", "ツィ": "tsi", "ツェ": "tse", "ツォ": "tso",
    "イェ": "ye",
}
_KANA_SINGLE = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o", "ン": "n", "ヴ": "vu",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
}


def _kana_reading_romaji(kana: str) -> str:
    """Katakana → lowercase romaji, tuned for ALIAS GENERATION.

    Distinct from :func:`_kana_to_romaji` above by design, not by accident:
    that one passes non-kana through (its consumers feed it mixed-script
    terms and compare against Latin values — a converter that drops Latin
    would gut the plausibility gate; a shadowing redefinition did exactly
    that once and "Justlove"→"Justice" sailed through). This one is the
    opposite trade: kana-only input, long-vowel marks dropped (Whisper's
    English output writes "Kato", not "Katoo"), ッ doubles the next
    consonant, and anything that is not kana is skipped."""
    out: list = []
    geminate = False
    i = 0
    s = kana or ""
    while i < len(s):
        pair = s[i:i + 2]
        if pair in _KANA_DIGRAPHS:
            syl = _KANA_DIGRAPHS[pair]
            i += 2
        else:
            ch = s[i]
            i += 1
            if ch == "ッ":
                geminate = True
                continue
            if ch == "ー":
                continue  # long vowel — macron dropped
            syl = _KANA_SINGLE.get(ch, "")
            if not syl:
                continue
        if geminate and syl:
            out.append(syl[0])
            geminate = False
        out.append(syl)
    return "".join(out)


def _alias_map_from_pairs(pairs: dict) -> dict:
    """``{lowercase-romaji-variant: official}`` from kana pairs.

    The variants cover how Whisper's translate head actually romanizes:
    the full Hepburn form, the final-``u`` clip ("katoru" → "kator"), and
    the final-syllable clip ("katoru" → "kato"). Variants that are ordinary
    English words, that collide across two different officials, or that ARE
    an official spelling already are dropped — an alias exists to catch
    garble, never to re-answer identity."""
    aliases: dict = {}
    contested: set = set()
    officials = {str(v).strip().lower() for v in (pairs or {}).values()}
    for kana, en in (pairs or {}).items():
        r = _kana_reading_romaji(kana)
        if len(r) < 3 or not r.isalpha():
            continue
        variants = {r}
        if r.endswith("u") and len(r) >= 4:
            variants.add(r[:-1])
        if r[-2:] in ("ru", "su", "tu") and len(r) >= 5:
            variants.add(r[:-2])
        for a in variants:
            if (len(a) < 3 or a in _ROSTER_COMMON_WORDS
                    or a in _ROSTER_SAFE_WORDS or a in officials):
                continue
            prev = aliases.get(a)
            if prev is not None and prev != en:
                contested.add(a)
                continue
            aliases[a] = en
    return {a: v for a, v in aliases.items() if a not in contested}


def _mine_glossary_names(text: str, cap: int = 80) -> list[str]:
    """TitleCase runs that read as proper names, most-frequent first.

    Mid-sentence occurrences only (sentence-initial capitalization proves
    nothing), ordinary/wiki words excluded, and a name must recur — an
    article mentions its cast constantly, so a once-only TitleCase run is
    prose, not a name."""
    counts: dict[str, int] = {}
    for m in _GLOSSARY_NAME_RE.finditer(text or ""):
        s = m.group(1).strip()
        words = [w.lower() for w in s.split()]
        if all(w in _ROSTER_SAFE_WORDS or w in _ROSTER_COMMON_WORDS
               or w in _WIKI_STOP for w in words):
            continue
        # Mid-sentence check: scan back over ALL whitespace to the previous
        # non-space character. A fixed 2-char peek treated a paragraph break
        # (``\n\n`` — ubiquitous in Wikipedia plaintext) and ``.  `` as
        # mid-sentence, counting every paragraph-initial word as a name.
        k = m.start() - 1
        while k >= 0 and text[k].isspace():
            k -= 1
        if k < 0 or text[k] in ".!?:=":
            continue
        counts[s] = counts.get(s, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, c in ranked if c >= 2][:cap]


async def _get_series_glossary(series: str, job_id: str = "") -> list[str]:
    """Official-spelling name list for ``series`` — durable-cached, one
    Wikipedia API round-trip per series ever. Empty on any failure (and
    failures are never cached, so a flaky network can retry next run)."""
    key = (series or "").strip().lower()
    if not key:
        return []
    # Remember which series this process resolved. The text-side respeller runs
    # much later, in a scope that never learns the series name, and without
    # this it can only fall back to the configured hint — unset by default.
    global _LAST_SERIES_KEY
    _LAST_SERIES_KEY = key
    try:
        from backend.config import settings as _gset
        if not bool(getattr(_gset, "TRANSLATION_SERIES_GLOSSARY", True)):
            return []
    except Exception:
        pass
    store = _glossary_store_load()
    _cached = store.get(key) if isinstance(store.get(key), list) else None
    _have_kana = isinstance(store.get(_KANA_STORE_PREFIX + key), dict)
    if _cached and (_have_kana or os.environ.get("PYTEST_CURRENT_TEST")):
        # A names-only entry from before kana mining existed falls through to
        # ONE backfill fetch (outside tests) so the katakana pairs get mined;
        # the cached names are kept either way.
        _remember_series(store, key)
        return list(_cached)
    # Never reach the network from a unit test: an order-dependent pass/fail
    # keyed on whether some earlier test populated the durable store (or on
    # the CI runner's connectivity) is exactly what this module's persist
    # docstring forbids. Cache hits above still work under pytest.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return []
    texts: list[str] = []
    try:
        import httpx
        async with httpx.AsyncClient(
                timeout=8.0,
                headers={"User-Agent": "ClipAI/1.0 (subtitle glossary)"}) as cl:
            for q in (f"List of {series} characters", series):
                r = await cl.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "opensearch", "search": q,
                            "limit": "1", "format": "json"})
                if r.status_code != 200:
                    continue
                titles = (r.json() or [None, []])[1] or []
                if not titles:
                    continue
                r2 = await cl.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "prop": "extracts",
                            "explaintext": "1", "redirects": "1",
                            "format": "json", "titles": titles[0]})
                if r2.status_code != 200:
                    continue
                pages = ((r2.json().get("query") or {}).get("pages") or {})
                for p in pages.values():
                    ex = (p.get("extract") or "")[:80000]
                    if ex:
                        texts.append(ex)
    except Exception as e:
        logger.info("[%s] series glossary fetch failed (%s) — names stay "
                    "model-resolved", job_id or "-", e)
        return list(_cached) if _cached else []
    _joined = "\n".join(texts)
    names = _mine_glossary_names(_joined) if texts else []
    kana = _mine_kana_pairs(_joined) if texts else {}
    if not names and _cached:
        names = list(_cached)   # backfill fetch mined nothing new — keep cache
    if kana:
        store[_KANA_STORE_PREFIX + key] = kana
        logger.info(
            "[%s] series glossary: %d katakana reading(s) for %r cached "
            "(e.g. %s)", job_id or "-", len(kana), series,
            "; ".join(f"{k}→{v}" for k, v in list(kana.items())[:4]))
    if names:
        store[key] = names
        store[_LAST_SERIES_STORE_KEY] = key
        logger.info(
            "[%s] series glossary: %d authoritative name(s) for %r cached "
            "(e.g. %s)", job_id or "-", len(names), series,
            ", ".join(names[:6]))
    if names or kana:
        _glossary_store_save(store)
    return names


def _glossary_tokens(glossary: list[str]) -> list[str]:
    """Individual name tokens plus full multi-word names, deduped."""
    toks: list[str] = []
    seen: set[str] = set()
    for name in glossary or []:
        for tok in name.split():
            tl = tok.lower()
            if len(tok) >= 3 and tl not in seen:
                seen.add(tl)
                toks.append(tok)
        nl = name.lower()
        if " " in name and nl not in seen:
            seen.add(nl)
            toks.append(name)
    return toks


_LAST_SERIES_KEY = ""
# Reserved key in the durable glossary store naming the series this machine
# most recently resolved. Prefixed so it can never collide with a real series.
_LAST_SERIES_STORE_KEY = "__last_series__"
_GLOSSARY_RESPELL_RATIO = 0.75
_GLOSSARY_RESPELL_MARGIN = 0.08
# Loose-match floor for kana-derived romaji aliases. Higher than the
# orthographic ratio would need to be, because an alias is already the
# expected garble shape — "dorian" vs "darian" scores 0.83; anything below
# 0.8 against a romaji reading is a different word, not a mishearing of it.
_KANA_ALIAS_RATIO = 0.8
# Mid-sentence capitalised tokens. A word capitalised only because it opens a
# sentence tells us nothing about whether it is a name, so the lookbehinds
# exclude that position — the same signal ``_mine_name_candidates`` uses.
_TEXT_NAME_TOKEN_RE = re.compile(
    r"(?<![.!?…]\s)(?<!^)\b([A-Z][A-Za-z'’]{2,})\b", re.MULTILINE)


def _spelling_variants(token: str) -> set:
    """``token`` plus its possessive and regular-plural bases.

    A token is only a misspelling if NONE of these is an official name.
    Without this, "Gundams" and "Marina's" both read as near-misses of
    "Gundam"/"Marina" and get "corrected" into the singular — measured on the
    professional reference, where every such hit was a correct plural or
    possessive, not an error."""
    out = {token}
    base = re.sub(r"['’]s$", "", token)
    out.add(base)
    low = base.lower()
    if low.endswith("es"):
        out.add(base[:-2])
    if low.endswith("s"):
        out.add(base[:-1])
    return out


def respell_text_from_glossary(texts: list,
                               glossary: list[str],
                               aliases: dict | None = None) -> tuple[list, int, list]:
    """Snap misspelled names in the SUBTITLE TEXT onto glossary spellings.

    ``_apply_glossary_spellings`` fixes the roster's mapping VALUES, which
    only helps when the roster produced a mapping at all. On a measured run it
    did not — "27 candidate(s) offered, none confidently corrected" — so the
    glossary sat in memory holding the right spellings while the shipped
    subtitles said "Dorlian", "Septum" and "Aires". The authority existed and
    never reached the text. This applies it directly, with no model in the
    loop, so a run whose roster declines still gets the spellings it already
    knows.

    Deliberately ORTHOGRAPHIC, not phonetic. A phonetic match against a cast
    list is not safe: measured on this episode it mapped "Trois" onto "Treize"
    — one character's name onto another's — because their consonant skeletons
    are identical. Letters are the conservative evidence, and at ratio
    ``_GLOSSARY_RESPELL_RATIO`` with a runner-up margin the measured result on
    the run was three corrections, zero false positives, and zero changes when
    the same pass was run over the professional reference track.

    The cost of that conservatism is real and worth stating: garbles that
    differ in spelling but agree in sound ("Jekks", "Xerxes", "Katrou",
    "U-Phi") are NOT reachable here on letters alone. ``aliases``
    (romaji variants of the series' official katakana readings, from
    :func:`kana_aliases_for_job`) supplies exactly that series knowledge:
    "Kato" is an exact hit on カトル's clipped romaji, and "Dorian" sits at
    0.83 against ダーリアン's "darian" — evidence from the sound the garble
    came FROM, not a phonetic guess between two English spellings. Alias
    matches are exact, or ≥ ``_KANA_ALIAS_RATIO`` with the same first letter,
    length within 2, and the margin rule against aliases of OTHER names.
    Returns ``(texts, replacements, samples)``."""
    toks = _glossary_tokens(glossary)
    aliases = {str(k).lower(): str(v) for k, v in (aliases or {}).items()
               if k and v}
    if not texts or not toks:
        return list(texts or []), 0, []
    tok_lower = {t.lower() for t in toks}
    single = [t for t in toks if " " not in t and len(t) >= 4]
    if not single:
        return list(texts or []), 0, []

    resolved: dict[str, str] = {}
    rejected: set = set()

    def _target(word: str):
        if word in resolved:
            return resolved[word]
        if word in rejected:
            return None
        if any(v.lower() in tok_lower for v in _spelling_variants(word)):
            rejected.add(word)          # already an official spelling
            return None
        base = re.sub(r"['’]s$", "", word)
        if len(base) < 4 or not base.isalpha():
            rejected.add(word)
            return None
        bl = base.lower()
        # Kana-derived aliases first — they carry the series' own readings,
        # which is stronger evidence than letter distance to an English
        # spelling. Exact hit, or a close hit vetted against aliases of
        # OTHER names (the Trois/Treize failure mode, restated for aliases).
        if aliases:
            hit = aliases.get(bl)
            if hit is None:
                ranked_a = sorted(
                    ((difflib.SequenceMatcher(None, bl, a).ratio(), a)
                     for a in aliases
                     if a[:1] == bl[:1] and abs(len(a) - len(bl)) <= 2),
                    reverse=True)
                if ranked_a and ranked_a[0][0] >= _KANA_ALIAS_RATIO:
                    _top = aliases[ranked_a[0][1]]
                    _rival = next(
                        (r for r, a in ranked_a[1:] if aliases[a] != _top),
                        0.0)
                    if ranked_a[0][0] - _rival >= _GLOSSARY_RESPELL_MARGIN:
                        hit = _top
            if hit and hit.lower() != bl:
                resolved[word] = hit
                return hit
        ranked = sorted(
            ((difflib.SequenceMatcher(
                None, bl, t.lower()).ratio(), t) for t in single),
            reverse=True)
        best, tok = ranked[0]
        # A near-tie means two official names are both plausible; identity is
        # ambiguous and guessing it would put one character's name on another.
        if best < _GLOSSARY_RESPELL_RATIO or (
                len(ranked) > 1
                and best - ranked[1][0] < _GLOSSARY_RESPELL_MARGIN):
            rejected.add(word)
            return None
        resolved[word] = tok
        return tok

    def _alias_exact(word: str):
        """EXACT alias hit for a token the positional regex refused.

        ``_TEXT_NAME_TOKEN_RE`` skips sentence-initial tokens because a
        capital there proves nothing — but the real data says "Mr. Dorian,
        sir…", and after the abbreviation's period the guard hides the one
        token that needs fixing. An exact dictionary hit does not rest on
        the capitalization at all, so position is irrelevant for it; only
        the LOOSE ratio match keeps the positional guard."""
        base = re.sub(r"['’]s$", "", word)
        if len(base) < 4 or not base.isalpha():
            return None
        if any(v.lower() in tok_lower for v in _spelling_variants(word)):
            return None                 # already an official spelling
        hit = aliases.get(base.lower())
        return hit if hit and hit.lower() != base.lower() else None

    out, n, samples = [], 0, []
    for raw in texts:
        line = str(raw or "")
        for word in set(_TEXT_NAME_TOKEN_RE.findall(line)):
            tok = _target(word)
            if not tok:
                continue
            line, k = re.subn(r"\b" + re.escape(word) + r"\b", tok, line)
            if k:
                n += k
                if len(samples) < 8:
                    samples.append(f"{word}→{tok}")
        if aliases:
            for word in set(re.findall(r"\b[A-Z][A-Za-z'’]{2,}\b", line)):
                tok = _alias_exact(word)
                if not tok:
                    continue
                line, k = re.subn(r"\b" + re.escape(word) + r"\b", tok, line)
                if k:
                    n += k
                    if len(samples) < 8:
                        samples.append(f"{word}→{tok}")
        out.append(line)
    return out, n, samples


def _apply_glossary_spellings(
        mapping: dict[str, str], glossary: list[str]) -> tuple[dict[str, str], int]:
    """Snap mapping VALUES onto authoritative glossary spellings.

    Only a near-variant is touched (similarity ≥ 0.72): the model already
    said essentially the right name with the wrong letters ("Hero"→"Heero",
    "Dorian"→"Darlian", "Aires"→"Aries"). A value that is already a glossary
    spelling, or that resembles nothing in the glossary, is left alone — the
    glossary corrects spelling, it never re-answers identity."""
    toks = _glossary_tokens(glossary)
    if not mapping or not toks:
        return dict(mapping or {}), 0
    out = dict(mapping)
    fixed = 0
    tok_lower = {t.lower() for t in toks}
    for src, val in mapping.items():
        if (val or "").lower() in tok_lower:
            continue
        # A value whose EVERY word is already an official token is not a
        # misspelling — it is a full name the mined list happens to store in
        # pieces, and "snapping" it would truncate ("Lucrezia Noin" →
        # "Lucrezia").
        _vwords = [w.lower() for w in (val or "").split()]
        if _vwords and all(w in tok_lower for w in _vwords):
            continue
        nv = _normalize(val)
        ranked = sorted(
            ((difflib.SequenceMatcher(None, nv, _normalize(tok)).ratio(), tok)
             for tok in toks), reverse=True)
        if not ranked:
            continue
        best_r, best = ranked[0]
        # The winner must clear the respelling bar, clearly beat the
        # runner-up (a near-tie means two official names are both close —
        # identity is ambiguous and must not be guessed), keep phonetic
        # faith with a katakana source, and not collide with a value some
        # OTHER term already holds.
        if best_r < 0.72:
            continue
        if len(ranked) > 1 and best_r - ranked[1][0] < 0.05 \
                and ranked[1][1].lower() != best.lower():
            continue
        if _normalize(_kana_to_romaji(src)) != _normalize(src) \
                and not _canonical_sounds_plausible(src, best):
            continue
        if any(o_src != src and (o_val or "").lower() == best.lower()
               for o_src, o_val in out.items()):
            continue
        out[src] = best
        fixed += 1
        logger.info(
            "canonical names: glossary respelled %r → %r (was %r, "
            "similarity %.2f)", src, best, val, best_r)
    return out, fixed


def _glossary_resolve_terms(
        terms: list[str], glossary: list[str]) -> dict[str, str]:
    """Resolve still-unanswered KATAKANA terms straight from the glossary by
    phonetics — retrieval where the model had no answer at all. Conservative:
    the winner must pass the phonetic gate and beat the runner-up clearly."""
    toks = [t for t in _glossary_tokens(glossary) if " " not in t]
    out: dict[str, str] = {}
    for term in terms or []:
        romaji = _normalize(_kana_to_romaji(term))
        if not romaji or romaji == _normalize(term):
            continue                       # Latin-script term — not phonetics' call
        scored = []
        for tok in toks:
            if not _canonical_sounds_plausible(term, tok):
                continue
            r = difflib.SequenceMatcher(None, romaji, _normalize(tok)).ratio()
            scored.append((r, tok))
        scored.sort(reverse=True)
        if scored and scored[0][0] >= 0.4 and (
                len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.1):
            out[term] = scored[0][1]
    return out


def _publish_series_evidence(job_id: str, mapping: dict) -> None:
    """Republish a resolved canonical map under the per-job key the roster pass
    reads its series evidence from.

    ``resolve_roster_corrections`` looks up ``job:<id>``, but the resolver caches
    under a CONTENT key (``sig:<title>|<terms>``) — a determinism change that
    keyed the cache on the inputs rather than the run, and moved the entry out
    from under the reader without updating it. Nothing wrote ``job:<id>``
    afterwards, so ``series_map`` was always empty; with no operator series hint
    the roster pass then returned before making any LLM call, and has therefore
    never run on a real job. The unit tests missed it because they seed
    ``job:<id>`` by hand.

    Keeping BOTH keys costs one dict copy and lets the content key stay the
    deterministic identity while the job key stays the reader's handle."""
    if not job_id or not mapping:
        return
    try:
        merged = dict(mapping)
        # Glossary names ride along as identity entries so the roster pass's
        # known-names corroboration accepts corrections whose RIGHT side is
        # an official spelling the mined terms never surfaced ("Hero Yu" →
        # "Heero Yuy" needs "Heero Yuy" in the evidence set to survive
        # vetting). Identity entries add evidence, never rewrites.
        for n in (_CACHE.get(f"glossary:{job_id}") or {}).values():
            merged.setdefault(n, n)
        _cache_put(f"job:{job_id}", merged)
    except Exception:
        pass


def roster_corrections_for_job(job_id: str) -> dict[str, str]:
    """The roster mapping resolved for this job (empty when none ran)."""
    return dict(_CACHE.get(f"roster:{job_id}") or {}) if job_id else {}


def series_glossary_for_job(job_id: str = "",
                            series: str = "") -> list[str]:
    """The authoritative name list actually available to THIS job.

    ``series_roster_terms`` reads a cache only ``expand_series_hint_to_names``
    fills, and that function is gated on ``TRANSLATION_SERIES_HINT`` — unset by
    default. So on a run with no hint it returns [] even when the glossary
    demonstrably exists: a measured run logged "glossary respelled 'ヒーロ' →
    'Heero'" and identified the series, then handed the text pass an empty
    list. The authority was in memory and the subtitles shipped misspelled.

    Three sources, most specific first: the per-job cache written when this
    job's own terms were resolved; the durable store keyed on the identified
    series; and finally the configured hint. Read-only and side-effect free."""
    try:
        if job_id:
            hit = _CACHE.get(f"glossary:{job_id}")
            if hit:
                return [str(v) for v in hit.values() if v]
        store = _glossary_store_load()
        key = ((series or "").strip().lower() or _LAST_SERIES_KEY
               or str(store.get(_LAST_SERIES_STORE_KEY) or ""))
        if key:
            names = store.get(key)
            if isinstance(names, list) and names:
                return [str(n) for n in names if n]
    except Exception:
        pass
    return series_roster_terms()


def kana_pairs_for_job(job_id: str = "", series: str = "") -> dict:
    """``{katakana: official-English-name}`` for the job's series, or {}.

    Same resolution chain as :func:`series_glossary_for_job` (explicit series
    → this process's last-identified series → the durable store's pointer).
    This is what lets the translator pin カトル → Quatre BEFORE the LLM ever
    guesses a romanization — the point where the name is still recoverable
    as an exact dictionary hit. Read-only and fail-soft."""
    try:
        store = _glossary_store_load()
        key = ((series or "").strip().lower() or _LAST_SERIES_KEY
               or str(store.get(_LAST_SERIES_STORE_KEY) or ""))
        if key:
            pairs = store.get(_KANA_STORE_PREFIX + key)
            if isinstance(pairs, dict) and pairs:
                return {str(k): str(v) for k, v in pairs.items() if k and v}
    except Exception:
        pass
    return {}


def kana_aliases_for_job(job_id: str = "", series: str = "") -> dict:
    """``{lowercase-romaji-variant: official}`` for the job's series, or {}.

    The English-side companion to :func:`kana_pairs_for_job`: Whisper's
    translate head romanizes the kana it heard ("Kato" from カトル), and the
    orthographic respeller can never bridge that to "Quatre" — 0.40
    similarity, below any safe threshold. The alias table makes those forms
    dictionary hits derived from the series' own katakana readings."""
    return _alias_map_from_pairs(kana_pairs_for_job(job_id, series))


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
        # Series evidence normally comes from the canonical-name pass (needs ≥3
        # confirmed names). An explicit operator series hint identifies the work
        # on its own, so it unlocks this pass even when the mined map is thin —
        # exactly the generic-filename case where the map came back empty.
        hint = str(getattr(s, "TRANSLATION_SERIES_HINT", "") or "").strip() if s else ""
        if len(series_map) < 3 and not hint:
            return {}
        candidates = _mine_name_candidates(texts)
        if not candidates:
            return {}
        # Terms already canonical need no second look, and FREQUENT
        # consistent spellings (4+ uses) are the glossary working — a real
        # run "corrected" Operation Meteor and Deathscythe into hallucinated
        # variants because they were offered at all. Only ASK about the rest.
        known = {v.lower() for v in series_map.values()}
        # Word-level view of the canonical values so a multi-word candidate
        # that merely CONTAINS a correct term ("Miss Relena", "Gundam Wing")
        # is recognised as noise, not just exact-phrase matches.
        known_words = {w for v in known for w in v.split()}
        frequent = {t for (t, _ex, c) in candidates if c >= 4}
        ask = [(t, ex) for (t, ex, _c) in candidates
               if t.lower() not in known and t not in frequent
               and _roster_worth_asking(t, known_words)]
        if not ask:
            return {}
        logger.info(
            "[%s] roster: %d candidate(s) mined, %d worth asking after "
            "noise/known/frequency filtering",
            job_id or "-", len(candidates), len(ask))
        evidence = "; ".join(f"{k} = {v}" for k, v in list(series_map.items())[:20])
        # With a thin/empty mined map, lead with the operator's series hint so
        # the model still knows which work's roster to correct against.
        evidence_line = (
            f"This episode is from: {hint}. "
            + (f"Canonical terms already verified: {evidence}. " if evidence else "")
            if hint else
            f"Canonical terms already verified for this episode: {evidence}. "
        )
        cand_block = "\n".join(f'- "{t}"  (e.g. “{ex}”)' for t, ex in ask)
        prompt = (
            "You are repairing machine subtitles for one specific episode. "
            "Speech recognition mis-heard some Japanese proper nouns and the "
            "translator spelled them phonetically.\n"
            f"{evidence_line}"
            "These identify the series precisely.\n\n"
            "For each candidate token below, decide whether it is a GARBLED "
            "rendering of a character, mecha, faction, place or term from "
            "this series. A correction is ONLY valid when the token is an "
            "obvious mis-hearing that SOUNDS like the official name "
            "('Gundarium' → 'Gundanium'). Most candidates are already correct "
            "or ordinary words — return the FEW that are clearly garbled "
            "character, mecha, faction or place names, and leave the rest. "
            "NEVER replace a name with a DIFFERENT character or term that "
            "merely fits the scene, and NEVER rewrite a term that is already "
            "a correct official spelling — omit those. Prefer consolidating "
            "a variant spelling toward the spelling that already appears in "
            "other candidates' example lines — never invent a new spelling. "
            "Return ONLY the "
            "corrections you are "
            "confident about, as a JSON array of objects "
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
        # Ground truth that can vouch for an UNATTESTED correction: the series
        # roster / canonical map resolved for this job, plus the operator's own
        # vocabulary. Without a corroborating source the vetting can only
        # consolidate spellings the transcript already uses (see _vet_roster_pairs).
        _known: set = set(series_map.values()) if series_map else set()
        try:
            from backend.services.custom_vocabulary import load_vocabulary
            _known |= {str(t) for t in (load_vocabulary() or [])}
        except Exception:
            pass
        try:
            _known |= {str(t) for t in (series_roster_terms() or [])}
        except Exception:
            pass
        mapping = _vet_roster_pairs(
            _parse_json_pairs(raw or ""), {t for t, _ in ask},
            corpus=" ".join(str(t) for t in texts),
            known_names=_known)
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
