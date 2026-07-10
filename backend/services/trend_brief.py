"""Live, daily-refreshed, STRUCTURED short-form trend brief.

Social platforms change day-by-day and an LLM's training data is months stale,
so "good for the platform TODAY" requires a LIVE source refreshed daily, then
injected into both clip scoring (find moments that ride a current trend) and
metadata generation (titles / captions / tags / hooks / keywords that match
today's search behavior).

The brief is a validated pydantic ``TrendBrief`` — per-platform hashtags,
search keywords, hook formats, topics and sounds — not freeform prose, so the
prompt builders can inject exactly one platform's section and the hygiene
layer (``seo_hygiene``) can scrub every tag before anything downstream sees it.

Three tiers, fail-soft:
  1. Web-research LLM (OpenRouter, ``LIVE_TRENDS_MODEL``, e.g.
     ``perplexity/sonar``) returning STRICT JSON for ALL platforms in one
     call/day — platform-native, genuinely current.
  2. YouTube Data API v3 ``mostPopular`` chart (``YOUTUBE_API_KEY``) — official
     and free; mines trending titles/tags into keywords + hook formats for the
     YouTube sections (other platforms keep the static tier's evergreen
     guidance).
  3. Static structured fallback shipped in-repo
     (``backend/data/evergreen_trends.json``) — durable guidance only, no
     dated hashtags, no generic discovery tags.

Cached on disk ~24h, keyed platforms × genre × region, storing the full
structured JSON + source + timestamp — one fetch per day per genre reused
across every clip in every job.

Back-compat: ``get_trend_brief`` / ``read_cached_brief`` keep their signatures
and still return the rendered-TEXT view; the structured object is available
via the ``*_struct`` variants.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Canonical platform keys a brief carries (matches PLATFORM_PROFILES slugs).
PLATFORM_KEYS = (
    "tiktok", "youtube_shorts", "reels", "x", "facebook", "linkedin", "youtube",
)

_EVERGREEN_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "evergreen_trends.json")


# ── schema ──────────────────────────────────────────────────────────────────

class PlatformTrends(BaseModel):
    hashtags: list[str] = []        # validated, '#'-prefixed, banlist-scrubbed
    keywords: list[str] = []        # natural-language search queries users type
    hook_formats: list[str] = []    # e.g. "POV: ...", "nobody talks about ..."
    topics: list[str] = []
    sounds: list[str] = []          # tiktok/reels only


class TrendBrief(BaseModel):
    as_of: str = ""                 # ISO date
    region: str = ""
    genre: str = ""
    source: str = "static"          # "sonar" | "youtube_api" | "static"
    platforms: dict[str, PlatformTrends] = {}


# ── cache location ─────────────────────────────────────────────────────────

def _cache_dir() -> str:
    docker = "/data/trends"
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        os.makedirs(docker, exist_ok=True)
        return docker
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai", "trends")
    os.makedirs(local, exist_ok=True)
    return local


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _region() -> str:
    """Two-letter region for the research query + the YouTube API. Maps the
    legacy pytrends-style names some installs still carry in their env."""
    from backend.config import settings
    raw = (getattr(settings, "LIVE_TRENDS_REGION", "US") or "US").strip()
    legacy = {
        "united_states": "US", "united_kingdom": "GB", "canada": "CA",
        "australia": "AU", "germany": "DE", "france": "FR", "japan": "JP",
        "brazil": "BR", "india": "IN", "mexico": "MX",
    }
    mapped = legacy.get(raw.lower())
    if mapped:
        return mapped
    return raw.upper() if len(raw) == 2 else "US"


def _cache_key(platforms: str, genre: str) -> str:
    p = (platforms or "both").strip().lower().replace("/", "_")
    g = (genre or "general").strip().lower().replace("/", "_")[:40] or "general"
    r = _region().lower()
    return f"{p}__{g}__{r}"


def _cache_path(platforms: str, genre: str) -> str:
    return os.path.join(_cache_dir(), f"trend_brief__{_cache_key(platforms, genre)}.json")


def _read_cache(platforms: str, genre: str, max_age_h: float) -> Optional[TrendBrief]:
    path = _cache_path(platforms, genre)
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("date") != _today():
            return None
        if (time.time() - float(data.get("ts", 0))) > max_age_h * 3600:
            return None
        struct = data.get("brief_struct")
        if not struct:
            return None
        return TrendBrief.model_validate(struct)
    except Exception:
        return None


def _write_cache(platforms: str, genre: str, brief: TrendBrief) -> None:
    try:
        with open(_cache_path(platforms, genre), "w") as f:
            json.dump({
                "date": _today(),
                "ts": time.time(),
                "source": brief.source,
                # Rendered text alongside the struct so the cache file stays
                # human-inspectable (and older readers see SOMETHING).
                "brief": render_brief_text(brief, platforms),
                "brief_struct": brief.model_dump(),
            }, f)
    except Exception as e:
        logger.debug("trend brief cache write failed: %s", e)


# ── rendering (struct → prompt-injectable text) ────────────────────────────

_PLATFORM_ALIASES = {
    "shorts": "youtube_shorts",
    "instagram": "reels",       # feed guidance rides the Reels trend section
    "ig": "reels",
    "twitter": "x",
}


def _requested_platform_keys(platforms: str) -> list[str]:
    p = (platforms or "both").strip().lower()
    if p in ("", "both", "default", "all"):
        return ["tiktok", "youtube_shorts"]
    p = _PLATFORM_ALIASES.get(p, p)
    return [p] if p in PLATFORM_KEYS else ["tiktok", "youtube_shorts"]


def _render_section(name: str, section: PlatformTrends) -> str:
    lines = [f"{name.upper().replace('_', ' ')}:"]
    if section.hashtags:
        lines.append("  trending hashtags: " + " ".join(section.hashtags))
    if section.keywords:
        lines.append("  search keywords: " + "; ".join(section.keywords))
    if section.hook_formats:
        lines.append("  hook formats: " + " | ".join(section.hook_formats))
    if section.topics:
        lines.append("  hot topics: " + "; ".join(section.topics))
    if section.sounds:
        lines.append("  trending sounds: " + "; ".join(section.sounds))
    return "\n".join(lines) if len(lines) > 1 else ""


def render_platform_section(brief: Optional[TrendBrief], platform: str) -> str:
    """ONE platform's section of the brief as compact prompt-ready text — what
    ``build_platform_seo_prompt`` injects (never another platform's trends)."""
    if brief is None:
        return ""
    key = _requested_platform_keys(platform)[0]
    section = (brief.platforms or {}).get(key)
    if section is None:
        return ""
    body = _render_section(key, section)
    if not body:
        return ""
    head = f"(source: {brief.source}, as of {brief.as_of or _today()})"
    return head + "\n" + body


def render_brief_text(brief: Optional[TrendBrief], platforms: str = "both") -> str:
    """The requested platform section(s) rendered as text — the back-compat
    string view ``get_trend_brief``/``read_cached_brief`` return."""
    if brief is None:
        return ""
    sections = []
    for key in _requested_platform_keys(platforms):
        section = (brief.platforms or {}).get(key)
        if section is None:
            continue
        body = _render_section(key, section)
        if body:
            sections.append(body)
    if not sections:
        return ""
    label = "LIVE" if brief.source != "static" else "EVERGREEN (static, not live)"
    head = (f"{label} TREND DATA — source: {brief.source}, "
            f"as of {brief.as_of or _today()}")
    if brief.genre:
        head += f", genre: {brief.genre}"
    return head + "\n" + "\n".join(sections)


# ── Tier 1: web-research LLM (OpenRouter) ──────────────────────────────────

def _trend_query(genre: str) -> str:
    region = _region()
    gclause = (f" Focus on {genre} content." if genre and genre.lower() not in
               ("", "general") else "")
    plat_shape = ", ".join(f'"{k}": {{...}}' for k in PLATFORM_KEYS)
    return (
        f"You are a social-SEO trends analyst. Today is {_today()}. "
        f"Target region: {region}.{gclause}\n"
        "Using CURRENT web data — TikTok Creative Center trend pages, YouTube "
        "trending, and current social-SEO reporting — return what is trending "
        "THIS WEEK on each platform.\n\n"
        "Return STRICT JSON only. No markdown fences, no commentary, no "
        "trailing text. Exact shape:\n"
        '{"platforms": {' + plat_shape + "}}\n"
        'Each platform object: {"hashtags": [...], "keywords": [...], '
        '"hook_formats": [...], "topics": [...], "sounds": [...]}\n\n'
        "Rules:\n"
        "- hashtags: 8-12 per platform, '#'-prefixed, each verifiably rising "
        "THIS WEEK on that specific platform. NEVER include generic discovery "
        "tags (#fyp, #foryou, #viral, #explorepage) — platforms penalize them.\n"
        "- keywords: 5-10 natural-language search queries users are typing "
        "into that platform's search bar this week.\n"
        "- hook_formats: 4-8 fill-in-the-blank title/hook templates getting "
        "views this week (e.g. 'POV: ...', 'nobody talks about ...').\n"
        "- topics: 5-8 topics/memes/events trending now on that platform.\n"
        "- sounds: 3-5 named trending audios for tiktok and reels ONLY; [] "
        "for every other platform.\n"
        "- Be factual and current; omit anything you cannot support with "
        "current data (an empty list beats a stale guess)."
    )


def _strip_json_fences(text: str) -> str:
    """Best-effort extraction of a JSON object from LLM output — strips
    ``` fences and any prose around the outermost {...}."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    return t


def _parse_sonar_brief(raw: str, genre: str) -> Optional[TrendBrief]:
    """Parse + validate the tier-1 JSON. None on anything unusable."""
    try:
        data = json.loads(_strip_json_fences(raw))
    except Exception as e:
        logger.info("trend brief: tier-1 JSON did not parse (%s)", e)
        return None
    plats_in = data.get("platforms") if isinstance(data, dict) else None
    if not isinstance(plats_in, dict):
        # Some models return the platform map at the top level.
        plats_in = data if isinstance(data, dict) else {}
    platforms: dict[str, PlatformTrends] = {}
    for key in PLATFORM_KEYS:
        raw_section = plats_in.get(key)
        if not isinstance(raw_section, dict):
            continue
        try:
            platforms[key] = PlatformTrends.model_validate(raw_section)
        except Exception:
            continue
    if not any(p.hashtags or p.keywords or p.hook_formats or p.topics
               for p in platforms.values()):
        return None
    brief = TrendBrief(as_of=_today(), region=_region(), genre=genre or "",
                       source="sonar", platforms=platforms)
    from backend.services.seo_hygiene import validate_brief
    return validate_brief(brief)


async def _fetch_web_search(platforms: str, genre: str) -> Optional[TrendBrief]:
    """Tier 1 — OpenRouter web-research model, strict JSON, ALL platforms in
    one call (one call/day/genre — same cost as the old prose brief). Returns
    None on any failure."""
    from backend.config import settings
    key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
    if not key:
        return None
    model = (getattr(settings, "LIVE_TRENDS_MODEL", "") or "perplexity/sonar").strip()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "HTTP-Referer": "http://localhost:1353", "X-Title": "ClipAI"},
                json={"model": model,
                      "messages": [{"role": "user", "content": _trend_query(genre)}],
                      "max_tokens": 2500, "temperature": 0.2},
            )
            resp.raise_for_status()
            txt = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.info("trend brief: web-research tier failed (%s) — trying YouTube API", e)
        return None
    return _parse_sonar_brief(txt, genre)


# ── Tier 2: YouTube Data API v3 (official, free, ToS-clean) ────────────────

# Free-text content_category → YouTube videoCategoryId (best-effort keyword
# match; no match = no category filter, which is always valid).
_YT_CATEGORY_KEYWORDS = (
    (("gam", "stream", "esport"), "20"),
    (("music", "song", "concert"), "10"),
    (("sport", "basketball", "soccer", "football", "racing"), "17"),
    (("comedy", "sketch", "funny"), "23"),
    (("education", "tutorial", "lecture", "explain"), "27"),
    (("howto", "how-to", "diy", "cook", "food", "recipe"), "26"),
    (("news", "politic"), "25"),
    (("tech", "science", "review", "unboxing"), "28"),
    (("anime", "animation", "cartoon", "film", "movie"), "1"),
    (("pet", "animal"), "15"),
    (("travel", "vlog"), "19"),
)

_TITLE_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in is it its me "
    "my not of on or our so that the this to was we what when why with you "
    "your | - vs".split())


def _genre_to_category_id(genre: str) -> Optional[str]:
    g = (genre or "").strip().lower()
    if not g:
        return None
    for keywords, cat_id in _YT_CATEGORY_KEYWORDS:
        if any(k in g for k in keywords):
            return cat_id
    return None


def _mine_youtube_items(items: list) -> PlatformTrends:
    """Mine trending videos into keywords / hashtags / hook formats.
    Pure (unit-testable): takes the API's ``items`` list."""
    titles: list[str] = []
    tag_counts: dict[str, int] = {}
    for item in items or []:
        sn = (item or {}).get("snippet") or {}
        title = str(sn.get("title") or "").strip()
        if title:
            titles.append(title)
        for t in sn.get("tags") or []:
            token = str(t).strip().lower()
            if token:
                tag_counts[token] = tag_counts.get(token, 0) + 1

    # Keywords: frequency-ranked title unigrams+bigrams, stopword-filtered.
    word_counts: dict[str, int] = {}
    bigram_counts: dict[str, int] = {}
    for title in titles:
        words = [w for w in re.findall(r"[\w']+", title.lower())
                 if w not in _TITLE_STOPWORDS and len(w) > 2 and not w.isdigit()]
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        for a, b in zip(words, words[1:]):
            bigram_counts[f"{a} {b}"] = bigram_counts.get(f"{a} {b}", 0) + 1
    keywords = [k for k, n in sorted(bigram_counts.items(), key=lambda kv: -kv[1])
                if n >= 2][:5]
    keywords += [k for k, n in sorted(word_counts.items(), key=lambda kv: -kv[1])
                 if n >= 3 and all(k not in b for b in keywords)][:5]

    # Hashtags: the most recurrent uploader tags (they mirror search terms).
    hashtags = [t for t, n in sorted(tag_counts.items(), key=lambda kv: -kv[1])
                if n >= 2][:12]

    # Hook formats: recurring title patterns (≥3 titles must match).
    patterns = (
        (r"\bhow (to|i)\b", '"How to/How I ..." explainer'),
        (r"\?\s*$", 'Question title ("why/what/is ...?")'),
        (r"\bvs\.?\b", '"X vs Y" comparison'),
        (r"^\d+\s|\b(top|best)\s+\d+\b", 'Numbered list ("Top N ...")'),
        (r"\bi (tried|tested|built|spent|survived)\b",
         '"I tried/tested ..." first-person experiment'),
        (r"\breact(s|ion)?\b", "Reaction format"),
    )
    hook_formats = []
    for pattern, label in patterns:
        hits = sum(1 for t in titles if re.search(pattern, t, re.IGNORECASE))
        if hits >= 3:
            hook_formats.append(label)

    return PlatformTrends(hashtags=hashtags, keywords=keywords,
                          hook_formats=hook_formats)


async def _fetch_youtube_trends(platforms: str, genre: str) -> Optional[TrendBrief]:
    """Tier 2 — YouTube Data API v3 mostPopular chart via httpx (no client
    lib). Populates only the youtube/youtube_shorts sections; every other
    platform keeps the static tier's evergreen guidance. None on any failure
    (including no API key)."""
    from backend.config import settings
    key = (getattr(settings, "YOUTUBE_API_KEY", "") or "").strip()
    if not key:
        return None
    params = {
        "part": "snippet",
        "chart": "mostPopular",
        "regionCode": _region(),
        "maxResults": "25",
        "key": key,
    }
    cat = _genre_to_category_id(genre)
    if cat:
        params["videoCategoryId"] = cat
    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
            resp = await client.get(
                "https://www.googleapis.com/youtube/v3/videos", params=params)
            if resp.status_code == 400 and cat:
                # Not every category supports the mostPopular chart — retry
                # unfiltered rather than losing the tier.
                params.pop("videoCategoryId", None)
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/videos", params=params)
            resp.raise_for_status()
            items = (resp.json() or {}).get("items") or []
    except Exception as e:
        logger.info("trend brief: YouTube API tier failed (%s)", e)
        return None
    if not items:
        return None
    mined = _mine_youtube_items(items)
    if not (mined.hashtags or mined.keywords or mined.hook_formats):
        return None
    # Start from the static brief so non-YouTube platforms keep evergreen
    # guidance, then overwrite the YouTube sections with the mined live data.
    brief = _static_brief(platforms, genre)
    brief.source = "youtube_api"
    brief.platforms["youtube"] = mined
    brief.platforms["youtube_shorts"] = mined.model_copy(deep=True)
    from backend.services.seo_hygiene import validate_brief
    return validate_brief(brief)


# ── Tier 3: static structured fallback ─────────────────────────────────────

def _static_brief(platforms: str, genre: str) -> TrendBrief:
    """Always-available structured fallback from the shipped evergreen file.
    Durable guidance only — no dated hashtags, no generic discovery tags
    (those are banned; see seo_hygiene). Never raises."""
    plat_map: dict[str, PlatformTrends] = {}
    try:
        with open(_EVERGREEN_FILE) as f:
            data = json.load(f)
        for key, section in (data.get("platforms") or {}).items():
            if key in PLATFORM_KEYS and isinstance(section, dict):
                try:
                    plat_map[key] = PlatformTrends.model_validate(section)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("evergreen_trends.json unreadable (%s) — minimal static "
                       "brief", e)
    if not plat_map:
        generic = PlatformTrends(hook_formats=[
            "POV: ...", "nobody talks about ...",
            "the ... mistake everyone makes",
        ])
        plat_map = {k: generic.model_copy(deep=True) for k in PLATFORM_KEYS}
    brief = TrendBrief(as_of=_today(), region=_region(), genre=genre or "",
                       source="static", platforms=plat_map)
    from backend.services.seo_hygiene import validate_brief
    try:
        return validate_brief(brief)
    except Exception:
        return brief


# ── public entry ───────────────────────────────────────────────────────────

async def get_trend_brief_struct(platforms: str = "both",
                                 genre: str = "") -> Optional[TrendBrief]:
    """Today's structured trend brief (cached 24h). Walks web-research →
    YouTube API → static, never raising. ``None`` only when live trends are
    disabled."""
    from backend.config import settings
    if not bool(getattr(settings, "LIVE_TRENDS_ENABLED", True)):
        return None
    max_age = float(getattr(settings, "LIVE_TRENDS_CACHE_HOURS", 24) or 24)

    cached = _read_cache(platforms, genre, max_age)
    if cached is not None:
        return cached

    brief: Optional[TrendBrief] = None
    for tier, fn in (("web_research", _fetch_web_search),
                     ("youtube_api", _fetch_youtube_trends)):
        try:
            brief = await fn(platforms, genre)
        except Exception as e:
            logger.info("trend brief: %s tier raised (%s)", tier, e)
            brief = None
        if brief is not None:
            break
    if brief is None:
        brief = _static_brief(platforms, genre)
        # WARNING, not debug: the user's "live" SEO is running on evergreen
        # fallback data and they should know why (source labeling surfaces
        # this in the sidecar + status messages too).
        logger.warning(
            "trend brief: all live tiers unavailable — using the STATIC "
            "fallback (set OPENROUTER_API_KEY for the web-research tier or "
            "YOUTUBE_API_KEY for the YouTube tier)")

    logger.info("trend brief refreshed (source=%s, platforms=%s, genre=%s, "
                "region=%s)", brief.source, platforms or "both",
                genre or "general", brief.region)
    _write_cache(platforms, genre, brief)
    return brief


async def get_trend_brief(platforms: str = "both", genre: str = "") -> str:
    """Back-compat text view of :func:`get_trend_brief_struct` — the rendered
    section(s) for ``platforms``, ready to inject into a prompt; an empty
    string only if disabled."""
    brief = await get_trend_brief_struct(platforms, genre)
    return render_brief_text(brief, platforms)


def read_cached_brief_struct(platforms: str = "both",
                             genre: str = "") -> Optional[TrendBrief]:
    """Sync read of today's cached structured brief (or ``None``) — for sync
    prompt builders / the sidecar. ``get_trend_brief*`` must have been awaited
    earlier (the pipeline warms it at job start)."""
    from backend.config import settings
    if not bool(getattr(settings, "LIVE_TRENDS_ENABLED", True)):
        return None
    max_age = float(getattr(settings, "LIVE_TRENDS_CACHE_HOURS", 24) or 24)
    return _read_cache(platforms, genre, max_age)


def read_cached_brief(platforms: str = "both", genre: str = "") -> str:
    """Sync read of today's cached brief rendered as text for the requested
    platform(s) (or ``''``) — back-compat signature."""
    return render_brief_text(read_cached_brief_struct(platforms, genre), platforms)
