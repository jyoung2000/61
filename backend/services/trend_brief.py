"""Live, daily-refreshed short-form trend brief.

Social platforms change day-by-day and an LLM's training data is months stale,
so "good for the platform TODAY" requires a LIVE source refreshed daily, then
injected into both clip scoring (find moments that ride a current trend) and
metadata generation (titles / captions / tags / hooks that match today's
hashtags, sounds, and hook formats).

Three tiers, fail-soft (per the user's choice):
  1. Web-search LLM (OpenRouter web-search model, e.g. ``perplexity/sonar`` or a
     ``:online`` variant) — rich, platform-native, genuinely current.
  2. Google Trends (``pytrends``) — free daily trending searches (topics only).
  3. Static lexicon (``trend_matcher``) — always-available last resort.

The brief is cached on disk for ~24h, keyed by date + platforms + genre, so it
is fetched ONCE per day and reused across every clip in every job — fresh
enough for day-by-day trends, cheap on API cost / rate limits.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


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


def _cache_key(platforms: str, genre: str) -> str:
    p = (platforms or "both").strip().lower().replace("/", "_")
    g = (genre or "general").strip().lower().replace("/", "_")[:40] or "general"
    return f"{p}__{g}"


def _cache_path(platforms: str, genre: str) -> str:
    return os.path.join(_cache_dir(), f"trend_brief__{_cache_key(platforms, genre)}.json")


def _read_cache(platforms: str, genre: str, max_age_h: float) -> Optional[str]:
    path = _cache_path(platforms, genre)
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("date") != _today():
            return None
        if (time.time() - float(data.get("ts", 0))) > max_age_h * 3600:
            return None
        brief = data.get("brief") or ""
        return brief or None
    except Exception:
        return None


def _write_cache(platforms: str, genre: str, brief: str, source: str) -> None:
    try:
        with open(_cache_path(platforms, genre), "w") as f:
            json.dump({"date": _today(), "ts": time.time(),
                       "source": source, "brief": brief}, f)
    except Exception as e:
        logger.debug("trend brief cache write failed: %s", e)


# ── Tier 1: web-search LLM (OpenRouter) ────────────────────────────────────

def _trend_query(platforms: str, genre: str) -> str:
    plats = "TikTok and YouTube Shorts"
    p = (platforms or "").lower()
    if "tiktok" in p and "youtube" not in p:
        plats = "TikTok"
    elif "youtube" in p and "tiktok" not in p:
        plats = "YouTube Shorts"
    gclause = f" for {genre} content" if genre and genre not in ("general", "") else ""
    return (
        f"You are a short-form social-media trends analyst. Today is {_today()}. "
        f"Using CURRENT web data, give a concise brief of what is trending RIGHT NOW "
        f"(this week) on {plats}{gclause}. Use these exact sections:\n"
        "TRENDING HASHTAGS: 8-12 currently-rising hashtags (with #).\n"
        "TRENDING SOUNDS/AUDIO: 3-5 popular sounds/audio trends, named.\n"
        "HOOK FORMATS: 4-6 title/hook templates getting views this week "
        "(e.g. 'POV: ...', 'the way ...', 'tell me you ... without ...').\n"
        f"HOT TOPICS: 5-8 topics/memes trending now{gclause}.\n"
        "Be factual and current; under 250 words; no preamble, no caveats."
    )


async def _fetch_web_search(platforms: str, genre: str) -> Optional[str]:
    """Tier 1 — OpenRouter web-search model. Returns None on any failure."""
    from backend.config import settings
    key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
    if not key:
        return None
    model = (getattr(settings, "LIVE_TRENDS_MODEL", "") or "perplexity/sonar").strip()
    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "HTTP-Referer": "http://localhost:1353", "X-Title": "ClipAI"},
                json={"model": model,
                      "messages": [{"role": "user",
                                    "content": _trend_query(platforms, genre)}],
                      "max_tokens": 700, "temperature": 0.3},
            )
            resp.raise_for_status()
            txt = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
            txt = (txt or "").strip()
            return txt or None
    except Exception as e:
        logger.info("trend brief: web-search tier failed (%s) — trying Google Trends", e)
        return None


# ── Tier 2: Google Trends (pytrends, optional dep) ─────────────────────────

async def _fetch_google_trends(platforms: str, genre: str) -> Optional[str]:
    """Tier 2 — free Google Trends daily trending searches (topics only)."""
    from backend.config import settings
    region = (getattr(settings, "LIVE_TRENDS_REGION", "") or "united_states").strip()

    def _run() -> Optional[str]:
        try:
            from pytrends.request import TrendReq  # optional dependency
        except Exception:
            logger.info("trend brief: pytrends not installed — skipping Google Trends tier")
            return None
        try:
            tr = TrendReq(hl="en-US", tz=0)
            df = tr.trending_searches(pn=region)
            topics = [str(x) for x in df[0].tolist()[:15]] if df is not None and not df.empty else []
            if not topics:
                return None
            return ("HOT TOPICS (Google Trends, " + _today() + "): "
                    + ", ".join(topics)
                    + "\n(No hashtag/sound/format data on this tier — topic hints only.)")
        except Exception as e:
            logger.info("trend brief: Google Trends tier failed (%s)", e)
            return None

    import asyncio
    return await asyncio.to_thread(_run)


# ── Tier 3: static lexicon ─────────────────────────────────────────────────

def _static_brief(platforms: str, genre: str) -> str:
    try:
        from backend.services.trend_matcher import format_trend_context
        ctx = format_trend_context(content_type=genre or "", platform=platforms or "")
        if ctx:
            return "EVERGREEN SHORT-FORM PATTERNS (static, not live):\n" + ctx.strip()
    except Exception:
        pass
    return ("EVERGREEN SHORT-FORM PATTERNS (static fallback): strong 3-second hooks, "
            "a clear payoff, emotional or surprising beats, and a clean loop. "
            "Tags: mix one broad discovery tag (#fyp / #shorts) with niche tags "
            "tied to the actual subject.")


# ── public entry ───────────────────────────────────────────────────────────

async def get_trend_brief(platforms: str = "both", genre: str = "") -> str:
    """Today's short-form trend brief for ``platforms`` (+ optional ``genre``).

    Cached 24h. Walks web-search → Google Trends → static, never raising. Returns
    a trend-context string ready to inject into the clip judge + SEO prompts; an
    empty string only if disabled.
    """
    from backend.config import settings
    if not bool(getattr(settings, "LIVE_TRENDS_ENABLED", True)):
        return ""
    max_age = float(getattr(settings, "LIVE_TRENDS_CACHE_HOURS", 24) or 24)

    cached = _read_cache(platforms, genre, max_age)
    if cached is not None:
        return cached

    brief, source = None, "none"
    for tier, fn in (("web_search", _fetch_web_search),
                     ("google_trends", _fetch_google_trends)):
        try:
            brief = await fn(platforms, genre)
        except Exception as e:
            logger.info("trend brief: %s tier raised (%s)", tier, e)
            brief = None
        if brief:
            source = tier
            break
    if not brief:
        brief, source = _static_brief(platforms, genre), "static"

    logger.info("trend brief refreshed (source=%s, platforms=%s, genre=%s, %d chars)",
                source, platforms or "both", genre or "general", len(brief))
    _write_cache(platforms, genre, brief, source)
    return brief


def read_cached_brief(platforms: str = "both", genre: str = "") -> str:
    """Sync read of today's cached brief (or '' if none) — for sync prompt
    builders. ``get_trend_brief`` must have been awaited earlier this job to
    warm the cache."""
    from backend.config import settings
    if not bool(getattr(settings, "LIVE_TRENDS_ENABLED", True)):
        return ""
    max_age = float(getattr(settings, "LIVE_TRENDS_CACHE_HOURS", 24) or 24)
    return _read_cache(platforms, genre, max_age) or ""
