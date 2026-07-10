"""Tag / keyword hygiene for the SEO layer.

Platforms stopped rewarding generic discovery tags years ago — by late 2025
TikTok ignores #fyp, Instagram hard-caps Reels at 5 hashtags and actively
discourages #reels / #explorepage, and X demotes hashtag-stuffed posts. Yet
LLMs keep emitting that 2023-era advice from their training data. This module
is the single scrubbing point between ANY tag source (LLM output, live trend
brief, static fallback) and anything persisted or shown to the user:

  * ``clean_tags``      — normalize, case-insensitively dedup, and strip
                          banned generic tags (with a per-call ``allow`` escape
                          hatch for tags today's LIVE brief explicitly lists).
  * ``validate_brief``  — scrub every platform section of a structured
                          ``TrendBrief`` and cap list lengths.

The banlist ships in ``backend/data/banned_tags.json`` (so the weekly
platform-rules self-researcher can update it without code changes) with an
optional ``banned_tags.live.json`` overlay in the writable state dir, and a
hardcoded fallback if both files are missing/corrupt. Pure stdlib — safe to
import from anywhere (no pydantic / httpx / model imports).
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BANNED_TAGS_FILE = os.path.join(_DATA_DIR, "banned_tags.json")

# Hard fallback if the JSON file is missing/corrupt — mirrors the shipped file.
_FALLBACK_BANNED = [
    "fyp", "foryou", "foryoupage", "fy", "viral", "viralvideo", "trending",
    "explore", "explorepage", "reels", "reelsinstagram", "instagood", "shorts",
]
_FALLBACK_YOUTUBE_ALLOWED = ["shorts"]

# Platforms where '#Shorts' is a real, platform-blessed feature tag rather
# than a generic discovery tag.
_YOUTUBE_PLATFORMS = {"youtube", "youtube_shorts", "shorts"}

# List-length caps for a validated brief (per platform section).
BRIEF_MAX_HASHTAGS = 12
BRIEF_MAX_KEYWORDS = 10
BRIEF_MAX_HOOK_FORMATS = 8
BRIEF_MAX_TOPICS = 10
BRIEF_MAX_SOUNDS = 5


def writable_state_dir() -> str:
    """Writable dir for SEO-intelligence state (rule overlays, banlist
    overlay). Same convention as the trend-brief cache: the Docker volume
    when present, else a repo-local ``.clipai`` dir."""
    docker = "/data/trends"
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        os.makedirs(docker, exist_ok=True)
        return docker
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai", "trends")
    os.makedirs(local, exist_ok=True)
    return local


_banlist_cache: "tuple[set, set] | None" = None


def _load_banlist() -> "tuple[set, set]":
    """(banned_lower, youtube_allowed_lower) from the shipped JSON plus the
    optional live overlay; hardcoded fallback on any failure."""
    banned = set(_FALLBACK_BANNED)
    yt_allowed = set(_FALLBACK_YOUTUBE_ALLOWED)
    try:
        with open(BANNED_TAGS_FILE) as f:
            data = json.load(f)
        banned = {str(t).lstrip("#").strip().lower()
                  for t in (data.get("banned") or []) if str(t).strip()}
        yt_allowed = {str(t).lstrip("#").strip().lower()
                      for t in (data.get("youtube_allowed") or []) if str(t).strip()}
    except Exception as e:
        logger.warning("banned_tags.json unreadable (%s) — using built-in banlist", e)
    # Live overlay (written by the self-researcher) only ADDS entries — a bad
    # overlay can never un-ban the shipped list.
    try:
        overlay = os.path.join(writable_state_dir(), "banned_tags.live.json")
        if os.path.exists(overlay):
            with open(overlay) as f:
                extra = json.load(f)
            banned |= {str(t).lstrip("#").strip().lower()
                       for t in (extra.get("banned") or []) if str(t).strip()}
    except Exception as e:
        logger.debug("banned_tags.live.json overlay skipped (%s)", e)
    return banned, yt_allowed


def banned_generic_tags() -> set:
    """The current banned generic-tag set (lowercase, no '#'). Cached."""
    global _banlist_cache
    if _banlist_cache is None:
        _banlist_cache = _load_banlist()
    return set(_banlist_cache[0])


def reload_banlist() -> None:
    """Drop the cached banlist so the next read picks up file changes."""
    global _banlist_cache
    _banlist_cache = None


# Convenience constant (spec name). Loaded lazily via function above for
# freshness; this snapshot is for callers that just want a set to look at.
BANNED_GENERIC_TAGS = frozenset(_FALLBACK_BANNED)


def normalize_tag(tag) -> str:
    """One tag → canonical ``#lowercase-free`` form (original casing kept).

    Strips the '#' for processing, removes whitespace / punctuation / emoji
    (keeps unicode letters, digits, underscore), re-prefixes '#'. Returns ''
    for anything empty or over 30 characters — junk is dropped, not repaired.
    """
    s = str(tag or "").strip().lstrip("#").strip()
    if not s:
        return ""
    s = re.sub(r"[^\w]", "", s, flags=re.UNICODE)
    if not s or set(s) == {"_"} or len(s) > 30:
        return ""
    return "#" + s


def clean_tags(tags, platform: str = "", allow=None) -> list:
    """Normalize + dedup + banlist-scrub a tag list for ``platform``.

    * every tag comes out '#'-prefixed, ≤30 chars, no whitespace/punct/emoji
    * dedup is case-INSENSITIVE and keeps the first-seen casing
    * banned generic tags are stripped UNLESS their lowercase form appears in
      ``allow`` — the escape hatch for tags today's LIVE trend brief lists for
      this platform (live data outranks the static banlist)
    * '#Shorts' survives on YouTube platforms (it's a feature tag there, a
      generic discovery tag everywhere else)
    """
    global _banlist_cache
    if _banlist_cache is None:
        _banlist_cache = _load_banlist()
    banned, yt_allowed = _banlist_cache
    allow_l = {str(a).lstrip("#").strip().lower()
               for a in (allow or ()) if str(a).strip()}
    plat = (platform or "").strip().lower()
    is_youtube = plat in _YOUTUBE_PLATFORMS
    out, seen = [], set()
    for raw in tags or []:
        tag = normalize_tag(raw)
        if not tag:
            continue
        core = tag[1:].lower()
        if core in seen:
            continue
        if core in banned and core not in allow_l and not (
                is_youtube and core in yt_allowed):
            continue
        seen.add(core)
        out.append(tag)
    return out


def clean_text_list(items, max_len: int = 120, cap: int = 50) -> list:
    """Free-text list hygiene (keywords / hook formats / topics / sounds):
    strings only, stripped, case-insensitive dedup keeping first casing,
    over-long entries dropped, capped at ``cap``."""
    out, seen = [], set()
    for item in items or []:
        s = str(item or "").strip()
        if not s or len(s) > max_len:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def validate_brief(brief):
    """Scrub every platform section of a structured ``TrendBrief`` in place
    (and return it). Duck-typed so this module never imports the schema.

    Hashtags go through ``clean_tags``; for a LIVE brief each section's own
    hashtag set is the ``allow`` list — if live platform data says a normally
    banned tag is genuinely trending this week, it survives; the STATIC tier
    gets no such pass (its file must never carry banned tags, and this
    enforces it). Free-text lists are deduped and capped.
    """
    if brief is None:
        return None
    live = (getattr(brief, "source", "") or "").strip().lower() != "static"
    for plat, section in (getattr(brief, "platforms", None) or {}).items():
        if section is None:
            continue
        own = set(getattr(section, "hashtags", None) or []) if live else set()
        section.hashtags = clean_tags(
            getattr(section, "hashtags", None), plat, allow=own,
        )[:BRIEF_MAX_HASHTAGS]
        section.keywords = clean_text_list(
            getattr(section, "keywords", None), cap=BRIEF_MAX_KEYWORDS)
        section.hook_formats = clean_text_list(
            getattr(section, "hook_formats", None), cap=BRIEF_MAX_HOOK_FORMATS)
        section.topics = clean_text_list(
            getattr(section, "topics", None), cap=BRIEF_MAX_TOPICS)
        section.sounds = clean_text_list(
            getattr(section, "sounds", None), cap=BRIEF_MAX_SOUNDS)
    return brief
