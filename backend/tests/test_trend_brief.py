"""Tests for the STRUCTURED live trend brief: 3-tier fail-soft fetch
(web-research JSON → YouTube Data API → static evergreen), daily cache keyed
platforms × genre × region, rendering, and injection into the SEO prompt.

The live fetches need keys (not available here), so the tier functions are
mocked — these cover the parsing, validation, orchestration, caching, and
fail-soft that matter regardless of the source.
"""

import asyncio
import json
import sys
import time
import types
from datetime import datetime, timedelta, timezone

import pytest

import backend.config as c
import backend.services.trend_brief as TB
from backend.services.trend_brief import PlatformTrends, TrendBrief


@pytest.fixture(autouse=True)
def _cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(TB, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_ENABLED", True, raising=False)
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_CACHE_HOURS", 24, raising=False)
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_REGION", "US", raising=False)
    return tmp_path


def _brief(source="sonar", **platform_kwargs):
    plats = {"tiktok": PlatformTrends(**(platform_kwargs or {
        "hashtags": ["#anitok"], "keywords": ["gundam explained"],
        "hook_formats": ["POV: ..."], "topics": ["mecha"]}))}
    return TrendBrief(as_of=TB._today(), region="US", genre="anime",
                      source=source, platforms=plats)


def _patch_tiers(monkeypatch, web=None, youtube=None):
    async def _web(p, g):
        return web
    async def _yt(p, g):
        return youtube
    monkeypatch.setattr(TB, "_fetch_web_search", _web)
    monkeypatch.setattr(TB, "_fetch_youtube_trends", _yt)


# ── tier order / fail-soft ────────────────────────────────────────────────

def test_web_research_is_primary(monkeypatch):
    _patch_tiers(monkeypatch, web=_brief("sonar"), youtube=_brief("youtube_api"))
    out = asyncio.run(TB.get_trend_brief_struct("both", "anime"))
    assert out.source == "sonar"
    assert "#anitok" in out.platforms["tiktok"].hashtags


def test_falls_back_to_youtube_when_web_empty(monkeypatch):
    _patch_tiers(monkeypatch, web=None, youtube=_brief("youtube_api"))
    out = asyncio.run(TB.get_trend_brief_struct("both", "anime"))
    assert out.source == "youtube_api"


def test_falls_back_to_static_when_all_live_fail(monkeypatch):
    _patch_tiers(monkeypatch, web=None, youtube=None)
    out = asyncio.run(TB.get_trend_brief_struct("tiktok", "anime"))
    assert out is not None and out.source == "static"
    text = asyncio.run(TB.get_trend_brief("tiktok", "anime"))
    assert "EVERGREEN" in text  # clearly labeled, never masquerades as live


def test_static_tier_never_recommends_banned_generic_tags(monkeypatch):
    _patch_tiers(monkeypatch, web=None, youtube=None)
    out = asyncio.run(TB.get_trend_brief_struct("both", ""))
    for section in out.platforms.values():
        for tag in section.hashtags:
            assert tag.lstrip("#").lower() not in (
                "fyp", "foryou", "viral", "shorts", "explorepage")
    text = asyncio.run(TB.get_trend_brief("both", ""))
    assert "#fyp" not in text.lower()


def test_web_tier_raising_does_not_crash(monkeypatch):
    async def _boom(p, g):
        raise RuntimeError("network down")
    async def _yt(p, g):
        return _brief("youtube_api")
    monkeypatch.setattr(TB, "_fetch_web_search", _boom)
    monkeypatch.setattr(TB, "_fetch_youtube_trends", _yt)
    assert asyncio.run(TB.get_trend_brief_struct()).source == "youtube_api"


def test_static_tier_survives_missing_evergreen_file(monkeypatch):
    monkeypatch.setattr(TB, "_EVERGREEN_FILE", "/nonexistent/evergreen.json")
    out = TB._static_brief("both", "")
    assert out.source == "static" and out.platforms  # built-in minimal brief


# ── tier-1 JSON parsing (strict / fenced / dirty / invalid) ───────────────

_VALID_JSON = json.dumps({"platforms": {
    "tiktok": {"hashtags": ["#anitok", "#gunpla"],
               "keywords": ["gundam explained"],
               "hook_formats": ["POV: ..."], "topics": ["mecha"],
               "sounds": ["some song"]},
    "linkedin": {"hashtags": ["#ProductDesign"], "keywords": [],
                 "hook_formats": [], "topics": [], "sounds": []},
}})


def test_parse_valid_sonar_json():
    out = TB._parse_sonar_brief(_VALID_JSON, "anime")
    assert out is not None and out.source == "sonar"
    assert out.platforms["tiktok"].hashtags == ["#anitok", "#gunpla"]
    assert out.platforms["linkedin"].hashtags == ["#ProductDesign"]


def test_parse_fenced_and_dirty_json():
    dirty = "Here you go!\n```json\n" + _VALID_JSON + "\n```\nHope this helps."
    out = TB._parse_sonar_brief(dirty, "anime")
    assert out is not None and "#anitok" in out.platforms["tiktok"].hashtags


def test_parse_top_level_platform_map():
    raw = json.dumps({"tiktok": {"hashtags": ["#x"], "keywords": ["y"],
                                 "hook_formats": [], "topics": [], "sounds": []}})
    out = TB._parse_sonar_brief(raw, "")
    assert out is not None and out.platforms["tiktok"].hashtags == ["#x"]


def test_parse_invalid_json_returns_none():
    assert TB._parse_sonar_brief("not json at all", "") is None
    assert TB._parse_sonar_brief("", "") is None
    assert TB._parse_sonar_brief('{"platforms": {}}', "") is None  # empty


def test_parse_scrubs_banned_tags_except_live_allowed():
    # A live brief may keep a normally-banned tag ONLY because live data says
    # it's trending — the validator gives live sections their own allow set.
    raw = json.dumps({"platforms": {"tiktok": {
        "hashtags": ["#fyp", "#anitok", "#ANITOK"], "keywords": [],
        "hook_formats": [], "topics": ["t"], "sounds": []}}})
    out = TB._parse_sonar_brief(raw, "")
    tags = out.platforms["tiktok"].hashtags
    assert "#anitok" in tags and len([t for t in tags if t.lower() == "#anitok"]) == 1


# ── daily cache ────────────────────────────────────────────────────────────

def test_second_call_same_day_is_cached(monkeypatch):
    calls = {"web": 0}
    async def _web(p, g):
        calls["web"] += 1
        return _brief("sonar")
    monkeypatch.setattr(TB, "_fetch_web_search", _web)
    _patch_tiers_yt_none(monkeypatch)

    a = asyncio.run(TB.get_trend_brief("both", "anime"))
    b = asyncio.run(TB.get_trend_brief("both", "anime"))
    assert a == b and "#anitok" in a
    assert calls["web"] == 1            # fetched once, then served from cache


def _patch_tiers_yt_none(monkeypatch):
    async def _yt(p, g):
        return None
    monkeypatch.setattr(TB, "_fetch_youtube_trends", _yt)


def test_genre_gets_its_own_cache_entry(monkeypatch):
    calls = {"web": 0}
    async def _web(p, g):
        calls["web"] += 1
        return _brief("sonar")
    monkeypatch.setattr(TB, "_fetch_web_search", _web)
    _patch_tiers_yt_none(monkeypatch)
    asyncio.run(TB.get_trend_brief("both", ""))
    asyncio.run(TB.get_trend_brief("both", "anime"))
    assert calls["web"] == 2            # generic + genre-specific fetched


def test_cache_from_a_previous_day_is_ignored(_cache_env, monkeypatch):
    path = TB._cache_path("both", "anime")
    y = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    with open(path, "w") as f:
        json.dump({"date": y, "ts": time.time(),
                   "brief_struct": _brief().model_dump()}, f)
    assert TB._read_cache("both", "anime", 24) is None


def test_read_cached_brief_sync_after_warm(monkeypatch):
    _patch_tiers(monkeypatch, web=_brief("sonar"))
    asyncio.run(TB.get_trend_brief("both", "anime"))
    txt = TB.read_cached_brief("both", "anime")
    assert "#anitok" in txt
    struct = TB.read_cached_brief_struct("both", "anime")
    assert struct is not None and struct.source == "sonar"
    # A platform/genre that was never warmed has no cached brief.
    assert TB.read_cached_brief("youtube", "cooking") == ""
    assert TB.read_cached_brief_struct("youtube", "cooking") is None


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_ENABLED", False)
    _patch_tiers(monkeypatch, web=_brief("sonar"))
    assert asyncio.run(TB.get_trend_brief()) == ""
    assert asyncio.run(TB.get_trend_brief_struct()) is None
    assert TB.read_cached_brief() == ""


# ── rendering: only the requesting platform's section is injected ─────────

def test_render_platform_section_isolates_platform():
    brief = TrendBrief(as_of="2026-07-10", region="US", source="sonar",
                       platforms={
                           "tiktok": PlatformTrends(hashtags=["#tiktoktag"]),
                           "linkedin": PlatformTrends(hashtags=["#LinkedInTag"]),
                       })
    li = TB.render_platform_section(brief, "linkedin")
    assert "#LinkedInTag" in li and "#tiktoktag" not in li
    tt = TB.render_platform_section(brief, "tiktok")
    assert "#tiktoktag" in tt and "#LinkedInTag" not in tt
    assert TB.render_platform_section(brief, "facebook") == ""  # no section
    assert TB.render_platform_section(None, "tiktok") == ""


def test_render_both_covers_shortform_only():
    brief = TrendBrief(as_of="2026-07-10", region="US", source="sonar",
                       platforms={
                           "tiktok": PlatformTrends(topics=["a"]),
                           "youtube_shorts": PlatformTrends(topics=["b"]),
                           "linkedin": PlatformTrends(topics=["c"]),
                       })
    txt = TB.render_brief_text(brief, "both")
    assert "TIKTOK" in txt and "YOUTUBE SHORTS" in txt and "LINKEDIN" not in txt


# ── region plumbing ────────────────────────────────────────────────────────

def test_region_maps_legacy_pytrends_names(monkeypatch):
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_REGION", "united_states")
    assert TB._region() == "US"
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_REGION", "gb")
    assert TB._region() == "GB"
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_REGION", "not-a-region")
    assert TB._region() == "US"


def test_region_and_genre_reach_the_research_query(monkeypatch):
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_REGION", "DE")
    q = TB._trend_query("cooking")
    assert "DE" in q and "cooking" in q


# ── tier 2: YouTube trending miner (pure) ─────────────────────────────────

def test_youtube_miner_extracts_keywords_and_formats():
    items = [{"snippet": {"title": f"How to build a gundam model kit part {i}?",
                          "tags": ["gundam", "model kit"]}} for i in range(5)]
    mined = TB._mine_youtube_items(items)
    assert "gundam model" in mined.keywords or "model kit" in mined.keywords
    assert any("How to" in f for f in mined.hook_formats)
    assert any("Question" in f for f in mined.hook_formats)
    # stopwords never become keywords
    assert not any(k in ("how", "the", "to") for k in mined.keywords)


def test_genre_category_mapping():
    assert TB._genre_to_category_id("gaming highlights") == "20"
    assert TB._genre_to_category_id("food review") == "26"
    assert TB._genre_to_category_id("") is None
    assert TB._genre_to_category_id("something niche") is None


# ── sequencing: job-start warm-up is non-blocking and fail-soft ───────────

def test_warm_seo_intelligence_swallows_failures(monkeypatch):
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from backend.services import pipeline as P

    async def _boom(*a, **k):
        raise RuntimeError("trend service down")
    monkeypatch.setattr(TB, "get_trend_brief", _boom)

    async def _run():
        P._warm_seo_intelligence("job-x")
        # Non-blocking: returns immediately; the task settles without raising.
        for _ in range(50):
            if not P._seo_warm_tasks:
                break
            await asyncio.sleep(0.01)
        assert not P._seo_warm_tasks
    asyncio.run(_run())


def test_warm_seo_intelligence_without_event_loop_is_noop():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from backend.services import pipeline as P
    # No running loop → create_task raises internally → swallowed.
    P._warm_seo_intelligence("job-y")


# ── SEO prompt injection ──────────────────────────────────────────────────

def test_seo_prompt_injects_brief():
    from backend.services.prompts import build_platform_seo_prompt
    with_brief = build_platform_seo_prompt("tiktok", trend_brief="#anitok #gundam")
    assert "LIVE TREND BRIEF" in with_brief and "#anitok" in with_brief
    without = build_platform_seo_prompt("tiktok")
    assert "LIVE TREND BRIEF" not in without


def test_seo_prompt_is_keyword_first_with_hook_contract():
    from backend.services.prompts import build_platform_seo_prompt
    p = build_platform_seo_prompt("reels")
    assert "primary_keyword" in p and '"hook"' in p and '"keywords"' in p
    assert "first 50 characters" in p.lower() or "FIRST 50 characters" in p
    assert "#fyp" in p  # named as a banned example
