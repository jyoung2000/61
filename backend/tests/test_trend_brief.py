"""Tests for the live trend brief: 3-tier fail-soft fetch (web-search → Google
Trends → static), daily cache, and injection into the SEO prompt.

The live fetches themselves need a key / pytrends (not available here), so the
tier functions are mocked — these cover the orchestration, caching, and
fail-soft that matter regardless of the source.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

import backend.config as c
import backend.services.trend_brief as TB


@pytest.fixture(autouse=True)
def _cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(TB, "_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_ENABLED", True, raising=False)
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_CACHE_HOURS", 24, raising=False)
    return tmp_path


def _patch_tiers(monkeypatch, web=None, google=None):
    async def _web(p, g):
        return web
    async def _goog(p, g):
        return google
    monkeypatch.setattr(TB, "_fetch_web_search", _web)
    monkeypatch.setattr(TB, "_fetch_google_trends", _goog)


# ── tier order / fail-soft ────────────────────────────────────────────────

def test_web_search_is_primary(monkeypatch):
    _patch_tiers(monkeypatch, web="WEB BRIEF #x", google="GOOGLE BRIEF")
    assert asyncio.run(TB.get_trend_brief("both", "anime")) == "WEB BRIEF #x"


def test_falls_back_to_google_when_web_empty(monkeypatch):
    _patch_tiers(monkeypatch, web=None, google="GOOGLE BRIEF")
    assert asyncio.run(TB.get_trend_brief("both", "anime")) == "GOOGLE BRIEF"


def test_falls_back_to_static_when_all_live_fail(monkeypatch):
    _patch_tiers(monkeypatch, web=None, google=None)
    out = asyncio.run(TB.get_trend_brief("tiktok", "anime"))
    assert "EVERGREEN" in out          # the static last-resort brief


def test_web_tier_raising_does_not_crash(monkeypatch):
    async def _boom(p, g):
        raise RuntimeError("network down")
    async def _goog(p, g):
        return "GOOGLE BRIEF"
    monkeypatch.setattr(TB, "_fetch_web_search", _boom)
    monkeypatch.setattr(TB, "_fetch_google_trends", _goog)
    assert asyncio.run(TB.get_trend_brief()) == "GOOGLE BRIEF"


# ── daily cache ────────────────────────────────────────────────────────────

def test_second_call_same_day_is_cached(monkeypatch):
    calls = {"web": 0}
    async def _web(p, g):
        calls["web"] += 1
        return "WEB BRIEF"
    monkeypatch.setattr(TB, "_fetch_web_search", _web)
    monkeypatch.setattr(TB, "_fetch_google_trends", lambda p, g: None)

    a = asyncio.run(TB.get_trend_brief("both", "anime"))
    b = asyncio.run(TB.get_trend_brief("both", "anime"))
    assert a == b == "WEB BRIEF"
    assert calls["web"] == 1            # fetched once, then served from cache


def test_cache_from_a_previous_day_is_ignored(_cache_env, monkeypatch):
    # Write a brief stamped yesterday → must be treated as stale.
    path = TB._cache_path("both", "anime")
    y = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    with open(path, "w") as f:
        json.dump({"date": y, "ts": time.time(), "brief": "OLD"}, f)
    assert TB._read_cache("both", "anime", 24) is None


def test_read_cached_brief_sync_after_warm(monkeypatch):
    _patch_tiers(monkeypatch, web="WARM BRIEF")
    asyncio.run(TB.get_trend_brief("both", "anime"))
    assert TB.read_cached_brief("both", "anime") == "WARM BRIEF"
    # A platform/genre that was never warmed has no cached brief.
    assert TB.read_cached_brief("youtube", "cooking") == ""


def test_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(c.settings, "LIVE_TRENDS_ENABLED", False)
    _patch_tiers(monkeypatch, web="WEB")
    assert asyncio.run(TB.get_trend_brief()) == ""


# ── SEO prompt injection ──────────────────────────────────────────────────

def test_seo_prompt_injects_brief():
    from backend.services.prompts import build_platform_seo_prompt
    with_brief = build_platform_seo_prompt("tiktok", trend_brief="#anitok #gundam")
    assert "LIVE TREND BRIEF" in with_brief and "#anitok" in with_brief
    without = build_platform_seo_prompt("tiktok")
    assert "LIVE TREND BRIEF" not in without
