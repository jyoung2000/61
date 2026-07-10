"""Tests for the SEO tag/keyword hygiene layer (seo_hygiene) and its wiring
into enforce_platform_caps.

Platforms penalize the 2023-era advice LLMs still emit (#fyp, hashtag
stuffing, 15+ Reels tags). These pin the single scrub point between any tag
source and anything persisted: banlist (case-insensitive, with the live-brief
allow escape hatch), normalization (emoji/punctuation/whitespace), dedup
(first casing wins), the per-platform caps, and backward compatibility with
old-shape SEO dicts.
"""

import pytest

from backend.services import seo_hygiene as H
from backend.services.prompts import enforce_platform_caps


# ── clean_tags: banlist ────────────────────────────────────────────────────

def test_banned_tags_stripped_case_insensitively():
    out = H.clean_tags(["#FYP", "#fyp", "#Viral", "#ExplorePage", "#anime"],
                       "tiktok")
    assert out == ["#anime"]


def test_brief_allowed_tag_survives():
    # Today's live brief explicitly lists #fyp for this platform → it survives.
    out = H.clean_tags(["#fyp", "#anime"], "tiktok", allow={"#fyp"})
    assert out == ["#fyp", "#anime"]
    # Allow matching is case-insensitive too.
    out = H.clean_tags(["#FYP"], "tiktok", allow={"#fyp"})
    assert out == ["#FYP"]


def test_shorts_allowed_on_youtube_only():
    assert H.clean_tags(["#Shorts"], "youtube_shorts") == ["#Shorts"]
    assert H.clean_tags(["#Shorts"], "youtube") == ["#Shorts"]
    assert H.clean_tags(["#Shorts"], "tiktok") == []
    assert H.clean_tags(["#shorts"], "reels") == []


# ── clean_tags: normalization + dedup ──────────────────────────────────────

def test_dedup_keeps_first_seen_casing():
    out = H.clean_tags(["#GundamWing", "#gundamwing", "#GUNDAMWING"], "tiktok")
    assert out == ["#GundamWing"]


def test_emoji_punctuation_whitespace_stripped():
    out = H.clean_tags(["#food tok!", "#🍜ramen", "  cooking  ", "#!!!"],
                       "tiktok")
    assert out == ["#foodtok", "#ramen", "#cooking"]


def test_overlong_and_empty_tags_dropped():
    out = H.clean_tags(["#" + "x" * 31, "", "#", None, "#ok"], "tiktok")
    assert out == ["#ok"]


def test_normalize_tag_shapes():
    assert H.normalize_tag("plain") == "#plain"
    assert H.normalize_tag("#Already") == "#Already"
    assert H.normalize_tag("  # spaced out ") == "#spacedout"
    assert H.normalize_tag("💯") == ""


# ── banlist file fallback ─────────────────────────────────────────────────

def test_banlist_falls_back_when_file_missing(monkeypatch):
    monkeypatch.setattr(H, "BANNED_TAGS_FILE", "/nonexistent/banned.json")
    H.reload_banlist()
    try:
        assert "fyp" in H.banned_generic_tags()
        assert H.clean_tags(["#fyp", "#anime"], "tiktok") == ["#anime"]
    finally:
        monkeypatch.undo()
        H.reload_banlist()


# ── validate_brief ────────────────────────────────────────────────────────

class _Section:
    def __init__(self, **kw):
        self.hashtags = kw.get("hashtags", [])
        self.keywords = kw.get("keywords", [])
        self.hook_formats = kw.get("hook_formats", [])
        self.topics = kw.get("topics", [])
        self.sounds = kw.get("sounds", [])


class _Brief:
    def __init__(self, source, platforms):
        self.source = source
        self.platforms = platforms


def test_validate_brief_caps_lengths_and_dedups():
    section = _Section(hashtags=[f"#tag{i}" for i in range(30)],
                       keywords=[f"kw {i}" for i in range(30)],
                       hook_formats=[f"hook {i}" for i in range(30)],
                       topics=["T", "t", "T"])
    brief = H.validate_brief(_Brief("sonar", {"tiktok": section}))
    assert len(brief.platforms["tiktok"].hashtags) <= 12
    assert len(brief.platforms["tiktok"].keywords) <= 10
    assert len(brief.platforms["tiktok"].hook_formats) <= 8
    assert brief.platforms["tiktok"].topics == ["T"]


def test_validate_brief_static_gets_no_banned_tag_pass():
    brief = H.validate_brief(_Brief("static", {
        "tiktok": _Section(hashtags=["#fyp", "#anime"])}))
    assert brief.platforms["tiktok"].hashtags == ["#anime"]


def test_validate_brief_live_keeps_its_own_tags():
    brief = H.validate_brief(_Brief("sonar", {
        "tiktok": _Section(hashtags=["#fyp", "#anime"])}))
    assert "#fyp" in brief.platforms["tiktok"].hashtags


# ── enforce_platform_caps (hygiene wiring + trim rules) ───────────────────

def test_reels_never_exceeds_five_tags():
    seo = {"title": "t", "description": "d",
           "tags": [f"#tag{i}" for i in range(15)], "platform_tips": ""}
    out = enforce_platform_caps(seo, "reels")
    assert len(out["tags"]) <= 5


def test_banned_tags_scrubbed_from_generated_seo():
    seo = {"title": "t", "description": "d",
           "tags": ["#fyp", "#viral", "#gunpla"], "platform_tips": ""}
    out = enforce_platform_caps(seo, "tiktok")
    assert out["tags"] == ["#gunpla"]


def test_trimmed_title_has_no_ellipsis_description_keeps_it():
    long_words = " ".join(["word"] * 100)
    out = enforce_platform_caps(
        {"title": long_words, "description": long_words, "tags": []}, "youtube")
    assert len(out["title"]) <= 70 and not out["title"].endswith("…")
    assert not out["title"].endswith(" ")  # clean word-boundary cut
    assert out["description"] == long_words  # under youtube's 5000 cap
    out2 = enforce_platform_caps({"title": "t", "description": long_words,
                                  "tags": []}, "x")
    assert len(out2["description"]) <= 280 and out2["description"].endswith("…")


def test_below_tag_min_is_not_padded():
    out = enforce_platform_caps(
        {"title": "t", "description": "d", "tags": ["#one"]}, "youtube")
    assert out["tags"] == ["#one"]  # youtube tag_min is 5 — no junk padding


def test_deterministic_trend_mixing_adds_at_most_two_brief_tags():
    seo = {"title": "t", "description": "d", "tags": ["#content"],
           "platform_tips": ""}
    out = enforce_platform_caps(
        seo, "tiktok",
        brief_hashtags=["#trend1", "#trend2", "#trend3", "#content"])
    assert out["tags"][0] == "#content"          # content-specific first
    assert out["tags"][1:] == ["#trend1", "#trend2"]  # then ≤2 brief tags


def test_brief_allowed_banned_tag_survives_caps():
    seo = {"title": "t", "description": "d", "tags": ["#fyp", "#anime"]}
    out = enforce_platform_caps(seo, "tiktok", brief_hashtags=["#fyp"])
    assert "#fyp" in out["tags"]


def test_old_shape_seo_dict_round_trips():
    """A pre-overhaul dict (no hook/keyword fields) still validates into
    ClipSEO and back — old persisted jobs must keep loading."""
    from backend.models import ClipSEO
    old = {"title": "My Title", "description": "Desc",
           "tags": ["#one", "#two"], "platform_tips": "tip"}
    capped = enforce_platform_caps(old, "tiktok")
    rec = ClipSEO(**capped)
    assert rec.title == "My Title"
    assert rec.hook == "" and rec.primary_keyword == "" and rec.keywords == []
    # And a full round-trip of an OLD ClipSEO dump still works.
    dumped = ClipSEO(title="a", description="b").model_dump()
    dumped.pop("hook"), dumped.pop("primary_keyword"), dumped.pop("keywords")
    assert ClipSEO(**dumped).hook == ""


def test_hook_capped_at_60_chars_no_ellipsis():
    out = enforce_platform_caps(
        {"title": "t", "description": "d", "tags": [],
         "hook": "why nobody talks about this incredibly specific gundam detail today"},
        "tiktok")
    assert len(out["hook"]) <= 60 and not out["hook"].endswith("…")


def test_keywords_and_primary_keyword_normalized():
    out = enforce_platform_caps(
        {"title": "t", "description": "d", "tags": [],
         "primary_keyword": "  gundam build guide  ",
         "keywords": ["a", "A", "", "  b  "] + [f"k{i}" for i in range(20)]},
        "tiktok")
    assert out["primary_keyword"] == "gundam build guide"
    assert out["keywords"][:2] == ["a", "b"]
    assert len(out["keywords"]) <= 10
