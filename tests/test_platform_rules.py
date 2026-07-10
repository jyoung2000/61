"""Tests for the data-driven platform rules: loader precedence
(overlay > shipped > hardcoded), corrupt-file fallback, sanity-range
rejection, and the self-research refresher's validation/diff plumbing.
"""

import asyncio
import json
import os

import pytest

from backend.services import prompts as P


@pytest.fixture()
def _isolated_rules(tmp_path, monkeypatch):
    """Point the overlay at a temp dir and restore the live dict afterward."""
    overlay = tmp_path / "platform_rules.live.json"
    monkeypatch.setattr(P, "platform_rules_overlay_path",
                        lambda: str(overlay))
    yield overlay
    monkeypatch.undo()
    P.reload_platform_rules()


# ── sanity-range validation ────────────────────────────────────────────────

def test_validate_rule_accepts_sane_values():
    out = P.validate_platform_rule(
        {"tag_min": 3, "tag_max": 5, "title_max": 100,
         "description_max": 4000, "label": "TikTok"})
    assert out == {"tag_min": 3, "tag_max": 5, "title_max": 100,
                   "description_max": 4000, "label": "TikTok"}


def test_validate_rule_rejects_out_of_range_fields():
    out = P.validate_platform_rule(
        {"tag_max": 99, "title_max": 5, "description_max": 10,
         "tag_min": -2})
    assert out == {}


def test_validate_rule_rejects_inverted_tag_range():
    out = P.validate_platform_rule({"tag_min": 8, "tag_max": 3})
    assert "tag_min" not in out and "tag_max" not in out


def test_validate_rule_joins_guidance_lines():
    out = P.validate_platform_rule(
        {"guidance": ["line one", "line two"], "tag_max": 5})
    assert out["guidance"] == "line one\nline two"


def test_validate_rule_non_dict_is_empty():
    assert P.validate_platform_rule("junk") == {}
    assert P.validate_platform_rule(None) == {}


# ── loader precedence ─────────────────────────────────────────────────────

def test_shipped_file_loads_appendix_values(_isolated_rules):
    P.reload_platform_rules()
    assert P.platform_rules_meta()["source"] == "shipped"
    # July-2026 verified values from the shipped platform_rules.json:
    assert P.PLATFORM_PROFILES["reels"]["tag_max"] == 5     # Dec-2025 hard cap
    assert P.PLATFORM_PROFILES["tiktok"]["tag_max"] == 5
    assert P.PLATFORM_PROFILES["tiktok"]["description_max"] == 4000
    assert P.PLATFORM_PROFILES["x"]["tag_max"] == 3
    # "both"/"default" aliases still resolve.
    assert P.PLATFORM_PROFILES["both"]["label"] == "TikTok"


def test_overlay_takes_field_level_precedence(_isolated_rules):
    _isolated_rules.write_text(json.dumps({
        "date": "2026-07-09",
        "platforms": {"tiktok": {"tag_max": 4}},
    }))
    P.reload_platform_rules()
    assert P.PLATFORM_PROFILES["tiktok"]["tag_max"] == 4       # overlay wins
    assert P.PLATFORM_PROFILES["tiktok"]["title_max"] == 150   # shipped kept
    assert "TIKTOK profile" in P.PLATFORM_PROFILES["tiktok"]["guidance"]
    meta = P.platform_rules_meta()
    assert meta["source"] == "live" and meta["refreshed"] == "2026-07-09"


def test_out_of_range_overlay_fields_are_ignored(_isolated_rules):
    _isolated_rules.write_text(json.dumps({
        "platforms": {"tiktok": {"tag_max": 99, "title_max": 3}},
    }))
    P.reload_platform_rules()
    # Nothing valid in the overlay → shipped values stand.
    assert P.PLATFORM_PROFILES["tiktok"]["tag_max"] == 5
    assert P.PLATFORM_PROFILES["tiktok"]["title_max"] == 150
    assert P.platform_rules_meta()["source"] == "shipped"


def test_corrupt_overlay_falls_back(_isolated_rules):
    _isolated_rules.write_text("{not json")
    P.reload_platform_rules()
    assert P.PLATFORM_PROFILES["tiktok"]["tag_max"] == 5
    assert P.platform_rules_meta()["source"] == "shipped"


def test_corrupt_shipped_file_falls_back_to_builtin(_isolated_rules,
                                                    monkeypatch, tmp_path):
    bad = tmp_path / "platform_rules.json"
    bad.write_text("][ definitely not json")
    monkeypatch.setattr(P, "_RULES_SHIPPED_FILE", str(bad))
    P.reload_platform_rules()
    assert P.platform_rules_meta()["source"] == "builtin"
    # The hardcoded fallback still carries complete, sane profiles.
    for slug in ("tiktok", "reels", "youtube", "x", "linkedin"):
        prof = P.PLATFORM_PROFILES[slug]
        assert prof["tag_min"] <= prof["tag_max"] <= 30
        assert prof["guidance"]


# ── researcher plumbing (no network) ──────────────────────────────────────

def test_research_diff_is_human_readable():
    from backend.services.platform_rules_research import _diff_rules
    current = {"tiktok": {"tag_max": 5, "title_max": 150}}
    proposed = {"tiktok": {"tag_max": 4, "title_max": 150, "notes": "x"}}
    diff = _diff_rules(current, proposed)
    assert diff == ["tiktok.tag_max: 5 → 4"]


def test_research_skips_without_api_key(monkeypatch, _isolated_rules):
    from backend.config import settings
    from backend.services import platform_rules_research as R
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    assert asyncio.run(R.maybe_refresh_platform_rules()) is False


def test_research_is_throttled_by_stamp(monkeypatch, tmp_path, _isolated_rules):
    from backend.config import settings
    from backend.services import platform_rules_research as R
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(settings, "PLATFORM_RULES_REFRESH_DAYS", 7)
    monkeypatch.setattr(R, "_stamp_path",
                        lambda: str(tmp_path / "stamp.json"))
    import time as _t
    (tmp_path / "stamp.json").write_text(json.dumps({"ts": _t.time()}))
    # Fresh stamp → throttled before any network is touched.
    assert asyncio.run(R.maybe_refresh_platform_rules()) is False


def test_research_never_raises_on_network_failure(monkeypatch, tmp_path,
                                                  _isolated_rules):
    from backend.config import settings
    from backend.services import platform_rules_research as R
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(R, "_stamp_path",
                        lambda: str(tmp_path / "stamp.json"))
    # No network in tests → the httpx call fails → fail-soft False.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    assert asyncio.run(R.maybe_refresh_platform_rules(force=True)) is False
    # The attempt stamp was written so failures are throttled too.
    assert os.path.exists(tmp_path / "stamp.json")


def test_seo_intel_endpoint_shape(_isolated_rules):
    from backend.routers.clips import seo_intelligence
    out = asyncio.run(seo_intelligence())
    assert "platform_rules" in out and "tiktok" in out["platform_rules"]
    assert out["platform_rules"]["reels"]["tag_max"] == 5
    assert "banned_tags" in out and "fyp" in out["banned_tags"]
    assert "trend_brief" in out and "platform_rules_meta" in out
