"""Tests: summary + SEO are emitted in the user's selected subtitle language,
not a hardcoded English (and not the raw transcript language).
"""

from backend.services.prompts import (
    output_language_name,
    summary_language_directive,
    build_platform_seo_prompt,
    DEFAULT_SUMMARY_PROMPT,
)


def test_language_name_mapping():
    assert output_language_name("en") == "English"
    assert output_language_name("ja") == "Japanese"
    assert output_language_name("es-ES") == "Spanish"     # strips region
    assert output_language_name("") == ""
    assert output_language_name("xx") == ""               # unknown


def test_summary_directive_targets_language():
    d = summary_language_directive("ja")
    assert "Japanese" in d and "OUTPUT LANGUAGE" in d
    assert "ONLY in Japanese" in d


def test_summary_directive_defaults_to_english():
    assert "English" in summary_language_directive("")     # historical default
    assert "English" in summary_language_directive("xx")


def test_default_summary_prompt_no_longer_hardcodes_english():
    assert "ALWAYS write your entire response in English" not in DEFAULT_SUMMARY_PROMPT


def test_seo_prompt_uses_subtitle_language():
    ja = build_platform_seo_prompt("tiktok", output_language="ja")
    assert "Japanese" in ja and "OUTPUT LANGUAGE" in ja
    es = build_platform_seo_prompt("youtube_shorts", output_language="es")
    assert "Spanish" in es


def test_seo_prompt_defaults_to_english():
    assert "English" in build_platform_seo_prompt("tiktok")


def test_seo_prompt_keeps_trend_and_language_together():
    p = build_platform_seo_prompt("tiktok", trend_brief="#anitok", output_language="es")
    assert "Spanish" in p and "#anitok" in p and "LIVE TREND BRIEF" in p
