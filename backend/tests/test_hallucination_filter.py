"""Tests for the multilingual Whisper hallucination filter.

A Gundam Wing job surfaced "Thank you for watching" twice (mid-content and
over the end credits). Root cause: Whisper hallucinated the canonical
Japanese phrase ご視聴ありがとうございました over music, and the English-only
blocklist couldn't catch it before translation turned it into English.
"""

import pytest

from backend.services.hallucination_filter import is_boilerplate_hallucination


@pytest.mark.parametrize("text", [
    "Thank you for watching.",
    "thanks for watching",
    "Please subscribe!",
    "Music",
    "[Applause]",  # brackets stripped
    # Japanese — the actual source-language hallucinations.
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "チャンネル登録をお願いします",
    "ありがとうございました",
    # Korean / Chinese
    "시청해주셔서 감사합니다",
    "谢谢观看",
    # "see you next time" family + bare JA "the end" — the phantoms that
    # flooded a mostly-silent 2 h video (English emitted directly on JA audio).
    "See you next time.",
    "I'll see you next time.",
    "おわり",
    "終わり",
    "おしまい",
    "また次回",
])
def test_known_hallucinations_detected(text):
    assert is_boilerplate_hallucination(text) is True


@pytest.mark.parametrize("text", [
    "But the United Earth Sphere Alliance gains military power.",
    "Operation Meteor.",
    "I am Heero Yuy.",
    "",
    "   ",
    "thank you so much for the detailed explanation",  # real speech, not the bare phrase
])
def test_real_speech_not_flagged(text):
    assert is_boilerplate_hallucination(text) is False


def test_trailing_cjk_punctuation_stripped():
    # The bare phrase plus a CJK terminator still matches.
    assert is_boilerplate_hallucination("ご視聴ありがとうございました！")
    assert is_boilerplate_hallucination("谢谢观看。")
