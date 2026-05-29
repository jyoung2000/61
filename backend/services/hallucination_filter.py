"""Multilingual Whisper hallucination filter.

Whisper emits fixed "filler" phrases over music, silence, and end credits.
For non-English audio the hallucination is emitted in the SOURCE language —
the canonical Japanese "ご視聴ありがとうございました" (thank-you-for-watching)
is the most common, and used to slip past an English-only blocklist and
surface post-translation as "Thank you for watching".

Kept dependency-light (no cv2/torch) so it can be unit-tested and reused by
both the main and gap-fill transcription passes.
"""

from __future__ import annotations

# Known fixed-phrase hallucinations across the languages we transcribe.
BOILERPLATE_HALLUCINATIONS = frozenset({
    # English
    'thank you for watching', 'thanks for watching',
    'please subscribe', 'like and subscribe',
    "don't forget to subscribe", 'see you in the next video',
    'bye bye', 'thanks for listening', 'music playing',
    'music', 'applause', 'subtitles by', 'captions by',
    'thank you', 'thanks', 'the end',
    # Japanese — the most common Whisper-JA hallucinations over music/credits.
    'ご視聴ありがとうございました', 'ご視聴ありがとうございます',
    'ご視聴いただきありがとうございました',
    'ご視聴いただきありがとうございます',
    '最後までご視聴いただきありがとうございました',
    'チャンネル登録お願いします', 'チャンネル登録をお願いします',
    'ありがとうございました', 'おやすみなさい', '次回もお楽しみに',
    'バイバイ',
    # Korean
    '시청해주셔서 감사합니다', '구독과 좋아요', '감사합니다',
    # Chinese
    '请订阅', '谢谢观看', '谢谢大家', '感谢观看',
})

# Punctuation stripped before matching — ASCII plus CJK terminators/quotes so
# a hallucination ending in 。！？ still matches its bare form.
_STRIP_CHARS = " .!,?。！？、…「」『』（）()[]{}\"'♪~-"


def is_boilerplate_hallucination(text: str) -> bool:
    """True when ``text`` is a known fixed-phrase Whisper hallucination
    (language-agnostic — English + JA/KO/ZH canonical forms)."""
    if not text:
        return False
    key = text.strip().lower().strip(_STRIP_CHARS).strip()
    return key in BOILERPLATE_HALLUCINATIONS
