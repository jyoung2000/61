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

import re

# Known fixed-phrase hallucinations across the languages we transcribe.
BOILERPLATE_HALLUCINATIONS = frozenset({
    # English
    'thank you for watching', 'thanks for watching',
    'please subscribe', 'like and subscribe',
    "don't forget to subscribe", 'see you in the next video',
    'bye bye', 'thanks for listening', 'music playing',
    'music', 'applause', 'subtitles by', 'captions by',
    'thank you', 'thanks', 'the end',
    # "see you next time" family — a very common Whisper end-of-segment phantom
    # (emitted directly in English even on JA audio); seen repeating across a
    # mostly-silent video. Standalone-only (exact cue match), so real dialogue
    # is unaffected.
    'see you next time', "i'll see you next time", 'see you next week',
    'see you again', 'see you',
    # Japanese — the most common Whisper-JA hallucinations over music/credits.
    'ご視聴ありがとうございました', 'ご視聴ありがとうございます',
    'ご視聴いただきありがとうございました',
    'ご視聴いただきありがとうございます',
    '最後までご視聴いただきありがとうございました',
    'チャンネル登録お願いします', 'チャンネル登録をお願いします',
    'ありがとうございました', 'おやすみなさい', '次回もお楽しみに',
    'バイバイ',
    # JA "the end" / "see you next time" phantoms (the bare 終わり/おわり cue
    # floods credits + silent stretches). Exact-cue match only.
    'おわり', '終わり', 'おしまい', 'また次回', 'また来週', 'また来年',
    'また会いましょう',
    # Korean
    '시청해주셔서 감사합니다', '구독과 좋아요', '감사합니다',
    # Chinese
    '请订阅', '谢谢观看', '谢谢大家', '感谢观看',
    # Transcription/caption vendor credits Whisper invents over intros + music
    # (e.g. "Transcription by CastingWords", "Subtitles by the amara.org
    # community"). The exact strings help; the regexes below catch any vendor.
    'transcription by castingwords', 'transcribed by castingwords',
    'subtitles by the amara.org community', 'amara.org', 'www.amara.org',
})

# Attribution "credits" Whisper hallucinates over silence / music / end cards.
# These are never real dialogue regardless of the vendor named, so match the
# whole "<credit> by …" / "… by <vendor>" family, not just fixed strings.
_ATTRIBUTION_RE = re.compile(
    r'^(?:the\s+)?(?:transcription|transcript|transcribed|subtitles?|subs?|'
    r'captions?|caption|closed\s+captions?|translation|translated)\s+by\b',
    re.IGNORECASE,
)
_CREDIT_VENDOR_RE = re.compile(
    r'\bby\s+(?:castingwords|amara\.org|the\s+amara\.org\s+community|'
    r'rev\.com|gotranscript|otter\.ai|happyscribe|verbit)\b',
    re.IGNORECASE,
)

# Punctuation stripped before matching — ASCII plus CJK terminators/quotes so
# a hallucination ending in 。！？ still matches its bare form.
_STRIP_CHARS = " .!,?。！？、…「」『』（）()[]{}\"'♪~-"


def is_boilerplate_hallucination(text: str) -> bool:
    """True when ``text`` is a known fixed-phrase Whisper hallucination
    (language-agnostic — English + JA/KO/ZH canonical forms)."""
    if not text:
        return False
    key = text.strip().lower().strip(_STRIP_CHARS).strip()
    if key in BOILERPLATE_HALLUCINATIONS:
        return True
    # Attribution credits ("Transcription by CastingWords", "Subtitles by the
    # Amara.org community", …) — always phantoms, whatever the vendor.
    if _ATTRIBUTION_RE.search(key) or _CREDIT_VENDOR_RE.search(key):
        return True
    return False
