"""Romaji (transliterated-Japanese) passthrough detection. A model that spells a
hard Japanese line out phonetically ("Katte ippai tsukuritaku naru toko") emits
NO CJK script, so the old CJK-only check scored it 0% translated and shipped it
in the English track. Detection is gated to a Japanese source so it can't
false-positive on open-syllable Romance languages."""

import sys
import types


def _stub():
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    for n, a in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                 ("anthropic", "AsyncAnthropic")):
        sys.modules.setdefault(n, type(sys)(n))
        setattr(sys.modules[n], a, object)


_stub()

from backend.services.translator import (  # noqa: E402
    _is_untranslated, _romaji_ja_ratio, fraction_untranslated,
)

ROMAJI = [
    "Katte ippai tsukuritaku naru toko aru ne, takusan doko?",
    "Kowayade ugokashite sou purupuru itteru purupuru itteru oppai no kanshoku tte iki",
    "Tsukuritaku naru toko ippai aru.",
    "Nani de koko soko?",
    "Nanika atte, chianbu no ippai aru ne e-e, sonna koto nai yo honto?",
]
ENGLISH = [
    "Let's eat!",
    "You're eating so much.",
    "I finished all of it — delicious! Thanks for your hard work.",
    "Big sister, shall we insert two?",
    "It really feels soft and smooth.",
    "Why hide your face with hands?",
]


def test_romaji_flagged_when_source_japanese():
    for line in ROMAJI:
        assert _is_untranslated(line, "ja"), line


def test_english_not_flagged_even_with_japanese_source():
    for line in ENGLISH:
        assert not _is_untranslated(line, "ja"), line


def test_romaji_not_flagged_without_japanese_source():
    # Unknown/auto source → don't apply romaji detection (avoid false positives).
    assert not _is_untranslated(ROMAJI[0], "")
    # Romance-language source with open syllables must never trip it.
    assert not _is_untranslated("nada mas la vida", "es")


def test_cjk_still_flagged_any_source():
    assert _is_untranslated("これはテストです", "")
    assert _is_untranslated("これはテストです", "ja")


def test_fraction_untranslated_counts_romaji_for_ja_source():
    segs = [{"text": t} for t in (ROMAJI[:2] + ENGLISH[:2])]  # 2 romaji, 2 english
    # Japanese source → romaji counted → 50%.
    assert abs(fraction_untranslated(segs, "en", "ja") - 0.5) < 1e-6
    # No source declared → CJK-only → 0% (romaji not counted).
    assert fraction_untranslated(segs, "en") == 0.0
