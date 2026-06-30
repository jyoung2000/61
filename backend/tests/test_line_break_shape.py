"""Netflix line-break rules in _smart_split: break at clause boundaries, never
mid-proper-noun or after an article/preposition, and prefer a bottom-heavy
shape (top line no longer than the bottom line)."""

from backend.services.subtitle_formatter import (
    _smart_split, _fix_trailing_function_words,
)


def _two_lines(text, max_chars):
    out = _smart_split(text, max_chars)
    assert "\n" in out, f"expected a 2-line wrap, got {out!r}"
    top, bottom = out.split("\n", 1)
    return top, bottom


def test_bottom_heavy_shape():
    top, bottom = _two_lines(
        "The quick brown fox jumps over the lazy dog today", 30)
    assert len(top) <= len(bottom)               # top line not longer
    assert top + " " + bottom == \
        "The quick brown fox jumps over the lazy dog today"


def test_breaks_at_clause_boundary():
    top, bottom = _two_lines("I went to the store, and then I came back home", 30)
    assert top.endswith(",")                     # broke after the comma


def test_never_breaks_proper_noun_pair():
    top, bottom = _two_lines("We met Steve Jobs at the conference hall today", 28)
    # "Steve Jobs" stays together on one line.
    assert ("Steve Jobs" in top) or ("Steve Jobs" in bottom)


def test_smart_split_keeps_article_with_noun():
    # A clause break is available, so the article never ends a line.
    top, bottom = _two_lines("I gave the heavy book to her best friend today", 28)
    assert top.split()[-1].lower() not in {"the", "a", "an", "of", "to", "by"}


def test_greedy_fallback_pushes_trailing_article_down():
    # The fallback wrap must also avoid stranding an article: "the" moves down.
    assert _fix_trailing_function_words("I gave the\nbook to her", 20) == \
        "I gave\nthe book to her"


def test_fixup_noop_when_move_would_overflow():
    # If moving the article would overflow the next line, leave it (no crash).
    out = _fix_trailing_function_words("I gave the\nbook to her now please", 18)
    assert out == "I gave the\nbook to her now please"


def test_short_text_unchanged():
    assert _smart_split("Short line", 42) == "Short line"
