"""Pure UI formatting helpers (tooltips, colors) — no Qt required."""

from __future__ import annotations

from inflect.config import SPAN_COLORS
from inflect.document.spans import EMOTIONS, Inflection
from inflect.ui.format import (
    NEUTRAL_COLOR_IDX,
    active_emotions,
    color_for_inflection,
    color_hex,
    dominant_emotion,
    summarize_inflection,
)


def _vec(**kw):
    v = [0.0] * 8
    for k, val in kw.items():
        v[EMOTIONS.index(k)] = val
    return v


def test_dominant_emotion():
    assert dominant_emotion(None) is None
    assert dominant_emotion([0.0] * 8) is None
    name, val = dominant_emotion(_vec(angry=0.7, happy=0.2))
    assert name == "angry"
    assert val == 0.7


def test_active_emotions_sorted():
    pairs = active_emotions(_vec(happy=0.3, angry=0.8, calm=0.1))
    assert [p[0] for p in pairs] == ["angry", "happy", "calm"]


def test_color_follows_dominant_emotion():
    # angry is index 1 in EMOTIONS -> color index 1.
    assert color_for_inflection(Inflection(emotion_vector=_vec(angry=0.9))) == 1
    # No emotion -> neutral color.
    assert color_for_inflection(Inflection(emo_text="whispering")) == NEUTRAL_COLOR_IDX
    assert color_for_inflection(Inflection(speed=1.3)) == NEUTRAL_COLOR_IDX


def test_color_hex_wraps():
    assert color_hex(0) == SPAN_COLORS[0]
    assert color_hex(len(SPAN_COLORS)) == SPAN_COLORS[0]


def test_summarize_default():
    assert summarize_inflection(Inflection()) == "default delivery"


def test_summarize_combined():
    inf = Inflection(
        emotion_vector=_vec(angry=0.7, surprised=0.3),
        speed=1.1,
        pause_after_ms=200,
        engine="fish",
    )
    s = summarize_inflection(inf)
    assert "angry 0.7" in s
    assert "surprised 0.3" in s
    assert "1.1×" in s
    assert "200ms" in s
    assert "Fish" in s


def test_summarize_truncates_long_text():
    inf = Inflection(emo_text="a" * 50)
    s = summarize_inflection(inf)
    assert "…" in s
