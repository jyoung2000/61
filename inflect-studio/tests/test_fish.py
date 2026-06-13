"""Fish tag mapping (pure) — inflection_to_tags / apply_tags."""

from __future__ import annotations

from inflect.document.spans import EMOTIONS, Inflection
from inflect.models.engine_fish import apply_tags, inflection_to_tags


def _vec(**kw):
    v = [0.0] * 8
    for k, val in kw.items():
        v[EMOTIONS.index(k)] = val
    return v


def test_neutral_has_no_tags():
    assert inflection_to_tags(Inflection()) == ""
    assert apply_tags("hello", Inflection()) == "hello"


def test_emotion_intensity_tiers():
    assert inflection_to_tags(Inflection(emotion_vector=_vec(angry=0.4))) == "[slightly annoyed]"
    assert inflection_to_tags(Inflection(emotion_vector=_vec(angry=0.6))) == "[angry]"
    assert inflection_to_tags(Inflection(emotion_vector=_vec(angry=0.9))) == "[furious]"


def test_below_threshold_ignored():
    # 0.2 is below the 0.35 tag threshold -> no emotion word.
    assert inflection_to_tags(Inflection(emotion_vector=_vec(happy=0.2))) == ""


def test_top_two_dims():
    tag = inflection_to_tags(Inflection(emotion_vector=_vec(angry=0.7, surprised=0.5)))
    assert tag == "[angry and surprised]"


def test_emo_text_preferred_verbatim():
    tag = inflection_to_tags(Inflection(emo_text="whispering, almost crying"))
    assert tag == "[whispering, almost crying]"


def test_emo_text_wins_over_vector():
    inf = Inflection(emotion_vector=_vec(angry=0.9), emo_text="professional broadcast tone")
    assert inflection_to_tags(inf) == "[professional broadcast tone]"


def test_speed_words_appended():
    assert inflection_to_tags(Inflection(emotion_vector=_vec(angry=0.6), speed=1.3)) == "[angry and fast]"
    assert inflection_to_tags(Inflection(speed=0.7)) == "[slow]"
    assert inflection_to_tags(Inflection(speed=1.1)) == "[a little fast]"
    assert inflection_to_tags(Inflection(speed=0.9)) == "[a little slow]"


def test_emo_text_plus_speed():
    inf = Inflection(emo_text="whispering", speed=1.25)
    assert inflection_to_tags(inf) == "[whispering and fast]"


def test_apply_tags_prepends():
    inf = Inflection(emotion_vector=_vec(happy=0.9))
    assert apply_tags("Good morning!", inf) == "[elated] Good morning!"
