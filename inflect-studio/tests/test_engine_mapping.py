"""Pure inflection→engine-parameter mappings (no model loading)."""

from __future__ import annotations

import numpy as np
import pytest

from inflect.document.spans import EMOTIONS, Inflection
from inflect.models.engine_chatterbox import (
    MAX_EXAGGERATION,
    MIN_EXAGGERATION,
    NEUTRAL_EXAGGERATION,
    emotion_to_exaggeration,
)
from inflect.models.engine_indextts2 import resolve_emotion
from inflect.synth.assemble import time_stretch


def _vec(**kw) -> list[float]:
    v = [0.0] * 8
    for name, val in kw.items():
        v[EMOTIONS.index(name)] = val
    return v


# --------------------------------------------------------------------------- #
# Chatterbox exaggeration mapping
# --------------------------------------------------------------------------- #
def test_neutral_exaggeration():
    assert emotion_to_exaggeration(Inflection()) == NEUTRAL_EXAGGERATION
    assert emotion_to_exaggeration(Inflection(emotion_vector=[0.0] * 8)) == NEUTRAL_EXAGGERATION


def test_angry_raises_exaggeration():
    angry = Inflection(emotion_vector=_vec(angry=1.0))
    assert emotion_to_exaggeration(angry) > NEUTRAL_EXAGGERATION


def test_energetic_beats_calm():
    angry = Inflection(emotion_vector=_vec(angry=0.8))
    calm = Inflection(emotion_vector=_vec(calm=0.8))
    assert emotion_to_exaggeration(angry) > emotion_to_exaggeration(calm)


def test_exaggeration_bounds():
    for vec in ([1.0] * 8, _vec(angry=1.0, happy=1.0, surprised=1.0), _vec(calm=0.01)):
        x = emotion_to_exaggeration(Inflection(emotion_vector=vec))
        assert MIN_EXAGGERATION <= x <= MAX_EXAGGERATION


# --------------------------------------------------------------------------- #
# IndexTTS-2 emotion resolution + precedence
# --------------------------------------------------------------------------- #
def test_resolve_vector_only():
    args = resolve_emotion(Inflection(emotion_vector=_vec(sad=0.8), emo_alpha=0.6))
    assert args.emo_vector == _vec(sad=0.8)
    assert args.use_emo_text is False
    assert args.emo_text is None
    assert args.emo_audio_prompt is None
    assert args.emo_alpha == pytest.approx(0.6)


def test_resolve_vector_wins_over_text():
    args = resolve_emotion(
        Inflection(emotion_vector=_vec(happy=1.0), emo_text="whispering")
    )
    assert args.emo_vector == _vec(happy=1.0)
    assert args.use_emo_text is False
    assert args.emo_text is None


def test_resolve_text_only():
    args = resolve_emotion(Inflection(emo_text="almost crying"))
    assert args.use_emo_text is True
    assert args.emo_text == "almost crying"
    assert args.emo_vector is None


def test_resolve_audio_only():
    args = resolve_emotion(Inflection(emo_audio="/tmp/perf.wav", emo_alpha=0.75))
    assert args.emo_audio_prompt == "/tmp/perf.wav"
    assert args.emo_vector is None
    assert args.use_emo_text is False
    assert args.emo_alpha == pytest.approx(0.75)


def test_resolve_neutral():
    args = resolve_emotion(Inflection())
    assert args.emo_vector is None
    assert args.emo_audio_prompt is None
    assert args.use_emo_text is False


def test_resolve_zero_vector_falls_through_to_text():
    # An all-zero vector is "no emotion" -> text channel should be used instead.
    args = resolve_emotion(Inflection(emotion_vector=[0.0] * 8, emo_text="tense"))
    assert args.emo_vector is None
    assert args.use_emo_text is True
    assert args.emo_text == "tense"


# --------------------------------------------------------------------------- #
# time_stretch fallback behaviour
# --------------------------------------------------------------------------- #
def test_time_stretch_noop_at_unit_speed():
    a = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = time_stretch(a, 1.0)
    assert np.array_equal(out, a)


def test_time_stretch_returns_float32_array():
    a = np.sin(np.linspace(0, 10, 4096)).astype(np.float32)
    out = time_stretch(a, 1.25)
    assert out.dtype == np.float32
    assert out.ndim == 1
    # With librosa present the length scales ~1/speed; without it, unchanged.
    # Either way it must be a sane, finite signal.
    assert np.all(np.isfinite(out))
