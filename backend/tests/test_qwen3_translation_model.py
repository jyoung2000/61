"""Qwen3-4B-Instruct-2507 as the local translation model.

Covers: param-size parse accepts a 4b tag, non-thinking classification of the
Instruct-2507 line, auto-rank preference, configured override beats auto-rank,
and the gated Qwen3 sampling options.
"""

import asyncio

import pytest

from backend.config import settings
from backend.services.local_models import (
    _parse_params_b, rank_local_editorial_models, select_local_editorial_models,
    qwen3_translation_options,
)
from backend.services.providers.base import AIProvider


# ── Task 1: param-size parse accepts a 4b tag (≤ 4.0, not excluded) ──

@pytest.mark.parametrize("tag,expected", [
    ("qwen3:4b-instruct-2507-q4_K_M", 4.0),
    ("qwen3:4b-instruct-2507-q8_0", 4.0),
    ("qwen3:4b-instruct-2507-fp16", 4.0),
    ("qwen3:4b", 4.0),
    ("qwen2.5:3b-instruct", 3.0),
    ("llama3.1:8b", 8.0),
])
def test_parse_params_b(tag, expected):
    assert _parse_params_b(tag) == expected


def test_4b_tag_not_excluded_by_max_params():
    # OFFLINE_EDITORIAL_MAX_PARAMS_B = 4.0 must still PERMIT a 4b tag.
    ranked = rank_local_editorial_models(
        ["qwen3:4b-instruct-2507-q4_K_M", "llama3.1:8b"], max_params_b=4.0)
    assert "qwen3:4b-instruct-2507-q4_K_M" in ranked
    assert "llama3.1:8b" not in ranked  # >4B correctly dropped


# ── Task 2: ranking + configured override ──

def test_2507_ranks_above_qwen25_and_plain_instruct():
    ranked = rank_local_editorial_models(
        ["qwen2.5:3b-instruct", "qwen3:4b-instruct", "qwen3:4b-instruct-2507-q4_K_M"],
        max_params_b=4.0)
    assert ranked[0] == "qwen3:4b-instruct-2507-q4_K_M"


def test_configured_override_beats_auto_rank(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen2.5:3b-instruct")
    installed = ["qwen3:4b-instruct-2507-q4_K_M", "qwen2.5:3b-instruct"]
    got = asyncio.run(select_local_editorial_models(limit=2, model_names=installed))
    assert got[0] == "qwen2.5:3b-instruct"  # explicit choice wins over auto-rank


def test_override_matches_with_latest_suffix(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    installed = ["qwen2.5:3b-instruct", "qwen3:4b-instruct-2507-q4_K_M:latest"]
    got = asyncio.run(select_local_editorial_models(limit=2, model_names=installed))
    assert got[0] == "qwen3:4b-instruct-2507-q4_K_M:latest"


# ── Task 3: non-thinking classification ──

class _FakeProvider(AIProvider):
    """Minimal concrete AIProvider so the real is_thinking_model property runs."""

    def __init__(self, model):
        self._model = model

    @property
    def provider_name(self):
        return "ollama"

    @property
    def text_model_name(self):
        return self._model

    # Abstract surface — trivial stubs (unused by the property under test).
    async def analyze_frames(self, *a, **k):
        return []

    async def generate_summary(self, *a, **k):
        return ""

    async def detect_viral_clips(self, *a, **k):
        return []

    async def generate_seo(self, *a, **k):
        return {}

    def supports_vision(self):
        return False


@pytest.mark.parametrize("model,is_thinking", [
    ("qwen3:4b-instruct-2507-q4_K_M", False),
    ("qwen3:4b-instruct", False),
    ("qwen3:4b-thinking-2507-q4_K_M", True),
    ("qwen3:4b", True),            # bare qwen3 → thinking line
    ("qwq:32b", True),
    ("deepseek-r1:7b", True),
    ("qwen2.5:3b-instruct", False),
    ("google/gemini-2.5-flash", True),
])
def test_is_thinking_classification(model, is_thinking):
    assert _FakeProvider(model).is_thinking_model is is_thinking


# ── Task 1: gated Qwen3 sampling ──

def test_qwen3_sampling_applies_only_to_qwen3():
    opts = qwen3_translation_options("qwen3:4b-instruct-2507-q4_K_M")
    assert opts["temperature"] == settings.QWEN3_TRANSLATION_TEMPERATURE
    assert opts["repeat_penalty"] == settings.QWEN3_TRANSLATION_REPEAT_PENALTY
    assert opts["presence_penalty"] == settings.QWEN3_TRANSLATION_PRESENCE_PENALTY
    assert opts["top_p"] == settings.QWEN3_TRANSLATION_TOP_P
    assert qwen3_translation_options("qwen2.5:3b-instruct") == {}
    assert qwen3_translation_options("llama3.1:8b") == {}
