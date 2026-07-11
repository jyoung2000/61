"""Tests for the source-before-translation polish parity move (Task 6).

When a job translates, the heavy readability reflow runs on the TARGET text, but
a single LIGHT source-language cleanup (punctuation / casing / filler) now runs
BEFORE translation so the translator works from clean input and the shipped
source transcript reads cleanly. Gated by TRANSLATION_POLISH_SOURCE_FIRST and
the master polish switches; fail-soft and cue-count / timing preserving.
"""

import asyncio
import json
import re

import pytest

from backend.models import TranscriptSegment
from backend.services import transcript_polisher as P


def _seg(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _raw_source():
    # Raw Whisper-style output: missing punctuation, filler, lowercase starts.
    return [
        _seg(0.0, 1.5, "um so he went to the store"),
        _seg(1.5, 3.0, "and uh she bought a book"),
        _seg(3.0, 4.5, "then they went home"),
    ]


class _FakeOrch:
    """Editorial-model stand-in returning a sized JSON array (no network)."""

    def __init__(self):
        self.calls = 0

    async def text_completion(self, prompt, timeout=90.0, **kw):
        self.calls += 1
        m = re.search(r"JSON array of (\d+) objects", prompt) or re.search(r"return EXACTLY (\d+)", prompt)
        n = int(m.group(1)) if m else 1
        return json.dumps([f"Polished line {i + 1}." for i in range(n)])


@pytest.fixture(autouse=True)
def _polish_settings(monkeypatch):
    monkeypatch.setattr(P.settings, "TRANSLATION_POLISH_SOURCE_FIRST", True, raising=False)
    monkeypatch.setattr(P.settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(P.settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(P.settings, "TRANSCRIPT_PRESERVE_WORDS", True, raising=False)
    monkeypatch.setattr(P.settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)


def test_source_polish_preserves_count_and_timing(monkeypatch):
    orch = _FakeOrch()
    raw = _raw_source()
    out = asyncio.run(P.polish_source_before_translation(raw, orch, "en"))

    assert orch.calls >= 1
    assert len(out) == len(raw)
    for o, r in zip(out, raw):
        assert o.start == r.start and o.end == r.end       # timing preserved
        assert o.speaker == r.speaker


def test_source_polish_disabled_returns_raw(monkeypatch):
    monkeypatch.setattr(P.settings, "TRANSLATION_POLISH_SOURCE_FIRST", False)
    orch = _FakeOrch()
    raw = _raw_source()
    out = asyncio.run(P.polish_source_before_translation(raw, orch, "en"))
    assert out == raw
    assert orch.calls == 0                                  # model never called


def test_source_polish_no_orchestrator_returns_raw():
    raw = _raw_source()
    out = asyncio.run(P.polish_source_before_translation(raw, None, "en"))
    assert out == raw


def test_source_polish_respects_master_switch(monkeypatch):
    monkeypatch.setattr(P.settings, "AI_TRANSCRIPT_CORRECTION", False)
    orch = _FakeOrch()
    raw = _raw_source()
    out = asyncio.run(P.polish_source_before_translation(raw, orch, "en"))
    assert out == raw
    assert orch.calls == 0


def test_source_polish_fail_soft_on_count_change(monkeypatch):
    raw = _raw_source()

    async def _bad(*a, **k):
        return raw[:-1]                                     # drops a cue

    monkeypatch.setattr(P, "correct_transcript", _bad)
    out = asyncio.run(P.polish_source_before_translation(raw, _FakeOrch(), "en"))
    assert out == raw                                       # raw kept


def test_source_polish_empty_input():
    assert asyncio.run(P.polish_source_before_translation([], _FakeOrch(), "en")) == []
