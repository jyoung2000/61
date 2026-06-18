"""Regression tests: the MTPE / LLM JSON parsers must never leak the prompt's
input objects (``{'index': 0, 'text': ...}``) or the source language into the
subtitles.

A small post-editor (qwen2.5:3b) sometimes ECHOES the input objects instead of
returning a flat string array; the old parsers ran ``str(x)`` on each element,
dumping ``{'index': 0, 'text': "...", 'source': 'スペースポート...'}`` straight into
the output. The parsers now extract ``text`` and reject anything that still
looks like a leaked object (fail-soft to the draft / source).
"""

import asyncio
import json
import sys
import types

import pytest


def _stub_provider_sdks():
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    if "groq" not in sys.modules:
        gr = types.ModuleType("groq")
        gr.AsyncGroq = object
        sys.modules["groq"] = gr


_stub_provider_sdks()

from backend.models import TranscriptSegment  # noqa: E402
from backend.services import transcript_polisher as P  # noqa: E402
from backend.services import translator as T  # noqa: E402


# ── _parse_polished_response (MTPE / ASR polish) ──────────────────────────

def test_echoed_objects_with_source_yield_clean_text_only():
    resp = json.dumps([
        {"index": 0, "text": "I'd obliterate everything.", "source": "すべてを消滅させる。"},
        {"index": 1, "text": "I have a combat record of 001.", "source": "戦闘記録001。"},
    ])
    out = P._parse_polished_response(resp, 2)
    assert out == ["I'd obliterate everything.", "I have a combat record of 001."]
    # No dict repr, no Japanese leaked.
    assert all("index" not in s and "source" not in s for s in out)
    assert all("戦闘" not in s and "消滅" not in s for s in out)


def test_stringified_dict_element_is_rejected():
    # The model returned the dict AS a string → must be rejected (keep draft).
    resp = json.dumps(["{'index': 0, 'text': 'Hello'}", "Fine."])
    assert P._parse_polished_response(resp, 2) is None


def test_plain_string_array_passes_through():
    resp = json.dumps(["Hello there.", "Goodbye."])
    assert P._parse_polished_response(resp, 2) == ["Hello there.", "Goodbye."]


def test_object_without_text_key_rejected():
    assert P._parse_polished_response(json.dumps([{"index": 0, "foo": "bar"}]), 1) is None


def test_count_mismatch_rejected():
    assert P._parse_polished_response(json.dumps(["a", "b"]), 3) is None


# ── _parse_json_array (LLM translate path) ────────────────────────────────

def test_translator_parser_extracts_text_from_echoed_objects():
    resp = json.dumps([{"index": 0, "text": "Run!"}, {"index": 1, "text": "Stop."}])
    assert T._parse_json_array(resp, 2) == ["Run!", "Stop."]


def test_translator_parser_rejects_stringified_dict():
    resp = json.dumps(["{'index': 0, 'text': 'Run!'}"])
    assert T._parse_json_array(resp, 1) is None


# ── correct_transcript end-to-end: echo must not corrupt the subtitles ────

def _seg(start, end, text):
    return TranscriptSegment(start=start, end=end, text=text, speaker="Speaker 1")


class _EchoOrch:
    """An orchestrator that echoes the prompt's input objects (incl. source) —
    the exact failure mode that leaked dicts + Japanese into the output."""

    async def text_completion(self, prompt, timeout=90.0, **kw):
        import re
        m = re.search(r"return EXACTLY (\d+)", prompt)
        n = int(m.group(1)) if m else 1
        return json.dumps([
            {"index": i, "text": f"draft line {i + 1}", "source": "日本語のソース"}
            for i in range(n)
        ])


def test_correct_transcript_echo_keeps_clean_draft_no_leak(monkeypatch):
    monkeypatch.setattr(P.settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(P.settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(P.settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)

    drafts = [_seg(0.0, 1.4, "draft line 1"), _seg(1.4, 3.0, "draft line 2")]
    sources = ["彼は店に行った。", "彼女は本を買った。"]
    out = asyncio.run(P.correct_transcript(
        list(drafts), _EchoOrch(), language="en",
        source_texts=sources, source_language="ja", mode="translation"))

    assert len(out) == len(drafts)
    for o, d in zip(out, drafts):
        assert o.start == d.start and o.end == d.end          # timing preserved
        txt = o.text
        assert "index" not in txt and "source" not in txt      # no dict leak
        assert "日本語" not in txt and "ソース" not in txt        # no source leak
        assert not txt.strip().startswith("{")
