"""Task 4 — translation/polish JSON parsers tolerate stray <think>/preamble.

The default local model (Qwen3-4B-Instruct-2507) is non-thinking, but a
mis-tagged or swapped model could emit <think> blocks or preamble. A malformed
batch must fall back (return None → caller keeps the draft), never ship garbage,
and the leaked-struct guard must still apply.
"""

import json
import sys
import types


def _stub_provider_sdks():
    """Stub optional provider SDKs not installed here so ai_orchestrator — and
    thus translator — imports (mirrors test_offline_mtpe.py)."""
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

from backend.services.translator import _parse_json_array  # noqa: E402
from backend.services import transcript_polisher as P  # noqa: E402


# ── translator._parse_json_array ──

def test_strips_closed_think_block():
    resp = "<think>let me reason about this</think>\n" + json.dumps(["Hola", "Adiós"])
    assert _parse_json_array(resp, 2) == ["Hola", "Adiós"]


def test_strips_unclosed_think_block():
    # Truncated mid-think then the array — the array must still be recovered.
    resp = json.dumps(["Hola", "Adiós"]) + "\n<think>oh wait I should reconsider"
    assert _parse_json_array(resp, 2) == ["Hola", "Adiós"]


def test_think_with_brackets_inside_does_not_poison():
    # A think block containing [brackets] must not be mistaken for the array.
    resp = "<think>options: [a, b, c]</think>" + json.dumps(["x", "y"])
    assert _parse_json_array(resp, 2) == ["x", "y"]


def test_markdown_fence_and_think_combined():
    resp = "<think>...</think>\n```json\n" + json.dumps(["one", "two"]) + "\n```"
    assert _parse_json_array(resp, 2) == ["one", "two"]


def test_leaked_struct_still_rejected_on_translation_path():
    resp = json.dumps(["{'index': 0, 'text': 'leak'}", "ok"])
    assert _parse_json_array(resp, 2) is None


def test_garbage_falls_back_to_none():
    assert _parse_json_array("totally not json", 2) is None


# ── transcript_polisher._parse_polished_response ──

def test_polisher_parser_strips_think():
    resp = "<think>reasoning</think>" + json.dumps(["A.", "B."])
    assert P._parse_polished_response(resp, 2) == ["A.", "B."]


def test_polisher_parser_leaked_struct_per_index():
    # Per-segment salvage: leaked element → None, good element kept.
    resp = json.dumps(["{'index': 0, 'text': 'leak'}", "Fine."])
    assert P._parse_polished_response(resp, 2) == [None, "Fine."]
