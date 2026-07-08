"""Tests for the editorial-LLM translation path — the fundamental rethink.

Whisper's task='translate' left music/narration in the source language, yielding
half-Japanese "translated" tracks. Translation now goes text-to-text through the
LLM (1:1, every cue), with a language-purity gate rejecting source-language
output. These pin that behaviour.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from backend.models import TranscriptSegment
from backend.services.translator import (
    _cjk_ratio, _parse_json_array, fraction_untranslated, translate_via_llm,
)


def _run(coro):
    return asyncio.run(coro)


def test_idiomatic_rule_toggle(monkeypatch):
    from backend.services import translator
    monkeypatch.setattr(translator.settings, "TRANSLATION_IDIOMATIC", True)
    rule = translator._idiomatic_rule()
    assert "idiomatic" in rule.lower()
    assert "{" not in rule          # placeholder-free → survives TRANSLATION_PROMPT.format()
    monkeypatch.setattr(translator.settings, "TRANSLATION_IDIOMATIC", False)
    assert translator._idiomatic_rule() == ""


class _Orch:
    """Minimal orchestrator stub: echoes back N English lines as a JSON array."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    async def text_completion(self, prompt, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no usable LLM")
        n = len(re.findall(r"^\d+\. ", prompt, re.M))
        return json.dumps([f"English line {i + 1}" for i in range(n)])


def _segs(n, prefix="日本語の文"):
    return [TranscriptSegment(text=f"{prefix}{i}", start=float(i), end=float(i + 1),
                              speaker="Speaker 1") for i in range(n)]


def test_llm_parallel_batches_preserve_order(monkeypatch):
    """Turbo-mode fan-out: batches run concurrently but reassemble in source
    order, 1:1, with timing preserved."""
    import asyncio
    from backend.services import translator as T

    monkeypatch.setattr(T.settings, "TRANSLATION_AUTO_GLOSSARY", False, raising=False)

    async def _fake_conc():
        return 4                                  # pretend a Turbo Companion

    monkeypatch.setattr(T, "_translation_batch_concurrency", _fake_conc)

    class _EchoOrch:
        def __init__(self):
            self.inflight = 0
            self.max_inflight = 0

        async def text_completion(self, prompt, **kwargs):
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            await asyncio.sleep(0.01)             # let concurrent batches overlap
            lines = re.findall(r"^\d+\.\s+(.*)$", prompt, re.M)
            self.inflight -= 1
            return json.dumps([f"EN[{ln}]" for ln in lines])

    orch = _EchoOrch()
    out = _run(translate_via_llm(_segs(40), "ja", "en", orch))   # BATCH=18 → 3 batches
    assert len(out) == 40
    # Reassembled in source order (cue i ← source i), not completion order.
    assert out[0].text == "EN[日本語の文0]"
    assert out[19].text == "EN[日本語の文19]"
    assert out[39].text == "EN[日本語の文39]"
    assert out[7].start == 7.0 and out[7].speaker == "Speaker 1"
    # The non-first batches actually ran concurrently.
    assert orch.max_inflight >= 2


def test_llm_sequential_when_no_turbo(monkeypatch):
    """With concurrency 1 (local card / Eco), batches run one at a time."""
    import asyncio
    from backend.services import translator as T

    async def _one():
        return 1

    monkeypatch.setattr(T, "_translation_batch_concurrency", _one)

    class _SeqOrch(_Orch):
        def __init__(self):
            super().__init__()
            self.inflight = 0
            self.max_inflight = 0

        async def text_completion(self, prompt, **kwargs):
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            await asyncio.sleep(0.005)
            self.inflight -= 1
            return await super().text_completion(prompt, **kwargs)

    orch = _SeqOrch()
    out = _run(translate_via_llm(_segs(40), "ja", "en", orch))
    assert len(out) == 40
    assert orch.max_inflight == 1                 # strictly sequential


def test_cjk_and_fraction_detectors():
    assert _cjk_ratio("宇宙コロニーでの生活に") > 0.9
    assert _cjk_ratio("The opponent was a mobile suit") == 0.0
    mixed = [TranscriptSegment(text="宇宙コロニー", start=0, end=1, speaker="S"),
             TranscriptSegment(text="Hello there", start=1, end=2, speaker="S")]
    # Content-based: detects CJK in the output regardless of declared source
    # (this is what makes it work when the source was detected as "auto").
    assert fraction_untranslated(mixed, "en") == 0.5           # half still CJK → English target
    assert fraction_untranslated(mixed, "ja") == 0.0           # CJK target → CJK is correct


def test_parse_json_array():
    assert _parse_json_array('["a", "b"]', 2) == ["a", "b"]
    assert _parse_json_array('```json\n["a", "b"]\n```', 2) == ["a", "b"]
    assert _parse_json_array('here you go: ["a", "b"] ok', 2) == ["a", "b"]
    assert _parse_json_array('["a"]', 2) is None               # count mismatch
    assert _parse_json_array('not json at all', 1) is None


def test_llm_translates_every_segment_1to1():
    segs = _segs(40)                                            # > 2 batches (BATCH=18)
    out = _run(translate_via_llm(segs, "ja", "en", _Orch()))
    assert len(out) == 40                                      # 1:1, nothing dropped
    assert all("English line" in s.text for s in out)         # every cue translated
    assert fraction_untranslated(out, "en") == 0.0            # none left Japanese
    assert out[5].start == 5.0 and out[5].speaker == "Speaker 1"  # timing/speaker kept


def test_llm_cleanup_retranslates_leftover_cjk():
    """The model sometimes echoes a hard line untranslated inside a valid array.
    The completeness pass must re-translate any cue still in CJK script so the
    final output has NONE of the source language left — even when the source was
    'auto'."""
    class _Leftover:
        def __init__(self):
            self.seen = set()

        async def text_completion(self, prompt, **kwargs):
            lines = re.findall(r"^\d+\.\s(.+)$", prompt, re.M)
            out = []
            for ln in lines:
                # Leave one specific Japanese line untranslated the FIRST time,
                # then translate it (pure English) on the cleanup retry.
                if ln == "日本語の文2" and ln not in self.seen:
                    self.seen.add(ln)
                    out.append(ln)                # echo → still CJK
                else:
                    out.append("Translated text")  # pure English
            return json.dumps(out)

    out = _run(translate_via_llm(_segs(5), "auto", "en", _Leftover()))
    assert len(out) == 5
    assert fraction_untranslated(out, "en") == 0.0            # cleanup fixed the leftover
    assert all(_cjk_ratio(s.text) == 0.0 for s in out)


def test_llm_applies_glossary():
    segs = _segs(2)
    out = _run(translate_via_llm(segs, "ja", "en", _Orch(),
                                 glossary={"English line 1": "Heero"}))
    assert out[0].text == "Heero"


def test_llm_bails_when_model_unusable():
    # First batch failing outright → return None so the caller falls back cleanly
    # instead of "translating" every line to itself.
    assert _run(translate_via_llm(_segs(20), "ja", "en", _Orch(fail=True))) is None


def test_llm_none_without_orchestrator():
    assert _run(translate_via_llm(_segs(3), "ja", "en", None)) is None


def test_translate_offline_prefers_llm_over_whisper():
    from backend.services.pipeline import translate_offline
    orch = _Orch()
    out, engine = _run(translate_offline(_segs(20), "ja", "en",
                                         orchestrator=orch, job_id=None))
    assert engine == "llm"                                     # LLM path won
    assert len(out) == 20
    assert all("English line" in s.text for s in out)
    assert orch.calls >= 1
