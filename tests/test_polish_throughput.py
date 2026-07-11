"""Polish-pass throughput + robustness (the 43-minute-translation-polish fix).

A real 128-min run spent 43 minutes in the translation polish: 57 batches ran
STRICTLY SEQUENTIALLY against the auto-upgraded qwen2.5:14b, 15 of them were
discarded whole on 'bad response shape (expected 15 items)', and only 582/851
cues got polished. These tests pin the fixes:

  * index-keyed response contract — a response with explicit {"index", "text"}
    objects parses even when the model loses count; missing indices keep the
    original text (exact mapping, never positional guessing);
  * batches run with bounded concurrency, results assembled in order;
  * SUBTITLE_POLISH_MAX_S caps the pass — cues after the budget keep drafts;
  * a first-batch latency probe DOWNSHIFTS the auto-upgraded model back to the
    base model when its throughput would blow the budget;
  * 5 consecutive whole-batch failures abort the remainder (drafts kept).
"""

import asyncio
import json
import re
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings
from backend.services import transcript_polisher as P


# ─────────────────────────────────────────────────────────────────────────────
# Parser: index-keyed salvage
# ─────────────────────────────────────────────────────────────────────────────

def test_indexed_objects_parse_exactly():
    resp = json.dumps([
        {"index": 0, "text": "Zero."},
        {"index": 2, "text": "Two."},
    ])
    slots = P._parse_polished_response(resp, expected=3)
    assert slots == ["Zero.", None, "Two."]


def test_indexed_objects_survive_wrong_count():
    # The old contract discarded this batch whole ("bad response shape").
    resp = json.dumps([{"index": 1, "text": "Only one came back."}])
    slots = P._parse_polished_response(resp, expected=15)
    assert slots is not None
    assert slots[1] == "Only one came back."
    assert all(s is None for i, s in enumerate(slots) if i != 1)


def test_numeric_keyed_dict_parses():
    resp = json.dumps({"0": "A.", "1": "B."})
    assert P._parse_polished_response(resp, expected=2) == ["A.", "B."]


def test_out_of_range_and_bogus_indices_skipped():
    resp = json.dumps([
        {"index": 99, "text": "nope"},
        {"index": "x", "text": "nope"},
        {"index": 0, "text": "Kept."},
    ])
    assert P._parse_polished_response(resp, expected=2) == ["Kept.", None]


def test_legacy_flat_array_still_parses():
    resp = json.dumps(["One.", "Two.", "Three."])
    assert P._parse_polished_response(resp, expected=3) == ["One.", "Two.", "Three."]


def test_flat_array_wrong_count_still_rejected():
    # Un-indexed mismatch stays ambiguous → whole batch kept raw.
    assert P._parse_polished_response(json.dumps(["A", "B"]), expected=3) is None


# ─────────────────────────────────────────────────────────────────────────────
# Batch loop: concurrency, budget, downshift, breaker
# ─────────────────────────────────────────────────────────────────────────────

def _segs(n):
    return [{"start": float(i), "end": float(i) + 1.0, "text": f"raw {i}"}
            for i in range(n)]


class _Orch:
    """Indexed-format responder; records call order/models; optional delay."""

    def __init__(self, delay_s=0.0, fail_all=False):
        self.calls = []
        self.delay_s = delay_s
        self.fail_all = fail_all
        self.max_inflight = 0
        self._inflight = 0

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            self.calls.append(model_override)
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.fail_all:
                return "not json at all"
            m = re.search(r"JSON array of (\d+) objects", prompt)
            n = int(m.group(1)) if m else 1
            idxs = re.findall(r'"index":\s*(\d+)', prompt)
            base = 0
            texts = re.findall(r'"text":\s*"raw (\d+)"', prompt)
            if texts:
                base = int(texts[0])
            return json.dumps([
                {"index": i, "text": f"Polished {base + i}."} for i in range(n)
            ])
        finally:
            self._inflight -= 1


def _run(segs, orch, monkeypatch, batch=4, conc=3, budget=0.0,
         model_override=None):
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_AUTO_GLOSSARY", False, raising=False)
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_BATCH_SIZE", batch, raising=False)
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_CONCURRENCY", conc, raising=False)
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_MAX_S", budget, raising=False)
    return asyncio.run(P.correct_transcript(
        segs, orch, language="en", mode="translation",
        model_override=model_override))


def test_all_batches_polished_in_order(monkeypatch):
    orch = _Orch()
    out = _run(_segs(20), orch, monkeypatch, batch=4, conc=3)
    texts = [P._coerce_segment(s)["text"] for s in out]
    assert texts == [f"Polished {i}." for i in range(20)]


def test_batches_actually_run_concurrently(monkeypatch):
    orch = _Orch(delay_s=0.05)
    _run(_segs(40), orch, monkeypatch, batch=4, conc=3)
    # Batch 0 runs solo; the remaining 9 batches must overlap.
    assert orch.max_inflight >= 2


def test_budget_exhaustion_keeps_drafts_for_tail(monkeypatch):
    # Budget so small that only the solo first batch (plus maybe one more)
    # fits; later cues must keep their draft text, never block forever.
    orch = _Orch(delay_s=0.2)
    out = _run(_segs(40), orch, monkeypatch, batch=4, conc=1, budget=0.3)
    texts = [P._coerce_segment(s)["text"] for s in out]
    assert texts[:4] == [f"Polished {i}." for i in range(4)]   # first batch ran
    assert any(t.startswith("raw ") for t in texts[4:])        # tail kept drafts


def test_downshift_when_upgraded_model_projects_over_budget(monkeypatch):
    # Auto-upgrade hands back a "big" model; the first batch is slow enough
    # that the projection blows the budget → remaining batches must run on
    # the BASE model instead.
    async def fake_upgrade(base):
        return "qwen2.5:14b"
    import backend.services.translator as T
    monkeypatch.setattr(T, "resolve_translation_polish_model", fake_upgrade)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_POLISH_AUTO", True, raising=False)

    orch = _Orch(delay_s=0.25)
    out = _run(_segs(40), orch, monkeypatch, batch=4, conc=1, budget=1.0,
               model_override="qwen3:4b-instruct-2507-q4_K_M")
    assert orch.calls[0] == "qwen2.5:14b"                      # probe batch
    assert "qwen3:4b" in (orch.calls[1] or "")                 # downshifted
    # Downshifted batches still polish (coverage preserved).
    texts = [P._coerce_segment(s)["text"] for s in out]
    assert texts[4].startswith("Polished")


def test_consecutive_failures_trip_the_breaker(monkeypatch):
    orch = _Orch(fail_all=True)
    out = _run(_segs(80), orch, monkeypatch, batch=4, conc=1)
    # Each failed batch costs 3 calls (main + halve-and-retry pair); the
    # breaker trips after 5 consecutive failed batches → ≤18 calls instead
    # of the 60 that running all 20 batches would burn.
    assert len(orch.calls) <= 18
    assert len(out) == 80                                       # drafts kept


def _run_downshift_base_model(monkeypatch, orch):
    return _run(_segs(40), orch, monkeypatch, batch=4, conc=1, budget=1.0)


def test_config_defaults():
    assert settings.SUBTITLE_POLISH_CONCURRENCY >= 1
    assert settings.SUBTITLE_POLISH_MAX_S > 0


class _FailFullSucceedHalf(_Orch):
    """Fails any batch larger than 2; succeeds on halves."""

    async def text_completion(self, prompt, timeout=90.0, model_override=None, **kw):
        import re as _re, json as _json
        self.calls.append(model_override)
        m = _re.search(r"JSON array of (\d+) objects", prompt)
        n = int(m.group(1)) if m else 1
        if n > 2:
            return "garbled not json"
        texts = _re.findall(r'"text":\s*"raw (\d+)"', prompt)
        base = int(texts[0]) if texts else 0
        return _json.dumps([{"index": i, "text": f"Polished {base + i}."}
                            for i in range(n)])


def test_halve_and_retry_recovers_failed_batches(monkeypatch):
    orch = _FailFullSucceedHalf()
    out = _run(_segs(8), orch, monkeypatch, batch=4, conc=1)
    texts = [P._coerce_segment(s)["text"] for s in out]
    # Full batches of 4 fail; the two halves of 2 succeed → all polished.
    assert texts == [f"Polished {i}." for i in range(8)]
