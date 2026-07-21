"""A Latin-dominant cue with a few stranded CJK glyphs must be re-translated,
not shipped raw. Run after run: "From Nagランチポイント AX, we're confirming…"
shipped with the katakana intact because the pipeline's per-cue cleanup net
selected only fully-CJK (_is_untranslated) and word-salad cues, missing the
_cjk_remnant_reason net that exists for exactly this case.
"""

import asyncio

import pytest

from backend.services.translator import _cjk_remnant_reason, _is_untranslated


SHIPPED = "From Nagランチポイント AX, we're confirming mass displacement toward GY."


def test_the_shipped_cue_slips_past_is_untranslated_but_cjkr_catches_it():
    # This is the root cause: cjk_ratio ~0.13 < 0.30, so _is_untranslated says
    # "translated"; only _cjk_remnant_reason flags the stranded katakana.
    assert not _is_untranslated(SHIPPED, "ja")
    assert _cjk_remnant_reason(SHIPPED) == "cjk-remnant"
    # And it's zero-false-positive on clean English (no CJK at all).
    assert _cjk_remnant_reason("From La Grange Point AX, confirming movement.") is None


def test_cleanup_selects_and_retranslates_the_katakana_remnant(monkeypatch):
    from backend.services import pipeline as pl

    # Isolate the per-cue path (skip the batch pre-pass) for a deterministic test.
    monkeypatch.setattr(pl.settings, "TRANSLATION_LLM_CLEANUP_BATCH", False,
                        raising=False)

    calls = []

    class _Fake:
        async def text_completion(self, prompt, **kw):
            calls.append(prompt)
            # The recovered, fully-English line.
            return "From La Grange Point AX, we're confirming mass movement toward GY."

    segs = [
        {"start": 0.0, "end": 3.0, "text": "Inform Zechs in the atmosphere."},
        {"start": 3.0, "end": 6.0, "text": SHIPPED},              # katakana remnant
        {"start": 6.0, "end": 9.0, "text": "Yes, radar indicates five contacts."},
    ]
    out = asyncio.run(pl._llm_cleanup_untranslated(
        segs, source_lang="ja", target_lang="en",
        orchestrator=_Fake(), glossary=None, job_id="t-cjkr"))

    texts = [s["text"] if isinstance(s, dict) else s.text for s in out]
    # The katakana is gone; the cue was re-translated.
    assert "ランチポイント" not in texts[1]
    assert "La Grange Point" in texts[1]
    # Exactly ONE cue was sent for recovery — the clean English cues were not
    # selected (no false positives).
    assert len(calls) == 1
    assert texts[0] == "Inform Zechs in the atmosphere."
    assert texts[2] == "Yes, radar indicates five contacts."
