"""P2 — per-segment salvage in the transcript polisher.

A single malformed element in a batch used to revert the ENTIRE batch to raw.
Now the good lines are polished and only the failing index keeps its original
text + words.
"""

import asyncio
import json

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services import transcript_polisher as P


def _seg(start, end, text, words=None):
    return TranscriptSegment(start=start, end=end, text=text,
                             speaker="Speaker 1", words=words)


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


class _OneBadElementOrch:
    """Returns a valid-length array where exactly one element is malformed
    (a leaked input object) — the rest are clean polished strings."""

    async def text_completion(self, prompt, timeout=90.0, **kw):
        import re
        m = re.search(r"return EXACTLY (\d+)", prompt)
        n = int(m.group(1)) if m else 1
        arr = []
        for i in range(n):
            if i == 1:
                # Malformed: stringified input object → must be rejected.
                arr.append("{'index': 1, 'text': 'leaked'}")
            else:
                arr.append(f"Polished line {i}.")
        return json.dumps(arr)


def test_one_bad_index_keeps_original_rest_polished(monkeypatch):
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_BATCH_SIZE", 10, raising=False)

    segs = [
        _seg(0.0, 1.0, "raw line zero",
             words=_words([(0.0, 0.5, "raw"), (0.5, 0.8, "line"), (0.8, 1.0, "zero")])),
        _seg(1.0, 2.0, "raw line one",
             words=_words([(1.0, 1.4, "raw"), (1.4, 1.7, "line"), (1.7, 2.0, "one")])),
        _seg(2.0, 3.0, "raw line two",
             words=_words([(2.0, 2.4, "raw"), (2.4, 2.7, "line"), (2.7, 3.0, "two")])),
    ]
    out = asyncio.run(P.correct_transcript(
        list(segs), _OneBadElementOrch(), language="en", mode="asr"))

    assert len(out) == 3
    # Good indices polished.
    assert out[0].text == "Polished line 0."
    assert out[2].text == "Polished line 2."
    # Bad index (1) kept its ORIGINAL text AND words verbatim.
    assert out[1].text == "raw line one"
    assert out[1].words is not None and len(out[1].words) == 3
    assert out[1].words[0].start == 1.0
    # No leaked dict reached any output.
    assert all("index" not in (s.text or "") and "{" not in (s.text or "") for s in out)
    # Timing preserved on all.
    for o, s in zip(out, segs):
        assert o.start == s.start and o.end == s.end


def test_whole_batch_failure_still_keeps_all_raw(monkeypatch):
    monkeypatch.setattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(settings, "CUSTOM_VOCABULARY_ENABLED", False, raising=False)

    class _GarbageOrch:
        async def text_completion(self, prompt, timeout=90.0, **kw):
            return "not json at all"

    segs = [_seg(0.0, 1.0, "one"), _seg(1.0, 2.0, "two")]
    out = asyncio.run(P.correct_transcript(
        list(segs), _GarbageOrch(), language="en", mode="translation"))
    assert [s.text for s in out] == ["one", "two"]
