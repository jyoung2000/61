"""P2 — deterministic, non-LLM punctuation-restore fallback.

When the editorial LLM is unavailable / leaves a segment with no terminator, a
lightweight restorer adds sentence punctuation so the resegmenter still has
boundaries to split on. The Latin restorer is optional + lazily imported; CJK
uses a rule-based terminator that needs no dependency.
"""

import asyncio

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services import transcript_polisher as P


def _seg(start, end, text, words=None, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text,
                             speaker=speaker, words=words)


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def test_cjk_gets_terminator_without_dependency():
    segs = [_seg(0.0, 2.0, "今日はいい天気")]
    out = P.restore_punctuation_fallback(segs, language="ja")
    assert out[0].text == "今日はいい天気。"


def test_already_terminated_cjk_untouched():
    segs = [_seg(0.0, 2.0, "今日はいい天気。")]
    out = P.restore_punctuation_fallback(segs, language="ja")
    assert out[0].text == "今日はいい天気。"


def test_flag_off_no_change(monkeypatch):
    monkeypatch.setattr(settings, "PUNCTUATION_RESTORE_FALLBACK_ENABLED", False)
    segs = [_seg(0.0, 2.0, "今日はいい天気")]
    out = P.restore_punctuation_fallback(segs, language="ja")
    assert out[0].text == "今日はいい天気"


def test_latin_uses_model_when_available(monkeypatch):
    # Simulate the optional dependency being present via a fake model.
    class _FakeModel:
        def restore_punctuation(self, text):
            return "Hello there everyone. Welcome back."

    monkeypatch.setattr(P, "_PUNCT_MODEL", _FakeModel(), raising=False)
    monkeypatch.setattr(P, "_PUNCT_MODEL_UNAVAILABLE", False, raising=False)
    segs = [_seg(0.0, 3.0, "hello there everyone welcome back",
                 words=_words([(0.0, 0.5, "hello"), (0.6, 1.0, "there"),
                               (1.1, 1.6, "everyone"), (2.0, 2.4, "welcome"),
                               (2.5, 3.0, "back")]))]
    out = P.restore_punctuation_fallback(segs, language="en")
    assert out[0].text == "Hello there everyone. Welcome back."
    # Timing preserved, word timing re-mapped (not nulled).
    assert out[0].start == 0.0 and out[0].end == 3.0
    assert out[0].words


def test_latin_skips_gracefully_when_dependency_missing(monkeypatch):
    # Force the "unavailable" state — Latin text must be left unchanged, no crash.
    monkeypatch.setattr(P, "_PUNCT_MODEL", None, raising=False)
    monkeypatch.setattr(P, "_PUNCT_MODEL_UNAVAILABLE", True, raising=False)
    segs = [_seg(0.0, 3.0, "hello there everyone")]
    out = P.restore_punctuation_fallback(segs, language="en")
    assert out[0].text == "hello there everyone"


def test_correct_transcript_no_orchestrator_restores_cjk():
    # LLM disabled (orchestrator=None): CJK still gets a terminator so the
    # resegmenter has something to split on.
    segs = [_seg(0.0, 2.0, "これはテストです", speaker="Speaker 1")]
    out = asyncio.run(P.correct_transcript(list(segs), orchestrator=None, language="ja"))
    assert out[0].text.endswith("。")


def test_ends_with_terminator_ignores_closers():
    assert P._ends_with_terminator('He said "hello."')
    assert P._ends_with_terminator("終わり。」")
    assert not P._ends_with_terminator("no terminator here")
