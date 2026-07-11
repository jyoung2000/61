"""Cue-segmentation + translation-coherence fixes from the 128-min run.

The exported transcript showed (a) two speakers' lines welded into single
cues, (b) fragmentary mid-utterance cues translated as finished sentences,
and (c) incoherent isolated-cue ja→en output. These pin the fixes:

  * terminator-carrying cues are additionally split at TURN-length word-gaps
    (machine-appended '。' no longer disarms the splitter);
  * the punctuation restorer withholds its CJK terminator when the next cue
    continues within the turn gap (mid-utterance fragments stay joinable);
  * same-speaker merges never bridge a turn-length silence, and a merged
    block only carries words when BOTH sides have them (partial word arrays
    can no longer erase a neighbour's text);
  * FuguMT/Opus-MT gains guarded per-cue contextual translation (prior
    source cues resolve dropped Japanese subjects/pronouns; tag-loss falls
    back to the isolated translation).
"""

import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.sentence_segmenter import (
    TranscriptSegment, resegment_by_sentence, _split_segment_by_sentence)
from backend.services import transcript_polisher as P


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


# ─────────────────────────────────────────────────────────────────────────────
# Turn-pause re-split of terminator-carrying cues
# ─────────────────────────────────────────────────────────────────────────────

def test_terminated_cue_splits_at_turn_pause():
    # One cue, terminator at the very end (machine-appended style), with a
    # 1.0 s silence in the middle — two utterances welded together.
    words = [_w("What?", 0.0, 0.4), _w("Congrats!", 0.5, 1.0),
             _w("Thanks.", 2.0, 2.5)]                 # 1.0 s gap before this
    seg = TranscriptSegment(text="What? Congrats! Thanks.",
                            start=0.0, end=2.5, speaker="S", words=words)
    out = _split_segment_by_sentence(seg)
    # "What?"/"Congrats!" split at their own terminators; the turn pause
    # separates "Thanks." even though the cue as a whole "had punctuation".
    assert len(out) >= 2
    assert any(o.text.strip().startswith("Thanks") for o in out)
    thanks = next(o for o in out if o.text.strip().startswith("Thanks"))
    assert thanks.start == pytest.approx(2.0)


def test_short_pause_does_not_split_terminated_cue():
    words = [_w("こんにちは", 0.0, 0.8), _w("みなさん。", 1.0, 1.6)]  # 0.2 s gap
    seg = TranscriptSegment(text="こんにちは みなさん。",
                            start=0.0, end=1.6, speaker="S", words=words)
    out = _split_segment_by_sentence(seg)
    assert len(out) == 1                              # ordinary rhythm kept


# ─────────────────────────────────────────────────────────────────────────────
# Merge guards
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_never_bridges_turn_silence():
    segs = [
        TranscriptSegment(text="A.", start=0.0, end=1.0, speaker="S", words=None),
        TranscriptSegment(text="B.", start=5.0, end=6.0, speaker="S", words=None),
    ]
    out = resegment_by_sentence(segs)
    assert len(out) == 2
    assert out[1].start == 5.0                        # timing intact


def test_partial_words_never_erase_neighbor_text():
    # First cue has words; second (post-polish, low remap confidence) has
    # none. The merged block must NOT rebuild text from the partial words.
    words = [_w("Hello", 0.0, 0.5), _w("there.", 0.5, 1.0)]
    segs = [
        TranscriptSegment(text="Hello there.", start=0.0, end=1.0,
                          speaker="S", words=words),
        TranscriptSegment(text="Long important continuation words here.",
                          start=1.1, end=2.5, speaker="S", words=None),
    ]
    out = resegment_by_sentence(segs)
    joined = " ".join(o.text for o in out)
    assert "continuation" in joined                   # text survived


# ─────────────────────────────────────────────────────────────────────────────
# Neighbour-aware CJK terminator restore
# ─────────────────────────────────────────────────────────────────────────────

def _cue(text, start, end, speaker="S"):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def test_mid_utterance_fragment_keeps_no_terminator(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "PUNCTUATION_RESTORE_FALLBACK_ENABLED",
                        True, raising=False)
    segs = [
        _cue("これはとても長い文章の", 0.0, 2.0),      # continues 0.2 s later
        _cue("続きです", 2.2, 3.5),                    # utterance ends (last)
    ]
    out = P.restore_punctuation_fallback(segs, language="japanese")
    assert not out[0]["text"].endswith("。")           # fragment left open
    assert out[1]["text"].endswith("。")               # real end terminated


def test_turn_gap_gets_terminator(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "PUNCTUATION_RESTORE_FALLBACK_ENABLED",
                        True, raising=False)
    segs = [
        _cue("わかりました", 0.0, 2.0),
        _cue("次の話", 4.0, 5.0),                      # 2 s gap — turn over
    ]
    out = P.restore_punctuation_fallback(segs, language="japanese")
    assert out[0]["text"].endswith("。")


def test_speaker_change_gets_terminator(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "PUNCTUATION_RESTORE_FALLBACK_ENABLED",
                        True, raising=False)
    segs = [
        _cue("そうですね", 0.0, 2.0, speaker="A"),
        _cue("はい", 2.1, 2.5, speaker="B"),           # different voice
    ]
    out = P.restore_punctuation_fallback(segs, language="japanese")
    assert out[0]["text"].endswith("。")


# ─────────────────────────────────────────────────────────────────────────────
# OpusMT/FuguMT contextual translation (guarded)
# ─────────────────────────────────────────────────────────────────────────────

def test_opusmt_context_used_when_tag_survives():
    from backend.services.nmt_translator import OpusMTTranslator

    eng = OpusMTTranslator.__new__(OpusMTTranslator)
    calls = []

    def fake_batch(texts, glossary=None):
        calls.append(texts[0])
        # Echo the tag back — simulating tag survival with context payoff.
        if "⟦1⟧" in texts[0]:
            return ["context junk ⟦1⟧She ate the cake.⟦/1⟧"]
        return ["It ate the cake."]                    # isolated = wrong subject
    eng.translate_batch = fake_batch

    out = eng.translate_with_context(
        ["ケーキを食べた"], ["ミカは"], [], "ja", "en")
    assert out == ["She ate the cake."]
    assert "⟦1⟧" in calls[0]                           # context path attempted


def test_opusmt_falls_back_when_tag_lost():
    from backend.services.nmt_translator import OpusMTTranslator

    eng = OpusMTTranslator.__new__(OpusMTTranslator)

    def fake_batch(texts, glossary=None):
        if "⟦1⟧" in texts[0]:
            return ["mangled output with no tags"]
        return ["Isolated translation."]
    eng.translate_batch = fake_batch

    out = eng.translate_with_context(
        ["ケーキを食べた"], ["ミカは"], [], "ja", "en")
    assert out == ["Isolated translation."]            # never worse than today


def test_opusmt_no_context_goes_straight_isolated():
    from backend.services.nmt_translator import OpusMTTranslator

    eng = OpusMTTranslator.__new__(OpusMTTranslator)
    calls = []

    def fake_batch(texts, glossary=None):
        calls.append(texts[0])
        return ["First line."]
    eng.translate_batch = fake_batch

    out = eng.translate_with_context(["最初の行"], [], [], "ja", "en")
    assert out == ["First line."]
    assert all("⟦" not in c for c in calls)            # no pointless tagging
