"""Forced alignment of the SHIPPED (translated) subtitle cues.

``refine_word_timestamps`` was only ever called on the source-language ASR
inside the reframer. The English cues that actually become subtitles never
reached an aligner, so roughly half of them carried word times distributed
across the cue window by character width. That one gap costs three things:

  * per-word highlighting advances by text length instead of speech;
  * the readability splitter refuses to cut a cue it has no real word time
    for, so over-long cues ship whole;
  * a cue's start cannot be tightened onto its first voiced word, which is
    why a cue appears before the speech it captions.

``align_translated_cues`` is the entry point that closes it. These tests pin
the contract that holds with no aligner backend present (this suite has no
torch), plus the shape adapters — the alignment maths itself needs audio and a
model, and is verified on a real run.
"""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.forced_aligner import (
    align_translated_cues, _as_word_rows, _iter_word_starts,
)


def _cue(start, end, text, words=None):
    return TranscriptSegment(start=start, end=end, text=text,
                             speaker="Speaker 1", words=words)


def test_no_backend_leaves_every_timing_untouched():
    # Fail-soft is the whole contract: a box without torchaudio, or a missing
    # audio file, must ship exactly what the timing tiers produced.
    cues = [_cue(1.0, 2.0, "hello there", words=[
        WordTimestamp(start=1.0, end=1.5, word="hello"),
        WordTimestamp(start=1.5, end=2.0, word="there")])]
    stats = align_translated_cues("/nonexistent/audio.wav", cues)
    assert stats["cues_aligned"] == 0
    assert stats["words_aligned"] == 0
    assert stats["starts_tightened"] == 0
    assert cues[0].start == 1.0 and cues[0].end == 2.0
    assert [w.start for w in cues[0].words] == [1.0, 1.5]


def test_empty_and_missing_inputs_are_safe():
    assert align_translated_cues("", []) ["cues_aligned"] == 0
    assert align_translated_cues("/x.wav", []) ["cues_aligned"] == 0
    assert align_translated_cues("", [_cue(0.0, 1.0, "hi")])["cues_aligned"] == 0


def test_disabled_by_setting():
    from backend.config import settings
    prev = settings.SUBTITLE_FORCED_ALIGN
    try:
        settings.SUBTITLE_FORCED_ALIGN = False
        stats = align_translated_cues("/x.wav", [_cue(0.0, 1.0, "hi there")])
        assert stats["enabled"] is False
    finally:
        settings.SUBTITLE_FORCED_ALIGN = prev


def test_word_rows_keep_the_row_type_the_cue_already_uses():
    # ``TranscriptSegment.words`` is typed ``list[WordTimestamp]`` and pydantic
    # does NOT validate on attribute assignment, so writing bare dicts there
    # would leave the model holding rows that every ``w.end`` consumer crashes
    # on. Dict rows (the persisted shape) must stay dicts.
    rows = [{"word": "hi", "start": 0.1, "end": 0.2}]
    model_rows = _as_word_rows(_cue(0.0, 1.0, "hi"), rows)
    assert isinstance(model_rows[0], WordTimestamp)
    assert model_rows[0].start == 0.1
    dict_rows = _as_word_rows({"text": "hi", "start": 0.0, "end": 1.0}, rows)
    assert isinstance(dict_rows[0], dict)


def test_iter_word_starts_reads_both_row_shapes():
    seg = _cue(0.0, 1.0, "a b", words=[
        WordTimestamp(start=0.0, end=0.4, word="a"),
        WordTimestamp(start=0.5, end=1.0, word="b")])
    assert _iter_word_starts(seg) == [0.0, 0.5]
    assert _iter_word_starts(
        {"words": [{"start": 2.0}, {"start": 3.0}]}) == [2.0, 3.0]
    assert _iter_word_starts({"words": None}) == []
    assert _iter_word_starts(_cue(0.0, 1.0, "x")) == []


def test_cue_shift_bound_is_configured_and_bounded():
    from backend.config import settings
    # The bound is what keeps a mis-anchored cue from wandering away from the
    # window the timing tiers established.
    v = float(getattr(settings, "SUBTITLE_ALIGN_MAX_CUE_SHIFT_S", 0.0))
    assert 0.0 < v <= 2.0


# ── End extension to the voiced extent ─────────────────────────────────────
# Tier B/C placement squeezes some cue windows well under their real voiced
# span; the guillotined end is what makes a subtitle vanish while its line is
# still being spoken, and the fake over-CPS reading then drives the splitter.
# When the aligner hears the last word running past the cue's end, the end
# moves to the voiced extent — bounded by the padded listening window and the
# next cue's start, so only idle time is ever borrowed.

def _fake_backend(spans):
    class _B:
        sample_rate = 16000
        def align_words(self, wav, tokens):
            return spans[:len(tokens)]
    return _B()


def _run_aligned(cues, spans, monkeypatch, tmp_path):
    import numpy as np
    import types, sys
    from backend.services import forced_aligner as FA
    # A fake torchaudio: load() returns 1 channel of silence at 16 kHz.
    fake_ta = types.ModuleType("torchaudio")
    class _T:
        def __init__(self, arr): self._a = arr
        @property
        def shape(self): return (1, len(self._a))
        def __getitem__(self, key): return self
        def mean(self, dim, keepdim): return self
    fake_ta.load = lambda p: (_T(np.zeros(16000 * 60)), 16000)
    fake_ta.functional = types.SimpleNamespace(
        resample=lambda w, a, b: w)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)
    monkeypatch.setattr(FA, "_pick_device", lambda: "cpu")
    monkeypatch.setattr(FA, "_get_backend", lambda lang, dev: _fake_backend(spans))
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    return FA.align_translated_cues(str(wav), cues)


def test_end_extends_to_the_voiced_extent(monkeypatch, tmp_path):
    # Cue window 10.0-11.0 but the aligner hears "world" ending at ~11.5
    # (window-relative: s0 = 10.0 - 0.3 pad = 9.7; 11.5 - 9.7 = 1.8).
    cues = [_cue(10.0, 11.0, "hello world"),
            _cue(13.0, 14.0, "next cue here")]
    stats = _run_aligned(cues, [(0.4, 0.9), (1.0, 1.8)], monkeypatch, tmp_path)
    assert stats["ends_extended"] >= 1
    # Extended toward the voiced extent, but never past window pad or the
    # next cue.
    assert 11.0 < cues[0].end <= 11.3 + 0.001
    assert cues[0].words[-1].end <= cues[0].end + 0.001


def test_end_extension_never_reaches_the_next_cue(monkeypatch, tmp_path):
    cues = [_cue(10.0, 11.0, "hello world"),
            _cue(11.05, 12.0, "tight next")]
    stats = _run_aligned(cues, [(0.4, 0.9), (1.0, 1.8)], monkeypatch, tmp_path)
    # Only ~0 room before the next cue: no meaningful extension.
    assert cues[0].end <= 11.05 - 0.084 + 0.02 or cues[0].end == 11.0
