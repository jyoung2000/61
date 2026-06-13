"""Pipeline orchestration: caching, engine batching, re-render, cancellation.

Uses a fake engine + fake model manager so the synthesis path is exercised
without any real model or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from inflect.config import Config
from inflect.document.spans import Document, Inflection
from inflect.ingest.profile import VoiceLibrary
from inflect.synth.pipeline import (
    PipelineCancelled,
    PipelineError,
    SynthesisPipeline,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEngine:
    def __init__(self, name: str, sr: int) -> None:
        self.name = name
        self.sample_rate = sr
        self.calls = 0

    def synthesize(self, request) -> np.ndarray:
        self.calls += 1
        n = max(2400, len(request.job.text) * 200)  # deterministic length
        t = np.arange(n) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


class FakeModelManager:
    def __init__(self) -> None:
        self.engines = {
            "chatterbox": FakeEngine("chatterbox", 24_000),
            "indextts2": FakeEngine("indextts2", 22_050),
            "fish": FakeEngine("fish", 44_100),
        }
        self.current: str | None = None
        self.load_events: list[str] = []

    def get_engine(self, name: str, device: str | None = None):
        if self.current != name:
            self.load_events.append(name)
            self.current = name
        return self.engines[name]

    def unload_current(self) -> None:
        self.current = None


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def env(tmp_path):
    cfg = Config.load(home=tmp_path)
    cfg.paths.ensure()
    lib = VoiceLibrary(cfg.paths.voices)
    # A real reference wav on disk so _resolve_speaker passes.
    ref = tmp_path / "ref.wav"
    sf.write(ref, (0.1 * np.sin(np.arange(24000) * 0.1)).astype(np.float32), 24000)
    profile = lib.add("Tester", ref)
    mm = FakeModelManager()
    pipe = SynthesisPipeline(cfg, mm, lib)
    return cfg, lib, mm, pipe, profile.id


def _doc(text: str, profile_id: str) -> Document:
    return Document(text=text, voice_profile_id=profile_id)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_basic_render(env):
    _, _, mm, pipe, pid = env
    # A span creates a second segment (short unstyled text is one segment).
    doc = _doc("Hello there. General Kenobi.", pid)
    doc.apply_inflection(0, 12, Inflection(emo_text="greeting"))
    result = pipe.render(doc, engine="chatterbox")
    assert result.audio.size > 0
    assert result.sample_rate == 24_000
    assert len(result.segments) == 2
    assert mm.load_events == ["chatterbox"]


def test_short_unstyled_text_is_one_segment(env):
    _, _, _, pipe, pid = env
    doc = _doc("Hello there. General Kenobi.", pid)  # no spans, < 400 chars
    result = pipe.render(doc, engine="chatterbox")
    assert len(result.segments) == 1


def test_caching_skips_resynthesis(env):
    _, _, mm, pipe, pid = env
    doc = _doc("Hello there. General Kenobi.", pid)
    pipe.render(doc, engine="chatterbox")
    calls_after_first = mm.engines["chatterbox"].calls
    # Second render: everything cached -> no engine loads, no new synth calls.
    result2 = pipe.render(doc, engine="chatterbox")
    assert mm.engines["chatterbox"].calls == calls_after_first
    assert mm.load_events == ["chatterbox"]  # not loaded again
    assert all(seg.from_cache for seg in result2.segments)


def test_editing_one_span_rerenders_only_that_segment(env):
    _, _, mm, pipe, pid = env
    # Two phrases already styled as their own segments.
    doc = _doc("First part here. Second part here.", pid)
    doc.apply_inflection(0, 16, Inflection(emo_text="a"))
    doc.apply_inflection(17, len(doc.text), Inflection(emo_text="b"))
    pipe.render(doc, engine="chatterbox")
    base_calls = mm.engines["chatterbox"].calls
    assert base_calls == 2
    # Tweak only the FIRST span's emotion -> only its hash changes.
    doc.apply_inflection(0, 16, Inflection(emotion_vector=[0, 0.9] + [0] * 6))
    result = pipe.render(doc, engine="chatterbox")
    assert mm.engines["chatterbox"].calls == base_calls + 1
    cached = [s for s in result.segments if s.from_cache]
    fresh = [s for s in result.segments if not s.from_cache]
    assert len(fresh) == 1
    assert len(cached) == 1


def test_engine_batching_minimizes_swaps(env):
    _, _, mm, pipe, pid = env
    # Alternate engines across spans: chatterbox / indextts2 / chatterbox / indextts2
    doc = _doc("Aaa one. Bbb two. Ccc three. Ddd four.", pid)
    doc.apply_inflection(0, 8, Inflection(engine="chatterbox"))
    doc.apply_inflection(9, 17, Inflection(engine="indextts2"))
    doc.apply_inflection(18, 28, Inflection(engine="chatterbox"))
    doc.apply_inflection(29, 38, Inflection(engine="indextts2"))
    pipe.render(doc, engine="chatterbox")
    # Each engine loaded exactly once despite interleaving -> 2 swaps total.
    assert mm.load_events.count("chatterbox") == 1
    assert mm.load_events.count("indextts2") == 1
    assert len(mm.load_events) == 2


def test_mixed_engine_canonical_rate_prefers_indextts2(env):
    _, _, _, pipe, pid = env
    doc = _doc("Draft bit. Final bit.", pid)
    doc.apply_inflection(0, 10, Inflection(engine="chatterbox"))
    doc.apply_inflection(11, 21, Inflection(engine="indextts2"))
    result = pipe.render(doc, engine="chatterbox")
    assert result.sample_rate == 22_050  # IndexTTS-2 native rate wins


def test_cancellation_between_segments(env):
    _, _, mm, pipe, pid = env
    # Two spans -> two segments, so cancel can fire between them.
    doc = _doc("One two three. Four five six.", pid)
    doc.apply_inflection(0, 14, Inflection(emo_text="x"))
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] >= 2  # allow seg 0, cancel before seg 1

    with pytest.raises(PipelineCancelled):
        pipe.render(doc, engine="chatterbox", should_cancel=cancel)
    # The first segment did get rendered before cancellation.
    assert mm.engines["chatterbox"].calls == 1


def test_missing_voice_profile_raises(env):
    _, _, _, pipe, _ = env
    doc = Document(text="No voice set.", voice_profile_id=None)
    with pytest.raises(PipelineError):
        pipe.render(doc, engine="chatterbox")


def test_segment_offsets_account_for_pauses(env):
    _, _, _, pipe, pid = env
    doc = _doc("First. Second.", pid)
    doc.apply_inflection(0, 6, Inflection(pause_after_ms=500))
    result = pipe.render(doc, engine="chatterbox")
    offsets = result.segment_offsets()
    assert len(offsets) == 2
    # There is a 0.5 s gap between segment 0's end and segment 1's start.
    _, _, end0 = offsets[0]
    _, start1, _ = offsets[1]
    assert start1 - end0 == pytest.approx(0.5, abs=0.02)


def test_render_one_uses_cache(env):
    from inflect.document.segmenter import segment_document

    _, _, mm, pipe, pid = env
    doc = _doc("Just one sentence here.", pid)
    job = segment_document(doc, "chatterbox")[0]
    seg1 = pipe.render_one(doc, job)
    assert not seg1.from_cache
    seg2 = pipe.render_one(doc, job)
    assert seg2.from_cache
    # force=True re-renders even though cached.
    seg3 = pipe.render_one(doc, job, force=True)
    assert not seg3.from_cache


# --------------------------------------------------------------------------- #
# Phase 6: Fish direct + hybrid Performance Transfer
# --------------------------------------------------------------------------- #
def test_fish_tag_change_invalidates_cache(env):
    _, _, mm, pipe, pid = env
    doc = _doc("A whole fish sentence.", pid)
    doc.apply_inflection(0, len(doc.text), Inflection(engine="fish"))
    pipe.render(doc, engine="indextts2")
    base = mm.engines["fish"].calls
    # Change emotion -> tag string changes -> fish segment re-renders.
    doc.apply_inflection(0, len(doc.text), Inflection(emotion_vector=[0, 0.9] + [0] * 6, engine="fish"))
    pipe.render(doc, engine="indextts2")
    assert mm.engines["fish"].calls == base + 1


def test_hybrid_two_stage_and_swap_budget(env):
    _, _, mm, pipe, pid = env
    doc = _doc("Calm narration here. PERFORM THIS LINE. More calm narration.", pid)
    # A hybrid span in the middle; the rest is plain (toolbar engine).
    doc.apply_inflection(20, 38, Inflection(emotion_vector=[0, 0.9] + [0] * 6, engine="hybrid"))
    result = pipe.render(doc, engine="chatterbox")
    # Fish renders stage 1, IndexTTS-2 stage 2, Chatterbox the plain parts.
    assert mm.engines["fish"].calls == 1
    assert mm.engines["indextts2"].calls == 1
    assert mm.engines["chatterbox"].calls >= 1
    # At most three swaps, in dependency order (fish before indextts2).
    assert mm.load_events == ["fish", "indextts2", "chatterbox"]
    # The hybrid segment's final audio comes from IndexTTS-2.
    hybrid_seg = next(s for s in result.segments if s.char_start == 20)
    assert hybrid_seg.engine == "indextts2"


def test_hybrid_writes_performance_wav(env):
    cfg, _, _, pipe, pid = env
    doc = _doc("PERFORM THIS.", pid)
    doc.apply_inflection(0, len(doc.text), Inflection(emo_text="furious", engine="hybrid"))
    pipe.render(doc, engine="indextts2")
    perf_files = list(cfg.paths.cache.glob("perf_*.wav"))
    assert len(perf_files) == 1


def test_pure_hybrid_doc_costs_two_swaps(env):
    _, _, mm, pipe, pid = env
    doc = _doc("One line here. Two line here.", pid)
    doc.apply_inflection(0, 14, Inflection(emo_text="excited", engine="hybrid"))
    doc.apply_inflection(15, len(doc.text), Inflection(emo_text="sad", engine="hybrid"))
    pipe.render(doc, engine="indextts2")
    # Exactly two model loads: all Fish stage-1, then all IndexTTS-2 stage-2.
    assert mm.load_events == ["fish", "indextts2"]
    assert mm.engines["fish"].calls == 2
    assert mm.engines["indextts2"].calls == 2


def test_render_performance_stage1_only(env):
    from inflect.document.segmenter import SegmentJob

    _, _, mm, pipe, pid = env
    doc = _doc("Audition this performance.", pid)
    job = SegmentJob(
        seg_id=0,
        text="Audition this performance.",
        inflection=Inflection(emo_text="furious"),
        voice_profile_id=pid,
        engine="hybrid",
        char_start=0,
        char_end=26,
    )
    seg = pipe.render_performance(doc, job)
    assert seg.engine == "fish"
    assert mm.engines["fish"].calls == 1
    assert mm.engines["indextts2"].calls == 0  # stage 2 not run
    assert seg.audio.size > 0


def test_hybrid_caching_no_reload(env):
    _, _, mm, pipe, pid = env
    doc = _doc("Perform this line.", pid)
    doc.apply_inflection(0, len(doc.text), Inflection(emo_text="furious", engine="hybrid"))
    pipe.render(doc, engine="indextts2")
    mm.load_events.clear()
    fish_calls = mm.engines["fish"].calls
    idx_calls = mm.engines["indextts2"].calls
    pipe.render(doc, engine="indextts2")  # fully cached now
    assert mm.load_events == []
    assert mm.engines["fish"].calls == fish_calls
    assert mm.engines["indextts2"].calls == idx_calls
