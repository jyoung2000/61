"""Unit tests for the offline Whisper-native audio→English translate helper
(``backend.services.pipeline._whisper_native_translate_segments``).

Importing the pipeline pulls in the AI orchestrator (optional provider SDKs)
and, lazily, the reframer audio module (OpenCV) + the NMT translator. None of
those are needed for this pure helper, so we stub the optional provider SDKs at
import time and inject lightweight fakes for the lazily-imported modules per
test (via ``monkeypatch.setitem`` so they're auto-restored — no leakage). No
network, no models, no GPU.
"""
import sys
import types

import pytest


def _stub_provider_sdks():
    for name in ["google", "google.generativeai", "groq", "openai", "anthropic"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["google"].generativeai = sys.modules["google.generativeai"]
    g = sys.modules["google.generativeai"]
    g.configure = lambda *a, **k: None
    g.GenerativeModel = object
    sys.modules["groq"].AsyncGroq = sys.modules["groq"].Groq = object
    sys.modules["openai"].AsyncOpenAI = sys.modules["openai"].OpenAI = object
    sys.modules["anthropic"].AsyncAnthropic = sys.modules["anthropic"].Anthropic = object


_stub_provider_sdks()

import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import backend.services.pipeline as pipeline  # noqa: E402
import backend.services.translator as translator  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402


# ── Fakes for the modules the helper imports lazily ──────────────────────────

def _real_apply_glossary(source_text, translated_text, glossary):
    """Copy of nmt_translator.apply_glossary (kept in sync) so the stub behaves
    like production for the glossary assertion."""
    if not glossary or not translated_text:
        return translated_text
    out = translated_text
    for src, tgt in glossary.items():
        s, t = (src or "").strip(), (tgt or "").strip()
        if not s or not t or s not in source_text:
            continue
        if s in out:
            out = out.replace(s, t)
    return out


class _FakeAI:
    last_args = None
    _loads = True
    _segments = ()

    def __init__(self, model_name="small"):
        self.model_name = model_name

    def try_load(self):
        return _FakeAI._loads

    def whisper_translate(self, video_path, source_lang=None, reuse_loaded=False):
        _FakeAI.last_args = (video_path, source_lang)
        return list(_FakeAI._segments)


def _install(monkeypatch, *, loads=True, segments=()):
    _FakeAI._loads = loads
    _FakeAI._segments = segments
    _FakeAI.last_args = None
    fake_reframer = types.ModuleType("backend.services.reframer_audio")
    fake_reframer.AudioIntelligence = _FakeAI
    fake_nmt = types.ModuleType("backend.services.nmt_translator")
    fake_nmt.apply_glossary = _real_apply_glossary
    monkeypatch.setitem(sys.modules, "backend.services.reframer_audio", fake_reframer)
    monkeypatch.setitem(sys.modules, "backend.services.nmt_translator", fake_nmt)


def test_maps_whisper_dicts_to_transcript_segments(monkeypatch):
    """start_sec/end_sec/text dicts → TranscriptSegment with correct fields."""
    _install(monkeypatch, segments=[
        {"start_sec": 0.0, "end_sec": 1.5, "text": "  Hello there  ", "words": []},
        {"start_sec": 1.5, "end_sec": 3.0, "text": "second line", "words": []},
    ])
    out = pipeline._whisper_native_translate_segments("/v.mp4", "ja", None)
    assert len(out) == 2
    assert out[0].text == "Hello there"                  # stripped
    assert out[0].start == 0.0 and out[0].end == 1.5     # start_sec/end_sec mapped
    assert out[1].start == 1.5 and out[1].end == 3.0
    assert out[0].speaker == "Speaker 1"                 # default when no source given
    assert _FakeAI.last_args == ("/v.mp4", "ja")         # video path + source forwarded


def test_speaker_inherited_from_source_by_overlap(monkeypatch):
    """Each translated cue inherits the diarized source speaker it overlaps most."""
    _install(monkeypatch, segments=[
        {"start_sec": 0.0, "end_sec": 2.0, "text": "first", "words": []},
        {"start_sec": 2.0, "end_sec": 4.0, "text": "second", "words": []},
    ])
    source = [
        {"start": 0.0, "end": 2.1, "speaker": "Alice", "text": "x"},
        {"start": 2.1, "end": 4.0, "speaker": "Bob", "text": "y"},
    ]
    out = pipeline._whisper_native_translate_segments("/v.mp4", "ja", None, source)
    assert out[0].speaker == "Alice"
    assert out[1].speaker == "Bob"


def test_glossary_fixes_leaked_source_terms(monkeypatch):
    """A glossary source term that leaked into the English output is replaced."""
    _install(monkeypatch, segments=[
        {"start_sec": 0.0, "end_sec": 2.0, "text": "ゼクス reporting in", "words": []},
    ])
    out = pipeline._whisper_native_translate_segments("/v.mp4", "ja", {"ゼクス": "Zechs"})
    assert out[0].text == "Zechs reporting in"


def test_blank_lines_are_dropped(monkeypatch):
    _install(monkeypatch, segments=[
        {"start_sec": 0.0, "end_sec": 1.0, "text": "   ", "words": []},
        {"start_sec": 1.0, "end_sec": 2.0, "text": "kept", "words": []},
    ])
    out = pipeline._whisper_native_translate_segments("/v.mp4", "ja", None)
    assert [s.text for s in out] == ["kept"]


def test_returns_empty_when_engine_fails_to_load(monkeypatch):
    _install(monkeypatch, loads=False, segments=[
        {"start_sec": 0.0, "end_sec": 1.0, "text": "ignored", "words": []},
    ])
    assert pipeline._whisper_native_translate_segments("/v.mp4", "ja", None) == []


def test_returns_empty_when_whisper_yields_nothing(monkeypatch):
    _install(monkeypatch, segments=[])
    assert pipeline._whisper_native_translate_segments("/v.mp4", "ja", None) == []


# ── Pipeline integration: the _background_post_processing translate branch ───

def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class _FakeDB:
    def __init__(self, initial=None):
        self.fields = dict(initial or {})
        self.update_calls = []

    async def update_job_status(self, job_id, **kw):
        self.update_calls.append(kw)
        self.fields.update(kw)

    async def load_job(self, job_id):
        return SimpleNamespace(**self.fields)


def _fake_orchestrator():
    return SimpleNamespace(
        reset_circuit_breaker=lambda: None,
        get_editorial_model_info=lambda: {"is_thinking": False},
    )


def _src(text, start, end, speaker="Speaker 1"):
    return {"text": text, "start": start, "end": end, "speaker": speaker}


def _en(text):
    """Map a known Japanese source cue to fixed English (non-CJK) text so a mock
    translator's output passes the pipeline's final language-purity gate, which
    rejects source-script output (prefixing the Japanese source would not)."""
    return {"こんにちは": "Hello", "世界": "World"}.get(text, "line")


def _install_pipeline(monkeypatch, db, *, engine="nllb"):
    async def _bcast(job_id, message):
        return None
    monkeypatch.setattr(pipeline.database, "update_job_status", db.update_job_status)
    monkeypatch.setattr(pipeline.database, "load_job", db.load_job)
    monkeypatch.setattr(pipeline, "broadcast_ws", _bcast)
    monkeypatch.setattr(translator, "_resolve_translation_engine", lambda s, t: engine)
    # Keep the heavy optional passes out of the unit under test.
    for flag in ("AI_TRANSCRIPT_CORRECTION", "SENTENCE_SEGMENTATION_ENABLED",
                 "SUBTITLE_CPS_ENFORCEMENT", "TRANSLATION_GLOSSARY_ENABLED",
                 "TRANSCRIPT_POLISHING_ENABLED"):
        monkeypatch.setattr(pipeline.settings, flag, False, raising=False)
    monkeypatch.setattr(pipeline.settings, "WHISPER_TRANSLATE_TO_EN", True, raising=False)
    # Pretend a capable GPU is free by default so the Whisper-native gate opens;
    # individual tests override this to exercise the low-VRAM skip.
    monkeypatch.setattr(pipeline, "_gpu_free_vram_gb", lambda: 8.0)


def test_pipeline_uses_whisper_native_for_ja_en_and_skips_nmt(monkeypatch):
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)

    nmt_calls = {"n": 0}
    async def _nmt(*a, **k):
        nmt_calls["n"] += 1
        return []
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)

    def _wn(video_path, source_lang, glossary=None, source_segments=None):
        assert video_path == "/v.mp4" and source_lang == "ja"
        return [TranscriptSegment(text="EN one", start=0.0, end=1.0, speaker="Speaker 1"),
                TranscriptSegment(text="EN two", start=1.0, end=2.0, speaker="Speaker 1")]
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments", _wn)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("こんにちは", 0.0, 1.0), _src("世界", 1.0, 2.0)]

    result = _run(pipeline._background_post_processing(
        "jobW", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    assert nmt_calls["n"] == 0  # Whisper-native handled it; NMT never called
    assert result["target_transcript"][0]["text"] == "EN one"
    assert db.fields.get("translation_status") == "translated"
    assert any("translated_transcript" in c for c in db.update_calls)


def test_pipeline_falls_back_to_nmt_when_whisper_native_empty(monkeypatch):
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    # Skip the LLM branch so this test deterministically exercises the
    # Whisper-native-vs-offline-NMT ordering it asserts.
    monkeypatch.setattr(pipeline.settings, "TRANSLATION_PREFER_LLM", False, raising=False)

    nmt_calls = {"n": 0}
    async def _nmt(segs, **k):
        nmt_calls["n"] += 1
        # Non-CJK English so the pipeline's final purity gate accepts it.
        return [TranscriptSegment(text="NMT " + _en(s.text), start=s.start, end=s.end,
                                  speaker=s.speaker) for s in segs]
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)
    # Whisper-native unavailable → empty → must fall back to offline NMT.
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments",
                        lambda *a, **k: [])

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("こんにちは", 0.0, 1.0), _src("世界", 1.0, 2.0)]

    result = _run(pipeline._background_post_processing(
        "jobN", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    assert nmt_calls["n"] == 1  # fell back to the offline NMT path
    assert result["target_transcript"][0]["text"].startswith("NMT ")
    assert db.fields.get("translation_status") == "translated"


def test_pipeline_skips_whisper_native_on_low_vram_and_uses_nmt(monkeypatch):
    """On a low-VRAM GPU the Whisper-native pass would fall back to CPU (~30 min),
    so the gate must skip it entirely and translate via offline NMT instead —
    Whisper-native must NOT even be attempted."""
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    # Only ~2.6 GB free (the 4 GB-card case) — below the 4 GB gate.
    monkeypatch.setattr(pipeline, "_gpu_free_vram_gb", lambda: 2.6)
    # Skip the LLM branch so this test deterministically exercises the low-VRAM
    # Whisper-native skip → offline-NMT ordering it asserts.
    monkeypatch.setattr(pipeline.settings, "TRANSLATION_PREFER_LLM", False, raising=False)

    whisper_calls = {"n": 0}
    def _wn(*a, **k):
        whisper_calls["n"] += 1
        return [TranscriptSegment(text="WN", start=0.0, end=1.0, speaker="Speaker 1")]
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments", _wn)

    nmt_calls = {"n": 0}
    async def _nmt(segs, **k):
        nmt_calls["n"] += 1
        # Non-CJK English so the pipeline's final purity gate accepts it.
        return [TranscriptSegment(text="NMT " + _en(s.text), start=s.start, end=s.end,
                                  speaker=s.speaker) for s in segs]
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("こんにちは", 0.0, 1.0), _src("世界", 1.0, 2.0)]

    result = _run(pipeline._background_post_processing(
        "jobLowVram", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    assert whisper_calls["n"] == 0   # the slow CPU pass was never attempted
    assert nmt_calls["n"] == 1       # offline NMT did the work
    assert result["target_transcript"][0]["text"].startswith("NMT ")


def test_pipeline_reuses_loaded_whisper_on_low_vram(monkeypatch):
    """When the transcription Whisper model is STILL loaded (the analyze stage
    deferred its release), Whisper-native translate REUSES it even on a low-VRAM
    card — no second load — instead of skipping to NMT."""
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    monkeypatch.setattr(pipeline, "_gpu_free_vram_gb", lambda: 2.6)   # below the reload gate
    monkeypatch.setattr(pipeline, "_whisper_engine_cached", lambda: True)  # but cached → reuse

    whisper_calls = {"n": 0}
    def _wn(*a, **k):
        whisper_calls["n"] += 1
        # Cover the full 0-2 s source so the coverage guard keeps this output
        # (this test exercises reuse, not the sparse-coverage fallback).
        return [TranscriptSegment(text="WN", start=0.0, end=2.0, speaker="Speaker 1")]
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments", _wn)

    nmt_calls = {"n": 0}
    async def _nmt(segs, **k):
        nmt_calls["n"] += 1
        return [TranscriptSegment(text="NMT " + s.text, start=s.start, end=s.end,
                                  speaker=s.speaker) for s in segs]
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("こんにちは", 0.0, 1.0), _src("世界", 1.0, 2.0)]

    result = _run(pipeline._background_post_processing(
        "jobReuse", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    assert whisper_calls["n"] == 1   # reused the loaded model instead of skipping
    assert nmt_calls["n"] == 0


def test_whisper_native_resegments_word_timed_before_polish(monkeypatch):
    """Whisper-native long cues are split into sentence cues using the model's
    OWN word timing, BEFORE the post-edit drops the word timestamps — so the
    boundaries are accurate (not the char-proportional fallback)."""
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    monkeypatch.setattr(pipeline.settings, "SENTENCE_SEGMENTATION_ENABLED", True, raising=False)
    monkeypatch.setattr(pipeline, "_whisper_engine_cached", lambda: True)

    def _wn(*a, **k):
        return [TranscriptSegment(
            text="Hi. This is a much longer sentence here.",
            start=0.0, end=40.0, speaker="Speaker 1",
            words=[{"word": "Hi.", "start": 0.0, "end": 30.0},
                   {"word": "This", "start": 30.0, "end": 31.0},
                   {"word": "is", "start": 31.0, "end": 32.0},
                   {"word": "a", "start": 32.0, "end": 33.0},
                   {"word": "much", "start": 33.0, "end": 34.0},
                   {"word": "longer", "start": 34.0, "end": 35.0},
                   {"word": "sentence", "start": 35.0, "end": 37.0},
                   {"word": "here.", "start": 37.0, "end": 40.0}])]
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments", _wn)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    result = _run(pipeline._background_post_processing(
        "jobReseg", [_src("おはよう", 0.0, 40.0)], _fake_orchestrator(), job,
        polished_already=False))

    assert result["translated"] is True
    tt = result["target_transcript"]
    assert len(tt) == 2                       # the long cue was split in two
    assert tt[1]["start"] >= 29.0             # 2nd sentence at ~30s (word-timed)


def test_llm_translation_skips_post_edit(monkeypatch):
    """The editorial LLM translates the source directly → its output is final.
    The MT post-edit (which compares each line to the SOURCE) must be SKIPPED for
    the LLM path; running it reverted good English back to Japanese in production.
    """
    import json as _json
    import re as _re

    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    monkeypatch.setattr(pipeline.settings, "AI_TRANSCRIPT_CORRECTION", True, raising=False)
    monkeypatch.setattr(pipeline.settings, "TRANSLATION_PREFER_LLM", True, raising=False)

    class _Orch(SimpleNamespace):
        async def text_completion(self, prompt, **k):
            n = len(_re.findall(r"^\d+\. ", prompt, _re.M))
            return _json.dumps([f"English {i + 1}" for i in range(n)])

    mtpe_calls = {"n": 0}

    async def _spy_mtpe(segments, *a, **k):
        mtpe_calls["n"] += 1
        return list(segments)

    monkeypatch.setattr("backend.services.transcript_polisher.correct_transcript", _spy_mtpe)

    orch = _Orch(reset_circuit_breaker=lambda: None,
                 get_editorial_model_info=lambda: {"is_thinking": False})
    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("日本語の文" + str(i), i * 1.0, i * 1.0 + 1.0) for i in range(6)]

    result = _run(pipeline._background_post_processing(
        "jobLLM", transcript, orch, job, polished_already=False))

    assert result["translated"] is True
    assert mtpe_calls["n"] == 0                                      # post-edit skipped
    tt = result["target_transcript"]
    assert tt and all(s["text"].startswith("English") for s in tt)  # all English, intact


def test_timeline_coverage_merges_overlaps():
    segs = [
        TranscriptSegment(text="a", start=0.0, end=10.0, speaker="S"),
        TranscriptSegment(text="b", start=5.0, end=15.0, speaker="S"),   # overlap → union 0-15
        TranscriptSegment(text="c", start=20.0, end=25.0, speaker="S"),
    ]
    assert pipeline._timeline_coverage_s(segs) == 20.0   # 15 + 5
    assert pipeline._timeline_coverage_s([]) == 0.0


def test_pipeline_rejects_sparse_whisper_native_and_uses_nmt(monkeypatch):
    """Whisper's translate task skips singing, so on a lyric-heavy video it can
    cover far less of the audio than the source transcript. The coverage guard
    must discard that sparse output and translate the full source via NMT — this
    is the regression that left long Japanese gaps in the subtitles."""
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)   # capable GPU → the Whisper gate opens

    # Whisper-native returns a single short cue (~2 s) for a 100 s source.
    def _wn(*a, **k):
        return [TranscriptSegment(text="WN", start=0.0, end=2.0, speaker="Speaker 1")]
    monkeypatch.setattr(pipeline, "_whisper_native_translate_segments", _wn)

    nmt_calls = {"n": 0}
    async def _nmt(segs, **k):
        nmt_calls["n"] += 1
        # Non-CJK English so the pipeline's final purity gate accepts it.
        return [TranscriptSegment(text="NMT " + _en(s.text), start=s.start, end=s.end,
                                  speaker=s.speaker) for s in segs]
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    transcript = [_src("こんにちは", i * 5.0, i * 5.0 + 5.0) for i in range(20)]  # dense 0-100s

    result = _run(pipeline._background_post_processing(
        "jobSparse", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    assert nmt_calls["n"] == 1                                  # NMT did the work
    assert result["target_transcript"][0]["text"].startswith("NMT ")


def test_nmt_translation_keeps_legitimately_repeated_lines(monkeypatch):
    """Offline NMT never hallucinates loops, so the global loop-drop (which
    targets Whisper hallucinations) must NOT run on NMT output — a chorus /
    recurring narration that genuinely repeats across the timeline must survive.
    """
    db = _FakeDB({"subtitle_language": "en", "language": "ja", "clips": [], "summary": None})
    _install_pipeline(monkeypatch, db)
    monkeypatch.setattr(pipeline.settings, "WHISPER_TRANSLATE_TO_EN", False, raising=False)

    async def _nmt(segs, **k):
        out = []
        for s in segs:
            txt = ("We must protect this colony until our dying breath"
                   if "守る" in s.text else "Distinct dialogue line " + s.text)
            out.append(TranscriptSegment(text=txt, start=s.start, end=s.end,
                                         speaker=s.speaker))
        return out
    monkeypatch.setattr(translator, "translate_segments_with_fallback", _nmt)

    job = SimpleNamespace(subtitle_language="en", language="ja", clips=[],
                          summary=None, file_path="/v.mp4")
    # A long chorus line that recurs 6× across the timeline, each time separated
    # by distinct dialogue (so it is NON-adjacent — only the global loop-drop
    # could remove it).
    transcript = []
    for i in range(6):
        transcript.append(_src("コロニーを守る", i * 20.0, i * 20.0 + 5.0))
        transcript.append(_src(f"べつのせりふ{i}", i * 20.0 + 5.0, i * 20.0 + 10.0))

    result = _run(pipeline._background_post_processing(
        "jobRepeat", transcript, _fake_orchestrator(), job, polished_already=False))

    assert result["translated"] is True
    kept = [s for s in result["target_transcript"]
            if "protect this colony" in s["text"]]
    # All 6 recurrences survive — the loop-drop (which would keep only 1) is
    # gated off for the NMT path.
    assert len(kept) == 6, f"expected 6 chorus cues, got {len(kept)}"
