"""Regression tests for defects found in the 2026-07-11 live-run logs.

  * Translation hard-failed with "staka/fugumt-japanese-en is not a valid
    model identifier": whisper.cpp reports FULL language names ("japanese")
    and every downstream consumer keys on ISO 639-1 codes. A shared
    normalizer now maps names → codes at every entry point.
  * qwen emitted bare (unquoted) JSON keys ("{ index: 3, text: ... }") —
    json.loads rejected the whole batch, costing a halve-and-retry round
    trip per failure. The parser now repairs bare keys / trailing commas
    and regex-salvages indexed pairs from truncated arrays.
  * The background polish loop ran 3 full passes after translation failed,
    each burning a 600 s budget to change <3%% of cues (~33 min total). The
    loop now stops on diminishing returns and an overall wall budget, and
    the light pre-translation source polish counts as the source polish.
  * The resegment merge concatenated Whisper's overlapping boundary text,
    doubling cue tails ("端っこから食べれるうん端っこから食べれるうん").
    The merge now trims the duplicated overlap.
  * Gap recovery silently skipped 272 s of uncovered speech (config default
    off, no log evidence). Skips over real gaps now log a reason, slice
    failures are counted, and a dead remote aborts early.
"""

import asyncio
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# F1: language-name → ISO normalization
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_lang_code_names_and_aliases():
    from backend.services.language_codes import normalize_lang_code as n
    assert n("japanese") == "ja"
    assert n("Japanese") == "ja"
    assert n("jpn") == "ja"
    assert n("jp") == "ja"
    assert n("english") == "en"
    assert n("chinese") == "zh"
    assert n("cantonese") == "yue"
    assert n("haitian creole") == "ht"
    assert n("japanese-jp") == "ja"
    assert n("pt_BR") == "pt"


def test_normalize_lang_code_passthroughs():
    from backend.services.language_codes import normalize_lang_code as n
    assert n("ja") == "ja"
    assert n("") == ""
    assert n("auto") == "auto"
    assert n(None) == ""
    # regional Chinese variants must survive (Flores Hans/Hant distinction)
    assert n("zh-TW") == "zh-tw"
    assert n("zh-cn") == "zh-cn"
    # unknown values pass through lowercased, never raise
    assert n("Klingon") == "klingon"


def test_iso_to_flores_accepts_full_names():
    from backend.services.nmt_translator import iso_to_flores
    assert iso_to_flores("japanese") == "jpn_Jpan"
    assert iso_to_flores("english") == "eng_Latn"
    assert iso_to_flores("zh-tw") == "zho_Hant"
    assert iso_to_flores("ja") == "jpn_Jpan"


def test_opus_translator_builds_valid_repo_from_full_name():
    from backend.services.nmt_translator import OpusMTTranslator
    t = OpusMTTranslator("japanese", "english", subdir="fugumt",
                         model_template="staka/fugumt-{src}-{tgt}")
    assert t.source == "ja"
    assert t.target == "en"
    assert t.hf_repo == "staka/fugumt-ja-en"


def test_opus_translator_get_caches_by_normalized_pair():
    from backend.services.nmt_translator import OpusMTTranslator
    a = OpusMTTranslator.get("japanese", "en", subdir="fugumt-test")
    b = OpusMTTranslator.get("ja", "english", subdir="fugumt-test")
    assert a is b


# ─────────────────────────────────────────────────────────────────────────────
# F2: lenient JSON repair in the polish parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_repairs_bare_keys():
    from backend.services.transcript_polisher import _parse_polished_response
    resp = '[{ index: 0, text: "こんにちは。" }, { index: 2, text: "はい。" }]'
    assert _parse_polished_response(resp, 3) == ["こんにちは。", None, "はい。"]


def test_parse_repairs_trailing_comma():
    from backend.services.transcript_polisher import _parse_polished_response
    resp = '[{"index": 0, "text": "a"}, {"index": 1, "text": "b"},]'
    assert _parse_polished_response(resp, 2) == ["a", "b"]


def test_parse_salvages_truncated_array():
    from backend.services.transcript_polisher import _parse_polished_response
    resp = ('[{"index": 0, "text": "first"}, {"index": 1, "text": "second"}, '
            '{"index": 2, "te')
    assert _parse_polished_response(resp, 3) == ["first", "second", None]


def test_parse_salvages_bare_keys_and_truncation_with_escapes():
    from backend.services.transcript_polisher import _parse_polished_response
    resp = '[{ index: 0, text: "first\\u3002" }, { index: 1, text: "sec'
    assert _parse_polished_response(resp, 2) == ["first。", None]


def test_parse_still_rejects_garbage_and_accepts_valid():
    from backend.services.transcript_polisher import _parse_polished_response
    assert _parse_polished_response("total nonsense", 2) is None
    assert _parse_polished_response('["a", "b"]', 2) == ["a", "b"]


# ─────────────────────────────────────────────────────────────────────────────
# F3: polish loop stops on diminishing returns + wall budget
# ─────────────────────────────────────────────────────────────────────────────

def _mk_segments(n=10):
    from backend.models import TranscriptSegment
    return [TranscriptSegment(start=float(i), end=float(i) + 0.9,
                              text=f"cue {i}", speaker="S1") for i in range(n)]


def _loop_env(monkeypatch, correct_fn, score=50.0):
    """Wire _polish_transcript_loop's collaborators to fakes."""
    from backend.services import transcript_polisher as tp
    from backend.services import subtitle_formatter as sf
    monkeypatch.setattr(tp, "correct_transcript", correct_fn)
    monkeypatch.setattr(settings, "SUBTITLE_CPS_ENFORCEMENT", False)
    monkeypatch.setattr(
        sf, "compute_readability_report", lambda segs: {"score": score})


def test_polish_loop_stops_after_low_yield_pass(monkeypatch):
    from backend.services import pipeline as pl

    calls = {"n": 0}

    async def _no_change(segs, orch, **kw):
        calls["n"] += 1
        return segs  # pass changes 0 cues → below min-yield → stop

    _loop_env(monkeypatch, _no_change, score=50.0)  # score < target: no break
    orch = types.SimpleNamespace(
        get_editorial_model_info=lambda: {"is_thinking": False})
    models, report = asyncio.run(pl._polish_transcript_loop(
        "job-t", _mk_segments(), orch, "ja"))
    assert calls["n"] == 1          # low yield stopped passes 2 and 3
    assert len(models) == 10


def test_polish_loop_runs_more_passes_when_yield_is_high(monkeypatch):
    from backend.services import pipeline as pl

    calls = {"n": 0}

    async def _always_change(segs, orch, **kw):
        calls["n"] += 1
        for s in segs:
            s.text = (s.text or "") + "!"
        return segs

    _loop_env(monkeypatch, _always_change, score=50.0)
    monkeypatch.setattr(settings, "TRANSCRIPT_READABILITY_MAX_PASSES", 3)
    orch = types.SimpleNamespace(
        get_editorial_model_info=lambda: {"is_thinking": False})
    asyncio.run(pl._polish_transcript_loop("job-t", _mk_segments(), orch, "ja"))
    assert calls["n"] == 3          # full-yield passes keep running


def test_polish_loop_respects_wall_budget(monkeypatch):
    from backend.services import pipeline as pl

    calls = {"n": 0}

    async def _slow_change(segs, orch, **kw):
        calls["n"] += 1
        for s in segs:
            s.text = (s.text or "") + "!"
        return segs

    _loop_env(monkeypatch, _slow_change, score=50.0)
    monkeypatch.setattr(settings, "TRANSCRIPT_READABILITY_MAX_PASSES", 3)
    # Budget already exhausted the moment the loop checks → stop after pass 1.
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_LOOP_MAX_S", 0.000001)
    orch = types.SimpleNamespace(
        get_editorial_model_info=lambda: {"is_thinking": False})
    asyncio.run(pl._polish_transcript_loop("job-t", _mk_segments(), orch, "ja"))
    assert calls["n"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# F4: overlap trim at the resegment merge
# ─────────────────────────────────────────────────────────────────────────────

def test_trim_boundary_overlap_cjk_full_duplicate():
    from backend.services.sentence_segmenter import _trim_boundary_overlap
    rest, n = _trim_boundary_overlap("端っこから食べれるうん", "端っこから食べれるうん")
    assert rest == "" and n == 11


def test_trim_boundary_overlap_keeps_short_echo():
    from backend.services.sentence_segmenter import _trim_boundary_overlap
    rest, n = _trim_boundary_overlap("はい", "はい")
    assert rest == "はい" and n == 0


def test_trim_boundary_overlap_latin_word_boundary_only():
    from backend.services.sentence_segmenter import _trim_boundary_overlap
    rest, n = _trim_boundary_overlap("and then we went home",
                                     "went home and ate dinner")
    assert rest == "and ate dinner" and n == 9
    # a mid-word coincidence must never trim
    rest, n = _trim_boundary_overlap("the cats", "catsup is red")
    assert rest == "catsup is red" and n == 0


def test_resegment_merge_drops_doubled_tail():
    from backend.services.sentence_segmenter import resegment_by_sentence
    segs = [
        {"start": 10.0, "end": 12.0, "text": "端っこから食べれるうん",
         "speaker": "S1"},
        {"start": 12.1, "end": 14.0, "text": "端っこから食べれるうん",
         "speaker": "S1"},
    ]
    out = resegment_by_sentence(segs)
    joined = "".join((s.text or "") for s in out)
    assert joined.count("端っこから食べれるうん") == 1


def test_readability_merge_drops_doubled_tail():
    from backend.models import TranscriptSegment
    from backend.services.subtitle_formatter import _merge_for_readability
    segs = [
        TranscriptSegment(start=10.0, end=12.0, text="端っこから食べれるうん",
                          speaker="S1"),
        TranscriptSegment(start=12.1, end=13.5, text="端っこから食べれるうん",
                          speaker="S1"),
    ]
    out = _merge_for_readability(
        segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=10.0, max_gap_s=1.0)
    joined = "".join((s.text or "") for s in out)
    assert joined.count("端っこから食べれるうん") == 1


def test_resegment_merge_without_overlap_unchanged():
    from backend.services.sentence_segmenter import resegment_by_sentence
    segs = [
        {"start": 0.0, "end": 1.0, "text": "今日はいい天気です。", "speaker": "S1"},
        {"start": 1.1, "end": 2.0, "text": "散歩に行きましょう。", "speaker": "S1"},
    ]
    out = resegment_by_sentence(segs)
    joined = "".join((s.text or "") for s in out)
    assert "今日はいい天気です。" in joined
    assert "散歩に行きましょう。" in joined


# ─────────────────────────────────────────────────────────────────────────────
# F5: gap-recovery visibility + early abort on a dead remote
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLog:
    def __init__(self):
        self.lines = []

    def log_stage(self, stage, msg):
        self.lines.append(f"[{stage}] {msg}")


def test_gap_recovery_default_is_on():
    from backend.config import Settings
    assert Settings.model_fields["SPEECH_GAP_RECOVERY_ENABLED"].default is True


def test_audit_logs_skip_reason_when_disabled(monkeypatch):
    from backend.services.reframer_audio import AudioIntelligence
    from backend.services import speech_coverage as SC

    monkeypatch.setattr(settings, "SPEECH_COVERAGE_AUDIT_ENABLED", True)
    monkeypatch.setattr(settings, "SPEECH_GAP_RECOVERY_ENABLED", False)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda path: [(0.0, 30.0)])
    log = _FakeLog()
    result = {"segments": [
        {"start_sec": 0.0, "end_sec": 10.0, "text": "covered"}]}
    out = AudioIntelligence._audit_and_recover_speech(
        None, result, "/nonexistent.wav", 30000, "ja",
        object(), log)
    assert any("Gap recovery skipped" in l and "RECOVERY_ENABLED is off" in l
               for l in log.lines), log.lines
    assert out.get("speech_coverage")


def test_audit_logs_skip_reason_when_no_engine(monkeypatch):
    from backend.services.reframer_audio import AudioIntelligence
    from backend.services import speech_coverage as SC

    monkeypatch.setattr(settings, "SPEECH_COVERAGE_AUDIT_ENABLED", True)
    monkeypatch.setattr(settings, "SPEECH_GAP_RECOVERY_ENABLED", True)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda path: [(0.0, 30.0)])
    log = _FakeLog()
    result = {"segments": [
        {"start_sec": 0.0, "end_sec": 10.0, "text": "covered"}]}
    AudioIntelligence._audit_and_recover_speech(
        None, result, "/nonexistent.wav", 30000, "ja", None, log)
    assert any("Gap recovery skipped" in l and "no remote engine" in l
               for l in log.lines), log.lines


def test_recover_aborts_early_when_remote_dead(monkeypatch, tmp_path):
    from backend.services import reframer_audio as ra

    # Fake ffmpeg: "extract" writes a big-enough wav so the upload is tried.
    def _fake_run(cmd, **kw):
        wav = cmd[-1]
        with open(wav, "wb") as f:
            f.write(b"\x00" * 4096)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ra.subprocess, "run", _fake_run)

    calls = {"n": 0}

    class _DeadEngine:
        def transcribe_wav(self, wav, lang):
            calls["n"] += 1
            return None

    log = _FakeLog()
    gaps = [(float(i * 10), float(i * 10 + 5)) for i in range(10)]
    out = ra.AudioIntelligence._recover_voice_gaps_remote(
        None, gaps, str(tmp_path / "audio.wav"), "ja", _DeadEngine(), log)
    assert out == []
    assert calls["n"] == 3          # aborted after 3 straight failures
    assert any("Gap recovery aborted" in l for l in log.lines), log.lines


def test_recover_logs_summary_on_success(monkeypatch, tmp_path):
    from backend.services import reframer_audio as ra

    def _fake_run(cmd, **kw):
        wav = cmd[-1]
        with open(wav, "wb") as f:
            f.write(b"\x00" * 4096)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(ra.subprocess, "run", _fake_run)

    class _GoodEngine:
        def transcribe_wav(self, wav, lang):
            return {"segments": [
                {"start_sec": 0.1, "end_sec": 1.2, "text": "こんにちは",
                 "no_speech_prob": 0.1}]}

    log = _FakeLog()
    out = ra.AudioIntelligence._recover_voice_gaps_remote(
        None, [(5.0, 8.0), (20.0, 24.0)], str(tmp_path / "audio.wav"),
        "ja", _GoodEngine(), log)
    assert len(out) == 2
    # timestamps shifted back to absolute time (slice start minus 0.2s pad)
    assert out[0]["start_sec"] == pytest.approx(4.9, abs=0.01)
    assert out[0]["source"] == "gap_recovery"
    assert any("Gap recovery finished: 2 cue(s)" in l for l in log.lines), log.lines


def test_resegment_merge_respects_duration_cap():
    """Unpunctuated same-speaker chains must not weld into paragraph cues —
    the 2026-07-12 run merged 950 → 488 with 30-40 s blobs that survived to
    the export. No merged cue may exceed SENTENCE_MERGE_MAX_CUE_S."""
    from backend.services.sentence_segmenter import resegment_by_sentence
    from backend.config import settings as _s
    cap = float(getattr(_s, "SENTENCE_MERGE_MAX_CUE_S", 12.0))
    # 20 unpunctuated 2.5s cues, 0.1s gaps, same speaker: an uncapped merge
    # would produce one ~50s cue that nothing downstream can re-split.
    segs = [{"start": i * 2.6, "end": i * 2.6 + 2.5,
             "text": "これはテストの発話です", "speaker": "S1"}
            for i in range(20)]
    out = resegment_by_sentence(segs)
    assert len(out) >= 4
    for s in out:
        assert (s.end - s.start) <= cap + 2.6, (
            f"cue {s.start}-{s.end} exceeds the merge cap")
