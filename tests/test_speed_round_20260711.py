"""Tests for the 30-40-minute speed round (overlap + batching, no quality loss).

  * Opus/FuguMT CT2 decode is now truly batched (one call per decode-length
    bucket instead of one call per cue) with per-chunk fallback on a batch
    failure, and the context path issues two batched calls per cue batch.
  * The perceiver's frame acquisition runs on a reader thread feeding a
    bounded queue (pipelined mode) — outputs must be identical to the
    serial path.
  * The clip judge skips the 4-ffmpeg-seek keyframe extraction once the
    model is known to reject images.
  * Config defaults: pipelined acquisition on, NMT batch 32, clip export
    3-wide.
"""

import asyncio
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# NMT: batched CT2 decode
# ─────────────────────────────────────────────────────────────────────────────

class _FakeTok:
    def encode_as_pieces(self, text):
        return list(text)          # 1 token per char

    def decode(self, pieces):
        return "".join(pieces)


class _FakeHyp:
    def __init__(self, hypotheses):
        self.hypotheses = hypotheses


class _FakeCT2:
    """Records every translate_batch call; 'translates' by upper-casing."""
    def __init__(self):
        self.calls = []

    def translate_batch(self, source_list, **kwargs):
        self.calls.append((len(source_list), kwargs))
        return [_FakeHyp([[t.upper() for t in toks if t != "</s>"]])
                for toks in source_list]


def _mk_opus():
    from backend.services.nmt_translator import OpusMTTranslator
    t = OpusMTTranslator("ja", "en", subdir="test-batching")
    t._translator = _FakeCT2()
    t._tokenizer = _FakeTok()
    t._loaded = True
    return t


def test_translate_batch_uses_one_ct2_call_per_bucket():
    t = _mk_opus()
    out = t.translate_batch(["abc", "def", "ghi", "jkl"])
    assert out == ["ABC", "DEF", "GHI", "JKL"]
    # 4 same-length texts → same decode bucket → exactly ONE CT2 call.
    assert len(t._translator.calls) == 1
    n_examples, kwargs = t._translator.calls[0]
    assert n_examples == 4
    assert kwargs.get("beam_size") == 5


def test_translate_batch_preserves_order_and_empties():
    t = _mk_opus()
    out = t.translate_batch(["abc", "", "xyz"])
    assert out == ["ABC", "", "XYZ"]


def test_translate_batch_bucket_failure_falls_back_per_chunk():
    t = _mk_opus()
    calls = {"n": 0}

    class _FlakyCT2(_FakeCT2):
        def translate_batch(self, source_list, **kwargs):
            calls["n"] += 1
            if len(source_list) > 1:
                raise RuntimeError("batch blew up")
            return super().translate_batch(source_list, **kwargs)

    t._translator = _FlakyCT2()
    out = t.translate_batch(["abc", "def"])
    assert out == ["ABC", "DEF"]       # recovered per chunk
    assert calls["n"] == 3             # 1 failed batch + 2 singles


def test_translate_with_context_issues_two_batched_calls():
    t = _mk_opus()
    seen = []
    orig = t.translate_batch

    def _spy(texts, glossary=None):
        seen.append(list(texts))
        return orig(texts, glossary=glossary)

    t.translate_batch = _spy
    batch = ["こんにちは", "元気ですか", "はい元気です"]
    out = t.translate_with_context(batch, [], [])
    assert len(out) == 3 and all(o for o in out)
    # ≤ 2 translate_batch calls for the whole batch: one for the context
    # minis, one for the isolated fallbacks — never per-cue.
    assert len(seen) <= 2


def test_translate_with_context_empty_cues_pass_through():
    t = _mk_opus()
    out = t.translate_with_context(["", "abc"], [], [])
    assert out[0] == ""
    assert out[1]


# ─────────────────────────────────────────────────────────────────────────────
# Perceiver: pipelined acquisition ≡ serial acquisition
# ─────────────────────────────────────────────────────────────────────────────

class _FakeFrame:
    def __init__(self, idx):
        self.idx = idx
        self.shape = (360, 640, 3)


class _FakeCap:
    """Deterministic 'video': read() returns a tagged frame + advances."""
    def __init__(self, total=10_000):
        self.pos = 0
        self.total = total

    def get(self, prop):
        return self.pos

    def set(self, prop, value):
        self.pos = int(value)

    def grab(self):
        if self.pos >= self.total:
            return False
        self.pos += 1
        return True

    def retrieve(self):
        return True, _FakeFrame(self.pos)

    def read(self):
        if self.pos >= self.total:
            return False, None
        self.pos += 1
        return True, _FakeFrame(self.pos)


@pytest.fixture()
def _fake_cv2(monkeypatch):
    from backend.services import reframer_perceiver as rp
    fake = types.SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        INTER_LINEAR=1,
        COLOR_BGR2GRAY=6,
        resize=lambda frame, size, interpolation=None: ("small", frame.idx),
        cvtColor=lambda small, code: ("gray", small[1]),
    )
    monkeypatch.setattr(rp, "cv2", fake)
    return fake


def _collect_samples(pipelined, monkeypatch, _n=25):
    from backend.services import reframer_perceiver as rp
    monkeypatch.setattr(settings, "REFRAMER_PIPELINED_ACQUISITION", pipelined,
                        raising=False)
    self = types.SimpleNamespace(cancelled=False)
    r = types.SimpleNamespace(fps=30.0, total_frames=10_000)
    cap = _FakeCap()
    sample_times_ms = [int(i * 1000) for i in range(_n)]   # 1 fps sampling
    tbuck = {"acquire": 0.0}
    items = list(rp.Perceiver._iter_samples(
        self, cap, sample_times_ms, r, det_w=640, det_h=360,
        det_scale=0.5, seek_gap=120, tbuck=tbuck))
    return items, tbuck


def test_pipelined_acquisition_matches_serial(monkeypatch, _fake_cv2):
    serial, _ = _collect_samples(False, monkeypatch)
    piped, _ = _collect_samples(True, monkeypatch)
    assert len(serial) == len(piped) == 25
    for s, p in zip(serial, piped):
        assert s[0] == p[0]            # index
        assert s[1] == p[1]            # time_ms
        assert s[2] == p[2]            # small_bgr (deterministic fake)
        assert s[3] == p[3]            # gray_small
        assert s[4] == p[4]            # gap_grays
    # 30-frame stride at 1 fps sampling → the grab path retrieves LK frames.
    assert any(s[4] for s in serial)


def test_pipelined_acquisition_accumulates_acquire_time(monkeypatch, _fake_cv2):
    _, tbuck = _collect_samples(True, monkeypatch)
    assert tbuck["acquire"] > 0.0


def test_pipelined_acquisition_stops_on_cancel(monkeypatch, _fake_cv2):
    from backend.services import reframer_perceiver as rp
    monkeypatch.setattr(settings, "REFRAMER_PIPELINED_ACQUISITION", True,
                        raising=False)
    self = types.SimpleNamespace(cancelled=False)
    r = types.SimpleNamespace(fps=30.0, total_frames=10_000)
    gen = rp.Perceiver._iter_samples(
        self, _FakeCap(), [i * 1000 for i in range(50)], r,
        det_w=640, det_h=360, det_scale=0.5, seek_gap=120,
        tbuck={"acquire": 0.0})
    first = next(gen)
    assert first[0] == 0
    self.cancelled = True
    remaining = list(gen)               # drains without hanging
    assert len(remaining) < 50


# ─────────────────────────────────────────────────────────────────────────────
# Clip judge: keyframe extraction skipped for text-only judges
# ─────────────────────────────────────────────────────────────────────────────

def test_judge_skips_keyframes_when_vision_unsupported():
    import inspect
    from backend.services import reframer_clipper as rc
    src = inspect.getsource(rc)
    # The worker must consult the judge's vision flag (including a
    # FallbackJudge's INNER judges) BEFORE paying the 4-ffmpeg-seek keyframe
    # extraction.
    assert '_vision_unsupported' in src
    idx_gate = src.find('getattr(j, "_vision_unsupported", False)')
    idx_extract = src.find("_extract_keyframes_b64(\n                    self.video_path")
    assert idx_gate != -1 and idx_extract != -1 and idx_gate < idx_extract
    # The gate must inspect the wrapper's primary/fallback, not just the top
    # object — a FallbackJudge never carries the flag itself.
    assert 'getattr(judge, "primary", None)' in src


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline wiring: summary overlap + translation warm-up
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_overlap_gated_on_cloud_editorial():
    import inspect
    from backend.services import pipeline as pl
    src = inspect.getsource(pl)
    assert "_summary_task = asyncio.create_task(_generate_summary_stage())" in src
    # Local-editorial rigs keep strict ordering (shared 4 GB card).
    assert "if not _editorial_is_local:" in src
    # The task is joined before Auto-SEO consumes the summary.
    assert src.find("_summary_task is not None") < src.find(
        "_final_clips = await _run_post_clip_followups")


def test_repair_and_bridge_overlap_wiring():
    import inspect
    from backend.services import pipeline as pl
    src = inspect.getsource(pl)
    assert "_plan_prep_task = asyncio.create_task(_repair_and_bridge())" in src
    # Joined before anything reads the bridge outputs.
    assert "render_plan, scenes, subject_track = await _plan_prep_task" in src


def test_warm_translation_model_noops_without_config(monkeypatch):
    from backend.services import pipeline as pl
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "", raising=False)

    async def _run():
        pl._warm_translation_model("job-x")
        # Let the fire-and-forget task run to completion.
        await asyncio.sleep(0.05)
        return True

    assert asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────────────────────────────────────

def test_speed_round_config_defaults():
    from backend.config import Settings
    f = Settings.model_fields
    assert f["REFRAMER_PIPELINED_ACQUISITION"].default is True
    assert f["REFRAMER_ACQUIRE_QUEUE_DEPTH"].default == 4
    assert f["NMT_BATCH_SIZE"].default == 32
    assert f["NMT_CT2_INTER_THREADS"].default == 0
    assert f["CLIP_EXPORT_CONCURRENCY"].default == 3
