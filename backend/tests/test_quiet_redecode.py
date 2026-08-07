"""Quiet-cue redecode (coverage item 1) — candidate selection, the
fail-soft acceptance gate, and the in-place text/confidence repair.

The pass re-decodes ONLY cues whose decode confidence is deep underwater,
from a loudness-normalized slice, and swaps text strictly on a measurable
confidence win. Its cardinal rule mirrors the rest of the recovery stack:
a wrong replacement is worse than the wrong original, so every doubtful
path must keep the original text.
"""
import asyncio
import os

from backend.services import vocal_gap_recovery as V


# ── candidate selection ─────────────────────────────────────────────────────

def _seg(i, lp=None, start=None, end=None, text=None):
    s = {"start": float(i * 3) if start is None else start,
         "end": float(i * 3 + 2) if end is None else end,
         "text": text if text is not None else f"セリフ{i}"}
    if lp is not None:
        s["avg_logprob"] = lp
    return s


def test_candidates_pick_low_logprob_worst_first():
    segs = [_seg(0, lp=-0.2), _seg(1, lp=-1.1), _seg(2), _seg(3, lp=-2.0)]
    assert V._quiet_candidates(segs, -0.8, 12) == [3, 1]


def test_candidates_respect_cap_floor_and_missing_logprob():
    segs = [_seg(i, lp=-1.0 - i * 0.1) for i in range(6)]
    # cap keeps the WORST n (5 has the lowest logprob)
    got = V._quiet_candidates(segs, -0.8, 2)
    assert got == [5, 4]
    # at/above the floor is not a suspect
    assert V._quiet_candidates([_seg(0, lp=-0.8)], -0.8, 12) == []
    # no avg_logprob at all → never a suspect (nothing to improve on)
    assert V._quiet_candidates([_seg(0)], -0.8, 12) == []


def test_candidates_skip_markers_and_bad_durations():
    segs = [
        _seg(0, lp=-1.5, text="[♪ Opening theme ♪]"),          # marker
        _seg(1, lp=-1.5, start=3.0, end=3.2),                   # too short
        _seg(2, lp=-1.5, start=6.0, end=20.0),                  # too long
        _seg(3, lp=-1.5, text=""),                              # empty
        _seg(4, lp=-1.5),                                       # the one suspect
    ]
    assert V._quiet_candidates(segs, -0.8, 12) == [4]


# ── acceptance gate ─────────────────────────────────────────────────────────

def test_accept_requires_clear_confidence_win():
    dec = [{"text": "こんにちは", "avg_logprob": -0.4, "no_speech_prob": 0.1}]
    ok, text, lp = V._accept_quiet_redecode(-1.2, dec, 0.3)
    assert ok and text == "こんにちは" and lp == -0.4
    # inside the margin → keep the original
    ok, _, _ = V._accept_quiet_redecode(-0.6, dec, 0.3)
    assert not ok
    # worse than the original → keep the original
    ok, _, _ = V._accept_quiet_redecode(-0.2, dec, 0.3)
    assert not ok


def test_accept_fails_soft_on_every_doubtful_decode():
    # empty decode
    assert not V._accept_quiet_redecode(-1.2, [], 0.3)[0]
    # no confidence reported → a win can't be proven
    assert not V._accept_quiet_redecode(
        -1.2, [{"text": "abc", "no_speech_prob": 0.1}], 0.3)[0]
    # hallucination staple
    assert not V._accept_quiet_redecode(
        -1.2, [{"text": "Thank you for watching.", "avg_logprob": -0.1,
                "no_speech_prob": 0.1}], 0.3)[0]
    # high no_speech_prob
    assert not V._accept_quiet_redecode(
        -1.2, [{"text": "abc", "avg_logprob": -0.1,
                "no_speech_prob": 0.95}], 0.3)[0]
    # blank text
    assert not V._accept_quiet_redecode(
        -1.2, [{"text": "   ", "avg_logprob": -0.1,
                "no_speech_prob": 0.1}], 0.3)[0]


def test_accept_joins_multi_segment_decodes_on_worst_logprob():
    dec = [
        {"text": "前半", "avg_logprob": -0.3, "no_speech_prob": 0.1},
        {"text": "後半", "avg_logprob": -0.5, "no_speech_prob": 0.1},
    ]
    ok, text, lp = V._accept_quiet_redecode(-1.4, dec, 0.3)
    assert ok and text == "前半 後半" and lp == -0.5


# ── stem normalization carries the acceptance signal ────────────────────────

def test_normalize_stem_segments_carries_avg_logprob():
    rows = V._normalize_stem_segments([
        {"start_sec": 1.0, "end_sec": 2.0, "text": "a", "avg_logprob": -0.55},
        {"start": 3.0, "end": 4.0, "text": "b"},
    ])
    assert rows[0]["avg_logprob"] == -0.55
    assert "avg_logprob" not in rows[1]
    assert rows[0]["start"] == 1.0 and rows[1]["start"] == 3.0


# ── end-to-end in-place repair ──────────────────────────────────────────────

def _run_redecode(tmp_path, monkeypatch, segments, decoded_by_call):
    """Drive redecode_quiet_segments with stubbed slicing + ASR."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 64)
    calls = {"n": 0}

    def _fake_slice(_a, out_path, _s, _e):
        with open(out_path, "wb") as f:
            f.write(b"\0" * 2048)
        return True

    def _fake_asr(_wav, _lang, _patience):
        i = calls["n"]
        calls["n"] += 1
        return decoded_by_call[min(i, len(decoded_by_call) - 1)]

    monkeypatch.setattr(V, "_slice_wav_boosted", _fake_slice)
    monkeypatch.setattr(V, "_transcribe_stem", _fake_asr)
    n = asyncio.run(V.redecode_quiet_segments(
        "job-q", str(audio), segments, "ja", str(tmp_path / "work")))
    return n, calls["n"]


def test_redecode_replaces_text_in_place_on_a_clear_win(tmp_path, monkeypatch):
    segs = [_seg(0, lp=-0.2), _seg(1, lp=-1.6), _seg(2, lp=-0.1)]
    n, asr_calls = _run_redecode(
        tmp_path, monkeypatch, segs,
        [[{"text": "本当のセリフ", "avg_logprob": -0.3, "no_speech_prob": 0.1}]])
    assert n == 1 and asr_calls == 1
    assert segs[1]["text"] == "本当のセリフ"
    assert segs[1]["avg_logprob"] == -0.3
    assert segs[1]["quiet_redecoded"] is True
    # neighbours untouched
    assert segs[0]["text"] == "セリフ0" and segs[2]["text"] == "セリフ2"
    # timing NEVER touched
    assert segs[1]["start"] == 3.0 and segs[1]["end"] == 5.0


def test_redecode_confirms_same_text_by_lifting_confidence(tmp_path, monkeypatch):
    segs = [_seg(0, lp=-1.6)]
    n, _ = _run_redecode(
        tmp_path, monkeypatch, segs,
        [[{"text": "セリフ0", "avg_logprob": -0.3, "no_speech_prob": 0.1}]])
    # same reading = 0 replacements, but the lifted confidence sticks so
    # downstream [UNRELIABLE ASR] marking stops firing on a vouched-for cue
    assert n == 0
    assert segs[0]["text"] == "セリフ0"
    assert segs[0]["avg_logprob"] == -0.3
    assert segs[0].get("quiet_redecoded") is True


def test_redecode_keeps_original_when_gate_rejects(tmp_path, monkeypatch):
    segs = [_seg(0, lp=-1.6)]
    n, _ = _run_redecode(
        tmp_path, monkeypatch, segs,
        [[{"text": "別の言葉", "avg_logprob": -1.5, "no_speech_prob": 0.1}]])
    assert n == 0
    assert segs[0]["text"] == "セリフ0"
    assert segs[0]["avg_logprob"] == -1.6
    assert "quiet_redecoded" not in segs[0]


def test_redecode_stops_on_transport_failure(tmp_path, monkeypatch):
    # None = the ASR transport is down: later slices fail identically, so
    # the pass must stop paying for them after the first None.
    segs = [_seg(0, lp=-1.6), _seg(1, lp=-1.5), _seg(2, lp=-1.4)]
    n, asr_calls = _run_redecode(tmp_path, monkeypatch, segs, [None])
    assert n == 0 and asr_calls == 1
    assert all("quiet_redecoded" not in s for s in segs)


def test_redecode_disabled_by_flag(tmp_path, monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "WHISPER_QUIET_REDECODE", False, raising=False)
    segs = [_seg(0, lp=-1.6)]
    called = {"n": 0}
    monkeypatch.setattr(V, "_transcribe_stem",
                        lambda *a: called.__setitem__("n", called["n"] + 1))
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF" + b"\0" * 64)
    n = asyncio.run(V.redecode_quiet_segments(
        "job-q", str(audio), segs, "ja", str(tmp_path / "w")))
    assert n == 0 and called["n"] == 0 and segs[0]["text"] == "セリフ0"
