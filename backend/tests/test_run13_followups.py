"""Follow-ups from the 2026-07-03 21:32 run (first run on build 720e347).

That run PROVED the duplication fix (403 cues, zero phantom copies) and the
new phase labels, and surfaced the next layer of issues:

  * bridge_conversion took 480s and ran ON the event loop — the browser got
    no updates from 34m14s to 42m15s, then CONVERT appeared with an 8m-stale
    clock (the 60% update only reached the socket when the bridge finished).
  * The batch-thumbnail ffmpeg decoded the WHOLE 128-min file per chunk
    (bare -i + select) and timed out at 120s, degrading to ~224 per-scene
    seeks — most of those 480s.
  * Speaker fusion + source polish (~10 min) then ran still labeled
    "render plan conversion".
  * The EXTRACT band interleaved frame (8-9%) and audio (9-14%) updates —
    visible percent regressions.
  * The gap-fill pass re-emitted transcription fractions, so "Transcribing
    audio with Whisper (100%)" popped up six minutes into refinement.
  * One shipped cue read "Sure, here is the translation:\\n\\nNothing really
    matters." — an LLM preamble echoed into the subtitle track.
"""

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _name, _attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                     ("anthropic", "AsyncAnthropic")):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod
if "google.generativeai" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.generativeai")
    _gg.configure = lambda *a, **k: None
    _gg.GenerativeModel = object
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules["google.generativeai"] = _gg

import os

import pytest

from backend.services.translator import strip_llm_preamble


# ── LLM-preamble strip ───────────────────────────────────────────────────

def test_strips_shipped_artifact():
    assert strip_llm_preamble(
        "Sure, here is the translation:\n\nNothing really matters."
    ) == "Nothing really matters."


@pytest.mark.parametrize("raw,want", [
    ("Here is the translation: Take off your pants.",
     "Take off your pants."),
    ("here's your English translation — It feels great.",
     "It feels great."),
    ("Translation: I love Mika.", "I love Mika."),
    ("English: See you next time.", "See you next time."),
])
def test_strips_label_variants(raw, want):
    assert strip_llm_preamble(raw) == want


def test_normal_lines_untouched():
    for line in ["How old are you now, huh?",
                 "Here is the cake I bought.",   # 'here is the' without 'translation'
                 "It's my birthday and yet..."]:
        assert strip_llm_preamble(line) == line


def test_never_strips_to_empty():
    assert strip_llm_preamble("Here is the translation:") \
        == "Here is the translation:"


def test_batch_parse_applies_preamble_strip():
    from backend.services.translator import _parse_json_array
    out = _parse_json_array(
        '["Sure, here is the translation: Hello.", "Plain line."]', 2)
    assert out == ["Hello.", "Plain line."]


# ── Windowed batch thumbnails ────────────────────────────────────────────

def test_batch_thumbnails_seek_windows_each_chunk(tmp_path, monkeypatch):
    from backend.services import reframer_bridge as RB
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        vf = cmd[cmd.index("-vf") + 1]
        n = vf.count("between(")
        d = os.path.dirname(cmd[-1])
        for i in range(1, n + 1):
            open(os.path.join(d, f"t_{i:06d}.jpg"), "wb").write(b"x")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(RB.subprocess, "run", fake_run)
    # Late-in-file timestamps (the 21:32 run's failing chunk: t≈6600-7650s)
    ts = [6600.0 + i * 40.0 for i in range(24)]
    outs = [str(tmp_path / f"s_{i:04d}.jpg") for i in range(24)]
    assert RB._extract_thumbnails_batch("/v.mp4", ts, outs) == 24
    cmd = calls[0]
    # Input seeking must bound the decode to the chunk's span…
    assert "-ss" in cmd and cmd.index("-ss") < cmd.index("-i")
    assert "-t" in cmd
    seek = float(cmd[cmd.index("-ss") + 1])
    assert seek == pytest.approx(6598.0, abs=0.01)
    # …and the select windows are rebased to the seek point (t≈0 onward),
    # not absolute file time.
    vf = cmd[cmd.index("-vf") + 1]
    assert "between(t,2.000,2.050)" in vf
    assert "between(t,6600" not in vf


def test_batch_thumbnails_timeout_scales_with_span(monkeypatch, tmp_path):
    from backend.services import reframer_bridge as RB
    seen = {}

    def fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        vf = cmd[cmd.index("-vf") + 1]
        d = os.path.dirname(cmd[-1])
        for i in range(1, vf.count("between(") + 1):
            open(os.path.join(d, f"t_{i:06d}.jpg"), "wb").write(b"x")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(RB.subprocess, "run", fake_run)
    # 60-min window → needs far more than the old flat 120s
    ts = [0.0, 3600.0]
    outs = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
    RB._extract_thumbnails_batch("/v.mp4", ts, outs)
    assert seen["timeout"] > 300


# ── pipeline progress plumbing (source-level contracts) ─────────────────

def _pipeline_src():
    import inspect
    from backend.services import pipeline
    return inspect.getsource(pipeline)


def test_bridge_runs_off_the_event_loop():
    src = _pipeline_src()
    assert "asyncio.to_thread(_run_bridge)" in src


def test_extract_band_percent_is_forward_only():
    src = _pipeline_src()
    assert "_extract_pct_floor" in src


def test_stale_transcribe_messages_latched_after_refine():
    src = _pipeline_src()
    assert "_hint_latch" in src


def test_post_bridge_phase_is_labeled():
    src = _pipeline_src()
    assert "source transcript polish" in src
