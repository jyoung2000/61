"""Voice attestation + LLM meta-response guard — the run-9 phantom fixes.

A measured run shipped two garbage blocks: an 11-cue block at 0:00-0:09 of
the timeline rendered over the silence before the opening theme, and a
12-cue block that was the LLM's refusal paragraph ("This sentence appears
to be in Japanese and seems to contain…") split across frame-width cues.
Two independent gates kill both classes at the last writer:

  * ``attest_cues_to_voice`` — a speech cue must overlap VAD-detected voice;
  * ``looks_like_meta_response`` — a "translation" that talks ABOUT the line
    is rejected instead of shipped.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.services.speech_coverage import (  # noqa: E402
    attest_cues_to_voice, voice_activity_regions_cached)
from backend.services.translator import looks_like_meta_response  # noqa: E402


def _rows():
    return [
        {"start": 0.0, "end": 0.6, "text": "Cut! Zechs rules!", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 1.7, "text": "Mission accomplished!", "speaker": "Speaker 1"},
        {"start": 27.8, "end": 31.8, "text": "[♪ Opening theme ♪]", "speaker": "Speaker 1"},
        {"start": 127.6, "end": 129.9, "text": "an operation called 'Operation Meteor'",
         "speaker": "Speaker 2"},
    ]


def test_attestation_drops_head_cues_over_silence(monkeypatch, tmp_path):
    # Voice exists only from 2:07 on — the head block has no audio under it.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(127.0, 135.0)])
    kept, dropped = attest_cues_to_voice(_rows(), str(wav))
    texts = [r["text"] for r in kept]
    assert "Cut! Zechs rules!" not in texts
    assert "Mission accomplished!" not in texts
    assert len(dropped) == 2 and "Zechs" in dropped[0]
    # The dialogue that sits on voice survives.
    assert "an operation called 'Operation Meteor'" in texts


def test_attestation_exempts_markers_over_music(monkeypatch, tmp_path):
    # [♪ Opening theme ♪] annotates MUSIC on purpose — it must never need
    # voice under it.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(127.0, 135.0)])
    kept, _ = attest_cues_to_voice(_rows(), str(wav))
    assert "[♪ Opening theme ♪]" in [r["text"] for r in kept]


def test_attestation_keeps_readability_extended_cues(monkeypatch, tmp_path):
    # The extension pass legitimately stretches a cue far past its voiced
    # audio (CPS relief). A small ABSOLUTE overlap must suffice — no
    # fractional requirement.
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(10.0, 10.5)])
    rows = [{"start": 10.0, "end": 14.0, "text": "Yes.", "speaker": "Speaker 1"}]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept == rows and not dropped


def test_attestation_drops_zero_width_and_timeless_cues(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions",
                        lambda p, **k: [(0.0, 600.0)])
    rows = [
        {"start": 5.0, "end": 5.0, "text": "Effect canceled", "speaker": "Speaker 1"},
        {"start": None, "end": None, "text": "Dialogue start", "speaker": "Speaker 1"},
        {"start": 5.0, "end": 6.0, "text": "Real line.", "speaker": "Speaker 1"},
    ]
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert [r["text"] for r in kept] == ["Real line."]
    assert len(dropped) == 2


def test_attestation_fails_soft_without_vad(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    monkeypatch.setattr(SC, "voice_activity_regions", lambda p, **k: [])
    rows = _rows()
    kept, dropped = attest_cues_to_voice(rows, str(wav))
    assert kept is rows and dropped == []
    # Missing file: same fail-soft contract.
    kept2, dropped2 = attest_cues_to_voice(rows, str(tmp_path / "nope.wav"))
    assert kept2 is rows and dropped2 == []


def test_vad_cache_one_pass_per_file(monkeypatch, tmp_path):
    from backend.services import speech_coverage as SC
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 64)
    calls = {"n": 0}

    def _fake(p, **k):
        calls["n"] += 1
        return [(1.0, 2.0)]

    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    SC._VAD_CACHE.clear()
    assert voice_activity_regions_cached(str(wav)) == [(1.0, 2.0)]
    assert voice_activity_regions_cached(str(wav)) == [(1.0, 2.0)]
    assert calls["n"] == 1, "second consult must hit the cache"
    # An empty result is NEVER cached — VAD can become available later.
    monkeypatch.setattr(SC, "voice_activity_regions", lambda p, **k: [])
    wav2 = tmp_path / "b.wav"
    wav2.write_bytes(b"RIFF" + b"\0" * 64)
    assert voice_activity_regions_cached(str(wav2)) == []
    monkeypatch.setattr(SC, "voice_activity_regions", _fake)
    assert voice_activity_regions_cached(str(wav2)) == [(1.0, 2.0)]


# ── The meta-response guard ─────────────────────────────────────────────────

_SHIPPED_REFUSAL = (
    "This sentence appears to be in Japanese and seems to contain multiple "
    "characters that do not form coherent words or sentences. Given the "
    "context provided, it's unclear what specific phrase is intended for "
    "translation as \"今回の殺さ\" does not appear to be a valid Japanese "
    "word. If we were to interpret this literally (which would likely result "
    "in nonsensical English), it could approximate: \"This time killing\".")


def test_meta_guard_catches_the_shipped_refusal_paragraph():
    assert looks_like_meta_response(_SHIPPED_REFUSAL, "今回の殺さ") is True
    # Pattern alone suffices — even without the source for the ratio check.
    assert looks_like_meta_response(_SHIPPED_REFUSAL) is True


def test_meta_guard_passes_real_dialogue():
    for line in (
        "Mission accomplished!",
        "That's definitely a Gundam.",
        "It seems our enemy possesses quite advanced technology.",
        "I'm sorry.",                       # a real apology LINE, not a refusal
        "Please, tell me what happened.",
    ):
        assert looks_like_meta_response(line, "了解、任務完了だ") is False, line


def test_meta_guard_rejects_paragraph_blowup_with_source():
    src = "今回の殺さ"
    blown = "word " * 60                    # 300 chars for a 5-char source
    assert looks_like_meta_response(blown, src) is True
    # The same text without a source is not flagged by ratio alone.
    assert looks_like_meta_response(blown) is False
    # A normal-length translation of a short line is fine.
    assert looks_like_meta_response("This time, the killing begins.", src) is False


def test_validate_translation_uses_the_meta_guard():
    from backend.services.pipeline import _validate_translation
    assert _validate_translation(_SHIPPED_REFUSAL, "ja", src_text="今回の殺さ") == ""
    assert _validate_translation(
        "Mission accomplished!", "ja", src_text="任務完了") == "Mission accomplished!"
