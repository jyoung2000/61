"""Profanity censor: text masking (first + last letter kept, user symbol),
beep-interval detection (word timestamps or char-weight interpolation),
source→output time mapping through trim + speed, the ffmpeg beep post-pass
command, and the Settings endpoints (block list / symbol / sound / default)."""

import asyncio
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.settings as S
import backend.services.censor as C
from backend.config import settings as cfg


# ── Text masking ────────────────────────────────────────────────────────────

def test_mask_word_keeps_first_and_last():
    assert C.mask_word("shit", "*") == "s**t"
    assert C.mask_word("fucking", "#") == "f#####g"
    assert C.mask_word("ass", "*") == "a*s"


def test_mask_word_short_words_still_read_censored():
    assert C.mask_word("a", "*") == "*"
    assert C.mask_word("ok", "*") == "o*"


def test_censor_text_masks_only_whole_words():
    text, hits = C.censor_text("Scunthorpe assessment of the ass", ["ass"], "*")
    assert text == "Scunthorpe assessment of the a*s"
    assert hits == 1


def test_censor_text_case_insensitive_preserves_case():
    text, hits = C.censor_text("SHIT happens, Shit happens", ["shit"], "*")
    assert text == "S**T happens, S**t happens"
    assert hits == 2


def test_censor_text_custom_symbol():
    text, _ = C.censor_text("what the fuck", ["fuck"], "#")
    assert text == "what the f##k"


def test_censor_text_default_list_covers_variants():
    text, hits = C.censor_text("fucking bullshit, dammit", C.DEFAULT_CENSOR_WORDS, "*")
    assert "f*****g" in text and "b******t" in text and "d****t" in text
    assert hits == 3


def test_censor_text_no_words_no_change():
    text, hits = C.censor_text("perfectly clean sentence", ["shit"], "*")
    assert hits == 0 and text == "perfectly clean sentence"


def test_censor_text_cjk_entry_matches_as_substring():
    text, hits = C.censor_text("これはクソです", ["クソ"], "*")
    assert hits == 1 and "ク*" in text


def test_censor_segments_masks_text_and_words_without_mutating():
    segs = [{
        "start": 0.0, "end": 2.0, "text": "oh shit",
        "words": [
            {"start": 0.0, "end": 0.5, "word": "oh"},
            {"start": 0.5, "end": 1.0, "word": "shit"},
        ],
    }]
    out, hits = C.censor_segments(segs, ["shit"], "*")
    assert hits == 2
    assert out[0]["text"] == "oh s**t"
    assert out[0]["words"][1]["word"] == "s**t"
    # Originals untouched — the beep scan still needs the raw words.
    assert segs[0]["text"] == "oh shit"
    assert segs[0]["words"][1]["word"] == "shit"


# ── Beep intervals ──────────────────────────────────────────────────────────

def test_profane_intervals_uses_word_timestamps():
    segs = [{
        "start": 10.0, "end": 14.0, "text": "well shit that broke",
        "words": [
            {"start": 10.0, "end": 10.4, "word": "well"},
            {"start": 10.5, "end": 11.0, "word": "shit"},
            {"start": 11.1, "end": 11.5, "word": "that"},
            {"start": 11.6, "end": 12.0, "word": "broke"},
        ],
    }]
    iv = C.profane_intervals(segs, 0.0, 60.0, ["shit"])
    assert len(iv) == 1
    a, b = iv[0]
    assert a == pytest.approx(10.5 - 0.06, abs=0.01)
    assert b == pytest.approx(11.0 + 0.06, abs=0.01)


def test_profane_intervals_interpolates_without_word_times():
    # "xxxx shit" — the word sits in the back half of a 2s cue.
    segs = [{"start": 0.0, "end": 2.0, "text": "hell no shit", "words": None}]
    iv = C.profane_intervals(segs, 0.0, 10.0, ["shit"])
    assert len(iv) == 1
    a, b = iv[0]
    assert 1.0 < a < 1.5 and 1.9 < b <= 2.06 + 0.01


def test_profane_intervals_merges_adjacent_and_clamps():
    segs = [{
        "start": 0.0, "end": 4.0, "text": "shit shit",
        "words": [
            {"start": 0.0, "end": 0.5, "word": "shit"},
            {"start": 0.55, "end": 1.0, "word": "shit"},
        ],
    }]
    iv = C.profane_intervals(segs, 0.2, 0.9, ["shit"])
    # Both hits merge into one interval, clamped to the clip window.
    assert iv == [(0.2, 0.9)]


def test_profane_intervals_ignores_segments_outside_clip():
    segs = [{"start": 100.0, "end": 102.0, "text": "shit", "words": None}]
    assert C.profane_intervals(segs, 0.0, 60.0, ["shit"]) == []


# ── Output-time mapping (trim + speed parity) ───────────────────────────────

def test_map_to_output_time_identity():
    assert C.map_to_output_time([(12.0, 13.0)], 10.0, 40.0) == [(2.0, 3.0)]


def test_map_to_output_time_global_speed():
    out = C.map_to_output_time([(12.0, 14.0)], 10.0, 40.0, speed=2.0)
    assert out == [(1.0, 2.0)]


def test_map_to_output_time_per_segment_speeds():
    # Clip 0..10; segment 0-4 at 2x (outputs 2s), rest at 1x.
    segs = [{"start": 0.0, "end": 4.0, "speed": 2.0}]
    out = C.map_to_output_time([(2.0, 3.0), (5.0, 6.0)], 0.0, 10.0,
                               speed=1.0, segments=segs)
    # 2.0s → 1.0 out (2x), 3.0 → 1.5; 5.0 → 2 (seg out) + 1 (gap) = 3.0
    assert out[0] == (1.0, 1.5)
    assert out[1] == (3.0, 4.0)


def test_map_to_output_time_drops_zero_length():
    assert C.map_to_output_time([(5.0, 5.001)], 0.0, 10.0) == []


# ── The ffmpeg beep post-pass command ───────────────────────────────────────

def test_build_censor_cmd_default_beep():
    cmd = C.build_censor_audio_cmd("in.mp4", "out.mp4", [(1.0, 1.5), (3.0, 3.4)],
                                   sample_rate=48000)
    joined = " ".join(cmd)
    fc = cmd[cmd.index("-filter_complex") + 1]
    # Mute both windows on the main track…
    assert "between(t,1.000,1.500)+between(t,3.000,3.400)" in fc
    assert ":volume=0[am]" in fc
    # …one sine beep input, delayed per interval, mixed over the top.
    assert "sine=frequency=1000:sample_rate=48000" in joined
    assert "adelay=1000:all=1[b0]" in fc and "adelay=3000:all=1[b1]" in fc
    assert "amix=inputs=3:duration=first:normalize=0[aout]" in fc
    # Video untouched, audio re-encoded.
    assert "-c:v copy" in joined and "[aout]" in joined


def test_build_censor_cmd_custom_sound_loops():
    cmd = C.build_censor_audio_cmd("in.mp4", "out.mp4", [(0.5, 4.5)],
                                   beep_path="/data/config/censor_sound.mp3",
                                   sample_rate=44100)
    joined = " ".join(cmd)
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "/data/config/censor_sound.mp3" in joined
    assert "sine=" not in joined
    # Looped + resampled so a short sound covers a long interval.
    assert "aloop=loop=-1" in fc and "aresample=44100" in fc
    assert "atrim=0:4.000" in fc


# ── Settings endpoints ──────────────────────────────────────────────────────

def _client():
    app = FastAPI()
    app.include_router(S.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", str(tmp_path / "user_settings.json"))
    monkeypatch.setattr(S, "_find_env_file", lambda: None)
    monkeypatch.setattr(C, "CENSOR_SOUND_DIR", str(tmp_path / "config"))
    old = {k: getattr(cfg, k) for k in (
        "CENSOR_ENABLED_DEFAULT", "CENSOR_WORDS", "CENSOR_MASK_CHAR",
        "CENSOR_BEEP_SOUND")}
    yield
    for k, v in old.items():
        setattr(cfg, k, v)


def test_get_censor_settings_defaults():
    d = _client().get("/api/censor/settings").json()
    assert d["enabled_default"] is False
    assert d["using_default_words"] is True
    assert "fuck" in d["words"] and "shit" in d["words"]
    assert d["mask_char"] == "*"
    assert d["sound"] == "beep"
    assert d["has_custom_sound"] is False


def test_post_words_symbol_and_default_roundtrip():
    c = _client()
    d = c.post("/api/censor/settings", json={
        "enabled_default": True,
        "words": ["Frak", "  smeg  ", ""],
        "mask_char": "#",
    }).json()
    assert d["enabled_default"] is True
    assert d["words"] == ["frak", "smeg"]
    assert d["using_default_words"] is False
    assert d["mask_char"] == "#"
    assert cfg.CENSOR_WORDS == "frak,smeg"
    # Empty list resets to the built-in list via the "default" sentinel —
    # so the reset itself persists (the settings store drops empty strings).
    d = c.post("/api/censor/settings", json={"words": []}).json()
    assert d["using_default_words"] is True
    assert cfg.CENSOR_WORDS == "default"


def test_post_rejects_alnum_mask_char():
    d = _client().post("/api/censor/settings", json={"mask_char": "x"}).json()
    assert d["mask_char"] == "*", "letters would leak into the mask"


def test_sound_custom_requires_uploaded_file():
    d = _client().post("/api/censor/settings", json={"sound": "custom"}).json()
    assert d["sound"] == "beep", "custom can't stick with no uploaded file"


def test_sound_upload_switch_and_delete(tmp_path):
    c = _client()
    r = c.post("/api/censor/sound",
               files={"file": ("bleep.wav", b"RIFF....WAVE", "audio/wav")})
    assert r.status_code == 200
    d = r.json()
    assert d["sound"] == "custom" and d["has_custom_sound"] is True
    assert d["custom_sound_name"] == "censor_sound.wav"
    # Now "custom" is selectable…
    assert c.post("/api/censor/settings",
                  json={"sound": "custom"}).json()["sound"] == "custom"
    # …and deleting falls back to the beep.
    d = c.delete("/api/censor/sound").json()
    assert d["sound"] == "beep" and d["has_custom_sound"] is False


def test_sound_upload_rejects_bad_type_and_empty():
    c = _client()
    assert c.post("/api/censor/sound",
                  files={"file": ("evil.exe", b"MZ", "application/x-dos")}
                  ).status_code == 400
    assert c.post("/api/censor/sound",
                  files={"file": ("empty.wav", b"", "audio/wav")}
                  ).status_code == 400


def test_censor_keys_are_persisted():
    for key in ("CENSOR_ENABLED_DEFAULT", "CENSOR_WORDS",
                "CENSOR_MASK_CHAR", "CENSOR_BEEP_SOUND"):
        assert key in S._PERSISTABLE_KEYS


def test_export_requests_accept_censor_flag():
    from backend.models import ExportRequest, FullVideoExportRequest
    assert ExportRequest(start=0, end=5, clip_id=1).censor_enabled is None
    assert ExportRequest(start=0, end=5, clip_id=1,
                         censor_enabled=True).censor_enabled is True
    assert FullVideoExportRequest().censor_enabled is None


# ── apply_censor_beeps failure contract ─────────────────────────────────────

def test_apply_censor_beeps_raises_on_ffmpeg_failure(monkeypatch, tmp_path):
    """Shipping uncensored audio after the user toggled the censor is the one
    outcome the feature must never produce — a failed pass must raise."""
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake")

    async def main():
        with pytest.raises(RuntimeError, match="Censor beep pass failed"):
            # ffmpeg isn't in the test image → subprocess fails → raise.
            await C.apply_censor_beeps(str(vid), [(1.0, 2.0)], sample_rate=48000)
    try:
        asyncio.run(main())
    except FileNotFoundError:
        # Environments with no ffmpeg binary at all surface FileNotFoundError
        # from create_subprocess_exec — equally a loud failure, never silent.
        pass
    assert vid.read_bytes() == b"fake", "original must be untouched on failure"


def test_apply_censor_beeps_noop_without_intervals():
    async def main():
        return await C.apply_censor_beeps("/nonexistent.mp4", [])
    assert asyncio.run(main()) == 0
