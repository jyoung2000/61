"""Tests for the duration-aware audio-preconditioning filter chain.

A 128-min video appeared to "hang at the face step with no VRAM": GPU frame
extraction finished in ~7 min but the concurrent audio-preconditioning ffmpeg
pass (highpass + afftdn FFT denoise + loudnorm) kept grinding on CPU, and the
faces step can't start until BOTH halves of the frame+audio stage finish.

``build_precondition_filters`` drops only the expensive ``afftdn`` step on long
tracks (keeping highpass + loudnorm) so the stall goes away while most of the
transcript-coverage benefit is retained.
"""

from backend.services.pipeline_helpers import build_precondition_filters


def test_precondition_off_returns_none():
    assert build_precondition_filters(False, 60.0) is None
    assert build_precondition_filters(False, 100000.0, denoise_max_min=45) is None


def test_short_video_keeps_full_chain_including_afftdn():
    af = build_precondition_filters(True, 10 * 60.0, denoise_max_min=45)
    assert af == "aresample=16000,highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:LRA=11:TP=-1.5"


def test_video_exactly_at_cap_keeps_afftdn():
    af = build_precondition_filters(True, 45 * 60.0, denoise_max_min=45)
    assert "afftdn" in af


def test_long_video_drops_afftdn_keeps_highpass_and_loudnorm():
    af = build_precondition_filters(True, 128 * 60.0, denoise_max_min=45)
    assert "afftdn" not in af
    assert af == "aresample=16000,highpass=f=80,loudnorm=I=-18:LRA=11:TP=-1.5"


def test_resample_to_16k_is_always_first_when_on():
    # The downsample-first speedup: filters must run at 16 kHz, not source rate.
    for dur in (5 * 60.0, 128 * 60.0):
        af = build_precondition_filters(True, dur, denoise_max_min=45)
        assert af.startswith("aresample=16000,")


def test_cap_zero_disables_the_cap_always_denoise():
    af = build_precondition_filters(True, 600 * 60.0, denoise_max_min=0)
    assert "afftdn" in af


def test_unknown_duration_keeps_full_chain():
    # duration 0 (unknown) → treated as short, full quality chain.
    af = build_precondition_filters(True, 0.0, denoise_max_min=45)
    assert "afftdn" in af
