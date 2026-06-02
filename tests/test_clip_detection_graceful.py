"""Tests for clip-detection graceful degradation under Replicate rate limits.

The clipper's signal-based pass always runs; the VLM/Replicate pass is an
enhancement. When Replicate rate-limits (429), the discovery must trip a circuit
breaker so it stops hammering the API and falls back to the signal-based clips,
rather than grinding every chunk through retries for many minutes.

reframer_clipper imports cleanly (its heavy deps are lazy), so no stubbing.
"""
import threading

from backend.services.reframer_clipper import (
    _is_rate_limited_error,
    ReplicateDiscoveryV3,
)


def test_is_rate_limited_error_detects_429_and_throttle():
    rate_limited = [
        "ReplicateError 429 Too Many Requests",
        "status: 429",
        "Your rate limit for creating predictions is reduced to 6 requests per minute",
        "Request was throttled.",
        "quota exceeded",
    ]
    for m in rate_limited:
        assert _is_rate_limited_error(Exception(m)) is True, m

    not_rate_limited = [
        "connection reset by peer",
        "Invalid input: unknown field 'fps'",
        "500 internal server error",
        "model not found",
    ]
    for m in not_rate_limited:
        assert _is_rate_limited_error(Exception(m)) is False, m


def test_coarse_pass_chunk_short_circuits_when_rate_limited():
    """Once the breaker is tripped, a chunk returns [] immediately without
    touching Replicate / the video (all args can be dummies)."""
    disc = ReplicateDiscoveryV3(api_key="x")
    disc._rate_limited = threading.Event()
    disc._rate_limited.set()

    called = {"n": 0}
    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("Replicate must not be called when rate-limited")
    disc._call_replicate_video = _boom  # would raise if reached

    out = disc._coarse_pass_chunk(
        None, None, None, None, [],      # replicate_sdk, model_ref, video_path, signals, transcript
        0.0, 10.0, 0.5,                  # start_s, end_s, sig_score
        "", "", ["tiktok"],              # preferred, avoid, platforms
        15, 60, 30, "",                  # min/max/ideal dur, discovery_prompt
        chunk_idx=2, n_total=6,
    )
    assert out == []
    assert called["n"] == 0  # Replicate never invoked — short-circuited at the top
