"""Tests for clip-detection graceful degradation under Replicate rate limits.

The clipper's signal-based pass always runs; the VLM/Replicate pass is an
enhancement. When Replicate rate-limits (429), the discovery must trip a circuit
breaker so it stops hammering the API and falls back to the signal-based clips,
rather than grinding every chunk through retries for many minutes.

reframer_clipper imports cleanly (its heavy deps are lazy), so no stubbing.
"""
import threading

from types import SimpleNamespace

from backend.services.reframer_clipper import (
    _is_rate_limited_error,
    _select_clip_pool,
    ReplicateDiscoveryV3,
)


def _cand(i, verdict, score):
    return SimpleNamespace(id=i, judge_verdict=verdict, composite_score=score)


def test_clip_pool_backfills_when_judge_keeps_few():
    # The judge over-skips (2 keep, 3 skip) but we target 4 → backfill with the
    # best-scored skipped moments so the diversity pass has spread to work with.
    cands = [_cand(1, "keep", 0.9), _cand(2, "keep", 0.8),
             _cand(3, "skip", 0.7), _cand(4, "skip", 0.6), _cand(5, "skip", 0.5)]
    pool = _select_clip_pool(cands, eff_max=4)
    ids = [c.id for c in pool]
    assert 1 in ids and 2 in ids                 # judge-approved always kept
    assert 3 in ids and 4 in ids and 5 in ids    # skipped backfilled toward target
    assert ids[0] in (1, 2) and ids[1] in (1, 2)  # keepers come first


def test_clip_pool_no_backfill_when_enough_keepers():
    # 5 keepers >= target → don't pull in a high-scoring skip (respect the judge).
    cands = [_cand(i, "keep", 1.0 / i) for i in range(1, 6)] + [_cand(9, "skip", 0.99)]
    pool = _select_clip_pool(cands, eff_max=3)
    assert all(c.judge_verdict == "keep" for c in pool)
    assert 9 not in [c.id for c in pool]


def test_clip_pool_all_skipped_falls_back_to_all():
    cands = [_cand(i, "skip", 0.5) for i in range(1, 4)]
    pool = _select_clip_pool(cands, eff_max=12)
    assert len(pool) == 3


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
    """Once the breaker is tripped, a chunk returns None (retryable — it will be
    resumed) immediately, without touching Replicate / the video."""
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
    assert out is None  # retryable skip — distinguishable from a ran-but-empty []
    assert called["n"] == 0  # Replicate never invoked — short-circuited at the top


def test_coarse_pass_with_resume_recovers_rate_limited_chunks():
    """A chunk that 429s on the first (concurrent) wave is resumed sequentially
    and succeeds, so no video span is silently dropped."""
    disc = ReplicateDiscoveryV3(api_key="x")
    disc._rate_limited = threading.Event()
    disc.rate_limit_backoff_s = 0.0   # don't actually sleep in the test
    disc.rate_limit_retries = 2
    disc.chunk_workers = 3

    seen = {}
    _lock = threading.Lock()
    def fake_run(args):
        with _lock:
            seen[args] = seen.get(args, 0) + 1
            n = seen[args]
        return None if n == 1 else [f"cand-{args}"]   # rate-limited once, then succeeds

    out = disc._coarse_pass_with_resume(["a", "b", "c"], fake_run, n_to_process=3)
    assert sorted(out) == ["cand-a", "cand-b", "cand-c"]   # every span recovered
    assert all(v == 2 for v in seen.values())              # each tried twice (wave0 + resume)
    assert not disc._rate_limited.is_set()                 # breaker cleared after full recovery


def test_coarse_pass_with_resume_gives_up_after_max_retries():
    """If chunks keep getting rate-limited, the resume loop is bounded and leaves
    the breaker tripped so the refine/keyframe passes skip too."""
    disc = ReplicateDiscoveryV3(api_key="x")
    disc._rate_limited = threading.Event()
    disc.rate_limit_backoff_s = 0.0
    disc.rate_limit_retries = 2

    attempts = {"n": 0}
    _lock = threading.Lock()
    def always_limited(args):
        with _lock:
            attempts["n"] += 1
        return None

    out = disc._coarse_pass_with_resume(["a", "b"], always_limited, n_to_process=2)
    assert out == []                       # nothing recovered
    assert disc._rate_limited.is_set()     # left tripped → downstream passes skip
    assert attempts["n"] == 6              # wave0 (2) + 2 resume waves (2 each), then stop
