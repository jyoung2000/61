"""Regression test: the clip editorial judge must fall back to the LOCAL judge
when the cloud primary hits a key-limit / billing 403.

A dead OpenRouter key (`403 Key limit exceeded`) was misclassified as a
*permanent* error, so FallbackJudge returned the error WITHOUT trying the local
qwen fallback. The judge then scored ZERO candidates and clip selection
collapsed to a couple of raw-signal picks (the "only 2 clips" bug). A cloud
key-limit must fall through to the local judge, which has no such limit.
"""

import sys
import types

# reframer_clipper lazily uses cv2; stub so the pure judge class imports.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframer_clipper import FallbackJudge  # noqa: E402


class _FakeJudge:
    def __init__(self, result):
        self._result = result
        self.calls = 0

    def judge(self, *a, **k):
        self.calls += 1
        return dict(self._result)


KEEP = {"verdict": "keep", "hook": 8, "title": "Local judge ran"}


def test_key_limit_403_falls_back_to_local():
    primary = _FakeJudge({"error": "Error code: 403 - Key limit exceeded (total limit)"})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    out = fj.judge("cand", "transcript", [], "summary")
    assert out == KEEP                 # local judge result, not the 403 error
    assert fallback.calls == 1         # fallback actually ran


def test_out_of_credits_falls_back():
    primary = _FakeJudge({"error": "402 insufficient credits — add billing"})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    assert fj.judge("c", "t", []) == KEEP
    assert fallback.calls == 1


def test_transient_429_still_falls_back():
    primary = _FakeJudge({"error": "429 rate limit exceeded"})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    assert fj.judge("c", "t", []) == KEEP


def test_primary_success_skips_fallback():
    primary = _FakeJudge({"verdict": "keep", "hook": 9})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    out = fj.judge("c", "t", [])
    assert out["hook"] == 9
    assert fallback.calls == 0         # never touched the fallback


def test_genuine_misconfig_does_not_fall_back():
    # A real config error (bad model) is still treated as permanent — don't
    # burn fallback budget masking a misconfiguration.
    primary = _FakeJudge({"error": "model not found: no such model"})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    out = fj.judge("c", "t", [])
    assert "error" in out
    assert fallback.calls == 0


def test_sticky_skips_primary_after_repeated_keylimit():
    # After STICKY_AFTER key-limit failures, later candidates skip the dead
    # primary and go straight to the local judge (no more wasted 403s).
    primary = _FakeJudge({"error": "403 Key limit exceeded"})
    fallback = _FakeJudge(KEEP)
    fj = FallbackJudge(primary, fallback)
    for _ in range(FallbackJudge.STICKY_AFTER + 2):
        assert fj.judge("c", "t", []) == KEEP
    # Primary stops being called once it's stickily skipped.
    assert primary.calls <= FallbackJudge.STICKY_AFTER
