"""Task 2 — the translator stops on sustained upstream rate-limiting and
fails loudly instead of silently crawling through every batch / returning the
untranslated source.

These tests stub the optional provider SDKs (``google.generativeai``, ``groq``)
that aren't installed in this environment so ``ai_orchestrator`` — and thus
``translator`` — can be imported, then exercise the pure rate-limit logic with
a fake orchestrator. No network, no models.
"""

import sys
import types

import pytest


def _stub_provider_sdks():
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    if "groq" not in sys.modules:
        gr = types.ModuleType("groq")
        gr.AsyncGroq = object
        sys.modules["groq"] = gr


_stub_provider_sdks()

from backend.services import translator as T  # noqa: E402
from backend.services.providers.base import ProviderRateLimitError  # noqa: E402
from backend.models import TranscriptSegment  # noqa: E402


def test_looks_rate_limited_matches_429_and_class():
    assert T._looks_rate_limited(ProviderRateLimitError("x"))
    assert T._looks_rate_limited(Exception("HTTP 429 Too Many Requests"))
    assert T._looks_rate_limited(Exception("upstream rate-limited"))
    assert not T._looks_rate_limited(Exception("connection reset"))
    assert not T._looks_rate_limited(Exception("invalid json"))


def test_exception_hierarchy():
    # Rate-limit is a kind of translation failure, so the pipeline can catch
    # the base class and still treat it as "did not complete".
    assert issubclass(T.TranslationRateLimitedError, T.TranslationFailedError)
    assert issubclass(T.TranslationFailedError, RuntimeError)


class _RateLimitedOrchestrator:
    """Fake orchestrator whose text_completion is always rate-limited —
    mirrors a free OpenRouter model returning sustained HTTP 429s."""

    def __init__(self):
        self.calls = 0

    def reset_circuit_breaker(self):
        pass

    async def text_completion(self, *a, **k):
        self.calls += 1
        raise ProviderRateLimitError("All providers rate-limited (HTTP 429)")


def _segs(n):
    return [
        TranscriptSegment(start=i, end=i + 1, text=f"セリフ{i}", speaker="Speaker 1")
        for i in range(n)
    ]


def test_translate_segments_aborts_fast_on_sustained_429():
    import asyncio
    orch = _RateLimitedOrchestrator()
    with pytest.raises(T.TranslationRateLimitedError):
        asyncio.run(
            T.translate_segments(_segs(200), "ja", "en", orch, batch_size=10)
        )
    # 2 attempts on the very first batch is enough to trip the guard — we must
    # NOT have ground through anywhere near all 20 batches.
    assert orch.calls <= 2, f"expected fast abort, made {orch.calls} calls"


def test_recovered_transient_429_does_not_abort():
    """A single 429 that then succeeds must NOT trip the abort (counter
    resets on success)."""
    import asyncio

    class _FlakyOnceOrchestrator:
        def __init__(self):
            self.calls = 0

        def reset_circuit_breaker(self):
            pass

        async def text_completion(self, prompt, *a, **k):
            self.calls += 1
            # Fail the very first call with a 429, succeed forever after with a
            # valid JSON array sized to the batch in the prompt.
            if self.calls == 1:
                raise ProviderRateLimitError("HTTP 429 transient")
            import json
            import re
            count = len(re.findall(r'"index"', prompt)) or 10
            return json.dumps([f"line {i}" for i in range(count)])

    orch = _FlakyOnceOrchestrator()
    out = asyncio.run(
        T.translate_segments(_segs(30), "ja", "en", orch, batch_size=10)
    )
    changed = sum(1 for t, o in zip(out, _segs(30)) if t.text != o.text)
    assert changed > 0, "translation should recover after a transient 429"
