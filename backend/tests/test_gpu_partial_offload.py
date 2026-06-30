"""Partial GPU offload for the 4B translation model on a small card.

A 4B-q4 model is too big to fully offload on a 4 GB GTX 1650 (forcing all
layers OOMs), but most of its layers DO fit — so we step DOWN through a
partial-offload ladder (num_gpu=32 → 24 → 16) before falling back to full CPU,
keeping the bulk of compute on the GPU instead of crawling on the CPU.
"""

import sys
import types

import httpx
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
    for name, attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                       ("anthropic", "AsyncAnthropic")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, attr, object)
            sys.modules[name] = mod


_stub_provider_sdks()

from backend.config import settings  # noqa: E402
from backend.services import local_models as LM  # noqa: E402
from backend.services import translator as T  # noqa: E402

XLATE = "qwen3:4b-instruct-2507-q4_K_M"
EDIT = "qwen2.5:3b-instruct"


# ── pure ladder ──────────────────────────────────────────────────────────────

def test_4b_gets_partial_ladder(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_PARTIAL_OFFLOAD_MIN_PARAMS_B", 3.5, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_MIDSIZE_GPU_LAYERS_START", 32, raising=False)
    ladder = LM.gpu_offload_ladder(XLATE)
    assert ladder[0] == 99            # try all-GPU first
    assert ladder[-1] == 0            # CPU is the last resort
    assert any(0 < x < 99 for x in ladder)   # has partial-GPU rungs
    assert ladder == sorted(ladder, reverse=True)   # strictly descending


def test_3b_keeps_simple_path(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_PARTIAL_OFFLOAD_MIN_PARAMS_B", 3.5, raising=False)
    # A 3B model already fits → all-GPU then CPU, no partial rungs (unchanged).
    assert LM.gpu_offload_ladder(EDIT) == [99, 0]


def test_disabled_flag_restores_legacy(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", False, raising=False)
    assert LM.gpu_offload_ladder(XLATE) == [99, 0]


# ── translator OOM step-down ─────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, status, text="", payload=None):
        self.status_code = status
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


class _FakeClient:
    """Records the num_gpu of each /api/chat call; OOMs until num_gpu drops to
    the configured threshold, then returns a normal response."""

    def __init__(self, succeed_at):
        self.succeed_at = succeed_at
        self.num_gpus = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        n = json["options"].get("num_gpu")
        self.num_gpus.append(n)
        if n is not None and n > self.succeed_at:
            return _FakeResp(500, text="CUDA error: out of memory")
        return _FakeResp(200, payload={"message": {"content": "translated"}})


@pytest.fixture(autouse=True)
def _reset_cache():
    T._GPU_LAYERS_GOOD.clear()
    yield
    T._GPU_LAYERS_GOOD.clear()


def test_translation_steps_down_to_partial_gpu(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    # All-GPU (99) OOMs; 32 fits.
    fake = _FakeClient(succeed_at=32)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)

    import asyncio
    out = asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert out == "translated"
    # It tried all-GPU first, OOM'd, then ran on a PARTIAL-GPU rung (not CPU).
    assert fake.num_gpus[0] == 99
    assert fake.num_gpus[-1] == 32 and fake.num_gpus[-1] > 0
    # The working rung is cached for next time.
    assert T._GPU_LAYERS_GOOD.get(XLATE) == 32


def test_translation_cache_skips_oom_dance(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    T._GPU_LAYERS_GOOD[XLATE] = 32          # a prior batch already found 32 works
    fake = _FakeClient(succeed_at=32)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)

    import asyncio
    out = asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert out == "translated"
    # Started at the cached rung — never re-tried the OOM'ing num_gpu=99.
    assert 99 not in fake.num_gpus
    assert fake.num_gpus[0] == 32


def test_translation_falls_to_cpu_when_no_gpu_rung_fits(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    # Every positive num_gpu OOMs → only CPU (0) succeeds.
    fake = _FakeClient(succeed_at=0)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)

    import asyncio
    out = asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert out == "translated"
    assert fake.num_gpus[-1] == 0            # last resort = CPU
    assert T._GPU_LAYERS_GOOD.get(XLATE) == 0
