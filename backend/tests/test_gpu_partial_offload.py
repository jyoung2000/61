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


# ── VRAM-aware ladder: proactive fit (don't OOM at 99 for a model that
#    can't fully offload; keep 99 when it fits) ─────────────────────────────

BIG = "qwen2.5:14b"  # ~8.4 GB q4 weights


def test_vram_ladder_unknown_falls_back_to_plain(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    # total<=0 (unknown card) → identical to the plain OOM-probe ladder.
    assert LM.gpu_offload_ladder_for_vram(BIG, 0) == LM.gpu_offload_ladder(BIG)
    assert LM.gpu_offload_ladder_for_vram(XLATE, 0) == LM.gpu_offload_ladder(XLATE)


def test_vram_ladder_keeps_99_when_model_fits(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    # A 14B (~8.4 GB) fully fits a 12 GB card → 99 first (loads fully on GPU).
    assert LM.gpu_offload_ladder_for_vram(BIG, 12.0)[0] == 99


def test_vram_ladder_drops_99_when_model_too_big(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    # A 14B does NOT fit an 8 GB card → never START at 99 (guaranteed OOM);
    # begin at a partial rung and end at CPU.
    ladder = LM.gpu_offload_ladder_for_vram(BIG, 8.0)
    assert 99 not in ladder
    assert ladder[-1] == 0
    assert any(0 < x < 99 for x in ladder)


def test_vram_ladder_cpu_only_when_no_budget(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    # A 14B against ~1.5 GB usable can't offload meaningfully → CPU-only.
    assert LM.gpu_offload_ladder_for_vram(BIG, 2.0) == [0]


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
    """Records the num_gpu (and full options) of each /api/chat call; OOMs until
    num_gpu drops to the configured threshold, then returns a normal response."""

    def __init__(self, succeed_at):
        self.succeed_at = succeed_at
        self.num_gpus = []
        self.options = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, **kw):
        n = json["options"].get("num_gpu")
        self.num_gpus.append(n)
        self.options.append(dict(json["options"]))
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
    # The working rung is cached for next time (with its timestamp).
    assert T._GPU_LAYERS_GOOD.get(XLATE)[0] == 32


def test_translation_cache_skips_oom_dance(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    import time as _time
    T._GPU_LAYERS_GOOD[XLATE] = (32, _time.monotonic())  # prior batch found 32
    fake = _FakeClient(succeed_at=32)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)

    import asyncio
    out = asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert out == "translated"
    # Started at the cached rung — never re-tried the OOM'ing num_gpu=99.
    assert 99 not in fake.num_gpus
    assert fake.num_gpus[0] == 32


def test_gpu_attempts_cap_context_cpu_keeps_full(monkeypatch):
    # The big VRAM cost is the KV cache (∝ num_ctx). GPU attempts must use the
    # small GPU context; only the CPU rung keeps the full (8192) context.
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_GPU_NUM_CTX", 2048, raising=False)
    fake = _FakeClient(succeed_at=0)        # every GPU rung OOMs → ends on CPU
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)

    import asyncio
    asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=8192))
    # Every positive-num_gpu attempt used the small context…
    for opt in fake.options:
        if opt["num_gpu"] > 0:
            assert opt["num_ctx"] == 2048
    # …and the CPU rung used the full context.
    assert fake.options[-1]["num_gpu"] == 0
    assert fake.options[-1]["num_ctx"] == 8192


def test_cpu_rung_is_never_remembered_as_a_preference(monkeypatch):
    """Measured failure: whisper held the card during batch 0, the ladder
    correctly fell to CPU — and the memo then made CPU the STARTING rung of
    every later call in the process. A 14B translated beside 10.3 GB of
    free VRAM. The CPU rung is the last resort, never the preference."""
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    fake = _FakeClient(succeed_at=0)        # every GPU rung OOMs this time
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)
    import asyncio
    asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert XLATE not in T._GPU_LAYERS_GOOD  # CPU run left NO memo
    # Next call starts from the top of the ladder again — the GPU gets its
    # chance back the moment the pressure clears.
    fake2 = _FakeClient(succeed_at=99)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake2)
    asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert fake2.num_gpus[0] > 0


def test_stale_gpu_rung_memo_reprobes_the_full_ladder(monkeypatch):
    """A partial rung probed under TEMPORARY pressure must not be paid
    forever: past the TTL the ladder starts from the top again."""
    monkeypatch.setattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True, raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://x", raising=False)
    import time as _time
    T._GPU_LAYERS_GOOD[XLATE] = (32, _time.monotonic() - T._GPU_LAYERS_GOOD_TTL_S - 1)
    fake = _FakeClient(succeed_at=99)
    monkeypatch.setattr(T.httpx, "AsyncClient", lambda *a, **k: fake)
    import asyncio
    asyncio.run(T._translate_batch_via_ollama("p", XLATE, num_ctx=4096))
    assert fake.num_gpus[0] == 99           # not the stale 32


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
    # …but the CPU rung is never memoized as a preference (see the sticky-CPU
    # failure documented on test_cpu_rung_is_never_remembered_as_a_preference).
    assert XLATE not in T._GPU_LAYERS_GOOD
