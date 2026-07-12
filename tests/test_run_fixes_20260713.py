"""Fixes from the 2026-07-12 23:42 run (build 74c3370, 41m50s).

That run PROVED the previous round (sticky num_ctx, tier-B word timings,
runaway clamp) worked — and exposed the next layer:

1. The polish auto-upgrade picked qwen2.5:14b while the Companion's live
   Ollama budget was 7 GB (whisper resident): the 14b partial-offloaded at
   90 s/batch. The downshift fired correctly, but the 14b STAYED resident
   (keep_alive 10 min), starving the downshifted 4b into partial offload
   too (~45 s/batch) — budget died at 31/62 batches, 702/925 cues raw.
   → gate the upgrade on /v1/health's vram_budget_gb + EVICT on downshift.
2. "MECA:"/"MIKA:" labels survived in cues that went through the per-cue
   untranslated-recovery path, which bypassed the output sanitizers.
3. The Compute card said face_detection=cpu (and fired the CPU warning)
   while YOLO-World ran on cuda:0 — ultralytics device index 0 is falsy.
4. The old-build Companion answers /v1/vision/health with 404; the probe
   retried every 60 s all run. A missing route can't heal — disable for
   the session and say why once.
"""

import asyncio
import inspect
import sys
import types


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


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_stub_provider_sdks()


# ── 3. Compute card: CUDA device index 0 is not "cpu" ───────────────────────

def test_detection_device_int_zero_reports_gpu():
    from backend.services.pipeline import _build_compute_summary

    class _P:
        detection_device = 0          # ultralytics CUDA index — falsy!
        detection_remote_frames = 0
        detection_local_frames = 40

    summary = _build_compute_summary(engine=None, perception=_P())
    assert summary["face_detection"]["device"] == "cuda:0"


def test_detection_device_cpu_still_reports_cpu():
    from backend.services.pipeline import _build_compute_summary

    class _P:
        detection_device = "cpu"
        detection_remote_frames = 0
        detection_local_frames = 40

    summary = _build_compute_summary(engine=None, perception=_P())
    assert summary["face_detection"]["device"] == "cpu"


# ── 1. Polish upgrade respects the Companion's live VRAM budget ─────────────

class _Host:
    url = "http://companion:11500/ollama"
    name = "4070"
    vram_total_mb = 0                 # registry doesn't know the card


class _Status:
    online = True
    models = ["qwen2.5:14b", "qwen3:4b-instruct-2507-q4_K_M"]


def _patch_registry(monkeypatch):
    from backend.services import ollama_registry as reg
    monkeypatch.setattr(reg, "primary_host", lambda: _Host())
    monkeypatch.setattr(reg, "is_local_gpu_host", lambda url: False)

    async def _probe(host, **kw):
        return _Status()

    monkeypatch.setattr(reg, "probe", _probe)


def test_upgrade_skipped_when_budget_too_small(monkeypatch):
    from backend.services import translator as T
    _patch_registry(monkeypatch)

    async def _budget():
        return 7.0                    # whisper resident → only 7 GB free

    monkeypatch.setattr(T, "_companion_vram_budget_gb", _budget)
    got = asyncio.run(T.resolve_translation_polish_model(
        "qwen3:4b-instruct-2507-q4_K_M"))
    assert got == "qwen3:4b-instruct-2507-q4_K_M"   # 14b would partial-offload


def test_upgrade_allowed_when_budget_fits(monkeypatch):
    from backend.services import translator as T
    _patch_registry(monkeypatch)

    async def _budget():
        return 11.0                   # whisper idle → nearly the whole card

    monkeypatch.setattr(T, "_companion_vram_budget_gb", _budget)
    got = asyncio.run(T.resolve_translation_polish_model(
        "qwen3:4b-instruct-2507-q4_K_M"))
    assert got == "qwen2.5:14b"


def test_unknown_budget_is_conservative(monkeypatch):
    """No health answer + no registry VRAM → a 9 GB model must not be picked
    blind (the conservative ceiling is 6 GB of weights)."""
    from backend.services import translator as T
    _patch_registry(monkeypatch)

    async def _budget():
        return 0.0

    monkeypatch.setattr(T, "_companion_vram_budget_gb", _budget)
    got = asyncio.run(T.resolve_translation_polish_model(
        "qwen3:4b-instruct-2507-q4_K_M"))
    assert got == "qwen3:4b-instruct-2507-q4_K_M"


# ── 1b. Downshift evicts the abandoned upgrade ──────────────────────────────

def test_downshift_block_fires_eviction():
    from backend.services import transcript_polisher as P
    src = inspect.getsource(P)
    i = src.find("downshifting remaining batches to")
    assert i > 0
    assert "_evict_model" in src[i:i + 1200]


def test_evict_helper_uses_keepalive_zero():
    from backend.services import transcript_polisher as P
    src = inspect.getsource(P._evict_model)
    assert '"keep_alive": 0' in src


# ── 2. Recovery path runs the output sanitizers ─────────────────────────────

def test_recovery_path_sanitizes_llm_output():
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    i = src.find("previously attempted and failed")
    assert i > 0
    window = src[i:i + 1200]
    assert "strip_invented_speaker_labels" in window
    assert "clamp_runaway_translation" in window


# ── 4. Remote vision 404 → disabled for the session ────────────────────────

def test_vision_404_disables_for_session(monkeypatch):
    from backend.services.remote_vision import RemoteVisionDetector

    det = RemoteVisionDetector()
    monkeypatch.setattr(det, "_base", lambda: "http://companion:11500")
    monkeypatch.setattr(det, "_headers", lambda: {})

    calls = {"n": 0}

    class _R:
        status_code = 404

    def _get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _R()

    import httpx
    monkeypatch.setattr(httpx, "get", _get)
    assert det.available() is False
    assert det._disabled is True
    # A dead route must not be re-probed.
    assert det.available() is False
    assert calls["n"] == 1
