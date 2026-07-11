"""Tests for the standalone/baseline-efficiency round.

  * Diarization overlaps the face pass in BOTH whisper modes now; the local
    path joins the diar thread before local Whisper claims the GPU.
  * Per-library CPU thread caps applied at startup (cpu_threads).
  * Opus/FuguMT picks CUDA only when VRAM is genuinely free, falls back to
    CPU on any failure, and frees the VRAM on unload.
  * Ollama compose: num_parallel 2, no CPU quota, 3.3 GiB VRAM budget.
"""

import os
import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Diarization overlap (local mode)
# ─────────────────────────────────────────────────────────────────────────────

def test_diar_concurrent_default_on():
    from backend.config import Settings
    assert Settings.model_fields["DIARIZE_CONCURRENT_WITH_FACES"].default is True


def test_diar_thread_ungated_and_joined_before_local_whisper():
    import inspect
    from backend.services import reframer_perceiver as rp
    src = inspect.getsource(rp)
    # The thread start is gated on the config flag, not on remote whisper.
    assert "DIARIZE_CONCURRENT_WITH_FACES" in src
    # The local-whisper branch joins the diar thread BEFORE try_load().
    i_join = src.find("before local Whisper loads (GPU handoff)")
    i_load = src.find("if self.audio_intel.try_load():")
    assert i_join != -1 and i_load != -1 and i_join < i_load


# ─────────────────────────────────────────────────────────────────────────────
# CPU thread caps
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def _fresh_caps(monkeypatch):
    from backend.services import cpu_threads as ct
    monkeypatch.setattr(ct, "_applied", {"cap": None})
    return ct


def test_thread_cap_auto_is_half_cores(monkeypatch, _fresh_caps):
    monkeypatch.delenv("CLIPAI_CPU_THREAD_CAP", raising=False)
    cap = _fresh_caps.recommended_thread_cap()
    assert cap == max(2, (os.cpu_count() or 4) // 2)


def test_thread_cap_explicit_override(monkeypatch, _fresh_caps):
    monkeypatch.setenv("CLIPAI_CPU_THREAD_CAP", "3")
    assert _fresh_caps.recommended_thread_cap() == 3


def test_thread_cap_disable(monkeypatch, _fresh_caps):
    monkeypatch.setenv("CLIPAI_CPU_THREAD_CAP", "-1")
    assert _fresh_caps.recommended_thread_cap() == -1
    assert _fresh_caps.apply_thread_caps() == -1


def test_apply_sets_env_defaults_but_respects_operator(monkeypatch, _fresh_caps):
    monkeypatch.setenv("CLIPAI_CPU_THREAD_CAP", "5")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "16")   # operator's explicit choice
    cap = _fresh_caps.apply_thread_caps()
    assert cap == 5
    assert os.environ["OMP_NUM_THREADS"] == "5"
    assert os.environ["MKL_NUM_THREADS"] == "16"  # untouched


def test_apply_is_idempotent(monkeypatch, _fresh_caps):
    monkeypatch.setenv("CLIPAI_CPU_THREAD_CAP", "4")
    assert _fresh_caps.apply_thread_caps() == 4
    assert _fresh_caps.apply_thread_caps() == 4


def test_main_applies_caps_at_startup():
    import inspect
    import backend.main as m
    src = inspect.getsource(m)
    assert "apply_thread_caps()" in src
    # Before the heavy imports: the call must precede the router imports.
    assert src.find("apply_thread_caps()") < src.find("from backend.routers")


# ─────────────────────────────────────────────────────────────────────────────
# Opus/FuguMT device selection
# ─────────────────────────────────────────────────────────────────────────────

class _FakeSPProcessor:
    def load(self, path):
        return None


def _fake_modules(monkeypatch, cuda_free_gb):
    """Inject fake ctranslate2 / sentencepiece / torch for load()."""
    created = {}

    class _FakeTranslator:
        def __init__(self, path, device="cpu", compute_type="int8", **kw):
            created["device"] = device
            created["compute_type"] = compute_type
            created["kwargs"] = kw

    ct2 = types.ModuleType("ctranslate2")
    ct2.Translator = _FakeTranslator
    sp = types.ModuleType("sentencepiece")
    sp.SentencePieceProcessor = _FakeSPProcessor
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_free_gb is not None,
        empty_cache=lambda: None,
        mem_get_info=lambda: (int((cuda_free_gb or 0) * 1_073_741_824),
                              4 * 1_073_741_824),
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setitem(sys.modules, "sentencepiece", sp)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return created


def _mk_model_dir(tmp_path, monkeypatch):
    from backend.services import nmt_translator as nt
    d = tmp_path / "opus"
    d.mkdir()
    (d / "model.bin").write_bytes(b"x")
    (d / "source.spm").write_bytes(b"x")
    monkeypatch.setattr(nt, "_opus_dir", lambda *a, **k: str(d))
    return d


def test_opus_load_picks_cuda_when_vram_free(tmp_path, monkeypatch):
    from backend.services.nmt_translator import OpusMTTranslator
    created = _fake_modules(monkeypatch, cuda_free_gb=2.5)
    _mk_model_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "NMT_OPUS_DEVICE", "auto", raising=False)
    t = OpusMTTranslator("ja", "en", subdir="dev-test-a")
    t.load()
    assert created["device"] == "cuda"
    assert created["compute_type"] == "int8_float16"
    assert t._device == "cuda"


def test_opus_load_stays_cpu_when_vram_tight(tmp_path, monkeypatch):
    from backend.services.nmt_translator import OpusMTTranslator
    created = _fake_modules(monkeypatch, cuda_free_gb=0.4)
    _mk_model_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "NMT_OPUS_DEVICE", "auto", raising=False)
    t = OpusMTTranslator("ja", "en", subdir="dev-test-b")
    t.load()
    assert created["device"] == "cpu"
    assert created["compute_type"] == "int8"
    assert created["kwargs"].get("inter_threads", 0) >= 1
    assert t._device == "cpu"


def test_opus_load_cuda_failure_retries_on_cpu(tmp_path, monkeypatch):
    from backend.services import nmt_translator as nt
    created = _fake_modules(monkeypatch, cuda_free_gb=2.5)
    _mk_model_dir(tmp_path, monkeypatch)
    attempts = []

    class _OOMTranslator:
        def __init__(self, path, device="cpu", compute_type="int8", **kw):
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            created["device"] = device

    sys.modules["ctranslate2"].Translator = _OOMTranslator
    monkeypatch.setattr(settings, "NMT_OPUS_DEVICE", "auto", raising=False)
    t = nt.OpusMTTranslator("ja", "en", subdir="dev-test-c")
    t.load()
    assert attempts == ["cuda", "cpu"]
    assert t._device == "cpu"


def test_opus_explicit_cpu_never_probes_cuda(tmp_path, monkeypatch):
    from backend.services.nmt_translator import OpusMTTranslator
    created = _fake_modules(monkeypatch, cuda_free_gb=3.0)
    _mk_model_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "NMT_OPUS_DEVICE", "cpu", raising=False)
    t = OpusMTTranslator("ja", "en", subdir="dev-test-d")
    t.load()
    assert created["device"] == "cpu"


def test_opus_unload_frees_cuda(tmp_path, monkeypatch):
    from backend.services.nmt_translator import OpusMTTranslator
    freed = {"n": 0}
    created = _fake_modules(monkeypatch, cuda_free_gb=2.5)
    sys.modules["torch"].cuda.empty_cache = lambda: freed.__setitem__("n", freed["n"] + 1)
    _mk_model_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "NMT_OPUS_DEVICE", "auto", raising=False)
    t = OpusMTTranslator("ja", "en", subdir="dev-test-e")
    t.load()
    assert t._device == "cuda"
    t.unload()
    assert freed["n"] >= 1
    assert t._device == "cpu" and not t._loaded


# ─────────────────────────────────────────────────────────────────────────────
# Compose: Ollama service tuning
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_service():
    import yaml
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(root, "docker-compose.yml")) as f:
        doc = yaml.safe_load(f)
    return doc["services"]["ollama"]


def test_compose_ollama_num_parallel_two():
    env = _ollama_service()["environment"]
    assert "OLLAMA_NUM_PARALLEL=2" in env


def test_compose_ollama_no_cpu_quota():
    svc = _ollama_service()
    assert "cpus" not in svc


def test_compose_ollama_vram_budget_raised():
    env = _ollama_service()["environment"]
    vram = [e for e in env if e.startswith("OLLAMA_MAX_VRAM=")]
    assert vram and int(vram[0].split("=")[1]) >= 3_500_000_000
