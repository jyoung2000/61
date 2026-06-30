"""A model that OOMs to CPU (e.g. qwen3:4b translation on a 4 GB card) must NOT
pin a smaller model (qwen2.5:3b editorial/SEO/summary, which fits the GPU) onto
the CPU. The force-CPU flag is scoped per-model and reset on model change."""

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


_stub_provider_sdks()

from backend.services.providers.ollama_provider import OllamaProvider  # noqa: E402

XLATE = "qwen3:4b-instruct-2507-q4_K_M"
EDIT = "qwen2.5:3b-instruct"


def _provider():
    p = OllamaProvider()
    p._gpu_available = True
    return p


def test_oom_is_scoped_to_the_model():
    p = _provider()
    p._editorial_model = XLATE
    p._note_oom_force_cpu(XLATE)
    assert p._force_cpu is True
    assert p._force_cpu_model == XLATE
    # The OOM'd model stays on CPU.
    assert p._get_num_gpu(XLATE) == 0


def test_different_model_resets_and_gets_gpu():
    p = _provider()
    p._note_oom_force_cpu(XLATE)            # qwen3 translation OOM'd
    p._editorial_model = EDIT               # pipeline switched back to editorial
    p._maybe_reset_force_cpu(EDIT)
    assert p._force_cpu is False
    assert p._force_cpu_model is None
    # qwen2.5:3b fits → forced fully onto the GPU (num_gpu=99).
    assert p._get_num_gpu(EDIT) == 99


def test_same_model_stays_forced():
    p = _provider()
    p._note_oom_force_cpu(XLATE)
    p._maybe_reset_force_cpu(XLATE)         # same model → don't reset
    assert p._force_cpu is True
    assert p._get_num_gpu(XLATE) == 0


def test_latest_suffix_matches_same_model():
    p = _provider()
    p._note_oom_force_cpu(XLATE)
    # The installed tag may carry ":latest"; it's still the same model.
    p._maybe_reset_force_cpu(XLATE + ":latest")
    assert p._force_cpu is True


def test_unknown_force_cpu_model_is_left_alone():
    # A global force-CPU with no recorded model (legacy/global signal) is not
    # reset by a model-change check.
    p = _provider()
    p._force_cpu = True
    p._force_cpu_model = None
    p._maybe_reset_force_cpu(EDIT)
    assert p._force_cpu is True
