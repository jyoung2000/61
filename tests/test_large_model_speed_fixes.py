"""Large-model translation speed fixes (2026-07-15).

The measured Gundam run finally used the 12B (batch-0 fix worked) — coherent
translation — but time exploded to ~58 min: a 12B q4 (~7.2 GB) + an 8192-ctx
fp16 KV (~2 GB) blew past the paired 4070's 9.5 GB budget → CPU spill →
60-90 s/batch translation AND "all providers failed" polish timeouts. Two fixes:
(A) lower the large-model num_ctx so the 12B stays GPU-resident, and (B) skip the
redundant post-edit when a large model already did the translation.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))


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


# ── Fix A: the large-model ctx fits the Companion budget ────────────────────

def test_large_num_ctx_fits_companion_budget():
    # 4096 (down from 8192) halves the fp16 KV so a 12B q4 stays GPU-resident on
    # a ~9.5 GB budget instead of spilling to CPU.
    assert settings.TRANSLATION_LARGE_NUM_CTX == 4096


def test_translation_plan_uses_lowered_ctx():
    # The size-aware plan for a large model now advises 4096 (the provider's
    # effective-ctx floor reads the same TRANSLATION_LARGE_NUM_CTX constant).
    from backend.services.local_models import translation_plan
    p = translation_plan("gemma3:12b-it-q4_K_M", 300, is_ollama=True, companion_parallel=3)
    assert p["num_ctx"] == 4096
    assert p["concurrency"] == 1


# ── Fix B: skip the redundant post-edit for large translation models ────────

def test_large_translation_model_flagged_for_postedit_skip(monkeypatch):
    from backend.services import pipeline as P
    monkeypatch.setattr(P, "_resolve_translation_model_override",
                        lambda o: "gemma3:12b-it-q4_K_M")
    assert P._llm_translation_model_is_large(None) is True


def test_small_translation_model_keeps_postedit(monkeypatch):
    from backend.services import pipeline as P
    monkeypatch.setattr(P, "_resolve_translation_model_override",
                        lambda o: "qwen3:4b-instruct-2507")
    assert P._llm_translation_model_is_large(None) is False


def test_missing_model_is_failsoft_false(monkeypatch):
    from backend.services import pipeline as P
    monkeypatch.setattr(P, "_resolve_translation_model_override", lambda o: "")
    assert P._llm_translation_model_is_large(None) is False

    def _boom(o):
        raise RuntimeError("resolver blew up")
    monkeypatch.setattr(P, "_resolve_translation_model_override", _boom)
    assert P._llm_translation_model_is_large(None) is False   # never raises → keep post-edit


def test_skip_flag_default_on():
    assert settings.TRANSLATION_SKIP_POSTEDIT_LARGE_MODEL is True
