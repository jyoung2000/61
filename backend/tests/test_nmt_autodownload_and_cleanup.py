"""Unit tests for the offline-NMT auto-download + model-lifecycle work.

Covers the pure-Python parts of ``backend.services.nmt_translator`` that do
NOT need ctranslate2 / torch installed:

  * disk-space guard before a download (Task 3)
  * HF intermediate-cache redirect + cleanup after a convert (Task 3)
  * partial / aborted convert cleanup so a retry isn't blocked (Task 3)
  * Opus-MT LRU pair cap (Task 3)
  * ``auto_download_for_pair`` routing (NLLB preferred; Opus-MT when asked
    or when the pair isn't in NLLB's Flores map) (Task 1)

The convert itself is faked by injecting a stub ``ctranslate2.converters``
module so we exercise the wrapper logic without the heavy dependency.
"""

import os
import sys
import types

import pytest


# ── A stub ``backend.config`` so nmt_translator's lazy ``from backend.config
#    import settings`` works without pydantic-settings installed. ───────────
def _install_config_stub(monkeypatch, **overrides):
    mod = types.ModuleType("backend.config")
    defaults = dict(
        NMT_NLLB_MODEL="facebook/nllb-200-distilled-600M",
        NMT_OPUS_MT_TEMPLATE="Helsinki-NLP/opus-mt-{src}-{tgt}",
        NMT_MAX_OPUS_PAIRS=5,
        NMT_AUTODOWNLOAD=True,
        NMT_DEVICE="auto",
    )
    defaults.update(overrides)
    mod.settings = types.SimpleNamespace(**defaults)
    # setitem (not a bare assignment) so the REAL backend.config is restored on
    # teardown — otherwise this partial stub leaks into every later test's
    # ``from backend.config import settings`` (it lacks most real settings).
    monkeypatch.setitem(sys.modules, "backend.config", mod)
    return mod


@pytest.fixture()
def N(monkeypatch, tmp_path):
    _install_config_stub(monkeypatch)
    from backend.services import nmt_translator as N
    # Pin the models dir to a temp dir for the whole test.
    monkeypatch.setattr(N, "_models_dir", lambda: str(tmp_path))
    return N


def test_disk_guard_raises_when_low(N, monkeypatch):
    monkeypatch.setattr(N, "_free_bytes", lambda _p: 100 * 1024 ** 2)  # 100 MB
    with pytest.raises(OSError) as ei:
        N._require_free_space(N._nllb_dir("x"), N._NLLB_MIN_FREE_BYTES, "NLLB x")
    assert "disk space" in str(ei.value).lower()


def test_disk_guard_passes_when_ample(N, monkeypatch):
    monkeypatch.setattr(N, "_free_bytes", lambda _p: 50 * 1024 ** 3)  # 50 GB
    # Should not raise.
    N._require_free_space(N._nllb_dir("x"), N._NLLB_MIN_FREE_BYTES, "NLLB x")


def _install_fake_converter(produce_files=True, raise_exc=None):
    """Inject a fake ``ctranslate2.converters.TransformersConverter``.

    The fake writes a ``model.bin`` + tokenizer into the target dir (so the
    real model-present check passes) unless ``raise_exc`` is set, in which
    case it raises after partially writing — to exercise cleanup.
    """
    ct2 = types.ModuleType("ctranslate2")
    conv_mod = types.ModuleType("ctranslate2.converters")

    class _FakeConverter:
        def __init__(self, model_id):
            self.model_id = model_id

        def convert(self, target_dir, quantization="int8", force=False):
            os.makedirs(target_dir, exist_ok=True)
            # Simulate the HF cache being populated during conversion.
            cache = os.environ.get("HF_HOME")
            if cache:
                os.makedirs(cache, exist_ok=True)
                with open(os.path.join(cache, "blob"), "wb") as f:
                    f.write(b"x" * 4096)
            # Partial write, then maybe fail.
            with open(os.path.join(target_dir, "model.bin"), "wb") as f:
                f.write(b"m" * 2048)
            if raise_exc is not None:
                raise raise_exc
            if produce_files:
                with open(os.path.join(target_dir, "sentencepiece.bpe.model"), "wb") as f:
                    f.write(b"t" * 64)

    conv_mod.TransformersConverter = _FakeConverter
    ct2.converters = conv_mod
    sys.modules["ctranslate2"] = ct2
    sys.modules["ctranslate2.converters"] = conv_mod


def test_convert_success_cleans_hf_cache_and_keeps_model(N, monkeypatch):
    monkeypatch.setattr(N, "_free_bytes", lambda _p: 50 * 1024 ** 3)
    _install_fake_converter(produce_files=True)
    target = N._nllb_dir("facebook/nllb-200-distilled-600M")

    # Capture the temp HF cache dir the redirect chooses so we can assert it's
    # gone afterwards.
    seen = {}
    real_redirect = N._hf_cache_redirect

    import contextlib

    @contextlib.contextmanager
    def _spy():
        with real_redirect() as d:
            seen["dir"] = d
            yield d

    monkeypatch.setattr(N, "_hf_cache_redirect", _spy)

    N._convert_with_cleanup("facebook/nllb-200-distilled-600M", target, "NLLB test")

    assert os.path.exists(os.path.join(target, "model.bin")), "int8 model kept"
    assert not os.path.exists(seen["dir"]), "HF intermediate cache removed"
    # Env restored (no dangling HF_HOME pointing at the temp cache).
    assert os.environ.get("HF_HOME") in (None, ""), "HF_HOME restored"


def test_convert_failure_removes_partial_dir(N, monkeypatch):
    monkeypatch.setattr(N, "_free_bytes", lambda _p: 50 * 1024 ** 3)
    _install_fake_converter(raise_exc=RuntimeError("boom mid-convert"))
    target = N._opus_dir("ja", "en")

    with pytest.raises(RuntimeError):
        N._convert_with_cleanup("Helsinki-NLP/opus-mt-ja-en", target, "Opus-MT ja-en")

    assert not os.path.exists(target), "partial/aborted model dir removed for clean retry"


def test_opus_lru_cap_evicts_oldest(N):
    import time
    opus_root = os.path.join(N._models_dir(), "opus-mt")
    os.makedirs(opus_root)
    pairs = ["ja-en", "de-en", "fr-en", "es-en", "it-en", "ko-en", "zh-en"]
    for i, p in enumerate(pairs):
        d = os.path.join(opus_root, p)
        os.makedirs(d)
        with open(os.path.join(d, "model.bin"), "wb") as f:
            f.write(b"m" * 100)
        marker = os.path.join(d, ".last_used")
        with open(marker, "w") as f:
            f.write("x")
        os.utime(marker, (1000 + i, 1000 + i))  # ja-en oldest, zh-en newest

    N._enforce_opus_pair_cap()  # cap=5 from the config stub
    remaining = set(os.listdir(opus_root))
    assert len(remaining) == 5
    assert "ja-en" not in remaining and "de-en" not in remaining  # 2 oldest gone
    assert "zh-en" in remaining and "ko-en" in remaining           # newest kept


def test_auto_download_returns_existing_without_downloading(N, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(N, "pick_local_engine", lambda s, t: sentinel)
    called = {"nllb": 0, "opus": 0}
    monkeypatch.setattr(N, "ensure_nllb_downloaded", lambda *a, **k: called.__setitem__("nllb", called["nllb"] + 1))
    monkeypatch.setattr(N, "ensure_opus_mt_downloaded", lambda *a, **k: called.__setitem__("opus", called["opus"] + 1))

    assert N.auto_download_for_pair("ja", "en") is sentinel
    assert called == {"nllb": 0, "opus": 0}, "no download when model already present"


def test_auto_download_prefers_nllb_for_flores_pair(N, monkeypatch):
    # No local model the first time; engine appears after the (faked) download.
    state = {"present": False}
    monkeypatch.setattr(N, "pick_local_engine", lambda s, t: "ENGINE" if state["present"] else None)

    def _fake_nllb(*a, **k):
        state["present"] = True
    nllb_calls = []
    opus_calls = []
    monkeypatch.setattr(N, "ensure_nllb_downloaded", lambda *a, **k: (nllb_calls.append(1), _fake_nllb()))
    monkeypatch.setattr(N, "ensure_opus_mt_downloaded", lambda *a, **k: opus_calls.append(1))

    out = N.auto_download_for_pair("ja", "en")  # ja+en are both in the Flores map
    assert out == "ENGINE"
    assert nllb_calls and not opus_calls, "NLLB preferred for a Flores-supported pair"


def test_auto_download_uses_opus_for_non_flores_pair(N, monkeypatch):
    state = {"present": False}
    monkeypatch.setattr(N, "pick_local_engine", lambda s, t: "ENGINE" if state["present"] else None)
    nllb_calls = []
    opus_calls = []
    monkeypatch.setattr(N, "ensure_nllb_downloaded", lambda *a, **k: nllb_calls.append(1))

    def _fake_opus(*a, **k):
        state["present"] = True
        opus_calls.append(1)
    monkeypatch.setattr(N, "ensure_opus_mt_downloaded", _fake_opus)

    # "xx" is not in NLLB's Flores map → must route to Opus-MT.
    out = N.auto_download_for_pair("xx", "en")
    assert out == "ENGINE"
    assert opus_calls and not nllb_calls, "Opus-MT used when NLLB can't serve the pair"
