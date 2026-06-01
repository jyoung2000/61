"""Task 1 — the settings router reports the EFFECTIVE (actually-loaded)
Whisper model, distinct from the configured one, so the Settings page can
show what really ran (incl. a low-VRAM downgrade).

``_effective_whisper_info`` reads ``AudioIntelligence`` class attrs out of
``sys.modules`` WITHOUT importing the heavy reframer_audio module — so we can
test it by injecting a tiny fake module, no cv2/torch needed.
"""

import sys
import types

import backend.routers.settings as S


class _FakeAI:
    _cached_model_name = None
    _last_loaded_model_name = None


def _install_fake_reframer(cached=None, last=None):
    mod = types.ModuleType("backend.services.reframer_audio")
    fake = type("AudioIntelligence", (), {})
    fake._cached_model_name = cached
    fake._last_loaded_model_name = last
    mod.AudioIntelligence = fake
    sys.modules["backend.services.reframer_audio"] = mod


def _set_selected(model, user_set=True):
    S.settings.WHISPER_MODEL = model
    S.settings.WHISPER_MODEL_USER_SET = user_set


def teardown_function(_):
    sys.modules.pop("backend.services.reframer_audio", None)


def test_nothing_loaded_yet_reports_null_effective():
    sys.modules.pop("backend.services.reframer_audio", None)
    _set_selected("large-v3-turbo", user_set=True)
    info = S._effective_whisper_info()
    assert info["whisper_model_selected"] == "large-v3-turbo"
    assert info["whisper_model_user_set"] is True
    assert info["whisper_model_effective"] is None
    assert info["whisper_model_loaded"] is False
    assert info["whisper_downgraded"] is False


def test_loaded_matching_model_is_not_downgraded():
    _install_fake_reframer(cached="medium")
    _set_selected("medium")
    info = S._effective_whisper_info()
    assert info["whisper_model_effective"] == "medium"
    assert info["whisper_model_loaded"] is True
    assert info["whisper_downgraded"] is False


def test_downgrade_is_flagged():
    # Selected turbo, but the 4 GB card actually loaded medium.
    _install_fake_reframer(cached="medium")
    _set_selected("large-v3-turbo")
    info = S._effective_whisper_info()
    assert info["whisper_model_selected"] == "large-v3-turbo"
    assert info["whisper_model_effective"] == "medium"
    assert info["whisper_downgraded"] is True
    assert info["whisper_model_loaded"] is True


def test_sticky_last_loaded_survives_vram_release():
    # After a job, _release_whisper_vram nulls _cached_model_name; the sticky
    # _last_loaded_model_name must still surface the model that ran.
    _install_fake_reframer(cached=None, last="base")
    _set_selected("large-v3-turbo")
    info = S._effective_whisper_info()
    assert info["whisper_model_effective"] == "base"     # from sticky
    assert info["whisper_model_loaded"] is False           # not resident now
    assert info["whisper_downgraded"] is True              # base != turbo


def test_whisper_models_list_includes_distil_english_variants():
    ids = {m["id"] for m in S._WHISPER_MODELS}
    for need in ("tiny", "base", "small", "medium", "large-v3",
                 "large-v3-turbo", "distil-small.en", "distil-medium.en",
                 "distil-large-v3"):
        assert need in ids, f"missing {need} from the Whisper model list"
    # The .en variants are flagged English-only for the UI label.
    by_id = {m["id"]: m for m in S._WHISPER_MODELS}
    assert by_id["distil-small.en"].get("english_only") is True
    assert by_id["distil-medium.en"].get("english_only") is True
