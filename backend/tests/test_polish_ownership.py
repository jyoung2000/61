"""One card owns cloud polish: the Polish Fallback card
(SUBTITLE_POLISH_CLOUD_FALLBACK / _CLOUD_MODEL / _LOCAL_ONLY) is the single
authority on whether polish may spend cloud money and on which model. The
second in-app picker (SUBTITLE_POLISH_MODEL) is gone; the env-only legacy
value never outranks the card."""

import asyncio

import pytest

import backend.routers.settings as S
from backend.config import settings as cfg


@pytest.fixture(autouse=True)
def _restore():
    keep = (cfg.SUBTITLE_POLISH_CLOUD_FALLBACK, cfg.SUBTITLE_POLISH_LOCAL_ONLY,
            cfg.SUBTITLE_POLISH_CLOUD_MODEL, cfg.SUBTITLE_POLISH_MODEL)
    yield
    (cfg.SUBTITLE_POLISH_CLOUD_FALLBACK, cfg.SUBTITLE_POLISH_LOCAL_ONLY,
     cfg.SUBTITLE_POLISH_CLOUD_MODEL, cfg.SUBTITLE_POLISH_MODEL) = keep


@pytest.fixture(autouse=True)
def _no_persist(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", str(tmp_path / "u.json"))


def test_card_choice_drives_both_cloud_flags():
    """"auto"/pinned must open BOTH gates; "none" must close both. Before
    this, the card only set CLOUD_FALLBACK and LOCAL_ONLY (default True)
    kept suppressing every cloud route — the card's choice half-worked."""
    asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="auto")))
    assert cfg.SUBTITLE_POLISH_CLOUD_FALLBACK is True
    assert cfg.SUBTITLE_POLISH_LOCAL_ONLY is False
    assert cfg.SUBTITLE_POLISH_CLOUD_MODEL == ""

    asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="google/gemini-2.5-flash")))
    assert cfg.SUBTITLE_POLISH_CLOUD_FALLBACK is True
    assert cfg.SUBTITLE_POLISH_LOCAL_ONLY is False
    assert cfg.SUBTITLE_POLISH_CLOUD_MODEL == "google/gemini-2.5-flash"

    asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="none")))
    assert cfg.SUBTITLE_POLISH_CLOUD_FALLBACK is False
    assert cfg.SUBTITLE_POLISH_LOCAL_ONLY is True


def test_cards_pin_outranks_the_legacy_env_model():
    """A stale SUBTITLE_POLISH_MODEL used to silently override the card's
    "auto" — the dropdown lied. The card's CLOUD_MODEL now wins."""
    pytest.importorskip("cv2")  # transcript_polisher pulls in the reframer stack
    from backend.services import transcript_polisher as TP
    cfg.SUBTITLE_POLISH_MODEL = "legacy/old-pin"
    cfg.SUBTITLE_POLISH_CLOUD_MODEL = "card/choice"
    assert TP._resolve_cloud_polish_model() == "card/choice"
    # Legacy still serves as the model of last resort when the card is on auto.
    cfg.SUBTITLE_POLISH_CLOUD_MODEL = ""
    assert TP._resolve_cloud_polish_model() == "legacy/old-pin"


def test_second_picker_is_gone_from_the_settings_surface():
    assert "SUBTITLE_POLISH_MODEL" not in S._PERSISTABLE_KEYS
    fields = set(S.SaveSubtitleQualityRequest.model_fields)
    assert "subtitle_polish_model" not in fields
