"""FuguMT wired as a first-class NMT engine — a Japanese-specialised Marian
model loaded through the Opus-MT machinery in its own cache dir. These cover
the routing/variant plumbing (no ctranslate2 / model needed)."""

import pytest

from backend.services import nmt_translator as n


def test_opus_dir_subdir_separates_variants():
    opus = n._opus_dir("ja", "en")                 # default
    fugu = n._opus_dir("ja", "en", "fugumt")
    assert opus.endswith("/opus-mt/ja-en")
    assert fugu.endswith("/fugumt/ja-en")
    assert opus != fugu                            # never collide on disk


def test_translator_get_caches_by_subdir():
    a = n.OpusMTTranslator.get("ja", "en")                       # opus-mt
    b = n.OpusMTTranslator.get("ja", "en", subdir="fugumt",
                               model_template="staka/fugumt-{src}-{tgt}")
    assert a is not b                              # distinct cache entries
    assert a is n.OpusMTTranslator.get("ja", "en")  # same key → same object


def test_hf_repo_resolves_template():
    default = n.OpusMTTranslator.get("ja", "en")
    fugu = n.OpusMTTranslator.get("ja", "en", subdir="fugumt",
                                  model_template="staka/fugumt-{src}-{tgt}")
    assert default.hf_repo == "Helsinki-NLP/opus-mt-ja-en"
    assert fugu.hf_repo == "staka/fugumt-ja-en"


def test_get_marian_variant_none_when_unavailable(monkeypatch):
    # No model on disk (ctranslate2 absent in CI → is_available False); with
    # autodownload off it must return None so the caller falls back to NLLB.
    monkeypatch.setattr(n.OpusMTTranslator, "is_available", lambda self: False)
    out = n.get_marian_variant("ja", "en", "fugumt",
                               "staka/fugumt-{src}-{tgt}", autodownload=False)
    assert out is None


def test_resolve_engine_passes_fugumt_through():
    # The router must not swallow an explicit fugumt request, and AUTO should
    # prefer FuguMT for Japanese↔English.
    t = pytest.importorskip(
        "backend.services.translator",
        reason="translator's cloud-provider imports aren't available in this env")
    from backend.config import settings
    # ja↔en pair detection (either direction, tolerant of spellings).
    assert t._is_ja_en_pair("ja", "en") and t._is_ja_en_pair("en", "ja")
    assert t._is_ja_en_pair("jpn_Jpan", "eng_Latn")
    assert not t._is_ja_en_pair("ja", "fr")
    eng, prefer = settings.TRANSLATION_ENGINE, settings.NMT_PREFER_FUGUMT_JA_EN
    try:
        settings.TRANSLATION_ENGINE = "fugumt"
        assert t._resolve_translation_engine("ja", "en") == "fugumt"
        # AUTO + ja→en → fugumt when the preference is on, NOT for other pairs.
        settings.TRANSLATION_ENGINE = "auto"
        settings.NMT_PREFER_FUGUMT_JA_EN = True
        settings.DEEPL_API_KEY = ""
        settings.GOOGLE_TRANSLATE_API_KEY = ""
        assert t._resolve_translation_engine("ja", "en") == "fugumt"
        settings.NMT_PREFER_FUGUMT_JA_EN = False
        assert t._resolve_translation_engine("ja", "en") != "fugumt"
    finally:
        settings.TRANSLATION_ENGINE, settings.NMT_PREFER_FUGUMT_JA_EN = eng, prefer
