"""Test the masked API-key fingerprints (last-4 + length) used to confirm which
key is active without ever exposing the full secret."""

import backend.config as c


def test_fingerprint_masks_to_last4_and_length(monkeypatch):
    monkeypatch.setattr(c.settings, "OPENROUTER_API_KEY", "sk-or-v1-abcdEFGH", raising=False)
    fps = c.settings.api_key_fingerprints()
    assert fps["OpenRouter"] == "…EFGH (17 chars)"
    # The full secret is never present in the fingerprint.
    assert "abcd" not in fps["OpenRouter"]


def test_unset_key_reported_unset(monkeypatch):
    monkeypatch.setattr(c.settings, "GROQ_API_KEY", "", raising=False)
    assert c.settings.api_key_fingerprints()["Groq"] == "(unset)"


def test_short_key_masked(monkeypatch):
    monkeypatch.setattr(c.settings, "GEMINI_API_KEY", "abc", raising=False)
    assert c.settings.api_key_fingerprints()["Gemini"] == "**** (set)"


def test_covers_all_external_providers():
    keys = set(c.settings.api_key_fingerprints())
    assert keys == {
        "OpenRouter", "Anthropic", "Gemini", "Groq",
        "Replicate", "GoogleTranslate", "DeepL", "HuggingFace",
    }
