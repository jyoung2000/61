"""Tests for VRAM-churn reduction: clear_vram must keep the model that's about
to be used resident, evicting only OTHER models.

The text path used to call clear_vram() (unload everything) before every healthy
Ollama call — including the text model it was about to reuse — so qwen2.5 was
unloaded + reloaded (~2 GB) before each summary / SEO / convert / MTPE call. Now
the in-use model stays loaded across consecutive calls in a step.
"""

import asyncio
import types

import pytest

from backend.services.providers.ollama_provider import (
    OllamaProvider, _ollama_names_match,
)


# ── name matching ─────────────────────────────────────────────────────────

def test_names_match_exact_and_tolerant():
    assert _ollama_names_match("qwen2.5:3b-instruct", "qwen2.5:3b-instruct")
    assert _ollama_names_match("qwen2.5", "qwen2.5:latest")
    assert _ollama_names_match("ollama/qwen2.5:3b-instruct", "qwen2.5:3b-instruct")


def test_names_dont_match_different_size():
    assert not _ollama_names_match("qwen2.5:3b", "qwen2.5:7b")
    assert not _ollama_names_match("qwen2.5:3b-instruct", "llava:7b")
    assert not _ollama_names_match("", "qwen2.5:3b")


def test_names_match_display_name_against_installed_tag():
    # The keep-list is asked with the CALLER's spelling — settings can hold the
    # model picker's display name while /api/ps reports the installed tag. A
    # strict compare answered "different model" and the keeper evicted the model
    # it was protecting, costing a multi-GB reload on the very next call.
    assert _ollama_names_match("qwen2.5:14b", "Qwen2.5-14B-Instruct")
    assert _ollama_names_match("Qwen2.5-14B-Instruct", "qwen2.5:14b")
    assert _ollama_names_match("qwen2.5:14b", "qwen2.5:14b-instruct-q4_K_M")
    # …without letting a different size or family through.
    assert not _ollama_names_match("qwen2.5:7b", "Qwen2.5-14B-Instruct")
    assert not _ollama_names_match("qwen2.5:7b", "qwen2.5vl:7b-q4_K_M")
    assert not _ollama_names_match("llava:7b", "Qwen2.5-14B-Instruct")


# ── clear_vram(except_model=...) ──────────────────────────────────────────

class _Resp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, ps_models):
        self._ps = ps_models
        self.unloaded = []

    async def get(self, url, timeout=None):
        return _Resp(200, {"models": self._ps})

    async def post(self, url, json=None, timeout=None):
        if json and json.get("keep_alive") == 0:
            self.unloaded.append(json.get("model"))
        return _Resp(200, {})


def _fake_provider(ps_models):
    client = _FakeClient(ps_models)
    return types.SimpleNamespace(_client=client, _host="http://h"), client


def test_clear_vram_keeps_target_evicts_others():
    prov, client = _fake_provider([
        {"name": "qwen2.5:3b-instruct", "size_vram": 1800, "size": 1900},
        {"name": "llava:7b", "size_vram": 0, "size": 4000},
    ])
    asyncio.run(OllamaProvider.clear_vram(prov, except_model="qwen2.5:3b-instruct"))
    assert client.unloaded == ["llava:7b"]          # text model kept resident


def test_clear_vram_resident_target_is_a_noop():
    # Only the in-use model is loaded → nothing is unloaded (no reload churn).
    prov, client = _fake_provider([
        {"name": "qwen2.5:3b-instruct", "size_vram": 1800, "size": 1900},
    ])
    asyncio.run(OllamaProvider.clear_vram(prov, except_model="qwen2.5:3b-instruct"))
    assert client.unloaded == []


def test_clear_vram_none_evicts_everything():
    prov, client = _fake_provider([
        {"name": "qwen2.5:3b-instruct", "size_vram": 1800, "size": 1900},
        {"name": "llava:7b", "size_vram": 0, "size": 4000},
    ])
    asyncio.run(OllamaProvider.clear_vram(prov, except_model=None))
    assert set(client.unloaded) == {"qwen2.5:3b-instruct", "llava:7b"}


def test_clear_vram_tolerates_latest_tag():
    prov, client = _fake_provider([
        {"name": "qwen2.5:latest", "size_vram": 1800, "size": 1900},
    ])
    asyncio.run(OllamaProvider.clear_vram(prov, except_model="qwen2.5"))
    assert client.unloaded == []                     # kept despite :latest tag


def test_clear_vram_keeps_target_named_by_display_id():
    # The observed regression: settings held "Qwen2.5-14B-Instruct", /api/ps
    # reported "qwen2.5:14b", and the keeper evicted the 8.9 GB translation
    # model before every single batch.
    prov, client = _fake_provider([
        {"name": "qwen2.5:14b", "size_vram": 8945, "size": 8945},
    ])
    asyncio.run(OllamaProvider.clear_vram(
        prov, except_model="Qwen2.5-14B-Instruct"))
    assert client.unloaded == []


def test_clear_vram_still_evicts_a_different_size_of_the_same_family():
    prov, client = _fake_provider([
        {"name": "qwen2.5:3b", "size_vram": 1800, "size": 1900},
    ])
    asyncio.run(OllamaProvider.clear_vram(
        prov, except_model="Qwen2.5-14B-Instruct"))
    assert client.unloaded == ["qwen2.5:3b"]


def test_gpu_fraction_finds_a_model_reported_under_its_installed_tag():
    # Same mismatch on the residency probe: it returned None ("not resident")
    # for a fully-resident model, which permanently swapped the tight warm
    # timeout for the cold-load ceiling and triggered pointless defrags.
    prov, _client = _fake_provider([
        {"name": "qwen2.5:14b", "size_vram": 8945, "size": 8945},
    ])
    frac = asyncio.run(OllamaProvider.model_gpu_fraction(
        prov, "Qwen2.5-14B-Instruct"))
    assert frac == pytest.approx(1.0)


def test_gpu_fraction_reports_a_cpu_spilled_model():
    prov, _client = _fake_provider([
        {"name": "qwen2.5:14b", "size_vram": 2000, "size": 8000},
    ])
    frac = asyncio.run(OllamaProvider.model_gpu_fraction(
        prov, "Qwen2.5-14B-Instruct"))
    assert frac == pytest.approx(0.25)


def test_gpu_fraction_is_none_for_a_model_that_is_not_loaded():
    prov, _client = _fake_provider([
        {"name": "llava:7b", "size_vram": 4000, "size": 4000},
    ])
    assert asyncio.run(OllamaProvider.model_gpu_fraction(
        prov, "Qwen2.5-14B-Instruct")) is None
