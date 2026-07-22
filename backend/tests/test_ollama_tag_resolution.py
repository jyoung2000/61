"""Ollama tag resolution: an installed 'qwen2.5:14b' must satisfy a configured
'qwen2.5:14b-instruct' (same model, different spelling) — routing to the
installed tag so /api/generate doesn't 404 and the pipeline stops silently
dropping to a small ladder model."""

import backend.services.ollama_registry as R


def test_resolve_installed_tag_instruct_to_base():
    installed = ["llava:7b", "qwen2.5:14b", "qwen2.5:3b"]
    # Configured with the -instruct spelling → routes to the installed base tag.
    assert R.resolve_installed_tag(installed, "qwen2.5:14b-instruct") == "qwen2.5:14b"
    # Configured with an explicit quant → still the installed base tag.
    assert R.resolve_installed_tag(installed, "qwen2.5:14b-instruct-q4_K_M") == "qwen2.5:14b"


def test_resolve_installed_tag_exact_keeps_requested_spelling():
    installed = ["qwen2.5:14b"]
    assert R.resolve_installed_tag(installed, "qwen2.5:14b") == "qwen2.5:14b"


def test_resolve_installed_tag_size_must_match():
    installed = ["qwen2.5:7b"]
    # A 7b never satisfies a 14b request.
    assert R.resolve_installed_tag(installed, "qwen2.5:14b-instruct") is None


def test_resolve_installed_tag_absent():
    assert R.resolve_installed_tag(["llama3.2:3b"], "qwen2.5:14b-instruct") is None


def test_model_size_key_ignores_instruct_and_quant():
    assert R._model_size_key("qwen2.5:14b") == ("qwen2.5", "14b")
    assert R._model_size_key("qwen2.5:14b-instruct") == ("qwen2.5", "14b")
    assert R._model_size_key("qwen2.5:14b-instruct-q4_K_M") == ("qwen2.5", "14b")
    assert R._model_size_key("qwen2.5:7b-instruct") == ("qwen2.5", "7b")
    # No size token → None (never equate two unrelated bare names).
    assert R._model_size_key("llava:latest") is None


def test_resolve_model_for_host_routes_instruct_to_installed_base():
    status = R.HostStatus(host_id="x", online=True,
                          models=["llava:7b", "qwen2.5:14b"])
    model, subbed = R.resolve_model_for_host(status, "qwen2.5:14b-instruct", "text")
    assert (model, subbed) == ("qwen2.5:14b", True)


def test_resolve_model_for_host_exact_unchanged():
    status = R.HostStatus(host_id="x", online=True, models=["qwen2.5:14b"])
    model, subbed = R.resolve_model_for_host(status, "qwen2.5:14b", "text")
    assert (model, subbed) == ("qwen2.5:14b", False)


def test_model_present_tolerates_instruct_suffix():
    installed = ["qwen2.5:14b", "llava:7b"]
    assert R.model_present(installed, "qwen2.5:14b-instruct") is True
    assert R.model_present(installed, "qwen2.5:14b") is True
    assert R.model_present(installed, "qwen2.5:32b-instruct") is False


# ── the friendly display-name id the model picker can store (hyphens, no colon)

def test_model_size_key_parses_friendly_display_name():
    # "Qwen2.5-14B-Instruct" (and with an ollama/ prefix) → same key as the tag.
    assert R._model_size_key("Qwen2.5-14B-Instruct") == ("qwen2.5", "14b")
    assert R._model_size_key("ollama/Qwen2.5-14B-Instruct") == ("qwen2.5", "14b")


def test_size_token_not_misread_from_base_name():
    # A base ending in "<n>b" must not be read as the size — the ":8b" wins.
    assert R._model_size_key("granite3b:8b") == ("granite3b", "8b")


def test_resolve_friendly_display_name_to_installed_tag():
    installed = ["qwen2.5:14b", "llava:7b"]
    assert R.resolve_installed_tag(installed, "ollama/Qwen2.5-14B-Instruct") == "qwen2.5:14b"
    assert R.resolve_installed_tag(installed, "Qwen2.5-14B-Instruct") == "qwen2.5:14b"
    assert R.model_present(installed, "ollama/Qwen2.5-14B-Instruct") is True


def test_resolve_friendly_name_routes_in_host():
    status = R.HostStatus(host_id="x", online=True, models=["qwen2.5:14b"])
    model, subbed = R.resolve_model_for_host(status, "ollama/Qwen2.5-14B-Instruct", "text")
    assert (model, subbed) == ("qwen2.5:14b", True)
