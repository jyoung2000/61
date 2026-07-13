"""Companion version handshake (the "is my Companion app up to date?" story).

An outdated Companion install fails SILENTLY — its proxy 404s the routes it
doesn't have (observed: /v1/vision/health 404 every minute, face detection
quietly stayed local). The handshake makes staleness loud:

  container job start → GET companion /v1/health (with the expected-version
  header) → compare "version" → Processing Log warning naming the fix.

The sync test is the linchpin: the backend's EXPECTED_COMPANION_VERSION must
equal every version the companion declares (Cargo.toml, tauri.conf.json,
package.json, the RELEASE tag), so a future bump can't drift.
"""

import json
import os
import re
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.companion_version import (  # noqa: E402
    EXPECTED_COMPANION_VERSION,
    is_outdated,
    outdated_message,
    parse_version,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPANION = os.path.join(REPO, "companion")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ── The repo-sync guarantee ─────────────────────────────────────────────────

def test_companion_version_sync():
    cargo = re.search(r'^version = "([^"]+)"',
                      _read(os.path.join(COMPANION, "src-tauri", "Cargo.toml")),
                      re.M).group(1)
    tauri = json.loads(_read(os.path.join(
        COMPANION, "src-tauri", "tauri.conf.json")))["version"]
    pkg = json.loads(_read(os.path.join(COMPANION, "package.json")))["version"]
    release_tag = _read(os.path.join(COMPANION, "RELEASE")).strip().splitlines()[0]

    assert cargo == EXPECTED_COMPANION_VERSION
    assert tauri == EXPECTED_COMPANION_VERSION
    assert pkg == EXPECTED_COMPANION_VERSION
    assert release_tag == f"companion-v{EXPECTED_COMPANION_VERSION}"


# ── Version comparison ──────────────────────────────────────────────────────

def test_parse_version_tolerant():
    assert parse_version("0.2.0") == (0, 2, 0)
    assert parse_version("v0.2") == (0, 2, 0)
    assert parse_version("companion-v0.2.0") == (0, 2, 0)
    assert parse_version("") == (0, 0, 0)
    assert parse_version(None) == (0, 0, 0)


def test_outdated_detection():
    assert is_outdated("0.1.0", "0.2.0") is True
    assert is_outdated("0.2.0", "0.2.0") is False
    assert is_outdated("0.3.0", "0.2.0") is False
    # Missing / garbage version = something older than the first release
    # (every Companion build has reported version in /v1/health).
    assert is_outdated("", "0.2.0") is True
    assert is_outdated(None, "0.2.0") is True


def test_outdated_message_names_the_fix():
    msg = outdated_message("0.1.0")
    assert "0.1.0" in msg
    assert EXPECTED_COMPANION_VERSION in msg
    assert "GPU Companion" in msg
    assert "Settings" in msg          # points at the in-app download


# ── Wiring pins ─────────────────────────────────────────────────────────────

def test_pipeline_checks_version_at_job_start():
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    i = src.find("ensure_companion_models(job_id)")
    assert i > 0
    assert "check_companion_version(job_id)" in src[i:i + 1000]


def test_diagnostics_exposes_version_fields():
    src = _read(os.path.join(REPO, "backend", "routers", "diagnostics.py"))
    assert '"app_version": app_version' in src
    assert '"app_outdated"' in src
    assert '"expected_app_version"' in src


def test_health_check_sends_expected_version_header():
    import inspect
    from backend.services import companion_version
    src = inspect.getsource(companion_version)
    assert "X-ClipAI-Expected-Companion" in src


def test_companion_health_reports_update_available():
    src = _read(os.path.join(COMPANION, "src-tauri", "src", "proxy.rs"))
    assert "x-clipai-expected-companion" in src
    assert '"update_available": update_available' in src
