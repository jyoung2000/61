"""Build-identity self-update (2026-07-15).

The Update button gated purely on semver, so a from-source rebuild that reused
the same version (0.2.4 → 0.2.4) reported "up to date" forever and the new
binary never installed. The fix threads a build id (the ClipAI repo's git SHA)
from the Docker build → the installer manifest AND into the compiled binary, so
a same-version rebuild is recognised by a DIFFERENT id.

This covers the Python half: make_local_manifest stamps build_id from the env,
and the downloads router surfaces it. (The Rust decision logic — should_update —
is unit-tested in companion/src-tauri/src/proxy.rs::update_decision_tests.)
"""
import importlib.util
import json
import os
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MLM_PATH = os.path.join(_REPO, "companion", "scripts", "make_local_manifest.py")


def _load_mlm():
    spec = importlib.util.spec_from_file_location("make_local_manifest", _MLM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_stamps_build_id_from_env(tmp_path, monkeypatch):
    mlm = _load_mlm()
    (tmp_path / "ClipAI GPU Companion_0.2.4_x64-setup.exe").write_bytes(b"x" * 32)
    monkeypatch.setenv("CLIPAI_COMPANION_BUILD_ID", "abc1234")
    assert mlm.main(str(tmp_path)) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["build_id"] == "abc1234"
    assert manifest["built_from_source"] is True
    assert "windows" in manifest["platforms"]


def test_manifest_build_id_blank_without_env(tmp_path, monkeypatch):
    mlm = _load_mlm()
    (tmp_path / "ClipAI GPU Companion_0.2.4_x64-setup.exe").write_bytes(b"x" * 32)
    monkeypatch.delenv("CLIPAI_COMPANION_BUILD_ID", raising=False)
    assert mlm.main(str(tmp_path)) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    # Empty (not missing) — the Companion then falls back to the semver gate.
    assert manifest["build_id"] == ""


def test_manifest_build_id_whitespace_trimmed(tmp_path, monkeypatch):
    mlm = _load_mlm()
    (tmp_path / "ClipAI GPU Companion_0.2.4_x64-setup.exe").write_bytes(b"x" * 32)
    monkeypatch.setenv("CLIPAI_COMPANION_BUILD_ID", "  def5678\n")
    assert mlm.main(str(tmp_path)) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["build_id"] == "def5678"
