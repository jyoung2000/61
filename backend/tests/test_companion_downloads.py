"""Companion installer downloads: serve order (cache → baked → GitHub
redirect), manifest merging with source labels, and sha256 verification of
the runtime refresh."""

import asyncio
import hashlib
import json
import os

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import backend.routers.downloads as D


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    baked = tmp_path / "baked"
    cache = tmp_path / "cache"
    baked.mkdir()
    cache.mkdir()
    monkeypatch.setattr(D, "BAKED_DIR", str(baked))
    monkeypatch.setattr(D, "CACHE_DIR", str(cache))
    D._github_manifest_cache.update({"at": 0.0, "data": None})
    D._refresh_state.update({"active": False, "message": "", "updated": False})
    return baked, cache


def _write_release(directory, version, platforms):
    manifest = {"version": version, "tag": f"companion-v{version}", "platforms": {}}
    for key, filename in platforms.items():
        payload = f"installer {key} {version}".encode()
        (directory / filename).write_bytes(payload)
        manifest["platforms"][key] = {
            "filename": filename,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "url": f"https://github.example/{filename}",
        }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def no_github(monkeypatch):
    async def none(force=False):
        return None
    monkeypatch.setattr(D, "_github_latest_manifest", none)


def test_manifest_reports_baked_source(dirs, no_github):
    baked, _cache = dirs
    _write_release(baked, "0.1.0", {"windows": "Companion_0.1.0.exe",
                                    "mac": "Companion_0.1.0.dmg"})
    out = asyncio.run(D.companion_manifest())
    assert out["version"] == "0.1.0"
    assert out["platforms"]["windows"]["source"] == "baked"
    assert out["platforms"]["mac"]["source"] == "baked"


def test_cache_wins_over_baked(dirs, no_github):
    baked, cache = dirs
    _write_release(baked, "0.1.0", {"windows": "Companion_0.1.0.exe"})
    _write_release(cache, "0.2.0", {"windows": "Companion_0.2.0.exe"})
    out = asyncio.run(D.companion_manifest())
    assert out["version"] == "0.2.0"
    assert out["platforms"]["windows"]["source"] == "cached"

    resp = asyncio.run(D.companion_download("windows"))
    assert resp.path.endswith("Companion_0.2.0.exe")
    assert resp.media_type == "application/vnd.microsoft.portable-executable"


def test_download_streams_baked_file(dirs, no_github):
    baked, _ = dirs
    _write_release(baked, "0.1.0", {"mac": "Companion_0.1.0.dmg"})
    app = FastAPI()
    app.include_router(D.router)
    client = TestClient(app)
    resp = client.get("/api/downloads/companion/mac")
    assert resp.status_code == 200
    assert resp.content == b"installer mac 0.1.0"
    assert "Companion_0.1.0.dmg" in resp.headers.get("content-disposition", "")


def test_download_redirects_to_github_when_not_local(dirs, monkeypatch):
    async def gh(force=False):
        return {"version": "0.3.0", "platforms": {"windows": {
            "filename": "C.exe", "url": "https://github.example/C.exe"}}}
    monkeypatch.setattr(D, "_github_latest_manifest", gh)
    resp = asyncio.run(D.companion_download("windows"))
    assert resp.status_code == 302
    assert resp.headers["location"] == "https://github.example/C.exe"


def test_download_404_when_nothing_exists(dirs, no_github):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(D.companion_download("windows"))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException):
        asyncio.run(D.companion_download("amiga"))


def test_manifest_falls_back_to_github_only(dirs, monkeypatch):
    async def gh(force=False):
        return {"version": "0.3.0", "platforms": {"mac": {
            "filename": "C.dmg", "url": "https://github.example/C.dmg",
            "sha256": "", "size": 5}}}
    monkeypatch.setattr(D, "_github_latest_manifest", gh)
    out = asyncio.run(D.companion_manifest())
    assert out["platforms"]["mac"]["source"] == "github-only"


def test_refresh_verifies_sha256(dirs, monkeypatch):
    _baked, cache = dirs
    payload = b"new installer bytes"
    good_sha = hashlib.sha256(payload).hexdigest()
    github = {"version": "0.4.0", "platforms": {"windows": {
        "filename": "Companion_0.4.0.exe", "size": len(payload),
        "sha256": good_sha, "url": "https://github.example/Companion_0.4.0.exe"}}}

    class _FakeStream:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def raise_for_status(self):
            pass
        def iter_bytes(self, n):
            yield payload

    monkeypatch.setattr(D.httpx, "stream", lambda *a, **k: _FakeStream())
    D._refresh_worker(github)
    dest = cache / "Companion_0.4.0.exe"
    assert dest.read_bytes() == payload
    assert json.loads((cache / "manifest.json").read_text())["version"] == "0.4.0"
    assert D._refresh_state["updated"] is True

    # Corrupted download (sha mismatch) must never be served.
    bad = {"version": "0.5.0", "platforms": {"windows": {
        "filename": "Companion_0.5.0.exe", "size": 1,
        "sha256": "0" * 64, "url": "https://github.example/Companion_0.5.0.exe"}}}
    D._refresh_worker(bad)
    assert not (cache / "Companion_0.5.0.exe").exists()
    assert "failed" in D._refresh_state["message"]
