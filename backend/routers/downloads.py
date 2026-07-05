"""GPU Companion installer downloads — served FROM the ClipAI container.

The Settings page's "GPU Companion" card downloads the Windows/.exe and
macOS/.dmg installers directly from these routes so a user on the LAN
never needs GitHub access:

  GET /api/downloads/companion/manifest      → merged manifest + sources
  GET /api/downloads/companion/{platform}    → stream the installer
                                               (windows | mac), or 302 →
                                               the GitHub Release asset
  POST /api/downloads/companion/refresh      → admin "Check for Companion
                                               updates": pull newer release
                                               assets into the persistent
                                               cache (no image rebuild)

Serve order per platform: runtime cache dir (Unraid-persistent, refreshed
without a rebuild) → baked-in image dir (Dockerfile fetch stage) → 302 to
GitHub Releases. Downloads pulled by the refresh endpoint are verified
against the release manifest's sha256 before being served to anyone.

File presence is authoritative: whichever installer FILES are in a
directory drive the buttons, whether or not a manifest.json describes
them. This lets a Windows .exe (cross-built by the from-source Docker
stage) and a macOS .dmg (built natively on a Mac) be dropped into the
same cache dir independently — neither build has to know about the
other's manifest, and neither can clobber the other's download.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/downloads", tags=["downloads"])

# Overridable for tests + non-Docker layouts.
BAKED_DIR = os.environ.get(
    "CLIPAI_COMPANION_BAKED_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "static", "companion"),
)
CACHE_DIR = os.environ.get("CLIPAI_COMPANION_CACHE_DIR", "/config/companion-cache")

GITHUB_REPO = os.environ.get("CLIPAI_COMPANION_REPO", "jyoung2000/61")
_RELEASES_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases"

PLATFORM_KEYS = {"windows": "windows", "win": "windows", "mac": "mac", "macos": "mac"}
_CONTENT_TYPES = {
    ".exe": "application/vnd.microsoft.portable-executable",
    ".msi": "application/x-msi",
    ".dmg": "application/x-apple-diskimage",
}
# Installer file extension → platform key. File presence alone decides what
# is offered, so a hand-dropped .dmg (or .exe) is served with no manifest.
_EXT_PLATFORM = {".exe": "windows", ".msi": "windows_msi", ".dmg": "mac"}
_VERSION_RE = re.compile(r"[_-](\d+\.\d+\.\d+(?:\.\d+)?)")

_refresh_state = {"active": False, "message": "", "updated": False}
_github_manifest_cache: dict = {"at": 0.0, "data": None}


def _read_manifest(directory: str) -> Optional[dict]:
    path = os.path.join(directory, "manifest.json")
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _scan_installers(directory: str) -> dict:
    """Map platform → {filename, size, path} from the installer FILES present in
    ``directory``. This — not a manifest.json — is the source of truth for what
    can be served, so independently produced installers coexist in one folder."""
    found: dict = {}
    try:
        names = sorted(os.listdir(directory))
    except Exception:
        return found
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        platform = _EXT_PLATFORM.get(ext)
        if not platform or platform in found:
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            found[platform] = {
                "filename": name,
                "size": os.path.getsize(path),
                "path": path,
            }
    return found


def _local_asset(directory: str, platform: str) -> Optional[str]:
    """Absolute path of a platform's installer file present in ``directory``."""
    entry = _scan_installers(directory).get(platform)
    return entry["path"] if entry else None


def _version_from_filename(name: str) -> str:
    m = _VERSION_RE.search(name)
    return m.group(1) if m else ""


async def _github_latest_manifest(force: bool = False) -> Optional[dict]:
    """manifest.json from the newest companion-v* GitHub Release (cached 10 min)."""
    now = time.time()
    if (not force and _github_manifest_cache["data"] is not None
            and now - _github_manifest_cache["at"] < 600):
        return _github_manifest_cache["data"]
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(_RELEASES_API, params={"per_page": 20})
            if resp.status_code != 200:
                return _github_manifest_cache["data"]
            releases = [r for r in resp.json()
                        if str(r.get("tag_name", "")).startswith("companion-v")]
            if not releases:
                return _github_manifest_cache["data"]
            latest = releases[0]
            manifest_asset = next(
                (a for a in latest.get("assets", []) if a.get("name") == "manifest.json"),
                None)
            if manifest_asset:
                mresp = await client.get(manifest_asset["browser_download_url"])
                if mresp.status_code == 200:
                    data = mresp.json()
                    _github_manifest_cache.update({"at": now, "data": data})
                    return data
            # No manifest asset (older release): synthesize one from assets.
            platforms = {}
            for asset in latest.get("assets", []):
                name = str(asset.get("name", ""))
                lower = name.lower()
                key = ("windows" if lower.endswith(".exe")
                       else "windows_msi" if lower.endswith(".msi")
                       else "mac" if lower.endswith(".dmg") else None)
                if key and key not in platforms:
                    platforms[key] = {
                        "filename": name,
                        "size": asset.get("size", 0),
                        "sha256": "",
                        "url": asset.get("browser_download_url", ""),
                    }
            data = {
                "version": str(latest.get("tag_name", "")).replace("companion-v", ""),
                "tag": latest.get("tag_name", ""),
                "published_at": latest.get("published_at", ""),
                "platforms": platforms,
            }
            _github_manifest_cache.update({"at": now, "data": data})
            return data
    except Exception as e:
        logger.debug("GitHub companion release lookup failed: %s", e)
        return _github_manifest_cache["data"]


def _merged_view(github: Optional[dict]) -> dict:
    """One manifest for the UI: newest local source per platform + source tag.

    Driven by the installer files present (``_scan_installers``); any
    manifest.json is consulted only for extra metadata (version, sha256,
    built_from_source), never to decide what exists."""
    cache_manifest = _read_manifest(CACHE_DIR)
    baked_manifest = _read_manifest(BAKED_DIR)
    cache_files = _scan_installers(CACHE_DIR)
    baked_files = _scan_installers(BAKED_DIR)
    platforms = {}
    version = ""
    for key in ("windows", "mac", "windows_msi"):
        for source, files, manifest in (
            ("cached", cache_files, cache_manifest),
            ("baked", baked_files, baked_manifest),
        ):
            f = files.get(key)
            if not f:
                continue
            entry = {"filename": f["filename"], "size": f["size"], "source": source}
            m_entry = (manifest.get("platforms") or {}).get(key) if manifest else None
            if m_entry and os.path.basename(m_entry.get("filename", "")) == f["filename"]:
                if m_entry.get("sha256"):
                    entry["sha256"] = m_entry["sha256"]
            platforms[key] = entry
            if not version:
                version = (manifest.get("version", "") if manifest else "") \
                    or _version_from_filename(f["filename"])
            break
        else:
            gh_entry = ((github or {}).get("platforms") or {}).get(key)
            if gh_entry:
                entry = dict(gh_entry)
                entry["source"] = "github-only"
                platforms[key] = entry
    if not version:
        version = ((cache_manifest or baked_manifest or github or {}).get("version", ""))
    # A from-source image-build installer (companion-builder Dockerfile
    # stage) marks its manifest — the UI explains the whisper-sidecar
    # difference vs official releases.
    built_from_source = bool(
        (cache_manifest or baked_manifest or {}).get("built_from_source", False))
    return {
        "version": version,
        "platforms": platforms,
        "built_from_source": built_from_source,
        "github_repo": GITHUB_REPO,
        "refresh": dict(_refresh_state),
    }


@router.get("/companion/manifest")
async def companion_manifest():
    """Merged installer manifest: what's available and where it's served from."""
    github = await _github_latest_manifest()
    return _merged_view(github)


@router.get("/companion/{platform}")
async def companion_download(platform: str):
    """Stream the installer for ``windows`` | ``mac`` (windows_msi also
    accepted), or 302 to the GitHub asset when it isn't stored locally."""
    key = PLATFORM_KEYS.get(platform.lower(), platform.lower())
    if key not in ("windows", "mac", "windows_msi"):
        raise HTTPException(status_code=404, detail=f"unknown platform {platform!r}")

    for directory in (CACHE_DIR, BAKED_DIR):
        path = _local_asset(directory, key)
        if path:
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()
            return FileResponse(
                path,
                media_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
                filename=filename,
            )

    github = await _github_latest_manifest()
    gh_entry = ((github or {}).get("platforms") or {}).get(key) or {}
    url = gh_entry.get("url", "")
    if url:
        return RedirectResponse(url, status_code=302)
    raise HTTPException(
        status_code=404,
        detail=("No Companion installer available yet — build one by pushing a "
                f"companion-v* tag to {GITHUB_REPO} (see companion/README.md)."),
    )


def _refresh_worker(github: dict):
    """Download newer release assets into CACHE_DIR, sha256-verified."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        platforms = github.get("platforms") or {}
        fetched = 0
        for key, entry in platforms.items():
            url = entry.get("url", "")
            filename = os.path.basename(entry.get("filename", "") or "")
            want_sha = (entry.get("sha256") or "").lower()
            if not url or not filename:
                continue
            dest = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(dest) and want_sha:
                digest = hashlib.sha256()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        digest.update(chunk)
                if digest.hexdigest().lower() == want_sha:
                    continue  # already current
            _refresh_state["message"] = f"downloading {filename}…"
            tmp = dest + ".part"
            digest = hashlib.sha256()
            with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)
                        digest.update(chunk)
            if want_sha and digest.hexdigest().lower() != want_sha:
                os.remove(tmp)
                raise RuntimeError(
                    f"sha256 mismatch for {filename} — refusing to serve it")
            os.replace(tmp, dest)
            fetched += 1
            logger.info("Companion cache: fetched %s (%s)", filename, key)
        with open(os.path.join(CACHE_DIR, "manifest.json"), "w") as f:
            json.dump(github, f, indent=2)
        _refresh_state.update({
            "message": (f"updated {fetched} installer(s) to "
                        f"v{github.get('version', '?')}" if fetched
                        else f"already current (v{github.get('version', '?')})"),
            "updated": fetched > 0,
        })
    except Exception as e:
        logger.warning("Companion cache refresh failed: %s", e)
        _refresh_state["message"] = f"refresh failed: {str(e)[:200]}"
    finally:
        _refresh_state["active"] = False


@router.post("/companion/refresh")
async def companion_refresh():
    """Admin action: pull the newest release installers into the persistent
    cache (``/config/companion-cache``) so an existing container serves new
    Companion versions without an image rebuild."""
    if _refresh_state["active"]:
        return {"status": "already_running", **_refresh_state}
    github = await _github_latest_manifest(force=True)
    if not github or not github.get("platforms"):
        return {"status": "error",
                "message": "No companion-v* release found on GitHub "
                           f"({GITHUB_REPO}) — nothing to fetch."}
    _refresh_state.update({"active": True, "message": "starting…", "updated": False})
    threading.Thread(target=_refresh_worker, args=(github,),
                     daemon=True, name="companion-cache-refresh").start()
    return {"status": "started", "target_version": github.get("version", "")}
