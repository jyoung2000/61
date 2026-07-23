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


def _installer_rank(name: str, path: str) -> tuple:
    """Sort key for choosing among several installers of one platform:
    highest embedded version wins, then the newest file. The old
    first-alphabetical pick served ``..._0.1.0_...exe`` FOREVER once a
    ``..._0.2.0_...exe`` landed beside it (0.1.0 sorts first), so every
    "update" re-downloaded the stale build."""
    ver = tuple(int(p) for p in (_version_from_filename(name) or "0").split(".")
                if p.isdigit()) or (0,)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (ver, mtime)


def _scan_installers(directory: str) -> dict:
    """Map platform → {filename, size, path} from the installer FILES present in
    ``directory``. This — not a manifest.json — is the source of truth for what
    can be served, so independently produced installers coexist in one folder.
    When several installers of the same platform coexist (an updated build
    copied next to an old one), the NEWEST version is served."""
    found: dict = {}
    try:
        names = sorted(os.listdir(directory))
    except Exception:
        return found
    for name in names:
        ext = os.path.splitext(name)[1].lower()
        platform = _EXT_PLATFORM.get(ext)
        if not platform:
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        prev = found.get(platform)
        if prev and _installer_rank(prev["filename"], prev["path"]) >= _installer_rank(name, path):
            continue
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
    build_id = ""
    for key in ("windows", "mac", "windows_msi"):
        # Newest version wins across sources (a stale cached installer used
        # to shadow a fresher baked build because "cached" was tried first).
        candidates = [
            (source, files, manifest)
            for source, files, manifest in (
                ("cached", cache_files, cache_manifest),
                ("baked", baked_files, baked_manifest),
            )
            if files.get(key)
        ]
        candidates.sort(
            key=lambda c: _installer_rank(c[1][key]["filename"], c[1][key]["path"]),
            reverse=True,
        )
        for source, files, manifest in candidates[:1]:
            f = files[key]
            entry = {"filename": f["filename"], "size": f["size"], "source": source}
            m_entry = (manifest.get("platforms") or {}).get(key) if manifest else None
            if m_entry and os.path.basename(m_entry.get("filename", "")) == f["filename"]:
                if m_entry.get("sha256"):
                    entry["sha256"] = m_entry["sha256"]
            platforms[key] = entry
            if not version:
                version = _version_from_filename(f["filename"]) \
                    or (manifest.get("version", "") if manifest else "")
            # Carry the build id from the SAME manifest that supplied the
            # winning installer — it's what the Companion's Update button
            # compares against its own build to catch same-semver rebuilds.
            if not build_id and manifest:
                build_id = (manifest.get("build_id", "") or "").strip()
            break
        else:
            gh_entry = ((github or {}).get("platforms") or {}).get(key)
            if gh_entry:
                entry = dict(gh_entry)
                entry["source"] = "github-only"
                platforms[key] = entry
    if not version:
        version = ((cache_manifest or baked_manifest or github or {}).get("version", ""))
    if not build_id:
        build_id = ((cache_manifest or baked_manifest or github or {}).get("build_id", "") or "").strip()
    # A from-source image-build installer (companion-builder Dockerfile
    # stage) marks its manifest — the UI explains the whisper-sidecar
    # difference vs official releases.
    built_from_source = bool(
        (cache_manifest or baked_manifest or {}).get("built_from_source", False))
    return {
        "version": version,
        "build_id": build_id,
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


_VISION_MODEL_NAME = "yolov8s-worldv2.pt"
_vision_model_cache: dict = {"path": None}


def _resolve_vision_model() -> Optional[str]:
    """Absolute path of the YOLO-World weight already on this server, if any.

    Checked in priority order: an explicit override, the companion cache, the
    repo model dir, and the places ultralytics caches a downloaded weight
    (the container almost always has it from prior reframe runs). Serving from
    here means the Companion never fetches the weight from GitHub itself."""
    import glob

    if _vision_model_cache["path"] and os.path.isfile(_vision_model_cache["path"]):
        return _vision_model_cache["path"]

    cands: list = []
    env = os.environ.get("CLIPAI_VISION_MODEL_PATH")
    if env:
        cands.append(env)
    cands.append(os.path.join(CACHE_DIR, _VISION_MODEL_NAME))
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cands.append(os.path.join(repo, "models", _VISION_MODEL_NAME))
    for d in (os.getcwd(), "/config", "/config/models",
              os.path.expanduser("~/.config/Ultralytics"),
              "/root/.config/Ultralytics", os.path.expanduser("~")):
        cands.append(os.path.join(d, _VISION_MODEL_NAME))
    for base in ("/config", "/root", os.getcwd()):
        try:
            cands.extend(glob.glob(os.path.join(base, "**", _VISION_MODEL_NAME),
                                   recursive=True)[:5])
        except Exception:
            pass
    for p in cands:
        try:
            if p and os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
                _vision_model_cache["path"] = p
                return p
        except OSError:
            continue
    return None


def _ensure_vision_model() -> Optional[str]:
    """Resolve the weight, or have ultralytics materialize it into the cache
    dir once (server-side, so the Companion still never touches GitHub)."""
    p = _resolve_vision_model()
    if p:
        return p
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        prev = os.getcwd()
        try:
            os.chdir(CACHE_DIR)  # ultralytics downloads into CWD
            from ultralytics import YOLO
            m = YOLO(_VISION_MODEL_NAME)
            src = getattr(m, "ckpt_path", None)
        finally:
            os.chdir(prev)
        cached = os.path.join(CACHE_DIR, _VISION_MODEL_NAME)
        if os.path.isfile(cached) and os.path.getsize(cached) > 1_000_000:
            _vision_model_cache["path"] = cached
            return cached
        if src and os.path.isfile(src):
            _vision_model_cache["path"] = src
            return src
    except Exception as e:
        logger.warning("vision model resolve/download failed: %s", e)
    return None


@router.get("/companion/vision-model")
async def companion_vision_model():
    """Stream the YOLO-World weight to the GPU Companion over the LAN so its
    from-source vision-offload install needs nothing from GitHub."""
    from starlette.concurrency import run_in_threadpool
    path = await run_in_threadpool(_ensure_vision_model)
    if not path:
        raise HTTPException(
            status_code=503,
            detail="YOLO-World weight not available on the server yet — "
                   "run one reframe so it caches, then retry")
    return FileResponse(path, media_type="application/octet-stream",
                        filename=_VISION_MODEL_NAME)


@router.get("/companion/{platform}")
async def companion_download(platform: str):
    """Stream the installer for ``windows`` | ``mac`` (windows_msi also
    accepted), or 302 to the GitHub asset when it isn't stored locally."""
    key = PLATFORM_KEYS.get(platform.lower(), platform.lower())
    if key not in ("windows", "mac", "windows_msi"):
        raise HTTPException(status_code=404, detail=f"unknown platform {platform!r}")

    # Newest wins ACROSS both locations: a stale cached copy must never
    # shadow a fresher image-baked build (the docker-cp publish flow leaves
    # old versions in the cache dir beside new ones).
    candidates = [p for p in (_local_asset(d, key) for d in (CACHE_DIR, BAKED_DIR)) if p]
    if candidates:
        path = max(candidates, key=lambda p: _installer_rank(os.path.basename(p), p))
        ext = os.path.splitext(os.path.basename(path))[1].lower()
        # Serve a CLEAN, token-safe ASCII download name. The real installer
        # name ("ClipAI GPU Companion_0.2.6_x64-setup.exe") has spaces, which
        # Starlette encodes as an RFC-5987 ``filename*=utf-8''…%20…`` header —
        # fragile across HTTP clients (curl, browsers, the Companion updater).
        # A token-safe name yields a plain ``filename="…"`` header everyone
        # parses. NOTE: this is hygiene, NOT the fix for the ``\\`` self-update
        # dialog — that was the Companion's inline ``cmd /C`` quoting (fixed in
        # v0.2.7, which now writes a .cmd file). Newer clients ignore this
        # header and save to a fixed name regardless.
        _clean = {
            ".exe": "ClipAI-GPU-Companion-Setup.exe",
            ".msi": "ClipAI-GPU-Companion-Setup.msi",
            ".dmg": "ClipAI-GPU-Companion.dmg",
        }.get(ext, "ClipAI-GPU-Companion-Setup" + ext)
        return FileResponse(
            path,
            media_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
            filename=_clean,
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


async def _probe_paired_companion() -> dict:
    """Best-effort: the paired GPU Companion's running version (and build id,
    when its build is new enough to report one) from ``/v1/health``. Never
    raises — {'paired': False} when there is no companion or it's unreachable."""
    try:
        from backend.services import ollama_registry as reg
        comp = reg.companion_host()
        if comp is None:
            return {"paired": False}
        base = reg.companion_base(comp)
        if not base:
            return {"paired": False}
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base}/v1/health",
                                 headers=dict(reg.auth_headers(comp) or {}))
        if r.status_code != 200:
            return {"paired": True, "reachable": False, "url": base}
        data = r.json() or {}
        if data.get("service") != "clipai-gpu-companion":
            return {"paired": True, "reachable": False, "url": base}
        return {
            "paired": True,
            "reachable": True,
            "url": base,
            "version": (data.get("version") or "").strip(),
            "build": (data.get("build") or "").strip(),
        }
    except Exception as e:
        logger.debug("companion probe for refresh failed: %s", e)
        return {"paired": False}


def _companion_update_available(remote: dict, served_version: str,
                                served_build: str) -> bool:
    """Same decision the Companion's own Update button makes (proxy.rs
    should_update): a strictly-newer served semver updates; a lower one never
    does; at an equal semver a DIFFERENT non-empty build id updates too."""
    from backend.services.companion_version import parse_version
    if not remote.get("reachable"):
        return False
    run_v = parse_version(remote.get("version"))
    srv_v = parse_version(served_version)
    if run_v < srv_v:
        return True
    if srv_v < run_v:
        return False
    run_b = (remote.get("build") or "").strip().lower()
    srv_b = (served_build or "").strip().lower()
    return bool(srv_b and srv_b not in ("unknown", "source")
                and run_b and run_b != srv_b)


@router.post("/companion/refresh")
async def companion_refresh():
    """Admin "Check for updates": (1) compare the PAIRED Companion's running
    version/build against what this server serves — that's the update that
    actually matters on a from-source setup; (2) when GitHub has a newer
    companion-v* release, pull it into the persistent cache. A from-source
    install with no GitHub releases is NORMAL, not an error."""
    if _refresh_state["active"]:
        return {"status": "already_running", **_refresh_state}
    github = await _github_latest_manifest(force=True)
    local = _merged_view(github)
    served_version = local.get("version", "")
    served_build = local.get("build_id", "")
    has_local = any((v or {}).get("source") in ("cached", "baked")
                    for v in (local.get("platforms") or {}).values())
    remote = await _probe_paired_companion()
    update_available = _companion_update_available(
        remote, served_version, served_build)
    companion_info = {**remote, "update_available": update_available}

    if github and github.get("platforms"):
        _refresh_state.update({"active": True, "message": "starting…",
                               "updated": False})
        threading.Thread(target=_refresh_worker, args=(github,),
                         daemon=True, name="companion-cache-refresh").start()
        return {"status": "started",
                "target_version": github.get("version", ""),
                "companion": companion_info}

    if has_local:
        srv = f"v{served_version}" + (f" (build {served_build})" if served_build else "")
        if update_available:
            msg = (f"Update available — the paired Companion at "
                   f"{remote.get('url', '?')} is running "
                   f"v{remote.get('version') or '?'} and this server hosts "
                   f"{srv}. On the GPU PC, open the Companion app and click "
                   f"Update (it downloads from this server over the LAN).")
        elif remote.get("reachable"):
            msg = (f"Paired Companion is up to date — it runs "
                   f"v{remote.get('version') or '?'} and this server hosts "
                   f"{srv}. New builds appear here after the container "
                   f"updates (bash update-all.sh).")
        else:
            msg = (f"This server hosts Companion {srv} (built from source). "
                   f"No paired Companion answered the version probe — open "
                   f"the Companion app on the GPU PC and use its Update "
                   f"button, or check pairing. (GitHub has no companion-v* "
                   f"release — normal for from-source builds.)")
        return {"status": "local", "message": msg,
                "served_version": served_version,
                "served_build": served_build,
                "companion": companion_info}

    return {"status": "error",
            "message": "No companion-v* release found on GitHub "
                       f"({GITHUB_REPO}) — nothing to fetch. The "
                       "companion-release workflow builds it: check the "
                       "repo's Actions tab — if runs fail instantly "
                       "before any step executes, GitHub is refusing to "
                       "start hosted runners for the account (billing / "
                       "spending limit / Actions permissions) and needs "
                       "fixing there first. Offline alternative: rebuild "
                       "the container with --build-arg "
                       "COMPANION_BUILD_FROM_SOURCE=1 to bake the Windows "
                       "installer locally; this card then serves it "
                       "without GitHub.",
            "companion": companion_info}


# ── Remote Companion update (push from this web UI, no one at the GPU PC) ───

@router.post("/companion/push-update")
async def companion_push_update():
    """Tell the paired Companion to self-update NOW: it downloads the installer
    this server hosts, sha256-verifies it, installs silently and relaunches.
    Progress is polled via GET /companion/push-update/status."""
    try:
        from backend.services import ollama_registry as reg
        comp = reg.companion_host()
        if comp is None:
            return {"status": "error",
                    "message": "No paired GPU Companion — pair one first."}
        base = reg.companion_base(comp)
        headers = dict(reg.auth_headers(comp) or {})
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{base}/v1/update/install", headers=headers)
    except Exception as e:
        return {"status": "error",
                "message": f"Couldn't reach the Companion to start the update: {e}"}
    if r.status_code == 404:
        # Old build — the /v1/update routes don't exist yet. One manual hop.
        return {"status": "unsupported",
                "message": "The Companion app on the GPU PC is too old for "
                           "remote updates (needs v0.2.5+). Do this ONE update "
                           "at the PC — open the Companion app and click "
                           "Update — and every update after that can be "
                           "pushed from here."}
    if r.status_code == 409:
        detail = ""
        try:
            detail = (r.json() or {}).get("error", "")
        except Exception:
            pass
        return {"status": "busy",
                "message": detail or "The Companion GPU is mid-job — retry "
                                     "when it finishes."}
    if r.status_code != 200:
        return {"status": "error",
                "message": f"Companion answered HTTP {r.status_code}."}
    return {"status": "started"}


@router.get("/companion/push-update/status")
async def companion_push_update_status():
    """Poll the remote update. States the UI renders:
    downloading (with %), verifying, launching, restarting (the app exited so
    the installer can replace it — expected offline window), failed, done."""
    local = _merged_view(None)
    served_version = local.get("version", "")
    served_build = local.get("build_id", "")
    out = {"served_version": served_version, "served_build": served_build}
    try:
        from backend.services import ollama_registry as reg
        comp = reg.companion_host()
        if comp is None:
            return {**out, "state": "no_companion"}
        base = reg.companion_base(comp)
        headers = dict(reg.auth_headers(comp) or {})
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{base}/v1/update/status", headers=headers)
            if r.status_code == 404:
                return {**out, "state": "unsupported"}
            if r.status_code != 200:
                return {**out, "state": "restarting"}
            st = r.json() or {}
            # Companion is reachable — figure out whether it's ALREADY the
            # served build (post-install relaunch = done).
            ver = (st.get("app_version") or "").strip()
            build = (st.get("app_build") or "").strip()
            if not ver:
                hr = await client.get(f"{base}/v1/health", headers=headers)
                if hr.status_code == 200:
                    hd = hr.json() or {}
                    ver = (hd.get("version") or "").strip()
                    build = (hd.get("build") or "").strip()
            up_to_date = not _companion_update_available(
                {"reachable": True, "version": ver, "build": build},
                served_version, served_build)
            state = st.get("state") or "idle"
            if state in ("idle", "launching") and up_to_date and ver:
                state = "done"
            return {**out, "state": state,
                    "progress_pct": st.get("progress_pct", 0.0),
                    "downloaded_mb": st.get("downloaded_mb", 0.0),
                    "total_mb": st.get("total_mb", 0.0),
                    "error": st.get("error", ""),
                    "companion_version": ver,
                    "companion_build": build,
                    "up_to_date": bool(up_to_date and ver)}
    except Exception:
        # Unreachable mid-update = the installer is swapping files. The UI
        # keeps polling; reappearance with the served build = done.
        return {**out, "state": "restarting"}


@router.post("/companion/vision-install")
async def companion_vision_install():
    """Tell the paired GPU Companion to install the vision (face-detection)
    offload FROM SOURCE — Python, torch, deps, and the model weight it streams
    back from us over the LAN. No GitHub. Returns immediately; poll
    ``/companion/vision-install/status`` for progress."""
    from backend.services import ollama_registry as reg
    comp = reg.companion_host()
    if comp is None:
        raise HTTPException(status_code=400, detail="No GPU Companion is paired")
    base = reg.companion_base(comp)
    headers = dict(reg.auth_headers(comp) or {})
    # Empty body: the Companion derives the model URL from the ClipAI base it
    # already talks to (this container), so we needn't know our own address.
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{base}/v1/vision/install", headers=headers, json={})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Companion unreachable: {e}")
    if r.status_code == 404:
        raise HTTPException(
            status_code=400,
            detail="This Companion build can't install the vision offload "
                   "remotely — update the Companion app first.")
    if r.status_code not in (200, 202):
        raise HTTPException(status_code=502,
                            detail=f"Companion returned {r.status_code}: {r.text[:200]}")
    return {"started": True}


@router.post("/companion/vision-uninstall")
async def companion_vision_uninstall():
    """Remove the vision offload on the paired Companion — stop the sidecar and
    delete its files so face detection returns to the fast faces-local path."""
    from backend.services import ollama_registry as reg
    comp = reg.companion_host()
    if comp is None:
        raise HTTPException(status_code=400, detail="No GPU Companion is paired")
    base = reg.companion_base(comp)
    headers = dict(reg.auth_headers(comp) or {})
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{base}/v1/vision/uninstall", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Companion unreachable: {e}")
    if r.status_code == 404:
        raise HTTPException(
            status_code=400,
            detail="This Companion build can't remove the vision offload "
                   "remotely — update the Companion app.")
    if r.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"Companion returned {r.status_code}: {r.text[:200]}")
    return {"removed": True}


@router.get("/companion/vision-install/status")
async def companion_vision_install_status():
    """Live progress of the remote vision-offload install on the Companion."""
    from backend.services import ollama_registry as reg
    comp = reg.companion_host()
    if comp is None:
        return {"state": "no_companion"}
    base = reg.companion_base(comp)
    headers = dict(reg.auth_headers(comp) or {})
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(f"{base}/v1/vision/install/status", headers=headers)
    except Exception:
        return {"state": "unreachable"}
    if r.status_code == 404:
        return {"state": "unsupported"}
    if r.status_code != 200:
        return {"state": "error", "message": f"HTTP {r.status_code}"}
    data = r.json() or {}
    data.setdefault("state", "ok")
    return data
