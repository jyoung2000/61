"""Project export / import as portable ``.clipai.zip`` bundles.

This router lets a user download an entire ClipAI project — source video,
analysis (``job.json``), the NLE timeline edits (``editor-state/``), and any
uploaded overlay media — as a single zip they can archive on their own
computer, then re-import later into a fresh job. It exists so a user can
free disk on the server (delete the job) WITHOUT losing the work, and move
projects between ClipAI instances.

Design notes
────────────
* **What a "project" is on disk.** Everything lives under
  ``/data/uploads/<job_id>/``:
    - ``job.json``            → the whole analysis + settings + transcript +
                                clips + subject_track + layout (essential)
    - ``video.<ext>``         → the source video (essential — the point)
    - ``media/`` (+ _meta)    → user-uploaded overlay assets referenced by the
                                timeline (essential)
    - ``editor-state/clip_*`` → the multi-track NLE edits per clip (essential)
    - ``glossary.json``       → custom vocabulary (kept if present)
    - ``audio.wav`` / ``frames/`` / ``demucs/`` → derivable caches (optional,
                                big; toggled by ``include_cache``)
    - derived debug/checkpoint files (render_plan.json, events.jsonl,
      reframe_trace.jsonl, checkpoint/, .content_hash, video.tmp) → skipped;
      they regenerate.
  Per-clip thumbnails live OUTSIDE the job dir in the global
  ``/data/thumbnails/<job_id>_clip*.jpg`` — bundled (small) unless disabled.
  Rendered clip outputs live in ``/data/outputs/<job_id>/`` — optional
  (``include_outputs``), since they re-render from the project.

* **The job_id rewrite is the linchpin of import.** ``job_id`` is a uuid4 and
  appears verbatim inside the project data in several places: ``file_path``,
  the browser media URLs (``/api/files/<job_id>/media/...``), and
  ``thumbnail_path``. On import we mint a NEW uuid4 (never reuse the old id —
  avoids collisions and re-assigns ownership) and do a text-level replace of
  the old id → new id across ``job.json`` and every ``editor-state/*.json``.
  Because a uuid4 is globally unique this is safe and fixes every embedded
  reference in one pass — file paths, media URLs and thumbnail refs all then
  resolve against the freshly-extracted directory.

* **Streaming, not buffering.** The source video can be multiple GB, so the
  zip is written to a temp file on disk and returned with ``FileResponse``
  (headers flush immediately; a ``BackgroundTask`` deletes the temp file after
  send). The video / wav / frames go in ``ZIP_STORED`` (already compressed —
  don't waste CPU); the JSON goes in ``ZIP_DEFLATED``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from backend import database
from backend.app.auth.deps import get_current_user
from backend.app.auth.models import User
from backend.models import JobResult
from backend.routers.jobs import _require_job_access
from backend.services.ingest import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["project-bundle"])

# ── layout constants (mirror the rest of the backend) ────────────────
UPLOADS_ROOT = "/data/uploads"
OUTPUTS_ROOT = "/data/outputs"
THUMBS_ROOT = os.environ.get("THUMBNAIL_DIR", "/data/thumbnails")

BUNDLE_FORMAT = "clipai-project"
BUNDLE_VERSION = 1
MANIFEST_NAME = "clipai_project.json"

# On-disk names that must NOT be included in an export — they are either
# transient, purely derived (regenerate on demand), or heavy debug traces.
_DERIVED_SKIP = {
    "video.tmp",
    ".content_hash",
    "render_plan.json",
    "detection_overlay.json",
    "reframe_trace.jsonl",
    "events.jsonl",
    "checkpoint",  # dir
}
# Heavy but derivable caches — included only when include_cache=True.
_CACHE_ONLY = {"audio.wav", "frames", "demucs"}

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_within(root: str, target: str) -> bool:
    """True iff ``target`` resolves to a path inside ``root`` (zip-slip guard)."""
    root_abs = os.path.realpath(root)
    tgt_abs = os.path.realpath(target)
    return tgt_abs == root_abs or tgt_abs.startswith(root_abs + os.sep)


def _source_video_name(job_dir: str) -> str | None:
    for ext in sorted(ALLOWED_EXTENSIONS):
        cand = f"video.{ext}"
        if os.path.isfile(os.path.join(job_dir, cand)):
            return cand
    # fall back to any video.* that exists
    try:
        for name in os.listdir(job_dir):
            if name.startswith("video.") and not name.endswith(".tmp"):
                return name
    except OSError:
        pass
    return None


# ═════════════════════════════════════════════════════════════════════
# EXPORT
# ═════════════════════════════════════════════════════════════════════

@router.get("/jobs/{job_id}/export")
async def export_project(
    job_id: str,
    include_cache: bool = False,
    include_outputs: bool = False,
    include_thumbnails: bool = True,
    user: User = Depends(get_current_user),
):
    """Stream the whole project as a ``<name>.clipai.zip`` bundle.

    Query params:
        include_cache      — also bundle audio.wav / frames / demucs (big,
                             but makes re-opening instant; default off).
        include_outputs    — also bundle already-rendered clips from
                             /data/outputs/<job_id> (default off — they
                             re-render from the project).
        include_thumbnails — bundle the small per-clip preview JPGs
                             (default on).
    """
    job = await _require_job_access(job_id, user)  # 404s if not owner/admin
    job_dir = os.path.join(UPLOADS_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Project files not found on disk")

    video_name = _source_video_name(job_dir)

    # Build the zip to a temp file (NOT in memory — the video may be GBs).
    tmp_dir = "/data" if os.path.isdir("/data") else None
    fd, tmp_zip = tempfile.mkstemp(suffix=".clipai.zip", dir=tmp_dir)
    os.close(fd)

    included: list[str] = []
    try:
        with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zf:

            def _add(disk_path: str, arcname: str, *, compress: bool):
                method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
                zf.write(disk_path, arcname, compress_type=method)
                included.append(arcname)

            def _add_tree(disk_root: str, arc_root: str, *, compress: bool):
                for base, _dirs, files in os.walk(disk_root):
                    for fn in files:
                        full = os.path.join(base, fn)
                        rel = os.path.relpath(full, disk_root)
                        _add(full, f"{arc_root}/{rel}", compress=compress)

            # 1) Walk the job dir, mirroring it under ``job/`` in the zip.
            for name in sorted(os.listdir(job_dir)):
                full = os.path.join(job_dir, name)
                if name in _DERIVED_SKIP:
                    continue
                if name in _CACHE_ONLY and not include_cache:
                    continue
                arc = f"job/{name}"
                # JSON + text state compresses well; media/video do not.
                json_like = name.endswith(".json") or name == "editor-state"
                if os.path.isdir(full):
                    _add_tree(full, arc, compress=json_like)
                elif os.path.isfile(full):
                    _add(full, arc, compress=json_like)

            # 2) Per-clip thumbnails (global dir, prefixed by job_id).
            if include_thumbnails and os.path.isdir(THUMBS_ROOT):
                prefix = f"{job_id}_"
                for fn in os.listdir(THUMBS_ROOT):
                    if fn.startswith(prefix):
                        _add(os.path.join(THUMBS_ROOT, fn),
                             f"thumbnails/{fn}", compress=False)

            # 3) Rendered outputs (optional).
            out_dir = os.path.join(OUTPUTS_ROOT, job_id)
            if include_outputs and os.path.isdir(out_dir):
                _add_tree(out_dir, "outputs", compress=False)

            # 4) Manifest.
            manifest = {
                "format": BUNDLE_FORMAT,
                "version": BUNDLE_VERSION,
                "app": "ClipAI",
                "exported_at": _iso_now(),
                "source_job_id": job_id,
                "filename": job.filename,
                "video_name": video_name,
                "source_sha256": getattr(job, "source_sha256", "") or "",
                "status": str(getattr(job, "status", "")),
                "duration": getattr(job, "duration", 0.0),
                "clips_count": len(getattr(job, "clips", []) or []),
                "included": sorted(included),
                "include_cache": include_cache,
                "include_outputs": include_outputs,
                "include_thumbnails": include_thumbnails,
            }
            zf.writestr(
                MANIFEST_NAME,
                json.dumps(manifest, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
    except Exception:
        try:
            os.remove(tmp_zip)
        except OSError:
            pass
        logger.exception("Project export failed for job %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to build project bundle")

    base = job.filename.rsplit(".", 1)[0] if "." in (job.filename or "") else (job.filename or job_id)
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_") or job_id
    download_name = f"{safe_base}.clipai.zip"

    logger.info("Exported project %s → %s (%d entries)", job_id, download_name, len(included))
    return FileResponse(
        tmp_zip,
        media_type="application/zip",
        filename=download_name,
        background=BackgroundTask(lambda: _quiet_remove(tmp_zip)),
    )


def _quiet_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ═════════════════════════════════════════════════════════════════════
# IMPORT
# ═════════════════════════════════════════════════════════════════════

@router.post("/jobs/import")
async def import_project(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Import a ``.clipai.zip`` bundle as a brand-new project.

    Returns the new ``job_id`` so the caller can navigate straight into the
    editor. Ownership is re-assigned to the importing user.
    """
    # 1) Spool the upload to a temp file.
    tmp_dir = "/data" if os.path.isdir("/data") else None
    fd, tmp_zip = tempfile.mkstemp(suffix=".clipai.zip", dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except Exception:
        _quiet_remove(tmp_zip)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    new_job_id = str(uuid.uuid4())
    new_job_dir = os.path.join(UPLOADS_ROOT, new_job_id)
    new_out_dir = os.path.join(OUTPUTS_ROOT, new_job_id)

    try:
        if not zipfile.is_zipfile(tmp_zip):
            raise HTTPException(status_code=400, detail="Not a valid zip file")

        with zipfile.ZipFile(tmp_zip) as zf:
            # 2) Validate the manifest.
            try:
                manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
            except KeyError:
                raise HTTPException(status_code=400, detail="Not a ClipAI project bundle (missing manifest)")
            except Exception:
                raise HTTPException(status_code=400, detail="Corrupt project manifest")

            if manifest.get("format") != BUNDLE_FORMAT:
                raise HTTPException(status_code=400, detail="Not a ClipAI project bundle")
            if int(manifest.get("version", 0)) > BUNDLE_VERSION:
                raise HTTPException(
                    status_code=400,
                    detail="Bundle was made by a newer ClipAI version; please update before importing",
                )

            old_job_id = str(manifest.get("source_job_id") or "")
            if not _UUID_RE.match(old_job_id):
                raise HTTPException(status_code=400, detail="Bundle manifest has an invalid source job id")

            os.makedirs(new_job_dir, exist_ok=True)

            # 3) Extract, routing each arc-prefix to its destination with a
            #    zip-slip guard, and renaming thumbnails to the new id.
            for info in zf.infolist():
                arc = info.filename
                if arc == MANIFEST_NAME or arc.endswith("/"):
                    continue

                if arc.startswith("job/"):
                    rel = arc[len("job/"):]
                    dest = os.path.join(new_job_dir, rel)
                    root = new_job_dir
                elif arc.startswith("outputs/"):
                    rel = arc[len("outputs/"):]
                    dest = os.path.join(new_out_dir, rel)
                    root = new_out_dir
                elif arc.startswith("thumbnails/"):
                    fn = os.path.basename(arc)
                    # rename <old_id>_clipN.jpg → <new_id>_clipN.jpg
                    if fn.startswith(f"{old_job_id}_"):
                        fn = f"{new_job_id}_{fn[len(old_job_id) + 1:]}"
                    os.makedirs(THUMBS_ROOT, exist_ok=True)
                    dest = os.path.join(THUMBS_ROOT, fn)
                    root = THUMBS_ROOT
                else:
                    continue  # unknown top-level entry — ignore

                if not _safe_within(root, dest):
                    raise HTTPException(status_code=400, detail="Bundle contains an unsafe path")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

        # 4) Rewrite job_id everywhere it is embedded (file_path, media URLs,
        #    thumbnail_path) across job.json + editor-state/*.json.
        _rewrite_job_id_in_tree(new_job_dir, old_job_id, new_job_id)

        # 5) Load the rewritten job.json, re-key + re-own, and persist so it
        #    registers in the cache / dashboard list.
        job_path = os.path.join(new_job_dir, "job.json")
        if not os.path.isfile(job_path):
            raise HTTPException(status_code=400, detail="Bundle is missing job.json")
        with open(job_path) as f:
            data = json.load(f)

        data["job_id"] = new_job_id
        data["owner_user_id"] = user.id
        data["owner_username"] = user.username
        data["updated_at"] = _iso_now()
        # Point file_path at the new source location regardless of what the
        # text-replace produced (belt and suspenders).
        video_name = manifest.get("video_name") or _source_video_name(new_job_dir)
        if video_name:
            data["file_path"] = os.path.join(new_job_dir, video_name)
            data["filename"] = data.get("filename") or manifest.get("filename") or video_name

        try:
            job = JobResult(**data)
        except Exception:
            logger.exception("Imported job.json failed validation for %s", new_job_id)
            raise HTTPException(status_code=400, detail="Bundle job.json is not a valid project")

        await database.save_job(job)

        logger.info("Imported project %s ← bundle (old id %s)", new_job_id, old_job_id)
        return {
            "job_id": new_job_id,
            "filename": job.filename,
            "status": str(job.status),
            "clips_count": len(job.clips or []),
        }

    except HTTPException:
        shutil.rmtree(new_job_dir, ignore_errors=True)
        shutil.rmtree(new_out_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(new_job_dir, ignore_errors=True)
        shutil.rmtree(new_out_dir, ignore_errors=True)
        logger.exception("Project import failed")
        raise HTTPException(status_code=500, detail="Failed to import project bundle")
    finally:
        _quiet_remove(tmp_zip)


def _rewrite_job_id_in_tree(job_dir: str, old_id: str, new_id: str) -> None:
    """Replace ``old_id`` → ``new_id`` in job.json and every editor-state JSON.

    Safe because a uuid4 is globally unique, so it never collides with any
    other substring in these files. Fixes ``file_path``, media URLs
    (``/api/files/<id>/media/...``) and ``thumbnail_path`` in one pass.
    """
    if not old_id or old_id == new_id:
        return
    targets = [os.path.join(job_dir, "job.json")]
    es_dir = os.path.join(job_dir, "editor-state")
    if os.path.isdir(es_dir):
        for fn in os.listdir(es_dir):
            if fn.endswith(".json"):
                targets.append(os.path.join(es_dir, fn))
    for path in targets:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if old_id in text:
                text = text.replace(old_id, new_id)
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, path)
        except (OSError, UnicodeDecodeError):
            continue
