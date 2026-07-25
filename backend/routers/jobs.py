import asyncio
import glob
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

from backend import database
from backend.app.auth.deps import get_current_user
from backend.app.auth.models import Role, User
from backend.models import FrameData, JobStatus, SceneDescription, TranscriptSegment, WordTimestamp
from backend.services.pipeline import run_analysis, request_cancel, is_cancel_requested
from backend.services.compat_stubs import retranscribe_job
from backend.services.srt_generator import generate_srt

logger = logging.getLogger(__name__)


async def _require_job_access(job_id: str, user: User):
    """Load a job and verify the caller owns it (or is admin).

    Admins see every job (including legacy jobs with no owner recorded).
    Regular users only see their own jobs — either by matching the
    stored ``owner_user_id`` UUID, or (fallback) by matching the
    stored ``owner_username`` when the UUID no longer resolves. The
    username fallback recovers access to jobs generated before
    ``/data/auth`` was persisted as a docker volume; without it a
    restart that regenerated the admin's UUID would hide every job
    they owned from them even though the job files survive on disk.
    Anything else → 404 so we don't leak the existence of other
    users' jobs.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    owner = getattr(job, "owner_user_id", "") or ""
    owner_name = (getattr(job, "owner_username", "") or "").strip().lower()
    if user.role == Role.ADMIN:
        return job
    if owner and owner == user.id:
        return job
    if owner_name and owner_name == (user.username or "").strip().lower():
        return job
    if not owner:
        # Legacy pre-auth job: leave invisible to regular users.
        raise HTTPException(status_code=404, detail="Job not found")
    raise HTTPException(status_code=404, detail="Job not found")


def _resolve_transcript_rows(job, target):
    """Pick the transcript list a single-track edit should target.

    Editing the displayed TRANSLATED subtitles must hit ``translated_transcript``,
    not the source: the two tracks have DIFFERENT segmentation (a JA source cue
    becomes several English cues), so applying a translated-list index to the
    source list lands on a different cue (or out of range) — which is why
    transcript-tab edits "didn't take." Returns ``(rows, use_translated)``."""
    use_translated = (
        (target or "").strip().lower() == "translated"
        and bool(getattr(job, "translated_transcript", None))
    )
    return (job.translated_transcript if use_translated else job.transcript), use_translated


async def _clean_translated_rows(job):
    """The translated track SANITIZED (drop source-script + dedup) — the same
    clean cues the ``/transcripts`` panel serves.

    Subtitle DOWNLOADS must use this. The SRT/VTT/subtitle endpoints otherwise
    read the raw ``translated_transcript``, so a corrupted resume that left a
    looped / source-language union in storage shipped that straight into the
    downloaded file (observed: the same hallucinated block repeated a dozen
    times, 38% still Japanese). Sanitize is idempotent (drop + dedup + sort), so
    it's safe to apply on every read.

    NOTE: this does NOT re-run the fragment MERGE. The merge is applied ONCE at
    translate time and the merged track is what's stored; re-merging on read
    formed a non-idempotent loop that eroded finished transcripts (412 → 13).
    The stored track is already merged, so sanitizing it is all a download needs.
    Off-thread (heavy on a bloated track) and fail-soft."""
    tt = getattr(job, "translated_transcript", []) or []
    if not tt:
        return tt
    try:
        from backend.services.transcript_sanitize import sanitize_translated_transcript
        lang = getattr(job, "subtitle_language", "") or "en"
        clean, _ = await asyncio.to_thread(sanitize_translated_transcript, tt, lang)
        return clean
    except Exception:
        return tt


async def _persist_transcript_rows(job_id, job, rows, use_translated):
    """Persist an edited transcript list to the correct track."""
    if use_translated:
        await database.update_job_status(
            job_id,
            translated_transcript=[
                s.model_dump() if hasattr(s, "model_dump") else s for s in rows],
        )
    else:
        job.transcript = rows
        await database.save_job(job)


class SpeakerRenameRequest(BaseModel):
    speaker_names: dict[str, str]  # {"Speaker 1": "Eric"}

router = APIRouter(prefix="/api", tags=["jobs"])


@router.get("/jobs/pipeline-stages")
async def get_pipeline_stages(_user: User = Depends(get_current_user)):
    """Return the static PIPELINE_STAGES list so the frontend can render
    the multi-segment stage tracker without hard-coding stage definitions."""
    from backend.services.pipeline import PIPELINE_STAGES
    return {"stages": PIPELINE_STAGES}


@router.get("/jobs")
async def list_jobs(user: User = Depends(get_current_user)):
    # Admin sees all jobs (including legacy unowned ones).
    # Regular users only see their own.
    # light=True: the list only needs summary fields + clip count, never the
    # transcripts — skip validating thousands of cues per job so the dashboard
    # loads fast even with a bloated job on disk.
    if user.role == Role.ADMIN:
        jobs = await database.list_jobs(light=True)
    else:
        jobs = await database.list_jobs(
            owner_user_id=user.id,
            owner_username=user.username,
            light=True,
        )
    return [
        {
            "job_id": j.job_id,
            "filename": j.filename,
            "duration": j.duration,
            "status": j.status,
            "progress": j.progress,
            "progress_message": j.progress_message,
            "created_at": j.created_at,
            "clips_count": len(j.clips),
            "provider_used": j.provider_used,
            "file_size_mb": j.file_size_mb,
            "estimated_cost_usd": j.estimated_cost_usd,
            "owner_user_id": getattr(j, "owner_user_id", "") or "",
        }
        for j in jobs
    ]


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, user: User = Depends(get_current_user)):
    job = await _require_job_access(job_id, user)
    data = job.model_dump(mode="json")
    # Safety: ensure status is always a plain string (not enum remnant)
    if "status" in data and not isinstance(data["status"], str):
        data["status"] = str(data["status"])
    return data


@router.get("/jobs/{job_id}/events")
async def get_job_events(
    job_id: str,
    limit: int = 2000,
    user: User = Depends(get_current_user),
):
    """Durable PROCESSING LOG events for a job.

    The Analysis page hydrates its activity log from this on mount so the log is
    viewable in full after a tab reload, from a new device, or after a container
    restart — the live WebSocket only carries FUTURE events. Events are persisted
    append-only to the job dir (see ``services.job_events``); this returns the
    most-recent ``limit`` of them in chronological order.
    """
    await _require_job_access(job_id, user)
    from backend.services.job_events import load_job_events
    return {"job_id": job_id, "events": load_job_events(job_id, limit=limit)}


@router.get("/jobs/{job_id}/transcripts")
async def get_transcripts(job_id: str, user: User = Depends(get_current_user)):
    """Lightweight transcript fetch — the source + translated transcripts and the
    translation status ONLY, not the full (often multi-MB) job payload.

    The Analysis page polls this so the translated subtitles load reliably even
    when the full ``GET /jobs/{id}`` is too large/slow to complete over a tunnel
    — the failure that left the UI stuck on the source-language transcript across
    every client (incognito / multiple devices ruled out caching). Small response
    => it always lands."""
    job = await _require_job_access(job_id, user)

    def _dump(rows):
        return [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r)
                for r in (rows or [])]

    # Self-heal a translated_transcript corrupted by an interrupted run / resume
    # (duplicated cues, source-language relapse). The pipeline persists a clean
    # track; this repairs damage that happened afterwards so the panel + the
    # download never show the garbled union. Persist the repair once (subsequent
    # polls find it clean → no-op) so edit indices stay consistent with storage.
    _tt = getattr(job, "translated_transcript", []) or []
    if _tt:
        try:
            from backend.services.transcript_sanitize import sanitize_translated_transcript
            # Sanitize OFF the event loop — on a transcript bloated by a corrupted
            # run (tens of thousands of cues) the sort + per-char CJK scan is heavy
            # enough to stall the loop on every poll ("Connection lost" + the
            # transcript "loading forever"). Sanitize is DROP + DEDUP + SORT only —
            # it is idempotent, so re-running it on read converges immediately.
            _clean, _changed = await asyncio.to_thread(
                sanitize_translated_transcript,
                _tt, getattr(job, "subtitle_language", "") or "en")
            if _changed:
                # Serve the cleaned/sorted track for DISPLAY ONLY — NEVER persist
                # it on read, and do not mutate the shared in-memory job either.
                #
                # Re-saving the sanitize output on every poll was a non-idempotent
                # read→persist loop that eroded a FINISHED transcript cue-by-cue
                # (this run: 168 → 111 over one viewing session; historically
                # 412 → 13). It also fought the frontend's reverse-sync writes,
                # churning the stored track. Corruption is repaired at WRITE time
                # (the pipeline builds a clean track; the save-guard in
                # update_job_status rejects dirtier incoming saves). Reads only
                # sanitize for the response — they must leave storage untouched.
                if len(_clean) < len(_tt):
                    import logging as _lg
                    _lg.getLogger("backend.routers.jobs").warning(
                        "[%s] Sanitized translated_transcript for display only: "
                        "%d → %d cue(s) (NOT persisted)",
                        job_id, len(_tt), len(_clean))
                _tt = _clean   # served below; job + storage are left as-is
        except Exception:
            pass

    # Dump both tracks OFF the event loop — model_dump per cue × thousands of
    # cues (each with per-word timestamps) is heavy enough to stall the loop on
    # every poll for a bloated transcript. ``_tt`` is the display-sanitized
    # translated track (storage itself is left untouched — see above).
    # ``raw_transcript`` (the direct pre-polish Whisper output) rides this poll
    # so the completed-job "Download raw transcript" button gets its data (the UI
    # refreshes from THIS lightweight endpoint, not the heavy full GET). It's the
    # largest single track (per-word timestamps) and only the completed view
    # consumes it, so we ONLY include it once the job is terminal — running polls
    # stay small, and the status->complete re-fetch delivers it exactly once.
    _terminal = getattr(job, "status", None) in (JobStatus.COMPLETE, JobStatus.FAILED)
    _src_rows, _tt_rows, _raw_rows, _traw_rows = await asyncio.to_thread(
        lambda: (_dump(getattr(job, "transcript", [])),
                 _dump(_tt),
                 _dump(getattr(job, "raw_transcript", [])) if _terminal else [],
                 _dump(getattr(job, "translated_raw_transcript", [])) if _terminal else []))

    return {
        "job_id": job_id,
        "status": str(getattr(job, "status", "") or ""),
        "translation_status": getattr(job, "translation_status", None),
        "subtitle_language": getattr(job, "subtitle_language", "") or "",
        "language": getattr(job, "language", "") or "",
        "transcript": _src_rows,
        "translated_transcript": _tt_rows,
        "raw_transcript": _raw_rows,
        # Translated-but-unpolished draft (raw machine translation, before the AI
        # post-edit) — powers the "raw" transcript download on translated jobs.
        "translated_raw_transcript": _traw_rows,
        # The video summary rides this lightweight poll too. It is persisted
        # mid-pipeline but otherwise only reaches the UI via the full (often
        # multi-MB) GET /jobs/{id} — the very request too large/slow to land over
        # a tunnel — leaving the Summary tab stuck on "Generating summary…" even
        # though the summary exists on disk. Small payload => it always lands.
        "summary": (job.summary.model_dump(mode="json")
                    if getattr(job, "summary", None) is not None
                    and hasattr(job.summary, "model_dump")
                    else getattr(job, "summary", None)),
        "speaker_names": getattr(job, "speaker_names", {}) or {},
        "speaker_colors": getattr(job, "speaker_colors", {}) or {},
    }


# Per-job single-flight lock so a user can't fire ten concurrent
# retranscribe jobs on the same video and blow the GPU budget. The
# lock is keyed by job_id and lives as long as the process — releases
# automatically when the task finishes.
_retranscribe_locks: dict[str, asyncio.Lock] = {}
# Track the in-flight task per job so we can detect "already running"
# even after we've handed control back to the HTTP client.
_retranscribe_tasks: dict[str, asyncio.Task] = {}


def _get_retranscribe_lock(job_id: str) -> asyncio.Lock:
    lock = _retranscribe_locks.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        _retranscribe_locks[job_id] = lock
    return lock


async def _run_retranscribe_in_background(job_id: str) -> None:
    """Background worker for ``POST /jobs/{job_id}/retranscribe``.

    Whisper on full-length audio routinely takes minutes. Holding the
    HTTP request open for that long means: (a) reverse proxies time
    out at 60–120s and the user sees a generic "request failed"; (b)
    the response is a long-lived ``StreamingResponse`` chain that re-
    enters every middleware on each chunk, which historically tripped
    the access-log filter and corrupted the body. Running in the
    background lets the endpoint return immediately and the existing
    job-status polling / WebSocket picks up progress + the new
    transcript when ``update_job_status`` writes them.
    """
    lock = _get_retranscribe_lock(job_id)
    async with lock:
        try:
            summary = await retranscribe_job(job_id)
            logger.info(
                "[%s] Retranscribe finished: before=%d after=%d task=%s",
                job_id,
                int(summary.get("segments_before", 0) or 0),
                int(summary.get("segments_after", 0) or 0),
                summary.get("task", ""),
            )
        except FileNotFoundError as e:
            logger.warning("[%s] Retranscribe aborted: %s", job_id, e)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe aborted: {e}",
                )
            except Exception:
                pass
        except ValueError as e:
            logger.warning("[%s] Retranscribe rejected: %s", job_id, e)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe rejected: {e}",
                )
            except Exception:
                pass
        except Exception as e:
            logger.exception("[%s] Retranscribe crashed", job_id)
            try:
                await database.update_job_status(
                    job_id,
                    status=JobStatus.COMPLETE,
                    progress_message=f"Retranscribe failed: {e}",
                )
            except Exception:
                pass
        finally:
            # Drop our handle so the next request isn't rejected as
            # "already running" once this task is done.
            _retranscribe_tasks.pop(job_id, None)


@router.post("/jobs/{job_id}/retranscribe", status_code=202)
async def retranscribe_job_endpoint(
    job_id: str,
    user: User = Depends(get_current_user),
):
    """Kick off a background re-run of Whisper on the job's full audio.

    Uses the language + subtitle-language (translate target) the user
    selected when uploading. If the user chose English subtitles for a
    non-English audio track, Whisper runs in native ``translate`` mode
    — the same logic the initial pipeline uses for the first pass.

    Returns ``202 Accepted`` immediately. Progress is exposed via the
    same ``GET /jobs/{job_id}`` polling and ``/ws/jobs/{job_id}``
    WebSocket the analysis page already watches:

        * ``status`` flips to ``transcribing`` while Whisper runs.
        * ``progress_message`` carries the human-readable phase.
        * ``status`` returns to ``complete`` with the new transcript
          attached when the run finishes.

    Replaces the old gap-fill approach. Only runs on terminal jobs
    (complete / failed / cancelled). The audio.wav is reused from the
    job's work directory, so frame extraction, scene analysis, clip
    detection etc. are NOT re-run — just the transcription phase plus
    inline diarization.
    """
    job = await _require_job_access(job_id, user)

    if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.TRANSCRIBING):
        raise HTTPException(
            status_code=409,
            detail=(
                "Retranscribe requires a finished analysis. This job is "
                f"still running (status={job.status}). Wait for the "
                "initial analysis to finish and try again."
            ),
        )

    existing = _retranscribe_tasks.get(job_id)
    if existing is not None and not existing.done():
        raise HTTPException(
            status_code=409,
            detail="A retranscribe is already running for this job.",
        )

    # Pre-flight: make sure audio.wav is still on disk, and surface the
    # 404 to the user synchronously instead of swallowing it into the
    # background task. This avoids a "click does nothing" experience on
    # jobs whose work dir has been pruned.
    from backend.services.compat_stubs import _locate_audio_path

    if _locate_audio_path(job) is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "audio.wav not found for this job. The source audio was "
                "likely purged — run the full analysis again."
            ),
        )

    # Mark the job as transcribing immediately so the UI button stays
    # disabled and the polling status reflects the new phase before the
    # first Whisper second has even elapsed.
    try:
        await database.update_job_status(
            job_id,
            status=JobStatus.TRANSCRIBING,
            progress_message="Retranscribe queued — Whisper warming up…",
        )
    except Exception:
        logger.exception("[%s] Failed to mark job as transcribing", job_id)

    task = asyncio.create_task(
        _run_retranscribe_in_background(job_id),
        name=f"retranscribe:{job_id}",
    )
    _retranscribe_tasks[job_id] = task

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": (
            "Retranscribe started in the background. Watch the job status "
            "for progress; the transcript will refresh automatically when "
            "Whisper finishes."
        ),
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, user: User = Depends(get_current_user)):
    """Cancel a running or queued job. Also cancels active exports/clip-generation
    for jobs that are already in a terminal state."""
    job = await _require_job_access(job_id, user)

    # Already cancelled — return success (idempotent for repeated clicks)
    if job.status == JobStatus.CANCELLED:
        return {"job_id": job_id, "status": "cancelled"}

    terminal = (JobStatus.COMPLETE, JobStatus.FAILED)
    if job.status in terminal:
        # Job is terminal, but there may be active exports or clip tasks —
        # try to cancel those before rejecting.
        from backend.routers.clips import _active_export_tasks, _export_cancel_events
        from backend.routers.clips import _active_clip_tasks, _clip_cancel_events

        cancelled_something = False

        cancel_evt = _clip_cancel_events.get(job_id)
        if cancel_evt:
            cancel_evt.set()
        clip_task = _active_clip_tasks.pop(job_id, None)
        if clip_task and not clip_task.done():
            clip_task.cancel()
            cancelled_something = True

        for key in list(_active_export_tasks.keys()):
            if key.startswith(f"{job_id}_"):
                evt = _export_cancel_events.get(key)
                if evt:
                    evt.set()
                t = _active_export_tasks.pop(key, None)
                if t and not t.done():
                    t.cancel()
                    cancelled_something = True

        if cancelled_something:
            return {"job_id": job_id, "status": "cancelled"}
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    # For queued jobs not yet running, mark cancelled directly
    if job.status == JobStatus.QUEUED:
        await database.update_job_status(
            job_id,
            status=JobStatus.CANCELLED,
            progress_message="Cancelled by user",
        )
        return {"job_id": job_id, "status": "cancelled"}

    # For running jobs: update DB immediately so polls see "cancelled",
    # then signal the pipeline to stop at its next checkpoint.
    await database.update_job_status(
        job_id,
        status=JobStatus.CANCELLED,
        progress_message="Cancelling...",
    )
    request_cancel(job_id)
    try:
        from backend.services import companion_progress
        companion_progress.job_ended(job_id)
    except Exception:
        pass
    return {"job_id": job_id, "status": "cancelled"}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, user: User = Depends(get_current_user)):
    job = await _require_job_access(job_id, user)
    deletable = (JobStatus.FAILED, JobStatus.COMPLETE, JobStatus.QUEUED, JobStatus.CANCELLED)
    if job.status not in deletable:
        raise HTTPException(status_code=409, detail="Cancel the job first before deleting")
    await database.delete_job(job_id)
    try:
        from backend.services import companion_progress
        companion_progress.job_ended(job_id)
    except Exception:
        pass
    return {"job_id": job_id, "deleted": True}


@router.post("/jobs/{job_id}/analyze")
async def trigger_analysis(job_id: str, background_tasks: BackgroundTasks):
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("queued", "failed", "complete"):
        raise HTTPException(status_code=409, detail="Analysis already in progress")
    # A user-initiated (re)analysis is a fresh start — reset the auto-resume
    # budget so a job that previously hit the restart cap can be retried and,
    # if interrupted again, still auto-resume.
    if getattr(job, "resume_attempts", 0):
        await database.update_job_status(job_id, resume_attempts=0)
    background_tasks.add_task(run_analysis, job_id)
    return {"job_id": job_id, "status": "analysis_started"}


def _filter_cue_range(segments: list, start: Optional[float], end: Optional[float]) -> list:
    """Keep cues overlapping ``[start, end]`` — the server-side equivalent of the
    transcript panel's time filter.

    Exists so a RANGE export can still go through ``generate_srt``. Without it
    the UI had to format a filtered subset itself, which bypassed every subtitle
    invariant (line wrapping, duration cap, min gap, frame alignment)."""
    if start is None and end is None:
        return segments
    lo = float(start) if start is not None else float("-inf")
    hi = float(end) if end is not None else float("inf")
    out = []
    for s in segments:
        try:
            s_start = float(getattr(s, "start", 0.0) or 0.0)
            s_end = float(getattr(s, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if s_start < hi and s_end > lo:
            out.append(s)
    return out


@router.get("/jobs/{job_id}/transcript.srt")
async def download_srt(
    job_id: str,
    speakers: bool = True,
    translated: bool = True,
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    """Download the transcript as a speaker-separated SRT subtitle file.

    When ``translated=true`` (default) and the job has a non-empty
    ``translated_transcript``, the translated version is served and the
    filename gets a ``_translated`` suffix. Pass ``translated=false`` to
    force the original-language transcript.

    ``start`` / ``end`` (seconds) export only the cues overlapping that window,
    mirroring the transcript panel's time filter, so a filtered export still gets
    the full readability / timing treatment.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    source = job.transcript
    lang_suffix = ""
    if translated and job.translated_transcript and len(job.translated_transcript) > 0:
        source = await _clean_translated_rows(job)
        lang_suffix = "_translated"

    if not source:
        raise HTTPException(status_code=404, detail="No transcript available")

    segments = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in source]
    segments = _filter_cue_range(segments, start, end)
    # Pass the video's frame rate so cue in/out points land on real frame
    # boundaries (what hand-authored subtitle tracks do).
    srt_content = generate_srt(
        segments, include_speakers=speakers, fps=getattr(job, "fps", 0.0))

    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    base = (base or "").strip() or "transcript"
    filename = f"{base}{lang_suffix}.srt"

    return Response(
        content=srt_content,
        media_type="text/srt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/transcript_original.srt")
async def download_original_srt(job_id: str, speakers: bool = True):
    """Download the original-language transcript (never the translation)."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=404, detail="No transcript available")
    segments = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in job.transcript]
    srt_content = generate_srt(
        segments, include_speakers=speakers, fps=getattr(job, "fps", 0.0))
    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    base = (base or "").strip() or "transcript"
    return Response(
        content=srt_content,
        media_type="text/srt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{base}_original.srt"'},
    )


@router.get("/jobs/{job_id}/transcript.vtt")
async def download_vtt(
    job_id: str,
    speakers: bool = True,
    include_position: bool = False,
    platform: str = "horizontal",
    translated: bool = True,
    start: Optional[float] = None,
    end: Optional[float] = None,
):
    """Download the transcript as a WebVTT subtitle file.

    Pass ``include_position=true`` and ``platform=tiktok|reels|shorts``
    to embed safe-zone position cues for short-form platforms. When
    ``translated=true`` (default) and a translated transcript exists,
    the translated version is served and the filename gets a
    ``_translated`` suffix.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    source = job.transcript
    lang_suffix = ""
    if translated and job.translated_transcript and len(job.translated_transcript) > 0:
        source = await _clean_translated_rows(job)
        lang_suffix = "_translated"

    if not source:
        raise HTTPException(status_code=404, detail="No transcript available")

    from backend.services.vtt_generator import generate_vtt
    segments = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in source]
    segments = _filter_cue_range(segments, start, end)
    vtt_content = generate_vtt(
        segments,
        include_speakers=speakers,
        include_position=include_position,
        platform=platform,
        fps=getattr(job, "fps", 0.0),
    )

    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    base = (base or "").strip() or "transcript"
    filename = f"{base}{lang_suffix}.vtt"

    return Response(
        content=vtt_content,
        media_type="text/vtt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/jobs/{job_id}/subtitles")
async def download_subtitles(
    job_id: str,
    format: str = "srt",
    speakers: bool = True,
    timestamps: bool = False,
    translated: bool = True,
    target_lang: str = "",
    order: str = "translation_top",
    include_position: bool = False,
    platform: str = "horizontal",
):
    """Unified subtitle export with Otter-style toggles.

    ``format``:
      * ``srt``           — SubRip.
      * ``vtt``           — WebVTT (web players); honors ``include_position``
                            + ``platform`` safe-zone cues.
      * ``bilingual_srt`` — translated + original stacked per cue. Requires
                            ``target_lang`` and translates on demand via the
                            NMT path (LLM fallback), matching the existing
                            router precedence.

    Toggles: ``speakers`` (speaker labels), ``timestamps`` (inline
    ``[mm:ss]`` in the cue text).
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    fmt = (format or "srt").strip().lower()

    if fmt == "bilingual_srt":
        if not job.transcript:
            raise HTTPException(status_code=404, detail="No transcript available")
        if not target_lang:
            raise HTTPException(
                status_code=400, detail="bilingual_srt requires a target_lang")
        from backend.services.srt_generator import generate_bilingual_srt
        source_segments = [
            TranscriptSegment(**s) if isinstance(s, dict) else s
            for s in job.transcript
        ]
        # Reuse an existing translation when it matches the target language,
        # otherwise translate on demand via the offline NMT engine.
        translated_segments = None
        if (getattr(job, "subtitle_language", "") == target_lang
                and job.translated_transcript):
            translated_segments = [
                TranscriptSegment(**s) if isinstance(s, dict) else s
                for s in await _clean_translated_rows(job)
            ]
        if translated_segments is None:
            from backend.services.translator import (
                translate_segments_with_fallback, TranslationFailedError,
            )
            from backend.services.ai_orchestrator import AIOrchestrator
            try:
                translated_segments = await translate_segments_with_fallback(
                    source_segments,
                    source_language=(job.language or "en"),
                    target_language=target_lang,
                    orchestrator=AIOrchestrator(),
                )
            except TranslationFailedError as e:
                # Offline-only translation (no LLM fallback) — return an
                # actionable 503 instead of a generic 500 for the SRT download.
                raise HTTPException(status_code=503, detail=str(e))
        content = generate_bilingual_srt(
            source_segments, translated_segments, order=order,
            include_speakers=speakers, include_timestamps_in_text=timestamps,
            fps=getattr(job, "fps", 0.0),
        )
        media_type = "text/srt; charset=utf-8"
        ext = f"_bilingual_{target_lang}.srt"
    else:
        source = job.transcript
        lang_suffix = ""
        if translated and job.translated_transcript and len(job.translated_transcript) > 0:
            source = await _clean_translated_rows(job)
            lang_suffix = "_translated"
        if not source:
            raise HTTPException(status_code=404, detail="No transcript available")
        segments = [
            TranscriptSegment(**s) if isinstance(s, dict) else s for s in source
        ]
        if fmt == "vtt":
            from backend.services.srt_generator import generate_vtt
            content = generate_vtt(
                segments, include_speakers=speakers,
                include_timestamps_in_text=timestamps,
                include_position=include_position, platform=platform,
                fps=getattr(job, "fps", 0.0),
            )
            media_type = "text/vtt; charset=utf-8"
            ext = f"{lang_suffix}.vtt"
        elif fmt == "srt":
            content = generate_srt(
                segments, include_speakers=speakers,
                include_timestamps_in_text=timestamps,
                fps=getattr(job, "fps", 0.0),
            )
            media_type = "text/srt; charset=utf-8"
            ext = f"{lang_suffix}.srt"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{fmt}' (use srt | vtt | bilingual_srt)")

    base = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    base = (base or "").strip() or "transcript"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{base}{ext}"'},
    )


@router.delete("/jobs/{job_id}/speakers/{speaker}")
async def delete_speaker(
    job_id: str,
    speaker: str,
    reassign_to: str | None = None,
):
    """Remove a speaker from the transcript.

    If ``reassign_to`` is provided, every transcript segment from
    ``speaker`` is reassigned to ``reassign_to`` and the original name
    is dropped from the speaker-name map. Otherwise every segment from
    ``speaker`` is deleted outright.

    Idempotent: if no segments match ``speaker`` the call still succeeds
    (affected_segments=0) and the rename map is still cleaned up.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to edit")
    if reassign_to is not None and reassign_to == speaker:
        raise HTTPException(
            status_code=400,
            detail="Cannot reassign a speaker to itself",
        )

    kept: list = []
    affected = 0
    for seg in job.transcript:
        s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
        if s.speaker == speaker:
            affected += 1
            if reassign_to:
                kept.append(s.model_copy(update={"speaker": reassign_to}))
            # else: drop the segment entirely
            continue
        kept.append(s)

    # Clean the rename map: drop both direct references (key == speaker)
    # AND any rename whose current value points at the deleted speaker.
    # When reassigning we do NOT remap value==speaker to reassign_to,
    # since the map's purpose is auditing the original → current name
    # transition, not the final destination.
    name_map = {
        k: v
        for k, v in job.speaker_names.items()
        if k != speaker and v != speaker
    }

    await database.update_job_status(
        job_id,
        transcript=[s.model_dump() for s in kept],
        speaker_names=name_map,
    )

    return {
        "job_id": job_id,
        "speaker": speaker,
        "action": "reassign" if reassign_to else "delete",
        "reassign_to": reassign_to,
        "affected_segments": affected,
        "remaining_segments": len(kept),
        "speaker_names": name_map,
    }


@router.put("/jobs/{job_id}/speakers")
async def rename_speakers(job_id: str, req: SpeakerRenameRequest):
    """Rename speakers in the transcript.

    Accepts a mapping like {"Speaker 1": "Eric", "Speaker 2": "Alice"}.
    Updates all transcript segments and persists the mapping.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to rename speakers in")

    # Build the full rename map: start from any existing renames, then apply new ones
    name_map = dict(job.speaker_names)
    name_map.update(req.speaker_names)

    # Update transcript segments
    updated_transcript = []
    for seg in job.transcript:
        s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
        # Check if this speaker's name should be replaced
        if s.speaker in req.speaker_names:
            s = s.model_copy(update={"speaker": req.speaker_names[s.speaker]})
        updated_transcript.append(s)

    await database.update_job_status(
        job_id,
        transcript=[s.model_dump() for s in updated_transcript],
        speaker_names=name_map,
    )

    # ── Voiceprint learning loop (Task 3) ──
    # Capture each renamed speaker's voiceprint and enroll/update the
    # registry so this voice is auto-named in future jobs. Fully optional:
    # no-ops silently when the embedding backend / audio are unavailable.
    try:
        from backend.services.voiceprint_registry import (
            enroll_from_segments, locate_job_audio, _voiceprint_enabled,
        )
        if _voiceprint_enabled():
            audio_path = locate_job_audio(job_id)
            if audio_path:
                for old_label, new_name in req.speaker_names.items():
                    spk_segments = [
                        seg for seg in job.transcript
                        if (seg.get("speaker") if isinstance(seg, dict)
                            else getattr(seg, "speaker", None)) == old_label
                    ]
                    if spk_segments:
                        enroll_from_segments(audio_path, spk_segments, new_name)
    except Exception as _vp_err:
        logger.warning("Voiceprint enrollment skipped (%s)", _vp_err)

    return {
        "job_id": job_id,
        "speaker_names": name_map,
        "transcript": [s.model_dump() for s in updated_transcript],
    }


# --- Transcript editing ---

class BulkUpdateSpeakerRequest(BaseModel):
    segment_indices: list[int]
    speaker: str
    # "translated" edits the translated track (when present), else "original".
    target: str | None = None


@router.put("/jobs/{job_id}/transcript/bulk-update-speaker")
async def bulk_update_speaker(job_id: str, req: BulkUpdateSpeakerRequest):
    """Update the speaker for multiple transcript segments at once."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    rows, use_translated = _resolve_transcript_rows(job, req.target)
    if not rows:
        raise HTTPException(status_code=404, detail="No transcript")

    updated = []
    for idx in req.segment_indices:
        if idx < 0 or idx >= len(rows):
            continue
        seg = rows[idx]
        if isinstance(seg, dict):
            seg = TranscriptSegment(**seg)
        rows[idx] = seg.model_copy(update={"speaker": req.speaker})
        updated.append(idx)

    if updated:
        await _persist_transcript_rows(job_id, job, rows, use_translated)
    return {"job_id": job_id, "updated_indices": updated, "speaker": req.speaker,
            "target": "translated" if use_translated else "original"}


class UpdateTranscriptSegmentRequest(BaseModel):
    text: str | None = None
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    words: list[dict] | None = None
    # "translated" edits the translated track (when present), else "original".
    target: str | None = None


@router.put("/jobs/{job_id}/transcript/{segment_index}")
async def update_transcript_segment(job_id: str, segment_index: int, req: UpdateTranscriptSegmentRequest):
    """Update a single transcript segment's text/speaker/timing/words.

    Targets the translated transcript when ``target='translated'`` and a
    translation exists, so edits to the displayed (translated) subtitles
    persist to the right track instead of silently editing the source text.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    use_translated = (
        (req.target or "").strip().lower() == "translated"
        and bool(job.translated_transcript)
    )
    rows = job.translated_transcript if use_translated else job.transcript
    if not rows or segment_index < 0 or segment_index >= len(rows):
        raise HTTPException(status_code=404, detail="Segment not found")

    seg = rows[segment_index]
    if isinstance(seg, dict):
        seg = TranscriptSegment(**seg)
    updates = {}
    if req.text is not None:
        updates["text"] = req.text
    if req.speaker is not None:
        updates["speaker"] = req.speaker
    if req.start is not None:
        updates["start"] = req.start
    if req.end is not None:
        updates["end"] = req.end
    if req.words is not None:
        updates["words"] = req.words
    if updates:
        seg = seg.model_copy(update=updates)
        rows[segment_index] = seg
        if use_translated:
            await database.update_job_status(
                job_id, translated_transcript=[
                    s.model_dump() if hasattr(s, "model_dump") else s for s in rows])
        else:
            await database.save_job(job)
    return {"job_id": job_id, "segment_index": segment_index, "target": "translated" if use_translated else "original",
            "text": seg.text, "speaker": seg.speaker, "start": seg.start, "end": seg.end}


@router.delete("/jobs/{job_id}/transcript/{segment_index}")
async def delete_transcript_segment(job_id: str, segment_index: int, target: str | None = None):
    """Delete a single transcript segment (from the translated track when
    ``target=translated`` and a translation exists, else the source)."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    rows, use_translated = _resolve_transcript_rows(job, target)
    if not rows or segment_index < 0 or segment_index >= len(rows):
        raise HTTPException(status_code=404, detail="Segment not found")

    rows.pop(segment_index)
    await _persist_transcript_rows(job_id, job, rows, use_translated)
    return {"job_id": job_id, "deleted_index": segment_index, "remaining": len(rows),
            "target": "translated" if use_translated else "original"}


class InsertTranscriptSegmentRequest(BaseModel):
    start: float
    end: float
    text: str
    speaker: str
    # "translated" inserts into the translated track (when present), else "original".
    target: str | None = None


@router.post("/jobs/{job_id}/transcript")
async def insert_transcript_segment(job_id: str, req: InsertTranscriptSegmentRequest):
    """Insert a new transcript segment, in chronological order, into the track
    being edited (translated when ``target=translated`` and present, else source)."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    new_seg = TranscriptSegment(start=req.start, end=req.end, text=req.text, speaker=req.speaker)
    rows, use_translated = _resolve_transcript_rows(job, req.target)

    if not rows:
        rows = [new_seg]
        insert_index = 0
    else:
        # Find insertion point to maintain chronological order
        insert_index = 0
        for i, seg in enumerate(rows):
            s = seg if isinstance(seg, TranscriptSegment) else TranscriptSegment(**seg)
            if s.start > new_seg.start:
                break
            insert_index = i + 1
        rows.insert(insert_index, new_seg)

    if use_translated:
        job.translated_transcript = rows
    await _persist_transcript_rows(job_id, job, rows, use_translated)
    return {"job_id": job_id, "inserted_index": insert_index, "segment": new_seg.model_dump(),
            "target": "translated" if use_translated else "original"}


class BulkTranscriptReplaceRequest(BaseModel):
    # Full ordered list of segments (start/end/text/speaker/words). This is the
    # single source of truth for subtitle timing + text, written by both the
    # transcript editor and the NLE timeline so the two stay in sync.
    segments: list[dict]
    # Which transcript to replace: "translated" (default when a translation
    # exists) or "original". Keeping them separate means editing the
    # translated subtitles never clobbers the source-language transcript.
    target: str = "translated"


@router.put("/jobs/{job_id}/transcript")
async def replace_transcript(job_id: str, req: BulkTranscriptReplaceRequest):
    """Replace a job's transcript wholesale, in chronological order.

    Used by the NLE timeline to write subtitle-element timing / text / word
    edits back to the canonical transcript (so the SRT/VTT/TXT downloads and
    the transcript panel all reflect timeline edits), and vice-versa. Edits
    persist to job.json and therefore survive container restarts.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Coerce + sort the incoming segments chronologically. Drop blank /
    # backwards cues so a corrupt timeline edit can't poison the transcript.
    from backend.services.transcript_sync import (
        clean_and_sort_segments, detect_union_write)
    rows = clean_and_sort_segments(req.segments)

    target = (req.target or "translated").strip().lower()
    # Only write the translated track when one already exists; otherwise the
    # edit belongs to the original transcript.
    use_translated = target == "translated" and bool(job.translated_transcript)

    # Union-write guard: refuse a replace that looks like a stale NLE timeline
    # union (fresh cues + phantom copies of the same lines at shifted times)
    # rather than a real edit. This is the write that turned a clean 406-cue
    # track into a 625-cue duplicated transcript (117 texts ~3x each) on the
    # 2026-07-03 run. A 409 tells the frontend to rebuild its subtitle track
    # from the stored transcript instead of pushing the corrupt one again.
    stored = job.translated_transcript if use_translated else job.transcript
    is_union, union_reason = detect_union_write(rows, stored or [])
    if is_union:
        logger.warning("[%s] REJECTED transcript replace (%s): %s",
                       job_id, "translated" if use_translated else "original",
                       union_reason)
        raise HTTPException(
            status_code=409,
            detail=f"Transcript replace rejected: {union_reason}. "
                   "Rebuild the timeline subtitles from the transcript and retry.")

    if use_translated:
        await database.update_job_status(job_id, translated_transcript=rows)
    else:
        await database.update_job_status(job_id, transcript=rows)

    return {
        "job_id": job_id,
        "target": "translated" if use_translated else "original",
        "segments": len(rows),
    }


# --- Word timestamp refresh ---

# Track active refresh tasks to prevent duplicate runs
_active_word_refresh: dict[str, asyncio.Task] = {}


async def _refresh_word_timestamps(job_id: str):
    """Background task: re-run Whisper to extract per-word timestamps and
    merge them onto the existing transcript segments (preserving text/speaker edits)."""
    from backend.services.compat_stubs import extract_word_timestamps
    from backend.services.pipeline import broadcast_ws

    try:
        job = await database.load_job(job_id)
        if not job or not job.transcript:
            return

        audio_path = f"/data/uploads/{job_id}/audio.wav"
        if not os.path.isfile(audio_path):
            logger.warning("[%s] No audio.wav for word timestamp refresh", job_id)
            await broadcast_ws(job_id, {
                "type": "error",
                "message": "Cannot refresh word timestamps: audio file not found. Re-analyze the video to regenerate it.",
            })
            return

        await broadcast_ws(job_id, {
            "type": "status",
            "status": "refreshing_words",
            "message": "Extracting per-word timestamps from audio...",
            "progress": job.progress,
        })

        all_words = await extract_word_timestamps(audio_path, language=job.language or "")
        if not all_words:
            await broadcast_ws(job_id, {
                "type": "error",
                "message": "Word timestamp extraction returned no words",
            })
            return

        # Re-load job in case it was edited during transcription
        job = await database.load_job(job_id)
        if not job or not job.transcript:
            return

        # Map extracted words onto existing segments by time overlap.
        # For each segment, collect words whose midpoint falls within
        # the segment's time range.
        updated = 0
        for idx, seg in enumerate(job.transcript):
            s = TranscriptSegment(**seg) if isinstance(seg, dict) else seg
            seg_words = [
                w for w in all_words
                if (w.start + w.end) / 2 >= s.start and (w.start + w.end) / 2 < s.end
            ]
            if seg_words:
                s = s.model_copy(update={"words": seg_words})
                job.transcript[idx] = s
                updated += 1

        await database.save_job(job)
        logger.info(
            "[%s] Word timestamps refreshed: %d/%d segments updated (%d total words)",
            job_id, updated, len(job.transcript), len(all_words),
        )

        await broadcast_ws(job_id, {
            "type": "word_timestamps_refreshed",
            "status": "complete",
            "message": f"Word timestamps updated for {updated}/{len(job.transcript)} segments",
            "progress": 100,
        })
    except Exception as e:
        logger.exception("[%s] Word timestamp refresh failed: %s", job_id, e)
        from backend.services.pipeline import broadcast_ws
        await broadcast_ws(job_id, {
            "type": "error",
            "message": f"Word timestamp refresh failed: {str(e)}",
        })
    finally:
        _active_word_refresh.pop(job_id, None)


@router.post("/jobs/{job_id}/refresh-word-timestamps")
async def refresh_word_timestamps(job_id: str):
    """Re-run Whisper on existing audio to extract per-word timestamps.

    Merges word-level timing onto the existing transcript segments without
    changing text or speaker assignments.  Useful for enabling accurate
    active-word highlighting on transcripts created before word timestamps
    were captured.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to refresh")

    # Check if already running
    existing = _active_word_refresh.get(job_id)
    if existing and not existing.done():
        return {"job_id": job_id, "status": "already_running"}

    # Check if all segments already have word timestamps
    segments = [
        TranscriptSegment(**s) if isinstance(s, dict) else s
        for s in job.transcript
    ]
    has_words = sum(1 for s in segments if s.words)
    if has_words == len(segments):
        return {"job_id": job_id, "status": "already_complete", "segments_with_words": has_words}

    task = asyncio.create_task(_refresh_word_timestamps(job_id))
    _active_word_refresh[job_id] = task

    return {
        "job_id": job_id,
        "status": "started",
        "segments_total": len(segments),
        "segments_with_words": has_words,
    }


# --- Post-processing diarization ---

class DiarizeRequest(BaseModel):
    num_speakers: int = 0  # 0 = auto-detect, >0 = exact count


@router.post("/jobs/{job_id}/diarize")
async def diarize_job(job_id: str, req: DiarizeRequest):
    """Run speaker diarization on an existing transcript (post-processing).

    The user specifies how many speakers are in the video. The system
    runs pyannote (if available) or the heuristic speaker assigner to
    label each segment with a speaker identity.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.transcript:
        raise HTTPException(status_code=400, detail="No transcript to diarize")

    # Find the audio file
    audio_path = None
    upload_dir = f"/data/uploads/{job_id}"
    for ext in ["wav", "mp3", "m4a", "aac", "ogg", "flac"]:
        matches = glob.glob(f"{upload_dir}/*.{ext}")
        if matches:
            audio_path = matches[0]
            break
    # Also check for extracted audio from video
    if not audio_path:
        extracted = os.path.join(upload_dir, "audio.wav")
        if os.path.isfile(extracted):
            audio_path = extracted

    if not audio_path:
        raise HTTPException(
            status_code=400,
            detail="Audio file not found — re-upload the video to enable diarization"
        )

    from backend.services.compat_stubs import diarize_transcript_post

    try:
        diarized = await diarize_transcript_post(
            audio_path=audio_path,
            segments=job.transcript,
            num_speakers=req.num_speakers,
        )

        await database.update_job_status(job_id, transcript=list(diarized))

        speaker_set = set(s.speaker for s in diarized)
        return {
            "status": "ok",
            "speakers_detected": len(speaker_set),
            "speakers_requested": req.num_speakers if req.num_speakers > 0 else "auto",
            "segments_updated": len(diarized),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)[:200]}")


# --- Scene management ---

class AddSceneRequest(BaseModel):
    timestamp: float
    description: str
    importance_score: int = 7  # 1-10


class UpdateSceneRequest(BaseModel):
    description: str | None = None
    importance_score: int | None = None
    subject_x: int | None = None


@router.post("/jobs/{job_id}/scenes")
async def add_scene(job_id: str, req: AddSceneRequest):
    """Add a user-defined keyscene to the job."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Find closest frame for thumbnail (if frames exist)
    thumbnail_path = ""
    frames_dir = f"/data/uploads/{job_id}/frames"
    if os.path.isdir(frames_dir):
        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png')))
        if frame_files and job.fps > 0:
            target_frame = int(req.timestamp * job.fps)
            # Find closest frame by index
            best = frame_files[0]
            best_diff = abs(target_frame)
            for ff in frame_files:
                try:
                    idx = int(ff.split('_')[1].split('.')[0])
                    diff = abs(idx - target_frame)
                    if diff < best_diff:
                        best_diff = diff
                        best = ff
                except (IndexError, ValueError):
                    pass
            thumbnail_path = os.path.join(frames_dir, best)

    scene = SceneDescription(
        timestamp=req.timestamp,
        description=req.description,
        importance_score=max(1, min(10, req.importance_score)),
        thumbnail_path=thumbnail_path,
        subject_x=50,
    )

    job.scenes.append(scene)
    # Keep scenes sorted by timestamp
    job.scenes.sort(key=lambda s: s.timestamp)
    await database.save_job(job)

    return {
        "job_id": job_id,
        "scene_count": len(job.scenes),
        "scene": scene.model_dump(),
    }


@router.put("/jobs/{job_id}/scenes/{scene_index}")
async def update_scene(job_id: str, scene_index: int, req: UpdateSceneRequest):
    """Update a scene's description, importance score, or subject_x."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes or scene_index < 0 or scene_index >= len(job.scenes):
        raise HTTPException(status_code=404, detail="Scene not found")

    scene = job.scenes[scene_index]
    if isinstance(scene, dict):
        scene = SceneDescription(**scene)

    updates = {}
    if req.description is not None:
        updates["description"] = req.description
    if req.importance_score is not None:
        updates["importance_score"] = max(1, min(10, req.importance_score))
    if req.subject_x is not None:
        updates["subject_x"] = max(0, min(100, req.subject_x))

    if updates:
        scene = scene.model_copy(update=updates)
        job.scenes[scene_index] = scene
        await database.save_job(job)

    return {"job_id": job_id, "scene_index": scene_index, "scene": scene.model_dump()}


@router.delete("/jobs/{job_id}/scenes/{scene_index}")
async def delete_scene(job_id: str, scene_index: int):
    """Delete a scene."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes or scene_index < 0 or scene_index >= len(job.scenes):
        raise HTTPException(status_code=404, detail="Scene not found")

    job.scenes.pop(scene_index)
    await database.save_job(job)
    return {"job_id": job_id, "scene_count": len(job.scenes)}


# --- Subtitle settings (server is source of truth) ---

@router.put("/jobs/{job_id}/subtitle-settings")
async def save_subtitle_settings(job_id: str, request: Request):
    """Save canonical subtitle settings for a job.

    The server stores these so the browser never relies on potentially-stale
    localStorage values.  On every export the frontend should read settings
    from the job object (populated by GET /api/jobs/{job_id}) rather than
    from local cache.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    settings = await request.json()
    job.subtitle_settings = settings
    await database.save_job(job)
    return {"job_id": job_id, "subtitle_settings": job.subtitle_settings}


# --- Subject tracking re-center ---

@router.post("/jobs/{job_id}/recenter-subject")
async def recenter_subject(job_id: str, background_tasks: BackgroundTasks):
    """Center the crop on the subject's detected position.

    Preserves per-scene AI-detected subject_x values so that both the
    preview player and the exported video use dynamic (keyframe-based)
    crop tracking.  If no AI analysis has been run yet (all subject_x
    still at the 50 default), triggers a background re-analysis first.
    The per-scene values are kept intact — no flattening to an average —
    so the crop follows the subject through the clip at every aspect ratio.
    """
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes:
        raise HTTPException(status_code=400, detail="No scenes to recenter")

    # Collect current subject_x values
    sx_values = []
    for scene in job.scenes:
        if isinstance(scene, dict):
            sx_values.append(scene.get("subject_x", 50))
        else:
            sx_values.append(scene.subject_x if hasattr(scene, "subject_x") else 50)

    # Check if AI analysis has been run (at least one non-default value)
    has_ai_data = any(v != 50 for v in sx_values)

    if not has_ai_data:
        # No AI analysis data — trigger background re-analysis.
        # center_after=False: preserve per-scene values for dynamic tracking.
        background_tasks.add_task(_reanalyze_subject_tracking, job_id, center_after=False)
        return {
            "job_id": job_id,
            "scenes_recentered": 0,
            "status": "reanalyzing",
            "message": "No subject data available — running AI analysis to detect subject position",
        }

    # AI data already exists — return the per-scene values as-is.
    # The preview player and export pipeline both build keyframes from
    # per-scene subject_x, so the crop dynamically follows the subject
    # at whatever aspect ratio is applied.
    avg_sx = round(sum(sx_values) / len(sx_values))
    avg_sx = max(10, min(90, avg_sx))
    return {
        "job_id": job_id,
        "scenes_recentered": len(job.scenes),
        "subject_x": avg_sx,
        "per_scene": True,
    }


async def _reanalyze_subject_tracking(job_id: str, center_after: bool = False):
    """Background task: re-run AI subject tracking on existing frames.

    If center_after=True, also computes the average subject_x after analysis
    and sets all scenes to that value for a static centered crop.
    """
    from backend.services.ai_orchestrator import AIOrchestrator
    from backend.services.frame_extractor import frame_to_base64
    from backend.services.prompts import load_prompts
    from backend.services.pipeline import broadcast_ws

    job = await database.load_job(job_id)
    if not job or not job.scenes:
        return

    frames_dir = f"/data/uploads/{job_id}/frames"
    if not os.path.isdir(frames_dir):
        logger.warning("[%s] No frames directory for re-analysis", job_id)
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "No extracted frames found — cannot re-analyze subject tracking",
        })
        return

    await broadcast_ws(job_id, {
        "type": "status",
        "status": "reanalyzing",
        "message": "Re-analyzing subject positions with AI...",
        "progress": job.progress,
    })

    # Build FrameData for each scene from existing frames on disk
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_files:
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_files:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "No frame images found — cannot re-analyze subject tracking",
        })
        return

    # Match existing scenes to their frame files by timestamp
    scene_frames = []
    for scene in job.scenes:
        if isinstance(scene, dict):
            scene = SceneDescription(**scene)
        # Try to find the exact frame file for this scene's thumbnail
        if scene.thumbnail_path and os.path.exists(scene.thumbnail_path):
            scene_frames.append(FrameData(timestamp=scene.timestamp, path=scene.thumbnail_path))
        else:
            # Find closest frame file by name (frame_00123.jpg -> timestamp)
            thumb_name = scene.thumbnail_path.split("/")[-1] if scene.thumbnail_path else ""
            matched = [f for f in frame_files if os.path.basename(f) == thumb_name]
            if matched:
                scene_frames.append(FrameData(timestamp=scene.timestamp, path=matched[0]))
            elif frame_files:
                # Use closest frame by index
                idx = min(len(frame_files) - 1, max(0, round(scene.timestamp / (job.duration or 1) * len(frame_files))))
                scene_frames.append(FrameData(timestamp=scene.timestamp, path=frame_files[idx]))

    if not scene_frames:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "Could not match scenes to frame files",
        })
        return

    # Encode frames to base64
    loop = asyncio.get_event_loop()
    for fr in scene_frames:
        try:
            fr.base64 = await loop.run_in_executor(None, lambda p=fr.path: frame_to_base64(p, skip_resize=True))
        except Exception as e:
            logger.warning("[%s] Failed to encode frame %s: %s", job_id, fr.path, e)
            fr.base64 = None

    scene_frames = [fr for fr in scene_frames if fr.base64]
    if not scene_frames:
        await broadcast_ws(job_id, {
            "type": "error",
            "message": "Failed to encode frames for re-analysis",
        })
        return

    # Run AI analysis
    try:
        custom_prompts = load_prompts()
        orchestrator = AIOrchestrator(custom_prompts=custom_prompts)
        new_scenes, provider = await orchestrator.analyze_frames(scene_frames, job_id)

        # Map updated subject_x back to existing scenes by timestamp matching
        new_sx_map = {round(s.timestamp, 1): s.subject_x for s in new_scenes}
        updated_count = 0
        for i, scene in enumerate(job.scenes):
            if isinstance(scene, dict):
                scene = SceneDescription(**scene)
                job.scenes[i] = scene
            key = round(scene.timestamp, 1)
            if key in new_sx_map:
                scene.subject_x = new_sx_map[key]
                updated_count += 1

        # Per-scene AI-detected values are preserved (no averaging/flattening).
        # The preview player and export pipeline both build keyframes from
        # per-scene subject_x values, enabling dynamic crop tracking at any
        # aspect ratio.
        if center_after:
            sx_vals = [s.subject_x for s in job.scenes if hasattr(s, "subject_x")]
            non_default = [v for v in sx_vals if v != 50]
            if non_default:
                logger.info(
                    "[%s] Re-analysis complete: preserving %d per-scene subject_x values "
                    "(range %d-%d, avg %d) for dynamic tracking",
                    job_id, len(non_default),
                    min(non_default), max(non_default),
                    round(sum(non_default) / len(non_default)),
                )

        await database.save_job(job)
        logger.info("[%s] Subject re-analysis complete: %d/%d scenes updated via %s", job_id, updated_count, len(job.scenes), provider)

        await broadcast_ws(job_id, {
            "type": "complete",
            "status": "complete",
            "message": f"Subject tracking re-analyzed: {updated_count} scenes updated via {provider}",
            "progress": 100,
        })
    except Exception as e:
        logger.exception("[%s] Subject re-analysis failed: %s", job_id, e)
        await broadcast_ws(job_id, {
            "type": "error",
            "message": f"Subject re-analysis failed: {str(e)}",
        })


@router.post("/jobs/{job_id}/reanalyze-subject")
async def reanalyze_subject(job_id: str, background_tasks: BackgroundTasks):
    """Re-run AI subject tracking analysis on existing frames. Updates subject_x values."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.scenes:
        raise HTTPException(status_code=400, detail="No scenes to re-analyze")
    background_tasks.add_task(_reanalyze_subject_tracking, job_id)
    return {"job_id": job_id, "status": "reanalyzing", "scenes_count": len(job.scenes)}


class RescoreRequest(BaseModel):
    """Optional payload for ``POST /api/jobs/{id}/rescore``.

    Lets the caller experiment with a different ``content_type`` (e.g.
    "what would the rankings look like if this were tagged as
    podcast?") without re-running detection. When omitted, uses the
    job's existing classified content type.
    """
    content_type: str | None = None
    focus_mode: bool = False


@router.post("/jobs/{job_id}/rescore")
async def rescore_clips(
    job_id: str,
    payload: RescoreRequest | None = None,
    user: User = Depends(get_current_user),
):
    """Re-run only the genre-weighted scoring composite on existing
    clips. NO LLM / vision / Whisper calls are made — this is purely a
    deterministic Python pass over the clips already on the job.

    Use cases:
      * Experiment with different ``content_type`` weights without
        paying for vision again.
      * Recompute after the user fixed a bad classification in the UI.
      * Backfill ``viral_score_composite`` on legacy jobs that ran
        before the four-axis composite landed.

    Returns the updated clips. Side-effect: persists them onto the job.
    """
    job = await _require_job_access(job_id, user)
    if not job.clips:
        raise HTTPException(status_code=400, detail="Job has no clips to rescore")

    from backend.models import ClipCandidate
    from backend.services.compat_stubs import finalize_clip_scores
    from backend.services.compat_stubs import ClipContentType

    # Resolve the content_type to use:
    #   1. explicit override from payload
    #   2. the job's classified content_type (stored on JobResult)
    #   3. the legacy ``content_type_override`` field
    #   4. None — uses the default weights table.
    requested = (payload.content_type if payload else None) or ""
    requested = requested.strip().lower()
    if not requested:
        requested = (
            getattr(job, "classified_content_type", "")
            or getattr(job, "content_type_override", "")
            or ""
        ).strip().lower()
    ct: ClipContentType | None = None
    if requested:
        try:
            ct = ClipContentType(requested)
        except ValueError:
            ct = None

    # Coerce dict clips back to model instances so `finalize_clip_scores`
    # can mutate them in place. Job storage round-trips them as dicts.
    clip_models: list[ClipCandidate] = []
    for raw in job.clips:
        if isinstance(raw, ClipCandidate):
            clip_models.append(raw)
        else:
            try:
                clip_models.append(ClipCandidate(**raw))
            except Exception as e:
                logger.warning("[%s] rescore: skipping malformed clip: %s", job_id, e)
    if not clip_models:
        raise HTTPException(status_code=400, detail="No valid clips found to rescore")

    rescored = finalize_clip_scores(
        clip_models,
        content_type=ct,
        focus_mode=bool(payload.focus_mode) if payload else False,
    )
    job.clips = [c.model_dump() for c in rescored]
    await database.save_job(job)
    return {
        "job_id": job_id,
        "rescored_count": len(rescored),
        "content_type": ct.value if ct else None,
        "focus_mode": bool(payload.focus_mode) if payload else False,
        "clips": job.clips,
    }


# Per-job diagnostic artifacts the reframer writes alongside the
# upload. Inlined into the log export so a remote reviewer (Claude
# Code or otherwise) sees the per-decision data without having to
# SSH in and grep /data/uploads/<job_id>/. Kept as raw text bodies
# so each section stays grep-able with the same tools the user
# already runs on the log itself (jq for the JSONL, less for the rest).
_JOB_DIAGNOSTIC_FILES = [
    "render_plan.json",       # per-scene signals + params + keyframe list
    "reframe_trace.jsonl",    # per-decision events (one per line)
    "detection_overlay.json", # face / person / motion / saliency timelines
                              # in source-pixel coords — the raw inputs
                              # the planner reasoned over. Without this
                              # the trace tells you the decision but not
                              # what alternatives were available.
]

# Keys whose values are secrets — printed as ``<set>`` / ``<unset>``
# placeholders in the ACTIVE CONFIG block rather than leaking into the
# exported log. The user mails this file around for debugging, so
# anything containing an API key, OAuth secret, encryption key, or
# token must be redacted here.
_REDACTED_CONFIG_KEYS = {
    # backend/config.py
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "REPLICATE_API_KEY", "GOOGLE_TRANSLATE_API_KEY",
    "DEEPL_API_KEY", "HF_AUTH_TOKEN", "CLIPAI_TOKEN_ENC_KEY",
    "GOOGLE_DRIVE_CLIENT_SECRET", "BOX_CLIENT_SECRET",
    # clipper_config.json
    "google_ai_key", "openrouter_key", "replicate_api_key",
}

# Subset of ``settings`` fields actually useful for reframer debugging.
# A full dump would include cloud-storage paths, FastAPI internals, and
# 60+ feature flags unrelated to the analysis pipeline — too much noise.
_CONFIG_KEYS_OF_INTEREST = [
    "AI_PROVIDER", "AI_FALLBACK_CHAIN", "SELF_HOSTED_MODE",
    "CLIP_ENGINE_SOURCE", "EDITORIAL_AI_SOURCE",
    "OPENROUTER_PRESET", "OPENROUTER_PRIMARY_MODEL",
    "OPENROUTER_EDITORIAL_MODEL", "OPENROUTER_SUMMARY_MODEL",
    "ANTHROPIC_MODEL", "GEMINI_EDITORIAL_MODEL", "GEMINI_VIDEO_MODEL",
    "GEMINI_USE_NATIVE_VIDEO", "GROQ_EDITORIAL_MODEL",
    "OLLAMA_HOST", "OLLAMA_PRIMARY_MODEL", "OLLAMA_EDITORIAL_MODEL",
    "VIDEOLLAMA2_ENABLED", "VIDEOLLAMA2_MODEL", "VIDEOLLAMA2_QUANTIZE",
    "VIDEOLLAMA3_ENHANCED", "VIDEOLLAMA3_FPS", "VIDEOLLAMA3_MAX_FRAMES",
    "VIDEOLLAMA3_REFINEMENT_PASS", "VIDEOLLAMA3_KEYFRAME_ANALYSIS",
    "VIDEOLLAMA3_AUDIO_ANNOTATION", "VIDEOLLAMA3_ADAPTIVE_CHUNKS",
    "REPLICATE_MODEL", "REPLICATE_ENABLED",
    "WHISPER_MODEL", "WHISPER_BEAM_SIZE", "WHISPER_VAD_FILTER",
    "WHISPER_VAD_ONSET", "WHISPER_NO_SPEECH_THRESHOLD",
    "WHISPER_AUDIO_PRECONDITION", "WHISPER_AUTO_UPGRADE",
    "VOCAL_SEPARATION_ENABLED", "VOCAL_SEPARATION_MODEL",
    "VOCAL_SEPARATION_DEVICE", "VOCAL_SEPARATION_SEGMENT",
    "VOCAL_SEPARATION_TIMEOUT",
    "FRAME_SAMPLE_RATE", "MIN_FRAMES", "FRAMES_PER_MINUTE",
    "MAX_CLIP_CANDIDATES", "CLIP_MIN_DURATION", "CLIP_MAX_DURATION",
    "CLIP_COUNT", "CLIP_PREFERRED_SUBJECTS", "CLIP_AVOID_SUBJECTS",
    "AI_TRANSCRIPT_CORRECTION", "TRANSCRIPT_POLISHING_ENABLED",
    "TRANSCRIPT_FILLER_REMOVAL", "TRANSCRIPT_SENTENCE_REPAIR",
    "TRANSCRIPT_PRESERVE_WORDS", "TRANSCRIPT_READABILITY_TARGET",
    "TRANSCRIPT_READABILITY_MAX_PASSES",
    "SUBTITLE_CPS_ENFORCEMENT", "SUBTITLE_MAX_CPS",
    "SUBTITLE_MAX_CHARS_PER_LINE", "SUBTITLE_PLATFORM_PROFILE",
    "TRANSLATION_ENGINE", "TRANSLATION_CONTEXT_WINDOW",
    "GPU_ACCELERATION_ENABLED", "GPU_VENDOR_OVERRIDE",
    "GPU_HWDECODE_ENABLED", "GPU_HEVC_FOR_4K", "GPU_DEVICE_INDEX",
    "GPU_FREE_BEFORE_ANALYSIS", "GPU_FREE_BEFORE_WHISPER",
    "FFMPEG_PRESET", "FFMPEG_CRF", "FFMPEG_THREADS", "FFMPEG_FASTSTART",
    "DIARIZATION_ENABLED", "AUDIO_EVENT_DETECTION",
    "SUBJECT_TRACKING_ENABLED", "CLIPAI_CAMERA_SOLVER",
    "CLIPAI_CONTENT_ROUTING",
]

# Cap how many job dirs we walk to keep the export under reasonable
# size on long-lived installs. Recent-first by mtime — the user's
# last few analyses are almost always what they want to inspect.
_DIAGNOSTIC_JOB_LIMIT = 20

# Per-file size guard so one runaway artifact can't blow the
# response up to hundreds of MB. ~2 MB is plenty for the trace
# on a 24-min video and well below the export's overall budget.
_DIAGNOSTIC_FILE_MAX_BYTES = 2 * 1024 * 1024

# The DB ``reframe_report`` carries a ``problems`` list with one entry per
# flagged keyframe — thousands on a long video (one export was 6,200 entries /
# ~100k lines, dwarfing everything else and making the bundle unreadable). The
# aggregate scores at the top are what matter for triage; cap the per-problem
# detail to a readable sample and note how many were omitted.
_DIAGNOSTIC_PROBLEMS_MAX = 40


def _recent_job_dirs(uploads_root: str, limit: int) -> list[tuple[str, str, float]]:
    """Return ``(job_id, job_dir, mtime)`` for the ``limit`` most-recent
    job dirs in ``uploads_root`` that look like they ran the reframer.

    "Looks like the reframer ran" = has at least one of the diagnostic
    files we'd want to inline. Jobs that are still uploading or that
    never finished analysis are skipped to keep the export focused on
    actionable data.
    """
    if not os.path.isdir(uploads_root):
        return []
    candidates: list[tuple[str, str, float]] = []
    for entry in os.listdir(uploads_root):
        job_dir = os.path.join(uploads_root, entry)
        if not os.path.isdir(job_dir):
            continue
        has_any = any(
            os.path.isfile(os.path.join(job_dir, name))
            for name in _JOB_DIAGNOSTIC_FILES
        )
        if not has_any:
            continue
        try:
            # Use the newest diagnostic file as the "analyzed at" mtime
            # so re-running an old job pulls it back to the top.
            mtime = max(
                os.path.getmtime(os.path.join(job_dir, name))
                for name in _JOB_DIAGNOSTIC_FILES
                if os.path.isfile(os.path.join(job_dir, name))
            )
        except OSError:
            continue
        candidates.append((entry, job_dir, mtime))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:limit]


def _read_diagnostic(path: str) -> tuple[str, int]:
    """Read a diagnostic file with size cap. Returns ``(body, bytes_read)``."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return (f"[clipai: cannot stat {path}: {e}]\n", 0)
    if size > _DIAGNOSTIC_FILE_MAX_BYTES:
        try:
            with open(path, "rb") as f:
                head = f.read(_DIAGNOSTIC_FILE_MAX_BYTES).decode(
                    "utf-8", errors="replace")
            return (
                head + (
                    f"\n[clipai: file truncated at "
                    f"{_DIAGNOSTIC_FILE_MAX_BYTES} bytes "
                    f"(actual size {size}); SSH in for the full artifact]\n"
                ),
                _DIAGNOSTIC_FILE_MAX_BYTES,
            )
        except OSError as e:
            return (f"[clipai: cannot read {path}: {e}]\n", 0)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return (f.read(), size)
    except OSError as e:
        return (f"[clipai: cannot read {path}: {e}]\n", 0)


def _build_active_config_block() -> str:
    """Snapshot the reframer-relevant config knobs once for the export.

    Two sources merged:

      * ``backend.config.settings`` — process-wide env-driven settings
        (Whisper tier, AI provider chain, GPU flags, FFmpeg knobs, …).
        Filtered to ``_CONFIG_KEYS_OF_INTEREST`` to keep the noise
        floor low. Secret keys redacted to ``<set>`` / ``<unset>``.

      * ``clipper_config.json`` (preferred ``/data/`` mount-backed
        copy, with ``/app/`` legacy fallback) — the clipper's
        per-install dataclass (judge selection, VLM toggle, platform
        list, max clips, etc.). Secret keys redacted the same way.

    Snapshot reflects current values at export time, not necessarily
    what was in effect when each job ran — but in practice the user
    rarely changes these between runs, and the most-recent jobs at
    the top of the diagnostics list are the ones a reviewer is
    actually looking at.
    """
    lines: list[str] = ["\n", "=" * 78 + "\n",
                        "== ACTIVE CONFIG (current values at export time)\n",
                        "=" * 78 + "\n"]

    # ── backend.config.settings ──
    try:
        from backend.config import settings
        lines.append("\n--- settings (backend/config.py) ---\n")
        for key in _CONFIG_KEYS_OF_INTEREST:
            value = getattr(settings, key, None)
            if value is None:
                continue
            lines.append(f"{key} = {value!r}\n")
        # Secret-bearing keys: render as <set>/<unset> only.
        lines.append("\n--- settings (secrets, redacted) ---\n")
        for key in sorted(_REDACTED_CONFIG_KEYS):
            if not hasattr(settings, key):
                continue
            value = getattr(settings, key, None)
            present = bool(value) and str(value) not in ("<set>", "")
            lines.append(f"{key} = {'<set>' if present else '<unset>'}\n")
    except Exception as e:
        lines.append(f"\n[clipai: failed to read settings: {e}]\n")

    # ── clipper_config.json ──
    try:
        from backend.services.pipeline import clipper_config_path
        # Prefers /data/clipper_config.json (mount-backed) then falls
        # back to the legacy /app/clipper_config.json — same resolver
        # the settings router and the pipeline loader use.
        clipper_path = clipper_config_path()
        if os.path.isfile(clipper_path):
            import json as _json
            with open(clipper_path, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
            for k in list(cfg.keys()):
                if k in _REDACTED_CONFIG_KEYS:
                    cfg[k] = "<set>" if cfg[k] else "<unset>"
            lines.append("\n--- clipper_config.json ---\n")
            lines.append(_json.dumps(cfg, indent=2, default=str))
            lines.append("\n")
        else:
            lines.append("\n[clipai: clipper_config.json not found]\n")
    except Exception as e:
        lines.append(f"\n[clipai: failed to read clipper_config.json: {e}]\n")

    return "".join(lines)


def _build_job_context_block(job, job_dir: str) -> str:
    """One-block summary so a reviewer doesn't have to scroll the app
    log to figure out what this job actually was.

    Pulls source dims / fps / crop geometry from ``render_plan.json``
    when present (most reliable — it's what the reframer actually
    saw), and the rest from the job DB record (file name, language,
    duration, content type, cost, compute summary).
    """
    fields: list[tuple[str, object]] = []
    if job is not None:
        for attr, label in [
            ("status", "status"),
            ("language", "detected_language"),
            ("subtitle_language", "subtitle_language"),
            ("analysis_duration_seconds", "analysis_duration_seconds"),
            ("estimated_cost_usd", "estimated_cost_usd"),
            ("content_type_override", "content_type_override"),
            ("game_type", "game_type"),
            ("anime_subtype", "anime_subtype"),
            ("created_at", "created_at"),
            ("updated_at", "updated_at"),
        ]:
            value = getattr(job, attr, None)
            if value not in (None, "", 0, 0.0):
                fields.append((label, value))
        # compute_summary is itself a dict — render compactly inline so
        # the reviewer can read it at a glance.
        cs = getattr(job, "compute_summary", None)
        if isinstance(cs, dict) and cs:
            badges = ", ".join(
                f"{stage}={(info or {}).get('device', '?')}"
                for stage, info in cs.items())
            fields.append(("compute_summary", badges))
        clips = getattr(job, "clips", None) or []
        fields.append(("clip_count", len(clips)))

    # Pull the source geometry + duration out of render_plan.json (the
    # planner's own record of what it saw). More reliable than the DB
    # record which can drift if the user kicks off a regenerate.
    plan_path = os.path.join(job_dir, "render_plan.json")
    if os.path.isfile(plan_path):
        try:
            import json as _json
            with open(plan_path, "r", encoding="utf-8") as f:
                plan = _json.load(f)
            for k in ("source_width", "source_height", "target_width",
                      "target_height", "fps", "duration_ms", "crop_w",
                      "crop_h", "crop_y", "version"):
                if k in plan:
                    fields.append((f"plan.{k}", plan[k]))
            scenes = plan.get("scenes") or []
            if scenes:
                strategies = {}
                for sc in scenes:
                    s = sc.get("strategy", "?")
                    strategies[s] = strategies.get(s, 0) + 1
                fields.append(("plan.scenes", len(scenes)))
                fields.append((
                    "plan.strategy_breakdown",
                    ", ".join(f"{s}={n}" for s, n in
                              sorted(strategies.items(), key=lambda x: -x[1])),
                ))
            fields.append(("plan.keyframes", len(plan.get("keyframes") or [])))
        except Exception as e:
            fields.append(("plan_read_error", str(e)))

    if not fields:
        return ""

    width = max(len(label) for label, _ in fields)
    out = ["\n--- job context ---\n"]
    for label, value in fields:
        out.append(f"{label:<{width}} : {value}\n")
    return "".join(out)


async def _collect_job_diagnostics() -> str:
    """Build the ``=== JOB DIAGNOSTICS ===`` tail section of the export.

    For each of the most-recent reframer jobs, dumps render_plan.json
    and reframe_trace.jsonl inline (so a single text file gives the
    full per-decision picture) plus the reframe_report from the job
    record (only persisted in the DB, not as a file).
    """
    uploads_root = "/data/uploads"
    jobs = _recent_job_dirs(uploads_root, _DIAGNOSTIC_JOB_LIMIT)
    if not jobs:
        return ""

    out: list[str] = [
        "\n",
        "=" * 78 + "\n",
        f"== JOB DIAGNOSTICS — {len(jobs)} most-recent reframer job(s)\n",
        "==   job context           source dims / fps / strategies / clips\n",
        "==   reframe_report        A-F grade + problems with keyframe brackets\n",
        "==   render_plan.json      per-scene signals + params + keyframes\n",
        "==   reframe_trace.jsonl   per-decision events\n",
        "==   detection_overlay.json face / person / motion / saliency timelines\n",
        "=" * 78 + "\n",
    ]
    # ACTIVE CONFIG goes once at the top — it's the same for every
    # job in the export. Cheaper to read once and cleaner to read.
    out.append(_build_active_config_block())

    for job_id, job_dir, mtime in jobs:
        ts = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(
            timespec="seconds")
        out.append("\n")
        out.append("-" * 78 + "\n")
        out.append(f"-- JOB {job_id} (artifact mtime: {ts})\n")
        out.append("-" * 78 + "\n")

        # The DB-persisted reframe_report + the per-job context block.
        # Loaded best-effort so a database failure can't block the rest
        # of the export — even if the DB read fails, we still emit the
        # on-disk artifacts below.
        job = None
        try:
            job = await database.load_job(job_id)
        except Exception as e:
            out.append(f"\n[clipai: failed to load job from db: {e}]\n")

        out.append(_build_job_context_block(job, job_dir))

        report = getattr(job, "reframe_report", None) if job else None
        if report:
            out.append("\n--- reframe_report ---\n")
            import json as _json
            # Cap the per-keyframe ``problems`` list so one job's thousands of
            # findings can't bloat the bundle into the 100k-line range — the
            # aggregate scores above the list are what triage needs.
            _rpt = report
            try:
                _probs = report.get("problems") if isinstance(report, dict) else None
                if isinstance(_probs, list) and len(_probs) > _DIAGNOSTIC_PROBLEMS_MAX:
                    _rpt = dict(report)
                    _rpt["problems"] = _probs[:_DIAGNOSTIC_PROBLEMS_MAX]
                    _rpt["problems_omitted"] = (
                        f"{len(_probs) - _DIAGNOSTIC_PROBLEMS_MAX} more "
                        f"(showing first {_DIAGNOSTIC_PROBLEMS_MAX} of "
                        f"{len(_probs)}; SSH in for the full reframe_report)")
            except Exception:
                _rpt = report
            out.append(_json.dumps(_rpt, indent=2, default=str))
            out.append("\n")

        # The on-disk artifacts.
        for fname in _JOB_DIAGNOSTIC_FILES:
            path = os.path.join(job_dir, fname)
            if not os.path.isfile(path):
                continue
            body, _bytes = _read_diagnostic(path)
            out.append(f"\n--- {fname} ---\n")
            out.append(body)
            if not body.endswith("\n"):
                out.append("\n")

    return "".join(out)


@router.get("/logs/export")
async def export_logs():
    """Export the full diagnostic bundle as a single text file.

    Includes:
      * The rotating ``/data/logs/app.log`` plus up to 3 rotated
        backups (chronological order: oldest backup first, current
        log last).
      * Per-job diagnostic artifacts for the most-recent reframer
        jobs (``render_plan.json``, ``reframe_trace.jsonl``, and
        the persisted ``reframe_report``). Inlined with clear
        ``=== JOB <id> ===`` section markers so a reviewer can grep
        the same single file for both server-level events and
        per-decision reframer events. Capped at
        ``_DIAGNOSTIC_JOB_LIMIT`` jobs and ``_DIAGNOSTIC_FILE_MAX_BYTES``
        per file to keep the response under control on busy installs.
    """
    log_file = "/data/logs/app.log"
    parts = []

    # Read rotated backups oldest-first (app.log.3, app.log.2, app.log.1)
    for i in range(3, 0, -1):
        rotated = f"{log_file}.{i}"
        if os.path.isfile(rotated):
            try:
                with open(rotated, "r", encoding="utf-8", errors="replace") as f:
                    parts.append(f.read())
            except OSError:
                pass

    # Read current log file
    if os.path.isfile(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
        except OSError:
            pass

    # Append per-job diagnostic artifacts. Failures here must not
    # abort the export — the rotating log is the priority, the
    # diagnostics are a bonus.
    try:
        diagnostics = await _collect_job_diagnostics()
        if diagnostics:
            parts.append(diagnostics)
    except Exception as e:
        logger.warning("Failed to collect job diagnostics for export: %s", e)
        parts.append(f"\n[clipai: job diagnostics collection failed: {e}]\n")

    if not parts:
        raise HTTPException(status_code=404, detail="No log files found")

    content = "".join(parts)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"clipai_logs_{ts}.log"

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/allocation")
async def get_allocation():
    """Return all jobs/processes currently consuming container resources."""
    import psutil

    jobs = await database.list_jobs()
    active_jobs = []
    terminal_statuses = {JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED}
    for j in jobs:
        if j.status not in terminal_statuses:
            elapsed = None
            if j.created_at:
                try:
                    created = datetime.fromisoformat(str(j.created_at))
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    elapsed = int((datetime.now(timezone.utc) - created).total_seconds())
                except Exception:
                    pass
            active_jobs.append({
                "job_id": j.job_id,
                "filename": j.filename,
                "status": j.status,
                "progress": j.progress,
                "progress_message": j.progress_message,
                "file_size_mb": j.file_size_mb,
                "duration": j.duration,
                "elapsed_seconds": elapsed,
                "type": "analysis",
            })

    # Get active exports from clips router
    from backend.routers.clips import _active_export_tasks, _active_clip_tasks
    for key, task in _active_export_tasks.items():
        if not task.done():
            parts = key.split("_", 1)
            active_jobs.append({
                "job_id": parts[0] if len(parts) > 1 else key,
                "clip_id": int(parts[1]) if len(parts) > 1 else 0,
                "export_key": key,
                "filename": f"Clip {parts[1]}" if len(parts) > 1 else key,
                "status": "encoding",
                "progress": None,
                "progress_message": "Encoding clip...",
                "type": "export",
            })
    for job_id, task in _active_clip_tasks.items():
        if not task.done():
            active_jobs.append({
                "job_id": job_id,
                "filename": "Clip Detection",
                "status": "detecting_clips",
                "progress": None,
                "progress_message": "AI generating clips...",
                "type": "clip_generation",
            })

    # System resource usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/data") if os.path.exists("/data") else psutil.disk_usage("/")

    return {
        "active_jobs": active_jobs,
        "resources": {
            "cpu_percent": round(cpu_percent, 1),
            "memory_used_mb": round(mem.used / (1024 * 1024)),
            "memory_total_mb": round(mem.total / (1024 * 1024)),
            "memory_percent": round(mem.percent, 1),
            "disk_used_gb": round(disk.used / (1024 ** 3), 1),
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_percent": round(disk.percent, 1),
        },
    }


@router.post("/jobs/{job_id}/force-fail")
async def force_fail_job(job_id: str):
    """Force-fail a job to free up resources. Works on any non-terminal job,
    and also cancels active exports/clip-generation for already-terminal jobs."""
    job = await database.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    from backend.routers.clips import _active_clip_tasks, _clip_cancel_events
    from backend.routers.clips import _active_export_tasks, _export_cancel_events

    terminal = (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED)
    cancelled_something = False

    # Cancel clip generation tasks for this job
    cancel_evt = _clip_cancel_events.get(job_id)
    if cancel_evt:
        cancel_evt.set()
    clip_task = _active_clip_tasks.pop(job_id, None)
    if clip_task and not clip_task.done():
        clip_task.cancel()
        cancelled_something = True

    # Cancel any exports for this job
    for key in list(_active_export_tasks.keys()):
        if key.startswith(f"{job_id}_"):
            evt = _export_cancel_events.get(key)
            if evt:
                evt.set()
            t = _active_export_tasks.pop(key, None)
            if t and not t.done():
                t.cancel()
                cancelled_something = True

    if job.status in terminal and not cancelled_something:
        raise HTTPException(status_code=409, detail=f"Job is already {job.status}")

    # Signal analysis pipeline cancellation and mark failed (if not already terminal)
    if job.status not in terminal:
        request_cancel(job_id)
        await database.update_job_status(
            job_id,
            status=JobStatus.FAILED,
            progress_message="Force-failed by admin to free resources",
        )

    return {"job_id": job_id, "status": "failed", "message": "Job force-failed"}
