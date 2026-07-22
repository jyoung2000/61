import json
import logging
import os
import shutil
import asyncio
import tempfile
import time
from collections import OrderedDict
from typing import Optional

import aiofiles

from backend.models import JobResult, TranscriptSegment
from backend.services import fastjson

logger = logging.getLogger(__name__)


def _numpy_safe_default(obj):
    """JSON serializer fallback for numpy types that slip through."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return str(obj)


_file_locks: dict[str, asyncio.Lock] = {}


def _get_lock(job_id: str) -> asyncio.Lock:
    if job_id not in _file_locks:
        _file_locks[job_id] = asyncio.Lock()
    return _file_locks[job_id]


# ── In-memory job cache + coalesced progress persistence ────────────
#
# Every ``update_job_status`` used to round-trip the FULL job.json through
# disk: read → json.loads → Pydantic-validate → mutate → model_dump →
# json.dumps → atomic rewrite. After the analysis save that file carries the
# full transcript, scenes, subject track and clips (multiple MB), and progress
# callbacks fire every ~1-3 s through extraction / the engine relay /
# translation / the clipper — hundreds of multi-MB parse+serialize+write
# cycles per job, competing with ffmpeg/Whisper for CPU and thrashing disk.
#
# Three cooperating mechanisms fix that without touching the atomic-write /
# per-job-lock / protect_terminal / save-guard semantics:
#   * ``_job_cache`` — an mtime-checked LRU of parsed ``JobResult`` objects,
#     keyed by the canonical job.json PATH (so tests that monkeypatch
#     ``_job_dir`` can never alias entries across directories). All writers go
#     through this module under the same per-job lock; the mtime check
#     additionally protects against out-of-band edits of job.json on disk.
#   * progress-only debounce — updates that change nothing but
#     ``progress`` / ``progress_message`` (and a same-value ``status``) mutate
#     the cached object immediately (readers see fresh progress) but persist
#     to disk at most once per ``JOB_PROGRESS_FLUSH_INTERVAL`` seconds, with a
#     guaranteed trailing flush. Anything else — field writes, status changes,
#     terminal statuses, whole-object saves — writes through unchanged, so
#     crash recovery loses at most ~2 s of progress-bar position.
#   * compact JSON via ``fastjson`` (orjson when installed) — pretty output
#     is available behind ``JOB_JSON_PRETTY`` for debugging.
#
# Everything here is fail-soft: any error falls back to the exact previous
# load-from-disk / write-through path.

_JOB_CACHE_MAX = 8
# path -> (JobResult, st_mtime_ns of the file the object mirrors)
_job_cache: "OrderedDict[str, tuple[JobResult, int]]" = OrderedDict()
# paths whose cached object carries progress not yet persisted to disk
_dirty_jobs: set[str] = set()
# path -> monotonic time of the last disk persist
_last_flush: dict[str, float] = {}
# path -> scheduled trailing-flush task
_pending_flush_tasks: dict[str, asyncio.Task] = {}


def _progress_flush_interval() -> float:
    try:
        from backend.config import settings
        return float(getattr(settings, "JOB_PROGRESS_FLUSH_INTERVAL", 2.0) or 0.0)
    except Exception:
        return 2.0


def _job_json_pretty() -> bool:
    try:
        from backend.config import settings
        return bool(getattr(settings, "JOB_JSON_PRETTY", False))
    except Exception:
        return False


def _cache_put(path: str, job: JobResult, mtime_ns: int) -> None:
    try:
        _job_cache[path] = (job, mtime_ns)
        _job_cache.move_to_end(path)
        while len(_job_cache) > _JOB_CACHE_MAX:
            _evicted, _ = _job_cache.popitem(last=False)
            _dirty_jobs.discard(_evicted)
            _last_flush.pop(_evicted, None)
    except Exception:
        pass


def _invalidate_job_cache(job_id: str) -> None:
    """Drop a job's cache entry (deletion, or an out-of-band direct write
    like the pipeline's ``_persist_complete_job``). Safe to call anytime."""
    try:
        path = _job_path(job_id)
        _job_cache.pop(path, None)
        _dirty_jobs.discard(path)
        _last_flush.pop(path, None)
        task = _pending_flush_tasks.pop(path, None)
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
    except Exception:
        pass


def _reset_job_cache_for_tests() -> None:
    for _t in list(_pending_flush_tasks.values()):
        try:
            _t.cancel()
        except Exception:
            pass
    _pending_flush_tasks.clear()
    _job_cache.clear()
    _dirty_jobs.clear()
    _last_flush.clear()


def _job_dir(job_id: str) -> str:
    return f"/data/uploads/{job_id}"


def _job_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "job.json")


def _coerce_segments(rows):
    """Coerce a transcript list to ``TranscriptSegment`` instances.

    Several callers assign ``transcript`` / ``translated_transcript`` as plain
    dicts — the transcript editor dumps segments to dicts before saving, and the
    on-read self-heal sanitizes to dicts. Pydantic does NOT re-validate on plain
    attribute assignment, so those dicts sit in a ``list[TranscriptSegment]``
    field unconverted, and every later ``job.model_dump()`` then emits a
    ``PydanticSerializationUnexpectedValue`` warning PER CUE. On a long
    transcript that is thousands of synchronous stderr writes on each save —
    enough to stall the event loop so the GUI can't load (the reported symptom).

    Coercing to models keeps serialization clean and fast. Defensive: fills
    missing required fields and never raises (a hopelessly malformed cue is
    dropped rather than crashing the save).
    """
    if not isinstance(rows, list):
        return rows
    out = []
    for r in rows:
        if isinstance(r, TranscriptSegment):
            out.append(r)
        elif isinstance(r, dict):
            try:
                out.append(TranscriptSegment(**r))
            except Exception:
                try:
                    out.append(TranscriptSegment(
                        start=float(r.get("start") or 0.0),
                        end=float(r.get("end") or r.get("start") or 0.0),
                        text=str(r.get("text") or ""),
                        speaker=str(r.get("speaker") or "Speaker 1"),
                    ))
                except Exception:
                    pass  # drop a malformed cue rather than break the save
        else:
            out.append(r)
    return out


async def _save_job_unlocked(job: JobResult, *, _preserve_terminal_status: bool = False) -> None:
    """Write a job to disk WITHOUT acquiring the per-job lock.

    Callers that already hold ``_get_lock(job_id)`` use this so the whole
    read-modify-write cycle in :func:`update_job_status` is atomic. Public
    :func:`save_job` wraps this with the lock.

    ``_preserve_terminal_status`` (set by :func:`save_job`, the whole-object
    write path used by transcript edits etc.) additionally refuses to revert a
    terminal status — guarding against a stale-snapshot save reverting a finished
    job. ``update_job_status`` leaves it False because its own ``protect_terminal``
    logic already decided the status, and a deliberate re-analysis reset there
    must be allowed through.
    """
    directory = _job_dir(job.job_id)
    os.makedirs(directory, exist_ok=True)
    path = _job_path(job.job_id)
    # Normalize transcript fields to TranscriptSegment instances before dumping,
    # so model_dump() doesn't emit a per-cue serialization warning for any dicts
    # a caller assigned (see _coerce_segments). This is the single chokepoint all
    # save paths pass through. Run the coercion + model_dump OFF the event loop:
    # serializing a job bloated by a corrupted run (tens of thousands of cues) is
    # heavy enough to stall the loop on every progress write during a run.
    def _coerce_and_dump():
        for _tk in ("transcript", "raw_transcript", "translated_transcript",
                    "translated_raw_transcript"):
            _rows = getattr(job, _tk, None)
            if isinstance(_rows, list) and any(isinstance(r, dict) for r in _rows):
                try:
                    setattr(job, _tk, _coerce_segments(_rows))
                except Exception:
                    pass
        dumped = job.model_dump(mode="json")
        # Re-validate the dump for the cache: callers assign raw strings /
        # dicts to fields (Pydantic does NOT validate plain assignment), and
        # before the cache existed every reader got a freshly-validated object
        # from disk. Caching this round-tripped model preserves exactly those
        # read semantics (status enums, nested models) at no extra I/O.
        try:
            revalidated = JobResult(**dumped)
        except Exception:
            revalidated = None
        return dumped, revalidated
    data, _cacheable_job = await asyncio.to_thread(_coerce_and_dump)

    # ── Anti-clobber guard (whole-object save_job path only) ────────────
    # A load → modify → save_job with a job captured just before a newer write
    # (e.g. a transcript-segment edit / reverse-sync that loaded the job
    # microseconds before the pipeline persisted the translation) would
    # otherwise WIPE the newer ``translated_transcript`` and revert a finished
    # status — exactly the corruption seen in production (translated_transcript
    # back to 0, status reverted to detecting_clips). Re-read the current on-disk
    # copy (we're under the per-job lock) and refuse to DOWNGRADE:
    #   • never replace a non-empty translated_transcript with an empty one
    #     (a real re-translation writes a NEW non-empty value, which still wins);
    #   • never revert a terminal status (complete/failed/cancelled).
    # ``update_job_status`` skips this — it loads fresh under the lock (so it
    # never carries a stale translation) and its protect_terminal logic already
    # governs status — which also avoids a re-read on every progress write.
    # When a guard below rewrites ``data`` (keeping on-disk fields the incoming
    # object lost), the in-memory ``job`` no longer mirrors what lands on disk
    # — so the cache entry is invalidated instead of updated (next load
    # re-parses the guarded on-disk copy, exactly as before the cache existed).
    _guard_adjusted = False
    if _preserve_terminal_status:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as _cf:
                    _cur = json.load(_cf)
                if (_cur.get("translated_transcript") or []) and not (data.get("translated_transcript") or []):
                    _guard_adjusted = True
                    data["translated_transcript"] = _cur["translated_transcript"]
                    logger.warning(
                        "Save guard [%s]: kept %d existing translated_transcript "
                        "segment(s) — incoming save had none (stale snapshot).",
                        job.job_id, len(_cur["translated_transcript"]))
                    # The readability card scores the TRANSLATED transcript. A
                    # stale snapshot that lost the translated cues also carries
                    # the PRELIMINARY source-language readability (the raw source
                    # transcript, scored with CJK limits → a lower grade), which
                    # would revert the card from the shipped-subtitle score (e.g.
                    # A) back to the source score (e.g. C). Keep the persisted
                    # readability alongside the translated cues it describes.
                    if (_cur.get("transcript_readability")
                            and _cur.get("transcript_readability")
                            != data.get("transcript_readability")):
                        _guard_adjusted = True
                        data["transcript_readability"] = _cur["transcript_readability"]
                        logger.warning(
                            "Save guard [%s]: kept existing transcript_readability — "
                            "incoming save carried the stale source-language score.",
                            job.job_id)
                # Refuse to replace a CLEANER translated track with a DIRTIER one.
                # A stale snapshot captured mid-translation (or a resume that merged
                # source cues back in after a long reconnect) can carry the SAME cue
                # count but more source-language (untranslated) text — the empty
                # check above misses it because neither side is empty. Compare the
                # source-script fraction and keep whichever is more fully translated;
                # a genuine re-translation is CLEANER (lower fraction) so it still
                # wins. Skipped for CJK targets (CJK output is correct there).
                _cur_tt = _cur.get("translated_transcript") or []
                _new_tt = data.get("translated_transcript") or []
                if _cur_tt and _new_tt:
                    try:
                        from backend.services.translator import fraction_untranslated
                        _tgt = (data.get("subtitle_language")
                                or _cur.get("subtitle_language") or "en")
                        _cur_frac = fraction_untranslated(_cur_tt, _tgt)
                        _new_frac = fraction_untranslated(_new_tt, _tgt)
                        if _new_frac > _cur_frac + 0.02:
                            _guard_adjusted = True
                            data["translated_transcript"] = _cur_tt
                            if _cur.get("transcript_readability"):
                                data["transcript_readability"] = _cur["transcript_readability"]
                            logger.warning(
                                "Save guard [%s]: kept cleaner translated_transcript "
                                "(%.0f%% source-script) over a dirtier incoming save "
                                "(%.0f%%) — stale/reconnect snapshot.",
                                job.job_id, 100 * _cur_frac, 100 * _new_frac)
                    except Exception:
                        pass
                # Same anti-wipe for the video summary. It is written ONCE by the
                # summary stage via ``update_job_status(job_id, summary=...)`` — a
                # separate DB write that does NOT update the in-memory pipeline job.
                # A later whole-object ``save_job()`` of that still-stale in-memory
                # job (summary=None) would otherwise clobber the persisted summary,
                # leaving the Summary tab on "No summary available" even though
                # generation succeeded (Stage 'summary' finished in the log). Keep
                # the existing summary whenever the incoming save lacks one; a real
                # re-analysis writes a NEW non-empty summary, which still wins.
                if _cur.get("summary") and not data.get("summary"):
                    _guard_adjusted = True
                    data["summary"] = _cur["summary"]
                    logger.warning(
                        "Save guard [%s]: kept existing summary — incoming save had "
                        "none (stale snapshot).", job.job_id)
                _cur_status = str(_cur.get("status", "") or "").lower()
                _new_status = str(data.get("status", "") or "").lower()
                if _cur_status in _TERMINAL_STATUSES and _new_status not in _TERMINAL_STATUSES:
                    _guard_adjusted = True
                    data["status"] = _cur.get("status")
                    if "progress" in _cur:
                        data["progress"] = _cur.get("progress")
                    # A stale pre-terminal snapshot also carries stale ``clips``
                    # (often 0 — captured before clip extraction finished). The
                    # finalize already persisted the real clips, so a late
                    # ``detecting_clips`` relay save must not wipe them back to 0.
                    # Tied to the terminal-revert signal so a legitimate clip
                    # delete (which keeps the status terminal) is unaffected.
                    if (_cur.get("clips") or []) and not (data.get("clips") or []):
                        data["clips"] = _cur["clips"]
                        logger.warning(
                            "Save guard [%s]: kept %d existing clip(s) — incoming save "
                            "reverted a terminal status (stale snapshot).",
                            job.job_id, len(_cur["clips"]))
                    logger.warning(
                        "Save guard [%s]: kept terminal status '%s' — incoming save "
                        "tried to revert it to '%s' (stale snapshot).",
                        job.job_id, _cur_status, _new_status)
        except Exception:
            pass

    # Encode off the event loop too (a multi-MB job dict is slow to stringify).
    # Compact by default (fastjson → orjson when installed); pretty output is
    # a debugging aid behind JOB_JSON_PRETTY.
    if _job_json_pretty():
        payload = (await asyncio.to_thread(
            json.dumps, data, indent=2, default=_numpy_safe_default)).encode("utf-8")
    else:
        payload = await asyncio.to_thread(
            fastjson.dumps_bytes, data, _numpy_safe_default)
    # Atomic write: write to temp file then rename to prevent readers
    # from seeing a truncated/empty file during concurrent access.
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        async with aiofiles.open(fd, "wb", closefd=True) as f:
            await f.write(payload)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # ── Cache upkeep (fail-soft: a failure here only costs a re-parse) ──
    try:
        _dirty_jobs.discard(path)
        _last_flush[path] = time.monotonic()
        _pending = _pending_flush_tasks.get(path)
        if (_pending is not None and _pending is not asyncio.current_task()
                and not _pending.done()):
            # A write-through supersedes any scheduled trailing flush.
            _pending_flush_tasks.pop(path, None)
            _pending.cancel()
        if _guard_adjusted or _cacheable_job is None:
            # Disk carries guard-restored fields the in-memory object lost
            # (or re-validation failed) — don't cache a divergent object;
            # the next load re-parses the file.
            _job_cache.pop(path, None)
        else:
            _cache_put(path, _cacheable_job, os.stat(path).st_mtime_ns)
    except Exception as _cache_err:
        logger.info("job cache update skipped for %s: %s", job.job_id, _cache_err)
        _job_cache.pop(path, None)


async def _load_job_unlocked(job_id: str) -> Optional[JobResult]:
    """Read a job from disk WITHOUT acquiring the per-job lock.

    Companion to :func:`_save_job_unlocked` for the atomic
    read-modify-write in :func:`update_job_status`.
    """
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    # Cache probe: serve the parsed object when the file hasn't changed since
    # it was cached (we hold the per-job lock; every writer in this module
    # updates or invalidates the entry, and the mtime check catches direct
    # writers like _persist_complete_job and manual on-disk edits).
    try:
        cached = _job_cache.get(path)
        if cached is not None:
            if os.stat(path).st_mtime_ns == cached[1]:
                _job_cache.move_to_end(path)
                return cached[0]
            _job_cache.pop(path, None)
            _dirty_jobs.discard(path)
    except Exception:
        pass
    try:
        stat_before = None
        try:
            stat_before = os.stat(path).st_mtime_ns
        except OSError:
            pass
        async with aiofiles.open(path, "rb") as f:
            content = await f.read()
        if not content.strip():
            logger.warning("Empty job.json for %s, treating as not found", job_id)
            return None
        # Parse + validate OFF the event loop. A job bloated by a corrupted run
        # (tens of thousands of transcript cues, each with per-word timestamps)
        # makes json.loads + Pydantic validation take long enough to stall the
        # loop — which surfaces as "Connection lost" in the UI and every other
        # request hanging while a transcript "loads forever". load_job runs on
        # every poll, so this is the hot path.
        job = await asyncio.to_thread(lambda: JobResult(**fastjson.loads(content)))
        # mtime captured BEFORE the read: if the file changed mid-read the
        # cached mtime won't match the file's next stat and we re-parse.
        if job is not None and stat_before is not None:
            _cache_put(path, job, stat_before)
        return job
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load job %s: %s", job_id, e)
        return None


async def save_job(job: JobResult) -> None:
    # Whole-object write (callers pass a job they loaded earlier and mutated),
    # so guard against a stale snapshot reverting a terminal status. The
    # translated_transcript anti-wipe guard in _save_job_unlocked applies to
    # every write regardless.
    async with _get_lock(job.job_id):
        await _save_job_unlocked(job, _preserve_terminal_status=True)


async def load_job(job_id: str) -> Optional[JobResult]:
    async with _get_lock(job_id):
        return await _load_job_unlocked(job_id)


def _parse_job_record(content: str, light: bool) -> JobResult:
    """Parse + validate a job.json string into a JobResult — meant to run OFF
    the event loop via ``asyncio.to_thread``.

    ``light=True`` drops the heavy per-cue transcript arrays before validation.
    ``list_jobs`` callers only read summary fields, status, clips and summary —
    never the transcripts — and validating thousands of TranscriptSegment /
    WordTimestamp models per job is what made the dashboard job list (and the
    ``/api/health`` ping behind the "Connecting to container…" banner) crawl once
    a job's transcript ballooned.
    """
    data = json.loads(content)
    if light and isinstance(data, dict):
        data.pop("transcript", None)
        data.pop("translated_transcript", None)
    return JobResult(**data)


async def list_jobs(
    *,
    owner_user_id: Optional[str] = None,
    owner_username: Optional[str] = None,
    include_unowned: bool = False,
    light: bool = False,
) -> list[JobResult]:
    """List all jobs on disk.

    When ``owner_user_id`` is given, only that user's jobs are returned.
    ``owner_username`` is an additional stable fallback match — a job
    is included when EITHER its stored UUID matches ``owner_user_id``
    OR its stored ``owner_username`` matches (case-insensitive). The
    username match recovers jobs whose stamped UUID no longer resolves
    (e.g. the auth store was regenerated after a restart because
    ``/data/auth`` wasn't persisted as a docker volume yet).
    ``include_unowned`` additionally surfaces jobs created before
    multi-user auth shipped (``owner_user_id == ""``); this is used by
    the admin view so legacy jobs remain accessible.
    """
    jobs = []
    # Derive the uploads root from _job_dir so tests that monkeypatch
    # _job_dir (and any future relocation) are honored by list_jobs too,
    # instead of this hard-coding /data/uploads independently.
    uploads_dir = os.path.dirname(_job_dir("_"))
    if not os.path.exists(uploads_dir):
        return jobs
    norm_username = (owner_username or "").strip().lower() or None
    for entry in os.listdir(uploads_dir):
        job_path = os.path.join(uploads_dir, entry, "job.json")
        if os.path.isfile(job_path):
            try:
                async with aiofiles.open(job_path, "r") as f:
                    content = await f.read()
                # Parse + validate off the loop; ``light`` skips the heavy
                # transcript arrays so a bloated job can't stall the whole list.
                job = await asyncio.to_thread(_parse_job_record, content, light)
                if owner_user_id is not None:
                    row_owner = job.owner_user_id or ""
                    row_name = (getattr(job, "owner_username", "") or "").strip().lower()
                    matches_uid = bool(row_owner) and row_owner == owner_user_id
                    matches_name = (
                        norm_username is not None
                        and bool(row_name)
                        and row_name == norm_username
                    )
                    if not (matches_uid or matches_name):
                        if not (include_unowned and row_owner == ""):
                            continue
                jobs.append(job)
            except Exception as e:
                logger.warning("Failed to load job from %s: %s", job_path, e)
                continue
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs


async def delete_job(job_id: str) -> bool:
    """Delete a job and all its associated files (uploads + outputs). Returns True if deleted.

    IMPORTANT: The global media library (_library) is never deleted via this path.
    Job-specific user-uploaded media in /data/uploads/{job_id}/media/ is preserved
    by moving it to the global library before the job directory is removed, so that
    user uploads survive job deletion and can only be deleted explicitly via the
    DELETE /api/media/{id} endpoint.
    """
    directory = _job_dir(job_id)
    if not os.path.exists(directory):
        return False

    # Guard: never delete the global media library
    if job_id == "_library":
        logger.warning("Blocked attempt to delete global media library via delete_job")
        return False

    lock = _get_lock(job_id)
    async with lock:
        # Preserve user-uploaded media by moving files to the global library.
        # This ensures uploads survive job deletion and can only be removed by
        # explicit user action (DELETE /api/media/{id}).
        job_media_dir = os.path.join(directory, "media")
        if os.path.isdir(job_media_dir):
            global_media_dir = os.path.join("/data/uploads", "_library", "media")
            os.makedirs(global_media_dir, exist_ok=True)

            # Load metadata from both source and destination
            src_meta_path = os.path.join(job_media_dir, "_meta.json")
            dst_meta_path = os.path.join(global_media_dir, "_meta.json")
            src_meta = {}
            dst_meta = {}
            try:
                if os.path.isfile(src_meta_path):
                    with open(src_meta_path) as f:
                        src_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
            try:
                if os.path.isfile(dst_meta_path):
                    with open(dst_meta_path) as f:
                        dst_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

            # Move each media file to the global library
            for fname in os.listdir(job_media_dir):
                if fname.startswith("_"):
                    continue  # skip metadata files
                src_path = os.path.join(job_media_dir, fname)
                if not os.path.isfile(src_path):
                    continue
                dst_path = os.path.join(global_media_dir, fname)
                if not os.path.exists(dst_path):
                    try:
                        shutil.move(src_path, dst_path)
                        media_id = os.path.splitext(fname)[0]
                        if media_id in src_meta:
                            dst_meta[media_id] = src_meta[media_id]
                    except OSError:
                        pass

            # Save updated global metadata
            try:
                with open(dst_meta_path, "w") as f:
                    json.dump(dst_meta, f)
            except OSError:
                pass

        _invalidate_job_cache(job_id)
        shutil.rmtree(directory, ignore_errors=True)
        # Also clean up exported clips / output files
        output_dir = f"/data/outputs/{job_id}"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
    _file_locks.pop(job_id, None)
    return True


async def update_job_thumbnail(job_id: str, thumbnail_path: str) -> Optional[JobResult]:
    """Store the thumbnail path for a job."""
    return await update_job_status(job_id, thumbnail_path=thumbnail_path)


async def get_job(job_id: str) -> Optional[JobResult]:
    """Alias for load_job, used by OG injection and share routes."""
    return await load_job(job_id)


# Statuses past which a job is finished. A late in-flight progress write
# must never drag a job back out of one of these.
_TERMINAL_STATUSES = {"complete", "failed", "cancelled"}


def _status_value(status) -> str:
    """Lower-cased string for a JobStatus enum or raw string."""
    if status is None:
        return ""
    return (status.value if hasattr(status, "value") else str(status)).lower()


def _classify_progress_only(cur_status: str, incoming_status, kwargs: dict) -> bool:
    """True when an ``update_job_status`` call changes nothing but progress.

    Progress-only ⇔ no extra fields, the status (when provided) equals the
    current one, and neither side is terminal. Terminal writes and field
    writes must always hit disk immediately.
    """
    if kwargs:
        return False
    if cur_status in _TERMINAL_STATUSES:
        return False
    if incoming_status is not None:
        if incoming_status != cur_status:
            return False
        if incoming_status in _TERMINAL_STATUSES:
            return False
    return True


def _schedule_trailing_flush(job_id: str, path: str, delay: float) -> bool:
    """Arm the trailing flush for a debounced progress write.

    Returns False when a task can't be scheduled (no running loop) so the
    caller falls back to an immediate write-through.
    """
    existing = _pending_flush_tasks.get(path)
    if existing is not None and not existing.done():
        return True  # one pending flush is enough — it writes the LATEST state

    async def _flush_later():
        try:
            await asyncio.sleep(max(0.05, delay))
            async with _get_lock(job_id):
                if path not in _dirty_jobs:
                    return
                entry = _job_cache.get(path)
                if entry is None:
                    _dirty_jobs.discard(path)
                    return
                await _save_job_unlocked(entry[0])
        except asyncio.CancelledError:
            pass  # superseded by a write-through, or the job was deleted
        except Exception as exc:
            logger.warning("[%s] trailing progress flush failed: %s", job_id, exc)
        finally:
            if _pending_flush_tasks.get(path) is asyncio.current_task():
                _pending_flush_tasks.pop(path, None)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    try:
        _pending_flush_tasks[path] = loop.create_task(_flush_later())
        return True
    except Exception:
        return False


def _maybe_defer_progress_write(job: JobResult) -> bool:
    """Debounce a progress-only persist. Caller holds the per-job lock and has
    already mutated ``job`` (which must be the cached object, so readers see
    the fresh progress immediately).

    Returns True when the disk write was deferred (a trailing flush is armed),
    False when the caller must write through now.
    """
    try:
        interval = _progress_flush_interval()
        if interval <= 0:
            return False
        path = _job_path(job.job_id)
        entry = _job_cache.get(path)
        if entry is None or entry[0] is not job:
            # Can't guarantee readers see this object — write through.
            return False
        last = _last_flush.get(path)
        if last is None:
            return False
        now = time.monotonic()
        elapsed = now - last
        if elapsed >= interval:
            return False
        if not _schedule_trailing_flush(job.job_id, path, interval - elapsed):
            return False
        _dirty_jobs.add(path)
        return True
    except Exception as exc:
        logger.info("[%s] progress debounce skipped (%s); writing through",
                    getattr(job, "job_id", "?"), exc)
        return False


def _cjk_heavy_text(text: str) -> bool:
    """True when a cue is predominantly CJK script (i.e. untranslated source).

    Mirrors ``translator._cjk_ratio > 0.30`` but dependency-free, so the DB
    layer can judge translation purity without importing the heavy
    translator/orchestrator chain (openai/genai/…)."""
    t = text or ""
    cjk = base = 0
    for c in t:
        if ("぀" <= c <= "ヿ" or "㐀" <= c <= "鿿"
                or "가" <= c <= "힣" or "ｦ" <= c <= "ﾟ"):
            cjk += 1
            base += 1
        elif c.isalpha():
            base += 1
    return base > 0 and (cjk / base) > 0.30


def _source_script_fraction(rows) -> float:
    """Fraction of cues still written predominantly in CJK source script."""
    rows = list(rows or [])
    if not rows:
        return 0.0
    n = sum(1 for s in rows if _cjk_heavy_text(
        (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")) or ""))
    return n / len(rows)


async def update_job_status(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    progress_message: Optional[str] = None,
    protect_terminal: bool = False,
    **kwargs,
) -> Optional[JobResult]:
    # The entire load → modify → save runs under the per-job lock so
    # concurrent writers can't clobber each other's fields. Previously
    # this read the job, then saved it in two separately-locked steps:
    # a progress callback fired from the clipper worker thread (via
    # ``run_coroutine_threadsafe``) could load the pre-COMPLETE snapshot,
    # then save it back AFTER the COMPLETE save landed — reverting status
    # to ``detecting_clips`` and wiping ``clips`` / ``translated_transcript``.
    # That single race is what surfaced as the pipeline being "stuck
    # finalizing clips", the clip list reading back empty, and the
    # translated subtitle track silently reverting to the source language.
    #
    # ``protect_terminal`` is opt-in (set by the progress relay in
    # ``_update_progress``). When True, a stale in-progress write can't
    # drag a COMPLETE / FAILED / CANCELLED job back to an unfinished
    # state. Deliberate restarts (re-analysis, retranscribe) leave it
    # False so they can legitimately move a finished job back into the
    # pipeline.
    async with _get_lock(job_id):
        job = await _load_job_unlocked(job_id)
        if job is None:
            return None

        cur_status = _status_value(getattr(job, "status", ""))
        incoming_status = _status_value(status) if status is not None else None
        is_terminal = cur_status in _TERMINAL_STATUSES

        if protect_terminal and is_terminal:
            # Block a stale relay from reverting the finished status or
            # ticking its progress bar backwards / wiping its fields.
            if incoming_status is not None and incoming_status not in _TERMINAL_STATUSES:
                logger.info(
                    "[%s] Ignoring stale '%s' progress update on terminal job (status=%s)",
                    job_id, incoming_status, cur_status,
                )
                return job

        if status is not None:
            # Same-value writes keep the existing (validated) attribute so a
            # debounced update can't swap the cached enum for a raw string;
            # the serialized value is identical either way.
            if incoming_status != cur_status:
                job.status = status
        if progress is not None:
            # Don't tick a finished job's bar backwards on a late
            # field-only update that happens to carry an old progress value.
            if not (protect_terminal and is_terminal and status is None):
                job.progress = progress
        if progress_message is not None:
            job.progress_message = progress_message
        from datetime import datetime, timezone
        job.updated_at = datetime.now(timezone.utc).isoformat()

        # ── Translation purity guard ───────────────────────────────────────
        # Never let a re-analyze / orphan-recovery pass that fell back to
        # Whisper-native translate (which leaves music/narration in the source
        # language) CLOBBER a clean editorial-LLM translation with a half-source
        # one. Seen in production: the LLM wrote 0%-source English, then a second
        # pass overwrote translated_transcript with ~60% Japanese — the recurring
        # "translated track came back half source-language" bug. A genuinely
        # better/equal translation still wins; only a REGRESSION to materially
        # more source-script is refused, and only when the target is non-CJK
        # (a →ja/zh/ko translation legitimately contains CJK).
        if "translated_transcript" in kwargs:
            _incoming_tt = kwargs.get("translated_transcript") or []
            _existing_tt = getattr(job, "translated_transcript", None) or []
            _tgt = (kwargs.get("subtitle_language")
                    or getattr(job, "subtitle_language", "") or "").lower().split("-")[0]
            if _incoming_tt and _existing_tt and _tgt not in ("ja", "ko", "zh", "yue"):
                try:
                    _ein = _source_script_fraction(_incoming_tt)
                    _eex = _source_script_fraction(_existing_tt)
                    if _eex <= 0.15 and _ein >= _eex + 0.15:
                        logger.warning(
                            "[%s] Translation purity guard: refused to overwrite a clean "
                            "translated_transcript (%.0f%% source-script, %d cues) with a "
                            "half-source one (%.0f%% source-script, %d cues) — kept the clean "
                            "translation.",
                            job_id, 100 * _eex, len(_existing_tt), 100 * _ein, len(_incoming_tt))
                        kwargs.pop("translated_transcript", None)
                except Exception:
                    pass

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        # Progress-only updates (no field changes, no status change, neither
        # side terminal) are debounced: the cached object above already carries
        # the new progress for every reader; disk sees it at most once per
        # JOB_PROGRESS_FLUSH_INTERVAL with a guaranteed trailing flush.
        if (_classify_progress_only(cur_status, incoming_status, kwargs)
                and _maybe_defer_progress_write(job)):
            return job
        await _save_job_unlocked(job)
        return job
