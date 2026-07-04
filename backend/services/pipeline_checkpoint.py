"""Engine checkpoint — persist the reframer's perception + plan so a
failed / interrupted job can RESUME instead of re-running the single most
expensive stage (face/motion detection + Whisper transcription + planning).

The analysis pipeline already caches frame + audio extraction (keyed on the
source SHA-256, see :mod:`pipeline_helpers`). Everything after that — the
``ReframeEngine.analyze()`` call that produces the in-memory
:class:`PerceptionResult` + :class:`RenderPlan` — used to re-run from scratch
on every re-analyze, and there was nothing to resume from when the container
died mid-run. This module serializes those two objects to a per-job
``checkpoint/`` directory right after the engine stage finishes, and reloads
them on the next run when the source + analysis parameters are unchanged.

A reload returns a perception the post-engine stages (bridge / clipper /
evaluator) consume exactly as a freshly-computed one: the int-keyed timeline
dicts (``face_timeline``, ``motion_timeline`` …) are restored to integer keys
(downstream code does ``int(t_ms / 1000)`` on them, which would crash on the
string keys JSON produces). The only field deliberately dropped is
``coverage_ledger`` — a millisecond-resolution structure read only by the
planner (which has already run by checkpoint time); see the note by
``_INT_KEYED_FIELDS``.

Validity is gated on a *signature* (source SHA + sample-fps + aspect ratio +
source language + vocal-separation setting + a planner-flag fingerprint +
ASR model), so a stale checkpoint is never silently reused after the source
or the analysis settings change.

FUTURE — mid-engine (sub-stage) resume, designed but deliberately not built
yet: the engine checkpoint is written only AFTER the whole engine finishes,
so a death during Whisper still re-runs the visual pass. The clean seam is
in ``Perceiver.run()`` right before the "Audio intelligence" section, where
every visual field on the PerceptionResult (face/motion timelines, scene
cuts, saliency, consolidated tracks) is final. Implementing it requires
splitting ``run()`` into ``_visual_pass(r)`` / ``_audio_pass(r)`` and adding
a signature-gated ``visual.json`` sub-checkpoint restored before the visual
pass — mechanical, but it restructures the guarded perception pipeline and
must be landed with a real end-to-end engine run to verify (cv2/YOLO/Whisper
are not available in the CI sandbox). Post-engine stages already have their
own resume layer (see :mod:`stage_checkpoints`), which removes most of the
re-paid wall clock on resume today.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes so old checkpoints are ignored
# (re-run the engine) instead of deserialized into the wrong fields.
# v2: invalidates checkpoints written before the transcribe-reuses-audio.wav
# fix — those could carry an empty (timed-out) transcript that a resume would
# otherwise keep reusing.
# v3: the ASR model joins the signature — switching Whisper models used to
# silently reuse the previous model's transcript on resume.
CHECKPOINT_VERSION = 3

_PERCEPTION_FILE = "engine_perception.json"
# Compressed variant (gzip'd orjson) — the perception payload is multi-MB
# (per-second timelines) and /data/uploads commonly sits on a parity array,
# so compressing cuts the fsync'd write AND the verify-after-save reload by
# ~5x. Loaders accept either name, so pre-existing plain checkpoints keep
# working; the plan/meta files stay plain (RenderPlan.load reads the plan).
_PERCEPTION_FILE_GZ = "engine_perception.json.gz"
_PLAN_FILE = "engine_plan.json"
_META_FILE = "engine_meta.json"

# Shared, content-addressed engine cache: the same three files keyed by
# (source SHA, signature hash) instead of job id. Re-uploading a video (or
# duplicating a job) hits this and skips detection + transcription entirely
# — the signature already carries exactly the validity semantics needed.
_SHARED_CACHE_ROOT = "/data/uploads/_engine_cache"


def _signature_hash(signature: dict) -> str:
    import hashlib
    canon = json.dumps(signature or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def shared_cache_dir(signature: dict) -> str:
    sha = (signature or {}).get("source_sha") or ""
    return os.path.join(_SHARED_CACHE_ROOT, sha[:16] or "_", _signature_hash(signature))

# Timeline fields whose keys are integers (time_ms, or track_id for the
# speaker map). JSON stringifies dict keys on write — these are restored to
# ``int`` on load so consumers that do arithmetic on the key keep working.
_INT_KEYED_FIELDS = (
    "face_timeline", "motion_timeline", "motion_hotspot", "speaker_timeline",
    "track_speaker_map", "speech_active", "audio_rms", "audio_events",
    "person_timeline", "saliency_hotspot",
)
# NOTE: ``coverage_ledger`` is deliberately NOT checkpointed. It is a
# millisecond-resolution structure (20ms bins → tens of thousands of
# ``LedgerBin`` dataclasses on a long video) consumed ONLY by the planner,
# which runs *inside* ``engine.analyze()`` — i.e. before this checkpoint is
# written. Nothing after the engine (bridge / summary / clip detection) reads
# it. Serializing + rebuilding all those bins in pure Python on a resume held
# the GIL long enough to starve the asyncio event loop, freezing the 2s
# ``/api/diagnostics/gpu-status`` poll (the "VRAM bar stopped working" report).
# Dropping it makes resume cheap; a restored perception simply has
# ``coverage_ledger=None``, which the model documents as the supported
# "fall back to speech_active" state.


def _numpy_safe_default(obj):
    """JSON fallback for numpy scalars/arrays that slip through detection."""
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


def _json_safe_key(k):
    """Coerce a dict key into something ``json.dumps`` accepts.

    ``json.dumps``'s ``default=`` hook rescues VALUES only — a non-str/int key
    (a numpy int from any detector, a float, a tuple) makes the whole
    ``dumps`` call raise *before* ``default`` is ever consulted, which would
    silently lose the entire checkpoint. The engine's timelines are int-keyed
    (``time_ms``), so integral keys are normalized to ``int`` (JSON stringifies
    them, and :func:`_int_keyed` restores them on load); everything else
    degrades to ``str``.
    """
    if isinstance(k, bool):  # bool is an int subclass — keep it distinct
        return k
    if isinstance(k, (str, int)):
        return k
    try:
        import numpy as np
        if isinstance(k, np.integer):
            return int(k)
        if isinstance(k, np.floating):
            f = float(k)
            return int(f) if f.is_integer() else str(f)
    except ImportError:
        pass
    if isinstance(k, float):
        return int(k) if k.is_integer() else str(k)
    return str(k)


def _json_safe(obj):
    """Recursively coerce ``obj`` into a structure ``json.dumps`` cannot choke
    on — including dict KEYS, which the ``default=`` hook never sees.

    This is the durable-write guarantee for the engine checkpoint: a single
    un-serializable leaf (a stray numpy key from a future detector, a ``set``,
    a ``Path``) used to make the save raise, get swallowed by the best-effort
    ``except``, and leave the next revive to re-run detection + transcription
    from scratch. Normalizing the payload up front means the save either writes
    a reloadable checkpoint or fails loudly — never silently.
    """
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_json_safe(v) for v in obj.tolist()]
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {_json_safe_key(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Unknown leaf (dataclass instance, Path, custom object): stringify — the
    # same ultimate fallback the old ``default=`` hook applied to values.
    return str(obj)


def checkpoint_dir(job_id: str) -> str:
    return f"/data/uploads/{job_id}/checkpoint"


def checkpoint_signature(
    *,
    source_sha: str,
    source_language: str,
    sample_fps: float,
    aspect_ratio: str,
    vocal_sep_key: str = "",
    planner_fingerprint: str = "",
    asr_model: str = "",
) -> dict:
    """Identity of the engine inputs. A reload only happens when the saved
    signature matches the one the current run would produce — otherwise the
    cached perception/plan no longer describe this source/config.

    asr_model: the CONFIGURED Whisper model (settings.WHISPER_MODEL at run
    start). Without it, switching models silently reused the old model's
    transcript on resume — a quality hole, not just a staleness one.
    """
    return {
        "version": CHECKPOINT_VERSION,
        "source_sha": source_sha or "",
        "source_language": (source_language or "auto").strip().lower() or "auto",
        "sample_fps": round(float(sample_fps or 0.0), 3),
        "aspect_ratio": (aspect_ratio or "9:16").strip(),
        "vocal_sep_key": vocal_sep_key or "",
        "planner_fingerprint": planner_fingerprint or "",
        "asr_model": (asr_model or "").strip().lower(),
    }


def _signatures_match(saved: dict, expected: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    # An empty source SHA can't be trusted to identify the source — never
    # reuse a checkpoint we can't tie to specific bytes.
    if not expected.get("source_sha"):
        return False
    return all(saved.get(k) == expected.get(k) for k in expected)


def _signature_diff(saved: dict, expected: dict) -> str:
    """Human-readable reason a saved signature doesn't match the expected one,
    for the load-miss log. Makes a non-resuming revive self-diagnosing instead
    of a mystery full re-run."""
    if not isinstance(saved, dict):
        return f"saved signature is not a dict ({type(saved).__name__})"
    if not expected.get("source_sha"):
        return "expected source_sha is empty — no checkpoint can be trusted"
    parts = []
    for k in expected:
        sv, ev = saved.get(k), expected.get(k)
        if sv != ev:
            # SHAs are long; show a prefix so the line stays readable.
            if k == "source_sha":
                sv = (str(sv)[:12] + "…") if sv else "(none)"
                ev = (str(ev)[:12] + "…") if ev else "(none)"
            parts.append(f"{k}: saved={sv!r} != expected={ev!r}")
    return "; ".join(parts) if parts else "saved signature is missing expected keys"


# ── Serialization ────────────────────────────────────────────────────


def _serialize_perception(p) -> dict:
    d = {
        "src_w": int(getattr(p, "src_w", 0) or 0),
        "src_h": int(getattr(p, "src_h", 0) or 0),
        "fps": float(getattr(p, "fps", 30.0) or 30.0),
        "duration_ms": int(getattr(p, "duration_ms", 0) or 0),
        "total_frames": int(getattr(p, "total_frames", 0) or 0),
        "scene_cuts": list(getattr(p, "scene_cuts", None) or []),
        "transcript_segments": list(getattr(p, "transcript_segments", None) or []),
        "detected_language": getattr(p, "detected_language", "") or "",
        # coverage_ledger intentionally omitted — see note by _INT_KEYED_FIELDS.
    }
    for field in _INT_KEYED_FIELDS:
        d[field] = getattr(p, field, None) or {}
    return d


def _int_keyed(d) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def _deserialize_perception(d: dict):
    from backend.services.reframer_models import PerceptionResult
    p = PerceptionResult(
        src_w=int(d.get("src_w", 0) or 0),
        src_h=int(d.get("src_h", 0) or 0),
        fps=float(d.get("fps", 30.0) or 30.0),
        duration_ms=int(d.get("duration_ms", 0) or 0),
        total_frames=int(d.get("total_frames", 0) or 0),
        scene_cuts=[int(c) for c in (d.get("scene_cuts") or [])],
        transcript_segments=list(d.get("transcript_segments") or []),
        detected_language=d.get("detected_language", "") or "",
        # coverage_ledger stays at its None default (not checkpointed).
    )
    for field in _INT_KEYED_FIELDS:
        setattr(p, field, _int_keyed(d.get(field)))
    return p


def _atomic_write(path: str, content) -> None:
    """Durably write ``content`` (str or bytes) to ``path`` via tmp+rename."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        _mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
        with os.fdopen(fd, _mode, **({} if _mode == "wb" else {"encoding": "utf-8"})) as f:
            f.write(content)
            # Flush to disk before the rename so an abrupt container kill (the
            # crash mode this whole module exists for) can't leave the file's
            # bytes in a buffer while the rename has already landed.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # fsync the directory too so the rename itself is durable.
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_sync(job_id: str, perception, reframer_plan, signature: dict,
               audio_meta: dict) -> bool:
    from dataclasses import asdict
    from backend.services import fastjson
    d = checkpoint_dir(job_id)
    # _json_safe() before dumps so a stray un-serializable key/value (numpy,
    # set, Path) can't make the whole save raise — see _json_safe's docstring.
    # default= is kept as a last-ditch net for anything _json_safe missed.
    # fastjson (orjson when installed) encodes the multi-MB perception/plan
    # payloads several times faster than stdlib json; _json_safe has already
    # normalized keys to str/int, which both encoders stringify identically.
    import gzip
    _atomic_write(
        os.path.join(d, _PERCEPTION_FILE_GZ),
        gzip.compress(
            fastjson.dumps_bytes(_json_safe(_serialize_perception(perception)),
                                 default=_numpy_safe_default),
            compresslevel=5),
    )
    # Drop a stale plain-format twin so the loader can't prefer old data.
    try:
        os.unlink(os.path.join(d, _PERCEPTION_FILE))
    except OSError:
        pass
    _atomic_write(
        os.path.join(d, _PLAN_FILE),
        fastjson.dumps_bytes(_json_safe(asdict(reframer_plan)),
                             default=_numpy_safe_default),
    )
    # Meta last: its presence is the "checkpoint is complete + valid" marker,
    # so a crash between the perception/plan writes never yields a half
    # checkpoint that load() would trust.
    _atomic_write(
        os.path.join(d, _META_FILE),
        json.dumps({"signature": signature, "audio": audio_meta or {}}, indent=2),
    )
    # Verify-after-save: reload what we just wrote and confirm it round-trips
    # with THIS signature, while the engine output is still in hand. A write
    # that "succeeded" but can't be reloaded (corrupt JSON, an empty source
    # SHA the loader always rejects, a partial flush) would otherwise surface
    # only on the next revive — as a silent, full re-run of detection +
    # transcription. Failing here turns that into one loud log line instead.
    if _load_sync(job_id, signature) is None:
        raise RuntimeError(
            "verify-after-save failed — the checkpoint just written does not "
            "reload with a matching signature (resume would not work)")
    # Mirror into the shared content-addressed cache so OTHER jobs on the
    # same source + config skip the engine too. Hardlink when the volume
    # allows (free), copy otherwise; best-effort — the per-job checkpoint
    # above is the durability guarantee.
    try:
        import shutil
        shared = shared_cache_dir(signature)
        if (signature or {}).get("source_sha"):
            os.makedirs(shared, exist_ok=True)
            for name in (_PERCEPTION_FILE_GZ, _PLAN_FILE, _META_FILE):
                src_p = os.path.join(d, name)
                dst_p = os.path.join(shared, name)
                try:
                    os.unlink(dst_p)
                except OSError:
                    pass
                try:
                    os.link(src_p, dst_p)
                except OSError:
                    shutil.copy2(src_p, dst_p)
    except Exception as _share_err:  # noqa: BLE001 — sharing is opportunistic
        logger.info("Shared engine-cache mirror skipped: %s", _share_err)
    return True


async def save_engine_checkpoint(
    job_id: str,
    perception,
    reframer_plan,
    *,
    signature: dict,
    audio_meta: Optional[dict] = None,
) -> bool:
    """Persist the engine's perception + plan so the next run can resume.

    Best-effort: serialization runs in a worker thread and any failure is
    logged and swallowed (the next run just re-runs the engine).
    """
    import asyncio
    try:
        await asyncio.to_thread(
            _save_sync, job_id, perception, reframer_plan, signature, audio_meta or {})
        logger.info(
            "[%s] Engine checkpoint saved + verified (%d transcript segs, %d face "
            "samples; sha=%s lang=%s fps=%s) — a resume will skip detection + "
            "transcription",
            job_id,
            len(getattr(perception, "transcript_segments", None) or []),
            sum(1 for v in (getattr(perception, "face_timeline", None) or {}).values() if v),
            (signature.get("source_sha") or "")[:12] or "(none)",
            signature.get("source_language"),
            signature.get("sample_fps"),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — checkpointing must never break a run
        # ERROR, not warning: a failed checkpoint means the NEXT revive of this
        # job will re-run the single most expensive stage from scratch (the
        # "died at translate → restarted faces" report). Make that visible.
        logger.error(
            "[%s] Engine checkpoint save FAILED (%s) — a future resume will have "
            "to RE-RUN detection + transcription from scratch", job_id, exc)
        return False


def _read_perception_bytes(directory: str) -> Optional[bytes]:
    """Perception payload from ``directory`` — gz preferred, plain legacy."""
    import gzip
    gz = os.path.join(directory, _PERCEPTION_FILE_GZ)
    if os.path.isfile(gz):
        with open(gz, "rb") as f:
            return gzip.decompress(f.read())
    plain = os.path.join(directory, _PERCEPTION_FILE)
    if os.path.isfile(plain):
        with open(plain, "rb") as f:
            return f.read()
    return None


def _probe_dir(directory: str, expected_signature: dict, label: str):
    """Try to load a full checkpoint from ``directory``. Returns the parsed
    (meta, perception_bytes, plan_path) or None (with reason logging)."""
    meta_path = os.path.join(directory, _META_FILE)
    plan_path = os.path.join(directory, _PLAN_FILE)
    has_perception = (os.path.isfile(os.path.join(directory, _PERCEPTION_FILE_GZ))
                      or os.path.isfile(os.path.join(directory, _PERCEPTION_FILE)))
    present = {"meta": os.path.isfile(meta_path),
               "plan": os.path.isfile(plan_path),
               "perception": has_perception}
    if not all(present.values()):
        missing = [k for k, ok in present.items() if not ok]
        # A wholly-absent checkpoint is the normal first-run / source-changed
        # case — stay quiet. A PARTIAL one (some files, not all) is a real
        # problem (interrupted write, manual deletion) worth flagging.
        if any(present.values()):
            logger.warning(
                "Engine checkpoint at %s incomplete — missing %s", label, missing)
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not _signatures_match(meta.get("signature") or {}, expected_signature):
        logger.info(
            "Engine checkpoint at %s present but signature differs (%s)",
            label,
            _signature_diff(meta.get("signature") or {}, expected_signature))
        return None
    payload = _read_perception_bytes(directory)
    if payload is None:
        return None
    return meta, payload, plan_path


def _load_sync(job_id: str, expected_signature: dict):
    # Per-job checkpoint first; on a miss, the shared content-addressed
    # cache — another job (or a previous upload of the same file) may have
    # already paid for the engine under this exact signature.
    hit = _probe_dir(checkpoint_dir(job_id), expected_signature, f"job {job_id}")
    if hit is None and (expected_signature or {}).get("source_sha"):
        shared = shared_cache_dir(expected_signature)
        hit = _probe_dir(shared, expected_signature, f"shared cache {shared}")
        if hit is not None:
            logger.info(
                "[%s] Engine checkpoint served from the SHARED cache — another "
                "job already analyzed this source with this config", job_id)
    if hit is None:
        return None
    meta, perception_bytes, plan_path = hit

    from backend.services import fastjson
    perception = _deserialize_perception(fastjson.loads(perception_bytes))

    from backend.services.reframer_models import RenderPlan
    reframer_plan = RenderPlan.load(plan_path)

    audio = meta.get("audio") or {}
    engine_stub = SimpleNamespace(
        _perceiver_audio_device=audio.get("device"),
        _perceiver_audio_model=audio.get("model"),
        _perceiver_audio_model_requested=audio.get("requested"),
        _resumed_from_checkpoint=True,
    )
    return perception, reframer_plan, engine_stub


async def load_engine_checkpoint(job_id: str, *, signature: dict):
    """Return ``(perception, reframer_plan, engine_stub)`` from a valid
    checkpoint, or ``None`` when absent / stale / unreadable.

    ``engine_stub`` is a lightweight stand-in carrying the Whisper device +
    model strings ``_build_compute_summary`` reads, so the Compute card stays
    faithful on a resumed run without a live engine object.
    """
    import asyncio
    try:
        return await asyncio.to_thread(_load_sync, job_id, signature)
    except Exception as exc:  # noqa: BLE001 — a bad checkpoint must fall back to fresh
        logger.warning("[%s] Engine checkpoint load failed (%s) — re-running engine", job_id, exc)
        return None
