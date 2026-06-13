"""Render a document to audio: segment → per-segment wav cache → assemble.

The pipeline is engine-agnostic and dependency-injected (it talks to a
``model_manager`` and a ``voice_library``), which keeps it unit-testable with a
fake engine and lets it enforce the project's two hard rules:

* **Cache by content hash** -- a segment whose text/inflection/engine is
  unchanged is loaded from ``project_cache/<hash>.wav`` instead of re-rendered,
  so editing one highlighted phrase only re-renders that phrase.
* **Batch by engine** -- all of one engine's segments render before swapping
  models, so a mixed-engine document costs the minimum number of VRAM swaps.

Assembly always re-runs (it is cheap) using whatever per-segment wavs are
current, which is what makes pause/crossfade tweaks instant.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import REFERENCE_SR, Config
from ..document.segmenter import SegmentJob, segment_document
from ..document.spans import Document
from ..models.engine_base import SynthRequest
from ..models.model_manager import resolve_device, vram_snapshot
from . import assemble

log = logging.getLogger("inflect.pipeline")

ProgressCb = Callable[[int, int, str], None]  # (done, total, message)
CancelCb = Callable[[], bool]


class PipelineCancelled(Exception):
    """Raised when a render is cancelled between segments."""


class PipelineError(RuntimeError):
    """A render could not proceed (e.g. missing voice profile)."""


@dataclass
class RenderedSegment:
    seg_id: int
    char_start: int
    char_end: int
    sample_rate: int
    n_samples: int
    pause_after_ms: int
    engine: str
    from_cache: bool
    synth_seconds: float
    audio: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate if self.sample_rate else 0.0


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    segments: list[RenderedSegment]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def segment_offsets(self) -> list[tuple[int, float, float]]:
        """(seg_id, start_seconds, end_seconds) in the assembled timeline.

        Approximate: ignores crossfade overlap shrink (a few ms per join) but is
        exact enough to drive timeline markers and click-to-seek.
        """
        offsets: list[tuple[int, float, float]] = []
        t = 0.0
        for seg in self.segments:
            start = t
            t += seg.duration_s
            offsets.append((seg.seg_id, start, t))
            t += seg.pause_after_ms / 1000.0
        return offsets


class SynthesisPipeline:
    def __init__(self, config: Config, model_manager, voice_library) -> None:
        self.config = config
        self.mm = model_manager
        self.voices = voice_library
        self.cache_dir = config.paths.cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------
    def render(
        self,
        document: Document,
        engine: str,
        engine_params: dict | None = None,
        *,
        should_cancel: CancelCb | None = None,
        progress: ProgressCb | None = None,
    ) -> RenderResult:
        jobs = segment_document(document, engine, engine_params or {})
        if not jobs:
            return RenderResult(np.zeros(0, np.float32), REFERENCE_SR, [])
        device = resolve_device(self.config.settings.use_cuda)
        speaker_wav = self._resolve_speaker(document)

        rendered: dict[int, RenderedSegment] = {}
        total = len(jobs)
        done = 0

        for engine_name, group in _group_by_engine(jobs):
            eng = None  # lazily fetched so cache-only renders never load a model
            for job in group:
                if should_cancel and should_cancel():
                    raise PipelineCancelled()
                if progress:
                    progress(done, total, f"Segment {done + 1}/{total} ({engine_name})")
                seg = self._render_job(job, engine_name, speaker_wav, device, lambda: self._lazy_engine(engine_name, device))
                rendered[job.seg_id] = seg
                done += 1

        if progress:
            progress(total, total, "Assembling…")
        return self._assemble(jobs, rendered)

    def render_one(
        self, document: Document, job: SegmentJob, *, force: bool = False
    ) -> RenderedSegment:
        """Render (or load-from-cache) a single segment -- the 'Preview' action."""
        device = resolve_device(self.config.settings.use_cuda)
        speaker_wav = self._resolve_speaker(document)
        return self._render_job(
            job, job.engine, speaker_wav, device,
            lambda: self.mm.get_engine(job.engine, device), force=force,
        )

    # -- internals ---------------------------------------------------------
    def _lazy_engine(self, engine_name: str, device: str):
        return self.mm.get_engine(engine_name, device)

    def _render_job(
        self,
        job: SegmentJob,
        engine_name: str,
        speaker_wav: str,
        device: str,
        engine_getter: Callable[[], object],
        *,
        force: bool = False,
    ) -> RenderedSegment:
        cache_path = self.cache_dir / f"{job.hash}.wav"
        if cache_path.exists() and not force:
            audio, sr = _read_wav(cache_path)
            log.debug("cache hit seg %s (%s)", job.seg_id, job.hash[:8])
            return RenderedSegment(
                seg_id=job.seg_id,
                char_start=job.char_start,
                char_end=job.char_end,
                sample_rate=sr,
                n_samples=len(audio),
                pause_after_ms=job.pause_after_ms,
                engine=engine_name,
                from_cache=True,
                synth_seconds=0.0,
                audio=audio,
            )

        engine = engine_getter()
        t0 = time.perf_counter()
        audio = engine.synthesize(SynthRequest(job=job, speaker_wav=speaker_wav, device=device))
        dt = time.perf_counter() - t0
        sr = int(getattr(engine, "sample_rate", 24_000))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        _write_wav(cache_path, audio, sr)
        vram = vram_snapshot()
        log.info(
            "seg %s rendered in %.2fs (%s, %d samp @ %dHz) %s",
            job.seg_id, dt, engine_name, len(audio), sr, vram,
        )
        return RenderedSegment(
            seg_id=job.seg_id,
            char_start=job.char_start,
            char_end=job.char_end,
            sample_rate=sr,
            n_samples=len(audio),
            pause_after_ms=job.pause_after_ms,
            engine=engine_name,
            from_cache=False,
            synth_seconds=dt,
            audio=audio,
        )

    def _assemble(
        self, jobs: list[SegmentJob], rendered: dict[int, RenderedSegment]
    ) -> RenderResult:
        ordered = [rendered[j.seg_id] for j in jobs if j.seg_id in rendered]
        if not ordered:
            return RenderResult(np.zeros(0, np.float32), 24_000, [])

        canonical = _canonical_rate(ordered)
        audio_segs: list[np.ndarray] = []
        pauses: list[int] = []
        for seg in ordered:
            audio = assemble.ensure_rate(seg.audio, seg.sample_rate, canonical)
            audio_segs.append(audio)
            pauses.append(seg.pause_after_ms)

        mix = assemble.assemble_segments(
            audio_segs, pauses, canonical, crossfade_ms=self.config.settings.crossfade_ms
        )
        mix = assemble.master(
            mix,
            canonical,
            target_lufs=self.config.settings.target_lufs,
            true_peak_dbtp=self.config.settings.true_peak_dbtp,
        )
        return RenderResult(mix, canonical, ordered)

    def _resolve_speaker(self, document: Document) -> str:
        pid = document.voice_profile_id
        if not pid:
            raise PipelineError("No voice profile selected. Pick or import a voice first.")
        ref = self.voices.reference_path(pid)
        if not Path(ref).exists():
            raise PipelineError(
                f"Voice profile {pid!r} is missing its reference audio on disk."
            )
        return str(ref)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _group_by_engine(jobs: list[SegmentJob]) -> list[tuple[str, list[SegmentJob]]]:
    """Group consecutive-by-first-appearance engines, minimizing model swaps.

    All jobs for an engine are collected together (even if interleaved in the
    document) so the engine loads once. First-appearance order is preserved.
    """
    order: list[str] = []
    buckets: dict[str, list[SegmentJob]] = {}
    for job in jobs:
        if job.engine not in buckets:
            buckets[job.engine] = []
            order.append(job.engine)
        buckets[job.engine].append(job)
    return [(name, buckets[name]) for name in order]


def _canonical_rate(segments: list[RenderedSegment]) -> int:
    """Pick the assembly rate: IndexTTS-2's native rate if present, else first."""
    for seg in segments:
        if seg.engine == "indextts2":
            return seg.sample_rate
    return segments[0].sample_rate if segments else 24_000


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32).reshape(-1), int(sr)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)
