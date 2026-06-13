"""Render a document to audio: segment → per-segment wav cache → assemble.

The pipeline is engine-agnostic and dependency-injected (it talks to a
``model_manager`` and a ``voice_library``), which keeps it unit-testable with a
fake engine and lets it enforce the project's hard rules:

* **Cache by content hash** -- a segment whose text/inflection/engine is
  unchanged is loaded from ``project_cache/<hash>.wav`` instead of re-rendered.
* **Batch by engine** -- every task for one engine renders before swapping
  models, so a mixed-engine document costs the minimum number of VRAM swaps.
* **Hybrid Performance Transfer** -- a span with ``engine="hybrid"`` expands into
  two tasks: Fish renders the *performance* (stage 1, default voice) and
  IndexTTS-2 reproduces it in the cloned voice using the stage-1 wav as the
  emotion reference (stage 2). Engine priority (fish → indextts2 → chatterbox)
  orders the batches so a mixed document needs at most three model swaps.

Assembly always re-runs (it is cheap) from whatever per-segment wavs are current.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..config import REFERENCE_SR, Config
from ..document.segmenter import SegmentJob, segment_document, segment_hash
from ..document.spans import Document, Inflection
from ..models.engine_base import SynthRequest
from ..models.engine_fish import inflection_to_tags
from ..models.model_manager import resolve_device, vram_snapshot
from . import assemble

log = logging.getLogger("inflect.pipeline")

ProgressCb = Callable[[int, int, str], None]  # (done, total, message)
CancelCb = Callable[[], bool]

# Lower number == loaded earlier. Fish must precede IndexTTS-2 so hybrid stage 1
# (Fish) is available as the emotion reference for stage 2 (IndexTTS-2).
ENGINE_PRIORITY = {"fish": 0, "indextts2": 1, "chatterbox": 2}


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


@dataclass
class _Task:
    """One unit of engine work. A hybrid job yields two (perf + segment)."""

    seg_id: int
    role: str  # "segment" (final audio) | "perf" (hybrid stage-1 emotion ref)
    engine: str
    text: str
    inflection: Inflection
    voice_profile_id: str | None  # None => engine default voice (Fish stage 1)
    engine_params: dict
    char_start: int
    char_end: int
    cache_path: Path
    hash: str


@dataclass
class _Rendered:
    audio: np.ndarray
    sample_rate: int
    from_cache: bool
    synth_seconds: float
    engine: str


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

        tasks, seg_final = self._plan_tasks(jobs, document.voice_profile_id)
        rendered = self._render_tasks(
            tasks, speaker_wav, device, should_cancel=should_cancel, progress=progress
        )
        if progress:
            progress(len(tasks), len(tasks), "Assembling…")

        ordered = self._segments_in_order(jobs, seg_final, rendered)
        return self._assemble(ordered)

    def render_one(
        self, document: Document, job: SegmentJob, *, force: bool = False
    ) -> RenderedSegment:
        """Render (or load-from-cache) a single segment -- the 'Preview' action.

        Handles hybrid jobs too (renders both stages), returning the final
        IndexTTS-2 segment.
        """
        device = resolve_device(self.config.settings.use_cuda)
        speaker_wav = self._resolve_speaker(document)
        tasks, seg_final = self._plan_tasks([job], document.voice_profile_id)
        force_hashes = {t.hash for t in tasks} if force else set()
        rendered = self._render_tasks(tasks, speaker_wav, device, force_hashes=force_hashes)
        return self._segments_in_order([job], seg_final, rendered)[0]

    def render_performance(self, document: Document, job: SegmentJob) -> RenderedSegment:
        """Render ONLY the Fish stage-1 'performance' for a hybrid job.

        Powers the inspector's "audition performance" button: hear Fish's
        expressive take (in its default voice) before committing to stage 2.
        """
        device = resolve_device(self.config.settings.use_cuda)
        speaker_wav = self._resolve_speaker(document)
        t_perf, _t_seg = self._plan_hybrid(job, document.voice_profile_id)
        rendered = self._render_tasks([t_perf], speaker_wav, device)
        r = rendered[t_perf.hash]
        return RenderedSegment(
            seg_id=job.seg_id,
            char_start=job.char_start,
            char_end=job.char_end,
            sample_rate=r.sample_rate,
            n_samples=len(r.audio),
            pause_after_ms=0,
            engine="fish",
            from_cache=r.from_cache,
            synth_seconds=r.synth_seconds,
            audio=r.audio,
        )

    # -- planning ----------------------------------------------------------
    def _plan_tasks(
        self, jobs: list[SegmentJob], profile_id: str | None
    ) -> tuple[list[_Task], dict[int, str]]:
        """Expand jobs into engine tasks; return tasks + seg_id→final-hash map."""
        tasks: list[_Task] = []
        seg_final: dict[int, str] = {}
        for job in jobs:
            if job.engine == "hybrid":
                t_perf, t_seg = self._plan_hybrid(job, profile_id)
                tasks.extend([t_perf, t_seg])
                seg_final[job.seg_id] = t_seg.hash
            else:
                task = self._plan_simple(job, profile_id)
                tasks.append(task)
                seg_final[job.seg_id] = task.hash
        return tasks, seg_final

    def _plan_simple(self, job: SegmentJob, profile_id: str | None) -> _Task:
        params = dict(job.engine_params)
        if job.engine == "fish":
            # Fish output depends on the tag string -> fold it into the hash.
            params["tags"] = inflection_to_tags(job.inflection)
        h = segment_hash(job.text, job.inflection, profile_id, job.engine, params)
        return _Task(
            seg_id=job.seg_id,
            role="segment",
            engine=job.engine,
            text=job.text,
            inflection=job.inflection,
            voice_profile_id=profile_id,
            engine_params=params,
            char_start=job.char_start,
            char_end=job.char_end,
            cache_path=self.cache_dir / f"{h}.wav",
            hash=h,
        )

    def _plan_hybrid(
        self, job: SegmentJob, profile_id: str | None
    ) -> tuple[_Task, _Task]:
        # Stage 1: Fish renders the performance in its default voice.
        tags = inflection_to_tags(job.inflection)
        stage1_inf = replace(job.inflection, emo_audio=None, pause_after_ms=0, engine=None)
        s1_params = {"tags": tags}
        s1_hash = segment_hash(job.text, stage1_inf, None, "fish", s1_params)
        perf_path = self.cache_dir / f"perf_{s1_hash}.wav"
        t_perf = _Task(
            seg_id=job.seg_id,
            role="perf",
            engine="fish",
            text=job.text,
            inflection=stage1_inf,
            voice_profile_id=None,  # default voice
            engine_params=s1_params,
            char_start=job.char_start,
            char_end=job.char_end,
            cache_path=perf_path,
            hash=s1_hash,
        )
        # Stage 2: IndexTTS-2 reproduces it in the cloned voice, perf wav as emo ref.
        stage2_inf = Inflection(
            emo_audio=str(perf_path),
            emo_alpha=job.inflection.emo_alpha,
            speed=job.inflection.speed,
        )
        s2_hash = segment_hash(job.text, stage2_inf, profile_id, "indextts2", {})
        t_seg = _Task(
            seg_id=job.seg_id,
            role="segment",
            engine="indextts2",
            text=job.text,
            inflection=stage2_inf,
            voice_profile_id=profile_id,
            engine_params={},
            char_start=job.char_start,
            char_end=job.char_end,
            cache_path=self.cache_dir / f"{s2_hash}.wav",
            hash=s2_hash,
        )
        return t_perf, t_seg

    # -- rendering ---------------------------------------------------------
    def _render_tasks(
        self,
        tasks: list[_Task],
        speaker_wav: str,
        device: str,
        *,
        should_cancel: CancelCb | None = None,
        progress: ProgressCb | None = None,
        force_hashes: set[str] | None = None,
    ) -> dict[str, _Rendered]:
        force_hashes = force_hashes or set()
        rendered: dict[str, _Rendered] = {}
        # Group by engine, ordered so hybrid stage 1 (fish) precedes stage 2.
        engines: dict[str, list[_Task]] = {}
        for task in tasks:
            engines.setdefault(task.engine, []).append(task)
        ordered_engines = sorted(engines, key=lambda e: ENGINE_PRIORITY.get(e, 99))

        total = len(tasks)
        done = 0
        for engine_name in ordered_engines:
            engine = None  # lazily loaded so an all-cache batch never loads a model
            for task in engines[engine_name]:
                if should_cancel and should_cancel():
                    raise PipelineCancelled()
                if progress:
                    progress(done, total, f"Segment {done + 1}/{total} ({engine_name})")
                if task.hash in rendered:
                    done += 1
                    continue
                if task.cache_path.exists() and task.hash not in force_hashes:
                    audio, sr = _read_wav(task.cache_path)
                    rendered[task.hash] = _Rendered(audio, sr, True, 0.0, engine_name)
                    done += 1
                    continue
                if engine is None:
                    engine = self.mm.get_engine(engine_name, device)
                rendered[task.hash] = self._render_single(task, engine, engine_name, speaker_wav, device)
                done += 1
        return rendered

    def _render_single(
        self, task: _Task, engine, engine_name: str, speaker_wav: str, device: str
    ) -> _Rendered:
        spk = speaker_wav if task.voice_profile_id else ""
        job = SegmentJob(
            seg_id=task.seg_id,
            text=task.text,
            inflection=task.inflection,
            voice_profile_id=task.voice_profile_id,
            engine=engine_name,
            engine_params=task.engine_params,
            char_start=task.char_start,
            char_end=task.char_end,
        )
        t0 = time.perf_counter()
        audio = engine.synthesize(SynthRequest(job=job, speaker_wav=spk, device=device))
        dt = time.perf_counter() - t0
        sr = int(getattr(engine, "sample_rate", 24_000))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        _write_wav(task.cache_path, audio, sr)
        log.info(
            "seg %s/%s rendered in %.2fs (%s, %d samp @ %dHz) %s",
            task.seg_id, task.role, dt, engine_name, len(audio), sr, vram_snapshot(),
        )
        return _Rendered(audio, sr, False, dt, engine_name)

    # -- assembly ----------------------------------------------------------
    def _segments_in_order(
        self,
        jobs: list[SegmentJob],
        seg_final: dict[int, str],
        rendered: dict[str, _Rendered],
    ) -> list[RenderedSegment]:
        segs: list[RenderedSegment] = []
        for job in jobs:
            h = seg_final.get(job.seg_id)
            if h is None or h not in rendered:
                continue
            r = rendered[h]
            segs.append(
                RenderedSegment(
                    seg_id=job.seg_id,
                    char_start=job.char_start,
                    char_end=job.char_end,
                    sample_rate=r.sample_rate,
                    n_samples=len(r.audio),
                    pause_after_ms=job.pause_after_ms,
                    engine=r.engine,
                    from_cache=r.from_cache,
                    synth_seconds=r.synth_seconds,
                    audio=r.audio,
                )
            )
        return segs

    def _assemble(self, ordered: list[RenderedSegment]) -> RenderResult:
        if not ordered:
            return RenderResult(np.zeros(0, np.float32), REFERENCE_SR, [])
        canonical = _canonical_rate(ordered)
        audio_segs = [assemble.ensure_rate(s.audio, s.sample_rate, canonical) for s in ordered]
        pauses = [s.pause_after_ms for s in ordered]
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
def _canonical_rate(segments: list[RenderedSegment]) -> int:
    """Pick the assembly rate: IndexTTS-2's native rate if present, else first."""
    for seg in segments:
        if seg.engine == "indextts2":
            return seg.sample_rate
    return segments[0].sample_rate if segments else REFERENCE_SR


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32).reshape(-1), int(sr)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)
