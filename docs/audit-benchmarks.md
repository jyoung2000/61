# Audit Benchmarks — how to measure the before/after numbers

The audit's Phase 1–5 changes were developed and unit-tested in an
environment without a GPU, FFmpeg, or the real-content fixtures, so the
headline numbers must be measured on the deployment box (the Unraid
host with the GTX 1650). This page lists each required number, the
exact command that produces it, and where to record it.

Record results in the table at the bottom and commit.

## 1. Perception decode time (Phase 5.1)

The seek-threshold fix (`REFRAMER_SEEK_GAP_FRAMES`, old `gap>5` vs new
`gap≤60 → grab()`) and the candidate piped-FFmpeg sampler are compared
by `scripts/bench_perception_decode.py`:

```bash
# inside the container (or any env with opencv + ffmpeg)
python scripts/bench_perception_decode.py /data/fixtures/30min_1080p.mp4
python scripts/bench_perception_decode.py /data/fixtures/30min_1080p.mp4 --hwaccel
```

Expected shape of the result: the `seek (old)` row decodes ~a full GOP
per sample on long-GOP H.264; `grab (new)` touches each frame once.
If the `ffmpeg pipe (+nvdec)` row wins consistently on your content,
open an issue to promote it to the default sampler.

## 2. End-to-end analysis time (Phases 2/3/5)

Run the same 30-minute 1080p fixture through a full analysis twice —
once on the commit before `Phase 2: reframing framed like a human
operator` and once on the current head — and compare the
`_stage_timer` lines in the job log:

```bash
grep "stage=" /data/logs/app.log | grep -E "vocal_separation|reframer_analysis|metadata"
```

Contributors to the delta: adaptive tiled detection (Phase 2.5 — the
2×2 tiled pass no longer runs on every frame), the decode fix (5.1),
and concurrent vocal separation (5.5, only on ≥6 GB GPUs).

Also grep the new per-stage VRAM ledger to confirm stage ordering:

```bash
grep "VRAM ledger" /data/logs/app.log
```

## 3. Export time — NVENC vs x264 (Phase 5.3)

Export the same clip twice from Settings → toggle
`GPU_ACCELERATION_ENABLED`, and compare with the quality/speed preset:

```bash
# NVENC (defaults: p5 + spatial/temporal AQ)
GPU_NVENC_PRESET=p5  # then export, note wall time from the job log
# x264 reference
GPU_ACCELERATION_ENABLED=false  # then export the same clip
```

For the quality check the audit asks for (NVENC vs
`libx264 -preset veryfast` at equal bitrate), reuse the parity harness's
SSIM tooling against the x264 render:

```bash
python scripts/parity_harness.py nvenc_export.mp4 x264_export.mp4 \
    --timestamps 1 5 10 20 --ssim-threshold 0.95
```

## 4. Subtitle timing error pre/post forced alignment (Phase 3.1)

`SUBTITLE_FORCED_ALIGN` logs its own measurement on every run:

```
Forced alignment (torchaudio/cpu): 42 segments / 512 words refined, mean shift 87.3ms
```

The `mean shift` value IS the pre/post timing delta — the average
distance between Whisper's word timestamps and the CTC-aligned ones.
Collect it over a few representative videos:

```bash
grep "Forced alignment" /data/logs/app.log
```

For an absolute (vs-human) number, spot-check ~20 cues against the
waveform in the NLE: cue-in should land within ~1 frame (33 ms) of
voice onset with alignment on, vs 50–200 ms drift with
`SUBTITLE_FORCED_ALIGN=false`.

## 5. Reframing stability metrics (Phase 2.7)

Every analysis now prints the measurable tuning targets in the
ReframeReport line:

```
ReframeReport | ... | jerk=0.031 hold_ratio=87.4% safe_area=96.2% cuts/min=3.20 | ...
```

Compare before/after tuning `REFRAMER_L1_DEADBAND_FRAC` /
`REFRAMER_SACCADE_CUT_FRAC`. Higher hold ratio + lower jerk with a
similar cuts/min is the goal; a cuts/min explosion means the saccade
threshold is too low for the content.

## Results

| Metric | Before | After | Fixture | Date |
|---|---|---|---|---|
| Perception decode (30 min 1080p) | | | | |
| End-to-end analysis (30 min 1080p) | | | | |
| Export (NVENC p5) vs x264 veryfast | | | | |
| NVENC vs x264 SSIM @ equal bitrate | | | | |
| Subtitle mean word shift (ms) | 50–200 (Whisper) | | | |
| Hold ratio / jerk / cuts/min | | | | |
