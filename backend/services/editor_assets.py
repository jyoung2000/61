"""Editor scrub-asset preparation, scheduled to stay OFF the analysis pipeline's
critical path.

The pipeline used to fire faststart + the fine sprite + peaks the moment a job
started — precisely when ffmpeg frame extraction, audio extraction, and the
source-hash read are all streaming the same multi-GB file. On spinning disks
(Unraid!) the faststart remux alone re-reads AND re-writes the entire source
while extraction is trying to read it, and the fine sprite adds a whole-file
keyframe scan on top. None of those assets are needed that early: the COARSE
sprite (a few dozen input-seeks — light, fast) covers the editor immediately.

New order:

  1. NOW:   coarse sprite (seconds, near-zero I/O footprint).
  2. WAIT:  until the pipeline's ``audio.wav`` lands (its atomic rename is the
            end-of-extraction signal), capped so a stall can't wedge us.
  3. THEN:  faststart remux → fine sprite → peaks. Peaks read the freshly
            extracted 16 kHz mono ``audio.wav`` via the zero-copy memmap path,
            so they're essentially free — the old early schedule decoded the
            VIDEO's audio track a second time because ``audio.wav`` didn't
            exist yet.

Everything is best-effort: any failure logs and moves on, and the lazy
``/api/jobs/{id}/filmstrip.*`` endpoints regenerate whatever is missing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

# End-of-extraction wait: poll cheaply, give up after a generous ceiling so a
# crashed extraction never wedges asset prep (the assets then build against
# the source video exactly as before).
_AUDIO_POLL_S = 5.0
_AUDIO_MAX_WAIT_S = 1800.0


async def prepare_editor_assets(
    video_path: str,
    job_dir: str,
    audio_path: str,
    poll_s: float = _AUDIO_POLL_S,
    max_wait_s: float = _AUDIO_MAX_WAIT_S,
    sleep=None,
) -> None:
    """Coarse sprite now; faststart + fine sprite + peaks after extraction.

    ``sleep`` is injectable for tests (defaults to ``asyncio.sleep``).
    Never raises.
    """
    _sleep = sleep or asyncio.sleep

    # 1) Coarse sprite immediately — the editor's filmstrip within seconds of
    #    import. Skips itself for short sources (fine pass is fast there) and
    #    when a fine sprite already exists (re-analysis).
    try:
        from backend.services.filmstrip_generator import generate_sprite_coarse
        await asyncio.to_thread(generate_sprite_coarse, video_path, job_dir)
    except Exception as e:  # noqa: BLE001
        logger.warning("bg coarse sprite failed: %s", e)

    # 2) Wait for the extraction's audio.wav (atomic rename = complete). A
    #    cached-extraction run may already have it; a failed run never will —
    #    the cap keeps us moving either way.
    try:
        deadline = time.monotonic() + max_wait_s
        while not os.path.isfile(audio_path) and time.monotonic() < deadline:
            await _sleep(poll_s)
    except Exception:
        pass

    # 3) Heavy passes, now off the extraction window.
    try:
        from backend.config import settings
        if getattr(settings, "FFMPEG_FASTSTART", True):
            from backend.services.faststart import ensure_faststart
            await asyncio.to_thread(ensure_faststart, video_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("bg faststart failed: %s", e)
    try:
        from backend.services.filmstrip_generator import generate_sprite, generate_peaks
        await asyncio.to_thread(generate_sprite, video_path, job_dir)
        _pk_src = audio_path if os.path.isfile(audio_path) else video_path
        await asyncio.to_thread(generate_peaks, _pk_src, job_dir)
        logger.info("editor scrub assets ready (background, post-extraction)")
    except Exception as e:  # noqa: BLE001
        logger.warning("bg sprite/peaks failed: %s", e)
