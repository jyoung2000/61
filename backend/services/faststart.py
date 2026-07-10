"""Make uploaded MP4/MOV sources stream progressively (faststart).

An MP4 whose ``moov`` atom sits *after* ``mdat`` cannot begin playback or
seek until the browser has downloaded the whole file — the metadata index
is at the very end. Phones and long videos over a tunnel feel broken:
seconds of spinner before the first frame, and every scrub re-stalls.

Muxing with ``-movflags +faststart`` relocates ``moov`` to the front so the
same file plays and seeks the moment the first bytes arrive. We do this once
at ingest, as a lossless stream copy (no re-encode), so it costs a single
disk rewrite and pays off on every subsequent open / seek in the editor.

Best-effort and non-fatal: on any probe/mux failure the original file is
left exactly as-is (there is a sibling ``moov`` scanner in
:mod:`backend.services.browser_preview` used by the preview-proxy path;
this module is the ingest-side remux and intentionally self-contained so
importing it never drags in the preview pipeline).
"""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

_REMUXABLE = (".mp4", ".mov", ".m4v")


def is_faststart(path: str) -> bool:
    """True when ``moov`` precedes ``mdat`` (or the container isn't MP4).

    Scans top-level atoms without decoding. Unknown/short reads return True
    so we never remux off a failed sniff.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _REMUXABLE:
        return True
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            pos = 0
            hops = 0
            while pos < size and hops < 32:
                fh.seek(pos)
                header = fh.read(8)
                if len(header) < 8:
                    return True
                atom_size = int.from_bytes(header[:4], "big")
                atom_type = header[4:8]
                if atom_type == b"moov":
                    return True
                if atom_type == b"mdat":
                    return False
                if atom_size == 1:  # 64-bit extended size
                    ext_size = fh.read(8)
                    if len(ext_size) < 8:
                        return True
                    atom_size = int.from_bytes(ext_size, "big")
                if atom_size < 8:
                    return True
                pos += atom_size
                hops += 1
    except OSError:
        return True
    return True


def ensure_faststart(path: str, timeout: int = 900) -> bool:
    """Relocate ``moov`` to the front in place if needed. Returns True if the
    file is faststart afterwards (already was, or was successfully remuxed)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _REMUXABLE:
        return True
    if is_faststart(path):
        return True

    tmp = path + ".faststart.tmp" + ext
    try:
        from backend.services.proc_priority import low_priority_popen_kwargs
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", path,
                "-c", "copy", "-map", "0",
                "-movflags", "+faststart",
                tmp,
            ],
            capture_output=True, timeout=timeout,
            # Background remux (full-file read+write) — must yield to a live
            # analysis pipeline streaming the same source file.
            **low_priority_popen_kwargs(),
        )
        if proc.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            logger.warning(
                "faststart remux failed (rc=%s): %s",
                proc.returncode, proc.stderr[:300].decode(errors="replace"),
            )
            _rm(tmp)
            return False
        os.replace(tmp, path)
        logger.info("faststart remux applied: %s", path)
        return True
    except Exception as e:  # noqa: BLE001 — best effort, keep original on failure
        logger.warning("faststart remux crashed: %s", e)
        _rm(tmp)
        return False


def _rm(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
