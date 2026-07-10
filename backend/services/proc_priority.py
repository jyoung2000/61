"""Run background media transcodes at low CPU priority.

The browser preview (a full re-encode on CPU-only boxes), the faststart remux
and the sprite scans all run WHILE the analysis pipeline is saturating the
machine with frame extraction, scene detection, and (locally) Whisper/YOLO.
They are wall-clock-insensitive — nobody is waiting on a background transcode
— so they should yield the CPU whenever the pipeline wants it.

``low_priority_popen_kwargs()`` returns kwargs for ``subprocess.run/Popen``
that lower the child's scheduling priority (``nice +10`` on POSIX,
BELOW_NORMAL on Windows). The OS scheduler then gives those processes the
idle cores only — faster analysis with zero effect on the transcode's output
bytes (priority changes WHEN it runs, never WHAT it produces).
"""

from __future__ import annotations

import os

_NICE_LEVEL = 10
# Windows: BELOW_NORMAL_PRIORITY_CLASS (subprocess exposes it on win32 only).
_BELOW_NORMAL = 0x00004000


def _renice_self() -> None:  # pragma: no cover — runs in the forked child
    try:
        os.nice(_NICE_LEVEL)
    except OSError:
        pass


def low_priority_popen_kwargs() -> dict:
    """kwargs to merge into a ``subprocess.run``/``Popen`` call so the child
    runs below normal priority. Empty dict when the platform offers nothing —
    callers can always ``**`` the result safely."""
    if os.name == "posix":
        return {"preexec_fn": _renice_self}
    if os.name == "nt":  # pragma: no cover — container is POSIX
        return {"creationflags": _BELOW_NORMAL}
    return {}
