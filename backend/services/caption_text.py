"""Shared helpers for cleaning clip caption / hook text.

The per-clip transcript slice is formatted with inline ``[m:ss]`` cue
timestamps (see ``pipeline._slice_transcript_for_clip``). Those markers are
fine when the slice is shown as a transcript, but they must not leak into the
social-facing ``hook_text`` / ``suggested_caption`` fields — those are shown on
the clip cards and used verbatim as on-screen text overlays, where "[0:00]"
is just noise.
"""

import re

# Matches "[m:ss]", "[mm:ss]" and "[h:mm:ss]" cue markers.
_CUE_TIMESTAMP = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def strip_cue_timestamps(text: str) -> str:
    """Remove inline ``[m:ss]`` cue markers, returning clean prose."""
    if not text or not isinstance(text, str):
        return text or ""
    cleaned = _CUE_TIMESTAMP.sub(" ", text)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)  # tidy space before punct
    cleaned = re.sub(r"\s{2,}", " ", cleaned)            # collapse gaps
    return cleaned.strip()
