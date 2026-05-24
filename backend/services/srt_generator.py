"""Generate SRT subtitle files from transcript segments with speaker labels."""

from typing import Optional

from backend.config import settings
from backend.models import TranscriptSegment


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
    enforce_readability_rules: Optional[bool] = None,
) -> str:
    """Convert transcript segments to SRT format with optional speaker labels.

    When ``SUBTITLE_CPS_ENFORCEMENT`` is enabled (default), segments are
    first run through ``subtitle_formatter.enforce_readability`` so the
    output honors Netflix-style CPS, line-length, and duration limits.
    Pass ``enforce_readability_rules=False`` to bypass the preprocessor.

    Example output:
        1
        00:00:01,200 --> 00:00:04,800
        [Speaker 1] Hello everyone, welcome to the show.

        2
        00:00:05,000 --> 00:00:08,300
        [Speaker 2] Thanks for having me!
    """
    if enforce_readability_rules is None:
        enforce_readability_rules = bool(
            getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True)
        )
    if enforce_readability_rules and segments:
        # Lazy import — keeps SRT generation working when the formatter
        # module fails to import for any reason.
        try:
            from backend.services.subtitle_formatter import enforce_readability
            segments = enforce_readability(
                segments,
                max_cps=float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
                max_chars_per_line=int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
                min_duration_ms=int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
                max_duration_ms=int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 7000)),
                smart_line_breaks=bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
            )
        except Exception:
            # Never let readability formatting break SRT generation.
            pass

    lines: list[str] = []
    counter = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        counter += 1
        start = _format_srt_time(seg.start)
        end = _format_srt_time(seg.end)
        if include_speakers and seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(str(counter))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line separator
    return "\n".join(lines)
