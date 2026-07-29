"""Generate WebVTT subtitle files from transcript segments.

WebVTT is preferred by YouTube, Vimeo, and most web players. It supports
inline speaker voicing (``<v Speaker 1>...``) and optional position /
alignment cues for platform-specific safe zones.

Wired into the export endpoints alongside the SRT and ASS generators.
"""

from __future__ import annotations

from typing import Optional

from backend.config import settings
from backend.models import TranscriptSegment
from backend.services.subtitle_formatter import (
    enforce_readability, get_safe_zone_margins,
)


def _format_vtt_time(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format: HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _escape_text(text: str) -> str:
    """Escape WebVTT-sensitive characters in subtitle text."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _position_cue(
    platform: str,
    video_width: int,
    video_height: int,
) -> str:
    """Build a WebVTT cue settings string from a platform safe-zone profile.

    Empty string when ``platform`` is unknown so the resulting cue uses
    the player's default positioning.
    """
    if not platform:
        return ""
    margins = get_safe_zone_margins(platform, video_width, video_height)
    pos = margins["recommended_position"]
    if pos == "middle":
        # 50% line, centered.
        return "line:50% align:middle"
    if pos == "top":
        return "line:10% align:start"
    # Default to bottom with a safe inset derived from the bottom margin.
    if video_height > 0:
        bottom_pct = max(0, 100 - int((margins["bottom_px"] / video_height) * 100))
        return f"line:{max(60, min(95, bottom_pct))}% align:middle"
    return ""


def generate_vtt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
    include_position: bool = False,
    platform: str = "horizontal",
    video_width: int = 1920,
    video_height: int = 1080,
    enforce_readability_rules: Optional[bool] = None,
    include_timestamps_in_text: bool = False,
    fps: Optional[float] = None,
) -> str:
    """Convert transcript segments to WebVTT format.

    Example output:
        WEBVTT

        1
        00:00:01.200 --> 00:00:04.800 line:90% align:middle
        <v Speaker 1>Hello everyone, welcome to the show.

        2
        00:00:05.000 --> 00:00:08.300
        <v Speaker 2>Thanks for having me!
    """
    if enforce_readability_rules is None:
        enforce_readability_rules = bool(
            getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True)
        )
    # Strip labels baked INTO cue text unconditionally (parity with
    # generate_srt): attribution lives in the ``speaker`` field and the voice
    # tag below is the only label renderer for VTT.
    try:
        from backend.services.subtitle_formatter import strip_baked_speaker_label
        for seg in (segments or []):
            _t = seg.text or ""
            _s = strip_baked_speaker_label(_t, getattr(seg, "speaker", None))
            if _s != _t:
                seg.text = _s
    except Exception:
        pass
    if enforce_readability_rules and segments:
        segments = enforce_readability(
            segments,
            max_cps=float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
            max_chars_per_line=int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
            min_duration_ms=int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
            max_duration_ms=int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 7000)),
            smart_line_breaks=bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
            word_timed_split_only=bool(getattr(
                settings, "SUBTITLE_EXPORT_WORD_TIMED_SPLIT_ONLY", True)),
        )

    cue_settings = (
        _position_cue(platform, video_width, video_height)
        if include_position else ""
    )

    # WebVTT cues MUST be chronological; sort defensively.
    segments = sorted(
        (s for s in segments if (s.text or "").strip()),
        key=lambda s: (s.start, s.end),
    )
    # Final invariant: frame-align cues + guarantee a one-frame gap so consecutive
    # cues never ship touching (exact parity with generate_srt). Fail-soft.
    from backend.services.srt_generator import _apply_min_gap
    segments = _apply_min_gap(segments, fps=fps)
    from backend.services.srt_generator import effective_include_speakers
    include_speakers = effective_include_speakers(segments, include_speakers)

    lines: list[str] = ["WEBVTT", ""]
    counter = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        if seg.end < seg.start:
            continue
        counter += 1
        start = _format_vtt_time(seg.start)
        end = _format_vtt_time(seg.end)
        # Render multi-line text with literal newlines — WebVTT supports
        # arbitrary line breaks inside a cue.
        body = _escape_text(text)
        if include_timestamps_in_text:
            from backend.services.srt_generator import inline_timestamp
            body = f"{inline_timestamp(seg.start)} {body}"
        if include_speakers and seg.speaker:
            body = f"<v {seg.speaker}>{body}"
        timing = f"{start} --> {end}"
        if cue_settings:
            timing = f"{timing} {cue_settings}"
        lines.append(str(counter))
        lines.append(timing)
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
