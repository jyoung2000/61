"""Generate SRT / WebVTT / bilingual subtitle files from transcript segments.

Otter.ai exports captions as TXT / DOCX / PDF / SRT with toggles for speaker
names, timestamps, and automatic line breaks. This module is ClipAI's
caption-export surface and adds, beyond the single SRT flavour:

  * ``generate_vtt``           — WebVTT for web players (delegates to
                                 ``vtt_generator`` so SRT/VTT don't fork).
  * ``generate_bilingual_srt`` — translated + original stacked per cue
                                 (a repurposing edge over Otter).
  * Otter-style export toggles — ``include_speakers`` and
    ``include_timestamps_in_text`` on every generator.
"""

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


def inline_timestamp(seconds: float) -> str:
    """Inline ``[mm:ss]`` marker for the ``include_timestamps_in_text`` toggle.

    Minutes are not wrapped at 60 so a 75-minute mark reads ``[75:03]``.
    """
    if seconds < 0:
        seconds = 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def _decorate_text(
    text: str,
    seg: TranscriptSegment,
    include_speakers: bool,
    include_timestamps_in_text: bool,
) -> str:
    """Apply the speaker-label + inline-timestamp toggles to a line of text."""
    if include_timestamps_in_text:
        text = f"{inline_timestamp(seg.start)} {text}"
    if include_speakers and seg.speaker:
        text = f"[{seg.speaker}] {text}"
    return text


def generate_srt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
    enforce_readability_rules: Optional[bool] = None,
    *,
    include_timestamps_in_text: bool = False,
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
        text = _decorate_text(
            text, seg, include_speakers, include_timestamps_in_text)
        lines.append(str(counter))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line separator
    return "\n".join(lines)


def generate_vtt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
    enforce_readability_rules: Optional[bool] = None,
    *,
    include_timestamps_in_text: bool = False,
    **kwargs,
) -> str:
    """WebVTT export. Delegates to ``vtt_generator.generate_vtt`` (so the
    SRT and VTT paths don't fork) and layers on the Otter export toggles.

    Extra keyword args (``include_position``, ``platform``, ...) pass
    straight through to the underlying generator.
    """
    from backend.services.vtt_generator import generate_vtt as _vtt
    return _vtt(
        segments,
        include_speakers=include_speakers,
        enforce_readability_rules=enforce_readability_rules,
        include_timestamps_in_text=include_timestamps_in_text,
        **kwargs,
    )


def _translated_text_for(translated_segments, idx: int) -> str:
    """Pull the translated text for source index ``idx`` from a parallel
    list of TranscriptSegments / dicts / strings."""
    if not translated_segments or idx >= len(translated_segments):
        return ""
    item = translated_segments[idx]
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("text", "") or "").strip()
    return str(getattr(item, "text", "") or "").strip()


def generate_bilingual_srt(
    segments: list[TranscriptSegment],
    translated_segments: list,
    order: str = "translation_top",
    *,
    include_speakers: bool = True,
    include_timestamps_in_text: bool = False,
    enforce_readability_rules: Optional[bool] = None,
) -> str:
    """Two-line bilingual SRT: translated text stacked with the original.

    Timings come from the source ``segments`` (the translation inherits
    them), keeping a strict 1:1 cue alignment. ``order`` controls the
    stacking:

      * ``"translation_top"`` (default) — translated line first, original below.
      * ``"original_top"``              — original line first, translation below.

    Readability enforcement is reused on the *translated* line (CJK→EN
    length changes), but only when it preserves the 1:1 cue count — a
    bilingual cue must stay paired with its source.
    """
    if enforce_readability_rules is None:
        enforce_readability_rules = bool(
            getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True))

    # Optionally normalise the translated line via the readability pass,
    # but only adopt the result if it keeps the cue count (1:1 alignment).
    translated_texts = [
        _translated_text_for(translated_segments, i) for i in range(len(segments))
    ]
    if enforce_readability_rules and translated_segments:
        try:
            from backend.services.subtitle_formatter import enforce_readability
            tmp = [
                TranscriptSegment(
                    start=s.start, end=s.end,
                    text=translated_texts[i] or s.text,
                    speaker=s.speaker,
                )
                for i, s in enumerate(segments)
            ]
            wrapped = enforce_readability(tmp)
            if len(wrapped) == len(segments):
                translated_texts = [(w.text or "").strip() for w in wrapped]
        except Exception:
            pass

    lines: list[str] = []
    counter = 0
    for i, seg in enumerate(segments):
        original = (seg.text or "").strip()
        translated = (translated_texts[i] or "").strip()
        if not original and not translated:
            continue
        # Collapse any internal newlines so each side is exactly one line.
        original = " ".join(original.split())
        translated = " ".join(translated.split())
        original = _decorate_text(
            original, seg, include_speakers, include_timestamps_in_text)
        # The speaker/timestamp prefix belongs on the leading line only.
        if order == "original_top":
            body = "\n".join([original, translated]) if translated else original
        else:
            top = _decorate_text(
                translated, seg, include_speakers, include_timestamps_in_text
            ) if translated else ""
            # When translation leads, strip the duplicate prefix from original.
            plain_original = (seg.text or "").strip()
            plain_original = " ".join(plain_original.split())
            body = "\n".join([top, plain_original]) if top else plain_original
        counter += 1
        lines.append(str(counter))
        lines.append(f"{_format_srt_time(seg.start)} --> {_format_srt_time(seg.end)}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)
