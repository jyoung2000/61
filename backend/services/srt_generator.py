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

import re
from typing import Optional

from backend.config import settings
from backend.models import TranscriptSegment


def _apply_min_gap(segments: list, fps: Optional[float] = None) -> list:
    """Apply the final cue-timing invariants before serialization.

    With a known video ``fps`` and ``SUBTITLE_FRAME_QUANTIZE`` on, every cue in/out
    is snapped to the video's frame grid with a one-frame gap — what a
    hand-authored track looks like (a reference YouTube track was 100 %
    frame-aligned with one-frame gaps). Without fps, degrade to the
    millisecond-based gap pass so cues still never ship touching. Fail-soft:
    returns the input unchanged on any error."""
    try:
        from backend.services.subtitle_formatter import (
            enforce_min_gap, quantize_to_frames,
        )
        min_gap_s = float(getattr(settings, "SUBTITLE_MIN_GAP_MS", 42)) / 1000.0
        if fps and float(fps) > 0 and bool(
                getattr(settings, "SUBTITLE_FRAME_QUANTIZE", True)):
            # Gap expressed in whole FRAMES so quantizing can't re-touch two cues.
            # Rounded (not ceiled): the default 42 ms is one frame at 24 fps, and
            # 0.042 × 23.976 = 1.007 must read as 1 frame, not 2.
            gap_frames = max(1, int(round(min_gap_s * float(fps))))
            return quantize_to_frames(segments, float(fps),
                                      min_gap_frames=gap_frames)
        return enforce_min_gap(segments, min_gap_s=min_gap_s)
    except Exception:
        return segments


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm

    Rounds to the nearest millisecond via a single integer conversion. The old
    per-field arithmetic TRUNCATED the fraction (``int((seconds % 1) * 1000)``),
    and because binary floats can't hold most decimals exactly — ``29.988 % 1``
    is 0.98799999… — that silently emitted every affected cue up to 1 ms EARLY,
    which was enough to knock frame-quantized times back off the frame grid."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
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


_PLACEHOLDER_SPEAKER = re.compile(r"^\s*speaker[\s_-]*\d+\s*$", re.IGNORECASE)


def effective_include_speakers(segments, include_speakers: bool) -> bool:
    """Speaker labels add no value (and clutter the file) when the whole
    transcript is a single speaker — the common case for narration / a solo
    talking-head. In that case suppress them even if ``include_speakers`` is
    True, matching how hand-authored SRTs omit "[Speaker 1]" everywhere.

    They also add no value when every label is still the diarizer's own
    placeholder. "[Speaker 2]" names nobody: it tells a viewer only that the
    voice changed, which the dialogue already tells them, and it costs 12
    characters off a 34-character line on every single cue. A reference
    broadcast track carries either a real name or nothing. So labels ship once
    at least one speaker has been given a name — renaming one speaker turns
    them on for the whole file. Set
    ``SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES=False`` to emit the placeholders."""
    if not include_speakers:
        return False
    distinct = {
        (s.get("speaker") if isinstance(s, dict) else getattr(s, "speaker", "")) or ""
        for s in (segments or [])
    }
    distinct = {d for d in distinct if d.strip()}
    if len(distinct) <= 1:
        return False
    if bool(getattr(settings, "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES", True)):
        if all(_PLACEHOLDER_SPEAKER.match(d) for d in distinct):
            return False
    return True


def _decorate_text(
    text: str,
    seg: TranscriptSegment,
    include_speakers: bool,
    include_timestamps_in_text: bool,
) -> str:
    """Apply the speaker-label + inline-timestamp toggles to a line of text.

    Both prefixes are added AFTER the readability pass has wrapped the cue to
    the per-line budget, so they push line 1 past it — "[Speaker 1] " alone
    costs 12 characters, and a 34-char line became 46 on screen. Re-wrap the
    decorated text so the budget survives decoration."""
    decorated = text
    if include_timestamps_in_text:
        decorated = f"{inline_timestamp(seg.start)} {decorated}"
    if include_speakers and seg.speaker:
        decorated = f"[{seg.speaker}] {decorated}"
    if decorated is text:
        return text
    return _rewrap(decorated)


def _rewrap(text: str) -> str:
    """Re-flow ``text`` into the configured line box, best-effort."""
    budget = int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42) or 0)
    if budget <= 0 or all(len(ln) <= budget for ln in text.splitlines()):
        return text
    try:
        from backend.services.subtitle_formatter import (
            _balanced_two_line, _hard_wrap_lines, _smart_split)
    except Exception:
        return text
    flat = " ".join(text.split())
    max_lines = max(1, int(getattr(settings, "SUBTITLE_MAX_LINES", 2) or 2))
    out = _smart_split(flat, budget, max_lines)
    if any(len(ln) > budget for ln in out.splitlines()):
        out = (_balanced_two_line(flat, budget) if max_lines == 2 else None) \
            or _hard_wrap_lines(out, budget, max_lines)
    return out


def generate_srt(
    segments: list[TranscriptSegment],
    include_speakers: bool = True,
    enforce_readability_rules: Optional[bool] = None,
    *,
    include_timestamps_in_text: bool = False,
    fps: Optional[float] = None,
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
                word_timed_split_only=bool(getattr(
                    settings, "SUBTITLE_EXPORT_WORD_TIMED_SPLIT_ONLY", True)),
            )
        except Exception:
            # Never let readability formatting break SRT generation — but say so.
            # Swallowing this silently meant one bad cue could ship the WHOLE file
            # unwrapped and un-capped with nothing to show it had happened.
            logger.warning(
                "SRT readability formatting failed — emitting unformatted cues "
                "(lines may exceed the character budget)", exc_info=True)

    # SRT cues MUST be chronological; sort defensively so corrupt upstream
    # ordering can't emit out-of-order / backwards cues.
    segments = sorted(
        (s for s in segments if (s.text or "").strip()),
        key=lambda s: (s.start, s.end),
    )
    # Final invariant: frame-align every cue and guarantee a one-frame gap (or,
    # with no fps, a millisecond gap) so consecutive cues never ship TOUCHING —
    # the post-readability sentence splitter emits contiguous pieces. Runs
    # UNCONDITIONALLY, even when readability enforcement is off, since it only
    # nudges endpoints (never merges/splits/reorders). Last mutation before write.
    segments = _apply_min_gap(segments, fps=fps)
    include_speakers = effective_include_speakers(segments, include_speakers)

    lines: list[str] = []
    counter = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        # Skip only backwards cues (corrupt timing). Zero-duration cues are
        # left as-is — the reference SRT format permits them.
        if seg.end < seg.start:
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
    fps: Optional[float] = None,
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
        fps=fps,
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

    # Guarantee a min inter-cue gap on the source timings that drive the cues
    # (same invariant as generate_srt), then pair each source segment with its
    # translated line and sort the pairs chronologically so cues emit in order.
    segments = _apply_min_gap(list(segments))
    paired = sorted(
        ((seg, translated_texts[i]) for i, seg in enumerate(segments)),
        key=lambda p: (p[0].start, p[0].end),
    )
    include_speakers = effective_include_speakers(segments, include_speakers)

    lines: list[str] = []
    counter = 0
    for seg, _translated_line in paired:
        original = (seg.text or "").strip()
        translated = (_translated_line or "").strip()
        if not original and not translated:
            continue
        if seg.end < seg.start:
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
