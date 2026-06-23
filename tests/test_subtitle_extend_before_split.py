"""Pass 0.7 — borrow idle time to satisfy CPS before splitting.

Translated CJK→EN cues are often longer than the short source window that
timed them, so the CPS splitter used to shatter them into 2-3 word flashes
even when there was silence after the line. ``SUBTITLE_EXTEND_BEFORE_SPLIT``
stretches such a cue into the following idle gap first (bounded by the next
cue and the max display duration), keeping it a single readable phrase.
"""

from backend.config import settings
from backend.models import TranscriptSegment
from backend.services.subtitle_formatter import enforce_readability, _cps


# A ~55-char English line timed to a 2.0 s source window → ~23 CPS (over the
# 20 cap), followed by ~15 s of silence (this clip was only ~24% speech).
_LINE = "I'm doing the pre-cumming by pressing down on them now."


def _mk():
    return [
        TranscriptSegment(start=20.0, end=22.0, text=_LINE, speaker="Speaker 1"),
        TranscriptSegment(start=37.0, end=40.0, text="Oh my god!", speaker="Speaker 1"),
    ]


def _run(extend):
    old = getattr(settings, "SUBTITLE_EXTEND_BEFORE_SPLIT", True)
    settings.SUBTITLE_EXTEND_BEFORE_SPLIT = extend
    try:
        return enforce_readability(
            _mk(), max_cps=20.0, max_chars_per_line=42, max_lines=2,
            min_duration_ms=833, max_duration_ms=9000, auto_cjk=False,
        )
    finally:
        settings.SUBTITLE_EXTEND_BEFORE_SPLIT = old


def test_extend_keeps_line_whole_instead_of_splitting():
    out = _run(extend=True)
    # The long line should survive as ONE cue (its full text intact), not be
    # cut into fragments.
    matches = [s for s in out if s.text.replace("\n", " ").strip() == _LINE]
    assert len(matches) == 1, [s.text for s in out]
    seg = matches[0]
    assert _cps(seg.text.replace("\n", " "), seg.end - seg.start) <= 20.0 + 0.5


def test_extend_never_overruns_the_next_cue():
    out = _run(extend=True)
    starts = sorted(s.start for s in out)
    for s in out:
        # No cue may extend past the next cue's start.
        later = [x for x in starts if x > s.start + 1e-6]
        if later:
            assert s.end <= later[0] + 1e-6, f"{s.text!r} overruns {later[0]}"


def test_extend_reduces_fragment_count_vs_splitting():
    with_ext = _run(extend=True)
    without_ext = _run(extend=False)
    # Borrowing idle time yields FEWER, fuller cues than shattering on CPS.
    assert len(with_ext) <= len(without_ext)
