"""Split-CPS tolerance — keep a cue whole rather than shatter it on reading speed.

The reading-speed cap alone re-split every merged sentence straight back into
2-3 word flashes (a logged 193→174 merge re-exploded to 500+ cues). With
``SUBTITLE_SPLIT_CPS_TOLERANCE`` a cue is kept intact up to ``max_cps × tol`` and
only split above it; extend-into-idle-time still targets the strict cap where
there's room.
"""

from backend.config import settings
from backend.models import TranscriptSegment as TS
from backend.services.subtitle_formatter import enforce_readability


# ~52 non-space chars in 2.0s ≈ 26 cps — over the 20 cap but under 20×1.5=30.
# The next cue is a DIFFERENT speaker starting right after, so this line can't
# merge into it and has no idle time to extend into: the only way to satisfy a
# strict cap would be to split. Tolerance should keep it whole instead.
_LINE = "and it is your boobs over here and they really do feel so soft now"


def _run(tol):
    old = getattr(settings, "SUBTITLE_SPLIT_CPS_TOLERANCE", 1.5)
    settings.SUBTITLE_SPLIT_CPS_TOLERANCE = tol
    try:
        segs = [
            TS(start=10.0, end=12.0, text=_LINE, speaker="Speaker 1"),
            TS(start=12.1, end=14.0, text="Next utterance here.", speaker="Speaker 2"),
        ]
        return enforce_readability(
            segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
            min_duration_ms=833, max_duration_ms=9000, auto_cjk=False)
    finally:
        settings.SUBTITLE_SPLIT_CPS_TOLERANCE = old


def test_tolerance_keeps_overcap_line_whole():
    out = _run(tol=1.5)
    whole = [s for s in out if s.text.replace("\n", " ").strip() == _LINE]
    assert len(whole) == 1, [s.text for s in out]


def test_strict_tolerance_still_splits():
    # tol=1.0 → strict 20 cap → the 25-cps line must be split (proves the knob
    # actually gates the splitter, not something else keeping it whole).
    out = _run(tol=1.0)
    whole = [s for s in out if s.text.replace("\n", " ").strip() == _LINE]
    assert len(whole) == 0
    assert len(out) > 2
