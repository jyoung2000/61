"""P0 acceptance — enforce_readability honors config for max display duration.

Two regressions are covered:

  1. The function-signature default for ``max_duration_ms`` used to be a stale
     hard-coded 4500 ms that disagreed with ``config.SUBTITLE_MAX_DURATION_MS``
     (9000 ms). A no-kwargs call must now resolve the cap from config, so a slow
     cue between 4.5 s and 9 s survives un-split.
  2. The cap is honored when passed explicitly: a low cap splits a long cue, a
     high cap keeps it whole.
"""

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services.subtitle_formatter import enforce_readability


def _words(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def _slow_cue():
    """A single readable English sentence spanning ~7 s — under the 9 s
    configured cap but over the old 4.5 s signature default. Word-timed so
    the duration splitter has real pause structure to (not) use."""
    spec = [
        (0.0, 0.6, "We"), (1.0, 1.6, "walked"), (2.0, 2.6, "slowly"),
        (3.0, 3.6, "along"), (4.0, 4.6, "the"), (5.0, 5.6, "quiet"),
        (6.0, 7.0, "shore."),
    ]
    text = " ".join(w for _, _, w in spec)
    return TranscriptSegment(start=0.0, end=7.0, text=text,
                             speaker="Speaker 1", words=_words(spec))


def test_no_kwargs_resolves_max_duration_from_config(monkeypatch):
    # With the configured 9 s cap, a ~7 s cue stays whole on a no-kwargs call.
    monkeypatch.setattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)
    out = enforce_readability([_slow_cue()])
    assert len(out) == 1, f"7s cue split despite 9s config cap (got {len(out)})"
    assert out[0].text.replace("\n", " ") == "We walked slowly along the quiet shore."


def test_no_kwargs_low_config_cap_splits(monkeypatch):
    # Drop the configured cap below the cue duration → it must split.
    monkeypatch.setattr(settings, "SUBTITLE_MAX_DURATION_MS", 3000)
    out_low = enforce_readability([_slow_cue()])
    assert len(out_low) > 1, "low config cap did not trigger a duration split"
    # A low cap must produce strictly more cues than the high (9s) cap, which
    # keeps the same cue whole — proving the config value drives the split.
    monkeypatch.setattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)
    out_high = enforce_readability([_slow_cue()])
    assert len(out_low) > len(out_high)


def test_explicit_high_cap_keeps_cue_whole():
    out = enforce_readability([_slow_cue()], max_duration_ms=9000)
    assert len(out) == 1


def test_explicit_low_cap_splits_cue():
    out = enforce_readability([_slow_cue()], max_duration_ms=3000)
    assert len(out) > 1


def test_timing_and_speaker_preserved_on_split():
    out = enforce_readability([_slow_cue()], max_duration_ms=3000)
    # Speaker preserved, timing monotonic and within the original span.
    assert all(s.speaker == "Speaker 1" for s in out)
    assert out[0].start >= 0.0
    assert out[-1].end <= 7.0 + 1e-6
    for i in range(len(out) - 1):
        assert out[i].start <= out[i].end <= out[i + 1].start
