"""Acceptance tests for Task 2 — Diarization→transcript fusion."""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.speaker_fusion import assign_speakers_from_timeline


def _timeline(spans, resolution_ms=200):
    """Build a {time_ms: speaker} timeline from (start_s, end_s, speaker)."""
    tl = {}
    for start_s, end_s, spk in spans:
        t = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        while t < end_ms:
            tl[t] = spk
            t += resolution_ms
    return tl


def _seg(start, end, text, speaker="Speaker 1", words=None):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker, words=words)


def _words(spec):
    """spec: list of (start, end, word)."""
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


# ── Majority-overlap assignment ──────────────────────────────────────────

def test_majority_overlap_assignment():
    # SPEAKER_00 owns 0-10s, SPEAKER_01 owns 10-20s.
    tl = _timeline([(0, 10, "SPEAKER_00"), (10, 20, "SPEAKER_01")])
    segs = [
        _seg(0.0, 4.0, "first speaker talking"),
        _seg(12.0, 18.0, "second speaker now"),
    ]
    out = assign_speakers_from_timeline(segs, tl)
    assert out[0].speaker == "Speaker 1"   # SPEAKER_00 → first label
    assert out[1].speaker == "Speaker 2"   # SPEAKER_01 → second label


def test_low_coverage_inherits_previous():
    # Timeline only covers 0-5s (SPEAKER_00); second segment at 50s has no
    # coverage and must inherit the previous speaker rather than guess.
    tl = _timeline([(0, 5, "SPEAKER_00")])
    segs = [
        _seg(0.0, 4.0, "covered"),
        _seg(50.0, 54.0, "uncovered tail"),
    ]
    out = assign_speakers_from_timeline(segs, tl)
    assert out[0].speaker == "Speaker 1"
    assert out[1].speaker == "Speaker 1"   # inherited


# ── Mid-segment changeover split (word-level) ─────────────────────────────

def test_changeover_splits_segment_with_words():
    # SPEAKER_00 owns 0-3s, SPEAKER_01 owns 3-6s. A single segment spans the
    # changeover and carries word timestamps.
    tl = _timeline([(0, 3, "SPEAKER_00"), (3, 6, "SPEAKER_01")])
    words = _words([
        (0.2, 0.8, "hello"), (0.9, 1.5, "there"), (1.6, 2.4, "friend"),
        (3.2, 3.8, "yes"), (3.9, 4.6, "indeed"), (4.7, 5.4, "sir"),
    ])
    seg = _seg(0.0, 6.0, "hello there friend yes indeed sir", words=words)
    out = assign_speakers_from_timeline([seg], tl)
    assert len(out) == 2
    assert out[0].speaker == "Speaker 1"
    assert out[1].speaker == "Speaker 2"
    assert "hello" in out[0].text and "sir" in out[1].text
    # Monotonic, non-overlapping timing.
    assert out[0].end <= out[1].start
    assert out[0].start < out[0].end < out[1].end


def test_single_speaker_segment_not_split():
    tl = _timeline([(0, 10, "SPEAKER_00")])
    words = _words([(0.2, 0.8, "all"), (0.9, 1.5, "one"), (1.6, 2.4, "speaker")])
    seg = _seg(0.0, 3.0, "all one speaker", words=words)
    out = assign_speakers_from_timeline([seg], tl)
    assert len(out) == 1
    assert out[0].speaker == "Speaker 1"


# ── Coarse split without word timestamps ──────────────────────────────────

def test_coarse_split_without_words():
    tl = _timeline([(0, 6, "SPEAKER_00"), (6, 10, "SPEAKER_01")])
    seg = _seg(0.0, 10.0, "the first half here and then the second half here")
    out = assign_speakers_from_timeline([seg], tl)
    assert len(out) == 2
    assert out[0].speaker == "Speaker 1"
    assert out[1].speaker == "Speaker 2"
    assert out[0].end == out[1].start


# ── Empty timeline → unchanged ─────────────────────────────────────────────

def test_empty_timeline_returns_unchanged():
    segs = [_seg(0.0, 4.0, "anything", speaker="Speaker 3")]
    out = assign_speakers_from_timeline(segs, {})
    assert out is segs
    assert out[0].speaker == "Speaker 3"


def test_none_timeline_returns_unchanged():
    segs = [_seg(0.0, 4.0, "anything")]
    assert assign_speakers_from_timeline(segs, None) is segs
