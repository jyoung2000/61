"""Preview <-> server-export parity for active-word (karaoke) highlighting.

The burned-in ASS export (ass_generator.generate_ass) must highlight the SAME
word at a given play time as the DOM/canvas preview
(frontend/src/utils/activeWordTiming.js getCurrentWordIndex). Two regressions
this guards:

  * FALLBACK (word-less cue) double-applied the anticipation lead, so every
    transition landed ~0.14s early vs the preview.
  * REAL-PER-WORD branch started each word at its own audio start and let the
    gap-fill hold the PREVIOUS word across an inter-word silence, while the
    preview advances to the UPCOMING word at the prior word's end.

We render the canonical styled ASS, parse the per-word colour events back into
(start, end, active_word), and assert the active word at sampled play times
matches the frontend model reimplemented here.
"""

import re

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.ass_generator import generate_ass, _hex_to_ass_color

_AW_HEX = "#FFD700"
_AW_COLOR = _hex_to_ass_color(_AW_HEX)          # &H0000D7FF&
_SPK_COLOR = "#00D9FF"                            # distinct from the AW colour
_ANTICIPATION_S = 0.10
_AUDIO_BUFFER_S = 0.12


def _gen(segments, start, end):
    return generate_ass(
        segments=segments, start_time=start, end_time=end,
        font="DM Sans", font_size=30, font_weight=700, font_color="#FFFFFF",
        position="bottom", speaker_colors={"Speaker 1": _SPK_COLOR},
        use_speaker_colors=True, video_width=1080, video_height=1920,
        background_enabled=False, background_color="#000000",
        background_opacity=0, background_radius=0,
        outline_color="#000000", outline_opacity=100, outline_width=2,
        max_width_pct=100, offset_v_pct=4,
        active_word_enabled=True, active_word_color=_AW_HEX,
        active_word_outline_color="#000000", active_word_bg_color="#000000",
        active_word_bg_opacity=0, active_word_bg_radius=4,
        enforce_readability_rules=False,
    )


def _parse_ts(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_ACTIVE_RE = re.compile(re.escape(_AW_COLOR) + r"[^}]*\}([^{]+)")


def _word_events(ass):
    """Extract the per-word colour events as (start, end, active_word).

    The active word is the token wrapped in the active-word colour tag; the
    non-active words carry the speaker colour, so the AW colour appears exactly
    once per colour event and marks the highlighted word."""
    events = []
    for line in ass.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        # "Dialogue: <layer>",Start,End,Style,Name,ML,MR,MV,Effect,Text...
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        start, end, text = parts[1], parts[2], parts[9]
        m = _ACTIVE_RE.search(text)
        if not m:
            continue  # Layer-0 border event (no AW colour) — skip
        events.append((_parse_ts(start), _parse_ts(end), m.group(1).strip()))
    events.sort(key=lambda e: e[0])
    return events


def _active_word_at(events, t):
    hit = [w for (s, e, w) in events if s <= t < e]
    return hit[-1] if hit else None


# ── Frontend model (mirror of activeWordTiming.getCurrentWordIndex) ──

_PUNCT_PAUSE = {",": .15, ";": .16, ":": .12, ".": .22, "!": .22, "?": .24, "—": .12, "–": .10}
_FAST_WORDS = set("the a an to in on at of for and but or is was are were it its this that".split())


def _fallback_windows(words, start, end, wps=3.0):
    rate_scale = max(0.6, min(1.6, 3.0 / wps))
    anticipation = _ANTICIPATION_S * rate_scale
    net = anticipation - _AUDIO_BUFFER_S
    duration = end - start
    total_chars = sum(len(w) for w in words) or 1
    punct = [_PUNCT_PAUSE.get(w[-1:], 0.0) * rate_scale for w in words]
    base_overhead = 0.04 * rate_scale * len(words)
    total_pause = base_overhead + sum(punct)
    char_time = max(duration - total_pause, duration * 0.45)
    pause_scale = (duration - char_time) / max(total_pause, 0.01)
    raw = []
    for i, w in enumerate(words):
        d = char_time * (len(w) / total_chars) + (0.04 * rate_scale + punct[i]) * pause_scale
        if w.lower().rstrip(".,!?;:—–") in _FAST_WORDS:
            d *= 0.75
        if i == 0:
            d *= 1.15
        elif i == len(words) - 1:
            d *= 1.10
        raw.append(d)
    tot = sum(raw) or 1.0
    raw = [d * duration / tot for d in raw]
    windows, cum = [], 0.0
    for d in raw:
        windows.append((start + cum - net, start + cum + d - net))
        cum += d
    return windows


def test_fallback_export_highlights_same_word_as_preview():
    # Word-LESS cue -> both preview and export use the char-proportional model.
    # start_time=0 so the ASS (clip-relative) event times share the segment's
    # coordinates. Sample times are clip-relative.
    text = "the quick brown fox jumps over the lazy dog today"
    dur = 6.0
    seg = TranscriptSegment(start=0.0, end=dur, text=text, speaker="Speaker 1")
    ass = _gen([seg], 0.0, dur)
    events = _word_events(ass)
    words = text.split()
    assert len(events) == len(words), (len(events), len(words))

    # generate_ass derives per-speaker wps from the segment (word count / span);
    # mirror it so rate_scale (hence the net offset) matches.
    wps = len(words) / dur
    windows = _fallback_windows(words, 0.0, dur, wps=wps)
    checked = 0
    for i, (ws, we) in enumerate(windows):
        mid = (ws + we) / 2.0
        if not (0.05 < mid < dur - 0.05):
            continue  # skip the clamped extremes
        got = _active_word_at(events, mid)
        assert got == words[i], (
            f"word {i} '{words[i]}' active window mid={mid:.3f} but export "
            f"highlighted '{got}' — preview/export drift (bug 1: ~0.14s early)")
        checked += 1
    assert checked >= 4


def test_realword_gap_highlights_upcoming_word_not_previous():
    # Two words with a 1.5s silence between them. The preview advances to the
    # UPCOMING word at the prior word's end; the export must not hold the
    # previous word across the pause.
    # start_time=0 so ASS times are in the segment's own coordinates.
    seg = TranscriptSegment(
        start=0.0, end=3.0, text="Alpha Bravo", speaker="Speaker 1",
        words=[
            WordTimestamp(start=0.0, end=0.5, word="Alpha"),
            WordTimestamp(start=2.0, end=2.8, word="Bravo"),
        ],
    )
    ass = _gen([seg], 0.0, 3.0)
    events = _word_events(ass)
    assert [w for _, _, w in events] == ["Alpha", "Bravo"], events

    # Mid-gap (1.0s): Alpha ended (0.5) and Bravo has not begun (2.0).
    # getCurrentWordIndex returns Bravo (first word whose end is still ahead of
    # the adjusted time). Export must agree — not hold Alpha across the pause.
    assert _active_word_at(events, 1.0) == "Bravo"
    # Right after Alpha's end it should already be Bravo, not Alpha.
    assert _active_word_at(events, 0.8) == "Bravo"


def test_realword_contiguous_is_unaffected():
    # Contiguous words (w[i-1].end == w[i].start): the gap-parity fix is a
    # no-op and each word lights within its own span.
    seg = TranscriptSegment(
        start=0.0, end=3.0, text="one two three", speaker="Speaker 1",
        words=[
            WordTimestamp(start=0.0, end=1.0, word="one"),
            WordTimestamp(start=1.0, end=2.0, word="two"),
            WordTimestamp(start=2.0, end=3.0, word="three"),
        ],
    )
    ass = _gen([seg], 0.0, 3.0)
    events = _word_events(ass)
    assert [w for _, _, w in events] == ["one", "two", "three"], events
    assert _active_word_at(events, 0.4) == "one"
    assert _active_word_at(events, 1.4) == "two"
    assert _active_word_at(events, 2.4) == "three"
