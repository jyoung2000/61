"""collapse_song_choruses: fold a sung opening/ending THEME (mis-transcribed as
duplicated dialogue) into a single "[♪ … theme ♪]" marker, keyed on chorus
repetition so real dialogue is never dropped and a preview narrated over the
ending theme survives."""

from backend.services.transcript_sanitize import collapse_song_choruses, _is_marker


def _cue(start, end, text):
    return {"start": float(start), "end": float(end), "text": text}


def _ending_theme_transcript():
    # ~24-min episode: dialogue tail, then the ending theme (chorus repeats
    # twice — the tell-tale mis-transcription), with a next-episode preview
    # (proper nouns) narrated over it.
    rows = [_cue(60 + i * 6, 66 + i * 6, f"Dialogue line number {i} here.") for i in range(200)]
    tail = [
        _cue(1316, 1319, "I'll kill you."),                       # climactic DIALOGUE — must survive
        _cue(1322, 1326, "What kind of person is he?!"),          # DIALOGUE — must survive
        _cue(1340, 1346, "I don't like just love. It irritates me."),     # chorus #1
        _cue(1348, 1354, "I love it when things get awkward."),           # chorus #2
        _cue(1403, 1409, "The Alliance's Marina sends out troops to find the Gundam."),  # PREVIEW
        _cue(1410, 1416, "But OZ's Zechs finds it first with the Cancer suit."),         # PREVIEW
        _cue(1420, 1426, "I don't like just love. It irritates me."),     # chorus #1 repeat
        _cue(1428, 1434, "I love it when things get awkward."),           # chorus #2 repeat
        _cue(1440, 1444, "That's why I care."),
    ]
    return rows + tail


def test_collapses_ending_theme_chorus_to_marker():
    rows = _ending_theme_transcript()
    out, changed = collapse_song_choruses(rows, "en")
    assert changed
    texts = [r["text"] for r in out]
    # A single ending-theme marker was inserted.
    markers = [t for t in texts if _is_marker(t) and "Ending theme" in t]
    assert len(markers) == 1
    # The repeated chorus lyric no longer appears as dialogue.
    assert "I don't like just love. It irritates me." not in texts


def test_preserves_dialogue_and_preview():
    rows = _ending_theme_transcript()
    out, _ = collapse_song_choruses(rows, "en")
    texts = [r["text"] for r in out]
    # Climactic dialogue survives.
    assert "I'll kill you." in texts
    assert "What kind of person is he?!" in texts
    # Preview narration (proper nouns) survives — never eaten by the theme.
    assert any("Marina sends out troops" in t for t in texts)
    assert any("Zechs finds it first" in t for t in texts)


def test_pure_dialogue_is_untouched():
    rows = [_cue(i * 5, i * 5 + 3, t) for i, t in enumerate([
        "Are we under attack?", "This is Duo here.", "I've destroyed the main motor.",
        "Yes, sir.", "But there's nothing here.", "Yes, sir.",
        "Drop your weapons and surrender.", "My name's Wufei.",
        "I'm not hiding anywhere.", "We're surrounded! Who are they!",
    ])]
    out, changed = collapse_song_choruses(rows, "en")
    assert not changed and out == rows


def test_midepisode_repeat_not_treated_as_theme():
    # A repeated sentence in the MIDDLE of the episode (outside head/tail
    # windows) is not a theme and must be left alone.
    rows = [_cue(600 + i * 4, 603 + i * 4, t) for i, t in enumerate([
        "The mission has changed completely.", "We must retreat now.",
        "The mission has changed completely.", "Understood, sir.",
    ])]
    # pad the tail so the window math has a late anchor far from these cues
    rows += [_cue(1400 + i * 4, 1403 + i * 4, f"Late unrelated line {i}.") for i in range(6)]
    out, changed = collapse_song_choruses(rows, "en")
    assert not changed


def test_cjk_target_passthrough():
    rows = _ending_theme_transcript()
    out, changed = collapse_song_choruses(rows, "ja")
    assert not changed and out == rows


def test_disabled_via_config(monkeypatch):
    from backend.config import settings
    monkeypatch.setattr(settings, "TRANSCRIPT_MARK_THEME_SONGS", False)
    out, changed = collapse_song_choruses(_ending_theme_transcript(), "en")
    assert not changed
