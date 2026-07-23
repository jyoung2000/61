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


def test_through_composed_opening_theme_collapses_without_a_repeated_chorus():
    """The Gundam Wing cold-open OP has all-distinct verse lines (no chorus to
    key on) and shipped as 'Speaker 1' dialogue. The repetition-independent
    theme-run branch must collapse it: a long, single-speaker, proper-noun-free
    run in the head window — while the narration that follows (proper nouns /
    different speaker) survives."""
    op = [
        _cue(30, 36, "The rain won't fall, so I can't cool this fever"),
        _cue(36, 48, "I want to convey my feelings in the air tonight"),
        _cue(48, 51, "Your trembling fingertips wander,"),
        _cue(51, 54, "seeking something"),
        _cue(54, 57, "Turn sorrow into love"),
        _cue(57, 62, "that changes its strength"),
        _cue(62, 70, "Believe in myself I wish to protect"),
        _cue(70, 78, "Because I won't turn away from it in my life"),
    ]
    for c in op:
        c["speaker"] = "Speaker 1"
    rest = [
        {**_cue(80, 90, "Raised on Earth, a person sought hope in space colonies"), "speaker": "Speaker 1"},   # proper noun ends run
        {**_cue(110, 118, "The Earth Sphere Unified Nation overwhelmed each colony."), "speaker": "Speaker 2"},
    ]
    rest += [{**_cue(300 + i * 6, 304 + i * 6, f"Dialogue line {i}."),
              "speaker": "Speaker 1" if i % 2 else "Speaker 2"}   # turn-taking
             for i in range(180)]
    rows = op + rest
    out, changed = collapse_song_choruses(rows, "en")
    assert changed
    texts = [r["text"] for r in out]
    assert any(_is_marker(t) and "theme" in t.lower() for t in texts)
    # None of the OP lyric lines survive as dialogue…
    assert not any("trembling fingertips" in t for t in texts)
    # …but the narration + dialogue do.
    assert any("Earth Sphere Unified Nation" in t for t in texts)
    assert any("Raised on Earth" in t for t in texts)


def test_ordinary_dialogue_scene_is_not_collapsed():
    """A normal opening dialogue scene must NOT be mistaken for a theme: it
    turn-takes between speakers (breaking any single-speaker run) and names
    people/places (proper nouns), so it never forms a long theme-run."""
    lines = [
        ("Speaker 2", "What's wrong, Lily?"),
        ("Speaker 1", "Do you dislike returning to Earth so much?"),
        ("Speaker 1", "Yes, very much."),
        ("Speaker 2", "I've been too busy with work lately."),
        ("Speaker 1", "Father, next time we go into space,"),
        ("Speaker 1", "please take more time."),
        ("Speaker 2", "Lord Darlian, our shuttle will enter the atmosphere."),
        ("Speaker 1", "What is that over there?"),
        ("Speaker 2", "Auto-lock engaged."),
        ("Speaker 1", "Earth's attack mobile suit."),
    ]
    rows = [{**_cue(30 + i * 6, 34 + i * 6, txt), "speaker": spk}
            for i, (spk, txt) in enumerate(lines)]
    rows += [{**_cue(300 + i * 6, 304 + i * 6, f"Later line {i}."),
              "speaker": "Speaker 1" if i % 2 else "Speaker 2"}   # turn-taking
             for i in range(180)]
    out, changed = collapse_song_choruses(rows, "en")
    assert not changed
    assert not any(_is_marker(r["text"]) for r in out)
