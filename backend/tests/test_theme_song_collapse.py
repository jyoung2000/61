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
    # The song is interrupted by the narrated preview, so each sung block
    # gets its own marker — one before the preview, one after. (A single
    # marker spanning both would claim the preview narration is part of the
    # song.) What matters: markers exist and no lyric ships as dialogue.
    markers = [t for t in texts if _is_marker(t) and "Ending theme" in t]
    assert 1 <= len(markers) <= 2
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


# ── A chorus run must be time-CONTIGUOUS ───────────────────────────────────
# Measured on a shipped run: chorus lines detected at 22:38-23:07 bounded a
# collapse run that reached back across a 60-second gap and absorbed a whole
# scene of dialogue starting at 21:34 — nineteen cues, including
# "I'll kill you.", the episode's signature line — into one 4-second marker.
# Meanwhile the song's own tail (past the last chorus repeat) shipped as
# dialogue. Contiguity + extension fix both directions.

def _run5_tail_transcript():
    # Turn-taking speakers on the body so the through-composed run detector
    # (single-speaker runs) has nothing to bite on — the chorus path is the
    # one under test here, exactly as in the measured run.
    rows = [{**_cue(60 + i * 6, 66 + i * 6, f"Dialogue line number {i} here."),
             "speaker": "Speaker 1" if i % 2 else "Speaker 2"}
            for i in range(200)]                      # ends ~1260s
    tail = [
        # A quiet scene WELL before the song — must never be absorbed.
        _cue(1294, 1298, "It's my birthday tomorrow."),
        _cue(1299, 1302, "I hope you can make it to the party."),
        _cue(1324, 1326, "I'll kill you."),
        _cue(1341, 1343, "What kind of person is he?!"),
        # The ending theme: verse, chorus ×2 ×2 lines, verse tail — contiguous.
        _cue(1359, 1365, "I give him a call out of the blue"),
        _cue(1366, 1372, "He gets on my nerves, just love."),      # chorus A
        _cue(1373, 1377, "He makes me wait, how dare he do that"),  # chorus B
        _cue(1378, 1384, "He gets on my nerves, just love."),      # chorus A repeat
        _cue(1385, 1389, "He makes me wait, how dare he do that"),  # chorus B repeat
        _cue(1390, 1394, "That's why."),                           # song tail (past last chorus)
        _cue(1396, 1401, "You're pushing your luck saying things you can't do"),
        # Next-episode preview — proper nouns — must survive.
        _cue(1410, 1416, "The Gundam that sank is retrieved by Zechs."),
    ]
    for c in tail:
        c.setdefault("speaker", "Speaker 1")
    return rows + tail


def test_chorus_run_cannot_bridge_a_gap_into_dialogue():
    out, changed = collapse_song_choruses(_run5_tail_transcript(), "en")
    assert changed
    texts = [r["text"] for r in out]
    # The dialogue scene before the song survives in full.
    assert "It's my birthday tomorrow." in texts
    assert "I hope you can make it to the party." in texts
    assert "I'll kill you." in texts
    assert "What kind of person is he?!" in texts
    # The preview survives.
    assert any("Zechs" in t for t in texts)


def test_song_tail_past_the_last_chorus_is_absorbed():
    out, _ = collapse_song_choruses(_run5_tail_transcript(), "en")
    texts = [r["text"] for r in out]
    # Every sung line — verses AND the tail after the final chorus repeat —
    # collapses into the marker instead of shipping as dialogue.
    assert "He gets on my nerves, just love." not in texts
    assert "I give him a call out of the blue" not in texts
    assert "That's why." not in texts
    assert "You're pushing your luck saying things you can't do" not in texts
    markers = [r for r in out if _is_marker(r["text"]) and "Ending theme" in r["text"]]
    assert len(markers) == 1, markers
    # The marker spans the SONG, not the dialogue a minute earlier.
    assert 1355.0 <= markers[0]["start"] <= 1360.0
    assert markers[0]["end"] >= 1400.0


def _spk(c, s):
    return {**c, "speaker": s}


def test_run62_opening_verses_collapse_despite_garbled_capitals():
    # The measured OP escape: 10 verse cues 0:30-1:31 whose mid-cue capitals
    # ("Uh", "There", "Right", "Turning") each scored a proper noun and broke
    # the strict run at every second cue — no marker shipped at all.
    op_lines = [
        "The rain isn't falling,",
        "so I can't cool the heat Uh",
        "I want to convey my feelings in the air tonight",
        "Holding you as if warming your wet shoulder There",
        "Your trembling fingertips wander seeking what?",
        "Protecting your gaze Right",
        "want to Turning sorrow into love that's strong trust",
        "in myself",
        "don't reveal the storm letting my hot sweat flow",
        "In my life",
    ]
    rows = [_spk(_cue(30 + i * 6, 34 + i * 6, t), "Speaker 1")
            for i, t in enumerate(op_lines)]
    # Narration follows after a >10s gap (its own group), then dialogue.
    rows += [_spk(_cue(112, 116, "The Earth Sphere Unified Nation overwhelmed each colony."), "Speaker 2")]
    rows += [_spk(_cue(300 + i * 6, 304 + i * 6, f"Dialogue line {i}."),
                  "Speaker 1" if i % 2 else "Speaker 2") for i in range(150)]
    out, changed = collapse_song_choruses(rows, "en")
    assert changed
    texts = [r["text"] for r in out]
    assert any(_is_marker(t) and "Opening theme" in t for t in texts), texts[:6]
    assert not any("trembling fingertips" in t for t in texts)
    assert not any("Turning sorrow" in t for t in texts)
    assert any("Earth Sphere Unified Nation" in t for t in texts)


def test_run62_ending_verses_chain_into_the_marker_and_preview_survives():
    # The measured ED shape: the chorus collapses to a marker, then after a
    # ~24s instrumental bridge the verses ("Just Love …" — the repeated
    # TitleCase HOOK scored 2 proper nouns and read as a preview) shipped as
    # dialogue. They must chain into the same marker; the next-episode
    # preview and the 22:03 signature line must survive.
    rows = [_spk(_cue(100 + i * 7, 104 + i * 7, f"Dialogue line {i}."),
                 "Speaker 1" if i % 2 else "Speaker 2") for i in range(150)]
    rows += [
        _spk(_cue(1290.0, 1294.0, "That's terrible."), "Speaker 1"),
        _spk(_cue(1323.4, 1330.9, "I'm gonna kill you What?"), "Speaker 1"),
        _spk(_cue(1341.2, 1343.1, "Who's that person over there?"), "Speaker 1"),
        # sung chorus (repeats → chorus path creates the marker)
        _spk(_cue(1353.0, 1357.0, "Just wild beat communication tonight"), "Speaker 1"),
        _spk(_cue(1358.0, 1362.0, "standing in the lashing rain forever"), "Speaker 1"),
        _spk(_cue(1363.0, 1367.0, "Just wild beat communication tonight"), "Speaker 1"),
        _spk(_cue(1368.0, 1372.0, "standing in the lashing rain forever"), "Speaker 1"),
        # instrumental bridge ~24s, then the verses with the TitleCase hook
        _spk(_cue(1396.0, 1401.0, "Just Love irritates me I Ts"), "Speaker 1"),
        _spk(_cue(1402.0, 1407.0, "It annoys me when you act like that"), "Speaker 1"),
        _spk(_cue(1408.0, 1414.0, "That's why I say this Just Love gets on my nerves Uh Again"), "Speaker 1"),
        _spk(_cue(1415.0, 1418.0, "don't make unreasonable demands on me"), "Speaker 1"),
        _spk(_cue(1419.0, 1422.0, "Please be careful"), "Speaker 1"),
        # next-episode preview — name-dense, must survive
        _spk(_cue(1431.0, 1437.0, "The Gundam sank to the bottom of the sea near Jack Star"), "Speaker 1"),
        _spk(_cue(1438.0, 1444.0, "But Zechs from Oz was using the new Mobile Suit Cancer"), "Speaker 1"),
        _spk(_cue(1445.0, 1451.0, "Mobile Suit Gundam Wing Episode 2 The Deathscythe Gundam"), "Speaker 1"),
    ]
    out, changed = collapse_song_choruses(rows, "en")
    assert changed
    texts = [r["text"] for r in out]
    assert not any("Just Love" in t for t in texts), \
        [t for t in texts if "Just Love" in t]
    assert not any("lashing rain" in t for t in texts)
    assert any("I'm gonna kill you" in t for t in texts)
    assert any("Zechs" in t for t in texts)
    assert any("Deathscythe" in t for t in texts)
    end_markers = [r for r in out if _is_marker(r["text"]) and "Ending theme" in r["text"]]
    assert len(end_markers) == 1
    # Marker extended across the bridge to cover the verse block.
    assert end_markers[0]["end"] >= 1422.0


def test_school_chatter_is_not_a_theme_despite_soft_tolerance():
    # Same-speaker, 6+ cues, ≥25s — but the cues close their sentences, which
    # is dialogue's signature. The punctuation gate must reject the run.
    lines = [
        "Well, there's nothing we can do about that.",
        "You see, she just came back yesterday.",
        "Of course, being the richest person at our school is just different.",
        "I'd love to go into space once in my life too.",
        "Oh, by the way, it's her birthday tomorrow.",
        "That's right, who is she inviting to the party?",
    ]
    rows = [_spk(_cue(30 + i * 6, 34 + i * 6, t), "Speaker 1")
            for i, t in enumerate(lines)]
    rows += [_spk(_cue(300 + i * 6, 304 + i * 6, f"Dialogue line {i}."),
                  "Speaker 1" if i % 2 else "Speaker 2") for i in range(150)]
    out, changed = collapse_song_choruses(rows, "en")
    texts = [r["text"] for r in out]
    assert any("richest person" in t for t in texts)
    assert not any(_is_marker(t) and "Opening theme" in t for t in texts)
