"""The editor-state corruption detector must catch the reverse-sync poison
(out-of-order, mass-duplicated, or backwards subtitle items) so a stale cache
is never persisted or served back — which previously re-poisoned the
transcript on load.
"""

from backend.services.transcript_sync import is_editor_state_corrupt


def _sub(start, end, text):
    return {"type": "subtitle", "start": start, "end": end, "subtitleText": text}


def _state(subs, extra=None):
    items = list(subs)
    if extra:
        items += extra
    return {"items": items}


def test_clean_state_not_corrupt():
    state = _state([_sub(0, 2, "a"), _sub(2, 4, "b"), _sub(4, 6, "c")])
    corrupt, _ = is_editor_state_corrupt(state)
    assert corrupt is False


def test_out_of_order_is_corrupt():
    # Mirrors the uploaded TXT: a cue jumps back to 0:58 after 24:19.
    state = _state([_sub(0, 2, "a"), _sub(1459, 1461, "late"), _sub(58, 60, "back")])
    corrupt, reason = is_editor_state_corrupt(state)
    assert corrupt is True
    assert "out-of-order" in reason


def test_backwards_cue_is_corrupt():
    state = _state([_sub(0, 2, "a"), _sub(9, 5, "backwards")])
    corrupt, reason = is_editor_state_corrupt(state)
    assert corrupt is True
    assert "backwards" in reason


def test_mass_duplicates_is_corrupt():
    # The OP-lyric block repeated ~8× at (nearly) the same timestamp.
    lyric = "Just wild beat communication"
    subs = [_sub(360 + i * 0.01, 380, lyric) for i in range(8)]
    subs += [_sub(0, 2, "intro"), _sub(2, 4, "next")]
    # Sort so it's not flagged out-of-order first — duplicates are the signal.
    subs.sort(key=lambda s: s["start"])
    corrupt, reason = is_editor_state_corrupt(_state(subs))
    assert corrupt is True
    assert "duplicate" in reason


def test_a_few_repeated_short_lines_ok():
    # Legit recurring interjections must NOT trip the duplicate heuristic.
    subs = [_sub(i * 5, i * 5 + 1, "Roger.") for i in range(3)]
    subs += [_sub(i * 5 + 100, i * 5 + 101, f"line {i}") for i in range(20)]
    subs.sort(key=lambda s: s["start"])
    corrupt, _ = is_editor_state_corrupt(_state(subs))
    assert corrupt is False


def test_non_subtitle_items_ignored():
    state = {"items": [
        {"type": "video", "start": 0, "end": 100},
        {"type": "audio", "start": 0, "end": 100},
    ]}
    assert is_editor_state_corrupt(state) == (False, "")


def test_empty_or_malformed_state_safe():
    assert is_editor_state_corrupt({}) == (False, "")
    assert is_editor_state_corrupt({"items": "nope"}) == (False, "")
    assert is_editor_state_corrupt(None) == (False, "")
