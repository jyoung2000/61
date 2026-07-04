"""Golden-file snapshot for generate_ass() (parity Phase 5).

One canonical settings object — font 30 / weight 700 / outline 2 /
background on / radius 8 / offset 4 / max-width 90 / active-word on —
rendered at 1080×1920. Any styling regression (style lines, margins,
karaoke tags, dialogue timing) diffs loudly against the stored golden
file instead of slipping through as a subtle preview↔export mismatch.

Regenerate intentionally after a REVIEWED styling change:

    REGEN_ASS_GOLDEN=1 python -m pytest backend/tests/test_ass_golden.py -q

and commit the updated fixture together with the change that caused it.
"""

import difflib
import os

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.ass_generator import generate_ass

GOLDEN_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "golden_subtitle_1080x1920.ass")


def _canonical_segments():
    return [
        TranscriptSegment(
            start=10.0, end=13.0, text="Welcome back to the channel",
            speaker="Speaker 1",
            words=[
                WordTimestamp(start=10.0, end=10.6, word="Welcome"),
                WordTimestamp(start=10.6, end=11.1, word="back"),
                WordTimestamp(start=11.1, end=11.4, word="to"),
                WordTimestamp(start=11.4, end=11.8, word="the"),
                WordTimestamp(start=11.8, end=13.0, word="channel"),
            ],
        ),
        TranscriptSegment(
            start=13.2, end=16.0, text="Today we ship the editor",
            speaker="Speaker 2",
            words=[
                WordTimestamp(start=13.2, end=13.7, word="Today"),
                WordTimestamp(start=13.7, end=14.1, word="we"),
                WordTimestamp(start=14.1, end=14.9, word="ship"),
                WordTimestamp(start=14.9, end=15.3, word="the"),
                WordTimestamp(start=15.3, end=16.0, word="editor"),
            ],
        ),
    ]


def _generate_canonical():
    return generate_ass(
        segments=_canonical_segments(),
        start_time=10.0,
        end_time=16.0,
        font="DM Sans",
        font_size=30,
        font_weight=700,
        font_color="#FFFFFF",
        position="bottom",
        speaker_colors={},
        use_speaker_colors=True,
        video_width=1080,
        video_height=1920,
        background_enabled=True,
        background_color="#000000",
        background_opacity=75,
        background_radius=8,
        outline_color="#000000",
        outline_opacity=100,
        outline_width=2,
        max_width_pct=90,
        offset_v_pct=4,
        active_word_enabled=True,
        active_word_color="#FFD700",
        active_word_outline_color="#000000",
        active_word_bg_color="#000000",
        active_word_bg_opacity=0,
        active_word_bg_radius=4,
        # Golden must not depend on the host's CPS-enforcement setting
        enforce_readability_rules=False,
    )


def test_golden_ass_snapshot():
    result = _generate_canonical()
    assert result and "[Script Info]" in result

    if os.environ.get("REGEN_ASS_GOLDEN") == "1" or not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            f.write(result)
        if os.environ.get("REGEN_ASS_GOLDEN") == "1":
            return  # explicit regeneration run
        raise AssertionError(
            "Golden file was missing and has been bootstrapped at "
            f"{GOLDEN_PATH} — review it, commit it, and re-run.")

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        expected = f.read()

    if result != expected:
        diff = "\n".join(difflib.unified_diff(
            expected.splitlines(), result.splitlines(),
            fromfile="golden", tofile="current", lineterm="", n=2))
        raise AssertionError(
            "generate_ass() output changed for the canonical settings. If "
            "this styling change is INTENTIONAL, regenerate with "
            "REGEN_ASS_GOLDEN=1 and commit the fixture.\n" + diff)


def test_golden_covers_the_contract_features():
    """The canonical render must actually exercise the styled features —
    guards against a refactor quietly dropping them from the fixture."""
    content = _generate_canonical()
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "DM Sans" in content
    # active-word highlighting renders per-word dialogue events
    assert content.count("Dialogue:") >= 10
    # background box style (BorderStyle=4 or 3 depending on build)
    assert "BorderStyle" in content
