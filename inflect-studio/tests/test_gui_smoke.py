"""Headless GUI smoke test (offscreen Qt).

Validates the Qt API usage that py_compile can't see: the editor's underline
rendering (QTextEdit.ExtraSelection + QTextCharFormat enums), the inspector's
inflection round-trip, and the timeline drawing a result. Skips cleanly when
PySide6 (or its system libs) are unavailable.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

try:  # the import itself can fail without libEGL/libGL present
    from PySide6.QtWidgets import QApplication
    _APP_OK = True
except Exception:  # pragma: no cover
    _APP_OK = False

pytestmark = pytest.mark.skipif(not _APP_OK, reason="Qt platform libs unavailable")


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


def test_editor_apply_and_highlight(app):
    from inflect.document.spans import Document, Inflection
    from inflect.ui.editor import InflectionTextEdit

    ed = InflectionTextEdit()
    ed.set_document_model(Document(text="Hello there. General Kenobi."))
    cur = ed.textCursor()
    cur.setPosition(0)
    cur.setPosition(12, cur.MoveMode.KeepAnchor)
    ed.setTextCursor(cur)
    ed.apply_inflection_to_selection(Inflection(emotion_vector=[0, 0.8] + [0] * 6, speed=1.1))
    assert len(ed.extraSelections()) == 1
    assert ed.model.sorted_spans()[0].start == 0
    # Typing before the span shifts it (offset remap through Qt's signal).
    cur2 = ed.textCursor()
    cur2.setPosition(0)
    ed.setTextCursor(cur2)
    ed.insertPlainText("X")
    assert ed.model.sorted_spans()[0].start == 1


def test_inspector_round_trip(app):
    from inflect.document.spans import Inflection
    from inflect.ui.inspector import Inspector

    insp = Inspector()
    inf = Inflection(
        emotion_vector=[0.0, 0.7, 0, 0, 0, 0, 0.3, 0],
        emo_text="tense",
        speed=1.2,
        pause_after_ms=250,
        engine="fish",
    )
    insp.set_context(inf, is_default=False, has_selection=True)
    out = insp.current_inflection()
    assert abs(out.emotion_vector[1] - 0.7) < 0.02
    assert abs(out.emotion_vector[6] - 0.3) < 0.02
    assert out.emo_text == "tense"
    assert out.speed == pytest.approx(1.2, abs=0.01)
    assert out.pause_after_ms == 250
    assert out.engine == "fish"


def test_timeline_set_result(app):
    from inflect.synth.pipeline import RenderedSegment, RenderResult
    from inflect.ui.timeline import TimelineDock

    audio = (0.3 * np.sin(np.arange(24000) * 0.05)).astype(np.float32)
    seg = RenderedSegment(
        seg_id=0, char_start=0, char_end=10, sample_rate=24000, n_samples=24000,
        pause_after_ms=0, engine="chatterbox", from_cache=False, synth_seconds=0.1,
        audio=audio,
    )
    tl = TimelineDock()
    tl.set_result(RenderResult(audio=audio, sample_rate=24000, segments=[seg]))
    tl.set_position(0.5)  # must not raise
