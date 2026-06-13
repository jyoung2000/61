"""Project save/load round-trips and defensive error handling."""

from __future__ import annotations

import json

import pytest

from inflect.document.project_io import (
    CURRENT_VERSION,
    FORMAT_ID,
    ProjectLoadError,
    load_project,
    save_project,
)
from inflect.document.spans import Document, Inflection


def _make_doc() -> Document:
    doc = Document(
        text="Hello there. General Kenobi! You are a bold one.",
        voice_profile_id="voice-xyz",
        default_inflection=Inflection(speed=0.95, emo_alpha=0.7),
    )
    doc.apply_inflection(
        0, 12, Inflection(emotion_vector=[0.8] + [0.0] * 7, pause_after_ms=200), color_idx=3
    )
    doc.apply_inflection(13, 28, Inflection(emo_text="surprised, loud"), color_idx=1)
    return doc


def test_round_trip_preserves_everything(tmp_path):
    doc = _make_doc()
    out = save_project(tmp_path / "demo", doc, engine="fish", engine_params={"x": 2})
    assert out.path.suffix == ".inflect"
    assert out.path.exists()

    loaded = load_project(out.path)
    assert loaded.document.text == doc.text
    assert loaded.document.voice_profile_id == "voice-xyz"
    assert loaded.engine == "fish"
    assert loaded.engine_params == {"x": 2}
    assert [(s.start, s.end, s.color_idx) for s in loaded.document.sorted_spans()] == [
        (0, 12, 3),
        (13, 28, 1),
    ]
    first = loaded.document.sorted_spans()[0].inflection
    assert first.emotion_vector[0] == pytest.approx(0.8)
    assert first.pause_after_ms == 200
    assert loaded.document.sorted_spans()[1].inflection.emo_text == "surprised, loud"
    assert loaded.document.default_inflection.speed == pytest.approx(0.95)


def test_suffix_is_appended_when_missing(tmp_path):
    out = save_project(tmp_path / "noext", Document(text="hi"))
    assert out.path.name == "noext.inflect"


def test_timestamps_present(tmp_path):
    out = save_project(tmp_path / "p", Document(text="hi"))
    data = json.loads(out.path.read_text("utf-8"))
    assert data["created"]
    assert data["modified"]
    assert data["version"] == CURRENT_VERSION
    assert data["format"] == FORMAT_ID


def test_created_preserved_on_resave(tmp_path):
    p = tmp_path / "p.inflect"
    first = save_project(p, Document(text="v1"))
    second = save_project(p, Document(text="v2"), created=first.created)
    assert second.created == first.created


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(ProjectLoadError):
        load_project(tmp_path / "nope.inflect")


def test_load_bad_json_raises(tmp_path):
    p = tmp_path / "broken.inflect"
    p.write_text("{not valid json", "utf-8")
    with pytest.raises(ProjectLoadError):
        load_project(p)


def test_load_wrong_format_raises(tmp_path):
    p = tmp_path / "wrong.inflect"
    p.write_text(json.dumps({"format": "something-else", "version": 1}), "utf-8")
    with pytest.raises(ProjectLoadError):
        load_project(p)


def test_load_non_object_root_raises(tmp_path):
    p = tmp_path / "arr.inflect"
    p.write_text(json.dumps([1, 2, 3]), "utf-8")
    with pytest.raises(ProjectLoadError):
        load_project(p)


def test_unknown_future_keys_ignored(tmp_path):
    p = tmp_path / "future.inflect"
    payload = {
        "format": FORMAT_ID,
        "version": CURRENT_VERSION,
        "created": "2026-01-01T00:00:00+00:00",
        "modified": "2026-01-01T00:00:00+00:00",
        "engine": "indextts2",
        "engine_params": {},
        "future_feature": {"nonsense": True},
        "document": {
            "text": "future",
            "voice_profile_id": None,
            "default_inflection": {"speed": 1.0, "unknown_field": 5},
            "spans": [],
        },
    }
    p.write_text(json.dumps(payload), "utf-8")
    loaded = load_project(p)
    assert loaded.document.text == "future"


def test_newer_version_is_best_effort_loaded(tmp_path):
    p = tmp_path / "newer.inflect"
    payload = {
        "format": FORMAT_ID,
        "version": CURRENT_VERSION + 5,
        "document": {"text": "from the future"},
    }
    p.write_text(json.dumps(payload), "utf-8")
    loaded = load_project(p)
    assert loaded.document.text == "from the future"
