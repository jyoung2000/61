"""Versioned ``.inflect`` project save / load.

A project bundles the :class:`Document` (text + spans + default delivery +
voice profile id) together with a little UI state (the last selected engine
mode and its params) and timestamps. The on-disk format is plain JSON so it is
diff-friendly and forward-debuggable.

Loading is defensive: unknown future keys are ignored, older versions are run
through :func:`_migrate`, and malformed files raise :class:`ProjectLoadError`
with a human-readable message instead of a raw ``json``/``KeyError`` traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .spans import Document

FORMAT_ID = "inflect-studio-project"
CURRENT_VERSION = 1
PROJECT_SUFFIX = ".inflect"


class ProjectLoadError(Exception):
    """Raised when a ``.inflect`` file cannot be parsed into a project."""


@dataclass
class ProjectFile:
    """In-memory representation of a loaded/saved project."""

    document: Document = field(default_factory=Document)
    engine: str = "indextts2"  # last selected engine mode (Final by default)
    engine_params: dict = field(default_factory=dict)
    created: str = ""
    modified: str = ""
    path: Path | None = None

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_ID,
            "version": CURRENT_VERSION,
            "created": self.created,
            "modified": self.modified,
            "engine": self.engine,
            "engine_params": self.engine_params,
            "document": self.document.to_dict(),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_project(
    path: str | Path,
    document: Document,
    engine: str = "indextts2",
    engine_params: dict | None = None,
    created: str | None = None,
) -> ProjectFile:
    """Write ``document`` to ``path`` (``.inflect`` is appended if missing)."""
    path = Path(path)
    if path.suffix != PROJECT_SUFFIX:
        path = path.with_suffix(PROJECT_SUFFIX)
    project = ProjectFile(
        document=document,
        engine=engine,
        engine_params=engine_params or {},
        created=created or _now_iso(),
        modified=_now_iso(),
        path=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically via a temp file then replace, so a crash mid-write does
    # not corrupt an existing project.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(project.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    tmp.replace(path)
    return project


def load_project(path: str | Path) -> ProjectFile:
    """Read and validate a project file, applying migrations as needed."""
    path = Path(path)
    try:
        raw = path.read_text("utf-8")
    except OSError as exc:
        raise ProjectLoadError(f"Could not read project file: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectLoadError(f"Project file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProjectLoadError("Project file root must be a JSON object.")
    if data.get("format") != FORMAT_ID:
        raise ProjectLoadError(
            "This does not look like an Inflect Studio project "
            f"(missing/incorrect 'format' field)."
        )

    data = _migrate(data)

    try:
        document = Document.from_dict(data.get("document", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectLoadError(f"Project document is malformed: {exc}") from exc

    return ProjectFile(
        document=document,
        engine=data.get("engine", "indextts2"),
        engine_params=data.get("engine_params", {}) or {},
        created=data.get("created", ""),
        modified=data.get("modified", ""),
        path=path,
    )


def _migrate(data: dict) -> dict:
    """Upgrade an older project dict in place to :data:`CURRENT_VERSION`."""
    version = int(data.get("version", 1))
    # No historical versions yet; this is where future migrations chain, e.g.:
    #   if version < 2: data = _v1_to_v2(data); version = 2
    if version > CURRENT_VERSION:
        # Forward-compatible best effort: warn-by-raising would be hostile, so we
        # just attempt to load it and let field-level defaults absorb new keys.
        pass
    data["version"] = CURRENT_VERSION
    return data
