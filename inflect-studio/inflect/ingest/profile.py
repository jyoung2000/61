"""Voice Profile model + persistent Voice Library (JSON index + wav files).

Layout under ``$INFLECT_HOME/voices`` (see :mod:`inflect.config`):

    voices/
      index.json                 # list of profile metadata
      <uuid>/
        reference.wav            # 24 kHz mono cloning reference
        audition.wav             # optional 44.1 kHz higher-fidelity preview
        meta.json                # same data as the index entry (self-describing)

The library is plain files so it survives crashes, is easy to back up, and is
fully unit-testable without any model dependency.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REFERENCE_WAV = "reference.wav"
AUDITION_WAV = "audition.wav"
META_JSON = "meta.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class VoiceProfile:
    """Metadata for one cloned voice. Audio lives next to it on disk."""

    id: str
    name: str
    source_filename: str = ""
    duration_s: float = 0.0
    sample_rate: int = 24_000
    created: str = field(default_factory=_now_iso)
    notes: str = ""
    consent: bool = False  # "I have permission to clone this voice."
    has_audition: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class VoiceLibraryError(RuntimeError):
    """Profile not found / could not be created."""


class VoiceLibrary:
    """CRUD over the on-disk voice library."""

    def __init__(self, voices_dir: str | Path) -> None:
        self.root = Path(voices_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def profile_dir(self, profile_id: str) -> Path:
        return self.root / profile_id

    def reference_path(self, profile_id: str) -> Path:
        return self.profile_dir(profile_id) / REFERENCE_WAV

    def audition_path(self, profile_id: str) -> Path:
        return self.profile_dir(profile_id) / AUDITION_WAV

    # -- index io ----------------------------------------------------------
    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text("utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_index(self, entries: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(self.index_path)

    # -- queries -----------------------------------------------------------
    def list_profiles(self) -> list[VoiceProfile]:
        return [VoiceProfile.from_dict(e) for e in self._read_index()]

    def get(self, profile_id: str) -> VoiceProfile:
        for entry in self._read_index():
            if entry.get("id") == profile_id:
                return VoiceProfile.from_dict(entry)
        raise VoiceLibraryError(f"No voice profile with id {profile_id!r}")

    def exists(self, profile_id: str) -> bool:
        return any(e.get("id") == profile_id for e in self._read_index())

    # -- mutations ---------------------------------------------------------
    def add(
        self,
        name: str,
        reference_wav: str | Path,
        *,
        source_filename: str = "",
        duration_s: float = 0.0,
        sample_rate: int = 24_000,
        notes: str = "",
        consent: bool = False,
        audition_wav: str | Path | None = None,
    ) -> VoiceProfile:
        """Copy ``reference_wav`` into a new profile dir and index it."""
        ref = Path(reference_wav)
        if not ref.exists():
            raise VoiceLibraryError(f"Reference wav does not exist: {ref}")
        profile_id = uuid.uuid4().hex
        pdir = self.profile_dir(profile_id)
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref, self.reference_path(profile_id))

        has_audition = False
        if audition_wav and Path(audition_wav).exists():
            shutil.copyfile(audition_wav, self.audition_path(profile_id))
            has_audition = True

        profile = VoiceProfile(
            id=profile_id,
            name=name.strip() or "Untitled voice",
            source_filename=source_filename,
            duration_s=float(duration_s),
            sample_rate=int(sample_rate),
            notes=notes,
            consent=bool(consent),
            has_audition=has_audition,
        )
        self._persist(profile)
        return profile

    def rename(self, profile_id: str, new_name: str) -> VoiceProfile:
        profile = self.get(profile_id)
        profile.name = new_name.strip() or profile.name
        self._persist(profile)
        return profile

    def update_notes(self, profile_id: str, notes: str) -> VoiceProfile:
        profile = self.get(profile_id)
        profile.notes = notes
        self._persist(profile)
        return profile

    def delete(self, profile_id: str) -> None:
        entries = [e for e in self._read_index() if e.get("id") != profile_id]
        self._write_index(entries)
        pdir = self.profile_dir(profile_id)
        if pdir.exists():
            shutil.rmtree(pdir, ignore_errors=True)

    # -- internals ---------------------------------------------------------
    def _persist(self, profile: VoiceProfile) -> None:
        # Update or insert the index entry, keeping creation order stable.
        entries = self._read_index()
        replaced = False
        for i, entry in enumerate(entries):
            if entry.get("id") == profile.id:
                entries[i] = profile.to_dict()
                replaced = True
                break
        if not replaced:
            entries.append(profile.to_dict())
        self._write_index(entries)
        # Self-describing meta.json next to the audio.
        (self.profile_dir(profile.id) / META_JSON).write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), "utf-8"
        )
