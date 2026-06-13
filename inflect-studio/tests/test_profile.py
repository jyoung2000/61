"""Voice library CRUD on a throwaway directory."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from inflect.ingest.profile import VoiceLibrary, VoiceLibraryError, VoiceProfile


@pytest.fixture()
def wav_file(tmp_path):
    path = tmp_path / "ref.wav"
    sig = (0.2 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)).astype(np.float32)
    sf.write(path, sig, 24000)
    return path


def test_add_and_get(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    prof = lib.add(
        "Alice", wav_file, source_filename="alice.mp4", duration_s=1.0, consent=True
    )
    assert prof.id
    assert lib.exists(prof.id)
    fetched = lib.get(prof.id)
    assert fetched.name == "Alice"
    assert fetched.consent is True
    assert lib.reference_path(prof.id).exists()
    # meta.json written next to audio
    assert (lib.profile_dir(prof.id) / "meta.json").exists()


def test_list_profiles_persisted(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.add("A", wav_file)
    lib.add("B", wav_file)
    # New library instance reads the same index from disk.
    lib2 = VoiceLibrary(tmp_path / "voices")
    names = sorted(p.name for p in lib2.list_profiles())
    assert names == ["A", "B"]


def test_rename_and_notes(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    prof = lib.add("Old", wav_file)
    lib.rename(prof.id, "New")
    lib.update_notes(prof.id, "calm narrator")
    again = lib.get(prof.id)
    assert again.name == "New"
    assert again.notes == "calm narrator"


def test_delete_removes_files_and_index(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    prof = lib.add("Temp", wav_file)
    pdir = lib.profile_dir(prof.id)
    assert pdir.exists()
    lib.delete(prof.id)
    assert not lib.exists(prof.id)
    assert not pdir.exists()


def test_add_with_audition_copy(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    prof = lib.add("WithAudition", wav_file, audition_wav=wav_file)
    assert prof.has_audition
    assert lib.audition_path(prof.id).exists()


def test_get_missing_raises(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    with pytest.raises(VoiceLibraryError):
        lib.get("does-not-exist")


def test_add_missing_reference_raises(tmp_path):
    lib = VoiceLibrary(tmp_path / "voices")
    with pytest.raises(VoiceLibraryError):
        lib.add("X", tmp_path / "nope.wav")


def test_profile_round_trip():
    prof = VoiceProfile(id="x", name="N", duration_s=3.5, consent=True, notes="z")
    assert VoiceProfile.from_dict(prof.to_dict()) == prof


def test_corrupt_index_recovers_gracefully(tmp_path, wav_file):
    lib = VoiceLibrary(tmp_path / "voices")
    lib.add("A", wav_file)
    lib.index_path.write_text("{ broken", "utf-8")
    # A corrupt index reads as empty rather than raising.
    assert lib.list_profiles() == []
    # And we can still add on top of it.
    prof = lib.add("B", wav_file)
    assert lib.exists(prof.id)
