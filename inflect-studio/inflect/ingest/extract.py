"""Audio extraction from any container via ffmpeg (subprocess).

ffmpeg/ffprobe are external binaries (not pip-installable). We shell out rather
than bind a library so any standard install on PATH works. All failures raise
:class:`FfmpegError` with a message suitable for a GUI dialog.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import REFERENCE_SR


class FfmpegError(RuntimeError):
    """ffmpeg/ffprobe missing, or a conversion failed."""


def find_ffmpeg(ffmpeg_path: str = "ffmpeg") -> str | None:
    """Return a usable ffmpeg path (resolving from PATH), or ``None``."""
    if Path(ffmpeg_path).exists():
        return ffmpeg_path
    return shutil.which(ffmpeg_path)


def ffmpeg_available(ffmpeg_path: str = "ffmpeg") -> bool:
    return find_ffmpeg(ffmpeg_path) is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise FfmpegError(
            f"'{cmd[0]}' was not found. Install ffmpeg and ensure it is on your "
            f"PATH (https://ffmpeg.org/download.html)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-5:]
        raise FfmpegError(
            f"{cmd[0]} failed (exit {exc.returncode}):\n" + "\n".join(tail)
        ) from exc


def probe_duration(input_path: str | Path, ffprobe_path: str = "ffprobe") -> float:
    """Return media duration in seconds (0.0 if unknown)."""
    exe = shutil.which(ffprobe_path) or ffprobe_path
    cmd = [
        exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        out = _run(cmd).stdout
        data = json.loads(out or "{}")
        return float(data.get("format", {}).get("duration", 0.0) or 0.0)
    except (FfmpegError, json.JSONDecodeError, ValueError):
        return 0.0


def extract_audio(
    input_path: str | Path,
    out_wav: str | Path,
    sample_rate: int = REFERENCE_SR,
    mono: bool = True,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Decode ``input_path`` (mp4/mp3/wav/mkv/...) to a PCM-16 wav.

    Produces a ``sample_rate`` Hz, optionally mono, WAV. Returns the output path.
    """
    exe = find_ffmpeg(ffmpeg_path)
    if exe is None:
        raise FfmpegError(
            "ffmpeg was not found on PATH. Install it from "
            "https://ffmpeg.org/download.html (or set its path in Settings)."
        )
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-i",
        str(input_path),
        "-vn",  # drop any video stream
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        str(out_wav),
    ]
    _run(cmd)
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise FfmpegError(
            f"No audio was produced from '{Path(input_path).name}'. The file may "
            f"have no audio track or an unsupported codec."
        )
    return out_wav


def extract_reference_pair(
    input_path: str | Path,
    work_dir: str | Path,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[Path, Path]:
    """Produce both the 24 kHz working wav and a 44.1 kHz audition copy.

    The 24 kHz mono wav feeds VAD/cloning; the 44.1 kHz copy is for auditioning
    candidate clips at higher fidelity in the import wizard.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav24 = extract_audio(
        input_path, work_dir / "ref_24k.wav", REFERENCE_SR, mono=True, ffmpeg_path=ffmpeg_path
    )
    wav44 = extract_audio(
        input_path, work_dir / "ref_44k.wav", 44_100, mono=True, ffmpeg_path=ffmpeg_path
    )
    return wav24, wav44


def encode_output(
    in_wav: str | Path,
    out_path: str | Path,
    fmt: str = "mp3",
    bitrate: str = "192k",
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Encode a finished wav mix to mp3 (or copy/convert to another format)."""
    exe = find_ffmpeg(ffmpeg_path)
    if exe is None:
        raise FfmpegError("ffmpeg is required to export MP3. Install it and retry.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-y", "-i", str(in_wav)]
    if fmt == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-b:a", bitrate]
    cmd += [str(out_path)]
    _run(cmd)
    return out_path
