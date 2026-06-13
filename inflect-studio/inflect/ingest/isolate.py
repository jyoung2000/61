"""Optional vocal isolation via Demucs (htdemucs), plus a pure suggestion heuristic.

Isolation is expensive (large model, lots of VRAM) so it is opt-in. The heuristic
:func:`should_suggest_isolation` -- pure and unit-tested -- decides whether the
import wizard *recommends* it: low speech ratio or a flat (music/noise-like)
spectrum both point at a non-clean source.

Demucs and a TTS engine must never be resident in VRAM at the same time, so the
caller routes loading through the model manager and unloads Demucs afterwards.
"""

from __future__ import annotations

import numpy as np


def spectral_flatness(audio: np.ndarray) -> float:
    """Spectral flatness in ``[0, 1]``: ~0 for a pure tone, ~1 for white noise.

    Geometric mean over arithmetic mean of the power spectrum. Clean speech is
    peaky (formants) → low flatness; broadband noise/music beds → higher.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio.size < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(audio)) ** 2
    spec = spec[1:]  # drop DC
    spec = spec[spec > 0]
    if spec.size == 0:
        return 0.0
    geo = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    return float(geo / (arith + 1e-12))


def should_suggest_isolation(
    speech_ratio: float,
    flatness: float,
    *,
    speech_threshold: float = 0.6,
    flatness_threshold: float = 0.30,
) -> bool:
    """Recommend Demucs when speech coverage is low or the spectrum is flat."""
    return speech_ratio < speech_threshold or flatness > flatness_threshold


# --------------------------------------------------------------------------- #
# Demucs wrapper (not unit-tested -- requires the model + torch)
# --------------------------------------------------------------------------- #
def isolate_vocals(
    audio: np.ndarray,
    sr: int,
    device: str = "cuda",
    progress_cb=None,
) -> tuple[np.ndarray, int]:  # pragma: no cover - requires model
    """Return the isolated vocal stem (mono) at the input sample rate.

    Resamples to Demucs' native 44.1 kHz, separates, takes the ``vocals`` stem
    and resamples back. Frees the model before returning so VRAM is reclaimed
    ahead of loading a TTS engine.
    """
    import gc

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    from ..synth.assemble import resample

    model = get_model("htdemucs")
    model.eval()
    model.to(device)
    model_sr = int(model.samplerate)

    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav = resample(mono, sr, model_sr)
    # Demucs expects (channels, length); duplicate mono to stereo.
    stereo = np.stack([wav, wav], axis=0)
    tensor = torch.from_numpy(stereo).float()

    ref = tensor.mean(0)
    std = ref.std() + 1e-8
    tensor = (tensor - ref.mean()) / std

    with torch.no_grad():
        sources = apply_model(
            model, tensor[None], device=device, progress=bool(progress_cb)
        )[0]
    sources = sources * std + ref.mean()

    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx].mean(0).cpu().numpy().astype(np.float32)

    # Free Demucs aggressively before any TTS engine loads.
    del model, sources, tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resample(vocals, model_sr, sr), sr
