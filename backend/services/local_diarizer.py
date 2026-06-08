"""Local, no-HF-token speaker diarization via SpeechBrain ECAPA embeddings.

When pyannote's gated ``speaker-diarization-3.1`` pipeline can't load (no
``HF_TOKEN``), this provides genuine AUDIO-based diarization with no token and
no per-request cost:

  1. Whisper already produced speech segments (start/end) — use them as the
     diarization units (no separate VAD needed).
  2. Embed each segment with SpeechBrain's ECAPA-TDNN speaker encoder
     (``speechbrain/spkrec-ecapa-voxceleb`` — a PUBLIC model, no token). The
     model is fetched once (~80 MB) into the persistent models dir, then runs
     fully offline.
  3. Cluster the embeddings (agglomerative, cosine) to discover who-spoke-when,
     and emit the SAME ``{time_ms: "SPEAKER_xx"}`` timeline shape the pyannote
     diarizer returns — so everything downstream (speaker_fusion, the
     "Speaker N" relabelling, track-speaker linking) works unchanged.

This sits BETWEEN pyannote (best, needs a token) and the visual left/right
mouth-motion heuristic (only on-screen, spatially-separated speakers): real
audio diarization that works for off-screen / audio-only voices too.

Everything heavy is imported lazily and guarded, so an image without
speechbrain simply reports ``is_available() == False`` and the caller falls
back to the visual heuristic — nothing breaks.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger("clipai.local_diarizer")

# Minimum speech to embed reliably; shorter chunks are zero-padded up to this.
_MIN_EMBED_SEC = 0.5
_RESOLUTION_MS = 200
_MAX_SPEAKERS = 8
# Cues whose Whisper ``no_speech_prob`` exceeds this are almost certainly music /
# singing, not speech (reuses the audio-event convention used elsewhere in the
# perceiver). They carry no clean speaker identity, so they're excluded from
# diarization — otherwise OP/ED vocals and instrumental beds embed as their own
# "speakers" and inflate the count. Cues without the field are kept.
_DIARIZE_MAX_NO_SPEECH = 0.6
# A lone-cue cluster is almost always a noise/music outlier rather than a real
# speaker; clusters smaller than this are folded into the nearest real one.
_MIN_CLUSTER_CUES = 2
_MUSIC_MARK = "♪"  # ♪ — text marker for music-only spans


# ── Pure helpers (unit-testable without speechbrain / torch) ───────────────

def _relabel_first_appearance(labels: List[int]) -> List[int]:
    """Remap arbitrary cluster ids to 0,1,2… in order of first appearance, so
    the eventual "Speaker N" numbering follows the timeline."""
    mapping: dict = {}
    out: List[int] = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return out


def _absorb_singleton_clusters(X, labels: List[int],
                               min_cues: int = _MIN_CLUSTER_CUES) -> List[int]:
    """Reassign cues in clusters smaller than ``min_cues`` to the nearest larger
    cluster (by centroid cosine similarity). A music/noise outlier that lands in
    its own one-cue cluster is almost never a real speaker; folding it in keeps
    the speaker count honest. ``X`` is the L2-normalised embedding matrix. Pure
    numpy — directly testable."""
    import numpy as np
    from collections import Counter

    labels = list(labels)
    counts = Counter(labels)
    small = {lab for lab, c in counts.items() if c < min_cues}
    big = [lab for lab, c in counts.items() if c >= min_cues]
    if not small or not big:
        return labels  # nothing to fold, or no larger cluster to fold into

    Xn = np.asarray(X, dtype=np.float64)
    centroids = {}
    for lab in big:
        idx = [i for i, l in enumerate(labels) if l == lab]
        c = Xn[idx].mean(axis=0)
        centroids[lab] = c / (np.linalg.norm(c) or 1.0)
    for i, lab in enumerate(labels):
        if lab in small:
            xn = Xn[i] / (np.linalg.norm(Xn[i]) or 1.0)
            labels[i] = max(big, key=lambda L: float(xn @ centroids[L]))
    return labels


def _cluster_embeddings(
    embeddings,
    num_speakers: Optional[int] = None,
    threshold: float = 0.70,
    max_speakers: int = _MAX_SPEAKERS,
    min_cues: int = _MIN_CLUSTER_CUES,
) -> List[int]:
    """Cluster L2-normalised embeddings by cosine distance.

    With ``num_speakers`` set, cut the dendrogram into exactly that many
    clusters; otherwise discover the count via a distance ``threshold`` (capped
    at ``max_speakers``). Returns a 0-based label per input row, in
    first-appearance order. Pure (numpy + scipy) so it can be tested directly.
    """
    import numpy as np

    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    X = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    dists = pdist(X, metric="cosine")
    Z = linkage(dists, method="average")
    if num_speakers and int(num_speakers) >= 1:
        k = min(int(num_speakers), n)
        labels = fcluster(Z, t=k, criterion="maxclust")
    else:
        labels = fcluster(Z, t=float(threshold), criterion="distance")
        if int(labels.max()) > max_speakers:
            labels = fcluster(Z, t=max_speakers, criterion="maxclust")
        # Fold lone-cue clusters (noise / music outliers) into the nearest real
        # speaker so they don't surface as phantom speakers.
        labels = _absorb_singleton_clusters(X, [int(x) for x in labels], min_cues)
    return _relabel_first_appearance([int(x) for x in labels])


def _build_timeline(spans: List[tuple], labels: List[int],
                    resolution_ms: int = _RESOLUTION_MS) -> Dict[int, str]:
    """Build a ``{time_ms: "SPEAKER_xx"}`` timeline (pyannote-compatible) from
    ``(start_s, end_s)`` spans + their cluster labels."""
    timeline: Dict[int, str] = {}
    for (start_s, end_s), lab in zip(spans, labels):
        start_ms = int(float(start_s) * 1000)
        end_ms = int(float(end_s) * 1000)
        if end_ms <= start_ms:
            end_ms = start_ms + resolution_ms
        sid = f"SPEAKER_{int(lab):02d}"
        for t in range(start_ms, end_ms, resolution_ms):
            timeline[t] = sid
    return timeline


def _coerce_spans(speech_segments, max_no_speech: float = 1.0) -> List[tuple]:
    """Pull ``(start_s, end_s)`` from reframer/transcript segments (dict or
    object), dropping zero/negative-length cues.

    Also drops cues that aren't clean speech so they never pollute speaker
    clustering: known hallucinations, music markers / empty text, and (when
    ``max_no_speech`` < 1.0) cues whose ``no_speech_prob`` exceeds it — the
    sung/instrumental cues that otherwise embed as phantom "speakers". Cues
    missing a given field are kept (backward-compatible)."""
    spans: List[tuple] = []
    for s in (speech_segments or []):
        if isinstance(s, dict):
            a = s.get("start", s.get("start_sec"))
            b = s.get("end", s.get("end_sec"))
            text = s.get("text")
            is_hall = s.get("is_hallucination")
            nsp = s.get("no_speech_prob")
        else:
            a = getattr(s, "start", getattr(s, "start_sec", None))
            b = getattr(s, "end", getattr(s, "end_sec", None))
            text = getattr(s, "text", None)
            is_hall = getattr(s, "is_hallucination", None)
            nsp = getattr(s, "no_speech_prob", None)
        if a is None or b is None:
            continue
        if is_hall:
            continue
        if nsp is not None and float(nsp) > max_no_speech:
            continue  # likely music / singing — no clean speaker identity
        if text is not None:
            t = str(text).strip()
            if not t or _MUSIC_MARK in t:
                continue
        a, b = float(a), float(b)
        if b > a:
            spans.append((a, b))
    return spans


# ── The diarizer ───────────────────────────────────────────────────────────

class LocalEmbeddingDiarizer:
    """SpeechBrain-ECAPA speaker diarizer. No HF token, fully offline after a
    one-time model fetch. Model is cached at class level across jobs."""

    _model = None  # cached EncoderClassifier

    def __init__(self, device: Optional[str] = None, threshold: Optional[float] = None):
        from backend.config import settings as _settings
        # ECAPA is tiny and runs during the perception stage while Whisper holds
        # the GPU — default to CPU to avoid VRAM contention (it's fast enough).
        self.device = (device or getattr(_settings, "LOCAL_DIARIZER_DEVICE", "cpu") or "cpu").lower()
        self.threshold = float(
            threshold if threshold is not None
            else getattr(_settings, "LOCAL_DIARIZER_THRESHOLD", 0.70))
        self.model_id = getattr(
            _settings, "LOCAL_DIARIZER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")

    @staticmethod
    def _has_dependencies() -> bool:
        try:
            import scipy  # noqa: F401
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            import speechbrain  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        from backend.config import settings as _settings
        if not getattr(_settings, "LOCAL_DIARIZATION_ENABLED", True):
            return False
        return self._has_dependencies()

    def _savedir(self) -> str:
        """Persistent dir for the fetched model (shares the NMT models mount)."""
        try:
            from backend.services.nmt_translator import _models_dir
            return os.path.join(_models_dir(), "ecapa")
        except Exception:
            d = os.path.join(tempfile.gettempdir(), "clipai_ecapa")
            os.makedirs(d, exist_ok=True)
            return d

    def _load_model(self):
        if LocalEmbeddingDiarizer._model is not None:
            return LocalEmbeddingDiarizer._model
        # speechbrain renamed pretrained → inference in 1.0; support both.
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            from speechbrain.pretrained import EncoderClassifier  # type: ignore
        savedir = self._savedir()
        os.makedirs(savedir, exist_ok=True)
        # Try the requested device first; on a CUDA failure (OOM / contention /
        # no driver) fall back to CPU. This makes LOCAL_DIARIZER_DEVICE=cuda
        # risk-free: ECAPA is tiny (~80 MB) and the GPU is free once Whisper has
        # released it, so on GPU it embeds the cues in seconds instead of the
        # minutes CPU takes — and if the GPU is unavailable it runs on CPU
        # exactly as before, never losing diarization.
        _devices = [self.device] + (["cpu"] if self.device != "cpu" else [])
        _last_err = None
        for _dev in _devices:
            try:
                logger.info(
                    "Local diarizer: loading SpeechBrain ECAPA (%s) on %s "
                    "(one-time fetch → %s)", self.model_id, _dev, savedir)
                model = EncoderClassifier.from_hparams(
                    source=self.model_id, savedir=savedir,
                    run_opts={"device": _dev},
                )
                self.device = _dev
                LocalEmbeddingDiarizer._model = model
                return model
            except Exception as e:
                _last_err = e
                logger.warning(
                    "Local diarizer: load on %s failed (%s)%s", _dev, e,
                    " — falling back to CPU" if _dev != "cpu" else "")
        raise _last_err

    def _extract_audio(self, media_path: str) -> Optional[str]:
        audio_path = os.path.join(tempfile.gettempdir(), "clipai_diarize_audio.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", media_path, "-vn", "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", audio_path],
                capture_output=True, timeout=180,
            )
        except Exception as e:
            logger.warning("Local diarizer: audio extraction failed (%s)", e)
            return None
        return audio_path if os.path.exists(audio_path) else None

    def diarize(self, media_path: str, speech_segments,
                duration_ms: int = 0, num_speakers: Optional[int] = None) -> Dict[int, str]:
        """Return a ``{time_ms: "SPEAKER_xx"}`` timeline (200 ms bins), or ``{}``
        when diarization can't run (caller falls back to the visual heuristic)."""
        spans = _coerce_spans(speech_segments, max_no_speech=_DIARIZE_MAX_NO_SPEECH)
        if len(spans) < 2:
            # 0/1 speech cue → nothing to separate; let the caller's fallback
            # (or the single-speaker default) handle it.
            return {}
        try:
            import numpy as np
            import torch
            import torchaudio
        except Exception as e:
            logger.info("Local diarizer: deps unavailable (%s)", e)
            return {}

        audio_path = self._extract_audio(media_path)
        if not audio_path:
            return {}

        try:
            model = self._load_model()
        except Exception as e:
            logger.warning("Local diarizer: model load failed (%s) — falling back", e)
            return {}

        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.dim() > 1 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                sr = 16000
            min_samples = int(_MIN_EMBED_SEC * sr)
            total = wav.shape[-1]

            embeddings = []
            kept_spans = []
            for (a, b) in spans:
                s = max(0, int(a * sr))
                e = min(total, int(b * sr))
                if e <= s:
                    continue
                chunk = wav[..., s:e]
                if chunk.shape[-1] < min_samples:
                    pad = min_samples - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad))
                with torch.no_grad():
                    emb = model.encode_batch(chunk)
                embeddings.append(emb.squeeze().detach().cpu().numpy().astype("float32"))
                kept_spans.append((a, b))

            if len(embeddings) < 2:
                return {}

            labels = _cluster_embeddings(
                embeddings, num_speakers=num_speakers, threshold=self.threshold)
            n_spk = len(set(labels))
            timeline = _build_timeline(kept_spans, labels)
            logger.info(
                "Local diarizer: %d speaker(s) over %d speech cues (ECAPA, no HF token)",
                n_spk, len(kept_spans),
            )
            return timeline
        except Exception as e:
            logger.warning("Local diarizer: diarization failed (%s) — falling back", e)
            return {}
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass
