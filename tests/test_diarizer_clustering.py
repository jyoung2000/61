"""Speaker-clustering fix: the 128-min run exported ~95% of 983 cues as
'Speaker 1' despite three real voices. Root cause: ECAPA embeddings of short
cues over shared BGM all carry a common recording-channel component, and the
fixed 0.70 cosine-distance cut merged everything it dominated. The fix is
per-recording mean-centering + silhouette-selected k, with a floor that keeps
genuine single-speaker content at one speaker, plus a margin-guarded pass
that smooths single-cue label flips without inventing speakers."""

import sys
import types

import numpy as np
import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.local_diarizer import (
    _cluster_embeddings, _mean_silhouette, _smooth_label_flips)


def _synth(n_per, speaker_dirs, bgm_strength=3.0, noise=0.25, seed=7):
    """Cues = strong shared BGM direction + weaker per-speaker direction."""
    rng = np.random.default_rng(seed)
    dim = 192
    bgm = rng.normal(size=dim)
    bgm /= np.linalg.norm(bgm)
    out, truth = [], []
    for si, _ in enumerate(speaker_dirs):
        d = speaker_dirs[si]
        for _ in range(n_per):
            v = bgm_strength * bgm + d + noise * rng.normal(size=dim)
            out.append(v.astype("float32"))
            truth.append(si)
    return out, truth


def _dirs(k, seed=3):
    rng = np.random.default_rng(seed)
    dirs = []
    for _ in range(k):
        d = rng.normal(size=192)
        dirs.append(d / np.linalg.norm(d))
    return dirs


def _purity(labels, truth):
    from collections import Counter
    total = 0
    for lab in set(labels):
        members = [t for l, t in zip(labels, truth) if l == lab]
        total += Counter(members).most_common(1)[0][1]
    return total / len(truth)


def test_bgm_dominated_multispeaker_separates():
    # The measured failure regime: shared BGM component ~3x the speaker
    # signal (calibrated: real ECAPA same-speaker cues cluster around
    # noise<=0.12 in this fixture; silhouette at true k = 0.25-0.67 vs a
    # single-speaker regime's ~0.03).
    emb, truth = _synth(40, _dirs(3), noise=0.1)
    labels = _cluster_embeddings(emb)
    n_spk = len(set(labels))
    assert 2 <= n_spk <= 4, f"expected ~3 speakers, got {n_spk}"
    assert _purity(labels, truth) >= 0.9


def test_inseparable_noise_gives_honest_single_speaker():
    # Beyond separability (noise swamps the voice signal) the silhouette
    # floor keeps the honest answer: one speaker — never an arbitrary split.
    emb, _ = _synth(40, _dirs(3), noise=0.3)
    assert len(set(_cluster_embeddings(emb))) == 1


def test_single_speaker_stays_single():
    # One voice + BGM must NOT be split into phantom speakers.
    emb, _ = _synth(60, _dirs(1))
    labels = _cluster_embeddings(emb)
    assert len(set(labels)) == 1


def test_two_speakers_alternating():
    emb, truth = _synth(30, _dirs(2, seed=11), noise=0.1, seed=13)
    labels = _cluster_embeddings(emb)
    assert len(set(labels)) == 2
    assert _purity(labels, truth) >= 0.9


def test_explicit_num_speakers_still_honored():
    emb, truth = _synth(25, _dirs(3, seed=5), noise=0.1, seed=9)
    labels = _cluster_embeddings(emb, num_speakers=3)
    assert len(set(labels)) == 3


def test_smooth_relabels_weak_singleton_flip():
    # A one-cue flip whose embedding is basically the neighbors' voice.
    d = _dirs(2, seed=21)
    X = np.stack([d[0], d[0], d[0] + 0.01, d[0], d[1], d[1], d[1]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = [0, 0, 1, 0, 1, 1, 1]          # index 2 is a weak flip
    out = _smooth_label_flips(X, labels)
    assert out[2] == 0                        # absorbed into the run
    assert out[4:] == [1, 1, 1]               # real speaker B untouched


def test_smooth_keeps_confident_interjection():
    d = _dirs(2, seed=31)
    X = np.stack([d[0], d[0], d[1], d[0], d[0], d[1], d[1]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = [0, 0, 1, 0, 0, 1, 1]           # index 2 is a REAL interjection
    out = _smooth_label_flips(X, labels)
    assert out[2] == 1                        # strong own-cluster affinity kept


def test_silhouette_sane():
    d = _dirs(2, seed=41)
    X = np.stack([d[0]] * 5 + [d[1]] * 5)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X, metric="cosine"))
    good = _mean_silhouette(D, [0] * 5 + [1] * 5)
    bad = _mean_silhouette(D, [0, 1] * 5)
    assert good > 0.8
    assert bad < good
