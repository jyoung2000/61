"""Unit tests for the local (no-HF-token) speaker diarizer's pure logic.

The SpeechBrain ECAPA model + audio I/O need the ML stack, but the clustering,
timeline assembly, and span coercion are pure numpy/scipy and tested here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# config resolves its data dir from HOME at import time.
os.environ.setdefault("HOME", tempfile.mkdtemp(prefix="clipai_diar_test_"))

ld = pytest.importorskip("backend.services.local_diarizer")
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")


def test_relabel_first_appearance():
    assert ld._relabel_first_appearance([5, 5, 2, 2, 5]) == [0, 0, 1, 1, 0]
    assert ld._relabel_first_appearance([]) == []
    assert ld._relabel_first_appearance([9]) == [0]


def test_cluster_embeddings_separates_two_speakers():
    # Two tight, well-separated clusters in embedding space.
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    embs = [a + 0.01, a - 0.01, b + 0.01, b - 0.01, a]
    labels = ld._cluster_embeddings(embs, threshold=0.5)
    # 5 cues → 2 speakers; the two 'a' rows share a label, the 'b' rows share another.
    assert len(labels) == 5
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[4]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_embeddings_respects_num_speakers_hint():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    embs = [a, a, b, b]
    # Force a single speaker even though there are two natural groups.
    assert len(set(ld._cluster_embeddings(embs, num_speakers=1))) == 1
    assert len(set(ld._cluster_embeddings(embs, num_speakers=2))) == 2


def test_cluster_embeddings_edge_cases():
    assert ld._cluster_embeddings([]) == []
    assert ld._cluster_embeddings([np.array([1.0, 2.0])]) == [0]


def test_build_timeline_pyannote_shape():
    spans = [(0.0, 0.4), (0.4, 0.6)]
    labels = [0, 1]
    tl = ld._build_timeline(spans, labels, resolution_ms=200)
    # 200 ms bins, "SPEAKER_xx" labels (matches pyannote output).
    assert tl[0] == "SPEAKER_00"
    assert tl[200] == "SPEAKER_00"
    assert tl[400] == "SPEAKER_01"
    assert all(v.startswith("SPEAKER_") for v in tl.values())


def test_coerce_spans_handles_dicts_objects_and_drops_invalid():
    from types import SimpleNamespace
    segs = [
        {"start": 0.0, "end": 1.0},
        {"start_sec": 1.0, "end_sec": 2.0},
        SimpleNamespace(start=2.0, end=3.0),
        {"start": 5.0, "end": 5.0},   # zero-length → dropped
        {"start": 9.0},               # missing end → dropped
    ]
    spans = ld._coerce_spans(segs)
    assert spans == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


def test_coerce_spans_drops_non_speech_cues():
    # Sung / music / hallucination / empty cues must not reach the clusterer,
    # or each becomes a phantom "speaker". Cues missing a field are kept.
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "no_speech_prob": 0.05},
        {"start": 1.0, "end": 2.0, "text": "la la la", "no_speech_prob": 0.85},   # sung → drop
        {"start": 2.0, "end": 3.0, "text": "[♪ music ♪]", "no_speech_prob": 0.1},  # marker → drop
        {"start": 3.0, "end": 4.0, "text": "   ", "no_speech_prob": 0.0},          # empty → drop
        {"start": 4.0, "end": 5.0, "text": "World", "is_hallucination": True},     # hallu → drop
        {"start": 5.0, "end": 6.0, "text": "Keep me", "no_speech_prob": 0.2},
    ]
    spans = ld._coerce_spans(segs, max_no_speech=0.6)
    assert spans == [(0.0, 1.0), (5.0, 6.0)]
    # Default (1.0) keeps everything no_speech-wise (back-compat), but still
    # drops the unambiguous non-speech (marker / empty / hallucination).
    assert ld._coerce_spans(segs) == [(0.0, 1.0), (1.0, 2.0), (5.0, 6.0)]


def test_cluster_absorbs_singleton_outlier():
    # 3 'a' cues + 3 'b' cues + 1 off-axis singleton nearer 'a'. The singleton
    # is a noise/music outlier, not a 9th-speaker — it must fold into 'a'.
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.3, 0.2, 0.93])   # its own cluster at t=0.4, closer to a than b
    embs = [a, a, a, b, b, b, c]
    labels = ld._cluster_embeddings(embs, threshold=0.4)
    assert len(labels) == 7
    assert len(set(labels)) == 2          # singleton absorbed, not a phantom speaker
    assert labels[6] == labels[0]         # c folded into the 'a' cluster (nearest)
    assert labels[0] != labels[3]         # a and b stay distinct
    # An explicit speaker-count hint bypasses absorption (honor the request).
    assert len(set(ld._cluster_embeddings(embs, num_speakers=3))) == 3
