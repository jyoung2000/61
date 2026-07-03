"""Transcript union-write guard + progress-label fixes.

Real-world failure this codifies (job 77198bd0, 2026-07-03 run): the pipeline
persisted a clean 406-cue translated transcript; afterwards the stored track
became 625 cues with 117 texts repeated ~3x each at timestamps minutes apart
(one phantom family at a constant ~+19min offset, another splayed further by
timeline overlap resolution). Diff review pinned the writer: the NLE
reverse-sync PUT — a timeline holding stacked stale generations of the same
subtitle cues replaced the canonical transcript wholesale. Existing dedup
passes could not catch it (drop_scattered_duplicates fires at 4+ occurrences,
collapse_repeated_runs needs CONSECUTIVE runs; the phantoms were 3x and
interleaved).

Also covers the progress-label fixes for the same run's UX report: heartbeat
saying "scene analysis" during FACES/CONVERT, an 8-minute silent diarization
gap, Whisper appearing frozen at 79% during redecode/alignment, and percent
regressions from interleaved callbacks.
"""

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _name, _attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                     ("anthropic", "AsyncAnthropic")):
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, object)
        sys.modules[_name] = _mod
if "google.generativeai" not in sys.modules:
    _g = types.ModuleType("google")
    _gg = types.ModuleType("google.generativeai")
    _gg.configure = lambda *a, **k: None
    _gg.GenerativeModel = object
    _g.generativeai = _gg
    sys.modules.setdefault("google", _g)
    sys.modules["google.generativeai"] = _gg

import pytest

from backend.services.transcript_sync import (
    clean_and_sort_segments,
    detect_union_write,
    _duplicate_text_share,
)


def _row(start, end, text, speaker="Speaker 1"):
    return {"start": float(start), "end": float(end), "text": text,
            "speaker": speaker}


def _unique_rows(n, t0=0.0, prefix="line"):
    return [_row(t0 + i * 3, t0 + i * 3 + 2.5,
                 f"{prefix} number {i} spoken with plenty of words")
            for i in range(n)]


# ── clean_and_sort_segments dedup ────────────────────────────────────────

def test_clean_and_sort_drops_same_position_duplicates():
    rows = _unique_rows(5)
    rows.append(dict(rows[2]))  # verbatim duplicate at the same start
    out = clean_and_sort_segments(rows)
    assert len(out) == 5


def test_clean_and_sort_keeps_repeated_dialogue_at_different_times():
    # Real dialogue legitimately repeats lines — dedup must be positional.
    rows = [_row(10, 12, "It feels great, doesn't it?"),
            _row(300, 302, "It feels great, doesn't it?")]
    out = clean_and_sort_segments(rows)
    assert len(out) == 2


# ── detect_union_write ───────────────────────────────────────────────────

def test_union_write_flags_stale_timeline_union():
    """The reproduced corruption shape: stored clean track + phantom copies
    of a contiguous block at two shifted time offsets."""
    stored = _unique_rows(100)
    block = stored[40:60]
    phantoms = []
    for offset in (533.0, 1162.0):  # the observed splay + constant offsets
        for r in block:
            phantoms.append(_row(r["start"] + offset, r["end"] + offset,
                                 r["text"]))
    incoming = sorted(stored + phantoms, key=lambda r: r["start"])
    is_union, reason = detect_union_write(incoming, stored)
    assert is_union, reason
    assert "union" in reason


def test_union_write_allows_first_write():
    assert detect_union_write(_unique_rows(50), []) == (False, "")


def test_union_write_allows_genuine_split_edit():
    # Splitting a handful of long cues adds a few rows of NEW text.
    stored = _unique_rows(100)
    incoming = list(stored)
    for i in range(5):
        r = stored[i]
        incoming.append(_row(r["end"], r["end"] + 1.5,
                             f"a brand new second half {i}"))
    assert detect_union_write(incoming, stored)[0] is False


def test_union_write_allows_large_unique_import():
    # Importing a denser SRT: many MORE cues but no duplicate-text growth.
    stored = _unique_rows(100)
    incoming = _unique_rows(160, prefix="fresh")
    assert detect_union_write(incoming, stored)[0] is False


def test_duplicate_text_share_ignores_short_interjections():
    rows = [_row(i, i + 1, "Yeah.") for i in range(10)]
    assert _duplicate_text_share(rows) == 0.0


# ── endpoint wiring ──────────────────────────────────────────────────────

def test_replace_transcript_endpoint_uses_guard():
    import inspect
    from backend.routers import jobs as jobs_router
    src = inspect.getsource(jobs_router.replace_transcript)
    assert "detect_union_write" in src
    assert "409" in src


# ── frontend mirrors (source-level contract) ─────────────────────────────

def test_frontend_reverse_sync_has_union_guard():
    with open("frontend/src/pages/Analysis.jsx", encoding="utf-8") as fh:
        src = fh.read()
    assert "dupShare" in src, "reverse-sync must mirror detect_union_write"


def test_store_backfill_is_idempotent():
    with open("frontend/src/stores/timelineStore.js", encoding="utf-8") as fh:
        src = fh.read()
    i = src.index("addSubtitlesFromTranscript: (")
    body = src[i:i + 1600]
    assert "filter(it => it.type !== 'subtitle')" in body, \
        "backfill must replace, not stack, subtitle generations"


def test_video_editor_detects_duplicate_indices():
    with open("frontend/src/components/VideoEditor.jsx", encoding="utf-8") as fh:
        src = fh.read()
    assert "idxDupes" in src
    assert "unionSuspect" in src


# ── progress-label fixes ─────────────────────────────────────────────────

def test_engine_progress_handles_phase_hints_and_is_monotonic():
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    assert "transcript_refine" in src
    assert "'diarization'" in src
    assert "_last_emitted_pct" in src, "percent must be forward-only"


def test_perceiver_emits_diarization_hint():
    with open("backend/services/reframer_perceiver.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "on_progress(1.0, 'diarization')" in src


def test_audio_emits_refinement_hint():
    with open("backend/services/reframer_audio.py", encoding="utf-8") as fh:
        src = fh.read()
    assert "on_progress(1.0, 'transcript_refine')" in src
