"""Turn a :class:`Document` into an ordered list of :class:`SegmentJob`.

Rules (see spec §4):

* Cut at every span boundary so each segment has a single, uniform inflection.
* Within any run longer than ~400 chars, also cut at sentence boundaries so the
  TTS engine receives manageable chunks. Over-long single sentences fall back to
  clause boundaries and finally a hard character wrap.
* Each job carries a stable content ``hash`` used by the pipeline to cache the
  rendered wav -- editing one highlighted phrase only re-renders its segment.

The hash deliberately covers only what changes a segment's *audio*: the text,
the inflection's audio signature (emotion/speed, **not** pause), the voice
profile, the engine name and the audio-affecting engine params. ``pause_after_ms``
is inserted as silence at assembly time and is tracked on the job separately so
that nudging a pause never invalidates a cached render.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace

from .spans import Document, Inflection, InflectionSpan

# Target maximum characters per segment before we start splitting at sentences.
MAX_SEG_CHARS = 400
# Absolute ceiling; a single chunk is hard-wrapped if it still exceeds this.
HARD_MAX_CHARS = 600

# Sentence terminator: one or more of . ? ! … optionally followed by closing
# quotes/brackets, then whitespace or end-of-string. We split *after* the
# terminator (+ trailing quote) so punctuation stays attached to its sentence.
_SENTENCE_RE = re.compile(r"[.?!…]+[\"'”’\)\]]*(?:\s+|$)")
# Clause boundaries used only when a single sentence is itself too long.
_CLAUSE_RE = re.compile(r"[,;:—][\"'”’\)\]]*\s+")


@dataclass
class SegmentJob:
    """A single unit of synthesis work."""

    seg_id: int
    text: str
    inflection: Inflection
    voice_profile_id: str | None
    engine: str
    engine_params: dict = field(default_factory=dict)
    # Character span in the source document (for timeline highlight / re-render).
    char_start: int = 0
    char_end: int = 0
    # Trailing silence to insert after this segment at assembly time. Computed by
    # the segmenter, NOT part of the audio hash.
    pause_after_ms: int = 0

    @property
    def hash(self) -> str:
        return segment_hash(
            self.text,
            self.inflection,
            self.voice_profile_id,
            self.engine,
            self.engine_params,
        )

    @property
    def is_blank(self) -> bool:
        return not self.text.strip()


def segment_hash(
    text: str,
    inflection: Inflection,
    voice_profile_id: str | None,
    engine: str,
    engine_params: dict | None = None,
) -> str:
    """Stable SHA-1 over everything that affects a segment's rendered audio."""
    params_json = json.dumps(engine_params or {}, sort_keys=True, ensure_ascii=False)
    payload = "\n".join(
        [
            text,
            inflection.audio_signature(),
            voice_profile_id or "",
            engine,
            params_json,
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def iter_regions(document: Document) -> list[tuple[int, int, Inflection]]:
    """Cover the whole text with ``(start, end, inflection)`` regions.

    Spans contribute their own inflection; the gaps between them are filled with
    the document's default inflection. Regions are returned in document order and
    are contiguous and non-overlapping.
    """
    text_len = len(document.text)
    spans = [s for s in document.sorted_spans() if s.start < s.end and s.start < text_len]
    regions: list[tuple[int, int, Inflection]] = []
    cursor = 0
    for span in spans:
        s = max(0, min(span.start, text_len))
        e = max(0, min(span.end, text_len))
        if s > cursor:
            regions.append((cursor, s, document.default_inflection))
        if e > s:
            regions.append((s, e, span.inflection))
        cursor = max(cursor, e)
    if cursor < text_len:
        regions.append((cursor, text_len, document.default_inflection))
    if not regions and text_len > 0:
        regions.append((0, text_len, document.default_inflection))
    return regions


def _split_long(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into chunks <= MAX_SEG_CHARS, returning local offsets.

    Strategy: group whole sentences greedily; if one sentence is still too long
    split it at clause boundaries; if a clause is *still* too long, hard-wrap at
    whitespace nearest HARD_MAX_CHARS.
    """
    if len(text) <= MAX_SEG_CHARS:
        return [(0, len(text))]

    # 1) Find sentence pieces.
    sentences = _split_by_regex(text, _SENTENCE_RE)

    # 2) Greedily pack sentences into <= MAX_SEG_CHARS chunks.
    chunks: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for s, e in sentences:
        piece_len = e - s
        if piece_len > MAX_SEG_CHARS:
            # Flush the accumulator first.
            if cur_start is not None:
                chunks.append((cur_start, cur_end))
                cur_start = None
            # Break this oversized sentence down further.
            chunks.extend((s + a, s + b) for a, b in _split_clauses(text[s:e]))
            continue
        if cur_start is None:
            cur_start, cur_end = s, e
        elif (e - cur_start) <= MAX_SEG_CHARS:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def _split_clauses(text: str) -> list[tuple[int, int]]:
    pieces = _split_by_regex(text, _CLAUSE_RE)
    out: list[tuple[int, int]] = []
    for s, e in pieces:
        if e - s <= HARD_MAX_CHARS:
            out.append((s, e))
        else:
            out.extend((s + a, s + b) for a, b in _hard_wrap(text[s:e]))
    return out


def _hard_wrap(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + HARD_MAX_CHARS, n)
        if end < n:
            # Back off to the last whitespace to avoid splitting a word.
            ws = text.rfind(" ", start + 1, end)
            if ws > start:
                end = ws + 1
        out.append((start, end))
        start = end
    return out


def _split_by_regex(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Split into pieces *after* each match, returning (start, end) offsets.

    Trailing delimiter (and its following whitespace) stays with the piece. A
    final piece without a terminator is still returned.
    """
    pieces: list[tuple[int, int]] = []
    last = 0
    for m in pattern.finditer(text):
        end = m.end()
        if end > last:
            pieces.append((last, end))
            last = end
    if last < len(text):
        pieces.append((last, len(text)))
    return pieces or [(0, len(text))]


def segment_document(
    document: Document,
    engine: str,
    engine_params: dict | None = None,
    default_engine: str | None = None,
) -> list[SegmentJob]:
    """Produce the ordered list of synthesis jobs for ``document``.

    ``engine`` is the active toolbar/document engine; a span whose inflection
    sets its own ``engine`` overrides it (Phase 6 per-span engines). Blank
    (whitespace-only) segments are skipped, but a pause attached to a blank
    region is folded onto the previous real segment.
    """
    engine_params = engine_params or {}
    jobs: list[SegmentJob] = []
    seg_id = 0
    last_real: SegmentJob | None = None

    for r_start, r_end, inflection in iter_regions(document):
        region_text = document.text[r_start:r_end]
        region_pause = inflection.pause_after_ms
        seg_engine = inflection.engine or engine
        chunks = _split_long(region_text)

        # Collect non-blank chunks for this region first so we know which is last.
        real_chunks = [(a, b) for a, b in chunks if region_text[a:b].strip()]

        if not real_chunks:
            # Whole region is whitespace -- push any pause onto the previous job.
            if region_pause and last_real is not None:
                last_real.pause_after_ms += region_pause
            continue

        for idx, (a, b) in enumerate(real_chunks):
            chunk_text = region_text[a:b]
            job = SegmentJob(
                seg_id=seg_id,
                text=chunk_text,
                inflection=inflection,
                voice_profile_id=document.voice_profile_id,
                engine=seg_engine,
                engine_params=dict(engine_params),
                char_start=r_start + a,
                char_end=r_start + b,
                # Only the final sub-chunk of a region carries the region pause.
                pause_after_ms=region_pause if idx == len(real_chunks) - 1 else 0,
            )
            jobs.append(job)
            last_real = job
            seg_id += 1

    return jobs


def jobs_signature(jobs: list[SegmentJob]) -> str:
    """Hash of the whole job list -- lets the pipeline detect "nothing changed"."""
    h = hashlib.sha1()
    for job in jobs:
        h.update(job.hash.encode("ascii"))
        h.update(str(job.pause_after_ms).encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()
