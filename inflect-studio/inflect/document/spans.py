"""Core document data model for Inflect Studio.

This module is intentionally free of any GUI / engine / torch dependencies so it
can be unit-tested in isolation and reused from the worker thread.

The central idea: a :class:`Document` holds plain ``text`` plus a list of
non-overlapping :class:`InflectionSpan` ranges that describe *how* a portion of
that text should be spoken. Text not covered by any span is spoken using the
document's :attr:`Document.default_inflection`.

Span algebra (apply / clear / remap) lives here and is the most heavily tested
part of the code base -- it behaves like rich-text character formatting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

# IndexTTS-2 emotion dimension order. Do not reorder -- the index of each label
# is meaningful and is mapped directly onto the engine's 8-dim emotion vector.
EMOTIONS: list[str] = [
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
]

# Number of colors used to underline spans in the editor. The model only stores
# an index; the actual RGB palette lives in the UI layer / config.
NUM_SPAN_COLORS = 8


@dataclass
class Inflection:
    """How a span (or the document default) should be delivered.

    All fields are optional knobs; ``None``/defaults mean "neutral / unset" and
    every engine adapter is responsible for translating the fields it supports.
    """

    emotion_vector: list[float] | None = None  # len 8, each 0.0-1.0, or None
    emo_text: str | None = None  # e.g. "whispering, hesitant" -> IndexTTS-2 T2E
    emo_audio: str | None = None  # path to an emotion reference wav (optional)
    emo_alpha: float = 0.8  # blend strength of emotion vs neutral
    speed: float = 1.0  # 0.5-1.5, duration control / post time-stretch
    pause_after_ms: int = 0
    # Phase 6: per-span engine override. None == use the document/toolbar engine.
    # One of: None, "chatterbox", "indextts2", "fish", "hybrid".
    engine: str | None = None

    def __post_init__(self) -> None:
        if self.emotion_vector is not None:
            if len(self.emotion_vector) != len(EMOTIONS):
                raise ValueError(
                    f"emotion_vector must have {len(EMOTIONS)} elements, "
                    f"got {len(self.emotion_vector)}"
                )
            # Clamp into range rather than rejecting -- sliders can only ever
            # produce 0..1 but loaded projects might be slightly out of range.
            self.emotion_vector = [float(min(1.0, max(0.0, v))) for v in self.emotion_vector]
        self.emo_alpha = float(min(1.0, max(0.0, self.emo_alpha)))
        self.speed = float(min(1.5, max(0.5, self.speed)))
        self.pause_after_ms = int(max(0, self.pause_after_ms))

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Inflection":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def is_neutral(self) -> bool:
        """True when this inflection carries no audible styling.

        Used to decide whether a span actually differs from the document
        default (purely cosmetic spans can be dropped).
        """
        if self.emotion_vector and any(v > 1e-6 for v in self.emotion_vector):
            return False
        if self.emo_text:
            return False
        if self.emo_audio:
            return False
        if abs(self.speed - 1.0) > 1e-6:
            return False
        return True

    def audio_signature(self) -> str:
        """Deterministic JSON of only the fields that change the rendered wav.

        Notably this EXCLUDES :attr:`pause_after_ms` (silence is inserted at
        assembly time, it does not change a segment's audio) and ``engine``
        (the engine is folded into the segment hash separately so the same
        inflection signature can be compared across engines).
        """
        payload = {
            "emotion_vector": self.emotion_vector,
            "emo_text": self.emo_text,
            "emo_audio": self.emo_audio,
            "emo_alpha": round(self.emo_alpha, 6),
            "speed": round(self.speed, 6),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


@dataclass
class InflectionSpan:
    """A half-open ``[start, end)`` character range with an attached inflection."""

    start: int
    end: int
    inflection: Inflection
    color_idx: int = 0  # for underline coloring in the editor

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("span offsets must be non-negative")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} < start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, pos: int) -> bool:
        """Caret/position membership: a caret sitting just past ``end`` is not in."""
        return self.start <= pos < self.end

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "color_idx": self.color_idx,
            "inflection": self.inflection.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InflectionSpan":
        return cls(
            start=int(data["start"]),
            end=int(data["end"]),
            color_idx=int(data.get("color_idx", 0)),
            inflection=Inflection.from_dict(data.get("inflection", {})),
        )


@dataclass
class Document:
    """Plain text plus the spans that style it and a default delivery."""

    text: str = ""
    spans: list[InflectionSpan] = field(default_factory=list)
    default_inflection: Inflection = field(default_factory=Inflection)
    voice_profile_id: str | None = None

    # -- queries -----------------------------------------------------------
    def span_at(self, pos: int) -> InflectionSpan | None:
        """Return the span containing ``pos`` (caret semantics), or ``None``."""
        for span in self.spans:
            if span.contains(pos):
                return span
        return None

    def sorted_spans(self) -> list[InflectionSpan]:
        return sorted(self.spans, key=lambda s: s.start)

    # -- mutations ---------------------------------------------------------
    def apply_inflection(
        self,
        start: int,
        end: int,
        inflection: Inflection,
        color_idx: int = 0,
    ) -> InflectionSpan | None:
        """Style ``[start, end)``, splitting/truncating any intersecting spans.

        Returns the newly created span, or ``None`` if the selection was empty.
        Mirrors how a rich-text editor replaces character formatting.
        """
        start, end = _normalize_range(start, end, len(self.text))
        if start >= end:
            return None
        self.spans = _carve_out(self.spans, start, end)
        new_span = InflectionSpan(start, end, inflection, color_idx)
        self.spans.append(new_span)
        self.spans.sort(key=lambda s: s.start)
        return new_span

    def clear_inflection(self, start: int, end: int) -> None:
        """Remove styling from ``[start, end)`` (truncates/splits spans)."""
        start, end = _normalize_range(start, end, len(self.text))
        if start >= end:
            return
        self.spans = _carve_out(self.spans, start, end)
        self.spans.sort(key=lambda s: s.start)

    def set_pause_after(self, pos: int, pause_after_ms: int) -> InflectionSpan:
        """Convenience: ensure a (possibly 1-char) span carries a trailing pause.

        Inserts a pause "marker" by styling the character just before ``pos``.
        Used by the "Insert pause" editor action.
        """
        anchor = max(0, min(pos, len(self.text)))
        start = max(0, anchor - 1)
        end = anchor
        if start >= end:  # empty document edge case
            start, end = 0, min(1, len(self.text))
        existing = self.span_at(start)
        base = replace(existing.inflection) if existing else replace(self.default_inflection)
        base.pause_after_ms = int(max(0, pause_after_ms))
        span = self.apply_inflection(start, end, base, existing.color_idx if existing else 0)
        assert span is not None
        return span

    def remap_for_edit(self, position: int, chars_removed: int, chars_added: int) -> None:
        """Shift span offsets after a text edit.

        Mirrors Qt's ``QTextDocument.contentsChange(position, removed, added)``.
        Spans that collapse to zero length are dropped.
        """
        new_text_len = len(self.text)
        survivors: list[InflectionSpan] = []
        for span in self.spans:
            new_start = _remap_point(
                span.start, position, chars_removed, chars_added, is_start=True
            )
            new_end = _remap_point(
                span.end, position, chars_removed, chars_added, is_start=False
            )
            new_start = max(0, min(new_start, new_text_len))
            new_end = max(0, min(new_end, new_text_len))
            if new_end > new_start:
                survivors.append(replace(span, start=new_start, end=new_end))
        survivors.sort(key=lambda s: s.start)
        self.spans = survivors

    def inflection_for(self, pos: int) -> Inflection:
        """The effective inflection at ``pos`` (span override or default)."""
        span = self.span_at(pos)
        return span.inflection if span else self.default_inflection

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "voice_profile_id": self.voice_profile_id,
            "default_inflection": self.default_inflection.to_dict(),
            "spans": [s.to_dict() for s in self.sorted_spans()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            text=data.get("text", ""),
            voice_profile_id=data.get("voice_profile_id"),
            default_inflection=Inflection.from_dict(data.get("default_inflection", {})),
            spans=[InflectionSpan.from_dict(s) for s in data.get("spans", [])],
        )


# ---------------------------------------------------------------------------
# Free functions implementing the span algebra. Kept module-level (and pure) so
# they are trivially unit-testable without constructing a Document.
# ---------------------------------------------------------------------------
def _normalize_range(start: int, end: int, text_len: int) -> tuple[int, int]:
    if start > end:
        start, end = end, start
    start = max(0, min(start, text_len))
    end = max(0, min(end, text_len))
    return start, end


def _carve_out(
    spans: list[InflectionSpan], start: int, end: int
) -> list[InflectionSpan]:
    """Return ``spans`` with the range ``[start, end)`` cleared.

    Handles every intersection case: disjoint (kept), engulfed (dropped),
    overlap-left (truncated to ``[a, start)``), overlap-right (truncated to
    ``[end, b)``), and engulfing (split into ``[a, start)`` + ``[end, b)``).
    """
    result: list[InflectionSpan] = []
    for span in spans:
        a, b = span.start, span.end
        # Disjoint -- entirely before or after the carve range.
        if b <= start or a >= end:
            result.append(span)
            continue
        # Left remainder survives if the span starts before the carve range.
        if a < start:
            result.append(replace(span, start=a, end=start))
        # Right remainder survives if the span ends after the carve range.
        if b > end:
            result.append(replace(span, start=end, end=b))
        # Anything fully inside [start, end) is dropped (no remainder appended).
    result.sort(key=lambda s: s.start)
    return result


def _remap_point(
    p: int, position: int, removed: int, added: int, *, is_start: bool
) -> int:
    """Remap a single boundary point through a ``replace`` edit.

    The edit deletes ``[position, position+removed)`` then inserts ``added``
    characters at ``position``. ``is_start`` selects the inclusive/exclusive
    convention so that text typed exactly at a span boundary lands *outside*
    the span (typing right before or right after a styled run is unstyled),
    while typing strictly inside a span extends it.
    """
    end_removed = position + removed
    # 1) Apply the deletion.
    if p >= end_removed:
        p -= removed
    elif p > position:  # point fell inside the deleted region -> clamp to start
        p = position
    # (p <= position: untouched by the deletion)

    # 2) Apply the insertion of `added` chars at `position`.
    if added:
        if is_start:
            if p >= position:
                p += added
        else:
            if p > position:
                p += added
    return p
