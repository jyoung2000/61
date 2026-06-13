"""Pure presentation helpers shared by the editor, inspector and timeline.

No Qt imports here so these are unit-testable: they turn an :class:`Inflection`
into the strings/colors the UI shows (tooltips, underline colors, labels).
"""

from __future__ import annotations

from ..config import SPAN_COLORS
from ..document.spans import EMOTIONS, NUM_SPAN_COLORS, Inflection

# A non-emotional span (emo_text / speed only) gets a neutral underline color.
NEUTRAL_COLOR_IDX = 3  # blue in the default palette

ENGINE_LABELS = {
    None: "Default",
    "chatterbox": "Chatterbox",
    "indextts2": "IndexTTS-2",
    "fish": "Fish",
    "hybrid": "Hybrid",
}


def dominant_emotion(vector: list[float] | None) -> tuple[str, float] | None:
    """Return ``(emotion_name, value)`` for the strongest active dim, or None."""
    if not vector or not any(v > 1e-6 for v in vector):
        return None
    idx = max(range(len(vector)), key=lambda i: vector[i])
    if vector[idx] <= 1e-6:
        return None
    return EMOTIONS[idx], float(vector[idx])


def active_emotions(vector: list[float] | None, threshold: float = 0.05) -> list[tuple[str, float]]:
    """All emotion dims above ``threshold``, strongest first."""
    if not vector:
        return []
    pairs = [(EMOTIONS[i], float(v)) for i, v in enumerate(vector) if v > threshold]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs


def color_for_inflection(inflection: Inflection) -> int:
    """Underline color index: follow the dominant emotion, else neutral."""
    dom = dominant_emotion(inflection.emotion_vector)
    if dom is not None:
        return EMOTIONS.index(dom[0]) % NUM_SPAN_COLORS
    return NEUTRAL_COLOR_IDX


def color_hex(color_idx: int) -> str:
    return SPAN_COLORS[color_idx % len(SPAN_COLORS)]


def summarize_inflection(inflection: Inflection, *, max_emotions: int = 3) -> str:
    """One-line summary, e.g. ``"angry 0.7 · surprised 0.3 · 1.1×"``."""
    parts: list[str] = []
    for name, value in active_emotions(inflection.emotion_vector)[:max_emotions]:
        parts.append(f"{name} {value:.1f}")
    if inflection.emo_text:
        text = inflection.emo_text.strip()
        if len(text) > 28:
            text = text[:27] + "…"
        parts.append(f'“{text}”')
    if inflection.emo_audio:
        parts.append("emo-ref")
    if abs(inflection.speed - 1.0) > 1e-3:
        parts.append(f"{inflection.speed:.2g}×")
    if inflection.pause_after_ms > 0:
        parts.append(f"⏸ {inflection.pause_after_ms}ms")
    if inflection.engine:
        parts.append(ENGINE_LABELS.get(inflection.engine, inflection.engine))
    return " · ".join(parts) if parts else "default delivery"
