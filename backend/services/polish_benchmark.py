"""Built-in subtitle-polish benchmark (audit Phase 4.2).

~20 canned Whisper-medium-style error segments with gold corrections.
``run_benchmark(model_id)`` pushes them through ``transcript_polisher``'s
own prompt (via ``correct_transcript`` with ``model_override``) and
scores:

  * exact_fix_rate    — fraction of cases whose text exactly matches the
                        gold correction after normalization
  * format_compliance — did the model return the right segment count and
                        keep every segment within the ±15% word budget?

The score is persisted by the settings router so the recommendation
endpoint can prefer measured winners over the static shortlist order.
"""
from __future__ import annotations

import re

from backend.models import TranscriptSegment

# (input_text, gold_text) — classic small-model failure modes: phonetic
# mis-hears, proper-noun spellings, homophones, dropped punctuation.
# Names/terms recur across cases so context-based correction is testable.
BENCH_CASES: list[tuple[str, str]] = [
    ("Aiken, let's get started with the demo.",
     "Okay, let's get started with the demo."),
    ("The areas constellation is visible in spring.",
     "The Aries constellation is visible in spring."),
    ("Their going to launch the update next week.",
     "They're going to launch the update next week."),
    ("You're code needs a review before merge.",
     "Your code needs a review before merge."),
    ("We shipped the new cash layer for the database.",
     "We shipped the new cache layer for the database."),
    ("Its important to profile before optimizing.",
     "It's important to profile before optimizing."),
    ("The team met with Dorian from the peace delegation.",
     "The team met with Darlian from the peace delegation."),
    ("Sex piloted the white mobile suit.",
     "Zechs piloted the white mobile suit."),
    ("Aaron was chosen as the pilot of Wing Gundam.",
     "Heero was chosen as the pilot of Wing Gundam."),
    ("Lets sink our watches before the race",
     "Let's sync our watches before the race."),
    ("The colonel gave the order to advance",
     "The colonel gave the order to advance."),
    ("I herd the results were announced yesterday.",
     "I heard the results were announced yesterday."),
    ("The customer once a refund for the broken unit.",
     "The customer wants a refund for the broken unit."),
    ("Please pole the sensor every five seconds.",
     "Please poll the sensor every five seconds."),
    ("The docker container mounts the volume at slash data.",
     "The Docker container mounts the volume at /data."),
    ("Who's turn is it to present the road map?",
     "Whose turn is it to present the roadmap?"),
    ("The gooey freezes when the render queue fills up.",
     "The GUI freezes when the render queue fills up."),
    ("We use fast API for the back end services.",
     "We use FastAPI for the backend services."),
    ("The affect of the patch was immediate.",
     "The effect of the patch was immediate."),
    ("Darlian met Sex and Aaron at the colony summit.",
     "Darlian met Zechs and Heero at the colony summit."),
]

_norm_re = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _norm_re.sub(" ", (text or "").strip().lower())


def _word_count(text: str) -> int:
    return len((text or "").split())


def score_results(polished_texts: list[str]) -> dict:
    """Score polished output against the gold corrections."""
    n = len(BENCH_CASES)
    exact = 0
    budget_ok = 0
    for (src, gold), out in zip(BENCH_CASES, polished_texts):
        if _norm(out) == _norm(gold):
            exact += 1
        wc_in, wc_out = _word_count(src), _word_count(out or "")
        if wc_in and abs(wc_out - wc_in) / wc_in <= 0.15 + 1e-9:
            budget_ok += 1
    count_ok = len(polished_texts) == n
    return {
        "cases": n,
        "exact_fixes": exact,
        "exact_fix_rate": round(exact / n, 3),
        "word_budget_ok": budget_ok,
        # Segment-count discipline is half the compliance score; the
        # per-segment word budget is the other half.
        "format_compliance": round(
            (0.5 * (1.0 if count_ok else 0.0)) + (0.5 * budget_ok / n), 3),
        "segment_count_ok": count_ok,
    }


async def run_benchmark(model_id: str, timeout_per_batch: float = 120.0) -> dict:
    """Run the benchmark against ``model_id`` via the polish pipeline."""
    from backend.services.ai_orchestrator import AIOrchestrator
    from backend.services.transcript_polisher import correct_transcript

    segments = [
        TranscriptSegment(start=float(i * 5), end=float(i * 5 + 4),
                          text=src, speaker="Speaker 1")
        for i, (src, _gold) in enumerate(BENCH_CASES)
    ]
    orchestrator = AIOrchestrator()
    try:
        polished = await correct_transcript(
            segments, orchestrator=orchestrator,
            language="en",
            timeout_per_batch=timeout_per_batch,
            model_override=model_id,
        )
    except Exception as e:
        return {"error": f"polish call failed: {e}", "model": model_id}
    texts = [getattr(s, "text", "") for s in polished]
    result = score_results(texts)
    result["model"] = model_id
    return result
