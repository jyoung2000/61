"""Centralized AI prompt management for ClipAI.

Stores default prompts (extracted from the best OpenRouter versions) and
handles loading / saving user customizations from disk.
"""

import json
import logging
import os

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROMPTS_FILE = "/data/logs/custom_prompts.json"
MAX_PROMPT_LENGTH = 10_000

# ── Default Prompts ─────────────────────────────────────────────────
# Canonical instruction text shared by all providers.
# JSON output format and strict requirements are appended by each
# provider separately so users can't accidentally break them.

DEFAULT_FRAME_ANALYSIS_PROMPT = (
    "You are analyzing frames from a video to identify key visual moments.\n\n"
    "For each frame, describe:\n"
    "1. What is visually happening (people, actions, setting, on-screen text)\n"
    "2. Whether this is a visually striking or spectacle moment (dramatic visuals, "
    "reactions, reveals, cool effects, beautiful scenery, action sequences)\n"
    "3. Social media potential — would this frame make someone stop scrolling?\n\n"
    "Rate importance from 1-10 where:\n"
    "  1-3 = mundane (talking head, static scene)\n"
    "  4-6 = interesting (good visuals, clear action)\n"
    "  7-8 = compelling (strong emotion, visual spectacle, key reveal)\n"
    "  9-10 = viral-worthy (jaw-dropping moment, perfect reaction, stunning visual)"
)

# ── VLM upgrade Phase 1 — grounding-output prompt ──

# ── 4-axis scoring rubric (Phase 1 of OpusClip parity gap) ───────────
# Every genre prompt below shares this base. The LLM must return the
# four axes (hook / flow / value / trend) per clip; the composite
# ``viral_score`` is computed in Python from those axes (see
# clip_scoring.py).
FOUR_AXIS_RUBRIC = (
    "SCORING — return four independent axis scores (0-100) per clip:\n\n"
    "1. hook_score (0-100): Does the FIRST 3 SECONDS pull a scrolling viewer in?\n"
    "   90+ — bold question, jaw-dropping claim, visual spectacle, or a clean\n"
    "         emotional spike (laughter, shouting, gasp) lands inside the first 3s.\n"
    "   60-80 — clear topic introduction with a confident opener and no dead air.\n"
    "   30-50 — generic exposition, mid-thought entry, or weak energy at t=0.\n"
    "   0-25 — dead air, mid-sentence start, filler word opener (\"um\", \"so\", \"and\"),\n"
    "         or context-dependent pronoun opener (\"that was\", \"it's\").\n\n"
    "2. flow_score (0-100): Does the clip stay on ONE topic, ONE scene, ONE exchange,\n"
    "   with a setup → payoff arc?\n"
    "   90+ — single coherent moment with a clear beginning / middle / end.\n"
    "   60-80 — mostly cohesive; one or two minor digressions but the through-line is clear.\n"
    "   30-50 — drifts between sub-topics or cuts across scenes.\n"
    "   0-25 — incoherent: stitches unrelated moments or ends mid-thought.\n\n"
    "3. value_score (0-100): Does the clip RESOLVE — answer a question, reveal\n"
    "   something, land a punchline, or deliver an emotional payoff?\n"
    "   90+ — explicit payoff: punchline, shocking reveal, hot take, satisfying answer.\n"
    "   60-80 — solid takeaway viewer would remember.\n"
    "   ~50 — interesting but unresolved: the clip is engaging but doesn't actually\n"
    "         arrive anywhere.\n"
    "   0-30 — no payoff, filler dialogue, no quotable moment.\n\n"
    "4. trend_score (0-100): How closely does the topic/vibe match current short-form\n"
    "   patterns for the detected genre and platform?\n"
    "   - If a TREND CONTEXT block is provided below, use it. Phrases that overlap\n"
    "     a listed trending topic should score 70+, phrases adjacent to one 50-65,\n"
    "     and unrelated content 30-45.\n"
    "   - When NO trend context is provided, default to 50 (neutral). Do NOT guess.\n\n"
    "Each axis ALSO needs a ONE-SENTENCE reason field explaining the score:\n"
    "  hook_reason, flow_reason, value_reason, trend_reason.\n"
    "Be concrete and specific. Bad: \"good hook\". Good: \"opens with the question\n"
    "'why does nobody talk about this' — strong scroll-stopper\".\n"
)


_VIRAL_BASE_INSTRUCTIONS = (
    "DETECTION METHODOLOGY (follow this two-phase process):\n\n"
    "PHASE 1 — SCAN: Read through the transcript and scene descriptions chronologically. "
    "Identify ALL potential clip-worthy moments. Look for:\n"
    " - Transcript energy spikes (exclamations, questions, rapid exchanges, laughter)\n"
    " - High-importance visual moments (score 7+ in scene descriptions, marked with ★)\n"
    " - Moments where strong dialogue COINCIDES with strong visuals\n"
    " - Natural story arcs: setup → tension → payoff within a contained segment\n"
    " - Speaker changes that mark the start or end of a distinct exchange\n\n"
    "PHASE 2 — SCORE each candidate against the 4-axis rubric below. Do NOT\n"
    "blend the axes into a single number — return them separately and we will\n"
    "compose the final score in code.\n\n"
    "SCENE & SUBJECT COHERENCE (CRITICAL):\n"
    "- The main subject MUST stay in focus throughout the entire clip\n"
    "- NEVER cut across unrelated scenes or topics — the clip must feel like ONE moment\n"
    "- If a clip covers a conversation, keep it within the same exchange\n"
    "- Avoid clips that start on one topic/scene and drift into a completely different one\n"
    "- The visual setting should remain consistent — don't span across location changes\n"
    "- Prefer segments where the camera stays on the main action without jarring cuts\n"
    "- If scene descriptions show different settings at different timestamps, do NOT combine them into one clip\n\n"
    "TITLE RULES:\n"
    "- Write titles as SEO-optimized social media captions about the TOPIC/SUBJECT\n"
    "- NEVER include speaker names, 'Speaker 1', 'Speaker 2', or any speaker labels\n"
    "- NEVER include timestamps, internal IDs, or technical labels like [317-405]\n"
    "- Good: 'The Truth About Celebrity Gossip', 'This Reaction Was Priceless'\n"
    "- Bad: 'Speaker 1 reacts to news', '[317-405] Speaker 1: rapid_exchange'\n\n"
    "BOUNDARY RULES:\n"
    "- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
    "- End at natural conclusions — punchlines, resolved thoughts, scene transitions\n"
    "- Must work standalone without context from the full video\n"
)


def _build_viral_prompt(role_line: str, genre_block: str = "") -> str:
    """Compose a genre prompt from the shared base + a per-genre block."""
    parts = [role_line.rstrip(), "", _VIRAL_BASE_INSTRUCTIONS, FOUR_AXIS_RUBRIC]
    if genre_block:
        parts.append(genre_block.rstrip())
    return "\n".join(parts)


# Generic / fallback prompt — same role line as the legacy default.
_GENERIC_ROLE = (
    "You are an expert social media video strategist who identifies the most "
    "viral-worthy, attention-grabbing moments in long-form content. Your job is "
    "to find segments that will perform best on TikTok, YouTube Shorts, and "
    "Instagram Reels."
)

DEFAULT_VIRAL_CLIP_PROMPT = _build_viral_prompt(_GENERIC_ROLE)


# ── Genre-specific prompt variants (Phase 2) ────────────────────────
# Each variant adds a genre block that re-tunes what a 90+ score on
# each axis looks like for that content type and lists genre-specific
# examples. The shared 4-axis rubric still applies — the genre block
# just refines it.

VIRAL_PROMPT_TALKING_HEAD = _build_viral_prompt(
    "You are a podcast / interview / vlog clip editor finding the moments most "
    "likely to be reposted as standalone shorts. Focus on quotable hot takes, "
    "clean exchanges, and reaction-worthy answers. Visual spectacle is rare in "
    "this genre — judge clips on what is SAID, not what is shown.",
    genre_block=(
        "GENRE TUNING — TALKING HEAD / PODCAST / INTERVIEW / VLOG:\n"
        "- A 90+ HOOK is a question the audience wants answered or a confident\n"
        "  declarative claim. Mid-question entries are penalised hard.\n"
        "- A 90+ FLOW stays inside a single exchange between speakers — never\n"
        "  glue together two questions that cover different topics.\n"
        "- A 90+ VALUE is a quotable line: \"Most people get this wrong because\n"
        "  ___\", a confessional reveal, or a punchline that lands.\n"
        "- Prefer 30-60s clips that capture one full Q→A or one self-contained\n"
        "  monologue beat. Do not pad to fill duration.\n"
    ),
)

VIRAL_PROMPT_GAMEPLAY = _build_viral_prompt(
    "You are a gameplay highlight editor finding clutch plays, kill streaks, "
    "skill moments, funny deaths, and reaction-worthy commentary. Most viewers "
    "are scrolling — the first second of action decides whether they stop.",
    genre_block=(
        "GENRE TUNING — GAMEPLAY (FPS / MOBA / TPS / RACING):\n"
        "- A 90+ HOOK opens on a moment of high stakes or a kinetic action beat:\n"
        "  the start of a teamfight, a clutch round, an enemy contact, or a\n"
        "  funny fail moment. Static menu / loadout screens are dead air — score 0-25.\n"
        "- A 90+ FLOW is a single play that resolves in a clear win or loss.\n"
        "  Avoid stitching plays from different rounds together.\n"
        "- A 90+ VALUE has a clear payoff: the kill, the clutch, the joke. The\n"
        "  commentary reaction (\"NO WAY\", \"OH MY GOD\") is a reliable payoff signal.\n"
        "- De-emphasise long stretches of dialogue between fights — those are\n"
        "  filler in this genre.\n"
        "- Prefer 15-45s clips. Anything longer than 60s loses scrollers.\n"
    ),
)

VIRAL_PROMPT_SPORTS = _build_viral_prompt(
    "You are a sports highlight editor finding scoring plays, close calls, "
    "crowd reactions, and dramatic moments. Score boundaries by play "
    "completion, not sentence boundaries — the cheering after the play is part "
    "of the clip.",
    genre_block=(
        "GENRE TUNING — SPORTS:\n"
        "- A 90+ HOOK opens a few seconds before the decisive moment so viewers\n"
        "  feel the build-up. Cold opens on a celebration are weaker (40-60).\n"
        "- A 90+ FLOW is one continuous play from setup to result, ending after\n"
        "  the crowd reaction. Cutting before the cheer kills the payoff.\n"
        "- A 90+ VALUE is the play itself + the reaction (crowd, commentary,\n"
        "  player). A scoring play with no reaction shot is ~70.\n"
        "- Boundary rule: snap to play start / whistle, not to mid-sentence\n"
        "  commentary. The commentator can still be mid-word at the start.\n"
        "- Prefer 15-40s clips.\n"
    ),
)

VIRAL_PROMPT_MUSIC_VIDEO = _build_viral_prompt(
    "You are a music video editor finding beat drops, chorus moments, and "
    "iconic visual motifs. The audio waveform is the primary signal — find the "
    "moments where the music peaks and the visuals support it.",
    genre_block=(
        "GENRE TUNING — MUSIC VIDEO:\n"
        "- A 90+ HOOK is a beat drop, vocal entry, or a striking visual motif\n"
        "  in the first 1-2 seconds. Long instrumental intros without a payoff\n"
        "  are weaker (30-50).\n"
        "- A 90+ FLOW is bar-aligned: starts on a downbeat, ends on a phrase\n"
        "  resolution. Do NOT cut mid-bar.\n"
        "- A 90+ VALUE is the chorus or the most memorable visual sequence —\n"
        "  the part viewers would loop or duet with.\n"
        "- Boundary rule: snap to beat boundaries. Even a 0.3s offset feels wrong.\n"
        "- Trend axis matters more than usual here — match against what is\n"
        "  currently going viral on TikTok sound.\n"
        "- Prefer 15-30s clips.\n"
    ),
)

VIRAL_PROMPT_ANIMATION = _build_viral_prompt(
    "You are an animation / anime clip editor finding reaction shots, "
    "punchline frames, action peaks, and emotional beats. Cuts must respect "
    "shot boundaries — never combine two scenes that show different characters "
    "in different settings.",
    genre_block=(
        "GENRE TUNING — ANIMATION / ANIME / CARTOON:\n"
        "- A 90+ HOOK is a striking pose, a sudden expression, or a sharp\n"
        "  motion beat in the first second. Static establishing shots are weak\n"
        "  (30-45).\n"
        "- A 90+ FLOW is one scene with one character set. Hard cut to a new\n"
        "  scene = drop flow to 30 or below.\n"
        "- A 90+ VALUE is a punchline frame, a reveal expression, an action peak,\n"
        "  or an emotional climax. Talking heads with no expression change ~50.\n"
        "- Character consistency is mandatory: do not glue together two scenes\n"
        "  with different protagonists.\n"
        "- Prefer 15-40s clips.\n"
    ),
)

VIRAL_PROMPT_NARRATIVE = _build_viral_prompt(
    "You are a narrative film / TV clip editor finding cinematic dialogue, "
    "reveal moments, and emotional beats. Respect shot boundaries hard — a "
    "clip must live inside a single scene.",
    genre_block=(
        "GENRE TUNING — NARRATIVE / CINEMATIC DIALOGUE:\n"
        "- A 90+ HOOK opens on a clean line delivery or a striking visual.\n"
        "  Mid-line entries lose 25 points.\n"
        "- A 90+ FLOW lives entirely inside one scene with one set of\n"
        "  characters. Scene cuts inside the clip = drop flow hard.\n"
        "- A 90+ VALUE is a reveal, a confession, a punchline, or a line that\n"
        "  hits hard out of context.\n"
        "- Boundary rule: snap to shot transitions, not to mid-line audio.\n"
        "- Prefer 20-50s clips so the beat has room to breathe.\n"
    ),
)

VIRAL_PROMPT_GENERIC = DEFAULT_VIRAL_CLIP_PROMPT


def get_genre_prompt(content_type) -> str:
    """Return the right viral-clip prompt variant for a content type.

    Accepts a ``ClipContentType`` enum value (or anything with a
    ``.value`` attribute / a string). Always returns a non-empty
    string — falls back to ``DEFAULT_VIRAL_CLIP_PROMPT`` for unknown
    types so the caller never has to null-check.
    """
    # Defer the import so this module stays cheap to import. The
    # content_classifier module is several hundred lines of heuristics
    # we do not need just to look up an enum.
    try:
        from backend.services.compat_stubs import ClipContentType
    except Exception:
        return DEFAULT_VIRAL_CLIP_PROMPT

    if content_type is None:
        return DEFAULT_VIRAL_CLIP_PROMPT

    if hasattr(content_type, "value"):
        key = content_type.value
    else:
        key = str(content_type)

    mapping = {
        ClipContentType.TALKING_HEAD.value: VIRAL_PROMPT_TALKING_HEAD,
        ClipContentType.MULTI_SPEAKER_PANEL.value: VIRAL_PROMPT_TALKING_HEAD,
        ClipContentType.GAMEPLAY.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_MOBA.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_TPS.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.GAMEPLAY_RACING.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.STREAM.value: VIRAL_PROMPT_GAMEPLAY,
        ClipContentType.SPORTS.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.SPORTS_BASKETBALL.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.SPORTS_RACING.value: VIRAL_PROMPT_SPORTS,
        ClipContentType.MUSIC_VIDEO.value: VIRAL_PROMPT_MUSIC_VIDEO,
        ClipContentType.ANIMATION.value: VIRAL_PROMPT_ANIMATION,
        ClipContentType.ANIMATION_DIALOGUE.value: VIRAL_PROMPT_ANIMATION,
        ClipContentType.CINEMATIC_DIALOGUE.value: VIRAL_PROMPT_NARRATIVE,
        ClipContentType.GENERIC.value: VIRAL_PROMPT_GENERIC,
    }
    return mapping.get(key, VIRAL_PROMPT_GENERIC)


# ── Output-language helpers ────────────────────────────────────────────────
# The summary + SEO must come out in the SAME language as the clips/subtitles
# (the user-selected subtitle language) — not always English, and not the raw
# source language. These build a "write in <language>" directive injected into
# the summary and SEO prompts at generation time.
_OUTPUT_LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "uk": "Ukrainian", "sv": "Swedish", "id": "Indonesian",
    "ms": "Malay", "tl": "Filipino", "fa": "Persian", "he": "Hebrew",
    "el": "Greek", "cs": "Czech", "ro": "Romanian", "hu": "Hungarian",
    "fi": "Finnish", "da": "Danish", "no": "Norwegian", "nb": "Norwegian",
}


def output_language_name(code: str) -> str:
    """Human language name for an ISO code ('en' → 'English'); '' if unknown."""
    c = (code or "").strip().lower().split("-")[0]
    return _OUTPUT_LANG_NAMES.get(c, "")


def summary_language_directive(code: str) -> str:
    """'Write the summary in <language>' instruction for the resolved subtitle
    language. Empty/unknown defaults to English (the historical behavior)."""
    name = output_language_name(code) or "English"
    return (
        f"OUTPUT LANGUAGE — write your ENTIRE response in {name}. The transcript "
        f"and scene notes may be in a different language; translate your "
        f"understanding and write ONLY in {name} (keep proper nouns in their "
        f"standard {name} spelling). Do not mix languages.\n\n")


DEFAULT_SUMMARY_PROMPT = (
    "You are writing a substantive video summary for a human audience.\n\n"
    "Write like a real person walking a friend through what they just "
    "watched — but be GENEROUS with detail. We want the reader to walk "
    "away feeling like they have a real sense of how the video unfolds, "
    "who appears, what happens in each section, and what makes it worth "
    "watching. Not a press release, not a one-liner.\n\n"
    "IMPORTANT: Do not reference or speculate about speakers by name "
    "unless the speaker names are explicitly present in the transcript. "
    "Focus on WHAT is discussed and shown — the speakers, the actions, "
    "the visuals, the topics, the emotional beats.\n\n"
    "Based on the transcript and scene descriptions below, return ONLY "
    "valid JSON:\n"
    '{"overview": "<6-10 sentence paragraph or 2-3 short paragraphs>", '
    '"narrative_arc": "<3-5 sentences describing how the video unfolds '
    'from opening to close — the beats, the turns, the payoff>", '
    '"key_topics": ["topic1", "topic2", ...], '
    '"highlight_moments": ["timestamp + 1-sentence description", ...], '
    '"tone": "<1-3 words>", "estimated_audience": "<who would watch>", '
    '"content_category": "<category>"}\n\n'
    "Field guidelines:\n"
    "- overview: 6-10 sentences (or 2-3 short paragraphs). Cover the "
    "  setup, the main thread, any meaningful turn / climax, and how it "
    "  ends. Use natural conversational language. Describe both the "
    "  audio (what is said) and the visuals (what is shown).\n"
    '   Good: "Two friends sit down at a beat-up picnic table to taste-'
    "  test five fast-food burgers. The bit opens with them ranking "
    "  the chains beforehand — Five Guys is favoured, In-N-Out is "
    "  doubted — then the food shows up and the rankings start to "
    "  fall apart. There's a sauce-spill disaster midway through, an "
    "  unexpected love for Wendy's, and a final scorecard where the "
    "  bottom-ranked chain ends up winning. Light, off-the-cuff, "
    "  obviously not staged.\"\n"
    "- narrative_arc: 3-5 sentences specifically on the *structure* — "
    "  hook → development → twist → resolution. This is the part "
    "  someone reads to decide if the pacing fits what they want.\n"
    "- key_topics: 5-10 specific topics. Use natural phrases, not "
    "  generic SEO keywords.\n"
    '   Good: ["fast food taste test", "In-N-Out vs Five Guys", '
    '   "sauce disaster", "Wendy\'s surprise winner"]\n'
    "- highlight_moments: 4-8 concrete beats with their approximate "
    '  timestamp (use mm:ss). Each line: "01:23 — sauce explodes on '
    '  shirt, both laugh", "04:10 — first In-N-Out bite, surprised '
    '  reaction".\n'
    '- tone: The vibe (e.g. "funny and casual", "tense and '
    '  thoughtful", "high-energy chaotic")\n'
    '- estimated_audience: Be specific (e.g. "foodies and fast food fans")\n'
    '- content_category: Specific (e.g. "food review", "tech unboxing", "comedy sketch")\n'
)

# ── VLM upgrade Phase 5 — editorial crop QA prompt ────────────────
# Used by backend.services.crop_qa.score_crop_quality to re-score
# each rendered segment's output frames. The VLM's answer is parsed
# as JSON and fed into decide_recovery() to decide whether to
# re-solve the crop with looser constraints or fall through to a
# safety-center crop. Gated behind CLIPAI_CROP_QA; see
# docs/vlm_upgrade/PHASE_5_NOTES.md.
DEFAULT_CROP_QA_PROMPT = (
    "You are reviewing a vertical 9:16 crop of a horizontal source video. "
    "Look at this output frame and answer:\n\n"
    "- head_in_frame: bool — is the main subject's head fully inside the "
    "frame (not cut off at top/bottom/sides)?\n"
    "- awkward_crop: bool — is there an awkward edge cut (hand chopped "
    "mid-gesture, half a face, body cut at the neck)?\n"
    "- subject_partially_off_frame: bool — is the subject visible but "
    "partially clipped at a frame edge?\n"
    "- dead_space_dominant: bool — is more than 50% of the frame empty/"
    "non-subject space?\n"
    "- quality_score: 0-10. 10 = perfect editorial framing (subject "
    "well-placed with appropriate headroom, no awkward cuts). 7+ = "
    "acceptable. 5-6 = noticeable problems but watchable. <5 = "
    "unwatchable, must re-frame.\n\n"
    "Return JSON only."
)


DEFAULT_SEO_PROMPT = (
    "You write social media captions and tags like a real person — not a marketer, "
    "not a robot. Think of how popular creators actually post on YouTube, TikTok, "
    "Instagram, and Tumblr. The text should feel natural and authentic.\n\n"
    "Given a video clip, generate:\n\n"
    "1. TITLE — Write it like a real post title. Keep it under 100 characters. "
    "It should sound like something a person would actually type, not an ad. "
    "Use lowercase naturally. No clickbait, no ALL CAPS spam, no excessive punctuation.\n"
    "   Good: \"when the beat dropped and nobody was ready\"\n"
    "   Good: \"This changed how I think about cooking\"\n"
    "   Bad: \"YOU WON'T BELIEVE What Happens Next!!!\"\n\n"
    "2. DESCRIPTION — Write 1-3 casual sentences like a real caption someone would "
    "post. Can include personality, humor, or a brief thought. Keep it 100-250 chars. "
    "Don't stuff keywords or write like a press release.\n\n"
    "3. TAGS — 8-15 hashtags (with # prefix) that a real person would actually use. "
    "Mix popular broad tags with specific niche ones. Use lowercase. "
    "These are the tags people search and browse on social platforms.\n"
    "   Example: [\"#cooking\", \"#foodtok\", \"#recipe\", \"#homemade\", \"#fyp\"]\n\n"
    "4. PLATFORM_TIPS — One short sentence of posting advice for this specific clip.\n\n"
    "Return ONLY valid JSON:\n"
    '{"title": "...", "description": "...", "tags": ["#tag1", "#tag2", ...], '
    '"platform_tips": "..."}'
)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-platform SEO — each platform has its own ranking signals, char caps,
#  and hashtag culture; using the same prompt for all of them produces
#  generic copy that wins on none of them. The slugs match clip.platform
#  values the bridge / clipper emit.
#
#  DATA-DRIVEN (2026 overhaul): the caps + guidance load from
#  backend/data/platform_rules.json (shipped, verified July-2026 values) with
#  a platform_rules.live.json overlay written by the weekly self-researcher
#  (backend/services/platform_rules_research.py) taking field-level
#  precedence, and the hardcoded dict below as the last-resort fallback.
#  Precedence: overlay > shipped file > hardcoded.
# ═══════════════════════════════════════════════════════════════════════════

_RULES_SHIPPED_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "platform_rules.json")
_RULES_OVERLAY_NAME = "platform_rules.live.json"

# Hard fallback — mirrors the shipped platform_rules.json (July-2026 verified:
# generic discovery tags penalized everywhere, Reels hard-capped at 5 tags,
# TikTok 3-5 targeted tags / 4000-char keyword-first captions, #Shorts
# optional). Keep in sync when editing the shipped file.
_PROFILES_FALLBACK: dict[str, dict] = {
    "tiktok": {
        "label": "TikTok", "title_max": 150, "description_max": 4000,
        "tag_min": 3, "tag_max": 5,
        "guidance": (
            "TIKTOK profile — keyword-first search surface (2026). Discovery is "
            "driven by the caption text, the spoken audio (ASR), and on-screen "
            "text (OCR); hashtags are only a supporting signal.\n"
            "TITLE: the first caption line (≤150 chars). Put the primary search "
            "keyword inside the first 50 characters. Lowercase, conversational, "
            "references a concrete moment from the clip.\n"
            "DESCRIPTION: keyword-rich natural sentences (total ≤4000 chars incl. "
            "hashtags). First 50 chars carry the primary keyword; then 1-3 casual "
            "sentences adding context the video doesn't say. Hashtags on their own "
            "line at the end.\n"
            "TAGS: 3-5 TARGETED hashtags tied to the actual subject. Generic "
            "discovery tags (#fyp, #foryou, #viral) are penalized/ignored — never "
            "use them.\n"
            "PLATFORM_TIPS: 1 line on the sound/duet/stitch angle that would "
            "amplify this clip."
        ),
    },
    "youtube_shorts": {
        "label": "YouTube Shorts", "title_max": 100, "description_max": 5000,
        "tag_min": 3, "tag_max": 5,
        "guidance": (
            "YOUTUBE SHORTS profile — keyword-front-loaded, search-discoverable. "
            "Shorts can run up to 3 minutes.\n"
            "TITLE: ≤100 chars, primary keyword inside the first 50. Title case. "
            "#Shorts in the title is OPTIONAL — auto-detection made it "
            "unnecessary; add it only when it reads naturally.\n"
            "DESCRIPTION: 3-5 sentences; the primary keyword must appear in the "
            "first 100 chars (that's what search shows). Reference things actually "
            "SAID in the transcript for keyword density. End with 3-5 hashtags "
            "mirroring the primary keywords on their own line.\n"
            "TAGS: 3-5 description hashtags mirroring the primary keywords. "
            "#Shorts allowed here.\n"
            "PLATFORM_TIPS: 1 line on the thumbnail moment + the query this "
            "should rank for in Shorts search."
        ),
    },
    "reels": {
        "label": "Instagram Reels", "title_max": 125, "description_max": 2200,
        "tag_min": 3, "tag_max": 5,
        "guidance": (
            "INSTAGRAM REELS profile — keyword-rich captions drive discovery; "
            "hashtags are HARD-CAPPED at 5 per post (enforced Dec 2025; some "
            "accounts limited to 3).\n"
            "TITLE: the first caption line (≤125 chars) — primary keyword inside "
            "the first 50 chars, most interesting thing first.\n"
            "DESCRIPTION: 2-4 natural-language sentences a real person would "
            "write, keyword-first, ≤2200 chars. End with a question/CTA that "
            "invites comments, then a line break, then the hashtags IN THE "
            "CAPTION (not the first comment).\n"
            "TAGS: 3-5 niche mid-tier hashtags tied to the subject. Generic tags "
            "(#reels, #explorepage, #viral) are actively discouraged — never use "
            "them.\n"
            "PLATFORM_TIPS: 1 line on the on-screen text overlay + any trending "
            "audio pairing."
        ),
    },
    "instagram": {
        "label": "Instagram Feed", "title_max": 125, "description_max": 2200,
        "tag_min": 3, "tag_max": 5,
        "guidance": (
            "INSTAGRAM FEED profile — keyword-rich caption, hashtag-light (the "
            "5-hashtag cap applies to feed posts too).\n"
            "TITLE: opening hook (≤125 chars), primary keyword early — it's the "
            "line before 'more'.\n"
            "DESCRIPTION: 2-5 sentences building context in a personal voice. "
            "Explicit CTA (\"save this if you...\"). Line break, then hashtags. "
            "≤2200 chars.\n"
            "TAGS: 3-5 micro-targeted niche hashtags. No generic discovery tags.\n"
            "PLATFORM_TIPS: 1 line on posting window + carousel vs single."
        ),
    },
    "youtube": {
        "label": "YouTube (Long-form)", "title_max": 70, "description_max": 5000,
        "tag_min": 5, "tag_max": 10,
        "guidance": (
            "YOUTUBE LONG-FORM profile — SEO-optimized for search ranking + "
            "watch time.\n"
            "TITLE: ≤70 chars (mobile cutoff), primary search keyword "
            "front-loaded. Title case. Numbers and brackets perform. No clickbait "
            "mismatch.\n"
            "DESCRIPTION: 1000-3000 chars. First 100 chars MUST contain the "
            "primary keyword (they show in search). Then 2-3 paragraphs weaving "
            "in secondary keywords from the transcript, a timestamps section when "
            "scenes are available, a CTA paragraph, and hashtags at the bottom.\n"
            "TAGS: 5-10 focused hashtags. The first 3 appear above the title — "
            "spend them on the highest-value ranking keywords. Mixed case OK.\n"
            "PLATFORM_TIPS: 1 line on the thumbnail keyword + the chapter where "
            "retention is highest."
        ),
    },
    "x": {
        "label": "X (Twitter)", "title_max": 280, "description_max": 280,
        "tag_min": 0, "tag_max": 3,
        "guidance": (
            "X / TWITTER profile — the hook IS the post; hashtags are "
            "deprioritized.\n"
            "TITLE: the post body, ≤280 chars INCLUDING hashtags. One punchy line "
            "carrying the primary keyword naturally; optionally a second line of "
            "context.\n"
            "DESCRIPTION: an alternate longer wording (≤280 chars) — return empty "
            "string if the title already says it.\n"
            "TAGS: 0-3 hashtags MAX — the best post often has zero. Never stuff.\n"
            "PLATFORM_TIPS: 1 line — standalone post, reply to a trending topic, "
            "or thread opener."
        ),
    },
    "facebook": {
        "label": "Facebook", "title_max": 100, "description_max": 63206,
        "tag_min": 2, "tag_max": 4,
        "guidance": (
            "FACEBOOK profile — conversational, comment-bait, low-hashtag.\n"
            "TITLE: ≤100 chars; the first 80 are what mobile shows before 'See "
            "more' — hook and keyword go there.\n"
            "DESCRIPTION: 2-5 sentences in a personal voice, 200-500 chars sweet "
            "spot, ending with a question or invitation (comments and shares are "
            "the ranking fuel).\n"
            "TAGS: 2-4 hashtags — three well-chosen tags beat fifteen.\n"
            "PLATFORM_TIPS: 1 line — Reel vs feed video vs boosted post."
        ),
    },
    "linkedin": {
        "label": "LinkedIn", "title_max": 150, "description_max": 3000,
        "tag_min": 3, "tag_max": 5,
        "guidance": (
            "LINKEDIN profile — professional voice, insight-led, hashtag-light.\n"
            "TITLE: insight-led hook ≤150 chars — lead with the takeaway, not the "
            "format. No emojis.\n"
            "DESCRIPTION: 3-7 sentences as single-line paragraphs (white space "
            "wins). Insight → one concrete example from the transcript → a "
            "question inviting professional comments. 500-1500 chars.\n"
            "TAGS: 3-5 industry-specific hashtags in CamelCase (#ProductDesign) — "
            "they drive topic clustering.\n"
            "PLATFORM_TIPS: 1 line on the target reader role + personal vs "
            "company page."
        ),
    },
}

# Sanity ranges for any file-loaded rule — a corrupt or hallucinated value
# must never make it into the live profiles (Part 6 hard requirement).
_RULE_BOUNDS = {
    "title_max": (20, 500),
    "description_max": (100, 100_000),
    "tag_min": (0, 30),
    "tag_max": (1, 30),
}


def validate_platform_rule(rule: dict) -> dict:
    """Return only the sane fields of one platform's rule dict.

    Numeric caps outside ``_RULE_BOUNDS`` (or a tag_min above tag_max) are
    dropped field-by-field, so a partially-bad overlay still contributes its
    good fields. ``guidance`` may be a string or a list of lines."""
    out: dict = {}
    if not isinstance(rule, dict):
        return out
    for key, (lo, hi) in _RULE_BOUNDS.items():
        try:
            v = int(rule[key])
        except (KeyError, TypeError, ValueError):
            continue
        if lo <= v <= hi:
            out[key] = v
    if ("tag_min" in out and "tag_max" in out
            and out["tag_min"] > out["tag_max"]):
        out.pop("tag_min")
        out.pop("tag_max")
    label = rule.get("label")
    if isinstance(label, str) and label.strip():
        out["label"] = label.strip()
    guidance = rule.get("guidance")
    if isinstance(guidance, list):
        guidance = "\n".join(str(line) for line in guidance)
    if isinstance(guidance, str) and guidance.strip():
        out["guidance"] = guidance
    return out


def platform_rules_overlay_path() -> str:
    """Where the weekly self-researcher writes its live rules overlay."""
    from backend.services.seo_hygiene import writable_state_dir
    return os.path.join(writable_state_dir(), _RULES_OVERLAY_NAME)


def _read_rules_file(path: str) -> dict:
    """{platform: validated-partial-rule} from a rules JSON; {} on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return {}
    plats = data.get("platforms") if isinstance(data, dict) else None
    if not isinstance(plats, dict):
        plats = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(plats, dict):
        return {}
    return {str(k): validate_platform_rule(v) for k, v in plats.items()
            if str(k) in _PROFILES_FALLBACK and validate_platform_rule(v)}


# Provenance for GET /api/seo/intel — which layer each profile came from.
_PLATFORM_RULES_META: dict = {"source": "builtin", "refreshed": ""}


def _load_platform_rules() -> dict[str, dict]:
    """Compose the effective profiles: hardcoded ← shipped file ← overlay.

    Field-level merge so a partial overlay (e.g. only tag caps) keeps the
    shipped guidance text. Never raises — the hardcoded fallback always
    stands underneath."""
    import copy
    profiles = copy.deepcopy(_PROFILES_FALLBACK)
    source = "builtin"
    refreshed = ""
    try:
        shipped = _read_rules_file(_RULES_SHIPPED_FILE)
        if shipped:
            source = "shipped"
            for plat, rule in shipped.items():
                profiles[plat].update(rule)
    except Exception as e:
        logger.warning("platform_rules.json unreadable (%s) — built-in rules", e)
    try:
        overlay_path = platform_rules_overlay_path()
        overlay = _read_rules_file(overlay_path)
        if overlay:
            source = "live"
            for plat, rule in overlay.items():
                profiles[plat].update(rule)
            try:
                with open(overlay_path) as f:
                    refreshed = str(json.load(f).get("date") or "")
            except Exception:
                pass
    except Exception as e:
        logger.debug("platform rules overlay skipped (%s)", e)
    _PLATFORM_RULES_META.update({"source": source, "refreshed": refreshed})
    # Default for unknown / "both" / legacy values — a balanced short-form
    # profile so generation doesn't silently degrade.
    profiles["both"] = profiles["tiktok"]
    profiles["default"] = profiles["tiktok"]
    return profiles


PLATFORM_PROFILES: dict[str, dict] = _load_platform_rules()


def reload_platform_rules() -> None:
    """Re-read the rules files into the LIVE dict (in place, so every module
    holding a reference to ``PLATFORM_PROFILES`` sees the update)."""
    fresh = _load_platform_rules()
    PLATFORM_PROFILES.clear()
    PLATFORM_PROFILES.update(fresh)


def platform_rules_meta() -> dict:
    """{'source': 'live'|'shipped'|'builtin', 'refreshed': iso-date} — where
    the effective rules came from (for /api/seo/intel)."""
    return dict(_PLATFORM_RULES_META)


def build_platform_seo_prompt(platform: str, trend_brief: str = "",
                              output_language: str = "") -> str:
    """Return the platform-specific, KEYWORD-FIRST SEO prompt for a platform.

    Falls back to ``PLATFORM_PROFILES['default']`` (TikTok-style) when the
    platform is unknown so a new clip type still gets reasonable output
    instead of crashing.

    ``trend_brief`` must be ONLY the requesting platform's section of today's
    structured brief (rendered text — see
    ``trend_brief.render_platform_section``), never the combined multi-platform
    blob: LinkedIn copy shaped by TikTok trends wins on neither platform.

    ``output_language`` (the user's subtitle language) forces the title,
    caption, hook and tips to come out in the SAME language as the clip —
    so a Japanese-source / English-subtitle clip gets English SEO, and vice
    versa — instead of defaulting to the transcript's language.
    """
    profile = PLATFORM_PROFILES.get(platform) or PLATFORM_PROFILES["default"]
    _lang_name = output_language_name(output_language) or "English"
    lang_block = (
        f"OUTPUT LANGUAGE — write the TITLE, DESCRIPTION/caption, HOOK and "
        f"PLATFORM_TIPS in {_lang_name} (the language of this clip's subtitles), "
        f"even if the transcript below is in another language. For TAGS, use "
        f"{_lang_name} hashtags relevant to the clip; widely-understood "
        f"subject tags in another language are fine when they're what users "
        f"actually search. Never mix languages in the title or caption.\n\n")
    trend_block = ""
    if (trend_brief or "").strip():
        trend_block = (
            "TODAY'S LIVE TREND BRIEF for THIS platform (CURRENT — prefer it "
            "over anything you 'remember'):\n"
            f"{trend_brief.strip()}\n"
            "Use AT MOST 2 trend hashtags from the brief in TAGS, and only if "
            "they genuinely fit the clip — NEVER force an off-topic trend. When "
            "one of the brief's hook formats genuinely fits, shape the HOOK "
            "with it. Prefer the brief's search keywords when picking the "
            "primary keyword. Do NOT mention dates or that you used a trend "
            "brief.\n\n"
        )
    return (
        "You write social media captions and tags like a real creator on the "
        "specific platform you're targeting — not a marketer, not a robot. "
        "The text should feel native to that platform's culture.\n\n"
        "KEYWORD-FIRST METHOD (discovery in 2026 is search-driven):\n"
        "1. Pick ONE primary search keyword/query — a phrase a real user would "
        "type into this platform's search bar to find exactly this clip. Choose "
        "it from the transcript's actual subject (and the trend brief's "
        "keywords when one genuinely matches).\n"
        "2. The primary keyword MUST appear within the FIRST 50 characters of "
        "both the title and the description/caption.\n"
        "3. Write a HOOK: one on-screen overlay line (≤60 chars) shown over the "
        "clip's opening frames. It must contain or strongly imply the primary "
        "keyword — platforms OCR-index on-screen text, so this line is a "
        "ranking signal, not decoration. Use one of the brief's hook formats "
        "when it fits; never sacrifice clarity for a format.\n"
        "4. List 3-8 secondary keywords/queries in \"keywords\".\n\n"
        f"{lang_block}"
        f"{profile['guidance']}\n\n"
        f"{trend_block}"
        "HARD CONSTRAINTS (the validator WILL truncate / reject if you miss):\n"
        f"  • title_max_chars: {profile['title_max']}\n"
        f"  • description_max_chars: {profile['description_max']}\n"
        f"  • tag_count: {profile['tag_min']}-{profile['tag_max']} hashtags\n"
        "  • hook_max_chars: 60\n"
        "  • NEVER use generic discovery tags (#fyp, #foryou, #viral, "
        "#explorepage, #trending) — platforms penalize them and the validator "
        "strips them.\n\n"
        "Use the clip's transcript, video summary, and title to ground the copy in "
        "specific things that actually happen in this clip. No generic filler — "
        "every sentence should reference a concrete moment, quote, or visual.\n"
        "The TITLE must DESCRIBE this specific clip — a named subject, action, "
        "line, or stakes a scroller can picture. It is NEVER a bare timestamp, "
        "a clip number, 'Highlight', 'Untitled', 'no speech', a filename, or a "
        "vague label like 'Amazing moment'. If the clip has no dialogue, title "
        "it from what happens ON SCREEN (any '[VISUAL CONTEXT]' block below is "
        "your source); never write about the absence of speech.\n\n"
        "Return ONLY valid JSON:\n"
        '{"title": "...", "description": "...", "tags": ["#tag1", "#tag2", ...], '
        '"platform_tips": "...", "primary_keyword": "...", '
        '"keywords": ["...", "..."], "hook": "..."}'
    )


def enforce_platform_caps(seo_data: dict, platform: str,
                          brief_hashtags: "list[str] | None" = None) -> dict:
    """Trim / scrub SEO output so it fits the platform's hard caps.

    The LLM occasionally blows the title/description length even with the
    constraint spelled out in the prompt. This is the last line of defense
    before the SEO is persisted:

      * title / description / hook are trimmed to the platform cap at a word
        boundary. Descriptions get an ellipsis; titles and hooks do NOT — a
        trailing '…' on a post title reads as truncated bot output.
      * tags are normalized, case-insensitively deduped (first casing wins)
        and scrubbed against the generic-tag banlist (``seo_hygiene``) — tags
        present in today's live brief (``brief_hashtags``) are exempt.
      * deterministic trend mixing: content-specific tags come first, then up
        to 2 of the brief's hashtags (when not already present), truncated to
        the platform ``tag_max``.
      * a tag count below ``tag_min`` is LOGGED and left as-is — padding with
        junk tags is worse for ranking than shipping fewer tags.

    Returns a NEW dict — does not mutate the input.
    """
    from backend.services.seo_hygiene import clean_tags, clean_text_list
    profile = PLATFORM_PROFILES.get(platform) or PLATFORM_PROFILES["default"]
    out = dict(seo_data or {})

    def _trim(text: str, cap: int, ellipsis: bool) -> str:
        if not isinstance(text, str):
            return ""
        if len(text) <= cap:
            return text
        # Trim at the last word boundary before the cap so we don't slice
        # mid-word; fall back to a hard cut if no whitespace fits.
        trimmed = text[: cap - 1]
        last_space = trimmed.rfind(" ")
        if last_space > cap * 0.6:
            trimmed = trimmed[:last_space]
        trimmed = trimmed.rstrip()
        return trimmed + "…" if ellipsis else trimmed

    out["title"] = _trim(out.get("title") or "", profile["title_max"],
                         ellipsis=False)
    out["description"] = _trim(out.get("description") or "",
                               profile["description_max"], ellipsis=True)

    allow = set(brief_hashtags or [])
    tags = out.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    cleaned = clean_tags(tags, platform, allow=allow)

    # Deterministic trend mixing — in code, not prompt-hope: content tags
    # first, then up to 2 brief hashtags not already present.
    if brief_hashtags:
        have = {t.lstrip("#").lower() for t in cleaned}
        added = 0
        for bt in clean_tags(brief_hashtags, platform, allow=allow):
            if added >= 2:
                break
            if bt.lstrip("#").lower() in have:
                continue
            cleaned.append(bt)
            added += 1

    if len(cleaned) > profile["tag_max"]:
        cleaned = cleaned[: profile["tag_max"]]
    if len(cleaned) < profile["tag_min"]:
        logger.info(
            "SEO tags below %s tag_min (%d < %d) — leaving as-is (padding "
            "with junk tags hurts ranking more than fewer tags)",
            platform, len(cleaned), profile["tag_min"])
    out["tags"] = cleaned

    out["platform_tips"] = (out.get("platform_tips") or "").strip()
    # Keyword-first fields (defensive — providers on the old JSON shape
    # simply produce empty values here).
    out["primary_keyword"] = str(out.get("primary_keyword") or "").strip()
    out["keywords"] = clean_text_list(out.get("keywords"), cap=10)
    out["hook"] = _trim(str(out.get("hook") or "").strip(), 60, ellipsis=False)
    return out


class PromptSet(BaseModel):
    frame_analysis: str = Field(default=DEFAULT_FRAME_ANALYSIS_PROMPT)
    viral_clip_detection: str = Field(default=DEFAULT_VIRAL_CLIP_PROMPT)
    summary: str = Field(default=DEFAULT_SUMMARY_PROMPT)
    seo: str = Field(default=DEFAULT_SEO_PROMPT)


def load_prompts() -> PromptSet:
    """Load custom prompts from disk, falling back to defaults."""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, "r") as f:
                data = json.load(f)
            return PromptSet(**data)
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")
    return PromptSet()


def save_prompts(prompts: PromptSet) -> None:
    """Persist custom prompts to disk."""
    os.makedirs(os.path.dirname(PROMPTS_FILE), exist_ok=True)
    with open(PROMPTS_FILE, "w") as f:
        json.dump(prompts.model_dump(), f, indent=2)


def get_defaults() -> PromptSet:
    """Return the hardcoded default prompts (for reset functionality)."""
    return PromptSet(
        frame_analysis=DEFAULT_FRAME_ANALYSIS_PROMPT,
        viral_clip_detection=DEFAULT_VIRAL_CLIP_PROMPT,
        summary=DEFAULT_SUMMARY_PROMPT,
        seo=DEFAULT_SEO_PROMPT,
    )
