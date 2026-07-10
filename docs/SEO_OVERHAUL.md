# Live SEO Intelligence Overhaul (July 2026)

The clip SEO system (titles, descriptions, tags, captions, hooks) kept its
architecture — a daily-cached live "trend brief" injected into clip scoring
and SEO generation — but two of its three data tiers were dead code, the
hardcoded platform rules were 2023/2024-era advice that platforms now
penalize, and hooks never received trend context. This overhaul fixes all of
that and makes the platform rules **self-researching**.

Everything here is fail-soft by contract: an SEO/trend failure can never
fail or slow clip export or the analysis pipeline.

## What changed

### Bug fixes
- **pytrends tier removed** (`trend_brief.py`, `requirements.txt`): the repo
  was archived April 2025 and Google's trending-searches endpoints are gone.
  Replaced with the official **YouTube Data API v3** tier (below).
- **`_static_brief` fixed**: it used to call `format_trend_context` with a
  signature that doesn't exist (TypeError swallowed on every call) and then
  recommend `#fyp`/`#shorts` — now it loads the structured
  `backend/data/evergreen_trends.json` (durable guidance only, no dated or
  generic tags, banlist-enforced).
- **Judge is no longer trend-blind on the first job of the day**:
  `_warm_seo_intelligence` fires `get_trend_brief()` as a fire-and-forget
  task at the very start of `run_analysis` (alongside transcription), so the
  cache is warm before the clip judge reads it synchronously. Non-blocking,
  swallows all failures.
- **Genre + region plumbing wired**: the auto-SEO stage and the manual
  `/seo/{clip_id}` endpoint fetch a genre-specific brief from
  `job.summary.content_category` (two-phase: generic brief warmed at job
  start, genre brief fetched at the SEO stage — cached independently).
  `LIVE_TRENDS_REGION` now feeds the tier-1 research query and the tier-2
  YouTube `regionCode` (legacy pytrends names like `united_states` are
  mapped to ISO codes).
- **Per-platform injection**: `build_platform_seo_prompt` receives ONLY the
  requesting platform's section of the brief
  (`trend_brief.render_platform_section`) — LinkedIn no longer sees TikTok
  trends.
- **`enforce_platform_caps`**: case-insensitive tag dedup (first casing
  wins), banned-tag scrub, no ellipsis on trimmed titles (descriptions keep
  it), and the docstring no longer claims padding — a tag count below
  `tag_min` is logged and left alone (junk padding hurts ranking).
- **Transcript slicing**: the manual SEO path uses overlap semantics
  (`s.end > start and s.start < end`), matching the sidecar — segments
  straddling clip boundaries are no longer dropped.

### Structured trend brief (`backend/services/trend_brief.py`)
Pydantic schema:

```python
class PlatformTrends(BaseModel):
    hashtags: list[str]; keywords: list[str]; hook_formats: list[str]
    topics: list[str]; sounds: list[str]          # sounds: tiktok/reels only

class TrendBrief(BaseModel):
    as_of: str; region: str; genre: str
    source: str                                   # "sonar" | "youtube_api" | "static"
    platforms: dict[str, PlatformTrends]          # tiktok, youtube_shorts, reels,
                                                  # x, facebook, linkedin, youtube
```

Tiers (fail-soft, in order):
1. **Web-research LLM** (`LIVE_TRENDS_MODEL`, default `perplexity/sonar` via
   OpenRouter): one strict-JSON call per day per genre fetches ALL platforms
   (cost identical to the old prose brief). Parsed defensively (markdown
   fences stripped), pydantic-validated, every hashtag scrubbed through the
   banlist.
2. **YouTube Data API v3** (`YOUTUBE_API_KEY`, empty = skipped):
   `videos.list(chart=mostPopular, regionCode, videoCategoryId?)` via httpx.
   Mines recurring title patterns → `hook_formats`, frequency-ranked
   stopword-filtered title keywords + uploader tags → `keywords`/`hashtags`
   for the YouTube sections; other platforms keep the static tier's
   evergreen guidance.
3. **Static fallback** (`backend/data/evergreen_trends.json`): durable
   guidance only; `source: "static"`; degradation is logged at WARNING and
   labeled everywhere downstream.

Cache: 24h on disk (`/data/trends` or `.clipai/trends`), keyed
`platforms × genre × region`, storing the full structured JSON + source +
timestamp. Back-compat: `get_trend_brief()` / `read_cached_brief()` keep
their signatures and return the rendered text; `get_trend_brief_struct()` /
`read_cached_brief_struct()` return the object.

### Tag hygiene (`backend/services/seo_hygiene.py`)
- Banlist in `backend/data/banned_tags.json` (+ optional
  `banned_tags.live.json` overlay the self-researcher may write): `fyp,
  foryou, foryoupage, fy, viral, viralvideo, trending, explore, explorepage,
  reels, reelsinstagram, instagood, shorts` — `#Shorts` stays allowed on
  YouTube platforms only.
- `clean_tags(tags, platform, allow=…)`: `#`-prefix, strips
  whitespace/punctuation/emoji, ≤30 chars, case-insensitive dedup keeping
  first casing, banned-tag scrub. `allow` = today's live brief's hashtag set
  for that platform: **live data outranks the static banlist** (if the
  research tier verifies a normally-banned tag is genuinely trending this
  week, it survives; the static tier gets no such pass).
- `validate_brief(brief)`: scrubs every platform section, caps lengths
  (≤12 hashtags, ≤10 keywords, ≤8 hook formats).
- Wired into `enforce_platform_caps`, so persisted SEO is always scrubbed
  regardless of source.

### Keyword-first SEO + trend-aware hooks
- `ClipSEO` gains `primary_keyword: str`, `keywords: list[str]`,
  `hook: str` (optional with defaults — old persisted jobs round-trip
  unchanged; verified by tests).
- The per-platform prompt is keyword-first: pick ONE primary search query,
  put it in the first 50 chars of title AND description, and produce a
  ≤60-char on-screen `hook` line that carries the keyword (TikTok/Reels
  OCR-index on-screen text). JSON contract:
  `{"title","description","tags","platform_tips","primary_keyword","keywords","hook"}` —
  providers returning the old shape still parse (defaults fill in).
- **Deterministic tag mixing in code**: after generation,
  `enforce_platform_caps` composes content-specific tags first, then ≤2
  brief hashtags, truncated to `tag_max`.
- **Hook feeds the video**: the export path prefers
  `clip.seo_by_platform[platform].hook` over the legacy detection-time
  `hook_text` for the opening-frame overlay (ClipAI's OCR-layer SEO
  advantage).
- **Judge nudge**: when the brief has keywords, the judge prompt asks to
  slightly favor candidates whose first ~5 s SPEAK a searchable topic (ASR
  indexing) — never overriding hook/payoff quality.

### Source labeling (truth in output)
- Sidecar header: `Trend data:    live (sonar) · 2026-07-09` or
  `Trend data:    static fallback — set OPENROUTER_API_KEY for live trends`;
  plus `PRIMARY KEYWORD` and SEO-hook sections when present.
- Auto-SEO WebSocket status includes the source once at stage start:
  `"Generating SEO for clips… (live trends: sonar)"`.
- Degradation to static logs at WARNING.

### Self-researching platform rules
- `PLATFORM_PROFILES` is data-driven: **overlay > shipped > hardcoded**.
  - Shipped defaults: `backend/data/platform_rules.json` (verified July-2026
    values — see Appendix below).
  - Overlay: `platform_rules.live.json` in the writable state dir, written
    by `backend/services/platform_rules_research.py`.
  - Hardcoded fallback: `_PROFILES_FALLBACK` in `prompts.py` (same values).
- The researcher runs at most every `PLATFORM_RULES_REFRESH_DAYS` (default
  7), triggered lazily from the job-start warm-up, throttled by an on-disk
  attempt stamp (failures throttle too). It asks the web-research model for
  per-platform caps + whether generic tags are penalized, **sanity-validates
  every value** (`tag_max ≤ 30`, `title_max ≥ 20`, `tag_min ≤ tag_max`, … —
  bad fields dropped individually), writes the overlay (never touching
  shipped defaults), hot-reloads `PLATFORM_PROFILES` in place, and logs a
  human-readable diff.
- `GET /api/seo/intel` returns the effective rules (+ source and refresh
  date), today's trend-brief summary, and the banlist — for the frontend
  ClipSEO page ("SEO intelligence: live, refreshed 2026-07-09").

## New env vars / settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `YOUTUBE_API_KEY` | `""` | Tier-2 trend source (official YouTube Data API v3). Empty = tier skipped. |
| `LIVE_TRENDS_REGION` | `"US"` | Two-letter region for tier 1 + 2 (legacy names mapped). |
| `PLATFORM_RULES_REFRESH_DAYS` | `7` | Self-research cadence; `0` disables. |
| `LIVE_TRENDS_ENABLED` / `LIVE_TRENDS_MODEL` / `LIVE_TRENDS_CACHE_HOURS` | unchanged | Existing knobs keep working. |

## How the self-research cycle works
1. Any job starts → `_warm_seo_intelligence` fires (non-blocking):
   trend brief warmed for today + `maybe_refresh_platform_rules()`.
2. The refresher checks the on-disk stamp; if the last attempt is older than
   `PLATFORM_RULES_REFRESH_DAYS`, it asks `LIVE_TRENDS_MODEL` to verify
   current per-platform rules, validates ranges, writes
   `platform_rules.live.json`, hot-reloads `PLATFORM_PROFILES`, and logs the
   diff (e.g. `tiktok.tag_max: 5 → 4`).
3. Shipped defaults are never overwritten; deleting the overlay reverts to
   them instantly.

## How to verify it's live
- `GET /api/seo/intel` → `platform_rules_meta.source` is `live` (overlay
  present) or `shipped`; `trend_brief.source` is `sonar` / `youtube_api` /
  `static`; `trend_brief.as_of` is today.
- Any exported clip's `.txt` sidecar header shows the `Trend data:` line.
- Logs: `trend brief refreshed (source=…)` daily; `platform rules refreshed`
  weekly; a WARNING when degraded to static.

## Decisions made autonomously (ambiguity → fail-soft)
- **Live-brief tags outrank the banlist**: `validate_brief` lets a LIVE
  tier's own hashtags through `clean_tags` (the banlist exists to stop the
  generation model's stale habits, not to override verified live data). The
  static tier gets no such pass.
- **Overlay merges field-by-field**: a partial or partially-invalid research
  result contributes its good fields; bad fields are dropped individually
  rather than rejecting the whole platform.
- **Researcher `notes` never touch guidance**: the curated guidance text
  stays shipped; research notes only surface via `/api/seo/intel`.
- **`instagram` (feed) profile** aligned to the Reels 5-tag cap (the
  Dec-2025 cap applies per post, feed included).
- **Trend mixing appends then truncates**: when content tags already fill
  `tag_max`, no trend tags are forced in — content specificity wins.
- **TikTok caption cap 4,000** (Appendix A) — cross-checked via web search
  July 2026; some secondary sources still say 2,200, but TikTok's own pages
  and 2026 counters confirm 4,000.

## Platform rule defaults (verified July 2026)
- TikTok: 3–5 targeted tags (generic tags penalized), caption ≤4000,
  keyword in first 50 chars, discovery = caption + ASR + OCR.
- Instagram Reels: HARD CAP 5 hashtags (Dec 2025), 3–5 niche tags, hashtags
  in caption, caption ≤2200.
- YouTube Shorts: `#Shorts` optional, 3–5 tags, keyword front-loaded
  (title ≤100, first 100 chars of description), Shorts up to 3 min.
- YouTube long-form: title ≤70 front-loaded, keyword in first 100 chars,
  timestamps, 5–10 tags.
- X: 0–3 tags (often zero), the hook is the post, ≤280 chars.
- Facebook: 2–4 tags, conversational, first 80 chars visible on mobile.
- LinkedIn: 3–5 CamelCase industry tags, insight-led hook, white space.
