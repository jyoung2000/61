"""SEO sidecar for exported clips — CSV with headers.

When a clip is exported to MP4, ClipAI also drops a ``.csv`` next to it
carrying everything you'd paste into a social upload form: the viral
score (and its breakdown), title, suggested caption, hashtags, the
recommended platform, per-platform SEO copy, and the clip's captions /
transcript. One header row + one data row per file, so any spreadsheet
app (or a concat of many sidecars) reads it directly. The exporter calls
:func:`write_clip_seo_sidecar` once the MP4 is finalized; the frontend
then auto-downloads the ``.csv`` alongside the video.

``format_clip_seo_text`` (the legacy human-readable ``.txt`` renderer) is
kept for anything still importing it, but every user-facing surface now
produces the CSV.

Pure stdlib (no cv2 / numpy / pydantic) so it stays unit-testable and
adds no weight to the encode path. Every field is read defensively so a
``ClipCandidate`` pydantic model and a plain persisted dict both work.
"""
from __future__ import annotations

import csv
import io
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Platform slug → display label. Mirrors the values ``ClipCandidate.platform``
# and the ``seo_by_platform`` keys can take.
_PLATFORM_NAMES = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "shorts": "YouTube Shorts",
    "youtube": "YouTube",
    "reels": "Instagram Reels",
    "instagram": "Instagram",
    "x": "X (Twitter)",
    "twitter": "X (Twitter)",
    "facebook": "Facebook",
    "both": "TikTok & YouTube Shorts",
}

_RULE = "─" * 78   # ────────
_HRULE = "═" * 78  # ════════


def _g(obj, key, default=None):
    """Read ``key`` from a pydantic model OR a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _platform_label(slug) -> str:
    if not slug:
        return ""
    s = str(slug).strip().lower()
    return _PLATFORM_NAMES.get(s, str(slug).strip())


def _fmt_ts(seconds) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "0:00"
    total = max(0, total)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _clean_tags(tags) -> list[str]:
    return [str(t).strip().lstrip("#").strip() for t in (tags or []) if str(t).strip()]


def _hashtagify(tags) -> list[str]:
    out = []
    for t in _clean_tags(tags):
        token = "".join(t.split())  # collapse whitespace into a single tag token
        if token:
            out.append("#" + token)
    return out


def find_clip(job, clip_id):
    """Locate the ``ClipCandidate`` (or dict) with ``id == clip_id``."""
    for c in _g(job, "clips", []) or []:
        if _g(c, "id") == clip_id:
            return c
    return None


def sidecar_path_for(output_path: str) -> str:
    """``.../[1080P] Title.mp4`` → ``.../[1080P] Title.csv``."""
    base, _ext = os.path.splitext(output_path)
    return base + ".csv"


def _section(title: str, body) -> str:
    body = ("" if body is None else str(body)).strip()
    if not body:
        return ""
    return f"{_RULE}\n{title}\n{_RULE}\n{body}\n\n"


def _trend_source_line(job) -> str:
    """One header line saying where today's trend data came from — truth in
    output: a static fallback must never masquerade as live. '' when the
    trend system is disabled or unreadable (best-effort, never raises)."""
    try:
        from backend.services.trend_brief import read_cached_brief_struct
        genre = ""
        summary = _g(job, "summary")
        if summary is not None:
            genre = (_g(summary, "content_category", "") or "").strip()
        brief = read_cached_brief_struct("both", genre)
        if brief is None and genre:
            brief = read_cached_brief_struct("both", "")
        if brief is None:
            return ""
        if (brief.source or "static") == "static":
            return ("static fallback — set OPENROUTER_API_KEY for live trends")
        return f"live ({brief.source}) · {brief.as_of}"
    except Exception:
        return ""


def _seo_record_for_platform(clip):
    """The clip's per-platform SEO record matching its declared platform
    (dict or pydantic), or None."""
    by_plat = _g(clip, "seo_by_platform", {}) or {}
    plat = str(_g(clip, "platform", "") or "").strip().lower()
    aliases = {"both": "tiktok", "shorts": "youtube_shorts", "twitter": "x",
               "instagram_reels": "reels"}
    plat = aliases.get(plat, plat)
    return by_plat.get(plat) if isinstance(by_plat, dict) else None


def _captions_for_range(transcript, start: float, end: float) -> list[str]:
    """Clip-relative caption lines whose segments overlap ``[start, end]``."""
    lines = []
    for seg in transcript or []:
        s = _g(seg, "start")
        e = _g(seg, "end")
        text = (_g(seg, "text", "") or "").strip()
        if s is None or e is None or not text:
            continue
        try:
            s = float(s)
            e = float(e)
        except (TypeError, ValueError):
            continue
        if e <= start or s >= end:  # no overlap with the clip window
            continue
        rel = max(0.0, s - start)
        speaker = (_g(seg, "speaker", "") or "").strip()
        prefix = f"[{_fmt_ts(rel)}]"
        lines.append(f"{prefix} {speaker}: {text}" if speaker else f"{prefix} {text}")
    return lines


def format_clip_seo_text(job, clip, output_basename: str,
                         export_info: dict | None = None,
                         transcript=None) -> str:
    """Render the full sidecar text for one exported clip."""
    export_info = export_info or {}

    title = (_g(clip, "title", "") or export_info.get("clip_title") or "").strip()

    # ── Header ──
    head = [_HRULE, "ClipAI — Clip SEO & Export Info", _HRULE,
            f"Clip file:     {output_basename}"]
    src = (_g(job, "filename", "") or "").strip()
    if src:
        head.append(f"Source video:  {src}")
    job_id = _g(job, "job_id", "") or export_info.get("job_id", "")
    if job_id:
        head.append(f"Job ID:        {job_id}")
    head.append("Generated:     " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    trend_src = _trend_source_line(job)
    if trend_src:
        head.append(f"Trend data:    {trend_src}")
    head.append("")
    head.append("")

    body = ""

    # ── Title ──
    seo_title = (_g(clip, "seo_title", "") or "").strip()
    title_block = title or "(untitled clip)"
    if seo_title and seo_title != title:
        title_block += f"\n\nSEO title: {seo_title}"
    body += _section("TITLE", title_block)

    # ── Viral score ──
    vs = _g(clip, "viral_score")
    if vs is not None:
        vlines = [f"Overall: {vs} / 100"]
        comp = _g(clip, "viral_score_composite")
        if comp is not None and comp != vs:
            vlines.append(f"Composite: {comp} / 100")
        axes = []
        for label, key in (("Hook", "hook_score"), ("Flow", "flow_score"),
                           ("Value", "value_score"), ("Trend", "trend_score")):
            av = _g(clip, key)
            if av:
                axes.append(f"{label} {av}")
        if axes:
            vlines.append("Breakdown: " + " · ".join(axes))
        reason = (_g(clip, "viral_score_reasoning", "") or "").strip()
        if reason:
            vlines += ["", reason]
        body += _section("VIRAL SCORE", "\n".join(vlines))

    # ── Recommended platform ──
    body += _section("RECOMMENDED PLATFORM", _platform_label(_g(clip, "platform", "")))

    # ── Social caption ──
    body += _section("CAPTION (social post)", _g(clip, "suggested_caption", ""))

    # ── Hook ── (prefer the trend-aware SEO hook — it carries the primary
    # search keyword and is what the export overlays on the opening frames —
    # falling back to the legacy detection-time hook_text)
    seo_rec = _seo_record_for_platform(clip)
    seo_hook = (_g(seo_rec, "hook", "") or "").strip()
    body += _section("HOOK", seo_hook
                     or _g(clip, "hook_text", "")
                     or export_info.get("hook_text", ""))

    # ── Primary keyword ── (the search query this clip is optimized to rank
    # for; secondary keywords on the next line when present)
    pk = (_g(seo_rec, "primary_keyword", "") or "").strip()
    if pk:
        kws = [str(k).strip() for k in (_g(seo_rec, "keywords", []) or [])
               if str(k).strip()]
        pk_block = pk
        if kws:
            pk_block += "\nSecondary: " + ", ".join(kws)
        body += _section("PRIMARY KEYWORD", pk_block)

    # ── Tags / hashtags ──
    tags = _clean_tags(_g(clip, "seo_tags", []))
    if tags:
        body += _section("TAGS / HASHTAGS",
                         ", ".join(tags) + "\n" + " ".join(_hashtagify(tags)))

    # ── Description ──
    body += _section("DESCRIPTION", _g(clip, "seo_description", ""))

    # ── Platform tips ──
    body += _section("PLATFORM TIPS", _g(clip, "seo_platform_tips", ""))

    # ── Why this works ──
    body += _section("WHY THIS WORKS", _g(clip, "why_this_works", ""))

    # ── Per-platform SEO ──
    by_plat = _g(clip, "seo_by_platform", {}) or {}
    if by_plat:
        blocks = []
        for slug, rec in by_plat.items():
            sub = [f"▸ {_platform_label(slug) or slug}"]
            t = (_g(rec, "title", "") or "").strip()
            d = (_g(rec, "description", "") or "").strip()
            rtags = _clean_tags(_g(rec, "tags", []))
            tips = (_g(rec, "platform_tips", "") or "").strip()
            pk = (_g(rec, "primary_keyword", "") or "").strip()
            hk = (_g(rec, "hook", "") or "").strip()
            if t:
                sub.append(f"  Title:       {t}")
            if pk:
                sub.append(f"  Keyword:     {pk}")
            if hk:
                sub.append(f"  Hook:        {hk}")
            if d:
                sub.append(f"  Description: {d}")
            if rtags:
                sub.append(f"  Tags:        {', '.join(rtags)}")
            if tips:
                sub.append(f"  Tips:        {tips}")
            if len(sub) > 1:
                blocks.append("\n".join(sub))
        body += _section("PER-PLATFORM SEO", "\n\n".join(blocks))

    # ── Export details ──
    start = export_info.get("start")
    end = export_info.get("end")
    elines = []
    q = export_info.get("export_quality")
    if q:
        elines.append(f"Quality:       {q}")
    elines.append(f"Aspect ratio:  {export_info.get('aspect_ratio') or 'original'}")
    if start is not None and end is not None:
        try:
            dur = max(0.0, float(end) - float(start))
            elines.append(f"Clip range:    {_fmt_ts(start)} → {_fmt_ts(end)}  ({dur:.1f}s)")
        except (TypeError, ValueError):
            pass
    se = export_info.get("subtitles_enabled")
    if se is not None:
        elines.append(f"Subtitles:     {'on' if se else 'off'}")
    body += _section("EXPORT DETAILS", "\n".join(elines))

    # ── Captions / transcript ──
    if start is not None and end is not None:
        try:
            cap = _captions_for_range(transcript, float(start), float(end))
        except (TypeError, ValueError):
            cap = []
        if cap:
            body += _section("CAPTIONS / TRANSCRIPT", "\n".join(cap))

    return "\n".join(head) + body + _HRULE + "\nGenerated by ClipAI\n"


def format_clip_seo_csv(job, clip, output_basename: str,
                        export_info: dict | None = None,
                        transcript=None) -> str:
    """Render the SEO sidecar as CSV: one header row + one data row.

    Headers cover every SEO field (core + per-platform, flattened as
    ``<platform>_title`` etc. for each platform the clip carries), plus the
    export details and the clip's caption lines. List fields are joined with
    ``"; "``; caption lines with ``" | "`` — every row stays single-line so
    naive CSV consumers cope, while the ``csv`` module still quotes anything
    that needs it."""
    export_info = export_info or {}

    def _s(v) -> str:
        return ("" if v is None else str(v)).strip()

    def _join(items, sep="; ") -> str:
        return sep.join(_s(x) for x in (items or []) if _s(x))

    title = _s(_g(clip, "title", "")) or _s(export_info.get("clip_title"))
    start = export_info.get("start")
    end = export_info.get("end")
    if start is None:
        start = _g(clip, "start_time")
    if end is None:
        end = _g(clip, "end_time")
    try:
        dur = f"{max(0.0, float(end) - float(start)):.1f}"
    except (TypeError, ValueError):
        dur = ""

    tags = _clean_tags(_g(clip, "seo_tags", []))
    seo_rec = _seo_record_for_platform(clip)

    cols: list[tuple[str, str]] = [
        ("clip_file", output_basename),
        ("source_video", _s(_g(job, "filename", ""))),
        ("job_id", _s(_g(job, "job_id", "") or export_info.get("job_id", ""))),
        ("generated_utc",
         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")),
        ("clip_title", title),
        ("seo_title", _s(_g(clip, "seo_title", ""))),
        ("viral_score", _s(_g(clip, "viral_score"))),
        ("hook_score", _s(_g(clip, "hook_score"))),
        ("flow_score", _s(_g(clip, "flow_score"))),
        ("value_score", _s(_g(clip, "value_score"))),
        ("trend_score", _s(_g(clip, "trend_score"))),
        ("viral_score_reasoning", _s(_g(clip, "viral_score_reasoning", ""))),
        ("recommended_platform", _platform_label(_g(clip, "platform", ""))),
        ("social_caption", _s(_g(clip, "suggested_caption", ""))),
        ("hook", _s(_g(seo_rec, "hook", "")) or _s(_g(clip, "hook_text", ""))
         or _s(export_info.get("hook_text", ""))),
        ("primary_keyword", _s(_g(seo_rec, "primary_keyword", ""))),
        ("secondary_keywords", _join(_g(seo_rec, "keywords", []))),
        ("tags", _join(tags)),
        ("hashtags", " ".join(_hashtagify(tags))),
        ("description", _s(_g(clip, "seo_description", ""))),
        ("platform_tips", _s(_g(clip, "seo_platform_tips", ""))),
        ("why_this_works", _s(_g(clip, "why_this_works", ""))),
        ("export_quality", _s(export_info.get("export_quality"))),
        ("aspect_ratio", _s(export_info.get("aspect_ratio")) or "original"),
        ("clip_start", _fmt_ts(start) if start is not None else ""),
        ("clip_end", _fmt_ts(end) if end is not None else ""),
        ("clip_duration_s", dur),
        ("subtitles_enabled",
         "" if export_info.get("subtitles_enabled") is None
         else ("yes" if export_info.get("subtitles_enabled") else "no")),
        ("trend_data", _trend_source_line(job)),
    ]

    # Per-platform SEO, flattened into stable-ordered columns.
    by_plat = _g(clip, "seo_by_platform", {}) or {}
    if isinstance(by_plat, dict):
        for slug in sorted(by_plat):
            rec = by_plat[slug]
            key = str(slug).strip().lower().replace("-", "_")
            cols += [
                (f"{key}_title", _s(_g(rec, "title", ""))),
                (f"{key}_primary_keyword", _s(_g(rec, "primary_keyword", ""))),
                (f"{key}_hook", _s(_g(rec, "hook", ""))),
                (f"{key}_description", _s(_g(rec, "description", ""))),
                (f"{key}_tags", _join(_clean_tags(_g(rec, "tags", [])))),
                (f"{key}_platform_tips", _s(_g(rec, "platform_tips", ""))),
            ]

    # Captions within the exported window, single-line joined.
    cap_lines: list[str] = []
    if start is not None and end is not None:
        try:
            cap_lines = _captions_for_range(transcript, float(start), float(end))
        except (TypeError, ValueError):
            cap_lines = []
    cols.append(("captions_transcript", " | ".join(cap_lines)))

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([c for c, _ in cols])
    w.writerow([v for _, v in cols])
    return buf.getvalue()


def write_clip_seo_sidecar(job, clip_id, output_path: str,
                           export_info: dict | None = None) -> str | None:
    """Write a ``.csv`` SEO sidecar next to ``output_path``.

    Best-effort: returns the sidecar path on success, ``None`` on any
    failure (a sidecar problem must never fail the actual clip export).
    Captions are taken from the job's translated transcript when present,
    else the source transcript, filtered to the clip window.
    """
    try:
        clip = find_clip(job, clip_id)
        transcript = _g(job, "translated_transcript", None) or _g(job, "transcript", None) or []
        text = format_clip_seo_csv(
            job, clip, os.path.basename(output_path),
            export_info=export_info, transcript=transcript,
        )
        path = sidecar_path_for(output_path)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        logger.info("Wrote clip SEO sidecar: %s", path)
        return path
    except Exception as e:  # pragma: no cover — best-effort, never break export
        logger.warning("Failed to write clip SEO sidecar for %s: %s", output_path, e)
        return None
