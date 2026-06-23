"""Human-readable SEO sidecar for exported clips.

When a clip is exported to MP4, ClipAI also drops a ``.txt`` next to it
carrying everything you'd paste into a social upload form: the viral
score (and its breakdown), title, suggested caption, hashtags, the
recommended platform, per-platform SEO copy, and the clip's captions /
transcript. The exporter calls :func:`write_clip_seo_sidecar` once the
MP4 is finalized; the frontend then auto-downloads the ``.txt`` alongside
the video.

Pure stdlib (no cv2 / numpy / pydantic) so it stays unit-testable and
adds no weight to the encode path. Every field is read defensively so a
``ClipCandidate`` pydantic model and a plain persisted dict both work.
"""
from __future__ import annotations

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
    """``.../[1080P] Title.mp4`` → ``.../[1080P] Title.txt``."""
    base, _ext = os.path.splitext(output_path)
    return base + ".txt"


def _section(title: str, body) -> str:
    body = ("" if body is None else str(body)).strip()
    if not body:
        return ""
    return f"{_RULE}\n{title}\n{_RULE}\n{body}\n\n"


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

    # ── Hook ──
    body += _section("HOOK", _g(clip, "hook_text", "") or export_info.get("hook_text", ""))

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
            if t:
                sub.append(f"  Title:       {t}")
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


def write_clip_seo_sidecar(job, clip_id, output_path: str,
                           export_info: dict | None = None) -> str | None:
    """Write a ``.txt`` SEO sidecar next to ``output_path``.

    Best-effort: returns the sidecar path on success, ``None`` on any
    failure (a sidecar problem must never fail the actual clip export).
    Captions are taken from the job's translated transcript when present,
    else the source transcript, filtered to the clip window.
    """
    try:
        clip = find_clip(job, clip_id)
        transcript = _g(job, "translated_transcript", None) or _g(job, "transcript", None) or []
        text = format_clip_seo_text(
            job, clip, os.path.basename(output_path),
            export_info=export_info, transcript=transcript,
        )
        path = sidecar_path_for(output_path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Wrote clip SEO sidecar: %s", path)
        return path
    except Exception as e:  # pragma: no cover — best-effort, never break export
        logger.warning("Failed to write clip SEO sidecar for %s: %s", output_path, e)
        return None
