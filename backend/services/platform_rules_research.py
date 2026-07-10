"""Self-researching platform SEO rules (Part 6 of the SEO overhaul).

Platform rules rot: the hashtag counts, char limits and "what the algorithm
rewards" advice that were correct in 2023 are penalized in 2026 (generic
discovery tags, hashtag stuffing, mandatory #Shorts). Instead of shipping
another snapshot that will rot the same way, ClipAI refreshes its OWN
knowledge on a schedule: a weekly web-research call verifies each platform's
current caps and writes a ``platform_rules.live.json`` OVERLAY (never touching
the shipped defaults). Loader precedence in ``prompts.py``:
overlay > shipped ``backend/data/platform_rules.json`` > hardcoded dict.

Every value coming back from the model is sanity-range-validated
(``prompts.validate_platform_rule`` — e.g. ``tag_max > 30`` or
``title_max < 20`` is rejected field-by-field), so a hallucinated number can
never corrupt the live rules. Fail-soft everywhere: any failure leaves the
current rules untouched and is retried after the throttle window.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Platforms the researcher verifies (the canonical profile slugs).
_RESEARCH_PLATFORMS = (
    "tiktok", "youtube_shorts", "reels", "instagram", "youtube", "x",
    "facebook", "linkedin",
)

_STAMP_NAME = "platform_rules.refresh_stamp.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _stamp_path() -> str:
    from backend.services.seo_hygiene import writable_state_dir
    return os.path.join(writable_state_dir(), _STAMP_NAME)


def _last_attempt_ts() -> float:
    """Timestamp of the last refresh ATTEMPT (success or failure) — failures
    are throttled too, so a dead key can't fire a research call per job."""
    try:
        with open(_stamp_path()) as f:
            return float(json.load(f).get("ts", 0))
    except Exception:
        return 0.0


def _write_stamp(ok: bool, detail: str = "") -> None:
    try:
        with open(_stamp_path(), "w") as f:
            json.dump({"ts": time.time(), "date": _today(), "ok": ok,
                       "detail": detail[:200]}, f)
    except Exception as e:
        logger.debug("platform rules refresh stamp write failed: %s", e)


def _research_prompt() -> str:
    plat_list = ", ".join(_RESEARCH_PLATFORMS)
    return (
        f"You are a social-platform SEO auditor. Today is {_today()}. Using "
        "CURRENT web data (platform help pages, creator documentation, and "
        "current social-SEO reporting), verify the CURRENT posting rules for "
        f"each of these platforms: {plat_list}.\n\n"
        "For each platform report:\n"
        "- tag_min / tag_max: the currently recommended hashtag count range "
        "(hard platform caps win over soft advice — e.g. if a platform "
        "enforces a 5-hashtag cap, tag_max is 5).\n"
        "- title_max: current title / first-line character limit.\n"
        "- description_max: current caption/description character limit.\n"
        "- generic_tags_penalized: whether generic discovery tags (#fyp, "
        "#viral, #explorepage) are currently penalized or ignored there.\n"
        "- notes: ONE sentence on the platform's current keyword/caption "
        "weighting (what actually drives discovery right now).\n\n"
        "Return STRICT JSON only — no markdown fences, no commentary:\n"
        '{"tiktok": {"tag_min": N, "tag_max": N, "title_max": N, '
        '"description_max": N, "generic_tags_penalized": true, '
        '"notes": "..."}, ...}\n'
        "Include every platform listed above. Report only values you can "
        "support with current data; omit a field you cannot verify."
    )


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    return t


def _diff_rules(current: dict, proposed: dict) -> list[str]:
    """Human-readable field diffs between the effective profiles and the
    validated research output."""
    lines = []
    for plat, rule in proposed.items():
        cur = current.get(plat) or {}
        for key, val in rule.items():
            if key in ("label", "guidance", "notes"):
                continue
            old = cur.get(key)
            if old != val:
                lines.append(f"{plat}.{key}: {old} → {val}")
    return lines


async def maybe_refresh_platform_rules(force: bool = False) -> bool:
    """Weekly (``PLATFORM_RULES_REFRESH_DAYS``) verification of the platform
    rules via the web-research model. Writes the validated diff to the
    ``platform_rules.live.json`` overlay and hot-reloads ``PLATFORM_PROFILES``.

    Called lazily from the same place the trend brief is warmed (job start),
    throttled by the on-disk attempt stamp. Returns True only when an overlay
    was written. Fail-soft: never raises.
    """
    try:
        from backend.config import settings
        key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
        if not key:
            return False
        days = float(getattr(settings, "PLATFORM_RULES_REFRESH_DAYS", 7) or 7)
        if days <= 0:
            return False
        if not force and (time.time() - _last_attempt_ts()) < days * 86400:
            return False
        _write_stamp(ok=False, detail="attempt started")  # throttle even on crash

        model = (getattr(settings, "LIVE_TRENDS_MODEL", "")
                 or "perplexity/sonar").strip()
        import httpx
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}",
                         "HTTP-Referer": "http://localhost:1353",
                         "X-Title": "ClipAI"},
                json={"model": model,
                      "messages": [{"role": "user", "content": _research_prompt()}],
                      "max_tokens": 1500, "temperature": 0.1},
            )
            resp.raise_for_status()
            raw = (resp.json().get("choices") or [{}])[0].get(
                "message", {}).get("content", "")

        try:
            data = json.loads(_strip_fences(raw))
        except Exception as e:
            logger.info("platform rules research: JSON did not parse (%s)", e)
            _write_stamp(ok=False, detail=f"parse: {e}")
            return False
        if not isinstance(data, dict):
            _write_stamp(ok=False, detail="non-dict response")
            return False

        from backend.services import prompts as P
        validated: dict[str, dict] = {}
        rejected: list[str] = []
        for plat in _RESEARCH_PLATFORMS:
            rule = data.get(plat)
            if not isinstance(rule, dict):
                continue
            ok_fields = P.validate_platform_rule(rule)
            # notes ride along for the /api/seo/intel display but never touch
            # the guidance text (that stays shipped/curated).
            notes = rule.get("notes")
            if isinstance(notes, str) and notes.strip():
                ok_fields["notes"] = notes.strip()[:300]
            dropped = {k for k in rule
                       if k in P._RULE_BOUNDS and k not in ok_fields}
            if dropped:
                rejected.append(f"{plat}: {sorted(dropped)}")
            if ok_fields:
                validated[plat] = ok_fields
        if rejected:
            logger.warning("platform rules research: rejected out-of-range "
                           "fields — %s", "; ".join(rejected))
        if not validated:
            _write_stamp(ok=False, detail="nothing validated")
            return False

        diff = _diff_rules(P.PLATFORM_PROFILES, validated)
        overlay_path = P.platform_rules_overlay_path()
        with open(overlay_path, "w") as f:
            json.dump({"date": _today(), "ts": time.time(),
                       "model": model, "platforms": validated}, f, indent=2)
        P.reload_platform_rules()
        if diff:
            logger.info("platform rules refreshed (live overlay) — changes:\n  "
                        + "\n  ".join(diff))
        else:
            logger.info("platform rules refreshed — current rules confirmed, "
                        "no changes")
        _write_stamp(ok=True, detail=f"{len(validated)} platforms, "
                                     f"{len(diff)} changes")
        return True
    except Exception as e:
        logger.info("platform rules research skipped (%s)", e)
        try:
            _write_stamp(ok=False, detail=str(e)[:200])
        except Exception:
            pass
        return False


def research_status() -> dict:
    """Last refresh attempt info for /api/seo/intel. Never raises."""
    try:
        with open(_stamp_path()) as f:
            stamp = json.load(f)
        return {"last_attempt": stamp.get("date", ""),
                "ok": bool(stamp.get("ok")),
                "detail": stamp.get("detail", "")}
    except Exception:
        return {"last_attempt": "", "ok": False, "detail": "never run"}
