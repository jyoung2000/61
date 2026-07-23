"""Conform ClipAI's subtitle track to an operator-supplied REFERENCE transcript
(e.g. YouTube's official captions), so the shipped words, timing and cue
segmentation match a known-good source instead of ClipAI's independent
best-effort.

The reference is the ground truth the operator wants to match. When supplied
(Settings → Transcription → Reference subtitles), this pass either:

  * ``adopt`` (default) — replaces ClipAI's cues wholesale with the reference
    cues (their exact wording, timing and one-thought-per-cue segmentation),
    inheriting each cue's SPEAKER from the ClipAI cue it overlaps most (so
    ClipAI's diarization / speaker colours survive). This is the surest
    "match YouTube" — words + timing + readability all come from the reference.

  * ``timing`` — keeps ClipAI's words but snaps each cue's start/end onto the
    nearest reference boundary (fixes linger / early-appearance without
    trusting the reference's wording).

Fail-soft everywhere: an empty / unparseable / implausibly-short reference is a
no-op and ClipAI's own track ships. Pure stdlib so it stays unit-testable.
"""
from __future__ import annotations

import re

# A reference this sparse can't be trusted to cover the video — ignore it
# rather than gut a full transcript with a handful of stray lines.
_MIN_REFERENCE_CUES = 8

_SRT_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})")
# A bare leading timestamp on a line: "0:30", "1:02:33", "[0:30]", "00:30".
_LINE_TS = re.compile(r"^\[?\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?\s*\]?\s*")
_SPEAKER_PREFIX = re.compile(r"^\s*(?:speaker\s*\d+|[A-Z][a-z]+)\s*:\s*", re.IGNORECASE)


def _hms(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + (int((ms or "0").ljust(3, "0")[:3]) / 1000.0)


def parse_reference(text: str) -> list:
    """Parse SRT / VTT / plain-timestamped captions into ``[{start, end, text}]``.

    Handles: SRT and VTT blocks (``HH:MM:SS,mmm --> …``), and plain lines that
    begin with a timestamp (``0:30 text`` / ``[0:30] text`` — YouTube's
    copy-paste + ClipAI's own .txt export). For start-only formats each cue's
    end is the next cue's start. Returns [] on anything unparseable."""
    if not text or not text.strip():
        return []
    raw = text.replace("\r\n", "\n").replace("\r", "\n")

    # ── SRT / VTT: arrow-delimited spans ──
    cues: list = []
    if "-->" in raw:
        blocks = re.split(r"\n\s*\n", raw)
        for blk in blocks:
            m = _SRT_TS.search(blk)
            if not m:
                continue
            start = _hms(m.group(1), m.group(2), m.group(3), m.group(4))
            end = _hms(m.group(5), m.group(6), m.group(7), m.group(8))
            # Text = every line after the timestamp line, minus a leading index.
            lines = [ln for ln in blk.split("\n")]
            body = []
            after_ts = False
            for ln in lines:
                if _SRT_TS.search(ln):
                    after_ts = True
                    continue
                if after_ts and ln.strip():
                    body.append(ln.strip())
            body_txt = " ".join(body).strip()
            body_txt = _SPEAKER_PREFIX.sub("", body_txt)
            if body_txt and end > start:
                cues.append({"start": round(start, 3), "end": round(end, 3), "text": body_txt})
        return cues if len(cues) >= _MIN_REFERENCE_CUES else []

    # ── Plain timestamped lines ("[0:30] text" / "0:30 text") ──
    starts: list = []
    for ln in raw.split("\n"):
        m = _LINE_TS.match(ln)
        if not m:
            continue
        h, mnt, sec, ms = m.group(1), m.group(2), m.group(3), m.group(4)
        start = _hms(h or 0, mnt, sec, ms)
        body = _LINE_TS.sub("", ln, count=1).strip()
        body = _SPEAKER_PREFIX.sub("", body).strip()
        if body:
            starts.append((start, body))
    if len(starts) < _MIN_REFERENCE_CUES:
        return []
    starts.sort(key=lambda p: p[0])
    for i, (st, body) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else st + 4.0
        if end <= st:
            end = st + 1.0
        cues.append({"start": round(st, 3), "end": round(min(end, st + 12.0), 3), "text": body})
    return cues


def parse_reference_lines(text: str) -> list:
    """Timestamp-LESS reference (e.g. DownloadYoutubeSubtitles.com plain text):
    one caption per non-empty line/paragraph, no times. Returns ``[str, …]``
    (or [] when the text actually carries timestamps — use parse_reference)."""
    if not text or not text.strip():
        return []
    raw = text.replace("\r\n", "\n").replace("\r", "\n")
    if "-->" in raw or any(_LINE_TS.match(ln) and _LINE_TS.match(ln).group(0).strip()
                           for ln in raw.split("\n") if ln.strip()):
        return []
    # Captions arrive as blank-line-separated blocks whose inner newlines are
    # display line-wraps — join within a block, split on blank lines.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw)]
    lines = [" ".join(b.split()) for b in blocks if b.strip()]
    lines = [_SPEAKER_PREFIX.sub("", ln).strip() for ln in lines]
    return [ln for ln in lines if ln] if len(lines) >= _MIN_REFERENCE_CUES else []


def _norm_tokens(s: str) -> set:
    return set(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def _sim(a: str, b: str) -> float:
    """Token-overlap similarity — cheap, order-free, robust to rephrasing."""
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def align_lines_to_cues(clip_rows: list, ref_lines: list) -> dict:
    """Monotone alignment ClipAI cue index → reference line index.

    Order-preserving DP (both tracks tell the same story in the same order);
    a cue may skip lines and vice versa; only pairs above a similarity floor
    are adopted. O(cues × lines) — fine at a few hundred each."""
    n, m = len(clip_rows), len(ref_lines)
    if not n or not m:
        return {}
    FLOOR = 0.34
    # dp[i][j] = best score aligning first i cues with first j lines.
    NEG = float("-inf")
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 1=match, 2=skip cue, 3=skip line
    for i in range(1, n + 1):
        txt_i = clip_rows[i - 1].get("text") or ""
        for j in range(1, m + 1):
            s = _sim(txt_i, ref_lines[j - 1])
            match = dp[i - 1][j - 1] + (s if s >= FLOOR else -0.05)
            skip_c = dp[i - 1][j]
            skip_l = dp[i][j - 1]
            best = max(match, skip_c, skip_l)
            dp[i][j] = best
            bt[i][j] = 1 if best == match else (2 if best == skip_c else 3)
    # Backtrack; keep only genuinely-similar pairs.
    out: dict = {}
    i, j = n, m
    while i > 0 and j > 0:
        step = bt[i][j]
        if step == 1:
            if _sim(clip_rows[i - 1].get("text") or "", ref_lines[j - 1]) >= FLOOR:
                out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == 2:
            i -= 1
        else:
            j -= 1
    return out


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_for(ref_start, ref_end, clip_rows) -> str:
    """Speaker of the ClipAI cue overlapping ``[ref_start, ref_end]`` most."""
    best_spk, best_ov = "", 0.0
    for r in clip_rows:
        ov = _overlap(ref_start, ref_end, float(r.get("start") or 0.0),
                      float(r.get("end") or 0.0))
        if ov > best_ov:
            best_ov, best_spk = ov, (r.get("speaker") or "")
    return best_spk


def conform_to_reference(segments, reference_text: str, mode: str = "adopt"):
    """Conform ClipAI cues to a reference transcript. Returns ``(rows, changed)``.

    ``adopt`` → reference cues (words + timing + segmentation), speakers mapped
    from ClipAI by time overlap. ``timing`` → ClipAI words with start/end snapped
    to the nearest overlapping reference cue. No-op + ``changed=False`` when the
    reference is empty / unparseable / too short to trust. Fail-soft."""
    try:
        rows = [dict(r) if isinstance(r, dict) else
                (r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r))
                for r in (segments or [])]
        ref = parse_reference(reference_text or "")
        if len(ref) < _MIN_REFERENCE_CUES:
            # Timestamp-less reference (plain caption lines): adopt the
            # reference WORDING onto ClipAI's own cue timing via monotone
            # fuzzy alignment. Timing/segmentation stay ClipAI's (there are
            # no reference times to take); unmatched cues keep their text.
            lines = parse_reference_lines(reference_text or "")
            if lines and rows:
                amap = align_lines_to_cues(rows, lines)
                # Trust the alignment only when a real fraction of cues found a
                # confident partner — a handful of coincidental matches must not
                # rewrite a transcript the reference doesn't actually cover.
                if len(amap) >= max(5, len(rows) // 5):
                    out = []
                    changed = False
                    for i, r in enumerate(rows):
                        nr = dict(r)
                        j = amap.get(i)
                        if j is not None and lines[j] != (nr.get("text") or ""):
                            nr["text"] = lines[j]
                            nr["words"] = []
                            changed = True
                        out.append(nr)
                    return out, changed
            return rows, False
        if not rows:
            return rows, False

        if (mode or "adopt").strip().lower() == "timing":
            # Snap each ClipAI cue to its best-overlapping reference boundary.
            out = []
            snapped = 0
            for r in rows:
                s, e = float(r.get("start") or 0.0), float(r.get("end") or 0.0)
                best, best_ov = None, 0.0
                for rc in ref:
                    ov = _overlap(s, e, rc["start"], rc["end"])
                    if ov > best_ov:
                        best_ov, best = ov, rc
                nr = dict(r)
                if best is not None and best_ov > 0:
                    if abs(nr.get("start", s) - best["start"]) > 0.05 or \
                       abs(nr.get("end", e) - best["end"]) > 0.05:
                        snapped += 1
                    nr["start"], nr["end"] = best["start"], best["end"]
                out.append(nr)
            return out, snapped > 0

        # ── adopt: the reference IS the transcript; inherit speakers by overlap ──
        out = []
        for rc in ref:
            out.append({
                "start": rc["start"],
                "end": rc["end"],
                "text": rc["text"],
                "speaker": _speaker_for(rc["start"], rc["end"], rows),
                "words": [],  # reference wording ≠ ClipAI word timings
            })
        return out, True
    except Exception:
        return list(segments or []), False
