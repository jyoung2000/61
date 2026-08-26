"""Profanity censor — mask curse words in burned subtitles and beep them in
the exported audio.

Two halves, both driven by the same user-configurable block list
(Settings > Profanity Censor; ``CENSOR_WORDS``):

  * TEXT: every blocked word in the burned subtitle track is masked with the
    user's chosen symbol, keeping the first and last letter ("shit" → "s**t").
    Applied to the transcript segments (including per-word karaoke text)
    right before ASS generation, so the active-word highlight can never
    flash the unmasked word.

  * AUDIO: the word's time window is muted and a beep (1 kHz tone, or the
    user's uploaded sound) is mixed over it. Runs as a dedicated post-pass
    on the finished export — video stream-copied, audio re-encoded once —
    so the fragile main filter graph (crop/overlay/speed/subtitles) is never
    touched. Word-level Whisper timestamps drive the window when present;
    otherwise the word's position is interpolated by character weight within
    its cue.

The post-pass works in OUTPUT time: intervals found in source time are
remapped through the export's trim + global/per-segment speed settings with
the same walk ``_compute_output_duration`` uses, so beeps stay on the word
even in speed-ramped exports.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Built-in block list used while CENSOR_WORDS is empty / "default". Common
# English profanity plus the variants word-boundary matching would miss.
# A censor feature has to spell these out — the list is exactly the point.
DEFAULT_CENSOR_WORDS = [
    "fuck", "fucking", "fucked", "fucker", "fuckers", "motherfucker",
    "motherfucking", "shit", "shitty", "bullshit", "shits", "ass",
    "asshole", "assholes", "bitch", "bitches", "bastard", "bastards",
    "dick", "dicks", "cock", "cocks", "pussy", "cunt", "cunts",
    "goddamn", "goddammit", "damn", "dammit", "piss", "pissed",
    "whore", "slut", "douche", "douchebag", "jackass", "dumbass",
]

# Where a user-uploaded beep replacement lives (mount-backed so it survives
# container rebuilds). Extension varies with the uploaded file.
CENSOR_SOUND_DIR = "/data/config"
CENSOR_SOUND_BASENAME = "censor_sound"
CENSOR_SOUND_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac")

# Padding applied around each censored word so the beep covers attack/decay
# of the spoken word instead of clipping its edges.
_INTERVAL_PAD_S = 0.06
# Hard cap on beep intervals per export — keeps the post-pass filter graph
# bounded on pathological transcripts. Intervals beyond the cap are merged
# into their predecessor's span by the merge step long before this matters.
_MAX_INTERVALS = 300
# Default beep: 1 kHz sine (the classic broadcast bleep), attenuated so a
# full-scale tone doesn't blast the viewer relative to speech.
_BEEP_FREQ_HZ = 1000
_BEEP_GAIN = 0.5


def censor_word_list() -> list[str]:
    """The active block list: the user's CENSOR_WORDS (comma/newline
    separated) or the built-in default when unset / "default"."""
    from backend.config import settings
    raw = str(getattr(settings, "CENSOR_WORDS", "") or "").strip()
    if not raw or raw.lower() == "default":
        return list(DEFAULT_CENSOR_WORDS)
    words = [w.strip().lower() for w in re.split(r"[,\n]+", raw)]
    return [w for w in words if w]


def mask_char() -> str:
    """The single masking symbol (Settings > Profanity Censor), default *."""
    from backend.config import settings
    ch = str(getattr(settings, "CENSOR_MASK_CHAR", "*") or "*")
    return ch[0] if ch else "*"


def custom_sound_path() -> Optional[str]:
    """Path of the user-uploaded censor sound, or None when none exists."""
    for ext in CENSOR_SOUND_EXTS:
        p = os.path.join(CENSOR_SOUND_DIR, CENSOR_SOUND_BASENAME + ext)
        if os.path.isfile(p):
            return p
    return None


def mask_word(word: str, symbol: str) -> str:
    """Mask a single word keeping its first and last letter: "shit" → "s**t".

    Words of 1–2 letters have no middle; mask everything after the first
    character so a 2-letter block-list entry still reads censored."""
    if len(word) <= 1:
        return symbol
    if len(word) == 2:
        return word[0] + symbol
    return word[0] + symbol * (len(word) - 2) + word[-1]


def _has_cjk(s: str) -> bool:
    return any(
        "一" <= c <= "鿿" or "぀" <= c <= "ヿ"
        or "가" <= c <= "힯" for c in s
    )


def _compile_pattern(words: Iterable[str]) -> Optional[re.Pattern]:
    """One case-insensitive pattern for the whole list. Space-delimited
    scripts get word-boundary guards; CJK entries (no word boundaries in
    those scripts) match as plain substrings."""
    bounded, raw = [], []
    for w in sorted({w.lower() for w in words if w}, key=len, reverse=True):
        (raw if _has_cjk(w) else bounded).append(re.escape(w))
    parts = []
    if bounded:
        parts.append(r"(?<!\w)(?:" + "|".join(bounded) + r")(?!\w)")
    if raw:
        parts.append("(?:" + "|".join(raw) + ")")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)


def censor_text(text: str, words: Optional[list[str]] = None,
                symbol: Optional[str] = None) -> tuple[str, int]:
    """Mask every blocked word in ``text``. Returns (masked_text, hits)."""
    if not text:
        return text, 0
    pattern = _compile_pattern(words if words is not None else censor_word_list())
    if pattern is None:
        return text, 0
    sym = (symbol if symbol is not None else mask_char())
    hits = 0

    def _sub(m: re.Match) -> str:
        nonlocal hits
        hits += 1
        return mask_word(m.group(0), sym)

    return pattern.sub(_sub, text), hits


def _seg_get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def censor_segments(segments: list, words: Optional[list[str]] = None,
                    symbol: Optional[str] = None) -> tuple[list, int]:
    """Masked deep-copies of transcript segments (dicts), covering both the
    cue ``text`` and per-word karaoke entries. Returns (segments, hits)."""
    wl = words if words is not None else censor_word_list()
    sym = symbol if symbol is not None else mask_char()
    out, total = [], 0
    for seg in segments or []:
        d = dict(seg) if isinstance(seg, dict) else seg.model_dump()
        masked, n = censor_text(d.get("text", "") or "", wl, sym)
        if n:
            d["text"] = masked
            total += n
        w_list = d.get("words")
        if w_list:
            new_words = []
            for w in w_list:
                wd = dict(w) if isinstance(w, dict) else w.model_dump()
                m_w, n_w = censor_text(wd.get("word", "") or "", wl, sym)
                if n_w:
                    wd["word"] = m_w
                    total += n_w
                new_words.append(wd)
            d["words"] = new_words
        out.append(d)
    return out, total


def _merge_intervals(intervals: list[tuple[float, float]],
                     gap: float = 0.05) -> list[tuple[float, float]]:
    """Merge overlapping / near-adjacent intervals into single beeps."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def profane_intervals(segments: list, clip_start: float, clip_end: float,
                      words: Optional[list[str]] = None,
                      pad: float = _INTERVAL_PAD_S) -> list[tuple[float, float]]:
    """Absolute-time (source) intervals containing blocked words, within
    [clip_start, clip_end], merged and clamped.

    Word-level Whisper timestamps give the exact window when present
    (synthetic word times are fine — they're char-weighted estimates, which
    is exactly the fallback anyway). Cues without word times interpolate the
    word's position by character offset within the cue."""
    wl = words if words is not None else censor_word_list()
    pattern = _compile_pattern(wl)
    if pattern is None:
        return []
    found: list[tuple[float, float]] = []
    for seg in segments or []:
        s0 = float(_seg_get(seg, "start", 0) or 0)
        s1 = float(_seg_get(seg, "end", 0) or 0)
        if s1 <= clip_start or s0 >= clip_end or s1 <= s0:
            continue
        text = str(_seg_get(seg, "text", "") or "")
        if not text or not pattern.search(text):
            continue
        w_list = _seg_get(seg, "words") or []
        used_word_times = False
        if w_list:
            for w in w_list:
                w_text = str(_seg_get(w, "word", "") or "").strip()
                if w_text and pattern.search(w_text):
                    ws = float(_seg_get(w, "start", 0) or 0)
                    we = float(_seg_get(w, "end", 0) or 0)
                    if we > ws:
                        found.append((ws - pad, we + pad))
                        used_word_times = True
        if not used_word_times:
            # Char-weight interpolation: the word's character span maps
            # linearly onto the cue's time span.
            dur = s1 - s0
            total = max(1, len(text))
            for m in pattern.finditer(text):
                t_a = s0 + dur * (m.start() / total)
                t_b = s0 + dur * (m.end() / total)
                found.append((t_a - pad, t_b + pad))
    clamped = [
        (max(clip_start, a), min(clip_end, b))
        for a, b in found if min(clip_end, b) > max(clip_start, a)
    ]
    return _merge_intervals(clamped)[:_MAX_INTERVALS]


def map_to_output_time(intervals_abs: list[tuple[float, float]],
                       clip_start: float, clip_end: float,
                       speed: float = 1.0,
                       segments: Optional[list] = None) -> list[tuple[float, float]]:
    """Source-absolute intervals → OUTPUT-time (post trim + speed) intervals.

    Mirrors the export's speed timeline (``_compute_output_duration``): gaps
    between editor segments run at the global speed, each segment at its own.
    With no per-segment speeds this collapses to (t - clip_start) / speed."""
    g_speed = speed if speed and speed > 0 else 1.0
    seg_list = []
    for seg in segments or []:
        try:
            a = max(clip_start, float(_seg_get(seg, "start", 0) or 0))
            b = min(clip_end, float(_seg_get(seg, "end", 0) or 0))
            sp = float(_seg_get(seg, "speed", g_speed) or g_speed)
            if b > a and sp > 0:
                seg_list.append((a, b, sp))
        except (TypeError, ValueError):
            continue
    seg_list.sort()
    has_seg_speed = any(abs(sp - g_speed) > 0.001 for _, _, sp in seg_list)

    def _to_out(t_abs: float) -> float:
        t = min(max(t_abs, clip_start), clip_end)
        if not has_seg_speed:
            return (t - clip_start) / g_speed
        out = 0.0
        pos = clip_start
        for a, b, sp in seg_list:
            if a > pos:                      # gap before segment: global speed
                span = min(a, t) - pos
                if span > 0:
                    out += span / g_speed
                pos = a
                if t <= a:
                    return out
            span = min(b, t) - pos           # inside segment: its own speed
            if span > 0:
                out += span / sp
            pos = max(pos, min(b, t))
            if t <= b:
                return out
        if t > pos:                          # tail gap
            out += (t - pos) / g_speed
        return out

    out_intervals = []
    for a, b in intervals_abs:
        oa, ob = _to_out(a), _to_out(b)
        if ob - oa > 0.01:
            out_intervals.append((round(oa, 3), round(ob, 3)))
    return out_intervals


def build_censor_audio_cmd(
    input_path: str,
    output_path: str,
    intervals_out: list[tuple[float, float]],
    beep_path: Optional[str] = None,
    sample_rate: int = 48000,
    faststart: bool = True,
    volume: float = 1.0,
    separate_track: bool = False,
) -> list[str]:
    """The ffmpeg post-pass: mute each interval on the main track and mix a
    beep over it. Video is stream-copied — only the audio is re-encoded, so
    the pass is fast and can't disturb the picture.

    volume: universal loudness multiplier on the censor sound (1.0 = the
    built-in baselines: tone at 0.5 full-scale, custom file at its own
    recorded loudness).

    separate_track: additionally mux a SECOND audio track carrying only the
    beeps (silence elsewhere, titled "Censor beeps") so an editor can grab
    or drop them; track 1 keeps the normal censored mix and stays the
    default, so ordinary players sound identical either way."""
    mute_terms = "+".join(
        f"between(t,{a:.3f},{b:.3f})" for a, b in intervals_out
    )
    if separate_track:
        # A filtergraph pad is single-use — split the source audio so both
        # the censored mix and the beep-only track's silent bed can tap it.
        fc = [
            "[0:a]asplit=2[a_main][a_bed]",
            f"[a_main]volume=enable='{mute_terms}':volume=0[am]",
            "[a_bed]volume=0[sil]",
        ]
    else:
        fc = [f"[0:a]volume=enable='{mute_terms}':volume=0[am]"]

    total_dur = max(b for _, b in intervals_out) + 1.0
    v = max(0.0, float(volume or 1.0))
    if beep_path:
        beep_input = ["-i", beep_path]
        # Loop the uploaded sound so it covers intervals longer than itself,
        # and resample to the main track's rate for amix.
        src_chain = (f"aloop=loop=-1:size=2147483647,"
                     f"aresample={sample_rate}")
        gain = 1.0 * v
    else:
        beep_input = [
            "-f", "lavfi", "-t", f"{total_dur:.3f}",
            "-i", f"sine=frequency={_BEEP_FREQ_HZ}:sample_rate={sample_rate}",
        ]
        src_chain = ""
        gain = _BEEP_GAIN * v

    beep_main, beep_sep = [], []
    for i, (a, b) in enumerate(intervals_out):
        dur = b - a
        chain = f"[1:a]{src_chain + ',' if src_chain else ''}" \
                f"atrim=0:{dur:.3f},asetpts=PTS-STARTPTS," \
                f"volume={gain:.3f},adelay={int(a * 1000)}:all=1"
        if separate_track:
            # Each beep feeds BOTH mixes — pads are single-use, so split.
            chain += f",asplit=2[bm{i}][bs{i}]"
            beep_main.append(f"[bm{i}]")
            beep_sep.append(f"[bs{i}]")
        else:
            chain += f"[b{i}]"
            beep_main.append(f"[b{i}]")
        fc.append(chain)

    n = len(beep_main) + 1
    fc.append(f"[am]{''.join(beep_main)}amix=inputs={n}:duration=first:"
              f"normalize=0[aout]")
    if separate_track:
        # Silent bed keeps the beep track exactly as long as the main one.
        fc.append(f"[sil]{''.join(beep_sep)}amix=inputs={n}:duration=first:"
                  f"normalize=0[beeps]")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        *beep_input,
        "-filter_complex", ";".join(fc),
        "-map", "0:v?", "-c:v", "copy",
        "-map", "[aout]",
    ]
    if separate_track:
        cmd += ["-map", "[beeps]"]
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    if separate_track:
        cmd += [
            "-metadata:s:a:0", "title=Censored audio",
            "-metadata:s:a:1", "title=Censor beeps",
            "-disposition:a:0", "default",
            "-disposition:a:1", "0",
        ]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(output_path)
    return cmd


async def apply_censor_beeps(
    video_path: str,
    intervals_out: list[tuple[float, float]],
    sample_rate: Optional[int] = None,
) -> int:
    """Run the beep post-pass in place (tmp + atomic replace). Returns the
    number of beeped intervals. Raises on ffmpeg failure — silently shipping
    UNCENSORED audio after the user toggled the censor is the one outcome
    this feature must never produce."""
    import asyncio
    from backend.config import settings

    if not intervals_out:
        return 0
    beep = None
    if str(getattr(settings, "CENSOR_BEEP_SOUND", "beep")) == "custom":
        beep = custom_sound_path()
        if beep is None:
            logger.warning("Censor: custom sound selected but no file found "
                           "— using the default beep tone")
    sr = sample_rate or 48000
    try:
        vol = float(getattr(settings, "CENSOR_BEEP_VOLUME", 1.0) or 1.0)
    except (TypeError, ValueError):
        vol = 1.0
    vol = max(0.1, min(3.0, vol))
    base, ext = os.path.splitext(video_path)
    tmp_path = f"{base}.censor_tmp{ext or '.mp4'}"
    cmd = build_censor_audio_cmd(
        video_path, tmp_path, intervals_out, beep_path=beep, sample_rate=sr,
        faststart=bool(getattr(settings, "FFMPEG_FASTSTART", True)),
        volume=vol,
        separate_track=bool(getattr(settings, "CENSOR_SEPARATE_TRACK", False)),
    )
    logger.info("Censor beep pass (%d interval(s)): %s",
                len(intervals_out), " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        tail = (stderr or b"").decode(errors="replace")[-1200:]
        raise RuntimeError(f"Censor beep pass failed:\n{tail}")
    os.replace(tmp_path, video_path)
    return len(intervals_out)
