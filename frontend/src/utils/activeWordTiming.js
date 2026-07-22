// Shared active-word timing — the SINGLE source of truth for the
// word-highlight algorithm used by BOTH the DOM preview
// (``SubtitleOverlay``) and the canvas compositor (``RenderEngine``,
// which serves preview *and* client export).
//
// These constants and functions were previously duplicated in
// ``SubtitleOverlay.jsx`` and ``RenderEngine.js`` with "MUST match
// exactly" comments. Duplication is exactly how they drift; importing
// from here makes the parity contract structural instead of
// aspirational. The backend ``ass_generator.py`` mirrors the same
// ``anticipation - AUDIO_BUFFER_S`` net offset for server exports.
//
// ``ClipPreview`` imports the constants but keeps its own (older)
// word-timestamp branch because its word times are always
// clip-relative by construction. ``ClipSEO`` intentionally uses a
// neutral anticipation of 0 and is NOT a consumer.

export const BASE_OVERHEAD_S = 0.04; // minimum gap between words
export const ANTICIPATION_S = 0.10;  // perceptual lead — highlight leads audio
export const AUDIO_BUFFER_S = 0.12;  // compensate for browser audio output lag

// Extra pause added AFTER a word that ends with punctuation
export const PUNCT_PAUSE = {
  ',': 0.15, ';': 0.16, ':': 0.12, '.': 0.22, '!': 0.22, '?': 0.24,
  '—': 0.12, '–': 0.10,
};

// Function words are spoken ~25% faster in natural speech
export const FAST_WORDS = new Set([
  'the', 'a', 'an', 'to', 'in', 'on', 'at', 'of', 'for',
  'and', 'but', 'or', 'is', 'was', 'are', 'were', 'it',
  'its', 'this', 'that',
]);

// Speaker color palette shared by SubtitleOverlay / RenderEngine /
// VideoEditor / Analysis. Index = order of first appearance.
export const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
  '#FB7185', '#38BDF8', '#FBBF24', '#34D399', '#C084FC', '#F472B6',
  '#22D3EE', '#A3E635', '#FB923C', '#2DD4BF', '#818CF8', '#F87171',
];

/**
 * Compute per-speaker word rates (words/sec) from subtitle segments.
 * Accepts segments carrying either ``text`` or ``subtitleText``.
 */
export function computeSpeakerRates(segments) {
  const stats = {};
  for (const seg of segments || []) {
    const wc = (seg.text || seg.subtitleText || '').split(/\s+/).filter(Boolean).length;
    const dur = seg.end - seg.start;
    if (dur <= 0 || wc === 0) continue;
    if (!stats[seg.speaker]) stats[seg.speaker] = { words: 0, time: 0 };
    stats[seg.speaker].words += wc;
    stats[seg.speaker].time += dur;
  }
  const rates = {};
  for (const [sp, s] of Object.entries(stats)) {
    rates[sp] = s.time > 0 ? s.words / s.time : 3.0;
  }
  return rates;
}

/**
 * Compute the active word index for a subtitle segment at a given time.
 *
 * Word-timestamp branch is coordinate-agnostic: word timestamps may be
 * timeline-absolute (Whisper's native output) or clip-relative (legacy
 * paths); we detect which by checking whether the first word's start
 * lies inside the segment window, then compare in the same coordinate
 * system. Per-speaker rate-scaled anticipation applies to BOTH
 * branches so fast/slow speakers highlight consistently.
 *
 * @param {object} segment - {subtitleText|text, start, end, speaker, words?}
 * @param {number} relativeTime - current time in the segment's own coordinates
 * @param {object} speakerRates - map speaker → words/sec (from computeSpeakerRates)
 * @returns {number} word index, or -1 when no word is active yet
 */
export function getCurrentWordIndex(segment, relativeTime, speakerRates) {
  const text = segment?.subtitleText || segment?.text || '';
  if (!text) return -1;
  const words = text.split(/\s+/).filter(Boolean);
  if (words.length <= 1) return words.length === 1 ? 0 : -1;

  const speakerWps = (speakerRates && speakerRates[segment.speaker]) || 3.0;
  const rateScale = Math.max(0.6, Math.min(1.6, 3.0 / speakerWps));
  const anticipation = ANTICIPATION_S * rateScale;

  if (segment.words && segment.words.length === words.length) {
    const n = segment.words.length;
    const segStart = segment.start;
    const segEnd = Math.max(segStart + 0.001, segment.end);
    const cueSpan = segEnd - segStart;
    const w0 = segment.words[0];
    const wLast = segment.words[n - 1];
    // Per-cue 0-based clock? (first word begins WELL before the cue's own start
    // AND the whole span fits the cue duration) → shift word times into the
    // cue's coordinate. A first word that begins a hair before segment.start is
    // the normal Whisper case, NOT a different clock — don't rebase for it.
    const perCueClock =
      Number.isFinite(w0.start) && w0.start < segStart - 1.0 &&
      Number.isFinite(wLast.end) && wLast.end <= cueSpan + 1.0;
    const toCue = (t) => (perCueClock ? t + segStart : t);
    // Build STRICTLY-INCREASING end boundaries that cover the whole cue, each
    // word guaranteed a ≥ minGap slice. Raw Whisper-EN projected word times can
    // be out of order / overlapping after alignment + interpolation; a naive
    // "first end > t" scan then SKIPS a word whose end is smaller than an
    // earlier word's, and lets an early word with a late end grab the highlight
    // mid-line. Clamping to a feasible monotonic schedule fixes both: word 0
    // begins at the cue start, the last word holds to the cue end, and no word
    // is skipped even when its raw timing is degenerate.
    const minGap = Math.min(0.05, cueSpan / n);
    const ends = new Array(n);
    let cur = segStart;
    for (let i = 0; i < n; i++) {
      const tail = n - 1 - i;
      const raw = segment.words[i] && Number.isFinite(segment.words[i].end)
        ? toCue(segment.words[i].end) : cur + minGap;
      let e = Math.min(raw, segEnd - tail * minGap); // leave room for the rest
      e = Math.max(e, cur + minGap);                 // this word gets a real slice
      ends[i] = Math.min(segEnd, e);
      cur = ends[i];
    }
    ends[n - 1] = segEnd; // the last word holds to the cue's own end
    const adjusted = relativeTime + anticipation - AUDIO_BUFFER_S;
    // On screen but before the first word's audio → light word 0 (karaoke starts
    // at the beginning of the line, never dark-then-jump-in-mid-sentence).
    if (adjusted < segStart) {
      return relativeTime >= segStart - 0.05 ? 0 : -1;
    }
    for (let i = 0; i < n; i++) {
      if (adjusted < ends[i]) return i;
    }
    return n - 1;
  }

  // Proportional timing fallback for segments without word data
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  if (totalChars === 0) return -1;
  const segDuration = segment.end - segment.start;
  const elapsed = (relativeTime - segment.start) + anticipation - AUDIO_BUFFER_S;
  // On screen but before the lead-in elapses → light word 0 (mirror the
  // word-timestamp branch; never sit dark then jump in mid-sentence).
  if (elapsed < 0) return relativeTime >= segment.start - 0.05 ? 0 : -1;

  const punctPauses = words.map((w) => {
    const last = w[w.length - 1];
    return (PUNCT_PAUSE[last] || 0) * rateScale;
  });
  const totalPunct = punctPauses.reduce((a, b) => a + b, 0);
  const baseOverhead = BASE_OVERHEAD_S * rateScale * words.length;
  const totalPause = baseOverhead + totalPunct;
  const charTime = Math.max(segDuration - totalPause, segDuration * 0.45);
  const pauseScale = (segDuration - charTime) / Math.max(totalPause, 0.01);

  // First pass: raw per-word durations with natural-speech rhythm.
  const rawDurations = words.map((w, i) => {
    const charDur = charTime * (w.length / totalChars);
    const pause = (BASE_OVERHEAD_S * rateScale + punctPauses[i]) * pauseScale;
    let wordDur = charDur + pause;
    const stripped = w.toLowerCase().replace(/[.,!?;:—–]+$/, '');
    if (FAST_WORDS.has(stripped)) wordDur *= 0.75;
    if (i === 0) wordDur *= 1.15;
    else if (i === words.length - 1) wordDur *= 1.10;
    return wordDur;
  });
  // Normalize so the durations sum EXACTLY to segDuration — matches the
  // export fallback (ass_generator.py), which scales raw_durations by
  // duration/total_raw. Without this the fast-word (×0.75) and first/last
  // (×1.15/×1.10) adjustments push the frontend total off the segment
  // length, so the previewed highlight drifts from the exported one on
  // word-less cues. Normalizing keeps the two in lock-step.
  const totalRaw = rawDurations.reduce((a, b) => a + b, 0);
  if (totalRaw > 0) {
    const norm = segDuration / totalRaw;
    for (let i = 0; i < rawDurations.length; i++) rawDurations[i] *= norm;
  }

  let t = 0;
  for (let i = 0; i < words.length; i++) {
    if (elapsed < t + rawDurations[i]) return i;
    t += rawDurations[i];
  }
  return words.length - 1;
}
