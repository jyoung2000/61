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
    const w0 = segment.words[0];
    const wLast = segment.words[segment.words.length - 1];
    const cueDur = Math.max(0.001, segment.end - segment.start);
    // Do the word timestamps use a PER-CUE 0-based clock (first word ≈ 0, whole
    // span ≈ the cue duration) rather than the cue's own coordinate? ONLY then
    // rebase onto a within-cue clock. A first word that begins a little BEFORE
    // segment.start is NOT a different clock — it's the normal case: Whisper's
    // first word routinely starts a few hundredths before the cue's (rounded /
    // overlap-resolved) start, and an overlap-pushed cue start sits later than
    // its own audio. The old ``w0.start < segment.start - 0.01`` test fired on
    // exactly those cues and double-subtracted segment.start, shoving the
    // highlight into the middle of the line ("doesn't start at the beginning").
    const perCueClock = w0.start < segment.start - 1.0 && wLast.end <= cueDur + 1.0;
    const baseT = perCueClock ? (relativeTime - segment.start) : relativeTime;
    const adjusted = baseT + anticipation - AUDIO_BUFFER_S;
    // Before the first word's audio but with the cue already on screen, light
    // the FIRST word — karaoke should begin at the start of the line, not sit
    // dark through the lead-in and then jump in mid-sentence.
    if (adjusted < w0.start) {
      return relativeTime >= segment.start - 0.05 ? 0 : -1;
    }
    for (let i = 0; i < segment.words.length; i++) {
      if (adjusted < segment.words[i].end) return i;
    }
    return segment.words.length - 1;
  }

  // Proportional timing fallback for segments without word data
  const totalChars = words.reduce((sum, w) => sum + w.length, 0);
  if (totalChars === 0) return -1;
  const segDuration = segment.end - segment.start;
  const elapsed = (relativeTime - segment.start) + anticipation - AUDIO_BUFFER_S;
  if (elapsed < 0) return -1;

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
