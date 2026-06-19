// Inline cue-timestamp markers the backend bakes into clip caption / hook
// text, e.g. "[0:00] you should ask [0:02] Relena-san about it." matches
// "[m:ss]", "[mm:ss]" and "[h:mm:ss]".
const CUE_TIMESTAMP = /\[\d{1,2}:\d{2}(?::\d{2})?\]/g;

/**
 * Strip inline cue-timestamp markers from caption / hook text so they never
 * appear in a clip card or as part of a text overlay. Leaves clean prose.
 *
 * @param {string} text
 * @returns {string}
 */
export function stripInlineTimestamps(text) {
  if (!text || typeof text !== 'string') return text || '';
  return text
    .replace(CUE_TIMESTAMP, ' ')
    .replace(/\s+([.,!?;:])/g, '$1') // tidy space left before punctuation
    .replace(/\s{2,}/g, ' ')         // collapse the gaps the markers leave
    .trim();
}

export default stripInlineTimestamps;
