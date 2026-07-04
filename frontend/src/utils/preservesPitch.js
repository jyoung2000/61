/**
 * Pitch behavior for speed-changed media elements — shared by the
 * preview (<video>/<audio> elements) so it can't drift per call site.
 *
 * ClipAI's default is VARISPEED (preservePitch=false): pitch shifts with
 * speed, like tape, matching the client export's
 * AudioBufferSourceNode.playbackRate and the server's asetrate path.
 * Browsers default HTMLMediaElement.preservesPitch to true, so the
 * preview must explicitly opt out or it silently diverges from both
 * export paths.
 */
export function applyPreservesPitch(el, preservePitch) {
  if (!el) return;
  const on = !!preservePitch;
  try {
    el.preservesPitch = on;
    // Legacy engine prefixes (older WebKit / Firefox)
    if ('webkitPreservesPitch' in el) el.webkitPreservesPitch = on;
    if ('mozPreservesPitch' in el) el.mozPreservesPitch = on;
  } catch {
    // Read-only in exotic engines — preview pitch may then differ from
    // export; nothing else breaks.
  }
}
