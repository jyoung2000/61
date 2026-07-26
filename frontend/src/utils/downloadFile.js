// Saving a generated file to disk from the browser.
//
// This looks trivial and isn't: a subtitle export shipped with NO filename at
// all because two details of the anchor dance were wrong. Both are encoded
// here so every caller gets them right.

/**
 * The filename a backend download endpoint chose, read out of its
 * ``Content-Disposition`` header. Returns '' when the header is absent or
 * unparseable.
 *
 * A plain navigation download takes its name from this header automatically.
 * A ``fetch`` + blob download does NOT — the blob has no name, so unless we
 * lift the name out ourselves the browser invents one from the blob URL.
 */
export function dispositionName(res) {
  try {
    const dispo = (res && res.headers && res.headers.get('content-disposition')) || '';
    const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(dispo);
    if (!m) return '';
    let raw = m[1].trim();
    try {
      raw = decodeURIComponent(raw);
    } catch {
      /* not percent-encoded — use it verbatim */
    }
    // Never let a server-supplied name walk out of the downloads folder.
    return raw.split(/[\\/]/).pop().trim();
  } catch {
    return '';
  }
}

/**
 * Save ``content`` (a string) to disk as ``filename``.
 *
 * Two load-bearing details:
 *   1. The anchor must be IN the document when clicked. A detached anchor's
 *      ``download`` attribute is honored inconsistently — Chrome usually takes
 *      it, but not when the click happens outside a user-gesture task (exactly
 *      the case once a handler awaits a fetch first), and the browser then
 *      names the file after the blob URL: no name, no extension.
 *   2. Revoking the object URL must wait. ``click()`` only SCHEDULES the
 *      download; revoking in the same tick can pull the blob out from under it,
 *      losing the name or the file entirely.
 */
export function saveTextAs(content, filename, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'download';
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return url;
}
