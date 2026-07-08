/**
 * Platform-aware keyboard-shortcut labels.
 *
 * The editor's shortcut HANDLERS are already cross-platform — every "mod"
 * binding matches ``ctrlKey || metaKey`` (see editorActions.js), so Ctrl+K,
 * Ctrl+Z, etc. all work on Windows/Linux. Only the on-screen LABELS were
 * Mac-exclusive: they hard-coded the ⌘/⌥/⌃/⇧ glyphs, which mean nothing to a
 * Windows user. ``formatKbd`` rewrites those glyphs to the platform's
 * conventional words so a Windows user sees "Ctrl+K" where a Mac user sees
 * "⌘K" — same universal binding, readable label.
 */

export const IS_MAC =
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || '');

/**
 * Localize a Mac-glyph shortcut string (e.g. "⌘⇧Z") for the current platform.
 * On macOS the glyphs are conventional, so they're kept as-is. Everywhere else
 * they become the familiar Windows/Linux words ("Ctrl+Shift+Z").
 */
export function formatKbd(kbd) {
  if (!kbd || IS_MAC) return kbd;
  return kbd
    .replace(/⌘/g, 'Ctrl+')
    .replace(/⌃/g, 'Ctrl+')
    .replace(/⌥/g, 'Alt+')
    .replace(/⇧/g, 'Shift+')
    .replace(/\s*\+\s*\+/g, '+') // collapse any accidental "++"
    .trim();
}
