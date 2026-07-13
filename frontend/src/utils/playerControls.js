/**
 * Pure helpers for the preview players' touch-aware controls (VideoPlayer,
 * ClipPreview). Kept side-effect-free so the fiddly bits — scrub geometry,
 * time clamping, when-to-show-controls, touch detection — are unit-tested
 * without rendering a full <video> component.
 */

/**
 * Fraction [0,1] of a seek bar that a pointer at ``clientX`` lands on, given
 * the bar's bounding rect. Clamped so a drag past either edge pins to 0 / 1
 * instead of seeking out of bounds.
 */
export function seekPct(clientX, rect) {
  if (!rect || !rect.width) return 0;
  return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
}

/** Clamp a target time into [0, duration] (NaN/negative duration → 0). */
export function clampTime(t, duration) {
  const d = Number.isFinite(duration) && duration > 0 ? duration : 0;
  if (!Number.isFinite(t)) return 0;
  return Math.max(0, Math.min(d, t));
}

/**
 * Whether the control bar should be shown. On touch there's no hover, so
 * controls are persistent; on a mouse they appear on hover; and they always
 * show while paused so the player never looks dead/frozen.
 */
export function controlsVisible({ hovered = false, touch = false, playing = false } = {}) {
  return Boolean(hovered || touch || !playing);
}

/**
 * Coarse-pointer / no-hover (touch) capability of a window-like object.
 * Accepts the target ``win`` so it's testable with a stub; falls back to the
 * global ``window`` in the app. Independent of viewport width — a wide tablet
 * or touch laptop is still "touch" for control-affordance purposes.
 */
export function detectTouch(win = typeof window !== 'undefined' ? window : undefined) {
  if (!win) return false;
  try {
    const mm = typeof win.matchMedia === 'function' ? win.matchMedia.bind(win) : null;
    if (mm && (mm('(pointer: coarse)').matches || mm('(hover: none)').matches)) return true;
    if ('ontouchstart' in win) return true;
    return (win.navigator && win.navigator.maxTouchPoints ? win.navigator.maxTouchPoints : 0) > 0;
  } catch {
    return 'ontouchstart' in win;
  }
}
