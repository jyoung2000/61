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
 * Whether a media element can honor a seek to ``t`` RIGHT NOW.
 *
 * The HTML seek algorithm silently ABORTS a ``currentTime`` write whose
 * target falls outside every ``video.seekable`` range — no error, no
 * ``seeking`` event, nothing. That is the normal state on touch, where the
 * players use ``preload='metadata'`` and no media data is loaded until the
 * first play. (Safari's ``fastSeek`` is worse: it clamps the target into the
 * empty range, landing at 0.) Callers latch a seek that fails this check and
 * re-issue it once the element gains data — see ``pendingSeekAction``.
 */
export function canSeekNow(video, t) {
  if (!video || !Number.isFinite(t)) return false;
  if ((video.readyState ?? 0) < 1) return false;
  const s = video.seekable;
  if (!s || !s.length) return false;
  for (let i = 0; i < s.length; i++) {
    // ±0.1s: a range boundary reported at 9.999 must still accept a seek to 10.
    if (t >= s.start(i) - 0.1 && t <= s.end(i) + 0.1) return true;
  }
  return false;
}

/**
 * Decide what to do with an owed ("pending") seek — the state machine behind
 * the mobile scrub-then-play fix. Pure so the fiddly branches are testable
 * without a real media element.
 *
 * ``pending`` is ``null`` or ``{ t, at }`` where ``at`` is a monotonic
 * timestamp (``performance.now()``). Returns one of:
 *   'idle'  — nothing owed
 *   'done'  — already at the target, or the latch expired; clear it
 *   'apply' — element can seek now; write ``currentTime`` and clear
 *   'wait'  — still unseekable; keep the latch and try again later
 *
 * ``tolerance`` (default 0.5s) treats a seek that landed close enough as
 * satisfied — browsers snap to keyframes, so an exact match never happens.
 * ``maxWaitMs`` (default 5s) is a safety valve: a source that never becomes
 * seekable at the target (load error, truncated file) must not wedge the
 * playhead forever. 5s covers opening the media on a slow mobile link — the
 * ``playing`` event, which is the real backstop, fires well inside it — while
 * bounding how long a pathological source can hold time-reporting silent.
 */
export function pendingSeekAction(
  video, pending, now,
  { tolerance = 0.5, maxWaitMs = 5000, retryMs = 300 } = {},
) {
  if (!pending || !video) return 'idle';
  if (Math.abs((video.currentTime || 0) - pending.t) <= tolerance) return 'done';
  if (Number.isFinite(pending.at) && Number.isFinite(now) && now - pending.at > maxWaitMs) {
    return 'done';
  }
  if (!canSeekNow(video, pending.t)) return 'wait';
  // ``canSeekNow`` is a heuristic, not a guarantee — iOS Safari can report a
  // seekable range it won't actually honor yet. So a re-issued seek is kept
  // latched until the element demonstrably lands on it (the tolerance branch
  // above), and re-attempted at most every ``retryMs``. Without the throttle
  // the rAF loop would rewrite ``currentTime`` 60×/s against an element that
  // keeps refusing; without the retry a single dropped re-issue would strand
  // the seek with nothing left to re-apply it.
  if (Number.isFinite(pending.lastApply) && Number.isFinite(now)
      && now - pending.lastApply < retryMs) {
    return 'wait';
  }
  return 'apply';
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
