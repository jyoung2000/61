/**
 * Seek latch for the preview players — the mobile "scrub, press play, and it
 * starts at 0:00" fix, shared so every player behaves identically.
 *
 * Why this exists: on touch the players render with ``preload='metadata'``
 * (deliberate — ``auto`` saturates a slow mobile link buffering a long
 * source). With only metadata loaded the element's ``seekable`` range list is
 * usually EMPTY until the first ``play()`` opens the media, and the HTML seek
 * algorithm SILENTLY ABORTS a ``currentTime`` write whose target lies outside
 * every seekable range — no exception, no ``seeking`` event, no ``seeked``
 * event, nothing a caller can test. (Safari's ``fastSeek`` is worse: it
 * clamps the target into the empty range, landing at 0.) Players updated
 * their React playhead optimistically, so the scrub LOOKED like it worked and
 * ``play()`` then started from the element's real position: 0.
 *
 * The latch remembers a seek the element could not honor and re-issues it the
 * moment it can. Two rules keep it honest while a seek is owed:
 *   * the element's clock is NOT trustworthy — see ``trustedTime`` — so time
 *     loops must not write it back over the UI playhead (that is what
 *     erased the user's scrub position), and
 *   * ``assertBeforePlay`` re-applies the UI position at the play() gesture,
 *     which is exactly when the media opens and the seek can finally land.
 */
import { useRef, useCallback, useEffect, useMemo } from 'react';
import { canSeekNow, pendingSeekAction } from '../utils/playerControls';

/** Media events after which a previously-impossible seek may become possible. */
const FLUSH_EVENTS = ['loadedmetadata', 'loadeddata', 'canplay', 'playing', 'progress'];

/**
 * @param videoRef  ref to the <video> (or <audio>) element
 * @param src       current source — resets/re-binds the latch when it changes
 */
export default function useSeekLatch(videoRef, src) {
  // { t, at } while a seek is owed, else null. A ref (not state) so reading it
  // inside rAF loops and event handlers never goes stale and never re-renders.
  const pendingRef = useRef(null);
  // Last position the UI asked for, latched or not. ``load()`` (the players'
  // error-retry path) resets the element to readyState 0 / currentTime 0 and
  // empties ``seekable``, silently discarding wherever the user was — this
  // is what lets us put them back.
  const lastAskedRef = useRef(null);

  /** Remember ``t`` iff the element cannot honor it right now. */
  const latch = useCallback((t) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(t)) return;
    lastAskedRef.current = t;
    pendingRef.current = canSeekNow(video, t) ? null : { t, at: performance.now() };
  }, [videoRef]);

  /**
   * Seek to ``t``, latching it if the element drops it. This is the call every
   * seek site should use instead of writing ``currentTime`` directly.
   * ``fast`` opts into ``fastSeek`` (keyframe-accurate, right for scrubbing).
   */
  const seek = useCallback((t, { fast = false } = {}) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(t)) return;
    latch(t);
    try {
      if (fast && typeof video.fastSeek === 'function') video.fastSeek(t);
      else video.currentTime = t;
    } catch {
      try { video.currentTime = t; } catch { /* noop */ }
    }
  }, [videoRef, latch]);

  /**
   * Resolve the latch. Returns true when nothing is owed any more (callers may
   * trust ``video.currentTime``), false while a seek is still outstanding.
   */
  const settle = useCallback(() => {
    const video = videoRef.current;
    const pend = pendingRef.current;
    const now = performance.now();
    const action = pendingSeekAction(video, pend, now);
    if (action === 'wait') return false;
    if (action === 'apply') {
      // Keep the latch: ``canSeekNow`` is a heuristic and this write can be
      // dropped too. It clears on the next settle, once the element is
      // demonstrably at the target ('done'). ``lastApply`` throttles retries.
      pend.lastApply = now;
      try { video.currentTime = pend.t; } catch { /* noop */ }
      return false;
    }
    if (action === 'done') pendingRef.current = null;
    return true;
  }, [videoRef]);

  /** True while a seek is owed, i.e. the element's clock is not trustworthy. */
  const isPending = useCallback(() => pendingRef.current != null, []);

  /**
   * The time the UI should believe: the owed target while a seek is pending,
   * otherwise the element's own clock. Use this as the base for RELATIVE
   * jumps (±10s skips) and for "what time is the playhead at" reads, so a
   * dropped seek can't teleport the user back to 0.
   */
  const trustedTime = useCallback((fallback = 0) => {
    const pend = pendingRef.current;
    if (pend) return pend.t;
    const video = videoRef.current;
    const t = video ? video.currentTime : fallback;
    return Number.isFinite(t) ? t : fallback;
  }, [videoRef]);

  /**
   * Call immediately before ``play()``: playback must start where the UI
   * playhead is, not where the element's (possibly stale) clock sits. Latches
   * the position too, so if this write is dropped as well the ``playing``
   * flush lands it as soon as the media opens.
   */
  const assertBeforePlay = useCallback((uiTime, tolerance = 0.75) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(uiTime)) return;
    lastAskedRef.current = uiTime;
    if (Math.abs((video.currentTime || 0) - uiTime) > tolerance) {
      pendingRef.current = { t: uiTime, at: performance.now() };
      try { video.currentTime = uiTime; } catch { /* noop */ }
    }
  }, [videoRef]);

  /** Forget any owed seek — for deliberate repositioning (loop wrap, reload). */
  const clear = useCallback(() => { pendingRef.current = null; }, []);

  // Land the owed seek as soon as the element gains data. ``playing`` is the
  // backstop that always fires: by then the media is open, and a currentTime
  // write mid-play seeks and keeps playing.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    const flush = () => settle();
    // ``emptied`` fires when the element is reset — notably the players'
    // error-retry ``video.load()``, which drops readyState, currentTime AND
    // seekable. Re-latch the last requested position so a transient load
    // failure doesn't silently restart the user at 0:00.
    const onEmptied = () => {
      const t = lastAskedRef.current;
      if (Number.isFinite(t) && t > 0) {
        pendingRef.current = { t, at: performance.now() };
      }
    };
    FLUSH_EVENTS.forEach((e) => video.addEventListener(e, flush));
    video.addEventListener('emptied', onEmptied);
    return () => {
      FLUSH_EVENTS.forEach((e) => video.removeEventListener(e, flush));
      video.removeEventListener('emptied', onEmptied);
    };
  }, [videoRef, settle, src]);

  // A new source invalidates any seek owed against the old one.
  useEffect(() => {
    pendingRef.current = null;
    lastAskedRef.current = null;
  }, [src]);

  // Memoized: an unstable identity here would propagate into every consumer's
  // useCallback deps — in VideoEditor that re-ran the PlayerContext
  // register/unregister effect on every render, letting a hidden editor steal
  // the active-player registration from the visible one.
  return useMemo(
    () => ({ seek, latch, settle, isPending, trustedTime, assertBeforePlay, clear }),
    [seek, latch, settle, isPending, trustedTime, assertBeforePlay, clear],
  );
}
