/**
 * Preview-player control helpers — touch scrubbing geometry, time clamping,
 * control visibility, and touch detection (the logic behind the mobile/tablet
 * player fix). Pure functions, no <video> render required.
 */
import { describe, it, expect } from 'vitest';
import {
  seekPct, clampTime, controlsVisible, detectTouch,
  canSeekNow, pendingSeekAction,
} from './playerControls';

const rect = { left: 100, width: 200 }; // bar spans clientX 100..300

describe('seekPct', () => {
  it('maps a pointer inside the bar to its fraction', () => {
    expect(seekPct(100, rect)).toBe(0);
    expect(seekPct(200, rect)).toBe(0.5);
    expect(seekPct(300, rect)).toBe(1);
  });

  it('clamps a drag past either edge to [0,1] (no out-of-bounds seek)', () => {
    expect(seekPct(40, rect)).toBe(0);   // dragged left of the bar
    expect(seekPct(999, rect)).toBe(1);  // dragged right of the bar
  });

  it('is safe with a zero-width / missing rect', () => {
    expect(seekPct(150, { left: 0, width: 0 })).toBe(0);
    expect(seekPct(150, null)).toBe(0);
  });
});

describe('clampTime', () => {
  it('keeps a valid target inside [0, duration]', () => {
    expect(clampTime(30, 120)).toBe(30);
    expect(clampTime(-5, 120)).toBe(0);       // rewind past start
    expect(clampTime(130, 120)).toBe(120);    // fast-forward past end
  });

  it('degrades safely on NaN/absent duration', () => {
    expect(clampTime(30, NaN)).toBe(0);
    expect(clampTime(NaN, 120)).toBe(0);
    expect(clampTime(30, 0)).toBe(0);
  });
});

describe('controlsVisible', () => {
  it('always shows on touch (no hover to rely on)', () => {
    expect(controlsVisible({ touch: true, hovered: false, playing: true })).toBe(true);
  });

  it('shows on hover with a mouse', () => {
    expect(controlsVisible({ touch: false, hovered: true, playing: true })).toBe(true);
  });

  it('always shows while paused so the player never looks dead', () => {
    expect(controlsVisible({ touch: false, hovered: false, playing: false })).toBe(true);
  });

  it('hides on a mouse while playing and not hovered (immersive)', () => {
    expect(controlsVisible({ touch: false, hovered: false, playing: true })).toBe(false);
  });

  it('defaults to visible with no args', () => {
    expect(controlsVisible()).toBe(true);
  });
});

describe('detectTouch', () => {
  const winWith = (mediaMatches, extras = {}) => ({
    matchMedia: (q) => ({ matches: !!mediaMatches[q] }),
    navigator: { maxTouchPoints: 0 },
    ...extras,
  });

  it('true when the pointer is coarse', () => {
    expect(detectTouch(winWith({ '(pointer: coarse)': true }))).toBe(true);
  });

  it('true when hover is unavailable', () => {
    expect(detectTouch(winWith({ '(hover: none)': true }))).toBe(true);
  });

  it('true when maxTouchPoints > 0 even without matchMedia signals', () => {
    expect(detectTouch(winWith({}, { navigator: { maxTouchPoints: 5 } }))).toBe(true);
  });

  it('false for a plain mouse desktop', () => {
    expect(detectTouch(winWith({ '(pointer: coarse)': false, '(hover: none)': false }))).toBe(false);
  });

  it('false with no window', () => {
    expect(detectTouch(undefined)).toBe(false);
  });
});

// ── Mobile scrub-then-play regression ──────────────────────────────────────
// The reported bug: on a phone, scrubbing (or dragging the playhead) and then
// pressing play started at 0:00 instead of the scrubbed position. Cause — with
// preload="metadata" the element has an EMPTY seekable range until the first
// play, so the currentTime write was silently aborted while the UI playhead
// had already moved. These cover the predicate + the latch state machine.

/** Minimal media-element stub: ``ranges`` is a list of [start, end] pairs. */
const fakeVideo = (currentTime, ranges, readyState = 1) => ({
  currentTime,
  readyState,
  seekable: {
    length: ranges.length,
    start: (i) => ranges[i][0],
    end: (i) => ranges[i][1],
  },
});

describe('canSeekNow', () => {
  it('false when the element has no seekable range yet (the mobile bug)', () => {
    expect(canSeekNow(fakeVideo(0, []), 42)).toBe(false);
  });

  it('false before metadata has loaded, even if a range is reported', () => {
    expect(canSeekNow(fakeVideo(0, [[0, 100]], 0), 42)).toBe(false);
  });

  it('true for a target inside a loaded range', () => {
    expect(canSeekNow(fakeVideo(0, [[0, 100]]), 42)).toBe(true);
  });

  it('accepts targets on the range boundary within 0.1s slop', () => {
    expect(canSeekNow(fakeVideo(0, [[10, 20]]), 10)).toBe(true);
    expect(canSeekNow(fakeVideo(0, [[10, 20]]), 20.05)).toBe(true);
    expect(canSeekNow(fakeVideo(0, [[10, 20]]), 25)).toBe(false);
  });

  it('searches every range, not just the first', () => {
    expect(canSeekNow(fakeVideo(0, [[0, 5], [60, 90]]), 75)).toBe(true);
  });

  it('false for a missing element or a non-finite target', () => {
    expect(canSeekNow(null, 10)).toBe(false);
    expect(canSeekNow(fakeVideo(0, [[0, 100]]), NaN)).toBe(false);
  });
});

describe('pendingSeekAction', () => {
  const NOW = 10_000;

  it('idle when nothing is owed', () => {
    expect(pendingSeekAction(fakeVideo(0, [[0, 100]]), null, NOW)).toBe('idle');
  });

  it('waits while the element still cannot honor the seek', () => {
    const pend = { t: 42, at: NOW - 500 };
    expect(pendingSeekAction(fakeVideo(0, []), pend, NOW)).toBe('wait');
  });

  it('applies the owed seek once the element gains a seekable range', () => {
    const pend = { t: 42, at: NOW - 500 };
    expect(pendingSeekAction(fakeVideo(0, [[0, 100]]), pend, NOW)).toBe('apply');
  });

  it('is done when the element already landed near the target', () => {
    const pend = { t: 42, at: NOW - 500 };
    // Browsers snap to keyframes, so "close enough" counts as satisfied.
    expect(pendingSeekAction(fakeVideo(41.8, [[0, 100]]), pend, NOW)).toBe('done');
  });

  it('gives up after the safety valve so the playhead can never wedge', () => {
    const pend = { t: 42, at: NOW - 20_000 };
    expect(pendingSeekAction(fakeVideo(0, []), pend, NOW)).toBe('done');
  });

  it('keeps waiting right up to the safety valve', () => {
    const pend = { t: 42, at: NOW - 14_000 };
    expect(pendingSeekAction(fakeVideo(0, []), pend, NOW)).toBe('wait');
  });

  it('honors caller-supplied tolerance and timeout', () => {
    const pend = { t: 42, at: NOW - 500 };
    expect(pendingSeekAction(fakeVideo(41, [[0, 100]]), pend, NOW, { tolerance: 2 })).toBe('done');
    expect(pendingSeekAction(fakeVideo(0, []), pend, NOW, { maxWaitMs: 100 })).toBe('done');
  });

  it('end-to-end: scrub while unseekable, then play — target survives', () => {
    // 1. User drags the playhead to 42s before any media data exists.
    const video = fakeVideo(0, []);
    const pend = { t: 42, at: NOW };
    expect(pendingSeekAction(video, pend, NOW)).toBe('wait');
    // 2. Element still at 0 a moment later — the latch must hold, NOT resolve
    //    to the element's wrong clock (this is what produced "plays from 0:00").
    expect(pendingSeekAction(video, pend, NOW + 200)).toBe('wait');
    // 3. play() opens the media: a range appears, the owed seek is applied.
    video.seekable = fakeVideo(0, [[0, 100]]).seekable;
    expect(pendingSeekAction(video, pend, NOW + 400)).toBe('apply');
    // 4. After the write lands, nothing is owed.
    video.currentTime = 42;
    expect(pendingSeekAction(video, pend, NOW + 500)).toBe('done');
  });
});
