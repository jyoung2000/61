/**
 * Preview-player control helpers — touch scrubbing geometry, time clamping,
 * control visibility, and touch detection (the logic behind the mobile/tablet
 * player fix). Pure functions, no <video> render required.
 */
import { describe, it, expect } from 'vitest';
import { seekPct, clampTime, controlsVisible, detectTouch } from './playerControls';

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
