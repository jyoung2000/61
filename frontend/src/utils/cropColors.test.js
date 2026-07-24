import { describe, it, expect } from 'vitest';
import { cropColorAt, CROP_HUE_SPAN } from './cropColors';

// Pull the hue out of an "hsla(H, S%, L%, A)" string.
const hueOf = (s) => Number(/hsla\((\d+)/.exec(s)[1]);

describe('cropColorAt — crop % → colour', () => {
  it('is deterministic: the same % always yields the same colour', () => {
    // The bug this guards: two "34%" crop segments rendered different colours.
    expect(cropColorAt(34)).toBe(cropColorAt(34));
    const a = cropColorAt(34), b = cropColorAt(34.0), c = cropColorAt(34);
    expect(a).toBe(b);
    expect(b).toBe(c);
  });

  it('spans the full hue range across 0–100 %', () => {
    expect(hueOf(cropColorAt(0))).toBe(0);              // left  → red
    expect(hueOf(cropColorAt(100))).toBe(CROP_HUE_SPAN); // right → violet-blue
    expect(hueOf(cropColorAt(50))).toBe(Math.round(0.5 * CROP_HUE_SPAN)); // centre → green
  });

  it('is monotonic in % (a smooth sweep, no wrap back to red)', () => {
    let prev = -1;
    for (let p = 0; p <= 100; p += 5) {
      const h = hueOf(cropColorAt(p));
      expect(h).toBeGreaterThanOrEqual(prev);
      prev = h;
    }
    expect(prev).toBeLessThanOrEqual(360); // never wraps past a full circle
  });

  it('rounds identical labels to one hue: any % that shows "34%" is one colour', () => {
    // Labels are `${Math.round(cropX)}%`, so every cropX in [33.5, 34.5) reads
    // "34%"; their hues must be visually identical (≤3° apart).
    const hues = [33.5, 33.9, 34.0, 34.4].map((p) => hueOf(cropColorAt(p)));
    expect(Math.max(...hues) - Math.min(...hues)).toBeLessThanOrEqual(3);
  });

  it('clamps out-of-range and non-finite input to the valid band', () => {
    expect(hueOf(cropColorAt(-20))).toBe(0);
    expect(hueOf(cropColorAt(140))).toBe(CROP_HUE_SPAN);
    expect(cropColorAt(NaN)).toBe(cropColorAt(50)); // NaN → centre default
  });

  it('honours the alpha argument', () => {
    expect(cropColorAt(34, 0.75)).toBe('hsla(95, 60%, 38%, 0.75)');
    expect(cropColorAt(34)).toBe('hsla(95, 60%, 38%, 1)');
  });
});
