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

  it('anchors the spectrum: 0 % is red, 100 % is blue, 50 % is magenta', () => {
    expect(CROP_HUE_SPAN).toBe(120);                     // the SHORT way round the wheel
    expect(hueOf(cropColorAt(0))).toBe(0);               // 0 %   → red
    expect(hueOf(cropColorAt(100))).toBe(240);           // 100 % → blue
    expect(hueOf(cropColorAt(50))).toBe(300);            // 50 %  → magenta (red+blue), not green
  });

  it('never passes through yellow/green/cyan (the regression: 35 % olive, 62 % green)', () => {
    // Every hue must live on the red↔blue side of the wheel: 240°–360° or 0°.
    for (let p = 0; p <= 100; p += 1) {
      const h = hueOf(cropColorAt(p));
      expect(h === 0 || (h >= 240 && h <= 360)).toBe(true);
    }
  });

  it('is monotonic in % (bluer is always higher %, no wrap past blue)', () => {
    // Hue DESCENDS 360°→240° as % rises (0 % renders as 0°, the same red).
    let prev = 361;
    for (let p = 1; p <= 100; p += 1) {
      const h = hueOf(cropColorAt(p));
      expect(h).toBeLessThanOrEqual(prev);
      expect(h).toBeGreaterThanOrEqual(240); // never runs past blue into cyan
      prev = h;
    }
    expect(prev).toBe(240);
  });

  it('rounds identical labels to one hue: any % that shows "34%" is one colour', () => {
    // Labels are `${Math.round(cropX)}%`, so every cropX in [33.5, 34.5) reads
    // "34%"; their hues must be visually identical (≤3° apart).
    const hues = [33.5, 33.9, 34.0, 34.4].map((p) => hueOf(cropColorAt(p)));
    expect(Math.max(...hues) - Math.min(...hues)).toBeLessThanOrEqual(3);
  });

  it('clamps out-of-range and non-finite input to the valid band', () => {
    expect(hueOf(cropColorAt(-20))).toBe(0);             // ≤0 % → red
    expect(hueOf(cropColorAt(140))).toBe(240);           // ≥100 % → blue
    expect(cropColorAt(NaN)).toBe(cropColorAt(50));      // NaN → centre default
  });

  it('honours the alpha argument', () => {
    expect(cropColorAt(34, 0.75)).toBe('hsla(319, 60%, 38%, 0.75)');
    expect(cropColorAt(34)).toBe('hsla(319, 60%, 38%, 1)');
  });
});
