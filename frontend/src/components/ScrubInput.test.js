/**
 * ScrubInput math tests (2.2 pro input ergonomics).
 * 1 horizontal pixel = 1 step; Shift = 10×, Alt = 0.1×; clamped to
 * min/max from the shared range tables.
 */
import { describe, it, expect } from 'vitest';
import { scrubValue, stepPrecision } from './ScrubInput';

describe('scrubValue', () => {
  it('moves one step per pixel', () => {
    expect(scrubValue(30, 5, { step: 1, min: 12, max: 72 })).toBe(35);
    expect(scrubValue(30, -5, { step: 1, min: 12, max: 72 })).toBe(25);
  });

  it('Shift = 10× step', () => {
    expect(scrubValue(30, 3, { step: 1, min: 0, max: 900, shiftKey: true })).toBe(60);
  });

  it('Alt = 0.1× step for fine control', () => {
    expect(scrubValue(1.0, 5, { step: 0.1, min: 0, max: 3, altKey: true }))
      .toBeCloseTo(1.05);
  });

  it('clamps to the shared range bounds', () => {
    expect(scrubValue(70, 100, { step: 1, min: 12, max: 72 })).toBe(72);
    expect(scrubValue(14, -100, { step: 1, min: 12, max: 72 })).toBe(12);
  });

  it('fractional steps keep their precision (no float dust)', () => {
    expect(scrubValue(0.5, 3, { step: 0.05, min: 0.1, max: 3 })).toBeCloseTo(0.65);
    expect(String(scrubValue(0.5, 3, { step: 0.05, min: 0.1, max: 3 })).length)
      .toBeLessThanOrEqual(4);
  });

  it('zero-pixel scrub only clamps (used for typed commits)', () => {
    expect(scrubValue(999, 0, { step: 1, min: 0, max: 100 })).toBe(100);
    expect(scrubValue(42, 0, { step: 1, min: 0, max: 100 })).toBe(42);
  });
});

describe('stepPrecision', () => {
  it('derives display decimals from the step', () => {
    expect(stepPrecision(1)).toBe(0);
    expect(stepPrecision(0.1)).toBe(1);
    expect(stepPrecision(0.05)).toBe(2);
    expect(stepPrecision(0)).toBe(0);
  });
});
