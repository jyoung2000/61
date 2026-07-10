import { describe, it, expect } from 'vitest';
import { computeCoverCrop, computeThumbStops } from './filmstrip';

// computeCoverCrop feeds createImageBitmap's source rect: the largest
// centered crop of a sprite tile whose aspect matches the target thumb box.
// Getting it wrong shows as stretched / letterboxed timeline thumbnails.
describe('computeCoverCrop', () => {
  it('is a no-op when tile and target share an aspect', () => {
    expect(computeCoverCrop(160, 90, 96, 54)).toEqual({ sx: 0, sy: 0, sw: 160, sh: 90 });
  });

  it('crops the sides of a wide tile for a squarer target', () => {
    const c = computeCoverCrop(160, 90, 90, 90); // square target
    expect(c.sh).toBe(90);                       // full height kept
    expect(c.sw).toBe(90);                       // width cropped to match
    expect(c.sx).toBe(Math.floor((160 - 90) / 2));
    expect(c.sy).toBe(0);
  });

  it('crops top/bottom of a tall tile for a wider target', () => {
    const c = computeCoverCrop(90, 160, 160, 90); // 9:16 tile → 16:9 box
    expect(c.sw).toBe(90);                        // full width kept
    expect(c.sh).toBe(Math.round((90 / 160) * 90));
    expect(c.sx).toBe(0);
    expect(c.sy).toBe(Math.floor((160 - c.sh) / 2));
  });

  it('never returns a zero-size rect', () => {
    const c = computeCoverCrop(2, 2, 1000, 2);
    expect(c.sw).toBeGreaterThan(0);
    expect(c.sh).toBeGreaterThan(0);
  });

  it('stays inside the tile bounds', () => {
    for (const [tw, th, w, h] of [[160, 90, 30, 56], [160, 66, 100, 48], [96, 96, 200, 40]]) {
      const c = computeCoverCrop(tw, th, w, h);
      expect(c.sx).toBeGreaterThanOrEqual(0);
      expect(c.sy).toBeGreaterThanOrEqual(0);
      expect(c.sx + c.sw).toBeLessThanOrEqual(tw);
      expect(c.sy + c.sh).toBeLessThanOrEqual(th);
    }
  });
});

describe('computeThumbStops', () => {
  it('produces viewport-density stops regardless of clip length', () => {
    const stops = computeThumbStops(0, 60, 10, 96); // 600px of clip, ~96px thumbs
    expect(stops.length).toBeGreaterThanOrEqual(2);
    expect(stops.length).toBeLessThanOrEqual(120);
    expect(stops[0]).toBeGreaterThan(0);
    expect(stops[stops.length - 1]).toBeLessThan(60);
  });
});
