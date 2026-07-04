/**
 * Safe-zone math tests (2.8) — the JS table must mirror the backend's
 * subtitle_formatter._PLATFORM_PROFILES exactly (1080×1920 reference,
 * proportional scaling).
 */
import { describe, it, expect } from 'vitest';
import { getSafeZoneMargins, getSafeZonePercents, PLATFORM_PRESETS, PLATFORM_PROFILES } from './safeZones';
import { EXPORT_QUALITIES } from './defaultSettings';

describe('getSafeZoneMargins', () => {
  it('returns reference values verbatim at 1080×1920', () => {
    expect(getSafeZoneMargins('tiktok', 1080, 1920)).toEqual({
      topPx: 140, bottomPx: 324, leftPx: 60, rightPx: 164, recommendedPosition: 'middle',
    });
    expect(getSafeZoneMargins('horizontal', 1080, 1920)).toEqual({
      topPx: 0, bottomPx: 80, leftPx: 40, rightPx: 40, recommendedPosition: 'bottom',
    });
  });

  it('scales proportionally (matches backend rounding: round(px * scale))', () => {
    // 1440×2560: scale_w = 1440/1080 = 1.333…, scale_h = 2560/1920 = 1.333…
    const m = getSafeZoneMargins('reels', 1440, 2560);
    expect(m.topPx).toBe(Math.round(120 * (2560 / 1920)));
    expect(m.bottomPx).toBe(Math.round(350 * (2560 / 1920)));
    expect(m.leftPx).toBe(Math.round(60 * (1440 / 1080)));
  });

  it('unknown platform is a zero-margin no-op', () => {
    expect(getSafeZoneMargins('myspace', 1080, 1920)).toEqual({
      topPx: 0, bottomPx: 0, leftPx: 0, rightPx: 0, recommendedPosition: 'bottom',
    });
  });
});

describe('platform presets', () => {
  it('every chip names a real profile, aspect and quality', () => {
    for (const p of PLATFORM_PRESETS) {
      expect(PLATFORM_PROFILES[p.profile], p.id).toBeTruthy();
      expect(['16:9', '9:16', '1:1', '4:5']).toContain(p.aspect);
      expect(EXPORT_QUALITIES).toContain(p.quality);
    }
  });
});

describe('getSafeZonePercents', () => {
  it('sums below 100% so the safe frame is never inverted', () => {
    for (const name of Object.keys(PLATFORM_PROFILES)) {
      const z = getSafeZonePercents(name, 1080, 1920);
      expect(z.top + z.bottom).toBeLessThan(100);
      expect(z.left + z.right).toBeLessThan(100);
    }
  });
});
