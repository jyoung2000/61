import { describe, it, expect } from 'vitest';
import {
  getCropXForTime,
  keyframesToCropSegments,
  detectPositionClusters,
  buildKeyframesFromSubjectTrack,
  interpolateSubjectX,
} from './subjectTracking.js';

// Build a dense ~2 Hz subject track from a list of x samples.
const track = (xs) => xs.map((x, i) => ({ t: i * 0.5, x, source: 'face' }));

describe('getCropXForTime — crop window resolver', () => {
  // Smooth pan 30 → 70 across t=2..4, held after.
  const kf = [{ t: 0, x: 30 }, { t: 2, x: 30 }, { t: 4, x: 70 }, { t: 6, x: 70 }];

  it('rides the smooth track when no segment is pinned', () => {
    expect(getCropXForTime(3, [], kf)).toBeCloseTo(interpolateSubjectX(kf, 3), 5);
  });

  it('ignores an auto (non-override) segment instead of freezing on it', () => {
    const segs = [{ startTime: 0, endTime: 6, cropX: 50, isManualOverride: false }];
    // At t=5 the subject is at 70; must NOT return the frozen 50.
    expect(getCropXForTime(5, segs, kf)).toBeCloseTo(interpolateSubjectX(kf, 5), 5);
    expect(getCropXForTime(5, segs, kf)).not.toBe(50);
  });

  it('honors a manual override only inside its window', () => {
    const segs = [
      { startTime: 0, endTime: 3, cropX: 30, isManualOverride: false },
      { startTime: 3, endTime: 5, cropX: 85, isManualOverride: true },
      { startTime: 5, endTime: 6, cropX: 30, isManualOverride: false },
    ];
    expect(getCropXForTime(4, segs, kf)).toBe(85);           // inside the pin
    expect(getCropXForTime(1, segs, kf)).toBeCloseTo(interpolateSubjectX(kf, 1), 5); // outside → smooth
  });

  it('past the clip end returns the last segment when it is pinned', () => {
    const segs = [{ startTime: 0, endTime: 6, cropX: 85, isManualOverride: true }];
    expect(getCropXForTime(99, segs, kf)).toBe(85);
  });

  it('falls back to a segment value only when the smooth track yields no number', () => {
    const segs = [{ startTime: 0, endTime: 6, cropX: 42, isManualOverride: false }];
    // A 1-element keyframe list with no x makes interpolateSubjectX return undefined.
    expect(getCropXForTime(3, segs, [{}])).toBe(42);
  });

  it('ultimate fallback is 50 when nothing is available', () => {
    expect(getCropXForTime(3, [], [])).toBe(50);
  });
});

describe('keyframesToCropSegments — settled cluster centers + debounce', () => {
  it('§2: a hold reads the cluster center (not the mid-transition sample) and the preview rides smooth', () => {
    const t = [];
    for (let s = 0; s <= 24; s++) {
      const tt = s * 0.5;
      let x;
      if (tt < 4) x = 30;                       // speaker A
      else if (tt < 6) x = 30 + 40 * ((tt - 4) / 2); // pan A→B
      else if (tt < 9) x = 70;                  // hold B
      else x = 30;                              // hard cut back to A
      t.push({ t: tt, x, source: 'face' });
    }
    const kf = buildKeyframesFromSubjectTrack(t, 0, 12);
    const clusters = detectPositionClusters(kf);
    const segs = keyframesToCropSegments(kf, 12, clusters);
    const centers = clusters.map((c) => c.center);

    // Every segment value is a settled cluster center, never a transitional sample.
    for (const s of segs) expect(centers).toContain(s.cropX);

    // Across the hold, the preview tracks the smooth signal exactly (no freeze).
    for (const tt of [6.0, 7.5, 8.7]) {
      expect(getCropXForTime(tt, segs, kf)).toBeCloseTo(interpolateSubjectX(kf, tt), 5);
    }
  });

  it('§3: debounce collapses noisy cluster churn into a few clean segments', () => {
    const xs = [45, 54, 49, 62, 60, 63, 61, 62, 38, 36, 35, 37, 64, 61, 63, 60, 62];
    const dur = xs.length * 0.5;
    const kf = buildKeyframesFromSubjectTrack(track(xs), 0, dur);
    const clusters = detectPositionClusters(kf);
    const segs = keyframesToCropSegments(kf, dur, clusters);
    const centers = clusters.map((c) => c.center);

    expect(segs.length).toBeLessThanOrEqual(4);   // was 6 without the debounce
    for (const s of segs) {
      expect(centers).toContain(s.cropX);          // honest cluster-center values
      expect(s.endTime - s.startTime).toBeGreaterThanOrEqual(0.4); // no sub-0.4s flip-flops
    }
  });

  it('auto segments are never flagged as manual overrides', () => {
    const xs = [30, 30, 30, 70, 70, 70];
    const kf = buildKeyframesFromSubjectTrack(track(xs), 0, 3);
    const clusters = detectPositionClusters(kf);
    const segs = keyframesToCropSegments(kf, 3, clusters);
    for (const s of segs) expect(s.isManualOverride).toBe(false);
  });

  it('produces a single segment when there are no distinct clusters', () => {
    const kf = buildKeyframesFromSubjectTrack(track([50, 50, 50, 50, 50, 50]), 0, 3);
    const segs = keyframesToCropSegments(kf, 3, null);
    expect(segs.length).toBe(1);
  });

  it('returns [] for empty input', () => {
    expect(keyframesToCropSegments([], 12, null)).toEqual([]);
    expect(keyframesToCropSegments([{ t: 0, x: 50 }], 0, null)).toEqual([]);
  });
});
