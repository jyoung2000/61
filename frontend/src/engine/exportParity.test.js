/**
 * Preview ↔ export parity tests for the pure math in ExportEngine.
 *
 * The WYSIWYG contract is "same renderFrame() for preview and export";
 * these tests codify the *time-mapping* and *audio-gain* halves of that
 * contract, which historically diverged (speed ignored on export,
 * fades not exported, muted clips audible).
 */
import { describe, it, expect } from 'vitest';
import { timelineToSourceTime, fadeGainAt, buildGainAutomation } from './ExportEngine';
import {
  getCurrentWordIndex,
  computeSpeakerRates,
  ANTICIPATION_S,
  AUDIO_BUFFER_S,
} from '../utils/activeWordTiming';

describe('timelineToSourceTime (frame-seek + audio scheduling share this)', () => {
  it('maps 1x clips as trimStart + elapsed', () => {
    const clip = { start: 10, end: 20, trimStart: 5, speed: 1 };
    expect(timelineToSourceTime(clip, 10)).toBeCloseTo(5);
    expect(timelineToSourceTime(clip, 15)).toBeCloseTo(10);
  });

  it('steps through source at 2x for a 2x clip (source-frame correspondence)', () => {
    const clip = { start: 10, end: 15, trimStart: 3, speed: 2 };
    // At timeline t=10 (clip start) → source 3
    expect(timelineToSourceTime(clip, 10)).toBeCloseTo(3);
    // 1 timeline second later → 2 source seconds consumed
    expect(timelineToSourceTime(clip, 11)).toBeCloseTo(5);
    // Clip end (5 timeline s) → 10 source seconds consumed
    expect(timelineToSourceTime(clip, 15)).toBeCloseTo(13);
  });

  it('supports slow motion (0.5x consumes half the source)', () => {
    const clip = { start: 0, end: 10, trimStart: 0, speed: 0.5 };
    expect(timelineToSourceTime(clip, 10)).toBeCloseTo(5);
  });

  it('defaults missing speed/trimStart to 1x/0', () => {
    const clip = { start: 2, end: 4 };
    expect(timelineToSourceTime(clip, 3)).toBeCloseTo(1);
  });
});

describe('fadeGainAt (audio twin of RenderEngine opacity fades)', () => {
  const clip = { start: 0, end: 10, fadeIn: 2, fadeOut: 4 };

  it('ramps 0 → 1 across fadeIn exactly like elapsed/fadeIn', () => {
    expect(fadeGainAt(clip, 0)).toBeCloseTo(0);
    expect(fadeGainAt(clip, 1)).toBeCloseTo(0.5);
    expect(fadeGainAt(clip, 2)).toBeCloseTo(1);
  });

  it('ramps 1 → 0 across fadeOut exactly like remaining/fadeOut', () => {
    expect(fadeGainAt(clip, 6)).toBeCloseTo(1);
    expect(fadeGainAt(clip, 8)).toBeCloseTo(0.5);
    expect(fadeGainAt(clip, 10)).toBeCloseTo(0);
  });

  it('is 1 in the body of the clip and multiplies overlapping fades', () => {
    expect(fadeGainAt(clip, 5)).toBeCloseTo(1);
    const short = { start: 0, end: 2, fadeIn: 2, fadeOut: 2 };
    // Midpoint: fadeIn gives 0.5, fadeOut gives 0.5 → product 0.25
    expect(fadeGainAt(short, 1)).toBeCloseTo(0.25);
  });
});

describe('buildGainAutomation', () => {
  it('emits base volume with no fades', () => {
    const clip = { start: 0, end: 10, volume: 0.8 };
    const pts = buildGainAutomation(clip, 0, 10);
    expect(pts.length).toBeGreaterThanOrEqual(2);
    for (const p of pts) expect(p.value).toBeCloseTo(0.8);
  });

  it('muted clips automate to zero everywhere', () => {
    const clip = { start: 0, end: 10, volume: 1, muted: true, fadeIn: 1 };
    const pts = buildGainAutomation(clip, 0, 10);
    for (const p of pts) expect(p.value).toBe(0);
  });

  it('produces the fade breakpoints in offline-context time', () => {
    const clip = { start: 5, end: 15, volume: 1, fadeIn: 2, fadeOut: 3 };
    const pts = buildGainAutomation(clip, 5, 15);
    // ctx time 0 = timeline 5 (clip start): gain 0
    expect(pts[0]).toEqual({ time: 0, value: 0 });
    const atFadeInEnd = pts.find(p => Math.abs(p.time - 2) < 1e-9);
    expect(atFadeInEnd.value).toBeCloseTo(1);
    const atFadeOutStart = pts.find(p => Math.abs(p.time - 7) < 1e-9);
    expect(atFadeOutStart.value).toBeCloseTo(1);
    const last = pts[pts.length - 1];
    expect(last.time).toBeCloseTo(10);
    expect(last.value).toBeCloseTo(0);
  });

  it('picks up mid-fade when the export range starts inside a fade', () => {
    const clip = { start: 0, end: 10, volume: 1, fadeIn: 4 };
    const pts = buildGainAutomation(clip, 2, 10);
    // Export starts halfway through the fade → initial gain 0.5
    expect(pts[0].time).toBeCloseTo(0);
    expect(pts[0].value).toBeCloseTo(0.5);
  });

  it('returns [] when the clip is outside the export range', () => {
    const clip = { start: 20, end: 30, volume: 1 };
    expect(buildGainAutomation(clip, 0, 10)).toEqual([]);
  });
});

describe('shared active-word timing (RenderEngine ↔ SubtitleOverlay)', () => {
  it('word-timestamp branch matches on timeline-absolute words', () => {
    const seg = {
      subtitleText: 'hello brave world',
      start: 10, end: 13, speaker: 'S1',
      words: [
        { start: 10.0, end: 11.0 },
        { start: 11.0, end: 12.0 },
        { start: 12.0, end: 13.0 },
      ],
    };
    const rates = { S1: 3.0 };
    // At 10.5 with anticipation 0.10 and audio buffer 0.12 → adjusted 10.48
    expect(getCurrentWordIndex(seg, 10.5, rates)).toBe(0);
    expect(getCurrentWordIndex(seg, 11.5, rates)).toBe(1);
    expect(getCurrentWordIndex(seg, 12.9, rates)).toBe(2);
  });

  it('detects clip-relative word timestamps automatically', () => {
    const seg = {
      subtitleText: 'one two',
      start: 100, end: 102, speaker: 'S1',
      words: [ { start: 0.0, end: 1.0 }, { start: 1.0, end: 2.0 } ],
    };
    expect(getCurrentWordIndex(seg, 100.5, {})).toBe(0);
    expect(getCurrentWordIndex(seg, 101.5, {})).toBe(1);
  });

  it('computeSpeakerRates aggregates words/sec per speaker', () => {
    const rates = computeSpeakerRates([
      { text: 'a b c', start: 0, end: 1, speaker: 'A' },   // 3 wps
      { text: 'd e f g h i', start: 1, end: 3, speaker: 'A' }, // 3 wps
      { text: 'x', start: 0, end: 2, speaker: 'B' },       // 0.5 wps
    ]);
    expect(rates.A).toBeCloseTo(3.0);
    expect(rates.B).toBeCloseTo(0.5);
  });

  it('exports the constants preview and export must agree on', () => {
    expect(ANTICIPATION_S).toBeCloseTo(0.10);
    expect(AUDIO_BUFFER_S).toBeCloseTo(0.12);
  });
});
