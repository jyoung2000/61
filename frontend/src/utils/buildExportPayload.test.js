/**
 * buildPlaybackPayload tests (parity fix 1.4 — speed pitch agreement).
 *
 * The a1 audio item and video item properties take priority over the
 * store globals, and preserve_pitch only travels with a real speed
 * change (backend default = varispeed, matching the preview element and
 * the client export's AudioBufferSourceNode).
 */
import { describe, it, expect } from 'vitest';
import { buildPlaybackPayload } from './buildExportPayload';

describe('buildPlaybackPayload', () => {
  it('returns an empty object at defaults', () => {
    expect(buildPlaybackPayload({ items: [], volume: 100, speed: 1.0 })).toEqual({});
  });

  it('sends speed from the video item', () => {
    const items = [{ type: 'video', speed: 2 }];
    expect(buildPlaybackPayload({ items, volume: 100, speed: 1 })).toEqual({ speed: 2 });
  });

  it('a1 audio item overrides the video item and the store globals', () => {
    const items = [
      { type: 'video', speed: 2, volume: 0.5 },
      { type: 'audio', trackId: 'a1', speed: 1.5, volume: 0.8 },
      { type: 'audio', trackId: 'a2', speed: 4, volume: 0.1 }, // other tracks ignored
    ];
    expect(buildPlaybackPayload({ items, volume: 100, speed: 1 }))
      .toEqual({ speed: 1.5, volume: 0.8 });
  });

  it('falls back to store globals when no items carry values', () => {
    expect(buildPlaybackPayload({ items: [], volume: 50, speed: 0.75 }))
      .toEqual({ volume: 0.5, speed: 0.75 });
  });

  it('sends preserve_pitch only with a speed change', () => {
    const withSpeed = [{ type: 'video', speed: 2, preservePitch: true }];
    expect(buildPlaybackPayload({ items: withSpeed, volume: 100, speed: 1 }))
      .toEqual({ speed: 2, preserve_pitch: true });

    // preservePitch without a speed change is a no-op — don't send it
    const noSpeed = [{ type: 'video', speed: 1, preservePitch: true }];
    expect(buildPlaybackPayload({ items: noSpeed, volume: 100, speed: 1 })).toEqual({});
  });

  it('default is varispeed: no preserve_pitch key unless opted in', () => {
    const items = [{ type: 'video', speed: 2 }];
    const payload = buildPlaybackPayload({ items, volume: 100, speed: 1 });
    expect('preserve_pitch' in payload).toBe(false);
  });
});
