/**
 * Settings → preview/export application sweep (parity task 1.5).
 *
 * Pins three invariants:
 *  1. Every subtitle-rendering key in DEFAULT_CLIP_SETTINGS reaches the
 *     server export via mapSubtitleSettings (perturbation diff) — a new
 *     panel control can't ship without an export mapping.
 *  2. Every one of those keys is consumed by the live preview
 *     (SubtitleOverlay reads settings.<key> — static source scan).
 *  3. Slider drags coalesce into ONE undo step (undoCoalesceHandlers
 *     pauses/resumes the zundo temporal middleware around the gesture).
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { DEFAULT_CLIP_SETTINGS } from './defaultSettings';
import { mapSubtitleSettings } from './buildExportPayload';
import {
  auditSubtitleSettingsCoverage,
  NON_SUBTITLE_SETTING_KEYS,
} from './editorQA';
import { undoCoalesceHandlers } from './undoCoalesce';
import useTimelineStore from '../stores/timelineStore';

const HERE = path.dirname(fileURLToPath(import.meta.url));

describe('settings → export mapping coverage', () => {
  const audit = auditSubtitleSettingsCoverage(DEFAULT_CLIP_SETTINGS, mapSubtitleSettings);

  it('every non-exempt DEFAULT_CLIP_SETTINGS key changes mapSubtitleSettings output', () => {
    expect(audit.missing).toEqual([]);
  });

  it('the exemption list carries no stale entries', () => {
    expect(audit.staleExemptions).toEqual([]);
  });

  it('every exemption documents where the value goes instead', () => {
    for (const [key, why] of Object.entries(NON_SUBTITLE_SETTING_KEYS)) {
      expect(typeof why === 'string' && why.length > 10, `${key} needs a reason`).toBe(true);
    }
  });

  it('mapping output keys are snake_case (backend contract)', () => {
    const mapped = mapSubtitleSettings(DEFAULT_CLIP_SETTINGS);
    for (const k of Object.keys(mapped)) {
      expect(k, `${k} must be snake_case`).toMatch(/^[a-z0-9]+(_[a-z0-9]+)*$/);
    }
  });
});

describe('settings → preview application', () => {
  it('SubtitleOverlay consumes every covered subtitle setting', () => {
    const src = readFileSync(
      path.join(HERE, '../components/SubtitleOverlay.jsx'), 'utf8');
    const audit = auditSubtitleSettingsCoverage(DEFAULT_CLIP_SETTINGS, mapSubtitleSettings);
    for (const key of audit.covered) {
      const consumed = new RegExp(`settings\\??\\.${key}\\b`).test(src);
      expect(consumed, `SubtitleOverlay never reads settings.${key} — the preview can't reflect this control`).toBe(true);
    }
  });
});

describe('slider drags coalesce into one undo step', () => {
  it('updates between pointerdown/pointerup form a single undo snapshot', () => {
    const store = useTimelineStore;
    const temporal = store.temporal.getState();

    // Seed one item directly (setState bypasses locked-track guards)
    store.setState({
      tracks: [{ id: 'v1', type: 'video', name: 'V1', order: 0 }],
      items: [{ id: 'it1', type: 'video', trackId: 'v1', start: 0, end: 10, opacity: 1 }],
    });
    temporal.clear();

    const handlers = undoCoalesceHandlers();
    handlers.onPointerDown();
    // Simulate a drag: many onChange ticks
    for (let i = 1; i <= 25; i++) {
      store.getState().updateItem('it1', { opacity: 1 - i * 0.02 });
    }
    handlers.onPointerUp();
    // One more tracked change after resume creates the post-gesture snapshot
    store.getState().updateItem('it1', { opacity: 0.42 });

    const past = store.temporal.getState().pastStates;
    // Without coalescing this would be 26 entries; the paused gesture
    // collapses to at most 2 (pre-gesture + post-gesture boundary).
    expect(past.length).toBeLessThanOrEqual(2);
    expect(past.length).toBeGreaterThan(0);
  });
});
