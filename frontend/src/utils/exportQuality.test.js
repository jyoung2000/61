/**
 * Export quality single-source-of-truth tests (parity fix 1.1).
 *
 * The 1440p bug: ExportDialog offered 1440p from a local preset table
 * while defaultSettings.EXPORT_QUALITIES and the backend quality tables
 * didn't know it — the server silently exported 1080p. These tests pin
 * the invariant that every quality surface derives from
 * defaultSettings.js and matches the backend table
 * (clip_exporter.py ASPECT_RATIO_DIMS_BY_QUALITY — mirrored in
 * backend/tests/test_export_pipeline.py EXPECTED_DIMS).
 */
import { describe, it, expect } from 'vitest';
import {
  EXPORT_QUALITIES,
  EXPORT_QUALITY_PRESETS,
  EXPORT_DIMS_BY_QUALITY,
  getExportDims,
} from './defaultSettings';
import { QUALITY_PRESETS } from '../components/ExportDialog';

// Canonical table — keep in sync with backend EXPECTED_DIMS.
const BACKEND_DIMS = {
  '720p':  { '16:9': [1280, 720],  '9:16': [720, 1280],  '1:1': [720, 720],   '4:5': [720, 900] },
  '1080p': { '16:9': [1920, 1080], '9:16': [1080, 1920], '1:1': [1080, 1080], '4:5': [1080, 1350] },
  '1440p': { '16:9': [2560, 1440], '9:16': [1440, 2560], '1:1': [1440, 1440], '4:5': [1440, 1800] },
  '4k':    { '16:9': [3840, 2160], '9:16': [2160, 3840], '1:1': [2160, 2160], '4:5': [2160, 2700] },
};

describe('export quality single source of truth', () => {
  it('EXPORT_QUALITIES includes 1440p between 1080p and 4k', () => {
    expect(EXPORT_QUALITIES).toEqual(['720p', '1080p', '1440p', '4k']);
  });

  it('every quality has a preset and a dims row (no partial tables)', () => {
    expect(Object.keys(EXPORT_QUALITY_PRESETS).sort()).toEqual([...EXPORT_QUALITIES].sort());
    expect(Object.keys(EXPORT_DIMS_BY_QUALITY).sort()).toEqual([...EXPORT_QUALITIES].sort());
  });

  it('ExportDialog presets are exactly EXPORT_QUALITIES, in order', () => {
    expect(QUALITY_PRESETS.map((p) => p.id)).toEqual(EXPORT_QUALITIES);
    // and each preset carries the shared label/dims (not a local copy)
    for (const p of QUALITY_PRESETS) {
      expect(p.label).toBe(EXPORT_QUALITY_PRESETS[p.id].label);
      expect(p.w).toBe(EXPORT_QUALITY_PRESETS[p.id].w);
      expect(p.h).toBe(EXPORT_QUALITY_PRESETS[p.id].h);
    }
  });

  it('dims match the backend table for every quality × aspect', () => {
    for (const [quality, aspects] of Object.entries(BACKEND_DIMS)) {
      for (const [ar, [w, h]] of Object.entries(aspects)) {
        expect(getExportDims(quality, ar), `${quality} ${ar}`).toEqual({ w, h });
      }
    }
  });

  it('getExportDims falls back sanely (unknown quality → 1080p, no aspect → 16:9)', () => {
    expect(getExportDims('999p', '9:16')).toEqual({ w: 1080, h: 1920 });
    expect(getExportDims('1440p', null)).toEqual({ w: 2560, h: 1440 });
    expect(getExportDims('1440p', 'bogus')).toEqual({ w: 2560, h: 1440 });
  });

  it('all output dims are even (encoder requirement)', () => {
    for (const aspects of Object.values(EXPORT_DIMS_BY_QUALITY)) {
      for (const [w, h] of Object.values(aspects)) {
        expect(w % 2).toBe(0);
        expect(h % 2).toBe(0);
      }
    }
  });
});
