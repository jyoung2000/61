/**
 * Effects coverage checklist test (audit Phase 1 item 7).
 *
 * Asserts the feature parity matrix:
 *  1. covers every property the editor panels expose (REQUIRED_KEYS),
 *  2. declares a valid status for all three render paths per feature,
 *  3. documents every non-'ok' status with a note (no silent gaps).
 */
import { describe, it, expect } from 'vitest';
import { FEATURE_PARITY, REQUIRED_KEYS, VALID_STATUSES } from './featureParityMatrix';

describe('feature parity matrix', () => {
  it('covers every panel-exposed property', () => {
    const keys = new Set(FEATURE_PARITY.map(f => f.key));
    const missing = REQUIRED_KEYS.filter(k => !keys.has(k));
    expect(missing).toEqual([]);
  });

  it('has no duplicate rows', () => {
    const keys = FEATURE_PARITY.map(f => f.key);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it('declares a valid status for preview, client export and server export', () => {
    for (const f of FEATURE_PARITY) {
      expect(f.key, 'row missing key').toBeTruthy();
      expect(f.label, `${f.key}: missing label`).toBeTruthy();
      for (const path of ['preview', 'clientExport', 'serverExport']) {
        expect(VALID_STATUSES.has(f[path]),
          `${f.key}.${path}='${f[path]}' is not a valid status`).toBe(true);
      }
    }
  });

  it('documents every divergence (partial/gap requires a note)', () => {
    for (const f of FEATURE_PARITY) {
      const diverges = ['preview', 'clientExport', 'serverExport']
        .some(p => f[p] === 'partial' || f[p] === 'gap');
      if (diverges) {
        expect(typeof f.notes === 'string' && f.notes.length > 10,
          `${f.key} diverges but has no explanatory note`).toBe(true);
      }
    }
  });

  it('audio-affecting properties are wired into client export', () => {
    // Regression guard for the audit's Phase 1 findings: these were the
    // properties that previously rendered in preview only.
    for (const key of ['speed', 'muted', 'fadeIn', 'fadeOut', 'volume']) {
      const row = FEATURE_PARITY.find(f => f.key === key);
      expect(row.clientExport, `${key} must be 'ok' on client export`).toBe('ok');
    }
  });
});
