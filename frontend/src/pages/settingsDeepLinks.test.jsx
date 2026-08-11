// @vitest-environment jsdom
/**
 * Settings deep links.
 *
 * Settings tabs are CONDITIONALLY rendered, so a control on an inactive tab
 * is not in the DOM at all — browser Ctrl+F finds nothing and the setting
 * reads as "missing" (this cost a user three rounds hunting for Concurrent
 * Analyses, which was present and deployed the whole time). Deep links are
 * the answer: ?tab=<name>&section=<id> opens the right tab and scrolls to
 * the control. These are source scans, pinning the two ways a deep link
 * silently rots:
 *
 *   1. TAB_NAME_TO_INDEX drifting out of sync with SETTINGS_TABS (a tab
 *      inserted or reordered → every ?tab= link opens the WRONG tab, with
 *      no error anywhere);
 *   2. a section id referenced by a link disappearing from the markup.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SETTINGS = readFileSync(path.join(HERE, 'Settings.jsx'), 'utf8');
const PANEL = readFileSync(
  path.join(HERE, '..', 'components', 'BulkImportPanel.jsx'), 'utf8');

/** Labels from the SETTINGS_TABS array literal, in order. */
function tabLabels() {
  const start = SETTINGS.indexOf('const SETTINGS_TABS = [');
  expect(start, 'SETTINGS_TABS array not found').toBeGreaterThan(-1);
  const body = SETTINGS.slice(start, SETTINGS.indexOf('];', start));
  return [...body.matchAll(/'([^']+)'/g)]
    .map((m) => m[1])
    .filter((s) => s !== 'Users' || true);   // conditional entries still count
}

/** The ?tab=<name> → index map. */
function nameToIndex() {
  const line = SETTINGS.match(/const TAB_NAME_TO_INDEX = \{([^}]*)\}/);
  expect(line, 'TAB_NAME_TO_INDEX not found').toBeTruthy();
  const out = {};
  for (const m of line[1].matchAll(/'?([\w-]+)'?\s*:\s*(\d+)/g)) out[m[1]] = Number(m[2]);
  return out;
}

describe('Settings deep links', () => {
  it('every ?tab= name maps to the tab whose label it slugifies', () => {
    const labels = tabLabels();
    const map = nameToIndex();
    const slug = (s) => s.toLowerCase().replace(/&/g, '').replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');

    for (const [name, idx] of Object.entries(map)) {
      expect(labels[idx], `?tab=${name} points at index ${idx}, which has no label`)
        .toBeTruthy();
      expect(slug(labels[idx]), `?tab=${name} → index ${idx} is "${labels[idx]}"`)
        .toBe(name);
    }
  });

  it('Concurrent Analyses renders on BOTH the AI Provider and Advanced tabs', () => {
    // The control was reported "missing" four times while it lived on one
    // tab only — it is now a shared card rendered on the tab users actually
    // scroll (AI Provider, index 0) AND at the top of Advanced (index 4).
    expect(nameToIndex().advanced).toBe(4);
    expect(SETTINGS).toContain('const concurrencyCard = (');
    const tab0 = SETTINGS.indexOf('{settingsTab === 0 && (');
    const tab4 = SETTINGS.indexOf('{settingsTab === 4 && (');
    const tab5 = SETTINGS.indexOf('{settingsTab === 5 && (');
    const renders = [...SETTINGS.matchAll(/\{concurrencyCard\}/g)].map((m) => m.index);
    expect(renders.length).toBe(2);
    expect(tab0).toBeGreaterThan(-1);
    expect(tab4).toBeGreaterThan(-1);
    // One render inside tab 0's block, one inside tab 4's.
    expect(renders.some((i) => i > tab0 && i < tab4)).toBe(true);
    expect(renders.some((i) => i > tab4 && (tab5 === -1 || i < tab5))).toBe(true);
  });

  it('links to ?section=<id> point at ids that exist in the markup', () => {
    const sources = [SETTINGS, PANEL].join('\n');
    const linked = [...sources.matchAll(/section=([\w-]+)/g)].map((m) => m[1]);
    expect(linked.length, 'expected at least one ?section= deep link').toBeGreaterThan(0);
    for (const id of new Set(linked)) {
      // Either a real DOM id, or the ref-based legacy section.
      const hasId = new RegExp(`id="${id}"`).test(sources)
        || new RegExp(`id='${id}'`).test(sources)
        || id === 'viral-algorithm';
      expect(hasId, `?section=${id} has no matching element id`).toBe(true);
    }
  });

  it('the bulk import panel links to the concurrency control', () => {
    // The setting is invisible to browser find from any other tab, so the
    // place bulk imports run must point at it.
    expect(PANEL).toMatch(/tab=advanced&section=concurrency/);
  });
});
