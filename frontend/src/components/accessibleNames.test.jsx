// @vitest-environment jsdom
/**
 * Accessible-name coverage (polish task 1).
 *
 * 1. Rendered check: every button in the Timeline toolbar/chrome and the
 *    ToolBar has a non-empty accessible name (visible text, aria-label,
 *    or the Tooltip-injected aria-label).
 * 2. Source scan: no interactive element in the core editor chrome files
 *    carries a bare native title= — the styled Tooltip (or aria-label)
 *    replaced them all, and this pins that they can't creep back.
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import useTimelineStore from '../stores/timelineStore';
import Timeline from './Timeline';
import ToolBar from './ToolBar';

const HERE = path.dirname(fileURLToPath(import.meta.url));

beforeAll(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  if (!window.ResizeObserver) {
    window.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
  }
  if (!window.matchMedia) {
    window.matchMedia = (q) => ({
      matches: false, media: q,
      addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {},
    });
  }
  vi.spyOn(window.HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
});

let container;
let root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  useTimelineStore.setState({
    tracks: [
      { id: 'v1', type: 'video', name: 'V1', order: 0 },
      { id: 's1', type: 'subtitle', name: 'Subtitles', order: 1 },
    ],
    items: [{ id: 'clip1', type: 'video', trackId: 'v1', start: 0, end: 10 }],
    duration: 10,
  });
});

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  container.remove();
});

const render = async (el) => {
  await act(async () => {
    root = createRoot(container);
    root.render(el);
  });
};

function accessibleName(btn) {
  return (btn.getAttribute('aria-label') || btn.textContent || '').trim();
}

describe('every editor chrome button has an accessible name', () => {
  it('Timeline', async () => {
    await render(<Timeline onSeek={() => {}} />);
    const buttons = [...container.querySelectorAll('button')];
    expect(buttons.length).toBeGreaterThan(5);
    for (const btn of buttons) {
      expect(accessibleName(btn), btn.outerHTML.slice(0, 120)).not.toBe('');
    }
  });

  it('ToolBar', async () => {
    await render(<ToolBar />);
    const buttons = [...container.querySelectorAll('button')];
    expect(buttons.length).toBeGreaterThan(3);
    for (const btn of buttons) {
      expect(accessibleName(btn), btn.outerHTML.slice(0, 120)).not.toBe('');
    }
  });
});

describe('no bare native title= on interactive editor chrome', () => {
  const FILES = ['VideoEditor.jsx', 'Timeline.jsx', 'ToolBar.jsx', 'ExportDialog.jsx'];

  it.each(FILES)('%s', (file) => {
    const src = readFileSync(path.join(HERE, file), 'utf8');
    const offenders = [];
    // Any title= attribute inside a <button/select/input/textarea ...> tag
    const tagRe = /<(button|select|input|textarea)\b[^>]*?\btitle=/gs;
    let m;
    while ((m = tagRe.exec(src)) !== null) {
      offenders.push(src.slice(m.index, m.index + 90).replace(/\s+/g, ' '));
    }
    expect(offenders, `native title= on interactive element in ${file}`).toEqual([]);
  });
});
