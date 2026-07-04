// @vitest-environment jsdom
/**
 * Responsive smoke tests (3.3) — phone 390×844, tablet 820×1180,
 * desktop 1440×900. Renders the real Timeline + PropertiesPanel against
 * the real store and asserts: the timeline mounts, a clip can be
 * selected, the inspector opens with the selection, and the editor
 * containers don't force horizontal overflow (jsdom does no layout, so
 * overflow is asserted structurally: no fixed pixel widths wider than
 * the viewport on the top-level editor containers).
 */
import { describe, it, expect, beforeEach, beforeAll, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import useTimelineStore from '../stores/timelineStore';
import Timeline from './Timeline';
import PropertiesPanel from './PropertiesPanel';
import BottomSheet from './BottomSheet';

const VIEWPORTS = [
  ['phone', 390, 844],
  ['tablet', 820, 1180],
  ['desktop', 1440, 900],
];

beforeAll(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  // jsdom gaps the editor code expects
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
  if (!window.HTMLCanvasElement.prototype.getContext) {
    window.HTMLCanvasElement.prototype.getContext = () => null;
  } else {
    // jsdom throws "not implemented" — the Timeline guards a null ctx
    vi.spyOn(window.HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  }
  global.fetch = vi.fn(() => Promise.resolve({ ok: false, json: () => Promise.resolve([]) }));
});

function setViewport(w, h) {
  Object.defineProperty(window, 'innerWidth', { value: w, configurable: true, writable: true });
  Object.defineProperty(window, 'innerHeight', { value: h, configurable: true, writable: true });
  window.matchMedia = (q) => {
    const mobile = q.includes('max-width: 767');
    const tablet = q.includes('min-width: 768') && q.includes('max-width: 1023');
    return {
      matches: mobile ? w < 768 : tablet ? (w >= 768 && w < 1024) : false,
      media: q,
      addEventListener() {}, removeEventListener() {},
      addListener() {}, removeListener() {},
    };
  };
  window.dispatchEvent(new Event('resize'));
}

function seedStore() {
  useTimelineStore.setState({
    tracks: [
      { id: 'v1', type: 'video', name: 'V1', order: 0 },
      { id: 'a1', type: 'audio', name: 'A1', order: 1 },
    ],
    items: [
      { id: 'clip1', type: 'video', trackId: 'v1', start: 0, end: 10, speed: 1 },
      { id: 'aud1', type: 'audio', trackId: 'a1', start: 0, end: 10, speed: 1 },
    ],
    duration: 10,
    selectedItemId: null,
    selectedItemIds: [],
  });
}

let container;
let root;

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  seedStore();
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

describe.each(VIEWPORTS)('editor smoke @ %s', (name, w, h) => {
  it('timeline renders with tracks and a clip can be selected', async () => {
    setViewport(w, h);
    await render(<Timeline onSeek={() => {}} />);
    // Track header rail is DOM — both tracks visible
    expect(container.querySelectorAll('.ve-multi-timeline__track-header').length).toBe(2);
    // Canvas mounted
    expect(container.querySelector('.ve-multi-timeline__canvas')).toBeTruthy();
    // Selection through the store (pointer hit-testing needs real canvas
    // layout, which jsdom doesn't do)
    await act(async () => {
      useTimelineStore.getState().setSelectedItemId('clip1');
    });
    expect(useTimelineStore.getState().selectedItemId).toBe('clip1');
    // No editor container demands more width than the viewport
    for (const el of container.querySelectorAll('[style]')) {
      const px = parseFloat(el.style.minWidth);
      if (Number.isFinite(px)) expect(px).toBeLessThanOrEqual(w);
    }
  });

  it('inspector opens for the selection with context header + tabs', async () => {
    setViewport(w, h);
    await act(async () => {
      useTimelineStore.getState().setSelectedItemId('clip1');
    });
    await render(<PropertiesPanel settings={{}} onSettingsChange={() => {}} />);
    expect(container.querySelector('.ve-properties__header--sticky')).toBeTruthy();
    const tabs = [...container.querySelectorAll('.ve-properties__tab')].map((b) => b.textContent);
    expect(tabs).toEqual(['Clip', 'Effects', 'Audio']);
  });

  it('empty state asks for a selection', async () => {
    setViewport(w, h);
    await render(<PropertiesPanel settings={{}} />);
    expect(container.textContent).toContain('Select a clip to edit');
  });
});

describe('mobile bottom sheet', () => {
  it('renders children with a drag handle at phone size', async () => {
    setViewport(390, 844);
    await render(
      <BottomSheet title="Inspector"><div data-testid="sheet-child">hello</div></BottomSheet>,
    );
    expect(container.querySelector('.ve-bottom-sheet')).toBeTruthy();
    expect(container.querySelector('.ve-bottom-sheet__grabber')).toBeTruthy();
    expect(container.textContent).toContain('hello');
  });
});
