// @vitest-environment jsdom
/**
 * UpdateNudge: the "ClipAI was updated on the server — reload" banner. It
 * compares the hashed bundle this tab loaded against the one a FRESH
 * index.html references; a mismatch means the container was updated under a
 * still-open (or cache-stale) tab — the exact condition that read as "the new
 * settings never appeared".
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import UpdateNudge from './UpdateNudge';

let container;
let root;
let scriptEl;

const indexHtmlWith = (bundle) =>
  `<!doctype html><html><head><script type="module" crossorigin src="${bundle}"></script></head><body></body></html>`;

const mount = async () => {
  await act(async () => {
    root = createRoot(container);
    root.render(<UpdateNudge />);
  });
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
};

beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
  // Simulate the bundle this "tab" is running.
  scriptEl = document.createElement('script');
  scriptEl.setAttribute('src', '/assets/index-OLDHASH1.js');
  document.head.appendChild(scriptEl);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  container.remove();
  scriptEl.remove();
  vi.unstubAllGlobals();
});

describe('UpdateNudge', () => {
  it('stays hidden while the served bundle matches this tab', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, text: async () => indexHtmlWith('/assets/index-OLDHASH1.js'),
    })));
    await mount();
    expect(document.body.textContent).not.toContain('updated on the server');
  });

  it('shows the reload banner when the server serves a NEWER bundle', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, text: async () => indexHtmlWith('/assets/index-NEWHASH2.js'),
    })));
    await mount();
    expect(document.body.textContent).toContain('ClipAI was updated on the server');
    const btn = [...document.body.querySelectorAll('button')]
      .find((b) => b.textContent.includes('Reload now'));
    expect(btn).toBeTruthy();
  });

  it('stays quiet when the check itself fails (offline)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('offline'); }));
    await mount();
    expect(document.body.textContent).not.toContain('updated on the server');
  });
});
