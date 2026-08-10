// @vitest-environment jsdom
/**
 * CompanionBrowser: path bookmarks, sequential bulk folder import (confirm →
 * shared BulkImportPanel), and "smooth local file browser" path handling —
 * Windows \\?\ verbatim normalization, working breadcrumbs, Up button, and
 * pasted paths (quoted, and file paths opening via their parent).
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import { MemoryRouter } from 'react-router-dom';
import CompanionBrowser from './CompanionBrowser';

beforeAll(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; });

let container;
let root;
let fetchCalls;

const DEFAULT_ROOTS = [{ path: 'D:\\media', name: 'media', exists: true }];
const LISTING = {
  path: 'D:\\media',
  entries: [
    { name: 'Anime', path: 'D:\\media\\Anime', is_dir: true, size: 0, ext: '', mtime_ms: 0, created_ms: 0 },
    { name: 'ep1.mp4', path: 'D:\\media\\ep1.mp4', is_dir: false, size: 9000, ext: 'mp4', mtime_ms: 1, created_ms: 1 },
    { name: 'ep2.mkv', path: 'D:\\media\\ep2.mkv', is_dir: false, size: 9000, ext: 'mkv', mtime_ms: 1, created_ms: 1 },
  ],
};

// Mutable per-test server state the fetch mock serves from.
let mockRoots;
let listingFor;       // (path) => { status, body }
let serverBookmarks;
let bulkProgress;

function mockFetch(url, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  fetchCalls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });
  const json = (data, status = 200) => Promise.resolve({
    ok: status < 400, status,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
  });

  if (url.startsWith('/api/providers/companion-files/roots')) {
    return json({ companions: [{ host_id: 'h1', name: '4070', online: true, roots: mockRoots }] });
  }
  if (url.startsWith('/api/providers/companion-files/bookmarks')) {
    if (method === 'POST') {
      const b = JSON.parse(opts.body);
      serverBookmarks = [{ path: b.path, name: b.name || b.path, is_dir: true, added_ms: 1 },
        ...serverBookmarks.filter((m) => m.path !== b.path)];
    }
    if (method === 'DELETE') {
      const path = decodeURIComponent(url.match(/[?&]path=([^&]*)/)[1]);
      serverBookmarks = serverBookmarks.filter((m) => m.path !== path);
    }
    return json({ ok: true, bookmarks: serverBookmarks });
  }
  if (url.startsWith('/api/providers/companion-files/list')) {
    const path = decodeURIComponent(url.match(/[?&]path=([^&]*)/)[1]);
    const r = listingFor(path);
    return json(r.body, r.status);
  }
  if (url.startsWith('/api/providers/companion-files/import-folder/active')) return json({ bulk_id: null });
  if (url.startsWith('/api/providers/companion-files/import-folder/progress')) return json(bulkProgress);
  if (url.startsWith('/api/providers/companion-files/import-folder')) {
    return json({ ok: true, bulk_id: 'bulk1', total: 2, folder: 'D:\\media' });
  }
  return json({}, 404);
}

beforeEach(() => {
  fetchCalls = [];
  serverBookmarks = [];
  mockRoots = DEFAULT_ROOTS;
  // Files 400 like the real Companion ("not a directory"); folders list.
  listingFor = (path) => (/\.(mp4|mkv|mov)$/i.test(path)
    ? { status: 400, body: { detail: 'not a directory' } }
    : { status: 200, body: LISTING });
  bulkProgress = { status: 'complete', total: 2, done: 2, ok: 2, failed: 0, current: -1, folder_name: 'media', items: [] };
  vi.stubGlobal('fetch', vi.fn(mockFetch));
  localStorage.clear();
  container = document.createElement('div');
  document.body.appendChild(container);
});

afterEach(async () => {
  await act(async () => { root?.unmount(); });
  container.remove();
  vi.unstubAllGlobals();
});

const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
const render = async (el) => {
  await act(async () => { root = createRoot(container); root.render(<MemoryRouter>{el}</MemoryRouter>); });
  await flush();
};
const click = (el) => act(async () => { el.click(); await Promise.resolve(); });
const byText = (text, sel = 'button') =>
  [...document.body.querySelectorAll(sel)].find((b) => b.textContent.trim() === text);
const setInput = async (el, value) => {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  await act(async () => {
    setter.call(el, value);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
};
const pressEnter = (el) => act(async () => {
  el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  await Promise.resolve();
});

describe('CompanionBrowser bookmarks', () => {
  it('shows a Bookmarks section on the home view and navigates on click', async () => {
    serverBookmarks = [{ path: 'D:\\media\\Anime', name: 'Anime', is_dir: true, added_ms: 1 }];
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);

    expect(document.body.textContent).toContain('Bookmarks');
    expect(document.body.textContent).toContain('Shared folders');
    const row = [...document.body.querySelectorAll('[role="button"]')]
      .find((r) => r.textContent.includes('D:\\media\\Anime'));
    expect(row).toBeTruthy();
    await click(row);
    await flush();
    expect(fetchCalls.some((c) => c.url.includes('/companion-files/list')
      && c.url.includes(encodeURIComponent('D:\\media\\Anime')))).toBe(true);
  });

  it('displays a legacy \\\\?\\-prefixed bookmark clean and navigates with the plain path', async () => {
    serverBookmarks = [{ path: '\\\\?\\D:\\media\\Anime', name: 'Anime', is_dir: true, added_ms: 1 }];
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);

    expect(document.body.textContent).toContain('D:\\media\\Anime');
    expect(document.body.textContent).not.toContain('\\\\?\\');
    const row = [...document.body.querySelectorAll('[role="button"]')]
      .find((r) => r.textContent.includes('D:\\media\\Anime'));
    await click(row);
    await flush();
    const nav = fetchCalls.find((c) => c.url.includes('/companion-files/list'));
    expect(nav.url).toContain(encodeURIComponent('D:\\media\\Anime'));
    expect(nav.url).not.toContain(encodeURIComponent('\\\\?\\'));
  });

  it('stars a folder row (POST) and un-stars it (DELETE)', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();

    // The toolbar has a star for the CURRENT folder too — take the folder
    // row's star, which renders after it.
    const star = [...document.body.querySelectorAll('[title="Bookmark this folder"]')].pop();
    expect(star).toBeTruthy();
    await click(star);
    await flush();
    const post = fetchCalls.find((c) => c.method === 'POST' && c.url.endsWith('/companion-files/bookmarks'));
    expect(post.body).toMatchObject({ host_id: 'h1', path: 'D:\\media\\Anime', name: 'Anime' });

    const unstar = document.body.querySelector('[title="Remove bookmark"]');
    expect(unstar).toBeTruthy();
    await click(unstar);
    await flush();
    expect(fetchCalls.some((c) => c.method === 'DELETE' && c.url.includes('/companion-files/bookmarks'))).toBe(true);
  });
});

describe('CompanionBrowser path smoothness', () => {
  it('normalizes \\\\?\\ listings into clean, working breadcrumbs', async () => {
    mockRoots = [{ path: 'C:\\', name: 'C:\\', exists: true }];
    listingFor = () => ({
      status: 200,
      body: {
        path: '\\\\?\\C:\\Users\\jalon\\Videos',
        entries: [{
          name: 'Videos', path: '\\\\?\\C:\\Users\\jalon\\Videos',
          is_dir: true, size: 0, ext: '', mtime_ms: 0, created_ms: 0,
        }],
      },
    });
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('C:\\')));
    await flush();
    await click([...document.body.querySelectorAll('[data-fbpath]')].find((r) => r.textContent.includes('Videos')));
    await flush();

    // No "?" crumb, no verbatim prefix anywhere on screen.
    expect(document.body.textContent).not.toContain('\\\\?\\');
    const crumbNames = [...document.body.querySelectorAll('button')]
      .map((b) => b.textContent.trim());
    expect(crumbNames).toContain('Users');
    expect(crumbNames).toContain('jalon');
    expect(crumbNames).not.toContain('?');

    // A parent crumb navigates to a REAL path (the old builder produced "?"
    // and bare "C:" crumbs that 403'd).
    fetchCalls.length = 0;
    await click(byText('Users'));
    await flush();
    const nav = fetchCalls.find((c) => c.url.includes('/companion-files/list'));
    expect(nav.url).toContain(encodeURIComponent('C:\\Users'));
    expect(nav.url).not.toContain('%3F');   // no "?" in the path
  });

  it('Up button walks to the parent folder, then back to the Shared view', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();
    await click([...document.body.querySelectorAll('[data-fbpath]')].find((r) => r.textContent.includes('Anime')));
    await flush();

    fetchCalls.length = 0;
    await click(document.body.querySelector('[aria-label="Up one level"]'));
    await flush();
    expect(fetchCalls.some((c) => c.url.includes('/companion-files/list')
      && c.url.includes(encodeURIComponent('D:\\media'))
      && !c.url.includes(encodeURIComponent('Anime')))).toBe(true);

    // From the shared root, Up exits to the Shared home view.
    await click(document.body.querySelector('[aria-label="Up one level"]'));
    await flush();
    expect(document.body.querySelector('input').placeholder).toContain('Paste a full path');
  });

  it('opens a pasted quoted FILE path via its parent, with the file selected', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    const input = document.body.querySelector('input');
    await setInput(input, '"D:\\media\\ep1.mp4"');
    await pressEnter(input);
    await flush();
    await flush();

    // Landed in the parent folder with the file selected, ready to Import.
    expect(document.body.textContent).toContain('Selected “ep1.mp4”');
    // The selection is real: the footer's Clear button only renders when
    // at least one file is selected.
    expect(byText('Clear')).toBeTruthy();
    // The file was probed first (400 → not a directory), then the parent.
    const listCalls = fetchCalls.filter((c) => c.url.includes('/companion-files/list'));
    expect(listCalls[0].url).toContain(encodeURIComponent('D:\\media\\ep1.mp4'));
    expect(listCalls.some((c) => c.url.endsWith(encodeURIComponent('D:\\media')))).toBe(true);
  });
});

describe('CompanionBrowser bulk folder import', () => {
  it('confirms with the video count, starts the run, and shows the shared progress panel', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();

    const pill = [...document.body.querySelectorAll('[title="Import every video in this folder, one at a time"]')].pop();
    expect(pill).toBeTruthy();
    await click(pill);
    await flush();

    expect(document.body.textContent).toContain('Import all');
    expect(document.body.textContent).toContain('2 videos');
    await click(byText('Start'));
    await flush();
    await flush();

    const post = fetchCalls.find((c) => c.method === 'POST' && c.url.endsWith('/companion-files/import-folder'));
    expect(post.body).toMatchObject({ host_id: 'h1', path: 'D:\\media\\Anime' });
    expect(document.body.textContent).toContain('Folder import done — 2 of 2 videos imported');
  });

  it('renders per-video statuses and an out-of-space stop honestly', async () => {
    bulkProgress = {
      status: 'out_of_space', total: 3, done: 1, ok: 1, failed: 0, current: -1,
      folder_name: 'media',
      items: [
        { name: 'a.mp4', path: 'a', size: 1, status: 'complete', job_id: 'j1', error: '', done_bytes: 1 },
        { name: 'b.mkv', path: 'b', size: 1, status: 'no_space', job_id: '', error: 'not enough free disk space on the ClipAI device', done_bytes: 0 },
        { name: 'c.mov', path: 'c', size: 1, status: 'skipped', job_id: '', error: '', done_bytes: 0 },
      ],
    };
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();
    const pill = [...document.body.querySelectorAll('[title="Import every video in this folder, one at a time"]')].pop();
    await click(pill);
    await flush();
    await click(byText('Start'));
    await flush();
    await flush();

    expect(document.body.textContent).toContain('Stopped — ClipAI ran out of disk space (1 of 3 done)');
    expect(document.body.textContent).toContain('Out of disk space');
    expect(document.body.textContent).toContain('Skipped');
    expect(byText('Dismiss')).toBeTruthy();
    expect(byText('Cancel import')).toBeFalsy();
  });

  it('does not offer bulk import for media/font browsing', async () => {
    await render(<CompanionBrowser kind="media" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();
    expect(document.body.querySelector('[title="Import every video in this folder, one at a time"]')).toBeFalsy();
    expect(document.body.querySelector('[title="Bookmark this folder"]')).toBeTruthy();
  });
});
