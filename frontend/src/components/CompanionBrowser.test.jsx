// @vitest-environment jsdom
/**
 * CompanionBrowser: path bookmarks (star folders, server-persisted) and the
 * sequential bulk folder import (confirm step → live progress panel).
 */
import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from 'vitest';
import React from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react';
import CompanionBrowser from './CompanionBrowser';

beforeAll(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; });

let container;
let root;
let fetchCalls;

const ROOTS = {
  companions: [{
    host_id: 'h1', name: '4070', online: true,
    roots: [{ path: 'D:\\media', name: 'media', exists: true }],
  }],
};
const LISTING = {
  path: 'D:\\media',
  entries: [
    { name: 'Anime', path: 'D:\\media\\Anime', is_dir: true, size: 0, ext: '', mtime_ms: 0, created_ms: 0 },
    { name: 'ep1.mp4', path: 'D:\\media\\ep1.mp4', is_dir: false, size: 9000, ext: 'mp4', mtime_ms: 1, created_ms: 1 },
    { name: 'ep2.mkv', path: 'D:\\media\\ep2.mkv', is_dir: false, size: 9000, ext: 'mkv', mtime_ms: 1, created_ms: 1 },
  ],
};

// Mutable per-test server state the fetch mock serves from.
let serverBookmarks;
let bulkProgress;

function mockFetch(url, opts = {}) {
  const method = (opts.method || 'GET').toUpperCase();
  fetchCalls.push({ url, method, body: opts.body ? JSON.parse(opts.body) : null });
  const json = (data, status = 200) =>
    Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(data), text: () => Promise.resolve('') });

  if (url.startsWith('/api/providers/companion-files/roots')) return json(ROOTS);
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
  if (url.startsWith('/api/providers/companion-files/list')) return json(LISTING);
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
  await act(async () => { root = createRoot(container); root.render(el); });
  await flush();
};
const click = (el) => act(async () => { el.click(); await Promise.resolve(); });
const byText = (text, sel = 'button') =>
  [...document.body.querySelectorAll(sel)].find((b) => b.textContent.trim() === text);

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
    // Navigated into the bookmarked folder → listing was requested for it.
    expect(fetchCalls.some((c) => c.url.includes('/companion-files/list')
      && c.url.includes(encodeURIComponent('D:\\media\\Anime')))).toBe(true);
  });

  it('stars a folder row (POST) and un-stars it (DELETE)', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    // Enter the shared root so folder rows render.
    await click(byText('media', 'button') || [...document.body.querySelectorAll('button')]
      .find((b) => b.textContent.includes('media')));
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

describe('CompanionBrowser bulk folder import', () => {
  it('confirms with the video count, starts the run, and shows the progress panel', async () => {
    await render(<CompanionBrowser kind="video" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();

    // Folder rows carry an "Import all" pill for video imports (the toolbar
    // has one too, for the current folder — take the row's, which renders last).
    const pill = [...document.body.querySelectorAll('[title="Import every video in this folder, one at a time"]')].pop();
    expect(pill).toBeTruthy();
    await click(pill);
    await flush();

    // Confirm step names the exact count (2 videos in the mocked listing).
    expect(document.body.textContent).toContain('Import all');
    expect(document.body.textContent).toContain('2 videos');
    const start = byText('Start');
    expect(start).toBeTruthy();
    await click(start);
    await flush();

    const post = fetchCalls.find((c) => c.method === 'POST' && c.url.endsWith('/companion-files/import-folder'));
    expect(post.body).toMatchObject({ host_id: 'h1', path: 'D:\\media\\Anime' });
    // The polled (terminal) state renders as the summary panel.
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

    expect(document.body.textContent).toContain('Stopped — ClipAI ran out of disk space (1 of 3 done)');
    expect(document.body.textContent).toContain('Out of disk space');
    expect(document.body.textContent).toContain('Skipped');
    // Terminal panel offers Dismiss, not Cancel.
    expect(byText('Dismiss')).toBeTruthy();
    expect(byText('Cancel import')).toBeFalsy();
  });

  it('does not offer bulk import for media/font browsing', async () => {
    await render(<CompanionBrowser kind="media" onClose={() => {}} onImported={() => {}} />);
    await click([...document.body.querySelectorAll('button')].find((b) => b.textContent.includes('media')));
    await flush();
    expect(document.body.querySelector('[title="Import every video in this folder, one at a time"]')).toBeFalsy();
    // Bookmarks still work for media browsing.
    expect(document.body.querySelector('[title="Bookmark this folder"]')).toBeTruthy();
  });
});
