import { useEffect, useRef, useCallback, useState } from 'react';
import { openDB } from 'idb';
import useTimelineStore from '../stores/timelineStore';

const DB_NAME = 'clipai-editor';
const DB_VERSION = 1;
const STORE_NAME = 'projects';
const AUTOSAVE_DEBOUNCE_MS = 500;
const SERVER_SYNC_INTERVAL_MS = 60000;

async function getDB() {
  return openDB(DB_NAME, DB_VERSION, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'key' });
      }
    },
  });
}

/**
 * Validate a persisted timeline state and report whether it is CORRUPT.
 *
 * Corruption manifested in prior runs as the reverse-sync writing a stale
 * timeline back over a fresh transcript: subtitle items out of chronological
 * order, duplicated cues (same start+text many times), and backwards cues
 * (end <= start). Such a cache must never be re-imported — it would re-poison
 * the transcript. We treat any of these as corrupt and purge the entry.
 *
 * Returns { corrupt: boolean, reason: string }.
 */
function validatePersistedState(state) {
  if (!state || !Array.isArray(state.items)) {
    return { corrupt: false, reason: '' };
  }
  const subs = state.items
    .filter((it) => it && it.type === 'subtitle' && (it.subtitleText || '').trim());
  if (subs.length === 0) return { corrupt: false, reason: '' };

  // 1) Backwards / non-finite cues.
  for (const s of subs) {
    const st = Number(s.start);
    const en = Number(s.end);
    if (!Number.isFinite(st) || !Number.isFinite(en) || en < st - 0.001) {
      return { corrupt: true, reason: `backwards/invalid cue (start=${s.start} end=${s.end})` };
    }
  }

  // 2) Out-of-chronological-order cues (the timeline is authored sorted; a
  //    later cue starting well before an earlier one signals a stale/merged
  //    cache, not a normal overlap).
  for (let i = 1; i < subs.length; i++) {
    if (Number(subs[i].start) < Number(subs[i - 1].start) - 0.05) {
      return { corrupt: true, reason: `out-of-order cue at index ${i}` };
    }
  }

  // 3) Excessive duplicate (start, text) pairs — the signature of the
  //    duplicated-OP-lyric corruption (same block repeated many times).
  const seen = new Map();
  let dupes = 0;
  for (const s of subs) {
    const k = `${Math.round(Number(s.start) * 10)}|${(s.subtitleText || '').trim()}`;
    const n = (seen.get(k) || 0) + 1;
    seen.set(k, n);
    if (n > 1) dupes += 1;
  }
  // A handful of repeated short lines is normal; a large share is not.
  if (dupes >= 5 && dupes > subs.length * 0.1) {
    return { corrupt: true, reason: `${dupes}/${subs.length} duplicate cues` };
  }

  return { corrupt: false, reason: '' };
}

/**
 * Drop a single corrupt project entry from IndexedDB. Best-effort.
 */
async function purgeCorruptEntry(db, key, reason) {
  try {
    await db.delete(STORE_NAME, key);
    // eslint-disable-next-line no-console
    console.warn(`[clipai] Purged corrupt editor cache for ${key}: ${reason}`);
  } catch {
    // ignore — nothing more we can do
  }
}

/**
 * Reconcile media library entries with the backend.
 *
 * After IndexedDB recovery or page reload, some media entries may have stale
 * blob: URLs (which are invalidated when the page unloads). This function
 * fetches the authoritative media list from the backend and:
 * 1. Replaces any stale blob: URLs with the backend URL
 * 2. Adds any backend media that is missing from the local store
 * 3. Leaves entries with valid backend URLs untouched
 *
 * This ensures uploaded media persists across container restarts and browser
 * cache clears — only explicit user deletion removes media.
 */
async function reconcileMediaLibrary(jobId) {
  const updateMedia = useTimelineStore.getState().updateMedia;
  const addMedia = useTimelineStore.getState().addMedia;

  // Fetch both job-specific and global library media
  const urls = ['/api/media/list'];
  if (jobId && jobId !== '_library') {
    urls.push(`/api/media/list?job_id=${jobId}`);
  }

  for (const url of urls) {
    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      const data = await res.json();
      const backendItems = data.items || [];

      const currentLibrary = useTimelineStore.getState().mediaLibrary;
      const currentById = new Map(currentLibrary.map(m => [m.id, m]));

      for (const backendItem of backendItems) {
        const existing = currentById.get(backendItem.id);
        if (existing) {
          // Entry exists locally — update URL if it's a stale blob: URL
          // or if the backend URL has changed
          if (
            existing.url.startsWith('blob:') ||
            (!existing.url.startsWith('/api/') && backendItem.url)
          ) {
            updateMedia(existing.id, { url: backendItem.url });
          }
        } else {
          // Entry missing locally — add from backend
          addMedia({
            id: backendItem.id,
            type: backendItem.type,
            filename: backendItem.filename,
            url: backendItem.url,
            thumbnailUrl: backendItem.type === 'image' ? backendItem.url : '',
            duration: 0,
          });
        }
      }
    } catch {
      // Backend unavailable — skip reconciliation
    }
  }
}

export default function useTimelinePersistence(jobId, clipId) {
  const exportState = useTimelineStore((s) => s.exportState);
  const importState = useTimelineStore((s) => s.importState);
  const tracks = useTimelineStore((s) => s.tracks);
  const items = useTimelineStore((s) => s.items);
  const mediaLibrary = useTimelineStore((s) => s.mediaLibrary);
  const [recovered, setRecovered] = useState(false);

  const saveTimerRef = useRef(null);
  const serverTimerRef = useRef(null);
  const reconcileRef = useRef(false);
  const sweptCorruptCacheRef = useRef(false);
  const key = `${jobId || 'unknown'}_${clipId || 'default'}`;

  // ── Load from IndexedDB on mount ──────────────────────────────────────────
  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    (async () => {
      try {
        const db = await getDB();
        // One-time global sweep: purge EVERY corrupt project entry on first
        // mount, not just this job's. This is the "clear corrupted cache at
        // start" guarantee — a stale/corrupt timeline from any prior run can
        // never be restored and re-poison its transcript via reverse-sync.
        if (!sweptCorruptCacheRef.current) {
          sweptCorruptCacheRef.current = true;
          try {
            const all = await db.getAll(STORE_NAME);
            for (const entry of all || []) {
              const { corrupt, reason } = validatePersistedState(entry && entry.state);
              if (corrupt) await purgeCorruptEntry(db, entry.key, reason);
            }
          } catch {
            // getAll unavailable — fall back to per-key validation below
          }
        }
        const saved = await db.get(STORE_NAME, key);
        if (saved && saved.state && !cancelled) {
          const state = saved.state;
          // Validate before importing — a corrupt entry is purged and skipped
          // so the fresh backend transcript shows through untouched.
          const { corrupt, reason } = validatePersistedState(state);
          if (corrupt) {
            await purgeCorruptEntry(db, key, reason);
          } else if (Array.isArray(state.items) && state.items.length > 0) {
            importState(state);
            setRecovered(true);
          }
        }
      } catch {
        // IndexedDB unavailable — no recovery
      }

      // Reconcile media URLs with backend after recovery (or on fresh load)
      if (!cancelled && !reconcileRef.current) {
        reconcileRef.current = true;
        await reconcileMediaLibrary(jobId);
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, clipId, key]);

  // ── Auto-save to IndexedDB (debounced) ────────────────────────────────────
  // Triggers on tracks, items, OR mediaLibrary changes so new uploads are
  // persisted immediately (not just on the 60s server sync interval).
  useEffect(() => {
    if (!jobId) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      try {
        const state = exportState();
        // Never persist a corrupt timeline — doing so would re-seed the cache
        // we purge on load and could feed the reverse-sync. Skip the write
        // (the on-disk transcript remains the source of truth).
        const { corrupt, reason } = validatePersistedState(state);
        if (corrupt) {
          // eslint-disable-next-line no-console
          console.warn(`[clipai] Skipped autosave of corrupt timeline for ${key}: ${reason}`);
          return;
        }
        const db = await getDB();
        await db.put(STORE_NAME, {
          key,
          state,
          lastModified: new Date().toISOString(),
        });
      } catch {
        // Silently fail
      }
    }, AUTOSAVE_DEBOUNCE_MS);

    return () => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    };
  }, [tracks, items, mediaLibrary, jobId, clipId, key, exportState]);

  // ── Server sync every 60s ─────────────────────────────────────────────────
  const syncToServer = useCallback(async () => {
    if (!jobId || clipId == null) return;
    try {
      const state = exportState();
      const { corrupt } = validatePersistedState(state);
      if (corrupt) return;  // never push a corrupt timeline to the server
      await fetch(`/api/jobs/${jobId}/clips/${clipId}/editor-state`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state),
      });
    } catch {
      // Server sync is best-effort
    }
  }, [jobId, clipId, exportState]);

  useEffect(() => {
    if (!jobId) return;
    serverTimerRef.current = setInterval(syncToServer, SERVER_SYNC_INTERVAL_MS);
    return () => {
      if (serverTimerRef.current) clearInterval(serverTimerRef.current);
    };
  }, [syncToServer, jobId]);

  return { recovered };
}
