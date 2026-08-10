/**
 * Filmstrip thumbnail source for the timeline.
 *
 * Two backends, transparent to callers:
 *
 *  1. **Server sprite sheet** (preferred, like Premiere Web / YouTube
 *     storyboards). When a source has a precomputed sprite — one JPEG of N
 *     evenly-spaced tiles plus a manifest — we fetch it once and *slice* the
 *     tile for each timestamp. No per-thumb video seek, no main-thread decode
 *     stall, and it survives reloads via the HTTP cache. This is what makes a
 *     2-hour video's filmstrip appear near-instantly.
 *
 *  2. **Client-side seeking** (fallback). If no sprite is available (old jobs,
 *     non-job media, or the sprite hasn't finished generating), we fall back
 *     to the original behavior: a hidden ``<video>`` seeks to each timestamp
 *     and we capture it to an ``OffscreenCanvas``.
 *
 * Both paths funnel into the same in-memory ``ImageBitmap`` cache keyed on
 * ``(src, t, w, h)`` so the canvas Timeline redraws are O(1). The public API
 * — ``getCachedThumbnail`` / ``ensureThumbnail`` — is unchanged; the Timeline
 * doesn't know or care which backend produced a tile.
 */

const _CACHE = new Map();         // key -> ImageBitmap
const _PENDING = new Map();       // key -> Promise<ImageBitmap>
const _ELEMENTS = new Map();      // src -> shared <video> element

// Server-sprite state, keyed by source URL.
const _SPRITE_JOB = new Map();    // src -> jobId (explicit registration)
const _SPRITE = new Map();        // src -> { manifest, img } once ready
const _SPRITE_PROMISE = new Map();// src -> in-flight Promise (dedupe; cleared on settle)
const _SPRITE_MISS = new Map();   // src -> last-miss timestamp (ms) for retry backoff
const _SPRITE_FIRST_MISS = new Map(); // src -> first-miss timestamp (ms)
const _SPRITE_UPGRADE_TIMER = new Map(); // src -> timer id (coarse→fine polling)
// Long videos generate their sprite in the background AFTER the editor opens,
// so a first miss must NOT be permanent — retry periodically until it appears.
const _SPRITE_RETRY_MS = 6000;
// While a COARSE sheet is being served, poll for the fine replacement.
const _SPRITE_UPGRADE_POLL_MS = 8000;
// Hidden-<video> fallback policy for JOB sources: seeking a hidden <video>
// per thumbnail on a 2-hour file is a storm of Range requests that competes
// with the preview player — the sprite (coarse in seconds) is the right
// source. Only allow the legacy fallback when the sprite has been missing
// for a while AND the source is short enough for seeks to be cheap.
const _CLIENT_FALLBACK_AFTER_MS = 90_000;
const _CLIENT_FALLBACK_MAX_DURATION_S = 900;
// Fired on window whenever a source's sprite appears or upgrades, so canvas
// timelines can redraw: new CustomEvent('clipai:filmstrip-updated', {detail:{src}})
export const FILMSTRIP_UPDATED_EVENT = 'clipai:filmstrip-updated';

const _MAX_CACHE = 600;           // keep the cache bounded

// Black-tile self-heal: how many consecutive near-black reads of the same
// tile we re-try (without caching) before accepting the frame really is dark.
const _BLACK_RETRY = new Map();   // key -> consecutive near-black count
const _BLACK_RETRY_MAX = 3;

// Reusable probe canvas — sampling a bitmap down to a few pixels is enough
// to tell "capture failed / undecoded region" (uniform black) from a real
// frame; done at 8x8 it costs microseconds per tile.
let _probeCtx = null;
function _isNearBlack(bitmap, w, h) {
  try {
    if (!_probeCtx) {
      const c = typeof OffscreenCanvas !== 'undefined'
        ? new OffscreenCanvas(8, 8)
        : Object.assign(document.createElement('canvas'), { width: 8, height: 8 });
      _probeCtx = c.getContext('2d', { willReadFrequently: true });
    }
    if (!_probeCtx) return false;
    _probeCtx.drawImage(bitmap, 0, 0, w || bitmap.width, h || bitmap.height, 0, 0, 8, 8);
    const d = _probeCtx.getImageData(0, 0, 8, 8).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) {
      sum += (d[i] + d[i + 1] + d[i + 2]) / 3;
    }
    return (sum / 64) < 10;        // mean luma < 10/255 ≈ pure black
  } catch {
    return false;                  // probe failure must never block caching
  }
}

function _key(src, t, w, h) {
  return `${src}|${t.toFixed(2)}|${w}x${h}`;
}

function _getVideoEl(src) {
  let el = _ELEMENTS.get(src);
  if (el) return el;
  el = document.createElement('video');
  el.crossOrigin = 'anonymous';
  el.preload = 'metadata';
  el.muted = true;
  el.playsInline = true;
  el.src = src;
  el.style.position = 'fixed';
  el.style.left = '-99999px';
  el.style.width = '1px';
  el.style.height = '1px';
  el.style.pointerEvents = 'none';
  document.body.appendChild(el);
  _ELEMENTS.set(src, el);
  return el;
}

// Cached feature support so we don't pay the typeof check per thumb.
const _SUPPORTS_OFFSCREEN = typeof OffscreenCanvas === 'function';
const _SUPPORTS_BITMAP = typeof globalThis.createImageBitmap === 'function';

// ── Server sprite ──────────────────────────────────────────────────────────

/**
 * Tell the filmstrip that ``src`` belongs to ``jobId`` so it can fetch that
 * job's precomputed sprite/manifest. Call this from the editor when both are
 * known; without it we still try to derive the id from the URL.
 */
export function registerFilmstripSource(src, jobId) {
  if (src && jobId) _SPRITE_JOB.set(src, String(jobId));
}

function _deriveJobId(src) {
  const explicit = _SPRITE_JOB.get(src);
  if (explicit) return explicit;
  // Known editor source shapes: /api/files/{jobId}/video.mp4,
  // /api/jobs/{jobId}/..., /files/{jobId}/video.*
  const m =
    /\/(?:api\/)?files\/([^/?#]+)\/video\./.exec(src) ||
    /\/(?:api\/)?jobs\/([^/?#]+)\//.exec(src);
  return m ? m[1] : null;
}

function _fetchManifest(jobId) {
  const base = `/api/jobs/${encodeURIComponent(jobId)}`;
  return fetch(`${base}/filmstrip.json`, { cache: 'no-cache' })
    .then((res) => (res.ok ? res.json() : null))
    .catch(() => null);
}

function _loadSheetImage(jobId, manifest) {
  const base = `/api/jobs/${encodeURIComponent(jobId)}`;
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    // ``v`` cache-busts the long-max-age sheet when a coarse sprite is
    // upgraded to the fine one (or a re-analysis rebuilds it).
    img.src = `${base}/filmstrip.jpg?v=${encodeURIComponent(manifest.v || 0)}`;
  });
}

function _announceUpdate(src) {
  try {
    window.dispatchEvent(new CustomEvent(FILMSTRIP_UPDATED_EVENT, { detail: { src } }));
  } catch { /* non-browser test env */ }
}

/**
 * While a coarse sheet is live, poll the manifest and hot-swap in the fine
 * sheet the moment the server finishes it. Cache entries for the src are
 * dropped so the timeline re-slices from the sharper tiles.
 */
function _scheduleUpgradePoll(src, jobId) {
  if (_SPRITE_UPGRADE_TIMER.has(src)) return;
  const timer = setTimeout(() => {
    _SPRITE_UPGRADE_TIMER.delete(src);
    const current = _SPRITE.get(src);
    if (!current || !current.manifest.coarse) return;
    _fetchManifest(jobId).then((manifest) => {
      const cur = _SPRITE.get(src);
      if (!cur) return;                       // disposed while polling
      if (!manifest || !manifest.cols || manifest.v === cur.manifest.v) {
        _scheduleUpgradePoll(src, jobId);     // not upgraded yet — keep polling
        return;
      }
      _loadSheetImage(jobId, manifest).then((img) => {
        if (!_SPRITE.has(src)) return;
        if (img) {
          _SPRITE.set(src, { manifest, img });
          _dropCachedTiles(src);
          _announceUpdate(src);
        }
        if (manifest.coarse) _scheduleUpgradePoll(src, jobId);
      });
    });
  }, _SPRITE_UPGRADE_POLL_MS);
  _SPRITE_UPGRADE_TIMER.set(src, timer);
}

function _dropCachedTiles(src) {
  for (const k of [..._CACHE.keys()]) {
    if (k.startsWith(`${src}|`)) {
      const bitmap = _CACHE.get(k);
      try { bitmap.close && bitmap.close(); } catch { /* noop */ }
      _CACHE.delete(k);
    }
  }
}

/**
 * Fetch (once) the sprite manifest + image for ``src``. Resolves to
 * ``{ manifest, img }`` when a usable sprite exists, or ``null`` to signal
 * "not (yet) available". Never rejects.
 */
function _loadSprite(src) {
  const ready = _SPRITE.get(src);
  if (ready) return Promise.resolve(ready);
  const pending = _SPRITE_PROMISE.get(src);
  if (pending) return pending;

  const jobId = _deriveJobId(src);
  if (!jobId) return Promise.resolve(null);

  // Back off between misses, but keep retrying — the sprite may still be
  // generating on the server (long videos) and should be picked up live.
  const lastMiss = _SPRITE_MISS.get(src);
  if (lastMiss && (Date.now() - lastMiss) < _SPRITE_RETRY_MS) return Promise.resolve(null);

  const p = _fetchManifest(jobId)
    .then((manifest) => {
      if (!manifest || !manifest.cols || !manifest.tileW || !manifest.interval) {
        return null;
      }
      return _loadSheetImage(jobId, manifest).then((img) => {
        if (!img) return null;
        const entry = { manifest, img };
        _SPRITE.set(src, entry);
        if (manifest.coarse) _scheduleUpgradePoll(src, jobId);
        _announceUpdate(src);
        return entry;
      });
    })
    .catch(() => null)
    .then((entry) => {
      _SPRITE_PROMISE.delete(src);           // allow a future retry
      if (!entry) {
        _SPRITE_MISS.set(src, Date.now());
        if (!_SPRITE_FIRST_MISS.has(src)) _SPRITE_FIRST_MISS.set(src, Date.now());
      } else {
        _SPRITE_MISS.delete(src);
        _SPRITE_FIRST_MISS.delete(src);
      }
      return entry;
    });
  _SPRITE_PROMISE.set(src, p);
  return p;
}

/**
 * Source-rect for ``cover``-fitting a tile into a w×h box: the largest
 * centered crop of the tile whose aspect matches the target. Pure —
 * exported for tests.
 */
export function computeCoverCrop(tileW, tileH, w, h) {
  let sw = tileW;
  let sh = tileH;
  if (tileW * h > w * tileH) {
    // tile is wider than target aspect — crop the sides
    sw = Math.max(1, Math.round((w / h) * tileH));
  } else {
    // tile is taller — crop top/bottom
    sh = Math.max(1, Math.round((h / w) * tileW));
  }
  return {
    sx: Math.floor((tileW - sw) / 2),
    sy: Math.floor((tileH - sh) / 2),
    sw,
    sh,
  };
}

// createImageBitmap(img, sx, sy, sw, sh, {resize*}) crops + scales on the
// compositor with NO JPEG round-trip. The old path drew to an OffscreenCanvas,
// re-ENCODED it to a JPEG blob, then DECODED that back into a bitmap — two
// image codecs per tile on the main thread, ~30 tiles per screenful.
let _bitmapCropBroken = false;

function _sliceSpriteCanvas(entry, t, w, h) {
  const { manifest, img } = entry;
  const idx = Math.max(
    0,
    Math.min(manifest.count - 1, Math.floor(t / manifest.interval)),
  );
  const col = idx % manifest.cols;
  const row = Math.floor(idx / manifest.cols);
  const sx = col * manifest.tileW;
  const sy = row * manifest.tileH;
  const off = new OffscreenCanvas(w, h);
  const ctx = off.getContext('2d');
  if (!ctx) return Promise.reject(new Error('OffscreenCanvas 2D context unavailable'));
  const scale = Math.max(w / manifest.tileW, h / manifest.tileH);
  const drawW = manifest.tileW * scale;
  const drawH = manifest.tileH * scale;
  const dx = (w - drawW) / 2;
  const dy = (h - drawH) / 2;
  ctx.drawImage(img, sx, sy, manifest.tileW, manifest.tileH, dx, dy, drawW, drawH);
  return createImageBitmap(off);
}

function _sliceSprite(entry, t, w, h) {
  const { manifest, img } = entry;
  if (_bitmapCropBroken) return _sliceSpriteCanvas(entry, t, w, h);
  const idx = Math.max(
    0,
    Math.min(manifest.count - 1, Math.floor(t / manifest.interval)),
  );
  const col = idx % manifest.cols;
  const row = Math.floor(idx / manifest.cols);
  const crop = computeCoverCrop(manifest.tileW, manifest.tileH, w, h);
  return createImageBitmap(
    img,
    col * manifest.tileW + crop.sx,
    row * manifest.tileH + crop.sy,
    crop.sw,
    crop.sh,
    { resizeWidth: w, resizeHeight: h, resizeQuality: 'medium' },
  ).catch((err) => {
    // Older Safari lacks crop/resize options — remember and use the canvas
    // path from now on instead of failing every tile.
    _bitmapCropBroken = true;
    console.warn('filmstrip: createImageBitmap crop unsupported, using canvas path', err);
    return _sliceSpriteCanvas(entry, t, w, h);
  });
}

// ── Client-side seeking (fallback) ──────────────────────────────────────────

function _seekAndCapture(el, t, w, h) {
  // Browsers without ``OffscreenCanvas`` (Safari < 16.4, older mobile)
  // would throw inside the seek handler and surface as silent thumbnail
  // gaps. Bail out cleanly so callers fall back to the colored bar.
  if (!_SUPPORTS_OFFSCREEN || !_SUPPORTS_BITMAP) {
    return Promise.reject(new Error('OffscreenCanvas/createImageBitmap unsupported'));
  }
  return new Promise((resolve, reject) => {
    let done = false;
    const onSeeked = () => {
      if (done) return;
      done = true;
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      try {
        const off = new OffscreenCanvas(w, h);
        const ctx = off.getContext('2d');
        if (!ctx) {
          reject(new Error('OffscreenCanvas 2D context unavailable'));
          return;
        }
        // ``cover``-style fit so the thumb shows what the user expects.
        const vw = el.videoWidth || w;
        const vh = el.videoHeight || h;
        const scale = Math.max(w / vw, h / vh);
        const drawW = vw * scale;
        const drawH = vh * scale;
        const dx = (w - drawW) / 2;
        const dy = (h - drawH) / 2;
        ctx.drawImage(el, dx, dy, drawW, drawH);
        off.convertToBlob({ type: 'image/jpeg', quality: 0.6 })
          .then(createImageBitmap)
          .then(resolve)
          .catch(reject);
      } catch (e) {
        reject(e);
      }
    };
    const onError = (e) => {
      if (done) return;
      done = true;
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('error', onError);
      reject(e?.error || e || new Error('thumbnail seek failed'));
    };
    el.addEventListener('seeked', onSeeked, { once: true });
    el.addEventListener('error', onError, { once: true });
    try {
      el.currentTime = Math.max(0.01, t);
    } catch (e) {
      onError(e);
    }
  });
}

function _clientCapture(src, t, w, h) {
  const el = _getVideoEl(src);
  const ready = el.readyState >= 1
    ? Promise.resolve()
    : new Promise((res) => el.addEventListener('loadedmetadata', res, { once: true }));
  return ready.then(() => _seekAndCapture(el, t, w, h));
}

// ── Public API ───────────────────────────────────────────────────────────

/** True iff this browser can actually generate filmstrip thumbnails. */
export function isFilmstripSupported() {
  return _SUPPORTS_OFFSCREEN && _SUPPORTS_BITMAP;
}

/**
 * True when the server sprite for ``src`` is loaded and sliceable RIGHT NOW.
 * Lets the timeline batch-schedule every missing visible tile in one draw
 * pass (slicing a loaded sheet is cheap and safe to parallelize) instead of
 * the one-tile-per-redraw trickle required by the hidden-<video> fallback.
 */
export function hasSpriteReady(src) {
  return _SPRITE.has(src);
}

/**
 * Get a cached thumbnail synchronously, or ``null`` if it has to be
 * generated. The caller is expected to schedule generation via
 * ``ensureThumbnail`` and trigger a redraw on the returned promise.
 */
export function getCachedThumbnail(src, t, w, h) {
  if (!isFilmstripSupported()) return null;
  return _CACHE.get(_key(src, t, w, h)) || null;
}

/**
 * Decide what to do when no sprite is available (yet) for ``src``.
 *
 * Non-job media (no jobId derivable) has no server sprite at all — hidden
 * <video> capture is the only option, as before. JOB sources DO get a
 * sprite (a coarse one within seconds on long videos), so falling back to
 * per-thumbnail seeks would just hammer the container with Range requests
 * that fight the preview player — the exact "filmstrip never loads on long
 * videos" failure. We wait for the sprite instead, and only allow the
 * legacy capture if the sprite has been missing for a long time on a SHORT
 * source (where seeks are cheap).
 */
function _fallbackAllowed(src, durationHint) {
  if (!_deriveJobId(src)) return true;       // non-job media: only option
  const firstMiss = _SPRITE_FIRST_MISS.get(src);
  if (!firstMiss || (Date.now() - firstMiss) < _CLIENT_FALLBACK_AFTER_MS) return false;
  const dur = Number(durationHint) || 0;
  return dur > 0 && dur <= _CLIENT_FALLBACK_MAX_DURATION_S;
}

/**
 * Lazily generate a thumbnail. Prefers the server sprite (slice a tile);
 * falls back to seeking a hidden ``<video>`` where allowed (see
 * ``_fallbackAllowed``). ``opts.durationHint`` — the source duration in
 * seconds, when the caller knows it — gates that fallback. Returns the same
 * Promise for repeat calls during generation so we don't do the same work
 * twice.
 */
export function ensureThumbnail(src, t, w, h, opts) {
  if (!src) return Promise.reject(new Error('no src'));
  if (!isFilmstripSupported()) {
    return Promise.reject(new Error('filmstrip thumbnails unsupported on this browser'));
  }
  const key = _key(src, t, w, h);
  const hit = _CACHE.get(key);
  if (hit) return Promise.resolve(hit);
  const inflight = _PENDING.get(key);
  if (inflight) return inflight;

  const p = _loadSprite(src)
    .then((entry) => {
      if (entry) return _sliceSprite(entry, t, w, h);
      if (_fallbackAllowed(src, opts && opts.durationHint)) {
        return _clientCapture(src, t, w, h);
      }
      // Sprite still generating server-side — the caller keeps its shimmer
      // and retries on later draws (the 404 retry/backoff lives in
      // _loadSprite). Soft, expected rejection.
      throw new Error('sprite pending');
    })
    .then((bitmap) => {
      _PENDING.delete(key);
      // Never LATCH a black tile. Captures taken while the editor is open
      // during analysis can be black through no fault of the frame — the
      // browser-preview transcode is still being built, a seek landed on an
      // unbuffered region, or the sprite pass raced the face loop for the
      // decoder. Caching those permanently painted long black runs on the
      // timeline that survived the real thumbnails becoming available.
      // Return the bitmap for THIS draw, but skip the cache so the next
      // redraw re-captures; after a few consistent black reads accept it —
      // genuinely dark scenes (fades, night shots) do exist.
      if (_isNearBlack(bitmap, w, h)) {
        const misses = (_BLACK_RETRY.get(key) || 0) + 1;
        if (misses <= _BLACK_RETRY_MAX) {
          _BLACK_RETRY.set(key, misses);
          return bitmap;               // shown now, re-tried on a later draw
        }
      }
      _BLACK_RETRY.delete(key);
      _CACHE.set(key, bitmap);
      if (_CACHE.size > _MAX_CACHE) {
        // Evict the oldest entry — Map preserves insertion order.
        const firstKey = _CACHE.keys().next().value;
        const oldBitmap = _CACHE.get(firstKey);
        _CACHE.delete(firstKey);
        try { oldBitmap.close && oldBitmap.close(); } catch {}
      }
      return bitmap;
    })
    .catch((err) => {
      _PENDING.delete(key);
      throw err;
    });
  _PENDING.set(key, p);
  return p;
}

/**
 * Drop every cached thumbnail for a source (e.g. when the source URL
 * changes or the editor unmounts).
 */
export function disposeFilmstrip(src) {
  _dropCachedTiles(src);
  const el = _ELEMENTS.get(src);
  if (el) {
    try { el.remove(); } catch {}
    _ELEMENTS.delete(src);
  }
  const timer = _SPRITE_UPGRADE_TIMER.get(src);
  if (timer) {
    clearTimeout(timer);
    _SPRITE_UPGRADE_TIMER.delete(src);
  }
  _SPRITE.delete(src);
  _SPRITE_PROMISE.delete(src);
  _SPRITE_MISS.delete(src);
  _SPRITE_FIRST_MISS.delete(src);
}

/** How many timestamps to thumbnail across a clip given pixel width. */
export function computeThumbStops(clipStart, clipEnd, pps, thumbW) {
  const widthPx = (clipEnd - clipStart) * pps;
  const count = Math.max(2, Math.min(120, Math.floor(widthPx / Math.max(thumbW, 24))));
  const stops = [];
  for (let i = 0; i < count; i++) {
    const t = clipStart + ((i + 0.5) / count) * (clipEnd - clipStart);
    stops.push(t);
  }
  return stops;
}
