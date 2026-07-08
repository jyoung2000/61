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
const _SPRITE_PROMISE = new Map();// src -> Promise<{manifest,img}|null> (dedupe)

const _MAX_CACHE = 600;           // keep the cache bounded

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

/**
 * Fetch (once) the sprite manifest + image for ``src``. Resolves to
 * ``{ manifest, img }`` when a usable sprite exists, or ``null`` to signal
 * "fall back to client-side seeking". Never rejects.
 */
function _loadSprite(src) {
  const ready = _SPRITE.get(src);
  if (ready) return Promise.resolve(ready);
  const pending = _SPRITE_PROMISE.get(src);
  if (pending) return pending;

  const jobId = _deriveJobId(src);
  if (!jobId) {
    const p = Promise.resolve(null);
    _SPRITE_PROMISE.set(src, p);
    return p;
  }

  const base = `/api/jobs/${encodeURIComponent(jobId)}`;
  const p = fetch(`${base}/filmstrip.json`)
    .then((res) => (res.ok ? res.json() : null))
    .then((manifest) => {
      if (!manifest || !manifest.cols || !manifest.tileW || !manifest.interval) {
        return null;
      }
      return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.decoding = 'async';
        img.onload = () => {
          const entry = { manifest, img };
          _SPRITE.set(src, entry);
          resolve(entry);
        };
        img.onerror = () => resolve(null);
        img.src = `${base}/filmstrip.jpg`;
      });
    })
    .catch(() => null);
  _SPRITE_PROMISE.set(src, p);
  return p;
}

function _sliceSprite(entry, t, w, h) {
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
  // ``cover``-fit the tile into the target box, matching the client path.
  const scale = Math.max(w / manifest.tileW, h / manifest.tileH);
  const drawW = manifest.tileW * scale;
  const drawH = manifest.tileH * scale;
  const dx = (w - drawW) / 2;
  const dy = (h - drawH) / 2;
  ctx.drawImage(img, sx, sy, manifest.tileW, manifest.tileH, dx, dy, drawW, drawH);
  return off.convertToBlob({ type: 'image/jpeg', quality: 0.7 }).then(createImageBitmap);
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
 * Get a cached thumbnail synchronously, or ``null`` if it has to be
 * generated. The caller is expected to schedule generation via
 * ``ensureThumbnail`` and trigger a redraw on the returned promise.
 */
export function getCachedThumbnail(src, t, w, h) {
  if (!isFilmstripSupported()) return null;
  return _CACHE.get(_key(src, t, w, h)) || null;
}

/**
 * Lazily generate a thumbnail. Prefers the server sprite (slice a tile);
 * falls back to seeking a hidden ``<video>``. Returns the same Promise for
 * repeat calls during generation so we don't do the same work twice.
 */
export function ensureThumbnail(src, t, w, h) {
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
    .then((entry) => (entry ? _sliceSprite(entry, t, w, h) : _clientCapture(src, t, w, h)))
    .then((bitmap) => {
      _PENDING.delete(key);
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
  for (const k of [..._CACHE.keys()]) {
    if (k.startsWith(`${src}|`)) {
      const bitmap = _CACHE.get(k);
      try { bitmap.close && bitmap.close(); } catch {}
      _CACHE.delete(k);
    }
  }
  const el = _ELEMENTS.get(src);
  if (el) {
    try { el.remove(); } catch {}
    _ELEMENTS.delete(src);
  }
  _SPRITE.delete(src);
  _SPRITE_PROMISE.delete(src);
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
