import React, { useRef, useEffect, useLayoutEffect, useCallback, useMemo, useState } from 'react';
import useTimelineStore, { getMaxItemDuration, hashGroupId } from '../stores/timelineStore';
import {
  computeThumbStops,
  ensureThumbnail,
  getCachedThumbnail,
  hasSpriteReady,
  FILMSTRIP_UPDATED_EVENT,
} from '../utils/filmstrip';
import ContextMenu from './ContextMenu';
import Tooltip from './Tooltip';
import useResponsive from '../hooks/useResponsive';
import { getCropXForTime } from '../utils/subjectTracking';

// ── Constants ────────────────────────────────────────────────────────────────
// Track sizing modeled after Premiere Pro / DaVinci Resolve / VEED — a clear
// gap between lanes reads as "stacked cards" instead of "one giant grid",
// which makes drag targets and selection states much easier to parse.
const TRACK_HEIGHT = 56;
// Mobile: lanes collapse to a slim bar; tapping a track header expands
// ONE lane at a time back to full height (see laneHeightsFor).
const COMPACT_TRACK_HEIGHT = 28;
// Desktop fit-to-window floor: lanes shrink (never below this) so EVERY
// track fits stacked in the visible canvas without vertical scrolling.
// 28px matches the mobile compact lane, which the whole draw/hit-test
// path already supports.
const MIN_FIT_TRACK_HEIGHT = 28;
const TRACK_GAP = 6;

/** Per-track lane heights. Desktop: uniform, shrunk uniformly when the
 * stack would overflow ``fitHeight`` (fit-to-window — no scrolling until
 * lanes hit the 28px floor). Mobile: compact except the expanded lane.
 * Pure so hit-tests and draw share the exact math. */
function laneHeightsFor(tracks, isMobile, expandedTrackId, fitHeight = 0) {
  if (isMobile) {
    return tracks.map((t) => (
      t.id === expandedTrackId ? TRACK_HEIGHT : COMPACT_TRACK_HEIGHT
    ));
  }
  let per = TRACK_HEIGHT;
  if (fitHeight > 0 && tracks.length > 0) {
    // Mirror canvasHeight's math: ruler + N lanes + N gaps + 12px pad.
    const usable = fitHeight - RULER_HEIGHT - 12;
    per = Math.floor(usable / tracks.length) - TRACK_GAP;
    per = Math.max(MIN_FIT_TRACK_HEIGHT, Math.min(TRACK_HEIGHT, per));
  }
  return tracks.map(() => per);
}

function laneTop(laneHs, idx) {
  let y = RULER_HEIGHT;
  for (let i = 0; i < idx; i++) y += (laneHs[i] ?? TRACK_HEIGHT) + TRACK_GAP;
  return y;
}

function laneIndexFromY(laneHs, y) {
  if (y < RULER_HEIGHT) return -1;
  let top = RULER_HEIGHT;
  for (let i = 0; i < laneHs.length; i++) {
    const bottom = top + laneHs[i] + TRACK_GAP;
    if (y < bottom) return i;
    top = bottom;
  }
  return -1;
}
// Touch devices need fatter invisible grab zones — a finger can't reliably
// land a 14px clip edge or a 16px playhead. Detected once at module load
// (pointer type effectively never changes mid-session).
const COARSE_POINTER = typeof window !== 'undefined' && typeof window.matchMedia === 'function'
  ? window.matchMedia('(pointer: coarse)').matches
  : false;
const LABEL_WIDTH = 140;
const HANDLE_WIDTH = 4;          // slim resting state
const HANDLE_WIDTH_HOVER = 8;    // fattened on hover for an easy grab
const HANDLE_HIT_AREA = COARSE_POINTER ? 24 : 14;
const RULER_HEIGHT = 32;
const PLAYHEAD_GRAB_WIDTH = COARSE_POINTER ? 26 : 16; // px on each side of playhead for grab detection

const TRACK_COLORS = {
  video: '#3B82F6',
  overlay: '#F59E0B',
  audio: '#10B981',
  subtitle: '#8B5CF6',
  text: '#EC4899',
  shape: '#F97316',
  crop: '#06B6D4',
};

const TRACK_ICONS = {
  video: '\uD83C\uDFAC',
  overlay: '\uD83D\uDDBC',
  audio: '\uD83C\uDFB5',
  subtitle: '\uD83D\uDCAC',
  text: 'T',
  shape: '\u25A1',
  crop: '\u2702',
};

// Crop segment cluster colors
const CROP_CLUSTER_COLORS = [
  '#3B82F6', // blue — speaker 0
  '#10B981', // green — speaker 1
  '#F59E0B', // amber — speaker 2
  '#EC4899', // pink — speaker 3
  '#8B5CF6', // purple — manual override / unknown
];

// ── Crop-track smooth-pan gradient ───────────────────────────────────────────
// Each crop element is filled with a horizontal gradient sampled from the
// SmoothDamp subject track, so a human-like pan (the crop X gliding across the
// shot) reads as a smooth colour transition and a held shot stays a flat band.
// The colour is the crop POSITION itself, mapped across a full hue spectrum: a
// left-biased crop is warm (red/orange), a centred crop is green, a
// right-biased crop is cool (blue/violet). That way the user literally sees the
// human-operator's framing sweep — and a glide from 20%→80% reads as a rainbow
// wipe rather than the old single-hue light/dark shimmer ("only blue & green").
const CROP_HUE_SPAN = 280; // 0° red (left) → 280° violet-blue (right); no wrap back to red

// Map a crop X (0–100 %) to a full-spectrum hue. ``baseHex`` is retained in the
// signature for callers but no longer tints the fill — the position drives the
// colour so the whole pan range is legible at a glance. Vivid saturation + mid
// lightness keep every hue readable on the dark timeline.
function cropColorAt(baseHex, cropX, alpha = 1) {
  const x = Math.max(0, Math.min(100, Number.isFinite(cropX) ? cropX : 50)) / 100; // 0..1
  const hue = Math.round(x * CROP_HUE_SPAN);
  return `hsla(${hue}, 82%, 54%, ${alpha})`;
}

function formatTime(s) {
  if (!s || isNaN(s) || s < 0) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function formatTimeMs(s) {
  if (!s || isNaN(s) || s < 0) return '0:00.00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.floor((s % 1) * 100);
  return `${m}:${sec.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

/**
 * Find the nearest snap target for a given time value.
 * Returns { snappedTime, snapTarget } or null if no snap found.
 */
function findSnapTarget(candidateTime, items, excludeItemId, playhead, duration, pps, sceneCuts = null) {
  // Adaptive threshold: larger zone when zoomed out, smaller when zoomed in
  const threshold = Math.max(5, Math.min(12, 600 / pps));

  const targets = new Set();
  targets.add(0);
  targets.add(playhead);
  if (duration > 0) targets.add(duration);

  for (const item of items) {
    if (item.id === excludeItemId) continue;
    targets.add(item.start);
    targets.add(item.end);
  }

  // Snap to scene cuts when the timeline knows about them. This makes
  // trim-handle dragging "click" into the cut detected by the analysis
  // pipeline, matching the expectation that a user would tend to cut
  // **on** a shot boundary rather than 200 ms inside it.
  if (Array.isArray(sceneCuts)) {
    for (const t of sceneCuts) {
      if (Number.isFinite(t)) targets.add(t);
    }
  }

  let bestDist = Infinity;
  let bestTarget = null;

  for (const target of targets) {
    const distPx = Math.abs((candidateTime - target) * pps);
    if (distPx < threshold && distPx < bestDist) {
      bestDist = distPx;
      bestTarget = target;
    }
  }

  if (bestTarget !== null) {
    return { snappedTime: bestTarget, snapTarget: bestTarget };
  }
  return null;
}

// ── Overview minimap ─────────────────────────────────────────────────────────
// Premiere / Resolve / Final Cut all surface a condensed projection of
// the entire timeline as a "navigator" so the user can jump anywhere
// without zooming out. Click anywhere to seek; drag the highlighted
// viewport rectangle to pan; drag its edges to zoom.
function TimelineMinimap({
  tracks, items, cropSegments, duration, playhead,
  scrollX, pps, labelWidth, canvasWidthRef, onScrollTo, onSeek, sceneCuts,
}) {
  const miniRef = useRef(null);
  const containerRef = useRef(null);
  const setZoom = useTimelineStore((s) => s.setZoom);
  const basePPS = 50; // must match the basePPS used in Timeline
  const [hoverPx, setHoverPx] = useState(null);
  const draggingRef = useRef(null);
  const HEIGHT = 38;

  // Total content extent in seconds — same heuristic as the ruler.
  const maxItemEnd = items.length > 0 ? Math.max(...items.map((it) => it.end || 0)) : 0;
  const totalDuration = Math.max(duration || 0, maxItemEnd, 30) * 1.05;

  useEffect(() => {
    const canvas = miniRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return;
    canvas.width = rect.width * dpr;
    canvas.height = HEIGHT * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width;
    const isDark = document.documentElement.dataset?.theme === 'dark';

    // Background
    ctx.clearRect(0, 0, W, HEIGHT);
    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
    ctx.fillRect(0, 0, W, HEIGHT);

    // Convert: seconds → minimap px
    const sPerPx = totalDuration / W;
    const pxPerSec = W / totalDuration;

    // Each track gets a horizontal lane in the minimap.
    const visTracks = tracks.filter((t) => t.visible !== false);
    const laneH = Math.max(2, (HEIGHT - 8) / Math.max(1, visTracks.length));
    visTracks.forEach((track, idx) => {
      const ly = 4 + idx * laneH;
      // Lane background
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)';
      ctx.fillRect(0, ly, W, Math.max(1, laneH - 1));

      // Paint each item as a colored slim bar
      const trackItems = items.filter((it) => it.trackId === track.id);
      for (const it of trackItems) {
        const ix = it.start * pxPerSec;
        const iw = Math.max(1, (it.end - it.start) * pxPerSec);
        const color = TRACK_COLORS[it.type] || TRACK_COLORS.video;
        ctx.fillStyle = color + 'B0';
        ctx.fillRect(ix, ly, iw, Math.max(1, laneH - 1));
      }

      // Crop track gets crop segments instead
      if (track.type === 'crop' && cropSegments?.length) {
        for (const seg of cropSegments) {
          const cx = seg.startTime * pxPerSec;
          const cw = Math.max(1, (seg.endTime - seg.startTime) * pxPerSec);
          const clrIdx = seg.isManualOverride ? 4 : Math.max(0, seg.clusterId);
          const color = CROP_CLUSTER_COLORS[clrIdx % CROP_CLUSTER_COLORS.length];
          ctx.fillStyle = color + 'C0';
          ctx.fillRect(cx, ly, cw, Math.max(1, laneH - 1));
        }
      }
    });

    // Viewport rectangle — what's currently visible in the main canvas.
    const mainCanvas = canvasWidthRef?.current;
    if (mainCanvas) {
      const mainWidth = mainCanvas.getBoundingClientRect().width - labelWidth;
      const viewStartSec = scrollX / pps;
      const viewEndSec = (scrollX + mainWidth) / pps;
      const vx = viewStartSec * pxPerSec;
      const vw = Math.max(8, (viewEndSec - viewStartSec) * pxPerSec);
      // Frosted glass viewport
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.07)';
      ctx.fillRect(vx, 0, vw, HEIGHT);
      ctx.strokeStyle = 'var(--accent, #6E7BFF)';
      ctx.fillStyle = 'rgba(110,123,255,0.18)';
      ctx.fillRect(vx, 0, vw, HEIGHT);
      ctx.strokeStyle = '#6E7BFF';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(vx + 0.5, 0.5, vw - 1, HEIGHT - 1);
      // Edge grippers for zoom-drag
      ctx.fillStyle = '#6E7BFF';
      ctx.fillRect(vx - 1, 8, 2, HEIGHT - 16);
      ctx.fillRect(vx + vw - 1, 8, 2, HEIGHT - 16);
    }

    // Scene-cut markers — small notches on the top edge so the user
    // can navigate cut-to-cut in one click without zooming in.
    if (Array.isArray(sceneCuts) && sceneCuts.length) {
      ctx.fillStyle = isDark ? 'rgba(251, 191, 36, 0.85)' : 'rgba(217, 119, 6, 0.85)';
      for (const c of sceneCuts) {
        const cx = c * pxPerSec;
        if (cx < 0 || cx > W) continue;
        // 3 × 6 notch hanging from the top
        ctx.fillRect(Math.round(cx) - 1, 0, 2, 5);
      }
    }

    // Playhead
    const phx = playhead * pxPerSec;
    if (phx >= 0 && phx <= W) {
      ctx.strokeStyle = '#FF5C5C';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(phx, 0);
      ctx.lineTo(phx, HEIGHT);
      ctx.stroke();
    }

    // Hover tooltip line
    if (hoverPx != null) {
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hoverPx, 0);
      ctx.lineTo(hoverPx, HEIGHT);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [tracks, items, cropSegments, duration, playhead, scrollX, pps,
      hoverPx, totalDuration, canvasWidthRef, labelWidth, sceneCuts]);

  const pxToTime = useCallback((px) => {
    const canvas = miniRef.current;
    if (!canvas) return 0;
    const W = canvas.getBoundingClientRect().width;
    return Math.max(0, (px / W) * totalDuration);
  }, [totalDuration]);

  const onMiniPointerDown = (e) => {
    const canvas = miniRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    let time = pxToTime(px);

    // Click-to-cut snapping: if the click lands within 6 minimap pixels
    // of a scene cut, snap to the cut so cut-to-cut navigation is
    // pixel-accurate even at very compressed scales.
    if (Array.isArray(sceneCuts) && sceneCuts.length) {
      const W = rect.width;
      const pxPerSecSnap = W / totalDuration;
      let best = null;
      let bestDist = Infinity;
      for (const c of sceneCuts) {
        const d = Math.abs(c * pxPerSecSnap - px);
        if (d < bestDist) { bestDist = d; best = c; }
      }
      if (best !== null && bestDist <= 6) time = best;
    }

    const mainCanvas = canvasWidthRef?.current;
    if (!mainCanvas) {
      onSeek(time);
      return;
    }
    const mainWidth = mainCanvas.getBoundingClientRect().width - labelWidth;
    const W = rect.width;
    const pxPerSec = W / totalDuration;
    const viewStartSec = scrollX / pps;
    const viewEndSec = (scrollX + mainWidth) / pps;
    const vx = viewStartSec * pxPerSec;
    const vw = (viewEndSec - viewStartSec) * pxPerSec;
    const edgeGrab = 6;
    let mode = 'seek';
    if (px >= vx - edgeGrab && px <= vx + edgeGrab) mode = 'zoom-left';
    else if (px >= vx + vw - edgeGrab && px <= vx + vw + edgeGrab) mode = 'zoom-right';
    else if (px >= vx && px <= vx + vw) mode = 'pan';
    draggingRef.current = {
      mode,
      startPx: px,
      startScrollX: scrollX,
      startTime: time,
      origVStart: viewStartSec,
      origVEnd: viewEndSec,
    };

    const onMove = (ev) => {
      const drag = draggingRef.current;
      if (!drag) return;
      const cRect = canvas.getBoundingClientRect();
      const curPx = ev.clientX - cRect.left;
      const dx = curPx - drag.startPx;
      const dSec = dx / pxPerSec;
      if (drag.mode === 'pan' || drag.mode === 'seek') {
        // Pan the main view so the same minimap-px point stays under
        // the cursor. For 'seek' mode we also move the playhead.
        const newScrollX = Math.max(0, drag.startScrollX + dSec * pps);
        onScrollTo(newScrollX);
        if (drag.mode === 'seek') {
          onSeek(pxToTime(curPx));
        }
      } else if (drag.mode === 'zoom-left') {
        const newVStart = Math.max(0, drag.origVStart + dSec);
        const newWindow = drag.origVEnd - newVStart;
        if (newWindow > 0.5) {
          const mainW = canvasWidthRef.current.getBoundingClientRect().width - labelWidth;
          const newZoom = Math.max(0.01, mainW / (newWindow * basePPS));
          setZoom(newZoom);
          onScrollTo(newVStart * newZoom * basePPS);
        }
      } else if (drag.mode === 'zoom-right') {
        const newVEnd = Math.max(drag.origVStart + 0.5, drag.origVEnd + dSec);
        const newWindow = newVEnd - drag.origVStart;
        if (newWindow > 0.5) {
          const mainW = canvasWidthRef.current.getBoundingClientRect().width - labelWidth;
          const newZoom = Math.max(0.01, mainW / (newWindow * basePPS));
          setZoom(newZoom);
          onScrollTo(drag.origVStart * newZoom * basePPS);
        }
      }
    };
    const onUp = () => {
      draggingRef.current = null;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);

    // Single click outside the viewport jumps the view to center on
    // the click + seeks the playhead.
    if (mode === 'seek') {
      onSeek(time);
      const mainW = mainWidth;
      const newScrollX = Math.max(0, time * pps - mainW / 2);
      onScrollTo(newScrollX);
    }
  };

  return (
    <div
      ref={containerRef}
      className="ve-multi-timeline__minimap"
      style={{
        height: HEIGHT,
        marginTop: 4,
        marginBottom: 4,
        marginLeft: 8,
        marginRight: 8,
        position: 'relative',
        borderRadius: 6,
        overflow: 'hidden',
        border: '1px solid var(--ve-chrome-border, rgba(127,127,127,0.15))',
        cursor: 'pointer',
      }}
      onPointerMove={(e) => {
        const rect = miniRef.current?.getBoundingClientRect();
        if (rect) setHoverPx(e.clientX - rect.left);
      }}
      onPointerLeave={() => setHoverPx(null)}
    >
      <canvas
        ref={miniRef}
        style={{ width: '100%', height: HEIGHT, display: 'block', touchAction: 'none' }}
        onPointerDown={onMiniPointerDown}
      />
      {hoverPx != null && (
        <div
          style={{
            position: 'absolute',
            left: hoverPx + 6,
            top: 4,
            fontSize: 10,
            color: '#fff',
            background: 'rgba(0,0,0,0.7)',
            padding: '1px 5px',
            borderRadius: 3,
            pointerEvents: 'none',
            fontFamily: 'var(--font-mono, monospace)',
            whiteSpace: 'nowrap',
          }}
        >
          {formatTime(pxToTime(hoverPx))}
        </div>
      )}
    </div>
  );
}

// ── Timecode input ───────────────────────────────────────────────────────────
// Click → type → Enter to seek the playhead to a specific frame.
// Accepts: ``H:MM:SS.mmm``, ``M:SS.mmm``, ``SS.mmm``, raw seconds.
function parseTimecodeInput(raw) {
  if (raw == null) return null;
  const s = String(raw).trim();
  if (!s) return null;
  const parts = s.split(':');
  let total = 0;
  try {
    if (parts.length === 1) {
      total = parseFloat(parts[0]);
    } else if (parts.length === 2) {
      total = parseInt(parts[0], 10) * 60 + parseFloat(parts[1]);
    } else if (parts.length === 3) {
      total = parseInt(parts[0], 10) * 3600 + parseInt(parts[1], 10) * 60 + parseFloat(parts[2]);
    } else {
      return null;
    }
  } catch (_) { return null; }
  if (!Number.isFinite(total) || total < 0) return null;
  return total;
}

function TimecodeInput({ playhead, onSeek }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState('');
  const display = formatTimeMs(playhead || 0);
  if (!editing) {
    return (
      <button
        className="ve-multi-timeline__tc"
        aria-label="Current timecode — click to jump to a specific time"
        onClick={() => {
          setValue(formatTimeMs(playhead || 0));
          setEditing(true);
        }}
        style={{
          fontFamily: 'var(--font-mono, "SF Mono", monospace)',
          fontSize: 11,
          padding: '3px 8px',
          minHeight: 24,
          background: 'rgba(127,127,127,0.10)',
          color: 'var(--ve-text, #ddd)',
          border: '1px solid rgba(127,127,127,0.18)',
          borderRadius: 4,
          cursor: 'pointer',
          letterSpacing: '0.02em',
        }}
      >
        {display}
      </button>
    );
  }
  return (
    <input
      autoFocus
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onBlur={() => {
        const t = parseTimecodeInput(value);
        if (t != null) onSeek(t);
        setEditing(false);
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          const t = parseTimecodeInput(value);
          if (t != null) onSeek(t);
          setEditing(false);
        } else if (e.key === 'Escape') {
          setEditing(false);
        }
        e.stopPropagation();
      }}
      style={{
        fontFamily: 'var(--font-mono, "SF Mono", monospace)',
        fontSize: 11,
        padding: '3px 8px',
        minHeight: 24,
        width: 110,
        background: 'var(--ve-surface, #1a1a1a)',
        color: 'var(--ve-text, #ddd)',
        border: '1px solid var(--accent, #6E7BFF)',
        borderRadius: 4,
        outline: 'none',
      }}
    />
  );
}

export default function Timeline({ compact = false, onSeek, onItemSelect, onSubtitleVisibilityChange }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  // Long-press on a track header opens its context menu on touch (the only
  // path to track ops — delete/duplicate/recolor — which is right-click only
  // on desktop).
  const trackLongPressRef = useRef({ timer: 0, x: 0, y: 0 });

  // Stable ref so async filmstrip generations can request a repaint
  // through the latest ``draw`` callback identity.
  const requestRedrawRef = useRef(null);

  // Imperative hover tracking — written from the mousemove handler and
  // read by ``draw``. Using a ref avoids forcing a React re-render on
  // every pixel of mouse movement; we just schedule a redraw via
  // ``requestRedrawRef`` when the hovered item / handle changes.
  const hoverRef = useRef({ itemId: null, cropId: null, edge: null });

  // Auto-scroll during drag: rAF loop that pans the timeline while the
  // user holds a clip / trim handle near a viewport edge. The
  // ``vx`` ref carries the current pan speed (px / frame), and
  // ``lastMoveEvent`` lets the rAF step re-run the drag onMove handler
  // so the dragged target tracks the new scroll position even when
  // the mouse itself is parked.
  const _autoScrollVxRef = useRef(0);
  const _autoScrollRafRef = useRef(0);
  const _lastMoveEventRef = useRef(null);

  // Subscribe per-slice for STATE that needs to trigger re-renders.
  // ACTIONS are pulled via ``_store.getState()`` below — they are
  // stable references for the lifetime of the store, so subscribing
  // to them was 25+ useless ``useSyncExternalStore`` registrations
  // each waking the component on every state change.
  const tracks = useTimelineStore((s) => s.tracks);
  const items = useTimelineStore((s) => s.items);
  const cropSegments = useTimelineStore((s) => s.cropSegments);
  const subjectKeyframes = useTimelineStore((s) => s.subjectKeyframes);

  // ── Mobile compact lanes (3.1) ──
  // Phones render slim lanes; tapping a track header expands one lane
  // at a time. laneHsRef feeds draw + hit tests (which read via refs).
  const { isMobile: isMobileViewport } = useResponsive();
  const [expandedTrackId, setExpandedTrackId] = useState(null);
  // Fit-to-window: the px budget the track stack may occupy. This MUST be the
  // real height of the scroll wrap, not a window-based guess — the timeline
  // lives in a fixed-height panel (``.ve-multitrack__content`` = timelineH),
  // which is usually shorter than the old ``innerHeight*0.72`` estimate. When
  // the two disagreed, lanes were sized for the taller guess and overflowed
  // the shorter panel, producing exactly the vertical scrollbar we don't want.
  // We measure the wrap directly (ResizeObserver) so lanes shrink to the
  // actual space and the stack always fits without scrolling.
  const wrapRef = useRef(null);
  const [fitHeight, setFitHeight] = useState(0);
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => setFitHeight(Math.max(120, el.clientHeight));
    measure();
    const ro = typeof ResizeObserver === 'function' ? new ResizeObserver(measure) : null;
    if (ro) ro.observe(el);
    window.addEventListener('resize', measure);
    return () => {
      if (ro) ro.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);
  const laneHs = useMemo(
    () => laneHeightsFor(tracks, isMobileViewport, expandedTrackId, fitHeight),
    [tracks, isMobileViewport, expandedTrackId, fitHeight],
  );
  const laneHsRef = useRef(laneHs);
  laneHsRef.current = laneHs;
  const selectedCropSegmentId = useTimelineStore((s) => s.selectedCropSegmentId);
  const playhead = useTimelineStore((s) => s.playhead);
  const duration = useTimelineStore((s) => s.duration);
  const zoom = useTimelineStore((s) => s.zoom);
  const scrollX = useTimelineStore((s) => s.scrollX);
  const snapEnabled = useTimelineStore((s) => s.snapEnabled);
  const rippleEnabled = useTimelineStore((s) => s.rippleEnabled);
  // Subscribed so a snap-guide change triggers a redraw via the
  // ``[draw]`` effect; ``draw`` reads the live value below.
  const snapLine = useTimelineStore((s) => s.snapLine);
  const sceneCuts = useTimelineStore((s) => s.sceneCuts);
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const selectedItemIds = useTimelineStore((s) => s.selectedItemIds);
  const activeTool = useTimelineStore((s) => s.activeTool);
  const isPlaying = useTimelineStore((s) => s.isPlaying);
  const segments = useTimelineStore((s) => s.segments);
  const hasOriginalSubtitles = useTimelineStore(
    (s) => (s._originalSubtitles || []).length > 0,
  );

  // Action references — stable across renders, so we read them off the
  // store reference once and avoid the per-action selector subscriptions.
  const _store = useTimelineStore;
  const selectCropSegment = _store.getState().selectCropSegment;
  const setPlayhead = _store.getState().setPlayhead;
  const setIsScrubbing = _store.getState().setIsScrubbing;
  const setZoom = _store.getState().setZoom;
  const setScrollX = _store.getState().setScrollX;
  const setSelectedItemId = _store.getState().setSelectedItemId;
  const setSelectedItemIds = _store.getState().setSelectedItemIds;
  const toggleSelectedItem = _store.getState().toggleSelectedItem;
  const updateItem = _store.getState().updateItem;
  const addItem = _store.getState().addItem;
  const splitItem = _store.getState().splitItem;
  const removeItem = _store.getState().removeItem;
  const toggleSnap = _store.getState().toggleSnap;
  const toggleRipple = _store.getState().toggleRipple;
  const addTrack = _store.getState().addTrack;
  const toggleTrackVisibility = _store.getState().toggleTrackVisibility;
  const toggleTrackMute = _store.getState().toggleTrackMute;
  const toggleTrackLock = _store.getState().toggleTrackLock;
  const updateTrack = _store.getState().updateTrack;
  const reorderTracks = _store.getState().reorderTracks;
  const resetSubtitleTimings = _store.getState().resetSubtitleTimings;
  const groupItems = _store.getState().groupItems;
  const ungroupItems = _store.getState().ungroupItems;

  // Track drag-to-reorder state
  const [dragTrackIdx, setDragTrackIdx] = useState(null);
  const [dragOverTrackIdx, setDragOverTrackIdx] = useState(null);
  const [renamingTrackId, setRenamingTrackId] = useState(null);

  const [isDragging, setIsDragging] = useState(false);
  const [dragInfo, setDragInfo] = useState(null);
  const [hoverTime, setHoverTime] = useState(null);
  const [contextMenu, setContextMenu] = useState(null);
  const [showAddTrack, setShowAddTrack] = useState(false);
  const [spaceHeld, setSpaceHeld] = useState(false);

  // Spacebar hold for pan mode
  useEffect(() => {
    const onKeyDown = (e) => { if (e.code === 'Space' && !e.repeat) setSpaceHeld(true); };
    const onKeyUp = (e) => { if (e.code === 'Space') setSpaceHeld(false); };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => { window.removeEventListener('keydown', onKeyDown); window.removeEventListener('keyup', onKeyUp); };
  }, []);

  const basePPS = compact ? 40 : 60;
  const pps = basePPS * zoom;

  // Ref for playhead pixel position — avoids stale closures in event handlers
  // without recreating callbacks on every playhead update (which happens every frame).
  const playheadRef = useRef(playhead);
  playheadRef.current = playhead;
  const ppsRef = useRef(pps);
  ppsRef.current = pps;
  // Drawn playhead position — trails the real playhead through a ~120 ms
  // ease on programmatic jumps (Home/End/ruler click) so the needle
  // glides instead of teleporting. Playback ticks and scrubs stay 1:1.
  const displayPlayheadRef = useRef(playhead);
  const glideRafRef = useRef(0);

  // ── Scrub-flush state lives on refs, not effect-local variables ──
  // The playhead drag effect previously stored ``_scrubRaf`` and
  // ``_scrubPending`` as ``let`` bindings inside the effect body. That
  // worked fine when the effect never re-ran mid-drag — but the parent
  // (VideoEditor) passes ``onSeek`` as a fresh inline arrow on every
  // render, which bumps Timeline's drag effect deps every frame and
  // tears the effect down. Each teardown drops the pending rAF handle
  // and the buffered scrub time with it → ``setPlayhead`` was never
  // flushed → playhead looked frozen during scrub.
  //
  // Refs survive effect rebuilds. The new effect body picks up the
  // same ``_scrubPending`` the old body queued, the same rAF handle
  // gets cancelled in cleanup, and the flush either fires or is
  // deterministically aborted on pointerup.
  const _scrubRafRef = useRef(0);
  const _scrubPendingRef = useRef(null);
  // Also ref the parent callbacks so the effect can read the latest
  // ``onSeek`` / ``setPlayhead`` without listing them as deps. Reading
  // through a ref is a cheap way to say "use the freshest value, but
  // don't trigger re-mounts when the value changes".
  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;
  const setPlayheadRef = useRef(setPlayhead);
  setPlayheadRef.current = setPlayhead;

  // ── Canvas rendering ──────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext && canvas.getContext('2d');
    // ``getContext`` can return null in restricted contexts (private
    // mode quirks, very old browsers, hardware-acceleration disabled).
    // Without this guard ``ctx.scale`` throws and the timeline goes
    // completely blank with no error boundary catch.
    if (!ctx) return;
    // ``roundRect`` polyfill — Safari < 16.4 and older Chromium don't
    // implement it, and this canvas calls it 7+ times per draw. Without
    // the polyfill the FIRST clip body draw throws ``TypeError:
    // ctx.roundRect is not a function`` and the entire multi-track view
    // blanks out (the parent error boundary then shows the failure UI).
    if (typeof ctx.roundRect !== 'function') {
      // eslint-disable-next-line no-param-reassign
      ctx.roundRect = function (x, y, w, h, r) {
        const radius = typeof r === 'number' ? Math.max(0, Math.min(r, Math.min(w, h) / 2)) : 0;
        if (radius <= 0) {
          this.rect(x, y, w, h);
          return this;
        }
        this.moveTo(x + radius, y);
        this.lineTo(x + w - radius, y);
        this.quadraticCurveTo(x + w, y, x + w, y + radius);
        this.lineTo(x + w, y + h - radius);
        this.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
        this.lineTo(x + radius, y + h);
        this.quadraticCurveTo(x, y + h, x, y + h - radius);
        this.lineTo(x, y + radius);
        this.quadraticCurveTo(x, y, x + radius, y);
        return this;
      };
    }
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const canvasW = rect.width;
    const canvasH = rect.height;
    const isDark = document.documentElement.dataset?.theme === 'dark';
    const contentLeft = LABEL_WIDTH;
    const contentWidth = canvasW - LABEL_WIDTH;
    const sx = scrollX;

    // ── Ruler ──
    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.03)';
    ctx.fillRect(0, 0, canvasW, RULER_HEIGHT);

    ctx.fillStyle = isDark ? 'rgba(255,255,255,0.62)' : 'rgba(0,0,0,0.58)';
    ctx.font = '500 11px "SF Mono", "Menlo", "Cascadia Code", monospace';
    ctx.textAlign = 'center';

    let interval = 1;
    if (pps < 15) interval = 10;
    else if (pps < 30) interval = 5;
    else if (pps < 60) interval = 2;
    else if (pps > 120) interval = 0.5;

    // Authoritative content extent = the clip/video duration. The timeline
    // represents EXACTLY this span, so the ruler ends here and every lane /
    // item is cut off cleanly at this boundary. Previously the extent padded
    // out by maxItemEnd*1.05, so a stale or mis-scoped item end (e.g. a
    // recovered full-video item on a 60s clip) drew colored bars far past the
    // last number marker. ``duration`` is the clip length when scoped to a
    // clip; fall back to the furthest item only when duration isn't known yet.
    const maxItemEnd = items.length > 0 ? Math.max(...items.map(it => it.end || 0)) : 0;
    const contentExtentSec = (duration && duration > 0.1)
      ? duration
      : Math.max(maxItemEnd, 30);
    const extentPx = contentLeft + contentExtentSec * pps - sx;
    for (let t = 0; t <= contentExtentSec + 1e-6; t += interval) {
      const x = contentLeft + t * pps - sx;
      if (x < contentLeft - 10 || x > canvasW + 10) continue;
      ctx.fillText(formatTime(t), x, 18);

      // Tick marks
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
      ctx.beginPath();
      ctx.moveTo(x, RULER_HEIGHT - 4);
      ctx.lineTo(x, RULER_HEIGHT);
      ctx.stroke();

      // Grid lines
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
      ctx.beginPath();
      ctx.moveTo(x, RULER_HEIGHT);
      ctx.lineTo(x, canvasH);
      ctx.stroke();
    }

    // Content-area width bounded by the clip extent, so lanes are cut off
    // cleanly at the clip's end marker (never a full-width lane running past
    // the last number marker).
    const laneContentW = Math.max(0, Math.min(contentWidth, extentPx - contentLeft));

    // ── Track lanes (ALL tracks always visible in timeline) ──
    tracks.forEach((track, idx) => {
      const laneHsD = laneHsRef.current;
      const y = laneTop(laneHsD, idx);
      const laneH = laneHsD[idx] ?? TRACK_HEIGHT;
      const isHidden = track.visible === false;

      // Card-style track lane with subtle alternating tint — Premiere
      // and VEED both use this trick to make adjacent rows easier to
      // scan even at a glance.
      const laneAlt = idx % 2 === 0;
      ctx.fillStyle = isDark
        ? (laneAlt ? 'rgba(255,255,255,0.035)' : 'rgba(255,255,255,0.02)')
        : (laneAlt ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.015)');
      ctx.fillRect(contentLeft, y, laneContentW, laneH);

      // Track label background — solid darker strip so headers read as
      // "side rail" instead of part of the timeline grid.
      ctx.fillStyle = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.04)';
      ctx.fillRect(0, y, LABEL_WIDTH - 1, laneH);

      // Soft inner border — same accent on both axes so the lane reads
      // as a single rounded "card" even though we don't actually round
      // the rect (would force a save/restore per track).
      ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';
      ctx.lineWidth = 1;
      ctx.strokeRect(contentLeft + 0.5, y + 0.5, Math.max(0, laneContentW - 1), laneH - 1);

      // Muted overlay
      if (track.muted) {
        ctx.fillStyle = isDark ? 'rgba(255,59,48,0.07)' : 'rgba(255,59,48,0.05)';
        ctx.fillRect(contentLeft, y, laneContentW, laneH);
      }

      // Hidden track overlay — dimmed with diagonal stripes pattern
      if (isHidden) {
        ctx.fillStyle = isDark ? 'rgba(0,0,0,0.35)' : 'rgba(128,128,128,0.15)';
        ctx.fillRect(contentLeft, y, laneContentW, laneH);
        ctx.fillRect(0, y, LABEL_WIDTH - 1, laneH);
      }

      // Track separator line — drawn in the gap between this track and the next
      if (idx < tracks.length - 1) {
        const sepY = y + laneH + Math.floor(TRACK_GAP / 2);
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, sepY);
        ctx.lineTo(canvasW, sepY);
        ctx.stroke();
      }
    });

    // ── Items (clips) ──
    // Precompute each item's next sibling start (on same track) so we can
    // cap the minimum render width without visually overlapping neighbors.
    const nextSiblingStartByItemId = {};
    {
      const byTrackSorted = {};
      for (const it of items) {
        (byTrackSorted[it.trackId] || (byTrackSorted[it.trackId] = [])).push(it);
      }
      for (const trackItems of Object.values(byTrackSorted)) {
        trackItems.sort((a, b) => a.start - b.start);
        for (let i = 0; i < trackItems.length; i++) {
          nextSiblingStartByItemId[trackItems[i].id] = i + 1 < trackItems.length
            ? trackItems[i + 1].start
            : Infinity;
        }
      }
    }

    items.forEach((item) => {
      const trackIdx = tracks.findIndex((t) => t.id === item.trackId);
      if (trackIdx < 0) return;
      const laneHsD = laneHsRef.current;
      const y = laneTop(laneHsD, trackIdx);
      const laneH = laneHsD[trackIdx] ?? TRACK_HEIGHT;
      const x1 = contentLeft + item.start * pps - sx;
      const x2 = contentLeft + item.end * pps - sx;
      const w = x2 - x1;

      // Clip rendering to track bounds AND to the clip content extent, so an
      // item never paints past the clip's end marker (a mis-scoped / stale
      // item end is cut off cleanly instead of bleeding to the panel edge).
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, y, Math.max(0, Math.min(canvasW, extentPx)), laneH);
      ctx.clip();

      if (x2 < contentLeft || x1 > canvasW || x1 >= extentPx) { ctx.restore(); return; }

      const color = TRACK_COLORS[item.type] || TRACK_COLORS.video;
      const isSelected = item.id === selectedItemId;
      const isMultiSelected = selectedItemIds.includes(item.id);
      const isHovered = hoverRef.current.itemId === item.id;

      // Clip body — enforce minimum visual width of 4px for visibility, but
      // cap against the next sibling's start position so short adjacent
      // items never visually overlap each other.
      const rr = 5;
      const clipX = Math.max(x1, contentLeft);
      const nextSiblingStart = nextSiblingStartByItemId[item.id];
      const maxRightX = nextSiblingStart !== undefined && nextSiblingStart !== Infinity
        ? contentLeft + nextSiblingStart * pps - sx
        : canvasW;
      const availableW = Math.max(0, maxRightX - clipX);
      // Right edge is the item's OWN end (x2), clamped to the canvas — NOT
      // clipX + full-width. The old ``min(w, canvasW - clipX)`` measured w
      // from the true (often off-screen-left) x1, so a long item that started
      // before the viewport was drawn all the way to the panel edge, well
      // past its real end — the "track element runs past the last marker" bug.
      const actualW = Math.min(x2, canvasW) - clipX;
      const clipW = Math.min(Math.max(actualW, 4), availableW || actualW);

      // Flat fill — selection/hover state changes opacity, not shading.
      const bodyY = y + 3;
      const bodyH = laneH - 6;
      const bodyAlpha = (isSelected || isMultiSelected) ? 'E0' : (isHovered ? 'B0' : '85');
      ctx.fillStyle = color + bodyAlpha;

      // Soft drop shadow when selected — only the selected item gets
      // the shadow so it visibly "lifts" off the lane.
      if (isSelected) {
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.45)';
        ctx.shadowBlur = 8;
        ctx.shadowOffsetY = 2;
      }
      ctx.beginPath();
      ctx.roundRect(clipX, bodyY, clipW, bodyH, rr);
      ctx.fill();
      if (isSelected) ctx.restore();

      // Glossy top highlight — 1px line at ~30% opacity gives the chip a
      // very Premiere-like specular sheen on dark themes.
      ctx.fillStyle = 'rgba(255,255,255,0.18)';
      ctx.fillRect(clipX + 1, bodyY + 1, Math.max(0, clipW - 2), 1);

      // ── Filmstrip thumbnails (long-zoom only) ──
      // Skip when:
      //   * the track isn't a video carrier (audio / text / shape items
      //     don't have meaningful thumbnails),
      //   * the clip is too narrow for thumbs to be useful,
      //   * we don't have a media URL on the item.
      // Throttled by computeThumbStops which targets ~thumbW pixels per
      // thumb so the count scales with zoom.
      const thumbCarrier = item.type === 'video' || item.type === 'image';
      const innerH = laneH - 8;
      const minClipPxForThumbs = 80;
      const mediaSrc = item.mediaSrc || item.src || null;
      if (thumbCarrier && clipW >= minClipPxForThumbs && innerH > 24 && mediaSrc) {
        const thumbW = Math.round(innerH * (16 / 9));
        // Thumbnail density must follow the VISIBLE pixels, not the whole clip.
        // computeThumbStops caps at 120 stops; spread across a 2-hour clip that
        // is one thumb every ~64s (~3800px apart), so none land in the ~1200px
        // viewport and the filmstrip looks blank. Generate stops only for the
        // clip ∩ viewport window (+ a thumb-width margin for edge tiles), so
        // density is identical whether the clip is 60s or 2h.
        const marginT = thumbW / Math.max(pps, 0.0001);
        const visT0 = Math.max(item.start, sx / pps - marginT);
        const visT1 = Math.min(item.end, (sx + contentWidth) / pps + marginT);
        const stops = visT1 > visT0
          ? computeThumbStops(visT0, visT1, pps, thumbW)
          : [];
        // With a loaded sprite sheet, slicing tiles is cheap, in-memory and
        // safe to parallelize — schedule EVERY missing visible tile in this
        // one pass so the strip fills in a frame or two. The hidden-<video>
        // fallback must stay serial (concurrent seeks on one element race),
        // so without a sprite we keep the old one-tile-per-redraw trickle.
        const spriteReady = hasSpriteReady(mediaSrc);
        const durationHint = item.end || 0;
        let scheduledRedraw = false;
        for (const t of stops) {
          const tx = contentLeft + t * pps - sx;
          // Skip thumbs entirely outside the clip's visible portion.
          if (tx + thumbW < clipX || tx > clipX + clipW) continue;
          const cached = getCachedThumbnail(mediaSrc, t, thumbW, innerH);
          if (cached) {
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(clipX, y + 4, clipW, innerH, rr - 1);
            ctx.clip();
            ctx.globalAlpha = 0.85;
            ctx.drawImage(cached, tx - thumbW / 2, y + 4, thumbW, innerH);
            ctx.globalAlpha = 1;
            ctx.restore();
          } else {
            // Skeleton shimmer while the tile generates — a moving
            // diagonal highlight instead of flat gray, so "loading"
            // reads differently from "empty clip".
            ctx.save();
            ctx.beginPath();
            ctx.roundRect(clipX, y + 4, clipW, innerH, rr - 1);
            ctx.clip();
            const phase = ((performance.now() / 1200) % 1) * (thumbW * 2);
            const gx = tx - thumbW / 2 - thumbW + phase;
            const grad = ctx.createLinearGradient(gx, 0, gx + thumbW, 0);
            grad.addColorStop(0, 'rgba(255,255,255,0.00)');
            grad.addColorStop(0.5, 'rgba(255,255,255,0.08)');
            grad.addColorStop(1, 'rgba(255,255,255,0.00)');
            ctx.fillStyle = grad;
            ctx.fillRect(tx - thumbW / 2, y + 4, thumbW, innerH);
            ctx.restore();
            if (spriteReady || !scheduledRedraw) {
              // ensureThumbnail dedupes per (src,t,w,h) so re-scheduling on
              // every draw is free; redraws are rAF-coalesced by the ref.
              ensureThumbnail(mediaSrc, t, thumbW, innerH, { durationHint })
                .then(() => requestRedrawRef.current && requestRedrawRef.current())
                .catch(() => {});
            }
            if (!scheduledRedraw) {
              scheduledRedraw = true;
              // Keep the shimmer moving while tiles generate
              setTimeout(() => requestRedrawRef.current && requestRedrawRef.current(), 120);
            }
          }
        }
      }

      // Selected border (solid white for primary, dashed cyan for
      // multi-select, soft white for hover). The hover state only
      // paints when nothing is selected so it doesn't fight the
      // selection border.
      if (isSelected) {
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(clipX, bodyY, clipW, bodyH, rr);
        ctx.stroke();
        ctx.lineWidth = 1;
      } else if (isMultiSelected) {
        ctx.strokeStyle = '#00D4FF';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.roundRect(clipX, bodyY, clipW, bodyH, rr);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      } else if (isHovered) {
        ctx.strokeStyle = 'rgba(255,255,255,0.55)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(clipX + 0.5, bodyY + 0.5, clipW - 1, bodyH - 1, rr);
        ctx.stroke();
      }

      // Group indicator: colored bottom bar
      if (item.groupId && clipW > 8) {
        const hue = hashGroupId(item.groupId) % 360;
        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
        ctx.fillRect(clipX + 2, y + laneH - 6, clipW - 4, 4);
      }

      // Transition indicator
      if (item.transition) {
        const transDur = item.transition.duration || 0.5;
        const transW = transDur * pps;
        ctx.fillStyle = 'rgba(255,255,255,0.25)';
        ctx.beginPath();
        ctx.moveTo(x1, y + 2);
        ctx.lineTo(x1 + transW, y + 2);
        ctx.lineTo(x1, y + laneH - 2);
        ctx.closePath();
        ctx.fill();
      }

      // Clip label
      if (w > 35) {
        ctx.fillStyle = '#fff';
        ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.textAlign = 'left';
        const label = item.textContent
          ? item.textContent.slice(0, 25)
          : item.subtitleText
            ? item.subtitleText.slice(0, 25)
            : item.type;
        // Subtle text shadow so labels stay readable on any track color.
        ctx.shadowColor = 'rgba(0,0,0,0.45)';
        ctx.shadowBlur = 2;
        ctx.fillText(label, Math.max(x1 + 10, contentLeft + 6), y + laneH / 2 + 4, w - 18);
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }

      // Trim handles. Two states:
      //   * Selected → bright fixed handles with a grip notch
      //   * Hovered  → translucent handles so the user knows where to
      //                grab without committing to a click.
      if (clipW > 24 && (isSelected || isHovered)) {
        // Hovering directly over an edge fattens + brightens THAT handle
        // so the 14px grab zone has a visible affordance before commit.
        const hoverEdge = isHovered ? hoverRef.current.edge : null;
        const handleWFor = (edge) => (isSelected || hoverEdge === edge)
          ? HANDLE_WIDTH_HOVER : HANDLE_WIDTH;
        const alphaFor = (edge) => isSelected ? 0.92 : (hoverEdge === edge ? 0.85 : 0.45);
        const handleY = bodyY + 3;
        const handleH = bodyH - 6;
        const handleRR = 2;
        ctx.fillStyle = '#FFFFFF';
        // Left handle
        const handleW = handleWFor('left');
        ctx.globalAlpha = alphaFor('left');
        ctx.beginPath();
        ctx.roundRect(clipX, handleY, handleW, handleH, handleRR);
        ctx.fill();
        // Right handle — anchored to the right edge of the visible clip,
        // not the off-screen x2, so it stays grabbable when the clip is
        // partially scrolled out.
        const rHandleW = handleWFor('right');
        const rightHandleX = Math.min(clipX + clipW - rHandleW, x2 - rHandleW);
        ctx.globalAlpha = alphaFor('right');
        ctx.beginPath();
        ctx.roundRect(rightHandleX, handleY, rHandleW, handleH, handleRR);
        ctx.fill();
        // Grip notch — two short vertical lines on each handle so the
        // user reads them as "drag here" the same way a Mac window
        // resize corner reads.
        if (isSelected) {
          ctx.globalAlpha = 0.5;
          ctx.fillStyle = '#000';
          const notchH = Math.min(10, handleH - 4);
          const notchY = handleY + (handleH - notchH) / 2;
          ctx.fillRect(clipX + handleW / 2 - 1, notchY, 1, notchH);
          ctx.fillRect(rightHandleX + rHandleW / 2 - 1, notchY, 1, notchH);
        }
        ctx.globalAlpha = 1;
      }

      // Effects indicator dot
      const effects = item.effects;
      if (effects && typeof effects === 'object' && !Array.isArray(effects)) {
        const hasEffects = Object.entries(effects).some(([k, v]) => v !== 0 && v !== undefined && v !== null);
        if (hasEffects) {
          ctx.fillStyle = '#FFD700';
          ctx.beginPath();
          ctx.arc(x2 - 12, y + 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore(); // End per-item track clip
    });

    // ── Crop segments (on the crop track) ──
    if (cropSegments && cropSegments.length > 0) {
      const cropTrackIdx = tracks.findIndex((t) => t.type === 'crop');
      if (cropTrackIdx >= 0) {
        const cropTrack = tracks[cropTrackIdx];
        if (cropTrack.visible !== false) {
          const laneHsC = laneHsRef.current;
          const cy = laneTop(laneHsC, cropTrackIdx);
          const cropLaneH = laneHsC[cropTrackIdx] ?? TRACK_HEIGHT;

          // Clip crop segment rendering to crop track bounds AND the clip
          // content extent, so a crop segment never paints past the clip's
          // end marker.
          ctx.save();
          ctx.beginPath();
          ctx.rect(0, cy, Math.max(0, Math.min(canvasW, extentPx)), cropLaneH);
          ctx.clip();

          // Precompute next-segment start times so min-width clamping doesn't
          // cause adjacent crop segments to visually overlap each other.
          const cropsSorted = [...cropSegments].sort((a, b) => a.startTime - b.startTime);
          const nextCropStartById = {};
          for (let i = 0; i < cropsSorted.length; i++) {
            nextCropStartById[cropsSorted[i].id] = i + 1 < cropsSorted.length
              ? cropsSorted[i + 1].startTime
              : Infinity;
          }

          cropSegments.forEach((seg) => {
            const cx1 = contentLeft + seg.startTime * pps - sx;
            const cx2 = contentLeft + seg.endTime * pps - sx;
            const cw = cx2 - cx1;
            if (cx2 < contentLeft || cx1 > canvasW) return;
            const clipCX = Math.max(cx1, contentLeft);
            const nextStart = nextCropStartById[seg.id];
            const maxRightCX = nextStart !== undefined && nextStart !== Infinity
              ? contentLeft + nextStart * pps - sx
              : canvasW;
            const availableCW = Math.max(0, maxRightCX - clipCX);
            // Right edge = the segment's OWN end (cx2), not clipCX + full width
            // (same off-screen-left overshoot fix as the main items).
            const actualCW = Math.min(cx2, canvasW) - clipCX;
            const clipCW = Math.min(Math.max(actualCW, 4), availableCW || actualCW);

            // Color by cluster or manual override
            const clrIdx = seg.isManualOverride ? 4 : Math.max(0, seg.clusterId);
            const baseColor = CROP_CLUSTER_COLORS[clrIdx % CROP_CLUSTER_COLORS.length];
            const isSelCrop = seg.id === selectedCropSegmentId;
            const isHoverCrop = hoverRef.current.cropId === seg.id;

            const sBodyY = cy + 4;
            const sBodyH = cropLaneH - 8;
            const sRr = 4;

            // Smooth-pan gradient — sample the SmoothDamp subject track across
            // the visible body so a human pan reads as a colour transition and
            // a held shot stays flat. A manually-pinned segment (or a missing
            // track) falls back to a flat fill at its own value.
            const fillAlpha = isSelCrop ? 0.9 : (isHoverCrop ? 0.72 : 0.55);
            const hasTrack = Array.isArray(subjectKeyframes) && subjectKeyframes.length > 1;
            if (!hasTrack || seg.isManualOverride) {
              ctx.fillStyle = cropColorAt(baseColor, seg.cropX, fillAlpha);
            } else {
              // Map gradient stops through the pixel→time inverse so the colour
              // stays aligned even when the segment is partly scrolled off.
              const grad = ctx.createLinearGradient(clipCX, 0, clipCX + clipCW, 0);
              const STOPS = 12;
              for (let gi = 0; gi <= STOPS; gi++) {
                const frac = gi / STOPS;
                const tt = (clipCX + frac * clipCW - contentLeft + sx) / pps;
                const cxPct = getCropXForTime(tt, cropSegments, subjectKeyframes);
                grad.addColorStop(frac, cropColorAt(baseColor, cxPct, fillAlpha));
              }
              ctx.fillStyle = grad;
            }

            if (isSelCrop) {
              ctx.save();
              ctx.shadowColor = 'rgba(0,0,0,0.45)';
              ctx.shadowBlur = 8;
              ctx.shadowOffsetY = 2;
            }
            ctx.beginPath();
            ctx.roundRect(clipCX, sBodyY, clipCW, sBodyH, sRr);
            ctx.fill();
            if (isSelCrop) ctx.restore();

            // Specular sheen
            ctx.fillStyle = 'rgba(255,255,255,0.18)';
            ctx.fillRect(clipCX + 1, sBodyY + 1, Math.max(0, clipCW - 2), 1);

            // Manual-override stripe — a vertical accent on the left so
            // a glance tells the user "this one is user-edited".
            if (seg.isManualOverride) {
              ctx.fillStyle = 'rgba(255,255,255,0.85)';
              ctx.fillRect(clipCX, sBodyY, 3, sBodyH);
            }

            // Borders for select / hover
            if (isSelCrop) {
              ctx.strokeStyle = '#FFFFFF';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.roundRect(clipCX, sBodyY, clipCW, sBodyH, sRr);
              ctx.stroke();
              ctx.lineWidth = 1;
            } else if (isHoverCrop) {
              ctx.strokeStyle = 'rgba(255,255,255,0.55)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.roundRect(clipCX + 0.5, sBodyY + 0.5, clipCW - 1, sBodyH - 1, sRr);
              ctx.stroke();
            }

            // Trim handles — match the main segment treatment so the
            // crop track resizes the same way an audio clip does.
            if (clipCW > 22 && (isSelCrop || isHoverCrop)) {
              const handleW = isSelCrop ? HANDLE_WIDTH_HOVER : HANDLE_WIDTH;
              const handleH = sBodyH - 6;
              const handleY = sBodyY + 3;
              ctx.fillStyle = '#FFFFFF';
              ctx.globalAlpha = isSelCrop ? 0.92 : 0.45;
              ctx.beginPath();
              ctx.roundRect(clipCX, handleY, handleW, handleH, 2);
              ctx.fill();
              const rightHX = Math.min(clipCX + clipCW - handleW, cx2 - handleW);
              ctx.beginPath();
              ctx.roundRect(rightHX, handleY, handleW, handleH, 2);
              ctx.fill();
              if (isSelCrop) {
                ctx.globalAlpha = 0.5;
                ctx.fillStyle = '#000';
                const notchH = Math.min(10, handleH - 4);
                const notchY = handleY + (handleH - notchH) / 2;
                ctx.fillRect(clipCX + handleW / 2 - 1, notchY, 1, notchH);
                ctx.fillRect(rightHX + handleW / 2 - 1, notchY, 1, notchH);
              }
              ctx.globalAlpha = 1;
            }

            // Label
            if (cw > 30) {
              ctx.fillStyle = '#fff';
              ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
              ctx.textAlign = 'left';
              ctx.shadowColor = 'rgba(0,0,0,0.45)';
              ctx.shadowBlur = 2;
              // Label the actual pan the gradient shows: "35→71%" for a glide,
              // "50%" for a hold. The segment's own cropX is a single
              // mid-transition value, so read the smoothed track at the
              // segment's ends instead (unless the user pinned it).
              let lbl = seg.label || `${Math.round(seg.cropX)}%`;
              const hasTrackLbl = Array.isArray(subjectKeyframes) && subjectKeyframes.length > 1;
              if (hasTrackLbl && !seg.isManualOverride) {
                const a = Math.round(getCropXForTime(seg.startTime, cropSegments, subjectKeyframes));
                const b = Math.round(getCropXForTime(Math.max(seg.startTime, seg.endTime - 0.001), cropSegments, subjectKeyframes));
                lbl = Math.abs(a - b) >= 2 ? `${a}→${b}%` : `${a}%`;
              }
              ctx.fillText(lbl, Math.max(cx1 + 8, contentLeft + 6), cy + cropLaneH / 2 + 4, cw - 16);
              ctx.shadowColor = 'transparent';
              ctx.shadowBlur = 0;
            }
          });
          ctx.restore(); // End crop track clip
        }
      }
    }

    // ── Segment boundaries ──
    if (segments && segments.length > 0) {
      segments.forEach((seg) => {
        const segStart = seg.start != null ? seg.start : 0;
        const segEnd = seg.end != null ? seg.end : 0;
        for (const edge of [segStart, segEnd]) {
          const ex = contentLeft + edge * pps - sx;
          if (ex < contentLeft || ex > canvasW) continue;
          ctx.strokeStyle = seg.color || '#FF9F0A';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.moveTo(ex, RULER_HEIGHT);
          ctx.lineTo(ex, canvasH);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.lineWidth = 1;
        }
        const sx1 = contentLeft + segStart * pps - sx;
        const sx2 = contentLeft + segEnd * pps - sx;
        if (sx2 > contentLeft && sx1 < canvasW && seg.label) {
          ctx.fillStyle = (seg.color || '#FF9F0A') + '18';
          ctx.fillRect(Math.max(sx1, contentLeft), RULER_HEIGHT, Math.min(sx2, canvasW) - Math.max(sx1, contentLeft), canvasH - RULER_HEIGHT);
          ctx.fillStyle = seg.color || '#FF9F0A';
          ctx.font = '9px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText(seg.label, Math.max(sx1 + 4, contentLeft + 4), RULER_HEIGHT + 10);
        }
      });
    }

    // ── Playhead ──
    // Read from ref for smooth rAF-driven updates (avoids stale closure).
    // The needle draws at the eased display position; the timecode bubble
    // always shows the REAL playhead so no fake frames are displayed.
    const phX = contentLeft + displayPlayheadRef.current * pps - sx;
    if (phX >= contentLeft && phX <= canvasW) {
      // Playhead line with subtle glow
      ctx.save();
      ctx.shadowColor = 'rgba(255, 92, 92, 0.55)';
      ctx.shadowBlur = 6;
      ctx.strokeStyle = '#FF5C5C';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(phX, 0);
      ctx.lineTo(phX, canvasH);
      ctx.stroke();
      ctx.restore();
      ctx.lineWidth = 1;

      // Playhead handle — wide flat-top "needle" with a grippy notch,
      // matching the Premiere / Resolve / Final Cut playhead idiom.
      // Wider top reads as a clear drag target; the tip points to the
      // exact frame.
      ctx.fillStyle = '#FF5C5C';
      ctx.beginPath();
      ctx.moveTo(phX - 11, 0);
      ctx.lineTo(phX + 11, 0);
      ctx.lineTo(phX + 11, RULER_HEIGHT - 14);
      ctx.lineTo(phX, RULER_HEIGHT - 2);
      ctx.lineTo(phX - 11, RULER_HEIGHT - 14);
      ctx.closePath();
      ctx.fill();

      // Glossy notch on the handle top
      ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.fillRect(phX - 6, 2, 12, 2);

      // Live time tooltip — readable bubble that follows the playhead
      // during scrub / playback. Sits just under the handle so it never
      // collides with the segment chips. Always painted so the user
      // doesn't lose track of the current frame.
      const tipText = formatTimeMs(playheadRef.current);
      ctx.font = '600 11px "SF Mono", "Cascadia Code", "Menlo", monospace';
      const tipPad = 6;
      const tipW = Math.ceil(ctx.measureText(tipText).width) + tipPad * 2;
      const tipH = 18;
      const tipY = RULER_HEIGHT - 1;
      let tipX = phX - tipW / 2;
      // Clamp so the bubble never escapes the visible canvas / track headers
      tipX = Math.max(contentLeft + 2, Math.min(canvasW - tipW - 2, tipX));
      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.45)';
      ctx.shadowBlur = 6;
      ctx.shadowOffsetY = 2;
      ctx.fillStyle = '#FF5C5C';
      ctx.beginPath();
      ctx.roundRect(tipX, tipY, tipW, tipH, 4);
      ctx.fill();
      ctx.restore();
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'center';
      ctx.fillText(tipText, tipX + tipW / 2, tipY + 13);
    }

    // ── Snap guide line ──
    // ``snapLine`` is subscribed at the top of the component so it
    // shows up in the ``draw`` useCallback's dep array; reading the
    // outer binding keeps the closure consistent with that dep.
    if (snapLine && isDragging) {
      const snapX = contentLeft + snapLine.time * pps - sx;
      if (snapX >= contentLeft && snapX <= canvasW) {
        ctx.save();
        ctx.strokeStyle = '#00D4FF';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(snapX, RULER_HEIGHT);
        ctx.lineTo(snapX, canvasH);
        ctx.stroke();

        // Small diamond indicator at top
        ctx.fillStyle = '#00D4FF';
        ctx.beginPath();
        ctx.moveTo(snapX, RULER_HEIGHT - 1);
        ctx.lineTo(snapX - 4, RULER_HEIGHT + 5);
        ctx.lineTo(snapX, RULER_HEIGHT + 11);
        ctx.lineTo(snapX + 4, RULER_HEIGHT + 5);
        ctx.closePath();
        ctx.fill();

        // Snap-target time bubble — readable timestamp + tiny "snap"
        // label so the user knows what they're locking onto. Sits just
        // below the diamond, clamped to the canvas so it can't escape.
        const snapText = formatTimeMs(snapLine.time);
        ctx.globalAlpha = 1;
        ctx.font = '600 10px "SF Mono", "Cascadia Code", "Menlo", monospace';
        const snapTipW = Math.ceil(ctx.measureText(snapText).width) + 12;
        const snapTipH = 16;
        const snapTipY = RULER_HEIGHT + 14;
        let snapTipX = snapX - snapTipW / 2;
        snapTipX = Math.max(contentLeft + 2, Math.min(canvasW - snapTipW - 2, snapTipX));
        ctx.fillStyle = 'rgba(0, 212, 255, 0.95)';
        ctx.beginPath();
        ctx.roundRect(snapTipX, snapTipY, snapTipW, snapTipH, 3);
        ctx.fill();
        ctx.fillStyle = '#001A26';
        ctx.textAlign = 'center';
        ctx.fillText(snapText, snapTipX + snapTipW / 2, snapTipY + 11);

        ctx.restore();
      }
    }

    // ── Hover indicator ──
    if (hoverTime !== null) {
      const hx = contentLeft + hoverTime * pps - sx;
      if (hx >= contentLeft && hx <= canvasW) {
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(hx, RULER_HEIGHT);
        ctx.lineTo(hx, canvasH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Tooltip
        ctx.fillStyle = 'rgba(0,0,0,0.85)';
        const tooltipText = formatTimeMs(hoverTime);
        const tw = ctx.measureText(tooltipText).width + 10;
        ctx.beginPath();
        ctx.roundRect(Math.min(hx - tw / 2, canvasW - tw - 4), 3, tw, 16, 4);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = '10px "SF Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(tooltipText, Math.min(hx, canvasW - tw / 2 - 4), 14);
      }
    }

    // ── Razor cursor indicator ──
    if (activeTool === 'razor' && hoverTime !== null) {
      const rx = contentLeft + hoverTime * pps - sx;
      if (rx >= contentLeft && rx <= canvasW) {
        ctx.strokeStyle = '#FF9500';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(rx, RULER_HEIGHT);
        ctx.lineTo(rx, canvasH);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
      }
    }
  }, [tracks, items, duration, zoom, scrollX, selectedItemId, selectedItemIds, hoverTime, pps, compact, activeTool, segments, cropSegments, subjectKeyframes, selectedCropSegmentId, snapLine]);
  // ``playhead`` is intentionally absent: every play tick was both
  // recreating ``draw`` (causing the ``[draw]`` effect below to re-fire)
  // AND running the rAF loop. Two redraw mechanisms stacked.
  // The rAF loop below is now the SINGLE source of repaint during
  // playback; ``playhead`` is read off ``playheadRef`` inside ``draw``.

  // ── Playhead seek glide (motion polish 2.6) ──
  // Programmatic jumps ease the DRAWN needle over ~120 ms. During
  // playback the rAF loop below repaints, so this effect only syncs the
  // display value; while paused it also requests the repaint. Scrub-size
  // deltas (< 0.2 s) and prefers-reduced-motion track 1:1.
  useEffect(() => {
    const from = displayPlayheadRef.current;
    const to = playhead;
    cancelAnimationFrame(glideRafRef.current);
    const reduceMotion = typeof window !== 'undefined' && window.matchMedia
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (isPlaying || reduceMotion || Math.abs(to - from) < 0.2) {
      displayPlayheadRef.current = to;
      if (!isPlaying) draw();
      return undefined;
    }
    const DUR = 120;
    const start = performance.now();
    const easeOutQuart = (t) => 1 - Math.pow(1 - t, 4);
    const step = (now) => {
      const p = Math.min(1, (now - start) / DUR);
      displayPlayheadRef.current = from + (to - from) * easeOutQuart(p);
      draw();
      if (p < 1) glideRafRef.current = requestAnimationFrame(step);
    };
    glideRafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(glideRafRef.current);
  }, [playhead, isPlaying, draw]);

  // Continuous redraw during playback. Throttled to ~30 Hz so a long
  // timeline doesn't repaint at full display refresh.
  useEffect(() => {
    if (!isPlaying) return;
    let rafId;
    let lastPaint = 0;
    const PAINT_INTERVAL_MS = 33;   // ~30 fps redraw is plenty for the
                                    // playhead bar; the canvas is static
                                    // otherwise.
    const tick = (now) => {
      if (now - lastPaint >= PAINT_INTERVAL_MS) {
        draw();
        lastPaint = now;
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, draw]);

  // Auto-scroll timeline to keep playhead visible during playback
  useEffect(() => {
    if (!isPlaying) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const visibleWidth = canvas.getBoundingClientRect().width - LABEL_WIDTH;
    if (visibleWidth <= 0) return;
    const playheadPx = playhead * pps;
    const viewStart = scrollX;
    const viewEnd = scrollX + visibleWidth;
    // When playhead moves past 80% of the visible area, scroll to keep it at 20%
    if (playheadPx > viewEnd - visibleWidth * 0.2) {
      setScrollX(Math.max(0, playheadPx - visibleWidth * 0.2));
    } else if (playheadPx < viewStart) {
      setScrollX(Math.max(0, playheadPx - visibleWidth * 0.1));
    }
  }, [isPlaying, playhead, pps, scrollX, setScrollX]);

  // Redraw on state changes
  useEffect(() => { draw(); }, [draw]);

  // Redraw when the playhead moves while the video is paused. The
  // primary ``draw`` callback intentionally omits ``playhead`` from
  // its dependency array (playback already owns its own rAF repaint
  // loop, and adding ``playhead`` there would stack two redraw
  // mechanisms at 60 Hz). That leaves paused-state playhead updates
  // — e.g. scrubbing the multi-track timeline, arrow-key nudges,
  // ``setPlayhead`` calls from the video's ``seeked`` event — without
  // any automatic redraw trigger, which is what made the playhead
  // appear to freeze under the cursor during scrub. Piggy-backing on
  // ``requestRedrawRef`` uses the latest ``draw`` closure without
  // pulling it into this effect's dep list (which would re-fire on
  // every state change the draw callback tracks).
  useEffect(() => {
    if (isPlaying) return;
    requestRedrawRef.current?.();
  }, [playhead, isPlaying]);

  // Track the latest ``draw`` so async filmstrip generations don't
  // hold a stale closure. We assign the ref in an effect so React
  // sees the canonical render order.
  useEffect(() => {
    requestRedrawRef.current = draw;
    return () => { requestRedrawRef.current = null; };
  }, [draw]);

  // Redraw when a source's sprite sheet appears or upgrades (coarse→fine):
  // the tile cache for that source was purged, so the next draw re-slices
  // from the new sheet. Without this, the sharper filmstrip only showed up
  // after the user next interacted with the timeline.
  useEffect(() => {
    const onFilmstripUpdate = () => { requestRedrawRef.current?.(); };
    window.addEventListener(FILMSTRIP_UPDATED_EVENT, onFilmstripUpdate);
    return () => window.removeEventListener(FILMSTRIP_UPDATED_EVENT, onFilmstripUpdate);
  }, []);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  // ── Pointer helpers ────────────────────────────────────────────────────────
  const getTimeFromX = useCallback((clientX) => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left - LABEL_WIDTH + scrollX;
    return Math.max(0, x / pps);
  }, [pps, scrollX]);

  const getTrackFromY = useCallback((clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const y = clientY - rect.top;
    const idx = laneIndexFromY(laneHsRef.current, y);
    // Read latest tracks from store to avoid stale closures
    const currentTracks = useTimelineStore.getState().tracks;
    if (idx < 0 || idx >= currentTracks.length) return null;
    return currentTracks[idx] || null;
  }, []);

  const hitTestItem = useCallback((clientX, clientY) => {
    // Read ALL values from getState() to avoid stale closures
    const { items: currentItems, scrollX: currentScrollX, selectedItemId: selectedId, tracks: currentTracks } = useTimelineStore.getState();
    const currentPps = ppsRef.current;

    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const x = px - LABEL_WIDTH + currentScrollX;
    const time = Math.max(0, x / currentPps);

    // Inline track-from-Y to use currentTracks from getState()
    const mouseY = clientY - rect.top;
    const trackIdx = laneIndexFromY(laneHsRef.current, mouseY);
    if (trackIdx < 0 || trackIdx >= currentTracks.length) return null;
    const track = currentTracks[trackIdx];
    if (!track) return null;

    // Collect ALL matching items, then pick the best one
    const matches = [];
    for (const item of currentItems) {
      if (item.trackId !== track.id) continue;
      if (time < item.start || time > item.end) continue;

      const x1 = LABEL_WIDTH + item.start * currentPps - currentScrollX;
      const x2 = LABEL_WIDTH + item.end * currentPps - currentScrollX;

      let edge = 'body';
      if (Math.abs(px - x1) < HANDLE_HIT_AREA) edge = 'left';
      else if (Math.abs(px - x2) < HANDLE_HIT_AREA) edge = 'right';

      matches.push({ item, edge, x1, x2 });
    }

    if (matches.length === 0) return null;

    // Priority 1: currently selected item (for boundary clicks between adjacent items)
    const selected = matches.find(m => m.item.id === selectedId);
    if (selected) return selected;

    // Priority 2: narrowest (most specific) clip at the click point, then newest (last in array)
    matches.sort((a, b) => {
      const widthDiff = (a.x2 - a.x1) - (b.x2 - b.x1);
      if (Math.abs(widthDiff) > 0.5) return widthDiff;
      // Same width: prefer the one later in the items array (newest)
      return currentItems.indexOf(b.item) - currentItems.indexOf(a.item);
    });
    return matches[0];
  }, []);

  // Hit-test a crop segment on the crop track. Returns { seg, edge } or null.
  const hitTestCropSegment = useCallback((clientX, clientY) => {
    const { cropSegments: segs, scrollX: currentScrollX, tracks: currentTracks } = useTimelineStore.getState();
    if (!segs?.length) return null;

    const cropTrackIdx = currentTracks.findIndex((t) => t.type === 'crop');
    if (cropTrackIdx < 0) return null;
    const cropTrack = currentTracks[cropTrackIdx];
    if (!cropTrack || cropTrack.visible === false || cropTrack.locked) return null;

    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const px = clientX - rect.left;
    const mouseY = clientY - rect.top;
    const currentPps = ppsRef.current;

    // Verify Y is on the crop track
    const trackIdx = laneIndexFromY(laneHsRef.current, mouseY);
    if (trackIdx !== cropTrackIdx) return null;

    const x = px - LABEL_WIDTH + currentScrollX;
    const time = Math.max(0, x / currentPps);

    for (const seg of segs) {
      if (time < seg.startTime || time > seg.endTime) continue;
      const x1 = LABEL_WIDTH + seg.startTime * currentPps - currentScrollX;
      const x2 = LABEL_WIDTH + seg.endTime * currentPps - currentScrollX;
      let edge = 'body';
      if (Math.abs(px - x1) < HANDLE_HIT_AREA) edge = 'left';
      else if (Math.abs(px - x2) < HANDLE_HIT_AREA) edge = 'right';
      return { seg, edge, x1, x2 };
    }
    return null;
  }, []);

  // ── Pointer events ─────────────────────────────────────────────────────────
  const onPointerDown = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    // Right-click context menu — clip menu over lanes, track menu over
    // the header column (both rendered by the shared ContextMenu).
    if (e.button === 2) {
      e.preventDefault();
      const px = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      if (px < LABEL_WIDTH && mouseY > RULER_HEIGHT) {
        const trackIdx = laneIndexFromY(laneHsRef.current, mouseY);
        const track = useTimelineStore.getState().tracks[trackIdx];
        if (track) {
          setContextMenu({ kind: 'track', x: e.clientX, y: e.clientY, trackId: track.id });
          return;
        }
      }
      const time = getTimeFromX(e.clientX);
      const hit = hitTestItem(e.clientX, e.clientY);
      if (hit?.item) {
        // Right-click selects like every commercial NLE, so the menu
        // and the inspector agree about the target.
        setSelectedItemId(hit.item.id);
      }
      setContextMenu({
        kind: 'item',
        x: e.clientX,
        y: e.clientY,
        time,
        item: hit?.item || null,
      });
      return;
    }

    setContextMenu(null);

    // Middle-click or space+left-click: pan/scroll
    if (e.button === 1 || (e.button === 0 && spaceHeld)) {
      e.preventDefault();
      setIsDragging(true);
      setDragInfo({ type: 'pan', startX: e.clientX, origScrollX: scrollX });
      return;
    }

    // Razor tool: split on click (blocked on locked tracks)
    if (activeTool === 'razor') {
      const time = getTimeFromX(e.clientX);
      const hit = hitTestItem(e.clientX, e.clientY);
      if (hit?.item) {
        const currentTracks = useTimelineStore.getState().tracks;
        const itemTrack = currentTracks.find((t) => t.id === hit.item.trackId);
        if (itemTrack?.locked) return; // Cannot split items on locked tracks
        splitItem(hit.item.id, time);
      }
      return;
    }

    // Playhead grab: check if click is near the playhead line or handle.
    // Only grab the playhead directly from the ruler area (triangle handle).
    // Clicks on items near the playhead should select the item, not grab the playhead.
    const currentScrollX = useTimelineStore.getState().scrollX;
    const playheadPixelX = rect.left + LABEL_WIDTH + playheadRef.current * ppsRef.current - currentScrollX;
    const mouseY = e.clientY - rect.top;
    const distToPlayhead = Math.abs(e.clientX - playheadPixelX);
    const isInRuler = mouseY <= RULER_HEIGHT + 8;
    if (isInRuler && distToPlayhead <= PLAYHEAD_GRAB_WIDTH) {
      // Grab the playhead directly from the ruler
      const time = getTimeFromX(e.clientX);
      setIsScrubbing(true);
      setPlayhead(time);
      onSeek?.(time);
      setIsDragging(true);
      setDragInfo({ type: 'scrub', startX: e.clientX });
      return;
    }

    const hit = hitTestItem(e.clientX, e.clientY);
    if (hit) {
      // Ctrl/Cmd+Click: toggle multi-select (no drag)
      if (e.ctrlKey || e.metaKey) {
        toggleSelectedItem(hit.item.id);
        onItemSelect?.(hit.item);
        return;
      }

      // Alt+Click: select all group members (group-aware selection)
      if (e.altKey && hit.item.groupId) {
        const currentItems = useTimelineStore.getState().items;
        const groupMembers = currentItems.filter(it => it.groupId === hit.item.groupId).map(it => it.id);
        setSelectedItemIds(groupMembers);
        useTimelineStore.setState({ selectedItemId: hit.item.id });
        onItemSelect?.(hit.item);
        // Continue to drag logic below
      } else {
        // Plain click: always single-select the clicked item
        setSelectedItemId(hit.item.id);
        onItemSelect?.(hit.item);
      }

      // Check if item is on a locked track — allow selection but block drag/trim
      const { tracks: latestTracks, items: latestItems } = useTimelineStore.getState();
      const itemTrack = latestTracks.find((t) => t.id === hit.item.trackId);
      if (itemTrack?.locked) {
        // Selection allowed, but no drag/trim
      } else if (hit.edge === 'left' || hit.edge === 'right') {
        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        setDragInfo({
          type: 'trim',
          itemId: hit.item.id,
          edge: hit.edge,
          origStart: hit.item.start,
          origEnd: hit.item.end,
          origTrimStart: hit.item.trimStart,
          origTrimEnd: hit.item.trimEnd,
          startX: e.clientX,
        });
      } else {
        // Multi-item drag: build snapshots for all selected items
        const currentSelectedIds = useTimelineStore.getState().selectedItemIds;
        const dragIds = currentSelectedIds.includes(hit.item.id)
          ? currentSelectedIds
          : [hit.item.id];
        const snapshots = dragIds.map(id => {
          const it = latestItems.find(i => i.id === id);
          return it ? { id, origStart: it.start, origEnd: it.end, origTrackId: it.trackId } : null;
        }).filter(Boolean);

        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        setDragInfo({
          type: 'move',
          itemId: hit.item.id,
          snapshots,
          origStart: hit.item.start,
          origEnd: hit.item.end,
          origTrackId: hit.item.trackId,
          startX: e.clientX,
          startY: e.clientY,
        });
      }
    } else {
      // No item hit — check crop track click, then playhead, then seek
      const time = getTimeFromX(e.clientX);

      // Check if click is on a crop segment (with edge detection for trim/move)
      const cropHit = hitTestCropSegment(e.clientX, e.clientY);
      if (cropHit) {
        selectCropSegment(cropHit.seg.id);
        setSelectedItemId(null);
        onItemSelect?.(null); // Open properties panel for crop segment

        // Pause undo history during drag so intermediate frames don't flood it
        useTimelineStore.temporal.getState().pause();
        setIsDragging(true);
        if (cropHit.edge === 'left' || cropHit.edge === 'right') {
          setDragInfo({
            type: 'trim-crop',
            segId: cropHit.seg.id,
            edge: cropHit.edge,
            origStart: cropHit.seg.startTime,
            origEnd: cropHit.seg.endTime,
            startX: e.clientX,
          });
        } else {
          setDragInfo({
            type: 'move-crop',
            segId: cropHit.seg.id,
            origStart: cropHit.seg.startTime,
            origEnd: cropHit.seg.endTime,
            startX: e.clientX,
          });
        }
        return;
      }

      if (!isInRuler && distToPlayhead <= Math.max(PLAYHEAD_GRAB_WIDTH / 2, 8)) {
        // Grab the playhead line directly
        setIsScrubbing(true);
        setPlayhead(time);
        onSeek?.(time);
        setIsDragging(true);
        setDragInfo({ type: 'scrub', startX: e.clientX });
      } else {
        // Click on empty area: seek + deselect
        setIsScrubbing(true);
        setPlayhead(time);
        setSelectedItemId(null);
        selectCropSegment(null);
        onSeek?.(time);
        setIsDragging(true);
        setDragInfo({ type: 'scrub', startX: e.clientX });
      }
    }
  }, [hitTestItem, hitTestCropSegment, getTimeFromX, setPlayhead, setSelectedItemId, setSelectedItemIds, toggleSelectedItem, selectCropSegment, onSeek, activeTool, splitItem, spaceHeld, scrollX, onItemSelect]);

  useEffect(() => {
    if (!isDragging || !dragInfo) return;

    // ── rAF-coalesced scrub flush ──
    // Pointer events fire at >100Hz on modern devices. Calling
    // ``setPlayhead`` + ``onSeek`` on every event triggers a Zustand
    // store update (re-renders Timeline + every other subscriber) AND
    // a video-element ``currentTime`` write (which has to seek to a
    // keyframe + decode) per frame — the result is the choppy scrub
    // the user reported. Coalesce both writes to one per animation
    // frame so the video gets at most ~60 seeks per second instead of
    // hundreds. The scrub thumb itself updates at full pointer-event
    // rate visually because the canvas redraw is decoupled.
    // Pending rAF handle + buffered time live on refs so they
    // survive an effect re-mount caused by a dep identity change
    // mid-drag. Without this, a fresh ``onSeek`` identity from the
    // parent tore down the effect every frame, cancelled the rAF, and
    // dropped the buffered time — making the playhead look frozen.
    const _flushScrub = () => {
      _scrubRafRef.current = 0;
      const t = _scrubPendingRef.current;
      if (t == null) return;
      _scrubPendingRef.current = null;
      try { setPlayheadRef.current?.(t); } catch { /* noop */ }
      try { onSeekRef.current?.(t); } catch { /* noop */ }
      // ``playhead`` is intentionally NOT in ``draw``'s dep list (so
      // playback doesn't recreate the callback 60×/sec), which means
      // a plain ``setPlayhead`` during scrub won't automatically
      // re-run the ``useEffect(() => draw(), [draw])``. Request a
      // redraw directly so the playhead line visually tracks the
      // cursor at rAF rate — without this call the playhead appeared
      // frozen under the mouse for paused scrubs.
      try { requestRedrawRef.current?.(); } catch { /* noop */ }
    };

    const onMove = (e) => {
      if (dragInfo.type === 'pan') {
        const dx = e.clientX - dragInfo.startX;
        setScrollX(Math.max(0, dragInfo.origScrollX - dx));
        return;
      }

      // Auto-scroll the timeline while dragging near either viewport
      // edge — the single most important affordance for "let me move
      // this clip 30 seconds to the right without zooming out first".
      // Every state-of-the-art NLE (Premiere, DaVinci, Final Cut, VEED)
      // does this. The further into the edge zone the cursor sits,
      // the faster the scroll, matching Premiere's behavior.
      const _affectsScroll = (
        dragInfo.type === 'trim' ||
        dragInfo.type === 'move' ||
        dragInfo.type === 'trim-crop' ||
        dragInfo.type === 'move-crop' ||
        dragInfo.type === 'scrub'
      );
      if (_affectsScroll) {
        try {
          const _canvas = canvasRef.current;
          if (_canvas) {
            const _rect = _canvas.getBoundingClientRect();
            const _edgeZone = 50; // px from edge that triggers scroll
            const _maxSpeed = 22;  // px per frame at the very edge
            const _contentLeft = _rect.left + LABEL_WIDTH;
            const _distLeft = e.clientX - _contentLeft;
            const _distRight = _rect.right - e.clientX;
            let _vx = 0;
            if (_distLeft < _edgeZone && _distLeft > -200) {
              _vx = -_maxSpeed * Math.min(1, Math.max(0, 1 - _distLeft / _edgeZone));
            } else if (_distRight < _edgeZone && _distRight > -200) {
              _vx = _maxSpeed * Math.min(1, Math.max(0, 1 - _distRight / _edgeZone));
            }
            _autoScrollVxRef.current = _vx;
            if (_vx !== 0 && !_autoScrollRafRef.current) {
              const _step = () => {
                const v = _autoScrollVxRef.current;
                if (!v) { _autoScrollRafRef.current = 0; return; }
                const cur = useTimelineStore.getState().scrollX;
                useTimelineStore.getState().setScrollX(Math.max(0, cur + v));
                // Re-run onMove with the last known clientX so the
                // drag target tracks the new scroll position even
                // when the mouse is parked.
                if (_lastMoveEventRef.current) {
                  onMove(_lastMoveEventRef.current);
                }
                _autoScrollRafRef.current = requestAnimationFrame(_step);
              };
              _autoScrollRafRef.current = requestAnimationFrame(_step);
            }
            _lastMoveEventRef.current = e;
          }
        } catch (_) { /* never break the drag because of auto-scroll */ }
      }

      const time = getTimeFromX(e.clientX);

      if (dragInfo.type === 'scrub') {
        // Buffer the latest time; the rAF flush picks the freshest
        // value when the browser is ready to paint.
        _scrubPendingRef.current = time;
        if (_scrubRafRef.current === 0) {
          _scrubRafRef.current = requestAnimationFrame(_flushScrub);
        }
        return;
      } else if (dragInfo.type === 'trim') {
        const item = items.find((i) => i.id === dragInfo.itemId);
        if (!item) return;
        const mediaLib = useTimelineStore.getState().mediaLibrary;
        const maxDur = getMaxItemDuration(item, mediaLib);
        const setSnapLine = useTimelineStore.getState().setSnapLine;
        const ripple = useTimelineStore.getState().rippleEnabled;
        const rippleShiftAfter = useTimelineStore.getState().rippleShiftAfter;

        if (dragInfo.edge === 'left') {
          let newStart = Math.max(0, Math.min(dragInfo.origEnd - 0.1, time));
          if (maxDur < Infinity) {
            const minStart = dragInfo.origEnd - maxDur;
            newStart = Math.max(newStart, minStart);
          }
          if (snapEnabled) {
            const snap = findSnapTarget(newStart, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
            if (snap) {
              newStart = Math.max(maxDur < Infinity ? dragInfo.origEnd - maxDur : 0, snap.snappedTime);
              setSnapLine({ time: snap.snapTarget });
            } else {
              setSnapLine(null);
            }
          }
          updateItem(dragInfo.itemId, { start: newStart });
          // Ripple: trimming the LEFT edge inward (later) means
          // downstream items should pull in by the same delta so the
          // gap doesn't grow. ``delta = newStart - origStart``.
          if (ripple) {
            const delta = newStart - dragInfo.origStart;
            if (delta !== 0) {
              rippleShiftAfter(dragInfo.origEnd, delta, [dragInfo.itemId]);
            }
          }
        } else {
          let newEnd = Math.max(dragInfo.origStart + 0.1, time);
          if (maxDur < Infinity) {
            newEnd = Math.min(newEnd, item.start + maxDur);
          }
          if (snapEnabled) {
            const snap = findSnapTarget(newEnd, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
            if (snap) {
              newEnd = maxDur < Infinity
                ? Math.min(snap.snappedTime, item.start + maxDur)
                : snap.snappedTime;
              setSnapLine({ time: snap.snapTarget });
            } else {
              setSnapLine(null);
            }
          }
          updateItem(dragInfo.itemId, { end: newEnd });
          // Ripple: trimming the RIGHT edge shifts everything that
          // started at or after the old end by ``newEnd - origEnd``.
          if (ripple) {
            const delta = newEnd - dragInfo.origEnd;
            if (delta !== 0) {
              rippleShiftAfter(dragInfo.origEnd, delta, [dragInfo.itemId]);
            }
          }
        }
      } else if (dragInfo.type === 'move') {
        const dx = (e.clientX - dragInfo.startX) / pps;
        const snapshots = dragInfo.snapshots || [{ id: dragInfo.itemId, origStart: dragInfo.origStart, origEnd: dragInfo.origEnd, origTrackId: dragInfo.origTrackId }];
        const primarySnap = snapshots[0];
        let newStart = Math.max(0, primarySnap.origStart + dx);
        const dur = primarySnap.origEnd - primarySnap.origStart;

        // Snap (both edges of the primary moving item)
        const setSnapLine = useTimelineStore.getState().setSnapLine;
        if (snapEnabled) {
          const snapStart = findSnapTarget(newStart, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);
          const snapEnd = findSnapTarget(newStart + dur, items, dragInfo.itemId, playheadRef.current, duration, pps, sceneCuts);

          if (snapStart && (!snapEnd || Math.abs(snapStart.snappedTime - newStart) * pps <= Math.abs(snapEnd.snappedTime - (newStart + dur)) * pps)) {
            newStart = snapStart.snappedTime;
            setSnapLine({ time: snapStart.snapTarget });
          } else if (snapEnd) {
            newStart = snapEnd.snappedTime - dur;
            setSnapLine({ time: snapEnd.snapTarget });
          } else {
            setSnapLine(null);
          }
        } else {
          setSnapLine(null);
        }

        // Compute actual delta from snapped primary position
        const actualDelta = newStart - primarySnap.origStart;

        // Track change (only for single-item drag on the primary item)
        const track = getTrackFromY(e.clientY);
        let primaryTrackId = dragInfo.origTrackId;
        if (track && snapshots.length === 1) {
          const draggedItem = useTimelineStore.getState().items.find((i) => i.id === dragInfo.itemId);
          const itemType = draggedItem?.type;
          const trackType = track.type;
          const compatible =
            (itemType === 'video' && trackType === 'video') ||
            (itemType === 'audio' && trackType === 'audio') ||
            ((itemType === 'text' || itemType === 'shape' || itemType === 'image' || itemType === 'overlay') && trackType === 'overlay') ||
            (itemType === 'subtitle' && trackType === 'subtitle') ||
            (itemType === 'crop' && trackType === 'crop');
          if (compatible && !track.locked) primaryTrackId = track.id;
        }

        // Apply to all items in the drag group
        for (const snap of snapshots) {
          const itemNewStart = Math.max(0, snap.origStart + actualDelta);
          const itemDur = snap.origEnd - snap.origStart;
          const itemTrackId = snap.id === dragInfo.itemId ? primaryTrackId : snap.origTrackId;
          updateItem(snap.id, { start: itemNewStart, end: itemNewStart + itemDur, trackId: itemTrackId });
        }
      } else if (dragInfo.type === 'trim-crop') {
        // Trim the left or right edge of a crop segment
        const { cropSegments: segs, updateCropSegment } = useTimelineStore.getState();
        const seg = segs.find((s) => s.id === dragInfo.segId);
        if (!seg) return;
        const idx = segs.findIndex((s) => s.id === dragInfo.segId);
        const prev = idx > 0 ? segs[idx - 1] : null;
        const next = idx < segs.length - 1 ? segs[idx + 1] : null;
        const MIN_DUR = 0.05; // minimum 50ms so handles don't collapse

        if (dragInfo.edge === 'left') {
          let newStart = Math.max(0, time);
          // Clamp against previous neighbor's end
          if (prev) newStart = Math.max(newStart, prev.endTime);
          // Keep at least MIN_DUR
          newStart = Math.min(newStart, dragInfo.origEnd - MIN_DUR);
          updateCropSegment({ ...seg, startTime: newStart, endTime: dragInfo.origEnd });
        } else {
          let newEnd = Math.max(dragInfo.origStart + MIN_DUR, time);
          // Clamp against next neighbor's start
          if (next) newEnd = Math.min(newEnd, next.startTime);
          // Clamp against clip duration (if known)
          if (duration > 0) newEnd = Math.min(newEnd, duration);
          updateCropSegment({ ...seg, startTime: dragInfo.origStart, endTime: newEnd });
        }
      } else if (dragInfo.type === 'move-crop') {
        // Move (slide) a crop segment along the timeline
        const { cropSegments: segs, updateCropSegment } = useTimelineStore.getState();
        const seg = segs.find((s) => s.id === dragInfo.segId);
        if (!seg) return;
        const idx = segs.findIndex((s) => s.id === dragInfo.segId);
        const prev = idx > 0 ? segs[idx - 1] : null;
        const next = idx < segs.length - 1 ? segs[idx + 1] : null;
        const dur = dragInfo.origEnd - dragInfo.origStart;
        const dx = (e.clientX - dragInfo.startX) / pps;
        let newStart = Math.max(0, dragInfo.origStart + dx);
        // Clamp within neighbor boundaries
        if (prev) newStart = Math.max(newStart, prev.endTime);
        if (next) newStart = Math.min(newStart, next.startTime - dur);
        if (duration > 0) newStart = Math.min(newStart, Math.max(0, duration - dur));
        updateCropSegment({ ...seg, startTime: newStart, endTime: newStart + dur });
      }
    };

    const onUp = () => {
      // Stop the drag auto-scroll loop.
      _autoScrollVxRef.current = 0;
      if (_autoScrollRafRef.current) {
        cancelAnimationFrame(_autoScrollRafRef.current);
        _autoScrollRafRef.current = 0;
      }
      _lastMoveEventRef.current = null;
      // Resume undo history so the final drag state is recorded as one snapshot
      if (dragInfo.type === 'move' || dragInfo.type === 'trim' ||
          dragInfo.type === 'trim-crop' || dragInfo.type === 'move-crop') {
        useTimelineStore.temporal.getState().resume();
      }
      // Make sure the FINAL scrub position lands on the video before we
      // tear down — otherwise releasing mid-frame can leave the
      // playhead one rAF behind the cursor.
      if (_scrubRafRef.current !== 0) {
        cancelAnimationFrame(_scrubRafRef.current);
        _scrubRafRef.current = 0;
      }
      if (_scrubPendingRef.current != null) {
        const finalT = _scrubPendingRef.current;
        _scrubPendingRef.current = null;
        try { setPlayheadRef.current?.(finalT); } catch { /* noop */ }
        try { onSeekRef.current?.(finalT); } catch { /* noop */ }
      }
      // Clear the scrub flag so VideoEditor's ``syncTime`` /
      // ``seeked`` handlers resume driving the playhead from
      // ``video.currentTime``. Doing this AFTER the final flush
      // guarantees the video has the freshest timestamp before the
      // timeupdate feedback loop wakes back up.
      if (dragInfo.type === 'scrub') {
        useTimelineStore.getState().setIsScrubbing(false);
      }
      useTimelineStore.getState().setSnapLine(null);
      setIsDragging(false);
      setDragInfo(null);
    };

    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      // Cancel any outstanding scrub flush so an unmount mid-drag
      // doesn't fire ``setPlayhead`` after the component is gone.
      // NOTE: this intentionally does NOT clear ``_scrubPendingRef`` —
      // if the effect is re-mounting (dep churn mid-drag, not a real
      // unmount), the freshly queued buffered time must survive. The
      // new effect body either schedules a new rAF on the next pointer
      // move or the ``onUp`` handler flushes it on release.
      if (_scrubRafRef.current !== 0) {
        cancelAnimationFrame(_scrubRafRef.current);
        _scrubRafRef.current = 0;
      }
      // Stop any in-flight auto-scroll rAF loop on unmount.
      _autoScrollVxRef.current = 0;
      if (_autoScrollRafRef.current) {
        cancelAnimationFrame(_autoScrollRafRef.current);
        _autoScrollRafRef.current = 0;
      }
      _lastMoveEventRef.current = null;
      // Safety: if component unmounts during a drag of any kind, resume
      // undo history. The previous version only handled ``move`` /
      // ``trim`` and would leave the temporal store paused after a
      // crop-segment drag mid-unmount, breaking undo for the rest of
      // the session.
      if (
        dragInfo.type === 'move' ||
        dragInfo.type === 'trim' ||
        dragInfo.type === 'trim-crop' ||
        dragInfo.type === 'move-crop'
      ) {
        useTimelineStore.temporal.getState().resume();
      }
      // Safety: clear the scrub flag so the video's timeupdate loop
      // stays wired up if this unmount was a teardown rather than a
      // dep-driven re-mount mid-drag.
      if (dragInfo.type === 'scrub') {
        useTimelineStore.getState().setIsScrubbing(false);
      }
    };
  // ``playhead`` / ``onSeek`` / ``setPlayhead`` are intentionally
  // absent from this dep list. They're read through refs
  // (``playheadRef``, ``onSeekRef``, ``setPlayheadRef``) so a parent
  // re-render with a new ``onSeek`` identity doesn't tear down the
  // drag listeners mid-gesture. Listing them here was the root cause
  // of the "playhead stays stationary" bug.
  //
  // ``sceneCuts`` is consumed by ``findSnapTarget`` inside ``onMove`` —
  // omitting it here meant new scene cuts (arriving asynchronously
  // while the user kept the drag handle held) wouldn't snap.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDragging, dragInfo, items, pps, scrollX, snapEnabled, sceneCuts, getTimeFromX, getTrackFromY, updateItem, setScrollX]);

  // ── Hover ──────────────────────────────────────────────────────────────────
  const onPointerMove = useCallback((e) => {
    if (isDragging) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x < LABEL_WIDTH) { setHoverTime(null); canvas.style.cursor = 'default'; return; }

    setHoverTime(getTimeFromX(e.clientX));

    if (spaceHeld) {
      canvas.style.cursor = 'grab';
      return;
    }

    if (activeTool === 'razor') {
      canvas.style.cursor = 'crosshair';
      return;
    }

    // Check if hovering near the playhead handle in the ruler area for grab cursor
    const phPixelX = rect.left + LABEL_WIDTH + playhead * pps - scrollX;
    const mouseY = e.clientY - rect.top;
    const isInRulerArea = mouseY <= RULER_HEIGHT + 8;
    if (isInRulerArea && Math.abs(e.clientX - phPixelX) <= PLAYHEAD_GRAB_WIDTH) {
      canvas.style.cursor = 'col-resize';
      return;
    }

    // Update imperative hover state — the draw fn uses these refs to
    // paint a soft outline and resize handles even before the user
    // commits to a click. Trigger a redraw only when the hovered id
    // actually changes so we don't burn CPU on every mouse pixel.
    const hit = hitTestItem(e.clientX, e.clientY);
    const cropHit = hitTestCropSegment(e.clientX, e.clientY);
    const prevHover = hoverRef.current;
    const newItemId = hit ? hit.item.id : null;
    const newCropId = cropHit ? cropHit.seg.id : null;
    const newEdge = (hit?.edge) || (cropHit?.edge) || null;
    if (prevHover.itemId !== newItemId
        || prevHover.cropId !== newCropId
        || prevHover.edge !== newEdge) {
      hoverRef.current = { itemId: newItemId, cropId: newCropId, edge: newEdge };
      requestRedrawRef.current?.();
    }

    if (hit) {
      canvas.style.cursor = hit.edge === 'left' || hit.edge === 'right' ? 'col-resize' : 'grab';
      return;
    }
    if (cropHit) {
      canvas.style.cursor = cropHit.edge === 'left' || cropHit.edge === 'right' ? 'col-resize' : 'grab';
      return;
    }
    if (!isInRulerArea && Math.abs(e.clientX - phPixelX) <= Math.max(PLAYHEAD_GRAB_WIDTH / 2, 8)) {
      // Show col-resize cursor on the playhead line outside ruler when no item is under cursor
      canvas.style.cursor = 'col-resize';
    } else {
      canvas.style.cursor = 'pointer';
    }
  }, [isDragging, getTimeFromX, hitTestItem, hitTestCropSegment, activeTool, spaceHeld, playhead, pps, scrollX]);

  const onPointerLeave = useCallback(() => {
    setHoverTime(null);
    if (hoverRef.current.itemId || hoverRef.current.cropId || hoverRef.current.edge) {
      hoverRef.current = { itemId: null, cropId: null, edge: null };
      requestRedrawRef.current?.();
    }
  }, []);

  // ── Zoom via Ctrl+Wheel ────────────────────────────────────────────────────
  // Ctrl/⌘ + wheel (and trackpad pinch, which browsers deliver as
  // ctrlKey wheel events) zooms CENTERED ON THE CURSOR: the timeline
  // time under the pointer stays put while the scale changes around it.
  // Multiplicative steps keep pinch (many tiny deltas) and wheel
  // (coarse deltas) feeling identical. Attached as a NATIVE non-passive
  // listener — React root wheel listeners are passive, so
  // e.preventDefault() there can't stop browser page-zoom.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const handler = (e) => {
      const { zoom: curZoom, scrollX: curScrollX } = useTimelineStore.getState();
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const cursorPx = Math.max(0, e.clientX - rect.left - LABEL_WIDTH);
        const anchorTime = (curScrollX + cursorPx) / (basePPS * curZoom);
        const factor = Math.exp(-e.deltaY * 0.01);
        const newZoom = Math.max(0.01, Math.min(10, curZoom * factor));
        setZoom(newZoom);
        setScrollX(Math.max(0, anchorTime * basePPS * newZoom - cursorPx));
      } else {
        setScrollX(Math.max(0, curScrollX + e.deltaX + (e.shiftKey ? e.deltaY : 0)));
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, [basePPS, setZoom, setScrollX]);

  // ── Double-click a clip → zoom to fill the view with it (2.7) ──
  const onCanvasDoubleClick = useCallback((e) => {
    const hit = hitTestItem(e.clientX, e.clientY);
    if (!hit?.item) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { start, end } = hit.item;
    const span = Math.max(0.1, end - start);
    const availableWidth = canvas.getBoundingClientRect().width - LABEL_WIDTH;
    // 10% breathing room either side
    const fitZoom = Math.max(0.01, Math.min(10, availableWidth / (span * 1.2 * basePPS)));
    setZoom(fitZoom);
    setScrollX(Math.max(0, (start - span * 0.1) * basePPS * fitZoom));
  }, [hitTestItem, basePPS, setZoom, setScrollX]);

  // ── Drop from media library ────────────────────────────────────────────────
  const onDrop = useCallback((e) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('application/x-clipai-media');
    if (!data) return;
    try {
      const media = JSON.parse(data);
      const time = getTimeFromX(e.clientX);
      const dropTrack = getTrackFromY(e.clientY);
      if (!dropTrack) return;
      if (dropTrack.locked) return; // Cannot drop items onto locked tracks

      // addItem auto-routes to the correct compatible track if the target is incompatible
      addItem({
        trackId: dropTrack.id,
        type: media.type,
        mediaRef: media.id,
        start: time,
        end: time + (media.duration || 5),
      });
    } catch { /* invalid data */ }
  }, [getTimeFromX, getTrackFromY, addItem]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  }, []);

  // ── Context menu ───────────────────────────────────────────────────────────
  const onContextMenu = useCallback((e) => e.preventDefault(), []);

  // Build the shared ContextMenu item list for the current target.
  // Fresh store reads keep the entries in sync with live state.
  const contextMenuItems = useMemo(() => {
    if (!contextMenu) return [];
    const store = useTimelineStore.getState();

    if (contextMenu.kind === 'track') {
      const track = store.tracks.find((t) => t.id === contextMenu.trackId);
      if (!track) return [];
      const audioMuted = track.audioMuted !== undefined ? !!track.audioMuted : !!track.muted;
      const hidden = track.videoVisible !== undefined ? !track.videoVisible : track.visible === false;
      const audible = (t) => t.type === 'audio' || t.type === 'video';
      const soloed = audible(track) && !audioMuted
        && store.tracks.filter((t) => audible(t) && t.id !== track.id)
          .every((t) => (t.audioMuted !== undefined ? !!t.audioMuted : !!t.muted));
      const tIdx = store.tracks.findIndex((t) => t.id === track.id);
      return [
        { heading: track.name || track.id },
        {
          id: 'rename', label: 'Rename…', onSelect: () => {
            const name = window.prompt('Track name', track.name || '');
            if (name != null && name.trim()) store.updateTrack(track.id, { name: name.trim() });
          },
        },
        // Reorder from the menu so touch users (no HTML5 drag) can restack
        // tracks; changing order also changes preview compositing (higher =
        // on top) for same-type tracks. Subtitles stay on top by type priority.
        { id: 'move-up', label: 'Move up', disabled: tIdx <= 0, onSelect: () => reorderTracks(tIdx, tIdx - 1) },
        { id: 'move-down', label: 'Move down', disabled: tIdx < 0 || tIdx >= store.tracks.length - 1, onSelect: () => reorderTracks(tIdx, tIdx + 1) },
        { separator: true },
        { id: 'mute', label: 'Mute', checked: audioMuted, disabled: !audible(track), onSelect: () => store.toggleTrackMute(track.id) },
        {
          id: 'solo', label: 'Solo', checked: soloed, disabled: !audible(track), onSelect: () => {
            // Solo = this track audible, every other A/V track muted.
            // Toggling solo off restores everything audible.
            for (const t of store.tracks) {
              if (!audible(t)) continue;
              const isMuted = t.audioMuted !== undefined ? !!t.audioMuted : !!t.muted;
              const shouldMute = soloed ? false : t.id !== track.id;
              if (isMuted !== shouldMute) store.toggleTrackMute(t.id);
            }
          },
        },
        { id: 'hide', label: hidden ? 'Show' : 'Hide', onSelect: () => store.toggleTrackVisibility(track.id) },
        { separator: true },
        { id: 'delete-track', label: 'Delete track', danger: true, disabled: track.locked, onSelect: () => store.removeTrack(track.id) },
      ];
    }

    const { item, time } = contextMenu;
    if (!item) return [];
    const itemTrack = store.tracks.find((t) => t.id === item.trackId);
    const locked = !!itemTrack?.locked;
    const isMedia = item.type === 'video' || item.type === 'audio';
    const canTransition = item.type === 'video' || item.type === 'image';
    const entries = [
      { id: 'split-playhead', label: 'Split at playhead', kbd: 'S', disabled: locked || store.playhead <= item.start || store.playhead >= item.end, onSelect: () => splitItem(item.id, store.playhead) },
      { id: 'split-here', label: 'Split here', disabled: locked || time <= item.start || time >= item.end, onSelect: () => splitItem(item.id, time) },
      { id: 'duplicate', label: 'Duplicate', kbd: '⌘D', disabled: locked, onSelect: () => store.duplicateItem(item.id) },
      { separator: true },
      {
        id: 'ripple-delete', label: 'Ripple delete', disabled: locked, danger: true, onSelect: () => {
          const dur = item.end - item.start;
          removeItem(item.id);
          useTimelineStore.getState().rippleShiftAfter(item.end - 0.001, -dur, [item.id]);
        },
      },
      { id: 'delete', label: 'Delete', kbd: '⌫', disabled: locked, danger: true, onSelect: () => removeItem(item.id) },
    ];
    if (isMedia) {
      entries.push({ separator: true });
      entries.push({ id: 'mute-item', label: 'Mute clip', checked: !!item.muted, disabled: locked, onSelect: () => updateItem(item.id, { muted: !item.muted }) });
      entries.push({ heading: 'Speed' });
      for (const s of [0.5, 1.0, 1.5, 2.0]) {
        entries.push({
          id: `speed-${s}`, label: `${s}×`, checked: (item.speed ?? 1) === s, disabled: locked,
          onSelect: () => updateItem(item.id, { speed: s }),
        });
      }
    }
    if (canTransition) {
      entries.push({ separator: true });
      entries.push({
        id: 'transition',
        label: item.transition ? 'Remove transition' : 'Add transition (dissolve)',
        disabled: locked,
        onSelect: () => updateItem(item.id, {
          transition: item.transition ? null : { type: 'dissolve', duration: 0.5 },
        }),
      });
    }
    if (selectedItemIds.length >= 2) {
      entries.push({ separator: true });
      entries.push({ id: 'group', label: 'Group selected', kbd: '⌘G', onSelect: () => groupItems(useTimelineStore.getState().selectedItemIds) });
    }
    if (item.groupId) {
      entries.push({ id: 'ungroup', label: 'Ungroup', onSelect: () => ungroupItems(useTimelineStore.getState().selectedItemIds) });
    }
    return entries;
  }, [contextMenu, selectedItemIds, splitItem, removeItem, updateItem, groupItems, ungroupItems]);

  // ── Touch gestures (3.1): pinch = zoom, two-finger pan = scroll,
  //    long-press = context menu action sheet ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const touches = new Map();
    let gesture = null;
    let longPressTimer = 0;
    let pressOrigin = null;

    const clearLongPress = () => { if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = 0; } };

    const onDown = (e) => {
      if (e.pointerType !== 'touch') return;
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (touches.size === 1) {
        const { clientX, clientY } = e;
        pressOrigin = { x: clientX, y: clientY };
        longPressTimer = setTimeout(() => {
          const hit = hitTestItem(clientX, clientY);
          if (hit?.item) {
            // Abort any drag the regular pointerdown started
            setIsDragging(false);
            setDragInfo(null);
            setSelectedItemId(hit.item.id);
            setContextMenu({
              kind: 'item', sheet: true,
              x: clientX, y: clientY,
              time: getTimeFromX(clientX),
              item: hit.item,
            });
          }
        }, 550);
      } else {
        clearLongPress();
      }
      if (touches.size === 2) {
        // Two fingers → cancel single-finger drag, start pinch/pan
        setIsDragging(false);
        setDragInfo(null);
        const pts = [...touches.values()];
        const st = useTimelineStore.getState();
        gesture = {
          startDist: Math.max(8, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)),
          startZoom: st.zoom,
          startScrollX: st.scrollX,
          startMidX: (pts[0].x + pts[1].x) / 2,
        };
      }
    };

    const onMove = (e) => {
      if (e.pointerType !== 'touch' || !touches.has(e.pointerId)) return;
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pressOrigin && Math.hypot(e.clientX - pressOrigin.x, e.clientY - pressOrigin.y) > 10) {
        clearLongPress();
      }
      if (touches.size === 2 && gesture) {
        e.preventDefault();
        const pts = [...touches.values()];
        const dist = Math.max(8, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y));
        const midX = (pts[0].x + pts[1].x) / 2;
        const newZoom = Math.max(0.01, Math.min(10, gesture.startZoom * (dist / gesture.startDist)));
        // Anchor the zoom on the gesture midpoint; horizontal midpoint
        // travel doubles as a two-finger pan.
        const rect = canvas.getBoundingClientRect();
        const cursorPx = Math.max(0, gesture.startMidX - rect.left - LABEL_WIDTH);
        const anchorTime = (gesture.startScrollX + cursorPx) / (basePPS * gesture.startZoom);
        const panDx = midX - gesture.startMidX;
        setZoom(newZoom);
        setScrollX(Math.max(0, anchorTime * basePPS * newZoom - cursorPx - panDx));
      }
    };

    const onUp = (e) => {
      if (e.pointerType !== 'touch') return;
      touches.delete(e.pointerId);
      clearLongPress();
      pressOrigin = null;
      if (touches.size < 2) gesture = null;
    };

    canvas.addEventListener('pointerdown', onDown);
    canvas.addEventListener('pointermove', onMove, { passive: false });
    canvas.addEventListener('pointerup', onUp);
    canvas.addEventListener('pointercancel', onUp);
    return () => {
      clearLongPress();
      canvas.removeEventListener('pointerdown', onDown);
      canvas.removeEventListener('pointermove', onMove);
      canvas.removeEventListener('pointerup', onUp);
      canvas.removeEventListener('pointercancel', onUp);
    };
  }, [basePPS, hitTestItem, getTimeFromX, setZoom, setScrollX, setSelectedItemId]);

  // ── Zoom to fit — toolbar Fit button, ⇧Z (action registry event) ──
  const zoomToFit = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const { items: curItems, duration: curDuration } = useTimelineStore.getState();
    const maxEnd = curItems.length > 0
      ? Math.max(...curItems.map(it => it.end || 0))
      : curDuration || 30;
    const fitDuration = maxEnd * 1.05 || 30;
    const availableWidth = canvas.getBoundingClientRect().width - LABEL_WIDTH;
    const fitZoom = Math.max(0.01, availableWidth / (fitDuration * basePPS));
    setZoom(fitZoom);
    setScrollX(0);
  }, [basePPS, setZoom, setScrollX]);

  useEffect(() => {
    window.addEventListener('ve:zoom-fit', zoomToFit);
    return () => window.removeEventListener('ve:zoom-fit', zoomToFit);
  }, [zoomToFit]);

  // ── Compute canvas height ──────────────────────────────────────────────────
  const canvasHeight = laneTop(laneHs, tracks.length) + 12;

  return (
    <div ref={containerRef} className="ve-multi-timeline" style={{ position: 'relative', height: '100%' }}>
      {/* Toolbar row */}
      <div className="ve-multi-timeline__toolbar">
        <Tooltip label="Zoom out">
          <button
            className="ve-btn"
            onClick={() => setZoom(Math.max(0.01, zoom - 0.2))}
            style={{ fontSize: 12, padding: '2px 6px', minWidth: 24, minHeight: 24 }}
          >
            -
          </button>
        </Tooltip>
        <input
          type="range"
          min="0.01"
          max="10"
          step="0.01"
          value={zoom}
          onChange={(e) => setZoom(parseFloat(e.target.value))}
          className="ve-multi-timeline__zoom-slider"
        />
        <Tooltip label="Zoom in">
          <button
            className="ve-btn"
            onClick={() => setZoom(Math.min(10, zoom + 0.2))}
            style={{ fontSize: 12, padding: '2px 6px', minWidth: 24, minHeight: 24 }}
          >
            +
          </button>
        </Tooltip>
        <Tooltip actionId="zoom-to-fit">
          <button
            className="ve-btn"
            onClick={zoomToFit}
            style={{ fontSize: 10, padding: '2px 8px', minWidth: 'auto', minHeight: 24, fontWeight: 600 }}
          >
            Fit
          </button>
        </Tooltip>
        <Tooltip label={`Snapping ${snapEnabled ? 'on' : 'off'}`} kbd="N">
          <button
            className={`ve-btn${snapEnabled ? ' ve-btn--active-snap' : ''}`}
            onClick={toggleSnap}
            style={{ fontSize: 10, padding: '2px 6px', minWidth: 'auto', minHeight: 24 }}
          >
            Snap {snapEnabled ? 'ON' : 'OFF'}
          </button>
        </Tooltip>
        <Tooltip label={`Ripple edit ${rippleEnabled ? 'on' : 'off'} — trims move downstream clips`} kbd="\\">
          <button
            className={`ve-btn${rippleEnabled ? ' ve-btn--active-ripple' : ''}`}
            onClick={toggleRipple}
            style={{ fontSize: 10, padding: '2px 6px', minWidth: 'auto', minHeight: 24 }}
          >
            Ripple {rippleEnabled ? 'ON' : 'OFF'}
          </button>
        </Tooltip>
        {/* Numeric timecode entry — click the display, type a time,
            press Enter to seek. Accepts H:MM:SS.mmm, M:SS.mmm, SS.mmm,
            and SSSS (raw seconds). */}
        <TimecodeInput
          playhead={playhead}
          onSeek={(t) => {
            setPlayhead(t);
            try { onSeek?.(t); } catch { /* noop */ }
          }}
        />
        <div style={{ flex: 1 }} />
        <div style={{ position: 'relative' }}>
          <button
            className="ve-btn"
            onClick={() => setShowAddTrack(!showAddTrack)}
            style={{ fontSize: 10, padding: '2px 8px', minHeight: 24 }}
          >
            + Track
          </button>
          {showAddTrack && (
            <div className="ve-multi-timeline__add-track-dropdown">
              {['video', 'audio', 'overlay', 'subtitle'].map(type => (
                <button
                  key={type}
                  onClick={() => { addTrack(type); setShowAddTrack(false); }}
                  className="ve-multi-timeline__add-track-option"
                >
                  {TRACK_ICONS[type]} {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Canvas area with track header overlay. Lanes shrink uniformly
          (fit-to-window, 28px floor — see laneHeightsFor) so every track
          fits stacked in view WITHOUT vertical scrolling. The overflow
          scroll below is only a backstop for extreme cases (dozens of
          tracks on a short window) where even floor-height lanes can't
          fit; the right-edge track rail then gives a Premiere-style
          overview + jump-to-track affordance. */}
      <div
        ref={wrapRef}
        className="ve-multi-timeline__canvas-wrap"
        style={{
          position: 'relative',
          // Fill the remaining panel height; lanes are sized (via fitHeight,
          // measured from this element) to fit exactly, so overflow scroll is
          // only a backstop for the extreme many-tracks-on-a-tiny-panel case.
          flex: '1 1 auto',
          minHeight: 0,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        {/* Track header controls — overlays the canvas label area */}
        {/* Styled like DaVinci Resolve / Premiere Pro: eye (visibility), mute, lock per track */}
        <div
          className="ve-multi-timeline__track-headers"
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            width: LABEL_WIDTH - 1,
            zIndex: 5,
            pointerEvents: 'none',
          }}
        >
          {/* Spacer for ruler */}
          <div style={{ height: RULER_HEIGHT }} />
          {tracks.map((track, trackIdx) => {
            const isHidden = track.visible === false;
            const isDragOver = dragOverTrackIdx === trackIdx && dragTrackIdx !== trackIdx;
            return (
              <div
                key={track.id}
                className="ve-multi-timeline__track-header"
                draggable
                onDragStart={(e) => {
                  setDragTrackIdx(trackIdx);
                  e.dataTransfer.effectAllowed = 'move';
                  e.dataTransfer.setData('text/plain', String(trackIdx));
                }}
                onDragOver={(e) => {
                  e.preventDefault();
                  e.dataTransfer.dropEffect = 'move';
                  setDragOverTrackIdx(trackIdx);
                }}
                onDragLeave={() => { if (dragOverTrackIdx === trackIdx) setDragOverTrackIdx(null); }}
                onDrop={(e) => {
                  e.preventDefault();
                  if (dragTrackIdx != null && dragTrackIdx !== trackIdx) {
                    reorderTracks(dragTrackIdx, trackIdx);
                  }
                  setDragTrackIdx(null);
                  setDragOverTrackIdx(null);
                }}
                onDragEnd={() => { setDragTrackIdx(null); setDragOverTrackIdx(null); }}
                onContextMenu={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setContextMenu({ kind: 'track', x: e.clientX, y: e.clientY, trackId: track.id });
                }}
                onPointerDown={(e) => {
                  if (e.pointerType !== 'touch') return;
                  const lp = trackLongPressRef.current;
                  lp.x = e.clientX; lp.y = e.clientY;
                  clearTimeout(lp.timer);
                  lp.timer = setTimeout(() => {
                    lp.timer = 0;
                    setContextMenu({ kind: 'track', sheet: true, x: lp.x, y: lp.y, trackId: track.id });
                  }, 500);
                }}
                onPointerMove={(e) => {
                  const lp = trackLongPressRef.current;
                  if (lp.timer && Math.hypot(e.clientX - lp.x, e.clientY - lp.y) > 10) {
                    clearTimeout(lp.timer); lp.timer = 0;
                  }
                }}
                onPointerUp={() => { const lp = trackLongPressRef.current; if (lp.timer) { clearTimeout(lp.timer); lp.timer = 0; } }}
                onPointerCancel={() => { const lp = trackLongPressRef.current; if (lp.timer) { clearTimeout(lp.timer); lp.timer = 0; } }}
                onClick={isMobileViewport ? () => {
                  // Tap a lane badge on mobile → expand that one lane
                  setExpandedTrackId((cur) => (cur === track.id ? null : track.id));
                } : undefined}
                style={{
                  height: laneHs[trackIdx] ?? TRACK_HEIGHT,
                  marginBottom: TRACK_GAP,
                  display: 'flex',
                  // Mobile: single row so the track NAME is visible inline next
                  // to the eye/mute/lock controls (the stacked column clipped
                  // the name in a 28px compact lane). Desktop keeps the column.
                  flexDirection: isMobileViewport ? 'row' : 'column',
                  alignItems: isMobileViewport ? 'center' : undefined,
                  justifyContent: 'center',
                  gap: isMobileViewport ? 4 : 2,
                  padding: '2px 4px',
                  pointerEvents: 'auto',
                  opacity: isHidden ? 0.5 : (dragTrackIdx === trackIdx ? 0.4 : 1),
                  cursor: 'grab',
                  borderTop: isDragOver ? '2px solid var(--accent, #6E7BFF)' : '2px solid transparent',
                  transition: 'opacity 0.15s, border-color 0.15s',
                }}
              >
                {/* Drag handle + track name */}
                <span style={{
                  fontSize: 10,
                  fontWeight: 500,
                  color: 'var(--ve-text, #ccc)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 3,
                  minWidth: 0,
                  flex: isMobileViewport ? 1 : undefined,
                  opacity: isHidden ? 0.5 : 0.8,
                }}>
                  {/* Drag-to-reorder handle — desktop only (touch uses the
                      long-press menu's Move up/down). */}
                  {!isMobileViewport && (
                    <svg width="8" height="10" viewBox="0 0 8 10" fill="currentColor" style={{ opacity: 0.35, flexShrink: 0 }}>
                      <circle cx="2" cy="2" r="1" /><circle cx="6" cy="2" r="1" />
                      <circle cx="2" cy="5" r="1" /><circle cx="6" cy="5" r="1" />
                      <circle cx="2" cy="8" r="1" /><circle cx="6" cy="8" r="1" />
                    </svg>
                  )}
                  {TRACK_ICONS[track.type] || ''}{' '}
                  {renamingTrackId === track.id ? (
                    <input
                      autoFocus
                      defaultValue={track.name}
                      onBlur={(e) => {
                        const val = e.target.value.trim();
                        if (val && val !== track.name) updateTrack(track.id, { name: val });
                        // Defer input removal so pointer events resolve their target
                        // before the DOM mutates (input → span swap)
                        requestAnimationFrame(() => setRenamingTrackId(null));
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') { e.target.blur(); }
                        else if (e.key === 'Escape') { setRenamingTrackId(null); }
                        e.stopPropagation();
                      }}
                      onClick={(e) => e.stopPropagation()}
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                      style={{
                        fontSize: 10, fontWeight: 500, width: '100%',
                        background: 'var(--ve-surface, #222)', color: 'var(--ve-text, #ccc)',
                        border: '1px solid var(--accent, #6E7BFF)', borderRadius: 2,
                        padding: '0 2px', outline: 'none', minWidth: 0,
                      }}
                    />
                  ) : (
                    <span
                      onDoubleClick={(e) => { e.stopPropagation(); setRenamingTrackId(track.id); }}
                      style={{ cursor: 'text', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}
                      aria-label="Double-click to rename"
                    >
                      {track.name}
                    </span>
                  )}
                  {track.type === 'subtitle' && isHidden && (
                    <span style={{
                      fontSize: 9,
                      color: 'var(--ve-text-muted, #999)',
                      marginLeft: 4,
                      opacity: 0.6,
                    }}>
                      (hidden)
                    </span>
                  )}
                </span>
                {/* Controls row */}
                <div style={{ display: 'flex', gap: 1, flexShrink: 0 }}>
                  {/* Visibility toggle (eye icon) — preview only */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleTrackVisibility(track.id);
                      // Sync subtitle track visibility to settings
                      if (track.type === 'subtitle' && onSubtitleVisibilityChange) {
                        onSubtitleVisibilityChange(!(track.visible !== false));
                      }
                    }}
                    aria-label={isHidden ? `Show ${track.name} in preview` : `Hide ${track.name} from preview (still in export)`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 12,
                      opacity: isHidden ? 0.4 : 0.7,
                      color: isHidden ? 'var(--ve-text-muted, #999)' : 'var(--ve-text, #666)',
                    }}
                  >
                    {isHidden ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94" />
                        <path d="M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19" />
                        <line x1="1" y1="1" x2="23" y2="23" />
                      </svg>
                    ) : (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                  {/* Mute toggle */}
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleTrackMute(track.id); }}
                    aria-label={track.muted ? `Unmute ${track.name}` : `Mute ${track.name}`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 9, fontWeight: 700,
                      opacity: track.muted ? 1 : 0.35,
                      color: track.muted ? 'var(--danger, #ef4444)' : 'var(--ve-text, #666)',
                    }}
                  >
                    M
                  </button>
                  {/* Lock toggle */}
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleTrackLock(track.id); }}
                    aria-label={track.locked ? `Unlock ${track.name}` : `Lock ${track.name}`}
                    className="ve-multi-timeline__track-ctrl"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      padding: '2px', lineHeight: 1, fontSize: 10,
                      opacity: track.locked ? 0.8 : 0.35,
                      color: 'var(--ve-text, #999)',
                    }}
                  >
                    {track.locked ? (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 0110 0v4" />
                      </svg>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                        <path d="M7 11V7a5 5 0 019.9-1" />
                      </svg>
                    )}
                  </button>
                  {/* Reset subtitle timings button — only on subtitle tracks */}
                  {track.type === 'subtitle' && hasOriginalSubtitles && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        resetSubtitleTimings();
                      }}
                      aria-label="Reset all subtitles to original timing from transcript"
                      className="ve-multi-timeline__track-ctrl"
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        padding: '2px', lineHeight: 1, fontSize: 10,
                        opacity: 0.7,
                        color: 'var(--ve-text, #666)',
                      }}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="1 4 1 10 7 10" />
                        <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          className="ve-multi-timeline__canvas"
          // Give a few-track canvas a comfortable minimum drop area, but never
          // taller than the measured wrap — otherwise the 280 floor would
          // reintroduce a scrollbar on a short panel.
          style={{ width: '100%', height: Math.max(canvasHeight, Math.min(280, fitHeight || 280)) }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onCanvasDoubleClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onContextMenu={onContextMenu}
        />

        {/* Track rail — colored swatches on the right edge for quick
            "jump to track" navigation. Only shows when the stack is
            tall enough to overflow the visible area (8+ tracks at
            current sizing); for typical 6-track jobs it's hidden. */}
        {tracks.length >= 8 && (
          <div
            className="ve-multi-timeline__rail"
            style={{ height: canvasHeight }}
            aria-label="Track rail — click a swatch to scroll its track into view"
          >
            {tracks.map((track, trackIdx) => {
              const trackY = laneTop(laneHs, trackIdx);
              const swatchColor = TRACK_COLORS[track.type] || TRACK_COLORS.video;
              return (
                <button
                  key={track.id}
                  className="ve-multi-timeline__rail-dot"
                  style={{
                    height: (laneHs[trackIdx] ?? TRACK_HEIGHT) - 6,
                    background: swatchColor,
                    marginTop: trackIdx === 0 ? RULER_HEIGHT + 3 : TRACK_GAP - 1,
                  }}
                  aria-label={`${track.name} — jump to track`}
                  onClick={(e) => {
                    e.stopPropagation();
                    const wrap = e.currentTarget.closest('.ve-multi-timeline__canvas-wrap');
                    if (wrap) {
                      wrap.scrollTo({
                        top: Math.max(0, trackY - RULER_HEIGHT - 8),
                        behavior: 'smooth',
                      });
                    }
                    setSelectedItemId(null);
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      {/* Overview ribbon — Premiere/Resolve-style mini-map.  Renders a
          condensed projection of every track plus the current viewport
          rectangle so the user can pan to any point on the timeline in
          one click, even when the working zoom only shows a few
          seconds at a time. */}
      <TimelineMinimap
        tracks={tracks}
        items={items}
        cropSegments={cropSegments}
        duration={duration}
        playhead={playhead}
        scrollX={scrollX}
        pps={pps}
        labelWidth={LABEL_WIDTH}
        canvasWidthRef={canvasRef}
        sceneCuts={sceneCuts}
        onScrollTo={(newScrollX) => setScrollX(Math.max(0, newScrollX))}
        onSeek={(t) => {
          setPlayhead(t);
          try { onSeek?.(t); } catch { /* noop */ }
        }}
      />

      {/* Context menu */}
      {contextMenu && contextMenuItems.length > 0 && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          items={contextMenuItems}
          sheet={!!contextMenu.sheet}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  );
}
