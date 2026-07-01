import React, { useRef, useEffect, useMemo } from 'react';
import { getCropXForTime } from '../utils/subjectTracking';
import useTimelineStore from '../stores/timelineStore';

// getCropXForTime now lives in utils/subjectTracking as the single source
// of truth shared by every preview surface (this canvas overlay, the
// VideoEditor / ClipPreview rAF loops, and ReframeStatsPanel) so they can
// never drift apart again.

// Binary search: index of the first key >= target (lower_bound).
function lowerBound(sortedKeys, target) {
  let lo = 0;
  let hi = sortedKeys.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedKeys[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// O(log n) nearest-key lookup within `maxDist`. Returns the matched
// key (number) or null when no key falls inside the tolerance window.
export function findNearestKey(sortedKeys, target, maxDist) {
  if (!sortedKeys.length) return null;
  const idx = lowerBound(sortedKeys, target);
  let bestKey = null;
  let bestDist = Infinity;
  for (const candidate of [sortedKeys[idx - 1], sortedKeys[idx]]) {
    if (candidate == null) continue;
    const d = Math.abs(candidate - target);
    if (d < bestDist) {
      bestDist = d;
      bestKey = candidate;
    }
  }
  return bestDist <= maxDist ? bestKey : null;
}

// Detect if the nearest sample is far enough that the box should be
// drawn as interpolated (dashed) instead of detected (solid).
const INTERPOLATED_DELTA_MS = 100;
const FACE_LOOKUP_TOL_MS = 250;
const SUBJECT_LOOKUP_TOL_MS = 250;
// How far past the last detection of a track we still advect its box using
// the emitted per-box velocity (optical-flow-style propagation, item 7).
const EXTRAPOLATE_MS = 350;

const _lerp = (a, b, u) => a + (b - a) * u;

// Numeric box fields we interpolate; everything else is carried from `from`.
const _BOX_NUM_FIELDS = ['x', 'y', 'w', 'h', 'cx', 'cy', 'confidence', 'mouth_motion'];

function _lerpBox(from, to, u) {
  const out = { ...from };
  for (const k of _BOX_NUM_FIELDS) {
    if (typeof from[k] === 'number' && typeof to[k] === 'number') {
      out[k] = _lerp(from[k], to[k], u);
    }
  }
  return out;
}

// Advect a single box forward from its sample time by dtMs using vx/vy
// (source px/sec). Used when a track has no bracketing "next" sample.
function _advectBox(box, dtMs) {
  const vx = Number(box.vx || 0);
  const vy = Number(box.vy || 0);
  if (!vx && !vy) return { ...box };
  const dt = dtMs / 1000;
  const dx = vx * dt;
  const dy = vy * dt;
  return {
    ...box,
    x: (box.x || 0) + dx,
    y: (box.y || 0) + dy,
    cx: (box.cx || 0) + dx,
    cy: (box.cy || 0) + dy,
  };
}

/**
 * Interpolate the boxes of a timeline at time `tMs`, matched by `track_id`.
 *
 * Instead of snapping to the nearest sample (which makes boxes teleport every
 * ~200ms and lag up to 250ms), we find the two bracketing samples and lerp
 * each track's box between them. Tracks present in only one bracket are
 * advected with their velocity (up to EXTRAPOLATE_MS) so they still move
 * smoothly. Boxes without a track_id fall back to nearest-sample selection.
 *
 * Returns `{ boxes, interpolated }` where `interpolated` is true when the
 * boxes are not sitting exactly on a detected sample (drawn dashed).
 */
function interpolateBoxes(timeline, sortedKeys, tMs, tolMs) {
  if (!timeline || !sortedKeys || !sortedKeys.length) {
    return { boxes: [], interpolated: false };
  }
  const idx = lowerBound(sortedKeys, tMs);
  const prevKey = sortedKeys[idx - 1];
  const nextKey = sortedKeys[idx];
  const prevBoxes = prevKey != null ? (timeline[String(prevKey)] || []) : [];
  const nextBoxes = nextKey != null ? (timeline[String(nextKey)] || []) : [];

  // Exactly on / adjacent to a sample: use it directly (solid).
  const nearestKey = findNearestKey(sortedKeys, tMs, tolMs);
  if (nearestKey == null) return { boxes: [], interpolated: false };
  const onSample = Math.abs(nearestKey - tMs) <= INTERPOLATED_DELTA_MS;

  const keyOf = (b) => (b && b.track_id != null && Number(b.track_id) >= 0
    ? Number(b.track_id) : null);
  const prevMap = new Map();
  const nextMap = new Map();
  const prevUntracked = [];
  const nextUntracked = [];
  for (const b of prevBoxes) { const k = keyOf(b); if (k == null) prevUntracked.push(b); else prevMap.set(k, b); }
  for (const b of nextBoxes) { const k = keyOf(b); if (k == null) nextUntracked.push(b); else nextMap.set(k, b); }

  const boxes = [];
  const haveBracket = prevKey != null && nextKey != null
    && prevKey <= tMs && tMs <= nextKey && nextKey > prevKey;
  const u = haveBracket ? (tMs - prevKey) / (nextKey - prevKey) : 0;

  const seen = new Set();
  for (const [k, pb] of prevMap.entries()) {
    seen.add(k);
    const nb = nextMap.get(k);
    if (nb && haveBracket) {
      boxes.push(_lerpBox(pb, nb, u));
    } else if (nextKey == null || (tMs - prevKey) <= EXTRAPOLATE_MS) {
      boxes.push(_advectBox(pb, Math.max(0, tMs - prevKey)));
    } else {
      boxes.push({ ...pb });
    }
  }
  // Tracks that only appear in the "next" bracket (just entered frame).
  for (const [k, nb] of nextMap.entries()) {
    if (seen.has(k)) continue;
    if (prevKey == null || (nextKey - tMs) <= EXTRAPOLATE_MS) {
      boxes.push(_advectBox(nb, -Math.max(0, nextKey - tMs)));
    } else {
      boxes.push({ ...nb });
    }
  }
  // Untracked boxes: fall back to whichever bracket sample is nearer.
  if (prevUntracked.length || nextUntracked.length) {
    const useNext = nextKey != null
      && (prevKey == null || Math.abs(nextKey - tMs) < Math.abs(prevKey - tMs));
    for (const b of (useNext ? nextUntracked : prevUntracked)) boxes.push({ ...b });
  }

  return { boxes, interpolated: !onSample };
}

/**
 * ReframePreview — canvas overlay drawn on top of the source <video>.
 *
 * The component does NOT render its own <video>; it sits inside the
 * existing .ve-stage and paints:
 *   - a dimming mask over the un-cropped region
 *   - an orange crop-window rectangle with centre tick marks
 *   - green face boxes (solid when detected, dashed when interpolated)
 *   - cyan YOLO subject boxes with class labels
 *   - a thicker pulsing green border on the inferred active speaker
 *
 * Detection data comes from /api/jobs/{id}/detection_overlay. The
 * timeline lookup uses binary search so each frame stays O(log n).
 */
export default function ReframePreview({
  videoRef,
  detectionData,
  subjectKeyframes,
  sourceWidth,
  sourceHeight,
  targetRatio,
  clipStart = 0,
}) {
  const canvasRef = useRef(null);

  // Pre-sort timeline keys once per detectionData change.
  const sortedKeys = useMemo(() => {
    const sortKeys = (obj) =>
      obj ? Object.keys(obj).map(Number).sort((a, b) => a - b) : [];
    return {
      face: sortKeys(detectionData?.face_timeline),
      person: sortKeys(detectionData?.person_timeline),
    };
  }, [detectionData]);

  const cropDims = useMemo(() => {
    if (!targetRatio || !sourceWidth || !sourceHeight) return null;
    let cropW = Math.round(sourceHeight * targetRatio);
    let cropH = sourceHeight;
    if (cropW > sourceWidth) {
      cropW = sourceWidth;
      cropH = Math.round(sourceWidth / targetRatio);
    }
    return { cropW, cropH, maxX: sourceWidth - cropW };
  }, [targetRatio, sourceWidth, sourceHeight]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef?.current;
    if (!canvas || !video || !cropDims) return;

    const ctx = canvas.getContext('2d');
    const srcRatio = sourceWidth / sourceHeight;
    let raf;
    let pulsePhase = 0;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      const targetW = Math.max(1, Math.round(cssW * dpr));
      const targetH = Math.max(1, Math.round(cssH * dpr));
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      // Compute the rect within the canvas where the video is actually
      // drawn (object-fit: contain).
      const canvasRatio = cssW / cssH;
      let displayW;
      let displayH;
      let offsetX;
      let offsetY;
      if (canvasRatio > srcRatio) {
        displayH = cssH;
        displayW = displayH * srcRatio;
        offsetX = (cssW - displayW) / 2;
        offsetY = 0;
      } else {
        displayW = cssW;
        displayH = displayW / srcRatio;
        offsetX = 0;
        offsetY = (cssH - displayH) / 2;
      }
      const scaleX = displayW / sourceWidth;
      const scaleY = displayH / sourceHeight;

      // Current crop X — read the live cropSegments from the store so
      // PropertiesPanel slider edits, drag-resizes on the timeline, and
      // split / merge actions all update the preview rectangle. Falls
      // back to interpolated subjectKeyframes only when no segments
      // exist (e.g. immediately after job analysis completes, before
      // the cropSegments effect has run).
      const relTime = Math.max(0, video.currentTime - clipStart);
      const { cropSegments } = useTimelineStore.getState();
      const sxPct = getCropXForTime(relTime, cropSegments, subjectKeyframes);
      const cropCenterPx = (sxPct / 100) * sourceWidth;
      const cropX0Src = Math.max(0, Math.min(
        cropDims.maxX,
        Math.round(cropCenterPx - cropDims.cropW / 2),
      ));
      const dx = offsetX + cropX0Src * scaleX;
      const dw = cropDims.cropW * scaleX;

      // 1) Dim outside the crop window.
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      if (dx > offsetX) ctx.fillRect(offsetX, offsetY, dx - offsetX, displayH);
      const rightEdge = dx + dw;
      const viewRight = offsetX + displayW;
      if (rightEdge < viewRight) {
        ctx.fillRect(rightEdge, offsetY, viewRight - rightEdge, displayH);
      }

      // 2) Crop rectangle (orange) + centre tick marks.
      ctx.strokeStyle = '#D85A30';
      ctx.lineWidth = 2;
      ctx.strokeRect(dx + 0.5, offsetY + 0.5, dw - 1, displayH - 1);
      const cx = dx + dw / 2;
      ctx.beginPath();
      ctx.moveTo(cx, offsetY);
      ctx.lineTo(cx, offsetY + 8);
      ctx.moveTo(cx, offsetY + displayH - 8);
      ctx.lineTo(cx, offsetY + displayH);
      ctx.stroke();

      // 3) Face boxes — interpolated per track_id between bracketing samples
      //    so the box tracks smoothly instead of teleporting each sample.
      const tMs = Math.round(video.currentTime * 1000);
      const faceResult = interpolateBoxes(
        detectionData?.face_timeline, sortedKeys.face, tMs, FACE_LOOKUP_TOL_MS);
      const faces = faceResult.boxes;
      const faceInterpolated = faceResult.interpolated;

      // Active speaker = highest mouth_motion above a small floor.
      let speakerTrackId = null;
      let bestMouth = 0;
      for (const f of faces) {
        const m = Number(f.mouth_motion || 0);
        if (m > bestMouth && m >= 0.12) {
          bestMouth = m;
          speakerTrackId = f.track_id;
        }
      }

      pulsePhase = (pulsePhase + 0.08) % (Math.PI * 2);
      const pulseAlpha = 0.55 + 0.35 * Math.sin(pulsePhase);

      ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      for (const f of faces) {
        const fx = offsetX + f.x * scaleX;
        const fy = offsetY + f.y * scaleY;
        const fw = f.w * scaleX;
        const fh = f.h * scaleY;
        const isSpeaker = speakerTrackId != null
          && f.track_id === speakerTrackId
          && Number(f.track_id) >= 0;

        if (faceInterpolated) ctx.setLineDash([4, 3]);
        if (isSpeaker) {
          ctx.strokeStyle = `rgba(34, 197, 94, ${pulseAlpha.toFixed(2)})`;
          ctx.lineWidth = 3;
        } else {
          ctx.strokeStyle = faceInterpolated ? '#3C8264' : '#1D9E75';
          ctx.lineWidth = 2;
        }
        ctx.strokeRect(fx, fy, fw, fh);
        ctx.setLineDash([]);

        const confPct = Math.round(Number(f.confidence || 0) * 100);
        const tid = Number(f.track_id ?? -1);
        const tidLabel = tid >= 0 ? `T${tid}` : 'T?';
        const speakerGlyph = isSpeaker ? ' \u{1F3A4}' : '';
        const label = `${tidLabel} ${confPct}%${speakerGlyph}`;
        const lw = ctx.measureText(label).width;
        const labelY = fy - 13 >= offsetY ? fy - 13 : fy + 1;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.fillRect(fx, labelY, lw + 6, 13);
        ctx.fillStyle = isSpeaker
          ? '#86efac'
          : (faceInterpolated ? '#7aaa96' : '#4ade80');
        ctx.fillText(label, fx + 3, labelY + 10);
      }

      // 4) Subject (YOLO) boxes — cyan, with class labels. Interpolated per
      //    track_id just like faces.
      const personResult = interpolateBoxes(
        detectionData?.person_timeline, sortedKeys.person, tMs, SUBJECT_LOOKUP_TOL_MS);
      const persons = personResult.boxes;
      for (const p of persons) {
        const sx = offsetX + p.x * scaleX;
        const sy = offsetY + p.y * scaleY;
        const sw = p.w * scaleX;
        const sh = p.h * scaleY;
        ctx.strokeStyle = personResult.interpolated ? '#3399B8' : '#00C8FF';
        ctx.lineWidth = 1.5;
        if (personResult.interpolated) ctx.setLineDash([4, 3]);
        ctx.strokeRect(sx, sy, sw, sh);
        ctx.setLineDash([]);

        const cls = String(p.class_name || 'person');
        const lw = ctx.measureText(cls).width;
        const labelY = sy - 13 >= offsetY ? sy - 13 : sy + 1;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.fillRect(sx, labelY, lw + 6, 13);
        ctx.fillStyle = '#67e8f9';
        ctx.fillText(cls, sx + 3, labelY + 10);
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [
    videoRef, cropDims, sourceWidth, sourceHeight,
    clipStart, subjectKeyframes, detectionData, sortedKeys,
  ]);

  return <canvas ref={canvasRef} className="reframe-preview__canvas" />;
}
