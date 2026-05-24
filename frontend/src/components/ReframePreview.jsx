import React, { useRef, useEffect, useMemo } from 'react';
import { interpolateSubjectX } from '../utils/subjectTracking';
import useTimelineStore from '../stores/timelineStore';

// Look up the active crop X (0–100%) at a given clip-relative time.
// Mirrors the legacy ClipPreview / VideoEditor rAF helper so the
// canvas overlay tracks the same user edits the export uses.
export function getCropXForTime(relTime, cropSegments, subjectKeyframes) {
  if (Array.isArray(cropSegments) && cropSegments.length > 0) {
    const seg = cropSegments.find((s) => relTime >= s.startTime && relTime < s.endTime);
    if (seg && Number.isFinite(seg.cropX)) return seg.cropX;
    const last = cropSegments[cropSegments.length - 1];
    if (last && relTime >= last.endTime && Number.isFinite(last.cropX)) return last.cropX;
  }
  try {
    const v = interpolateSubjectX(subjectKeyframes, relTime);
    if (typeof v === 'number' && !Number.isNaN(v)) return v;
  } catch (_) { /* fall through */ }
  return 50;
}

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

      // 3) Face boxes via binary-search lookup.
      const tMs = Math.round(video.currentTime * 1000);
      const faceKey = findNearestKey(sortedKeys.face, tMs, FACE_LOOKUP_TOL_MS);
      const faces = faceKey != null
        ? (detectionData?.face_timeline?.[String(faceKey)] || [])
        : [];
      const faceInterpolated = faceKey != null
        && Math.abs(faceKey - tMs) > INTERPOLATED_DELTA_MS;

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

      // 4) Subject (YOLO) boxes — cyan, with class labels.
      const personKey = findNearestKey(sortedKeys.person, tMs, SUBJECT_LOOKUP_TOL_MS);
      const persons = personKey != null
        ? (detectionData?.person_timeline?.[String(personKey)] || [])
        : [];
      for (const p of persons) {
        const sx = offsetX + p.x * scaleX;
        const sy = offsetY + p.y * scaleY;
        const sw = p.w * scaleX;
        const sh = p.h * scaleY;
        ctx.strokeStyle = '#00C8FF';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(sx, sy, sw, sh);

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
