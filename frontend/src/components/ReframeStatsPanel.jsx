import React, { useRef, useEffect, useMemo } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { findNearestKey, getCropXForTime } from './ReframePreview';

const STATS_UPDATE_HZ = 12;        // ~12 stat refreshes per second
const SCENE_CUT_WARN_MS = 500;     // flag scene cuts within 500ms
const FACE_LOOKUP_TOL_MS = 250;
const SUBJECT_LOOKUP_TOL_MS = 250;
const MOTION_LOOKUP_TOL_MS = 250;
const SPEECH_LOOKUP_TOL_MS = 250;

function classBreakdown(persons) {
  if (!persons?.length) return '';
  const counts = {};
  for (const p of persons) {
    const k = String(p.class_name || 'person');
    counts[k] = (counts[k] || 0) + 1;
  }
  return Object.entries(counts)
    .map(([k, n]) => (n > 1 ? `${k}×${n}` : k))
    .join(', ');
}

function findActiveOp(ops, tSec) {
  if (!Array.isArray(ops) || !ops.length) return null;
  // Linear scan is fine — typical clip has <30 ops.
  for (const op of ops) {
    if (tSec >= op.start_sec && tSec < op.end_sec) return op;
  }
  return ops[ops.length - 1];
}

function StatRow({ label, valueRef, color, mono = true }) {
  return (
    <div className="reframe-stats__row">
      <span className="reframe-stats__label">{label}</span>
      <span
        ref={valueRef}
        className={`reframe-stats__value${mono ? ' reframe-stats__value--mono' : ''}`}
        style={color ? { color } : undefined}
      >
        —
      </span>
    </div>
  );
}

/**
 * ReframeStatsPanel — live per-frame inspector next to the source view.
 *
 * Updates are throttled to ~12 Hz and applied imperatively via refs so
 * the React tree doesn't re-render on every animation frame.
 */
export default function ReframeStatsPanel({
  videoRef,
  detectionData,
  subjectKeyframes,
  renderPlan,
  sourceWidth,
  sourceHeight,
  targetRatio,
  clipStart = 0,
}) {
  const refs = {
    strategy: useRef(null),
    cropX: useRef(null),
    faces: useRef(null),
    subjects: useRef(null),
    motion: useRef(null),
    motionBar: useRef(null),
    speech: useRef(null),
    speechDot: useRef(null),
    keyframes: useRef(null),
    audio: useRef(null),
    analysis: useRef(null),
    content: useRef(null),
    sceneCut: useRef(null),
  };

  const sortedKeys = useMemo(() => {
    const sortKeys = (obj) =>
      obj ? Object.keys(obj).map(Number).sort((a, b) => a - b) : [];
    return {
      face: sortKeys(detectionData?.face_timeline),
      person: sortKeys(detectionData?.person_timeline),
      motion: sortKeys(detectionData?.motion_timeline),
      speech: sortKeys(detectionData?.speech_active),
    };
  }, [detectionData]);

  const sceneCuts = useMemo(() => {
    const list = Array.isArray(detectionData?.scene_cuts) ? detectionData.scene_cuts : [];
    return [...list].sort((a, b) => a - b);
  }, [detectionData]);

  const cropDims = useMemo(() => {
    if (!targetRatio || !sourceWidth || !sourceHeight) return null;
    let cropW = Math.round(sourceHeight * targetRatio);
    let cropH = sourceHeight;
    if (cropW > sourceWidth) {
      cropW = sourceWidth;
      cropH = Math.round(sourceWidth / targetRatio);
    }
    return { cropW, cropH, maxX: Math.max(0, sourceWidth - cropW) };
  }, [targetRatio, sourceWidth, sourceHeight]);

  // Static stats that don't change per frame.
  const staticStats = useMemo(() => {
    if (!detectionData) {
      return { keyframes: 0, audio: '—', analysis: '—', content: '—' };
    }
    const m = detectionData.metadata || {};
    const kf = Array.isArray(renderPlan?.ops)
      ? renderPlan.ops.length
      : (subjectKeyframes?.length || 0);
    const audioBits = [];
    const speechKeys = sortedKeys.speech;
    const speechCount = speechKeys.length;
    const activeSpeech = speechKeys.filter(
      (k) => detectionData.speech_active?.[String(k)]).length;
    if (speechCount) {
      const durMs = activeSpeech * (m.duration_ms && speechCount
        ? Math.max(1, Math.round(m.duration_ms / speechCount))
        : 100);
      audioBits.push(`${(durMs / 1000).toFixed(1)}s speech`);
    }
    const lang = m.language || detectionData.language || '';
    if (lang) audioBits.push(String(lang));
    return {
      keyframes: kf,
      audio: audioBits.join(' · ') || (speechCount ? `${speechCount} samples` : '—'),
      analysis: typeof m.analysis_sec === 'number' ? `${m.analysis_sec.toFixed(1)}s` : '—',
      content: m.is_live_action ? 'Live-action' : 'Animated',
    };
  }, [detectionData, renderPlan, subjectKeyframes, sortedKeys.speech]);

  useEffect(() => {
    if (refs.keyframes.current) refs.keyframes.current.textContent = String(staticStats.keyframes);
    if (refs.audio.current) refs.audio.current.textContent = staticStats.audio;
    if (refs.analysis.current) refs.analysis.current.textContent = staticStats.analysis;
    if (refs.content.current) refs.content.current.textContent = staticStats.content;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [staticStats]);

  useEffect(() => {
    const video = videoRef?.current;
    if (!video) return;
    const minDelta = 1000 / STATS_UPDATE_HZ;
    let raf;
    let lastUpdate = 0;

    const tick = () => {
      const now = performance.now();
      if (now - lastUpdate >= minDelta) {
        lastUpdate = now;
        const t = video.currentTime;
        const tMs = Math.round(t * 1000);
        const relT = Math.max(0, t - clipStart);

        // Strategy from render plan ops.
        const op = findActiveOp(renderPlan?.ops, t);
        if (refs.strategy.current) {
          refs.strategy.current.textContent = op?.strategy_label || op?.kind || '—';
        }

        // Crop X position — match the live cropSegments the preview
        // canvas uses so the "Crop X" stat reflects the user's edits.
        if (refs.cropX.current && cropDims) {
          const { cropSegments } = useTimelineStore.getState();
          const sxPct = getCropXForTime(relT, cropSegments, subjectKeyframes);
          const center = (sxPct / 100) * sourceWidth;
          const cropX = Math.max(0, Math.min(
            cropDims.maxX, Math.round(center - cropDims.cropW / 2),
          ));
          refs.cropX.current.textContent = `${cropX} / ${cropDims.maxX}`;
        }

        // Faces (with unique track count).
        const faceKey = findNearestKey(sortedKeys.face, tMs, FACE_LOOKUP_TOL_MS);
        const faces = faceKey != null
          ? (detectionData?.face_timeline?.[String(faceKey)] || [])
          : [];
        if (refs.faces.current) {
          const tracks = new Set();
          for (const f of faces) {
            const tid = Number(f.track_id ?? -1);
            if (tid >= 0) tracks.add(tid);
          }
          refs.faces.current.textContent = faces.length
            ? `${faces.length} (${tracks.size} ${tracks.size === 1 ? 'track' : 'tracks'})`
            : '0';
        }

        // Subjects with class breakdown.
        const personKey = findNearestKey(sortedKeys.person, tMs, SUBJECT_LOOKUP_TOL_MS);
        const persons = personKey != null
          ? (detectionData?.person_timeline?.[String(personKey)] || [])
          : [];
        if (refs.subjects.current) {
          if (persons.length) {
            const breakdown = classBreakdown(persons);
            refs.subjects.current.textContent = breakdown
              ? `${persons.length} (${breakdown})`
              : String(persons.length);
          } else {
            refs.subjects.current.textContent = '0';
          }
        }

        // Motion magnitude + bar.
        const motionKey = findNearestKey(sortedKeys.motion, tMs, MOTION_LOOKUP_TOL_MS);
        const motion = motionKey != null
          ? Number(detectionData?.motion_timeline?.[String(motionKey)] || 0)
          : 0;
        if (refs.motion.current) {
          refs.motion.current.textContent = motionKey != null ? motion.toFixed(2) : '—';
        }
        if (refs.motionBar.current) {
          const pct = Math.max(0, Math.min(100, (motion / 10) * 100));
          refs.motionBar.current.style.width = `${pct}%`;
        }

        // Speech activity dot.
        const speechKey = findNearestKey(sortedKeys.speech, tMs, SPEECH_LOOKUP_TOL_MS);
        const speechOn = speechKey != null
          ? !!detectionData?.speech_active?.[String(speechKey)]
          : false;
        if (refs.speech.current) {
          refs.speech.current.textContent = speechOn ? 'active' : 'silent';
          refs.speech.current.style.color = speechOn ? '#4ade80' : '#666';
        }
        if (refs.speechDot.current) {
          refs.speechDot.current.style.background = speechOn ? '#22c55e' : '#3a3a3a';
          refs.speechDot.current.style.boxShadow = speechOn
            ? '0 0 6px rgba(34,197,94,0.7)'
            : 'none';
        }

        // Scene cut proximity warning.
        if (refs.sceneCut.current) {
          let near = false;
          if (sceneCuts.length) {
            const idx = sceneCuts.findIndex((c) => c >= tMs - SCENE_CUT_WARN_MS);
            if (idx >= 0) {
              const c = sceneCuts[idx];
              if (Math.abs(c - tMs) <= SCENE_CUT_WARN_MS) near = true;
            }
          }
          refs.sceneCut.current.textContent = near ? '⚡ cut soon' : 'clear';
          refs.sceneCut.current.style.color = near ? '#fbbf24' : '#666';
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoRef, detectionData, sortedKeys, sceneCuts, cropDims, subjectKeyframes, renderPlan, clipStart, sourceWidth]);

  return (
    <aside className="reframe-preview__stats reframe-stats">
      <div className="reframe-stats__title">Reframer Inspector</div>
      <StatRow label="Strategy" valueRef={refs.strategy} />
      <StatRow label="Crop X" valueRef={refs.cropX} />
      <StatRow label="Faces" valueRef={refs.faces} />
      <StatRow label="Subjects" valueRef={refs.subjects} />
      <div className="reframe-stats__row">
        <span className="reframe-stats__label">Motion</span>
        <span ref={refs.motion} className="reframe-stats__value reframe-stats__value--mono">—</span>
      </div>
      <div className="reframe-stats__bar">
        <div ref={refs.motionBar} className="reframe-stats__bar-fill" />
      </div>
      <div className="reframe-stats__row">
        <span className="reframe-stats__label">Speech</span>
        <span className="reframe-stats__speech">
          <span ref={refs.speechDot} className="reframe-stats__dot" />
          <span ref={refs.speech} className="reframe-stats__value reframe-stats__value--mono">—</span>
        </span>
      </div>
      <StatRow label="Scene Cut" valueRef={refs.sceneCut} />
      <div className="reframe-stats__divider" />
      <StatRow label="Keyframes" valueRef={refs.keyframes} />
      <StatRow label="Audio" valueRef={refs.audio} />
      <StatRow label="Analysis" valueRef={refs.analysis} />
      <StatRow label="Content" valueRef={refs.content} />
    </aside>
  );
}
