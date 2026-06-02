import React, { useEffect, useRef, useState } from 'react';

// Stage color palette keyed by stage group
const STAGE_COLORS = {
  queue:         '#6b7280', // gray
  metadata:      '#06b6d4', // cyan
  extraction:    '#0ea5e9', // sky
  face_detection:'#a78bfa', // purple
  transcription: '#8b5cf6', // violet
  diarization:   '#7c3aed', // indigo
  conversion:    '#f59e0b', // amber
  summary:       '#f97316', // orange
  clips:         '#22c55e', // green
  saving:        '#16a34a', // dark green
  // background tasks
  polishing:     '#ec4899', // pink
  translation:   '#14b8a6', // teal
  seo:           '#64748b', // slate
};

// Order matches the backend pipeline: translation runs right after conversion
// (so clips/summary use the translated transcript) and BEFORE the summary.
const STAGE_ORDER = [
  'metadata', 'extraction', 'face_detection', 'transcription',
  'diarization', 'conversion', 'translation', 'summary', 'clips', 'saving',
];

function fmtElapsed(secs) {
  if (!secs && secs !== 0) return '';
  const s = Math.round(secs);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
}

function fmtTotal(secs) {
  if (!secs && secs !== 0) return '';
  const s = Math.round(secs);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export default function PipelineTracker({
  currentStageId,
  stageTimes = {},
  pipelineElapsed = 0,
  isComplete = false,
  isFailed = false,
  stages = null, // optional override from API
}) {
  const displayStages = stages || STAGE_ORDER.map((id) => ({ id }));

  // Compute total weight for bar sizing (approximate)
  const STAGE_WEIGHTS = {
    metadata: 3, extraction: 9, face_detection: 27, transcription: 14,
    diarization: 2, conversion: 2, summary: 18, translation: 6, clips: 18, saving: 2,
  };
  const totalWeight = displayStages.reduce((sum, s) => sum + (STAGE_WEIGHTS[s.id] || 5), 0);

  const STAGE_LABELS = {
    metadata: 'Metadata', extraction: 'Frames', face_detection: 'Faces',
    transcription: 'Whisper', diarization: 'Speakers', conversion: 'Convert',
    summary: 'Summary', translation: 'Translate', clips: 'Clips', saving: 'Save',
  };

  const currentIdx = displayStages.findIndex((s) => s.id === currentStageId);

  return (
    <div style={{
      background: 'var(--bg-panel)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      padding: '10px 12px',
      marginBottom: 8,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 8,
      }}>
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', textTransform: 'uppercase',
          letterSpacing: '0.08em', color: 'var(--text-secondary)' }}>
          Pipeline Progress
        </span>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
          {isComplete
            ? `Completed in ${fmtTotal(pipelineElapsed)}`
            : isFailed
              ? 'Failed'
              : pipelineElapsed > 0 ? `${fmtTotal(pipelineElapsed)} elapsed` : ''}
        </span>
      </div>

      {/* Stage bar */}
      <div style={{ display: 'flex', gap: 2, height: 6, borderRadius: 3, overflow: 'hidden' }}>
        {displayStages.map((stage, idx) => {
          const weight = STAGE_WEIGHTS[stage.id] || 5;
          const widthPct = (weight / totalWeight) * 100;
          const isDone = stageTimes[stage.id] != null || (isComplete && idx < displayStages.length);
          const isActive = stage.id === currentStageId && !isComplete;
          const color = STAGE_COLORS[stage.id] || '#6b7280';

          return (
            <div
              key={stage.id}
              title={`${stage.label || STAGE_LABELS[stage.id] || stage.id}${stageTimes[stage.id] != null ? ` — ${fmtElapsed(stageTimes[stage.id])}` : ''}`}
              style={{
                flex: `0 0 ${widthPct}%`,
                background: isDone || isActive ? color : 'var(--border)',
                opacity: isActive ? 1 : isDone ? 0.85 : 0.3,
                borderRadius: 2,
                position: 'relative',
                overflow: 'hidden',
                transition: 'background 0.4s, opacity 0.4s',
              }}
            >
              {isActive && (
                <div style={{
                  position: 'absolute', top: 0, left: '-100%',
                  width: '100%', height: '100%',
                  background: `linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)`,
                  animation: 'shimmer 1.5s ease-in-out infinite',
                }} />
              )}
            </div>
          );
        })}
      </div>

      {/* Stage labels row */}
      <div style={{ display: 'flex', gap: 2, marginTop: 5 }}>
        {displayStages.map((stage, idx) => {
          const weight = STAGE_WEIGHTS[stage.id] || 5;
          const widthPct = (weight / totalWeight) * 100;
          const isDone = stageTimes[stage.id] != null || isComplete;
          const isActive = stage.id === currentStageId && !isComplete;
          const color = STAGE_COLORS[stage.id] || '#6b7280';
          const label = stage.label || STAGE_LABELS[stage.id] || stage.id;
          const elapsed = stageTimes[stage.id];

          return (
            <div
              key={stage.id}
              style={{
                flex: `0 0 ${widthPct}%`,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                overflow: 'hidden',
              }}
            >
              <span style={{
                fontSize: 9,
                fontFamily: 'var(--font-mono)',
                color: isActive ? color : isDone ? 'var(--text-secondary)' : 'var(--text-muted)',
                fontWeight: isActive ? 600 : 400,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                maxWidth: '100%',
                textAlign: 'center',
                transition: 'color 0.3s',
              }}>
                {label}
              </span>
              {elapsed != null && (
                <span style={{
                  fontSize: 8, fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)', whiteSpace: 'nowrap',
                }}>
                  {fmtElapsed(elapsed)}
                </span>
              )}
            </div>
          );
        })}
      </div>

      <style>{`
        @keyframes shimmer {
          0% { left: -100%; }
          100% { left: 100%; }
        }
      `}</style>
    </div>
  );
}
