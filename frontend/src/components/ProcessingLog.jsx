import React, { useEffect, useMemo, useRef, useState } from 'react';

// A "progress-like" entry is one a stage emits repeatedly as it advances — a
// percent, an N/M counter, an "Extracted N frames", a "Still processing …
// elapsed" heartbeat, or an identical retry line. Left as-is the log balloons
// (summary alone emits 21 "chunk N/21" lines; Whisper a tick per %, export/SEO
// one per clip; translation a batch tick every few seconds). The display policy
// is "live trail, fold when done": while a job is LIVE the currently-running
// stage shows a short TRAIL of its last few progress steps so you can watch the
// current step advance (e.g. 88→96→104→108 of 108); every FINISHED stage — and
// the whole log once the job ends — folds to the single latest line. Milestones
// (success / warning / error / checkpoint and one-off status lines) are never
// touched. Purely presentational: the durable event list is untouched.
const _PROGRESS_RE = /\d+\s*\/\s*\d+|\d+\s*%|elapsed\)|Extracted\s+\d+\s+frames|chunk\s+\d+|Attempting SEO generation|Translating subtitles with/i;

// How many recent progress steps the ACTIVE (currently-running) stage keeps
// visible while live, so the user perceives motion instead of one updating
// number. Finished stages always fold to 1.
const _PROGRESS_TRAIL = 5;

function _progressSig(entry) {
  if (!entry || (entry.type !== 'status' && entry.type !== 'info')) return null;
  const msg = typeof entry.message === 'string' ? entry.message : '';
  if (!msg || !_PROGRESS_RE.test(msg)) return null;
  const stage = entry.stage_id || entry._stageId || '';
  // Mask run-specific numbers so "chunk 5/21" and "chunk 6/21" share one key.
  return stage + '|' + msg.replace(/\d+/g, '#');
}

// Apply the "live trail, fold when done" policy. Every non-progress (milestone)
// entry is preserved in order. For progress entries:
//   • While ``isLive``, the ACTIVE stage (the stage of the most recent entry)
//     keeps the last ``_PROGRESS_TRAIL`` occurrences of each signature — a short
//     accumulating trail you watch advance.
//   • Every other stage — and ALL stages once the job is no longer live — folds
//     to the single latest occurrence of each signature (clean saved log).
function collapseProgress(entries, isLive = false) {
  // The active stage is only meaningful while live: it's the stage_id of the
  // most recent entry that carries one.
  let activeStage = null;
  if (isLive) {
    for (let i = entries.length - 1; i >= 0; i--) {
      const sid = entries[i]?.stage_id || entries[i]?._stageId;
      if (sid) { activeStage = sid; break; }
    }
  }
  const lastIndex = new Map();     // sig -> last index (for folded stages)
  const activeIdxs = new Map();    // sig -> [indices] (active stage only)
  for (let i = 0; i < entries.length; i++) {
    const sig = _progressSig(entries[i]);
    if (!sig) continue;
    lastIndex.set(sig, i);
    const sid = entries[i].stage_id || entries[i]._stageId || '';
    if (activeStage && sid === activeStage) {
      const arr = activeIdxs.get(sig) || [];
      arr.push(i);
      activeIdxs.set(sig, arr);
    }
  }
  // Indices to keep for the active stage: the last _PROGRESS_TRAIL per signature.
  const keepTrail = new Set();
  for (const arr of activeIdxs.values()) {
    for (const idx of arr.slice(-_PROGRESS_TRAIL)) keepTrail.add(idx);
  }
  return entries.filter((e, i) => {
    const sig = _progressSig(e);
    if (!sig) return true;                       // milestones always shown
    if (activeIdxs.has(sig)) return keepTrail.has(i);  // active stage: trail only
    return lastIndex.get(sig) === i;             // folded: only the latest
  });
}

const STAGE_COLORS = {
  queue:         '#6b7280',
  metadata:      '#06b6d4',
  extraction:    '#0ea5e9',
  face_detection:'#a78bfa',
  transcription: '#8b5cf6',
  diarization:   '#7c3aed',
  conversion:    '#f59e0b',
  summary:       '#f97316',
  clips:         '#22c55e',
  saving:        '#16a34a',
  polishing:     '#ec4899',
  translation:   '#14b8a6',
  seo:           '#64748b',
};

const STAGE_BADGES = {
  queue:         'QUEUE',
  metadata:      'META',
  extraction:    'EXTRACT',
  face_detection:'FACES',
  transcription: 'WHISPER',
  diarization:   'SPEAKER',
  conversion:    'CONVERT',
  summary:       'SUMMARY',
  clips:         'CLIPS',
  saving:        'SAVE',
  polishing:     'POLISH',
  translation:   'XLATE',
  seo:           'SEO',
};

const TYPE_ICONS = {
  status:     '›',
  success:    '✓',
  warning:    '⚠',
  error:      '✕',
  info:       '·',
  checkpoint: '⚑',
};

const TYPE_COLORS = {
  status:     'var(--text-secondary)',
  success:    'var(--success, #22c55e)',
  warning:    'var(--accent-amber, #f59e0b)',
  error:      'var(--danger, #ef4444)',
  info:       'var(--text-muted)',
  // A saved resume point — make it stand out so the user can see where a
  // restart would pick up.
  checkpoint: 'var(--accent-cyan, #06b6d4)',
};

function fmtRelative(absTime, startTime) {
  if (!absTime || !startTime) return '';
  const diffMs = absTime - startTime;
  if (diffMs < 0) return '+0s';
  const s = Math.floor(diffMs / 1000);
  if (s < 60) return `+${s}s`;
  return `+${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
}

export default function ProcessingLog({
  entries = [],
  pipelineStartTime = null,
  isLive = false,
  initialExpanded = true,
}) {
  const [expanded, setExpanded] = useState(initialExpanded);
  const scrollRef = useRef(null);
  const prevStageRef = useRef(null);

  // Live trail for the running stage; fold finished stages (and the whole log
  // once the job is done) to one line each.
  const visible = useMemo(() => collapseProgress(entries, isLive), [entries, isLive]);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (expanded && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visible.length, expanded]);

  const warningCount = entries.filter((e) => e.type === 'warning').length;
  const errorCount = entries.filter((e) => e.type === 'error').length;

  return (
    <div style={{
      background: 'var(--bg-panel)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      marginBottom: 16,
      overflow: 'hidden',
    }}>
      {/* Header */}
      <button
        onClick={() => setExpanded((p) => !p)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between',
          padding: '8px 12px', background: 'none', border: 'none',
          cursor: 'pointer', color: 'var(--text-secondary)',
          fontSize: 11, fontFamily: 'var(--font-mono)',
          textTransform: 'uppercase', letterSpacing: '0.05em',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isLive && (
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: '#22c55e',
              animation: 'pulse 1.5s ease-in-out infinite',
              flexShrink: 0,
            }} />
          )}
          Processing Log
          <span style={{ color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0 }}>
            ({visible.length} steps
            {warningCount > 0 && <span style={{ color: 'var(--accent-amber, #f59e0b)', marginLeft: 4 }}> {warningCount}⚠</span>}
            {errorCount > 0 && <span style={{ color: 'var(--danger, #ef4444)', marginLeft: 4 }}> {errorCount}✕</span>}
            )
          </span>
        </span>
        <span style={{ fontSize: 14 }}>{expanded ? '▴' : '▾'}</span>
      </button>

      {expanded && (
        <div
          ref={scrollRef}
          style={{
            maxHeight: 280, overflowY: 'auto',
            padding: '8px 12px',
            fontFamily: 'var(--font-mono)', fontSize: 10,
            lineHeight: 1.75,
          }}
        >
          {visible.map((entry, i) => {
            const stageId = entry.stage_id || entry._stageId || '';
            const stageColor = STAGE_COLORS[stageId] || 'var(--border)';
            const stageBadge = STAGE_BADGES[stageId] || '';
            const showStageSep = stageId && stageId !== (visible[i - 1]?.stage_id || visible[i - 1]?._stageId || '');
            const icon = TYPE_ICONS[entry.type] || '·';
            const iconColor = TYPE_COLORS[entry.type] || TYPE_COLORS.info;
            const relTime = fmtRelative(entry._absTime, pipelineStartTime);

            return (
              <React.Fragment key={i}>
                {showStageSep && stageBadge && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    margin: '6px 0 4px', opacity: 0.7,
                  }}>
                    <div style={{ flex: 1, height: 1, background: stageColor, opacity: 0.4 }} />
                    <span style={{
                      fontSize: 8, fontFamily: 'var(--font-mono)',
                      color: stageColor, textTransform: 'uppercase',
                      letterSpacing: '0.1em', fontWeight: 700,
                    }}>
                      {stageBadge}
                    </span>
                    <div style={{ flex: 1, height: 1, background: stageColor, opacity: 0.4 }} />
                  </div>
                )}
                <div style={{
                  display: 'flex', gap: 6, alignItems: 'baseline',
                  // Make a checkpoint read as a milestone, not just another line.
                  ...(entry.type === 'checkpoint' ? {
                    fontWeight: 600,
                    background: 'var(--accent-cyan, #06b6d4)11',
                    borderLeft: '2px solid var(--accent-cyan, #06b6d4)',
                    padding: '2px 4px 2px 6px', margin: '2px 0', borderRadius: 3,
                  } : {}),
                }}>
                  {/* Relative elapsed time */}
                  <span style={{ color: 'var(--text-muted)', flexShrink: 0, minWidth: 40 }}>
                    {relTime || entry.ts || ''}
                  </span>
                  {/* Stage badge (inline, small) */}
                  {stageBadge && (
                    <span style={{
                      fontSize: 8, padding: '1px 4px',
                      background: `${stageColor}22`,
                      border: `1px solid ${stageColor}44`,
                      color: stageColor,
                      borderRadius: 3, flexShrink: 0,
                      textTransform: 'uppercase', letterSpacing: '0.05em',
                      fontWeight: 700,
                    }}>
                      {stageBadge}
                    </span>
                  )}
                  {/* Type icon */}
                  <span style={{ color: iconColor, flexShrink: 0 }}>{icon}</span>
                  {/* Progress % (if present) */}
                  {entry.progress != null && entry.type === 'status' && (
                    <span style={{ color: 'var(--accent-cyan, #06b6d4)', flexShrink: 0, minWidth: 24, textAlign: 'right' }}>
                      {typeof entry.progress === 'number' ? entry.progress : String(entry.progress ?? '')}%
                    </span>
                  )}
                  {/* Message */}
                  <span style={{ color: TYPE_COLORS[entry.type] || TYPE_COLORS.status, wordBreak: 'break-word' }}>
                    {typeof entry.message === 'string' ? entry.message : String(entry.message ?? '')}
                  </span>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}
