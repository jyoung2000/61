import React, { useEffect, useMemo, useRef, useState } from 'react';

// The Processing Log shows a line for EACH progress step a stage emits — the
// per-batch translation ticks ("Translating subtitles… (N/M)"), Whisper/face
// percentages, summary chunks, export "(N/M)" — so you can follow exactly where
// a run is and what it did, live AND in the saved log you review afterward. The
// durable event log is already throttled server-side (≈1 event / stage / 8 s),
// which keeps the volume sane without hiding the progression; here we only drop a
// run of the EXACT same (type + stage + message + progress) line — which can
// appear when the persisted history overlaps the live socket, or a status is
// re-broadcast. Everything distinct is kept, in order. Purely presentational.
function dedupeConsecutive(entries) {
  const out = [];
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const prev = out[out.length - 1];
    if (prev
        && prev.type === e.type
        && (prev.stage_id || prev._stageId || '') === (e.stage_id || e._stageId || '')
        && (prev.message || '') === (e.message || '')
        && (prev.progress ?? null) === (e.progress ?? null)) {
      continue;
    }
    out.push(e);
  }
  return out;
}

// Max rows rendered into the DOM WHILE A JOB IS LIVE. A long run can emit 700+
// live ticks; re-rendering all of them on every websocket event is what made the
// tab feel slow, so the live view shows a trailing window by default. This cap
// does NOT apply when reviewing a finished job (no socket updates → no re-render
// storm, so the full log renders), and the user can override it live via the
// "show all" toggle. The header always reports the true total.
const _MAX_VISIBLE_ROWS = 250;

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
  // Live runs window to the last N rows for performance; the user can opt into
  // the full log mid-run with this toggle. A finished job always shows the full
  // log regardless (see below).
  const [showAll, setShowAll] = useState(false);
  const scrollRef = useRef(null);
  const prevStageRef = useRef(null);

  // Show every distinct progress step; only collapse exact consecutive
  // duplicates.
  const deduped = useMemo(() => dedupeConsecutive(entries), [entries]);
  // Cap the RENDERED rows for performance ONLY while live + not "show all".
  // The re-render-per-websocket-event storm is what made the tab sluggish; a
  // finished job has no socket updates, so render its full log. Reviewing the
  // complete run is the whole point of the saved log.
  const hardCapActive = isLive && !showAll;
  const truncated = hardCapActive && deduped.length > _MAX_VISIBLE_ROWS;
  const visible = truncated ? deduped.slice(-_MAX_VISIBLE_ROWS) : deduped;
  // Whether a "show all / show recent" toggle is worth offering (live + over cap).
  const canToggle = isLive && deduped.length > _MAX_VISIBLE_ROWS;

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
            ({deduped.length} steps
            {truncated && <span> · showing last {_MAX_VISIBLE_ROWS}</span>}
            {canToggle && (
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); setShowAll((s) => !s); }}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.stopPropagation(); setShowAll((s) => !s); } }}
                style={{ color: 'var(--accent-cyan, #06b6d4)', marginLeft: 6, cursor: 'pointer', textDecoration: 'underline' }}
              >
                {showAll ? 'show recent' : 'show all'}
              </span>
            )}
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
