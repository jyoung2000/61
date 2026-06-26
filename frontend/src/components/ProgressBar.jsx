import React, { useRef } from 'react';

export default function ProgressBar({ progress, message, variant = 'cyan' }) {
  const colorMap = { amber: 'var(--accent-amber)', green: 'var(--success)', cyan: 'var(--accent-cyan)' };
  const color = colorMap[variant] || colorMap.cyan;

  // Track peak progress to ensure monotonically increasing display.
  // This prevents the bar from jumping backward on transient state updates.
  const peakRef = useRef(0);
  const safeProgress = Math.min(100, Math.max(0, progress));
  // A LARGE backward drop is not jitter — it's a genuine restart. A job revived
  // after a container crash re-runs the early stages (e.g. it died at 95% during
  // clip export and comes back at ~5% re-extracting/transcribing). Follow that
  // down so the bar tells the truth, instead of sticking at the old peak and
  // implying the job is about to finish when it actually started over.
  const RESTART_DROP = 15; // percentage points below peak that means "restarted"
  if (
    safeProgress >= peakRef.current ||              // advancing
    safeProgress === 0 ||                            // explicit reset
    safeProgress <= peakRef.current - RESTART_DROP   // restart / revive
  ) {
    peakRef.current = safeProgress;
  }
  const displayProgress = peakRef.current;

  return (
    <div style={{ width: '100%' }}>
      {message && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 8,
            marginBottom: 6,
          }}
        >
          {/* Truncate to one line so a long status string can't wrap and shove
              the % off-screen on a narrow (mobile) viewport. */}
          <span style={{
            fontSize: 12, color: 'var(--text-secondary)',
            flex: 1, minWidth: 0, overflow: 'hidden',
            textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>{typeof message === 'string' ? message : String(message ?? '')}</span>
          {/* % never shrinks — stays pinned and readable on the right. */}
          <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'var(--font-mono)', color, flexShrink: 0 }}>{displayProgress}%</span>
        </div>
      )}
      <div
        style={{
          height: 10,
          background: 'var(--bg-elevated)',
          position: 'relative',
          overflow: 'hidden',
          borderRadius: 5,
          border: '1px solid var(--border)',
        }}
      >
        <div
          className={displayProgress < 100 ? 'shimmer' : ''}
          style={{
            height: '100%',
            width: `${displayProgress}%`,
            background: color,
            transition: 'width 0.3s ease',
            borderRadius: 5,
          }}
        />
      </div>
    </div>
  );
}
