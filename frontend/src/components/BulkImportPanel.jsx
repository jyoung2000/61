import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

// Live status for a Companion bulk folder import (sequential download →
// analyze, one video at a time). The run lives on the SERVER, so this panel
// is deliberately standalone: it discovers a running import by itself (via
// /import-folder/active), polls it, and renders wherever it's mounted — the
// Dashboard (so the user can check on it after closing the import dialog or
// from another device) and inside the Companion import dialog itself.
//
// Props:
//   startId          — a bulk_id this page just started (adopted immediately;
//                      without it the panel self-discovers on mount)
//   onRunningChange  — optional (bool) callback so a host page can disable
//                      its own "start another import" affordances
//
// Completed/analyzing items link straight to their /analysis/<job_id> page.

const T = {
  panel: 'var(--bg-panel, rgba(28,28,30,0.92))',
  elev: 'var(--bg-elevated, rgba(255,255,255,0.06))',
  border: 'var(--border, rgba(255,255,255,0.1))',
  accent: 'var(--accent-cyan, #0A84FF)',
  tp: 'var(--text-primary, #F5F5F7)',
  ts: 'var(--text-secondary, rgba(235,235,245,0.68))',
  tm: 'var(--text-muted, rgba(235,235,245,0.42))',
  track: 'var(--border, rgba(255,255,255,0.16))',
  ok: 'var(--success, #30D158)',
  danger: 'var(--danger, #FF453A)',
  dangerDim: 'var(--danger-dim, rgba(255,59,48,0.14))',
  mono: 'var(--font-mono, ui-monospace, monospace)',
};

const sv = { fill: 'none', stroke: 'currentColor', strokeWidth: 2.2, strokeLinecap: 'round', strokeLinejoin: 'round' };
const CheckIcon = () => <svg width={15} height={15} viewBox="0 0 24 24" {...sv}><path d="M20 6 9 17l-5-5" /></svg>;
const XIcon = () => <svg width={15} height={15} viewBox="0 0 24 24" {...sv}><path d="M18 6 6 18" /><path d="m6 6 12 12" /></svg>;
const LayersIcon = ({ s = 17 }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" {...sv} strokeWidth={1.8}>
    <path d="m12 2 9 4.9-9 4.9-9-4.9z" /><path d="m3 11.9 9 4.9 9-4.9" /><path d="m3 16.9 9 4.9 9-4.9" />
  </svg>
);

const spinner = (
  <span style={{
    width: 14, height: 14, flex: 'none', borderRadius: '50%', display: 'inline-block',
    border: `2px solid ${T.track}`, borderTopColor: T.accent,
    animation: 'bipSpin 0.8s linear infinite',
  }} />
);

const itemGlyph = (it, active) => {
  if (it.status === 'complete') return <span style={{ color: T.ok, display: 'flex' }}><CheckIcon /></span>;
  if (it.status === 'failed' || it.status === 'no_space') return <span style={{ color: T.danger, display: 'flex' }}><XIcon /></span>;
  if (it.status === 'cancelled' || it.status === 'skipped') return <span style={{ color: T.tm, fontSize: 13 }}>—</span>;
  if (active) return spinner;
  return <span style={{ color: T.tm, fontSize: 13 }}>·</span>;
};

const itemDetail = (it) => {
  if (it.status === 'downloading') {
    const pct = it.size > 0 ? Math.min(100, Math.round((it.done_bytes / it.size) * 100)) : null;
    return pct == null ? 'Downloading…' : `Downloading ${pct}%`;
  }
  if (it.status === 'analyzing') {
    return it.analysis_message || `Analyzing… ${it.analysis_progress != null ? `${it.analysis_progress}%` : ''}`;
  }
  if (it.status === 'no_space') return 'Out of disk space';
  if (it.status === 'failed') return it.error || 'Failed';
  if (it.status === 'skipped') return 'Skipped';
  if (it.status === 'complete') return 'Done';
  if (it.status === 'cancelled') return 'Cancelled';
  return 'Queued';
};

export default function BulkImportPanel({ startId = '', onRunningChange }) {
  const [bulkId, setBulkId] = useState('');
  const [bulk, setBulk] = useState(null);
  const [cancelling, setCancelling] = useState(false);
  const navigate = useNavigate();

  // Adopt a run this page just started.
  useEffect(() => { if (startId) setBulkId(startId); }, [startId]);

  // Discover a run already going on the server (mount without startId —
  // e.g. the Dashboard, or the dialog reopened mid-run).
  useEffect(() => {
    if (startId) return;
    let alive = true;
    (async () => {
      try {
        const r = await fetch('/api/providers/companion-files/import-folder/active');
        const d = await r.json();
        if (alive && d && d.bulk_id) setBulkId(d.bulk_id);
      } catch { /* no companion / no run — render nothing */ }
    })();
    return () => { alive = false; };
  }, [startId]);

  // Poll while attached; stop on terminal states (the last state stays
  // rendered as a summary until dismissed).
  useEffect(() => {
    if (!bulkId) return undefined;
    let stopped = false;
    let timer = 0;
    const tick = async () => {
      try {
        const r = await fetch(`/api/providers/companion-files/import-folder/progress?bulk_id=${encodeURIComponent(bulkId)}`);
        if (r.status === 404) { if (!stopped) { setBulk(null); setBulkId(''); } return; }
        const d = await r.json();
        if (stopped) return;
        setBulk(d);
        if (d.status === 'running') timer = setTimeout(tick, 1200);
      } catch {
        if (!stopped) timer = setTimeout(tick, 2500);
      }
    };
    tick();
    return () => { stopped = true; clearTimeout(timer); };
  }, [bulkId]);

  const running = !!(bulk && bulk.status === 'running');
  useEffect(() => { onRunningChange && onRunningChange(running); }, [running, onRunningChange]);

  if (!bulk) return null;

  const cancel = async () => {
    if (cancelling) return;
    setCancelling(true);
    try {
      const r = await fetch(`/api/providers/companion-files/import-folder/cancel?bulk_id=${encodeURIComponent(bulkId)}`, { method: 'POST' });
      if (r.ok) {
        // The endpoint returns with the run ALREADY terminal (it aborts the
        // running analysis and stops the runner before answering), so reflect
        // it now instead of leaving "Cancelling…" up until the next poll.
        setBulk((b) => (b ? {
          ...b,
          status: 'cancelled',
          items: (b.items || []).map((it) => (
            it.status === 'queued' ? { ...it, status: 'skipped' }
              : (it.status === 'downloading' || it.status === 'analyzing')
                ? { ...it, status: 'cancelled' } : it)),
        } : b));
      }
    } catch { /* next poll shows the real state */ }
    finally { setCancelling(false); }
  };
  const dismiss = () => { setBulk(null); setBulkId(''); };

  const items = bulk.items || [];
  const cur = items.find((it) => it.status === 'downloading' || it.status === 'analyzing');
  // Overall bar: finished items + a fraction for the one in flight
  // (download ≈ first 20 % of a video's wall-clock, analysis the rest).
  let frac = 0;
  if (cur) {
    if (cur.status === 'downloading') frac = 0.2 * (cur.size > 0 ? cur.done_bytes / cur.size : 0);
    else frac = 0.2 + 0.8 * (Math.min(100, cur.analysis_progress || 0) / 100);
  }
  const overallPct = bulk.total ? Math.min(100, Math.round(((bulk.done + frac) / bulk.total) * 100)) : 0;
  const headline = bulk.status === 'running'
    ? `Importing “${bulk.folder_name}” — video ${Math.min(bulk.done + 1, bulk.total)} of ${bulk.total}`
    : bulk.status === 'complete'
      ? `Folder import done — ${bulk.ok} of ${bulk.total} video${bulk.total === 1 ? '' : 's'} imported`
      : bulk.status === 'out_of_space'
        ? `Stopped — ClipAI ran out of disk space (${bulk.ok} of ${bulk.total} done)`
        : bulk.status === 'cancelled'
          ? `Folder import cancelled (${bulk.ok} of ${bulk.total} done)`
          : `Folder import failed: ${bulk.error || 'unknown error'}`;
  const headColor = bulk.status === 'out_of_space' || bulk.status === 'error' ? T.danger : T.tp;

  return (
    <div style={{
      padding: '12px 14px', borderRadius: 12, marginBottom: 10,
      background: T.elev, border: `1px solid ${T.border}`,
      display: 'flex', flexDirection: 'column', gap: 9,
      fontFamily: 'var(--font-sans, -apple-system, BlinkMacSystemFont, sans-serif)',
    }}>
      <style>{'@keyframes bipSpin { to { transform: rotate(360deg); } }'}</style>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        {running ? spinner : <span style={{ color: T.accent, display: 'flex', flex: 'none' }}><LayersIcon /></span>}
        <span style={{ flex: 1, minWidth: 0, fontSize: 13.5, fontWeight: 600, color: headColor, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {headline}
        </span>
        {running ? (
          <button onClick={cancel} disabled={cancelling}
            title="Stop everything now: the running analysis is aborted and nothing else in the queue starts"
            style={{ flex: 'none', fontSize: 12.5, fontWeight: 600, color: T.danger, background: T.dangerDim, padding: '6px 12px', borderRadius: 8, border: 'none', cursor: cancelling ? 'default' : 'pointer', opacity: cancelling ? 0.6 : 1 }}>
            {cancelling ? 'Stopping…' : 'Cancel import'}
          </button>
        ) : (
          <button onClick={dismiss}
            style={{ flex: 'none', fontSize: 12.5, fontWeight: 500, color: T.ts, background: 'transparent', padding: '6px 12px', borderRadius: 8, border: `1px solid ${T.border}`, cursor: 'pointer' }}>
            Dismiss
          </button>
        )}
      </div>
      <div style={{ height: 5, borderRadius: 3, background: T.track, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${overallPct}%`, background: T.accent, borderRadius: 3, transition: 'width 0.4s' }} />
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 168, overflowY: 'auto' }}>
        {items.map((it) => {
          const active = it.status === 'downloading' || it.status === 'analyzing';
          const linkable = !!it.job_id && (it.status === 'complete' || it.status === 'analyzing');
          return (
            <div key={it.path} style={{ display: 'flex', alignItems: 'center', gap: 9, minHeight: 24 }}>
              <span style={{ width: 16, flex: 'none', display: 'flex', justifyContent: 'center' }}>{itemGlyph(it, active)}</span>
              <span
                onClick={linkable ? () => navigate(`/analysis/${it.job_id}`) : undefined}
                title={linkable ? 'Open this video’s analysis page' : undefined}
                style={{
                  flex: 1, minWidth: 0, fontSize: 12.5,
                  color: active ? T.tp : T.ts,
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  cursor: linkable ? 'pointer' : 'default',
                  textDecoration: linkable ? 'underline' : 'none',
                  textDecorationColor: 'rgba(127,127,127,0.4)', textUnderlineOffset: 3,
                }}>
                {it.name}
              </span>
              <span style={{ flex: 'none', maxWidth: '46%', fontFamily: T.mono, fontSize: 11, color: it.status === 'failed' || it.status === 'no_space' ? T.danger : T.tm, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {itemDetail(it)}
              </span>
            </div>
          );
        })}
      </div>
      {running && (
        <div style={{ fontSize: 11.5, color: T.tm }}>
          Runs on the ClipAI server — safe to close this window or navigate away; progress also shows on the Dashboard.
          {' '}Videos run one at a time —{' '}
          {/* The setting lives on an inactive Settings tab, so it is invisible
              to browser find; link to it from where bulk imports actually
              happen (deep link opens the tab AND scrolls to the control). */}
          <span
            role="link"
            tabIndex={0}
            onClick={() => navigate('/settings?tab=advanced&section=concurrency')}
            onKeyDown={(e) => { if (e.key === 'Enter') navigate('/settings?tab=advanced&section=concurrency'); }}
            style={{ color: T.accent, cursor: 'pointer', textDecoration: 'underline' }}
          >
            change how many run at once
          </span>.
        </div>
      )}
    </div>
  );
}
