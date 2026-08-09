import React, { useState, useEffect, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import useResponsive from '../hooks/useResponsive';
import { exportProjectBlocking, exportProjectWithProgress, importProject } from '../utils/projectBundle';

function formatDuration(seconds) {
  if (!seconds) return '-';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/* "9m 44s" / "1h 03m" — analysis wall-clock, distinct from the video's
   m:ss duration so the two numbers on a card never read as the same thing. */
function formatAnalysisTime(seconds) {
  if (!seconds || seconds <= 0) return '';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

const STATUS_STYLES = {
  queued: { className: 'badge-gray', label: 'Queued' },
  extracting_frames: { className: 'badge-cyan pulse', label: 'Processing' },
  transcribing: { className: 'badge-cyan pulse', label: 'Processing' },
  analyzing_scenes: { className: 'badge-cyan pulse', label: 'Processing' },
  generating_summary: { className: 'badge-cyan pulse', label: 'Processing' },
  translating: { className: 'badge-cyan pulse', label: 'Translating' },
  detecting_clips: { className: 'badge-cyan pulse', label: 'Processing' },
  complete: { className: 'badge-green', label: 'Complete' },
  failed: { className: 'badge-red', label: 'Failed' },
  cancelled: { className: 'badge-amber', label: 'Cancelled' },
};

const CANCELLABLE = [
  'queued', 'extracting_frames', 'transcribing',
  'analyzing_scenes', 'generating_summary', 'translating', 'detecting_clips',
];

export default function Dashboard() {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [cancelling, setCancelling] = useState({});
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [sortBy, setSortBy] = useState('newest');
  // Power-user multi-select for bulk deleting projects.
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  // Card the pointer is over — drives the accent highlight ring.
  const [hoverId, setHoverId] = useState(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const navigate = useNavigate();
  const { isMobile } = useResponsive();

  // ── Project import / export ───────────────────────────────────────────────
  const importInputRef = useRef(null);
  const [importing, setImporting] = useState(false);
  const [importPct, setImportPct] = useState(0);
  const [exportingId, setExportingId] = useState(null);
  const [exportPct, setExportPct] = useState(0);   // -1 = size unknown (indeterminate)
  // Bulk export progress: { done, total } while exporting a multi-selection.
  const [bulkExport, setBulkExport] = useState(null);
  // Delete confirmation modal (single or bulk) with an "export first" option.
  const [deleteModal, setDeleteModal] = useState(null); // { ids: string[] } | null
  const [exportBeforeDelete, setExportBeforeDelete] = useState(true);

  const handleExport = async (e, jobId) => {
    if (e) e.stopPropagation();
    setExportingId(jobId);
    setExportPct(0);
    try {
      await exportProjectWithProgress(jobId, {}, setExportPct);
    } catch (err) {
      window.alert(`Export failed: ${err.message || err}`);
    } finally {
      setExportingId(null);
      setExportPct(0);
    }
  };

  const handleImportPick = () => importInputRef.current?.click();

  const handleImportFile = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setImporting(true);
    setImportPct(0);
    try {
      const res = await importProject(file, setImportPct);
      navigate(`/analysis/${res.job_id}`);
    } catch (err) {
      window.alert(`Import failed: ${err.message || err}`);
    } finally {
      setImporting(false);
      setImportPct(0);
    }
  };

  // Export several selected projects SEQUENTIALLY with a progress bar — one at a
  // time so we never buffer multiple multi-GB bundles at once, and the user sees
  // "Exporting k/N (pct%)" rather than a silent burst of downloads.
  const exportSelected = async () => {
    const ids = Array.from(selected);
    if (!ids.length || bulkExport) return;
    setBulkExport({ done: 0, total: ids.length });
    for (let i = 0; i < ids.length; i++) {
      setExportingId(ids[i]);
      setExportPct(0);
      try {
        await exportProjectWithProgress(ids[i], {}, setExportPct);
      } catch { /* skip a failed one, keep going */ }
      setBulkExport({ done: i + 1, total: ids.length });
    }
    setExportingId(null);
    setExportPct(0);
    setBulkExport(null);
  };

  // Track cancelled job IDs so the 5-second poll never brings them back
  const cancelledIdsRef = React.useRef(new Set());

  const handleCancel = async (e, jobId) => {
    e.stopPropagation();
    if (cancelling[jobId]) return; // debounce
    setCancelling((prev) => ({ ...prev, [jobId]: true }));

    // Mark this ID as cancelled so polls will filter it out
    cancelledIdsRef.current.add(jobId);

    // Optimistically remove from UI immediately
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));

    try {
      await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      // Wait for status to transition then delete the job files
      _waitAndDelete(jobId);
    } catch {
      // If cancel fails, still try to clean up
      _waitAndDelete(jobId);
    }
  };

  // Poll for cancelled status then auto-delete job files
  const _waitAndDelete = async (jobId) => {
    for (let i = 0; i < 15; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) {
          // Already gone — clean up tracking
          cancelledIdsRef.current.delete(jobId);
          setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
          return;
        }
        const job = await res.json();
        if (job.status === 'cancelled' || job.status === 'failed') {
          await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
          cancelledIdsRef.current.delete(jobId);
          setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
          return;
        }
      } catch {
        cancelledIdsRef.current.delete(jobId);
        setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
        return;
      }
    }
    // Timed out — try deleting anyway
    try { await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' }); } catch {}
    cancelledIdsRef.current.delete(jobId);
    setCancelling((prev) => { const n = { ...prev }; delete n[jobId]; return n; });
  };

  const handleDelete = (e, jobId) => {
    if (e) e.stopPropagation();
    // Route single deletes through the same confirm modal (offers export first).
    setExportBeforeDelete(true);
    setDeleteModal({ ids: [jobId] });
  };

  // ── Multi-select bulk delete ──────────────────────────────────────────────
  const toggleSelect = (jobId) => {
    setSelected((prev) => {
      const n = new Set(prev);
      if (n.has(jobId)) n.delete(jobId); else n.add(jobId);
      return n;
    });
  };
  const exitSelectMode = () => { setSelectMode(false); setSelected(new Set()); };
  const bulkDelete = () => {
    const ids = Array.from(selected);
    if (!ids.length) return;
    setExportBeforeDelete(true);
    setDeleteModal({ ids });
  };

  // Runs the confirmed delete. When "export first" is on, each project is
  // exported (and fully saved) BEFORE deletion; any project whose export
  // fails is NOT deleted, so nothing is lost.
  const runDelete = async () => {
    const ids = deleteModal?.ids || [];
    setDeleteModal(null);
    if (!ids.length) return;
    setBulkBusy(true);
    let toDelete = ids;
    if (exportBeforeDelete) {
      const ok = [];
      for (const id of ids) {
        try { await exportProjectBlocking(id); ok.push(id); } catch { /* keep it */ }
      }
      if (ok.length < ids.length) {
        window.alert(`${ids.length - ok.length} export(s) failed — those projects were NOT deleted.`);
      }
      toDelete = ok;
    }
    const CONC = 5;
    for (let i = 0; i < toDelete.length; i += CONC) {
      const batch = toDelete.slice(i, i + CONC);
      await Promise.all(batch.map((id) =>
        fetch(`/api/jobs/${id}`, { method: 'DELETE' }).catch(() => {})));
      setJobs((prev) => prev.filter((j) => !batch.includes(j.job_id)));
    }
    setBulkBusy(false);
    if (selectMode) exitSelectMode();
  };

  const [fetchError, setFetchError] = useState(false);
  const [lastFetchTime, setLastFetchTime] = useState(null);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const res = await fetch('/api/jobs');
        if (res.ok) {
          const data = await res.json();
          // Filter out jobs that are being cancelled — prevents them from reappearing
          const filtered = data.filter((j) => !cancelledIdsRef.current.has(j.job_id));
          setJobs(filtered);
          setFetchError(false);
          setLastFetchTime(Date.now());
        } else {
          setFetchError(true);
        }
      } catch {
        setFetchError(true);
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  // Filter and sort jobs
  let filteredJobs = jobs;
  if (searchQuery.trim()) {
    const q = searchQuery.trim().toLowerCase();
    filteredJobs = filteredJobs.filter((j) =>
      (j.filename || '').toLowerCase().includes(q) ||
      (j.progress_message || '').toLowerCase().includes(q) ||
      (j.job_id || '').toLowerCase().includes(q)
    );
  }
  if (statusFilter !== 'all') {
    if (statusFilter === 'processing') {
      filteredJobs = filteredJobs.filter((j) => !['complete', 'failed', 'queued', 'cancelled'].includes(j.status));
    } else {
      filteredJobs = filteredJobs.filter((j) => j.status === statusFilter);
    }
  }
  filteredJobs = [...filteredJobs].sort((a, b) => {
    if (sortBy === 'oldest') return (a.created_at || '').localeCompare(b.created_at || '');
    if (sortBy === 'name') return (a.filename || '').localeCompare(b.filename || '');
    if (sortBy === 'clips') return (b.clips_count || 0) - (a.clips_count || 0);
    return (b.created_at || '').localeCompare(a.created_at || ''); // newest
  });

  const totalClips = jobs.reduce((sum, j) => sum + (j.clips_count || 0), 0);
  const processingCount = jobs.filter((j) => !['complete', 'failed', 'queued', 'cancelled'].includes(j.status)).length;

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
        <div style={{
          width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--accent-cyan)',
          borderRadius: '50%', animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px',
        }} />
        Connecting to container...
      </div>
    );
  }

  return (
    <div>
      <input
        ref={importInputRef}
        type="file"
        accept=".zip,application/zip"
        onChange={handleImportFile}
        style={{ display: 'none' }}
      />

      {/* Delete confirmation — offers to export the project(s) first. */}
      {deleteModal && (
        <div
          onClick={() => !bulkBusy && setDeleteModal(null)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-lg)', width: 'min(460px, 96vw)', padding: 20,
            }}
          >
            <h3 style={{ margin: '0 0 8px', fontSize: 16 }}>
              Delete {deleteModal.ids.length} project{deleteModal.ids.length === 1 ? '' : 's'}?
            </h3>
            <p style={{ margin: '0 0 14px', fontSize: 13, color: 'var(--text-secondary)' }}>
              This frees server disk and can&rsquo;t be undone. Export first to keep a
              portable <code>.clipai.zip</code> you can re-import later.
            </p>
            <label style={{ display: 'flex', gap: 8, alignItems: 'flex-start', cursor: 'pointer', marginBottom: 16, fontSize: 13 }}>
              <input
                type="checkbox"
                checked={exportBeforeDelete}
                onChange={(e) => setExportBeforeDelete(e.target.checked)}
                style={{ marginTop: 2 }}
              />
              <span>
                <strong>Export before deleting</strong> — download each project as a
                bundle first. A project is only deleted once its export is saved.
              </span>
            </label>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                onClick={() => setDeleteModal(null)}
                disabled={bulkBusy}
                style={{ padding: '8px 14px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 13, cursor: 'pointer' }}
              >
                Cancel
              </button>
              <button
                onClick={runDelete}
                disabled={bulkBusy}
                style={{ padding: '8px 16px', background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 13, fontWeight: 600, cursor: bulkBusy ? 'default' : 'pointer' }}
              >
                {bulkBusy ? (exportBeforeDelete ? 'Exporting…' : 'Deleting…') : (exportBeforeDelete ? 'Export & Delete' : 'Delete')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connection error banner */}
      {fetchError && (
        <div style={{
          padding: '10px 16px', marginBottom: 16,
          background: 'var(--amber-dim)', border: '1px solid var(--accent-amber)',
          borderRadius: 'var(--radius-sm)', fontSize: 13, color: 'var(--accent-amber)',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <span style={{ fontSize: 10, animation: 'pulse 1.5s ease-in-out infinite' }}>{'\u25CF'}</span>
          Unable to reach the backend — retrying automatically...
        </div>
      )}

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: isMobile ? 10 : 16, marginBottom: isMobile ? 20 : 24, flexWrap: 'wrap' }}>
        {[
          { label: 'Total Videos', value: jobs.length, color: 'var(--accent-cyan)' },
          { label: 'Clips Extracted', value: totalClips, color: 'var(--accent-amber)' },
          { label: 'Processing', value: processingCount, color: processingCount > 0 ? 'var(--accent-amber)' : 'var(--text-primary)' },
        ].map(({ label, value, color }) => (
          <div key={label} style={{
            background: 'var(--bg-panel)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', padding: isMobile ? '10px 14px' : '12px 20px',
            flex: 1, minWidth: isMobile ? 0 : 140, boxShadow: 'var(--shadow-sm)',
          }}>
            <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              {label}
            </div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: isMobile ? 22 : 28, fontWeight: 700, color }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Search & Filter Bar */}
      {jobs.length > 0 && (
        <div style={{ marginBottom: selectMode ? 12 : 16, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch' }}>
          {/* Search input */}
          <div style={{ position: 'relative', flex: '1 1 220px', minWidth: 0 }}>
            <div style={{
              position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-muted)', fontSize: 15, pointerEvents: 'none', lineHeight: 1,
            }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search videos by name..."
              style={{
                width: '100%',
                padding: '10px 14px 10px 40px',
                fontSize: 14,
                background: 'var(--bg-panel)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                color: 'var(--text-primary)',
                outline: 'none',
                transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
              }}
              onFocus={(e) => {
                e.target.style.borderColor = 'var(--accent-cyan)';
                e.target.style.boxShadow = '0 0 0 3px var(--accent-cyan-dim)';
              }}
              onBlur={(e) => {
                e.target.style.borderColor = 'var(--border)';
                e.target.style.boxShadow = 'none';
              }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
                  background: 'var(--bg-elevated)', border: 'none', borderRadius: '50%',
                  width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'var(--text-muted)', fontSize: 12, cursor: 'pointer', lineHeight: 1,
                }}
              >
                &times;
              </button>
            )}
          </div>

          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{
              padding: '10px 12px', borderRadius: 'var(--radius-md)', fontSize: 13,
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', outline: 'none', cursor: 'pointer',
            }}
          >
            <option value="all">All Status</option>
            <option value="processing">Processing</option>
            <option value="complete">Complete</option>
            <option value="failed">Failed</option>
            <option value="queued">Queued</option>
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            style={{
              padding: '10px 12px', borderRadius: 'var(--radius-md)', fontSize: 13,
              background: 'var(--bg-panel)', border: '1px solid var(--border)',
              color: 'var(--text-primary)', outline: 'none', cursor: 'pointer',
            }}
          >
            <option value="newest">Newest First</option>
            <option value="oldest">Oldest First</option>
            <option value="name">By Name</option>
            <option value="clips">Most Clips</option>
          </select>

          {/* Multi-select toggle — inline with the search / filter bar. */}
          <button
            onClick={() => (selectMode ? exitSelectMode() : setSelectMode(true))}
            style={{
              padding: '10px 14px', fontSize: 13, cursor: 'pointer',
              borderRadius: 'var(--radius-md)',
              background: selectMode ? 'var(--accent-cyan-dim)' : 'var(--bg-panel)',
              border: `1px solid ${selectMode ? 'var(--accent-cyan)' : 'var(--border)'}`,
              color: selectMode ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              fontWeight: selectMode ? 600 : 400, whiteSpace: 'nowrap',
            }}
            title="Select multiple projects to delete at once"
          >
            {selectMode ? 'Done' : 'Multi-select'}
          </button>

          {/* Import a .clipai.zip project bundle. */}
          {!selectMode && (
            <button
              onClick={handleImportPick}
              disabled={importing}
              style={{
                padding: '10px 14px', fontSize: 13, borderRadius: 'var(--radius-md)',
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                color: 'var(--text-secondary)', whiteSpace: 'nowrap',
                cursor: importing ? 'default' : 'pointer', opacity: importing ? 0.7 : 1,
              }}
              title="Import a .clipai.zip project bundle"
            >
              {importing ? `Importing… ${importPct}%` : 'Import Project'}
            </button>
          )}
        </div>
      )}

      {/* Bulk-action bar — appears once multi-select is on. */}
      {jobs.length > 0 && selectMode && (
        <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)', marginRight: 4 }}>
            {selected.size} selected
          </span>
          <button
            onClick={() => setSelected(new Set(filteredJobs.map((j) => j.job_id)))}
            style={{ padding: '8px 12px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 13, cursor: 'pointer' }}
          >
            Select all
          </button>
          <button
            onClick={() => setSelected(new Set())}
            style={{ padding: '8px 12px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 13, cursor: 'pointer' }}
          >
            Clear
          </button>
          <button
            onClick={exportSelected}
            disabled={selected.size === 0 || bulkBusy || !!bulkExport}
            style={{ position: 'relative', overflow: 'hidden', padding: '8px 12px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 13, cursor: selected.size > 0 && !bulkBusy && !bulkExport ? 'pointer' : 'default', minWidth: bulkExport ? 180 : undefined }}
            title="Download each selected project as a .clipai.zip"
          >
            {bulkExport && (
              <span
                aria-hidden="true"
                style={{
                  position: 'absolute', left: 0, top: 0, bottom: 0,
                  width: `${Math.round(((bulkExport.done + (exportPct > 0 ? exportPct / 100 : 0)) / Math.max(1, bulkExport.total)) * 100)}%`,
                  background: 'var(--accent-cyan-dim, rgba(56,189,248,0.25))',
                  transition: 'width 0.2s ease',
                }}
              />
            )}
            <span style={{ position: 'relative' }}>
              {bulkExport
                ? `Exporting ${Math.min(bulkExport.done + 1, bulkExport.total)}/${bulkExport.total}${exportPct >= 0 ? ` (${exportPct}%)` : '…'}`
                : `Export${selected.size ? ` (${selected.size})` : ''}`}
            </span>
          </button>
          <button
            onClick={bulkDelete}
            disabled={selected.size === 0 || bulkBusy}
            style={{
              padding: '8px 14px', background: selected.size > 0 ? 'var(--danger)' : 'var(--bg-elevated)',
              color: selected.size > 0 ? '#fff' : 'var(--text-muted)', border: 'none',
              borderRadius: 'var(--radius-sm)', fontSize: 13, fontWeight: 600,
              cursor: selected.size > 0 && !bulkBusy ? 'pointer' : 'default',
            }}
          >
            {bulkBusy ? 'Deleting…' : `Delete${selected.size ? ` (${selected.size})` : ''}`}
          </button>
        </div>
      )}

      {/* Job list or empty state */}
      {jobs.length === 0 ? (
        <div
          style={{
            textAlign: 'center',
            padding: isMobile ? '48px 20px' : '80px 24px',
            border: '2px dashed var(--border)',
            borderRadius: 'var(--radius-lg)',
          }}
        >
          <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>&#x1F3AC;</div>
          <h3 style={{ fontSize: 18, marginBottom: 8, color: 'var(--text-secondary)' }}>
            No videos yet
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
            Drop your first video to get started
          </p>
          <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link
              to="/upload"
              style={{
                display: 'inline-block',
                padding: '10px 24px',
                background: 'var(--accent-cyan)',
                color: 'var(--bg-base)',
                fontWeight: 600,
                borderRadius: 'var(--radius-sm)',
                textDecoration: 'none',
              }}
            >
              Upload Video
            </Link>
            <button
              onClick={handleImportPick}
              disabled={importing}
              style={{
                padding: '10px 24px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontWeight: 600,
                cursor: importing ? 'default' : 'pointer', opacity: importing ? 0.7 : 1,
              }}
              title="Import a .clipai.zip project bundle"
            >
              {importing ? `Importing… ${importPct}%` : 'Import Project'}
            </button>
          </div>
        </div>
      ) : filteredJobs.length === 0 ? (
        <div style={{
          textAlign: 'center', padding: '48px 24px',
          color: 'var(--text-muted)', fontSize: 14,
        }}>
          {searchQuery.trim()
            ? `No videos match "${searchQuery.trim()}".`
            : 'No videos match the selected filter.'}
        </div>
      ) : (
        <div
          className="responsive-grid"
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(min(320px, 100%), 1fr))',
            gap: isMobile ? 12 : 16,
          }}
        >
          {filteredJobs.map((job, i) => {
            const statusInfo = STATUS_STYLES[job.status] || STATUS_STYLES.queued;
            const isSel = selected.has(job.job_id);
            return (
              <div
                key={job.job_id}
                className="card-hover slide-in"
                onClick={() => (selectMode ? toggleSelect(job.job_id) : navigate(`/analysis/${job.job_id}`))}
                onMouseEnter={() => setHoverId(job.job_id)}
                onMouseLeave={() => setHoverId((h) => (h === job.job_id ? null : h))}
                style={{
                  background: 'var(--bg-panel)',
                  // Hovering highlights the card with the same accent ring the
                  // multi-select checkmark uses, so "this card is live" reads
                  // identically everywhere.
                  border: `1px solid ${isSel || hoverId === job.job_id
                    ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  boxShadow: isSel || hoverId === job.job_id
                    ? '0 0 0 2px var(--accent-cyan-dim)' : 'var(--shadow-sm)',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  position: 'relative',
                  animationDelay: `${i * 50}ms`,
                }}
              >
                {selectMode && (
                  <div style={{
                    position: 'absolute', top: 10, right: 10, zIndex: 2,
                    width: 22, height: 22, borderRadius: '50%',
                    border: `2px solid ${isSel ? 'var(--accent-cyan)' : 'var(--border)'}`,
                    background: isSel ? 'var(--accent-cyan)' : 'var(--bg-panel)',
                    color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, fontWeight: 700,
                  }}>
                    {isSel ? '✓' : ''}
                  </div>
                )}
                {/* Progress bar for active jobs */}
                {job.progress > 0 && job.progress < 100 && (
                  <div style={{ height: 2, background: 'var(--bg-elevated)' }}>
                    <div
                      className="shimmer"
                      style={{ height: '100%', width: `${job.progress}%`, background: 'var(--accent-cyan)' }}
                    />
                  </div>
                )}

                <div style={{ padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <div>
                      <h4 style={{ fontSize: 14, marginBottom: 4, wordBreak: 'break-word' }}>
                        {String(job.filename || '')}
                      </h4>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        {formatDate(job.created_at)}
                      </span>
                    </div>
                    <span className={`badge ${statusInfo.className}`}>
                      {statusInfo.label}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: 16, fontSize: 12, color: 'var(--text-secondary)', flexWrap: 'wrap' }}>
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {formatDuration(job.duration)}
                    </span>
                    {job.file_size_mb > 0 && (
                      <span style={{ fontFamily: 'var(--font-mono)' }}>
                        {job.file_size_mb.toFixed(1)}MB
                      </span>
                    )}
                    {job.clips_count > 0 && (
                      <span style={{ color: 'var(--accent-amber)', fontFamily: 'var(--font-mono)' }}>
                        {job.clips_count} clips
                      </span>
                    )}
                    {job.status === 'complete' && formatAnalysisTime(job.analysis_duration_seconds) && (
                      <span
                        style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}
                        title="Total analysis time (upload to complete)"
                      >
                        ⏱ {formatAnalysisTime(job.analysis_duration_seconds)}
                      </span>
                    )}
                  </div>

                  {job.progress_message && job.status !== 'complete' && (
                    <div style={{ marginTop: 8, fontSize: 11, color: 'var(--accent-cyan)' }}>
                      {String(job.progress_message || '')}
                    </div>
                  )}

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                    {!CANCELLABLE.includes(job.status) && (
                      <button
                        onClick={(e) => handleExport(e, job.job_id)}
                        disabled={exportingId === job.job_id}
                        style={{
                          position: 'relative', overflow: 'hidden',
                          padding: '6px 14px', background: 'var(--bg-elevated)',
                          border: '1px solid var(--border)', color: 'var(--text-secondary)',
                          fontSize: 12, fontWeight: 600,
                          cursor: exportingId === job.job_id ? 'default' : 'pointer',
                          borderRadius: 'var(--radius-sm)', opacity: exportingId === job.job_id ? 0.85 : 1,
                          minWidth: exportingId === job.job_id ? 128 : undefined,
                        }}
                        title="Download this project as a .clipai.zip you can re-import later"
                      >
                        {exportingId === job.job_id && (
                          <span
                            aria-hidden="true"
                            style={{
                              position: 'absolute', left: 0, top: 0, bottom: 0,
                              width: exportPct < 0 ? '100%' : `${exportPct}%`,
                              background: 'var(--accent-cyan-dim, rgba(56,189,248,0.25))',
                              transition: 'width 0.2s ease',
                              opacity: exportPct < 0 ? 0.4 : 1,
                            }}
                          />
                        )}
                        <span style={{ position: 'relative' }}>
                          {exportingId === job.job_id
                            ? (exportPct < 0 ? 'Exporting…' : `Exporting… ${exportPct}%`)
                            : 'Export'}
                        </span>
                      </button>
                    )}
                    {CANCELLABLE.includes(job.status) && (
                      <button
                        onClick={(e) => handleCancel(e, job.job_id)}
                        disabled={cancelling[job.job_id]}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--amber-dim)',
                          border: '1px solid var(--accent-amber)',
                          color: 'var(--accent-amber)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: cancelling[job.job_id] ? 'default' : 'pointer',
                          borderRadius: 'var(--radius-sm)',
                          opacity: cancelling[job.job_id] ? 0.6 : 1,
                        }}
                      >
                        {cancelling[job.job_id] ? 'Cancelling...' : 'Cancel'}
                      </button>
                    )}
                    {(job.status === 'failed' || job.status === 'cancelled') && (
                      <button
                        onClick={(e) => handleDelete(e, job.job_id)}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--danger-dim)',
                          border: '1px solid var(--danger)',
                          color: 'var(--danger)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: 'pointer',
                          borderRadius: 'var(--radius-sm)',
                        }}
                      >
                        Remove
                      </button>
                    )}
                    {job.status === 'complete' && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm(`Delete "${job.filename}" and all its clips? This cannot be undone.`)) {
                            handleDelete(e, job.job_id);
                          }
                        }}
                        style={{
                          padding: '6px 14px',
                          background: 'var(--danger-dim)',
                          border: '1px solid var(--danger)',
                          color: 'var(--danger)',
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: 'pointer',
                          borderRadius: 'var(--radius-sm)',
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
