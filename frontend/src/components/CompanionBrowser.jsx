import React, { useEffect, useState, useCallback } from 'react';

// Remote file browser for the paired GPU Companion's shared folders.
//
// Only useful when a Companion is connected AND the user shared folders in the
// Companion app. Lets the user navigate those folders (server-jailed to the
// shared roots) and import a file into ClipAI: a video becomes a queued job,
// media/fonts land in the library. Rendered as a modal overlay.
//
// Props:
//   kind: 'video' | 'media' | 'font'   — what to import + which extensions to show
//   onClose(): void
//   onImported(result): void           — result is the import endpoint's JSON

const VIDEO_EXT = ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'mpg', 'mpeg', 'wmv', 'flv'];
const MEDIA_EXT = ['mp4', 'mov', 'mkv', 'webm', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'mp3', 'wav', 'm4a', 'aac', 'ogg'];
const FONT_EXT = ['ttf', 'otf', 'ttc', 'woff', 'woff2'];

function extsFor(kind) {
  if (kind === 'font') return FONT_EXT;
  if (kind === 'media') return MEDIA_EXT;
  return VIDEO_EXT;
}

function fmtSize(n) {
  if (!n) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export default function CompanionBrowser({ kind = 'video', onClose, onImported }) {
  const exts = extsFor(kind);
  const [companions, setCompanions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [rootsError, setRootsError] = useState('');
  const [hostId, setHostId] = useState('');
  const [cwd, setCwd] = useState('');          // '' = show this host's shared roots
  const [entries, setEntries] = useState([]);
  const [listError, setListError] = useState('');
  const [listing, setListing] = useState(false);
  const [importing, setImporting] = useState('');
  const [importMsg, setImportMsg] = useState('');

  // Load connected companions + their shared roots.
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/settings/providers/companion-files/roots');
        const data = await res.json();
        const comps = (data && data.companions) || [];
        setCompanions(comps);
        const firstOnline = comps.find((c) => c.online && (c.roots || []).length > 0);
        if (firstOnline) setHostId(firstOnline.host_id);
        else if (!comps.length) setRootsError('No Companion connected.');
        else setRootsError(comps.map((c) => c.error).filter(Boolean).join(' · ') || 'No shared folders. Add one in the Companion app.');
      } catch (e) {
        setRootsError(`Could not reach the Companion: ${e}`);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const activeHost = companions.find((c) => c.host_id === hostId);
  const roots = (activeHost && activeHost.roots) || [];

  const listDir = useCallback(async (path) => {
    if (!hostId || !path) { setEntries([]); return; }
    setListing(true);
    setListError('');
    try {
      const res = await fetch(`/api/settings/providers/companion-files/list?host_id=${encodeURIComponent(hostId)}&path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        const t = await res.text();
        setListError(`Could not open folder (${res.status}) ${t.slice(0, 120)}`);
        setEntries([]);
        return;
      }
      const data = await res.json();
      setEntries((data && data.entries) || []);
    } catch (e) {
      setListError(`${e}`);
      setEntries([]);
    } finally {
      setListing(false);
    }
  }, [hostId]);

  useEffect(() => { if (cwd) listDir(cwd); }, [cwd, listDir]);

  const doImport = async (entry) => {
    setImporting(entry.path);
    setImportMsg('');
    try {
      const res = await fetch('/api/settings/providers/companion-files/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host_id: hostId, path: entry.path, kind }),
      });
      const data = await res.json();
      if (!res.ok) {
        setImportMsg(`Import failed: ${data.detail || res.status}`);
        return;
      }
      setImportMsg(`Imported "${entry.name}" ✓`);
      onImported && onImported(data);
    } catch (e) {
      setImportMsg(`Import failed: ${e}`);
    } finally {
      setImporting('');
    }
  };

  const importable = (e) => !e.is_dir && exts.includes((e.ext || '').toLowerCase());

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-panel, #12151b)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-md, 10px)', width: 'min(720px, 96vw)',
          maxHeight: '86vh', display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}
      >
        <div className="row spread" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
          <strong>Import from Companion shared folder</strong>
          <button className="secondary" onClick={onClose} style={{ padding: '2px 10px' }}>✕</button>
        </div>

        <div style={{ padding: '10px 16px', overflow: 'auto' }}>
          {loading ? (
            <div className="muted small">Loading shared folders…</div>
          ) : rootsError && !roots.length ? (
            <div className="muted small">{rootsError}</div>
          ) : (
            <>
              {companions.filter((c) => c.online && (c.roots || []).length).length > 1 && (
                <select value={hostId} onChange={(e) => { setHostId(e.target.value); setCwd(''); }}
                  style={{ marginBottom: 10, width: '100%' }}>
                  {companions.filter((c) => c.online).map((c) => (
                    <option key={c.host_id} value={c.host_id}>{c.name}</option>
                  ))}
                </select>
              )}

              {/* Breadcrumb / up button */}
              <div className="row small" style={{ gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                <button className="secondary" style={{ padding: '2px 8px' }}
                  onClick={() => setCwd('')} title="Back to shared folders">⌂ Shared</button>
                {cwd && (
                  <span className="mono small" style={{ color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis' }} title={cwd}>{cwd}</span>
                )}
              </div>

              {/* Root list (cwd empty) or directory entries */}
              {!cwd ? (
                roots.length ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {roots.map((r) => (
                      <button key={r.path} className="secondary"
                        onClick={() => setCwd(r.path)}
                        disabled={!r.exists}
                        style={{ textAlign: 'left', padding: '8px 10px', opacity: r.exists ? 1 : 0.5 }}
                        title={r.exists ? r.path : `${r.path} (folder not found)`}>
                        📂 {r.name}{!r.exists ? ' (missing)' : ''}
                      </button>
                    ))}
                  </div>
                ) : <div className="muted small">This Companion has no shared folders. Add one in the Companion app.</div>
              ) : listing ? (
                <div className="muted small">Loading…</div>
              ) : listError ? (
                <div className="muted small">{listError}</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {entries.length === 0 && <div className="muted small">Empty folder.</div>}
                  {entries.map((e) => (
                    <div key={e.path} className="row spread"
                      style={{ padding: '6px 8px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-elevated)' }}>
                      {e.is_dir ? (
                        <button className="secondary" onClick={() => setCwd(e.path)}
                          style={{ textAlign: 'left', flex: 1, padding: '2px 6px', background: 'transparent', border: 'none' }}>
                          📁 {e.name}
                        </button>
                      ) : (
                        <span className="small" style={{ flex: 1, opacity: importable(e) ? 1 : 0.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={e.name}>
                          📄 {e.name} <span className="muted mono">{fmtSize(e.size)}</span>
                        </span>
                      )}
                      {!e.is_dir && importable(e) && (
                        <button onClick={() => doImport(e)} disabled={!!importing}
                          style={{ padding: '2px 10px' }}>
                          {importing === e.path ? 'Importing…' : 'Import'}
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {importMsg && (
          <div className="small" style={{ padding: '8px 16px', borderTop: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
            {importMsg}
          </div>
        )}
      </div>
    </div>
  );
}
