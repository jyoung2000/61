import React, { useEffect, useState, useCallback, useMemo } from 'react';

// Finder + Spotlight-inspired remote file browser for a paired GPU Companion's
// shared folders. Browse (server-jailed to the shared roots), search/paste a
// path, sort, preview thumbnails, and import a file into ClipAI: a video
// becomes a queued job (with a live progress bar while it downloads over the
// LAN); media/fonts land in the library.
//
// Props:
//   kind: 'video' | 'media' | 'font'
//   onClose(): void
//   onImported(result): void

const VIDEO_EXT = ['mp4', 'mov', 'mkv', 'avi', 'webm', 'm4v', 'mpg', 'mpeg', 'wmv', 'flv'];
const IMAGE_EXT = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'heic'];
const AUDIO_EXT = ['mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac'];
const FONT_EXT = ['ttf', 'otf', 'ttc', 'woff', 'woff2'];
const MEDIA_EXT = [...VIDEO_EXT, ...IMAGE_EXT, ...AUDIO_EXT];

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
function fmtDate(ms) {
  if (!ms) return '';
  try { return new Date(ms).toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' }); }
  catch { return ''; }
}
function iconFor(e) {
  if (e.is_dir) return '📁';
  const x = (e.ext || '').toLowerCase();
  if (VIDEO_EXT.includes(x)) return '🎬';
  if (IMAGE_EXT.includes(x)) return '🖼️';
  if (AUDIO_EXT.includes(x)) return '🎵';
  if (FONT_EXT.includes(x)) return '🔤';
  return '📄';
}
const looksLikePath = (s) => /^([a-zA-Z]:[\\/]|\/|\\\\)/.test((s || '').trim());

const SORTS = [
  { key: 'az', label: 'Name (A–Z)' },
  { key: 'za', label: 'Name (Z–A)' },
  { key: 'size', label: 'Size (largest)' },
  { key: 'modified', label: 'Date modified' },
  { key: 'created', label: 'Date created' },
  { key: 'type', label: 'Type' },
];

// A thumbnail that loads via the backend ffmpeg endpoint; falls back to a type
// icon on error / non-previewable files.
function Thumb({ entry, hostId }) {
  const x = (entry.ext || '').toLowerCase();
  const previewable = !entry.is_dir && (IMAGE_EXT.includes(x) || VIDEO_EXT.includes(x));
  const [failed, setFailed] = useState(false);
  const box = {
    width: 44, height: 44, flexShrink: 0, borderRadius: 8, overflow: 'hidden',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'var(--bg-elevated)', fontSize: 22,
  };
  if (previewable && !failed) {
    const src = `/api/providers/companion-files/thumb?host_id=${encodeURIComponent(hostId)}&path=${encodeURIComponent(entry.path)}&v=${entry.mtime_ms || ''}`;
    return (
      <div style={box}>
        <img src={src} loading="lazy" alt="" onError={() => setFailed(true)}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
    );
  }
  return <div style={box}>{iconFor(entry)}</div>;
}

export default function CompanionBrowser({ kind = 'video', onClose, onImported }) {
  const exts = extsFor(kind);
  const [companions, setCompanions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [rootsError, setRootsError] = useState('');
  const [hostId, setHostId] = useState('');
  const [cwd, setCwd] = useState('');
  const [entries, setEntries] = useState([]);
  const [listError, setListError] = useState('');
  const [listing, setListing] = useState(false);
  const [query, setQuery] = useState('');
  const [sortBy, setSortBy] = useState('az');
  const [importingPath, setImportingPath] = useState('');
  const [importPct, setImportPct] = useState(null);
  const [importMsg, setImportMsg] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/api/providers/companion-files/roots');
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
    setListing(true); setListError('');
    try {
      const res = await fetch(`/api/providers/companion-files/list?host_id=${encodeURIComponent(hostId)}&path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        const t = await res.text();
        setListError(`Could not open folder (${res.status}) ${t.slice(0, 140)}`);
        setEntries([]); return;
      }
      const data = await res.json();
      setEntries((data && data.entries) || []);
    } catch (e) {
      setListError(`${e}`); setEntries([]);
    } finally {
      setListing(false);
    }
  }, [hostId]);

  useEffect(() => { if (cwd) listDir(cwd); }, [cwd, listDir]);
  const goTo = (path) => { setQuery(''); setCwd(path); };

  // Sort (dirs first, always A–Z) then filter by the live query.
  const shown = useMemo(() => {
    const dirs = entries.filter((e) => e.is_dir);
    const files = entries.filter((e) => !e.is_dir);
    const byName = (a, b) => (a.name || '').toLowerCase().localeCompare((b.name || '').toLowerCase());
    const sorters = {
      az: byName,
      za: (a, b) => byName(b, a),
      size: (a, b) => (b.size || 0) - (a.size || 0),
      modified: (a, b) => (b.mtime_ms || 0) - (a.mtime_ms || 0),
      created: (a, b) => (b.created_ms || 0) - (a.created_ms || 0),
      type: (a, b) => ((a.ext || '').localeCompare(b.ext || '')) || byName(a, b),
    };
    dirs.sort(byName);
    files.sort(sorters[sortBy] || byName);
    const all = [...dirs, ...files];
    const q = query.trim().toLowerCase();
    return q && !looksLikePath(query) ? all.filter((e) => (e.name || '').toLowerCase().includes(q)) : all;
  }, [entries, sortBy, query]);

  const importable = (e) => !e.is_dir && exts.includes((e.ext || '').toLowerCase());

  const pollImport = (importId, entry) => {
    setImportPct(0);
    const tick = async () => {
      try {
        const r = await fetch(`/api/providers/companion-files/import-progress?import_id=${encodeURIComponent(importId)}`);
        if (!r.ok) { setImportingPath(''); return; }
        const p = await r.json();
        const total = p.total || entry.size || 0;
        setImportPct(total > 0 ? Math.min(100, Math.round((p.done / total) * 100)) : null);
        if (p.status === 'complete') {
          setImportingPath(''); setImportMsg(`Imported “${entry.name}” ✓`);
          onImported && onImported({ kind: 'video', ok: true, job_id: p.job_id });
          return;
        }
        if (p.status === 'error') { setImportingPath(''); setImportMsg(`Import failed: ${p.error}`); return; }
        setTimeout(tick, 500);
      } catch { setTimeout(tick, 900); }
    };
    tick();
  };

  const doImport = async (entry) => {
    setImportingPath(entry.path); setImportMsg(''); setImportPct(null);
    try {
      const res = await fetch('/api/providers/companion-files/import', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ host_id: hostId, path: entry.path, kind, size: entry.size || 0 }),
      });
      const data = await res.json();
      if (!res.ok) { setImportMsg(`Import failed: ${data.detail || res.status}`); setImportingPath(''); return; }
      if (kind === 'video' && data.import_id) {
        pollImport(data.import_id, entry);
      } else {
        setImportingPath(''); setImportMsg(`Imported “${entry.name}” ✓`);
        onImported && onImported(data);
      }
    } catch (e) {
      setImportMsg(`Import failed: ${e}`); setImportingPath('');
    }
  };

  const onSearchKey = (e) => {
    if (e.key === 'Enter' && looksLikePath(query)) goTo(query.trim());
  };

  const C = 'var(--border)';
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(3px)',
      zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-panel, #14171d)', border: `1px solid ${C}`,
        borderRadius: 16, width: 'min(1080px, 96vw)', height: 'min(760px, 88vh)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        boxShadow: '0 24px 64px rgba(0,0,0,0.45)',
      }}>
        {/* Title bar */}
        <div className="row spread" style={{ padding: '12px 16px', borderBottom: `1px solid ${C}` }}>
          <strong style={{ fontSize: 14 }}>Import from Companion</strong>
          <button className="secondary" onClick={onClose} style={{ padding: '2px 10px' }}>✕</button>
        </div>

        {/* Spotlight-style search */}
        <div style={{ padding: '14px 16px 8px' }}>
          <div className="row" style={{
            gap: 8, background: 'var(--bg-elevated)', border: `1px solid ${C}`,
            borderRadius: 12, padding: '10px 14px',
          }}>
            <span style={{ fontSize: 16, opacity: 0.6 }}>🔍</span>
            <input
              autoFocus
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={onSearchKey}
              placeholder="Search this folder, or paste a full path and press Enter…"
              style={{
                flex: 1, minWidth: 0, background: 'transparent', border: 'none',
                outline: 'none', color: 'var(--text)', fontSize: 15,
              }}
            />
            {looksLikePath(query) && (
              <button onClick={() => goTo(query.trim())} style={{ padding: '4px 12px' }}>Go</button>
            )}
          </div>
        </div>

        {/* Toolbar: host + breadcrumb + sort */}
        <div className="row spread" style={{ padding: '0 16px 10px', gap: 8, flexWrap: 'wrap' }}>
          <div className="row" style={{ gap: 8, minWidth: 0, flexWrap: 'wrap' }}>
            {companions.filter((c) => c.online && (c.roots || []).length).length > 1 && (
              <select value={hostId} onChange={(e) => { setHostId(e.target.value); setCwd(''); }}>
                {companions.filter((c) => c.online).map((c) => (
                  <option key={c.host_id} value={c.host_id}>{c.name}</option>
                ))}
              </select>
            )}
            <button className="secondary" style={{ padding: '4px 10px' }} onClick={() => goTo('')} title="Shared folders">⌂ Shared</button>
            {cwd && (
              <span className="mono" style={{ fontSize: 12, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 380 }} title={cwd}>{cwd}</span>
            )}
          </div>
          {cwd && (
            <label className="row" style={{ gap: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
              Sort
              <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                {SORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
              </select>
            </label>
          )}
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: 'auto', padding: '0 16px 12px' }}>
          {loading ? (
            <div className="muted" style={{ padding: 24 }}>Loading shared folders…</div>
          ) : rootsError && !roots.length ? (
            <div className="muted" style={{ padding: 24 }}>{rootsError}</div>
          ) : !cwd ? (
            roots.length ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {roots.map((r) => (
                  <button key={r.path} className="secondary" onClick={() => goTo(r.path)} disabled={!r.exists}
                    style={{ textAlign: 'left', padding: '12px 12px', display: 'flex', gap: 12, alignItems: 'center', opacity: r.exists ? 1 : 0.5 }}
                    title={r.exists ? r.path : `${r.path} (folder not found)`}>
                    <span style={{ fontSize: 22 }}>📁</span>
                    <span style={{ minWidth: 0 }}>
                      <div style={{ fontWeight: 600 }}>{r.name}{!r.exists ? ' (missing)' : ''}</div>
                      <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.path}</div>
                    </span>
                  </button>
                ))}
              </div>
            ) : <div className="muted" style={{ padding: 24 }}>This Companion has no shared folders.</div>
          ) : listing ? (
            <div className="muted" style={{ padding: 24 }}>Loading…</div>
          ) : listError ? (
            <div className="muted" style={{ padding: 24 }}>{listError}</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {shown.length === 0 && <div className="muted" style={{ padding: 24 }}>{query ? 'No matches.' : 'Empty folder.'}</div>}
              {shown.map((e) => {
                const isImporting = importingPath === e.path;
                return (
                  <div key={e.path} style={{
                    display: 'flex', gap: 12, alignItems: 'center', padding: '8px 10px',
                    borderRadius: 10, background: 'var(--bg-elevated)',
                  }}>
                    <Thumb entry={e} hostId={hostId} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {e.is_dir ? (
                        <button onClick={() => goTo(e.path)} style={{
                          background: 'transparent', border: 'none', color: 'var(--text)',
                          padding: 0, textAlign: 'left', fontSize: 14, fontWeight: 600, cursor: 'pointer',
                        }}>{e.name}</button>
                      ) : (
                        <div style={{ fontSize: 14, overflowWrap: 'anywhere' }}>{e.name}</div>
                      )}
                      <div className="mono" style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                        {e.is_dir ? 'Folder' : fmtSize(e.size)}{!e.is_dir && (e.mtime_ms ? ` · ${fmtDate(e.mtime_ms)}` : '')}
                      </div>
                      {isImporting && (
                        <div style={{ marginTop: 6 }}>
                          <div style={{ height: 5, borderRadius: 3, background: 'var(--border)', overflow: 'hidden' }}>
                            <div style={{
                              height: '100%', width: importPct == null ? '35%' : `${importPct}%`,
                              background: 'var(--accent-cyan, #37b6ff)', transition: 'width 0.3s',
                              animation: importPct == null ? 'indet 1.1s ease-in-out infinite' : 'none',
                            }} />
                          </div>
                          <div className="small" style={{ color: 'var(--text-secondary)', marginTop: 3 }}>
                            {importPct == null ? 'Importing…' : `Importing… ${importPct}%`}
                          </div>
                        </div>
                      )}
                    </div>
                    {e.is_dir ? (
                      <span style={{ flexShrink: 0, opacity: 0.4, fontSize: 18 }}>›</span>
                    ) : importable(e) ? (
                      <button onClick={() => doImport(e)} disabled={!!importingPath} style={{ flexShrink: 0, padding: '6px 14px' }}>
                        {isImporting ? '…' : 'Import'}
                      </button>
                    ) : (
                      <span className="small muted" style={{ flexShrink: 0 }}>—</span>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {importMsg && (
          <div className="small" style={{ padding: '10px 16px', borderTop: `1px solid ${C}`, color: 'var(--text-secondary)' }}>
            {importMsg}
          </div>
        )}
      </div>
    </div>
  );
}
