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

// ── Line-icon kit (matches the Cloud File Picker design comp) ────────────────
const _sv = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' };
const Svg = ({ s = 18, fill, sw, children }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" {..._sv} strokeWidth={sw ?? 1.8} fill={fill || 'none'} style={{ display: 'block', flex: 'none' }}>{children}</svg>
);
const Ic = {
  Search: (p) => <Svg {...p}><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></Svg>,
  X: (p) => <Svg {...p}><path d="M18 6 6 18" /><path d="m6 6 12 12" /></Svg>,
  ChevR: (p) => <Svg {...p}><path d="m9 18 6-6-6-6" /></Svg>,
  ChevL: (p) => <Svg {...p}><path d="m15 18-6-6 6-6" /></Svg>,
  Check: (p) => <Svg {...p}><path d="M20 6 9 17l-5-5" /></Svg>,
  List: (p) => <Svg {...p}><path d="M8 6h13" /><path d="M8 12h13" /><path d="M8 18h13" /><path d="M3 6h.01" /><path d="M3 12h.01" /><path d="M3 18h.01" /></Svg>,
  Grid: (p) => <Svg {...p}><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /></Svg>,
  Download: (p) => <Svg {...p}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m7 10 5 5 5-5" /><path d="M12 15V3" /></Svg>,
  Folder: (p) => <Svg {...p}><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z" /></Svg>,
  Film: (p) => <Svg {...p}><rect x="2" y="3" width="20" height="18" rx="2" /><path d="M7 3v18" /><path d="M17 3v18" /><path d="M2 12h20" /><path d="M2 7.5h5" /><path d="M2 16.5h5" /><path d="M17 7.5h5" /><path d="M17 16.5h5" /></Svg>,
  Image: (p) => <Svg {...p}><rect x="3" y="3" width="18" height="18" rx="2" /><circle cx="9" cy="9" r="2" /><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21" /></Svg>,
  Music: (p) => <Svg {...p}><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></Svg>,
  FileText: (p) => <Svg {...p}><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z" /><path d="M14 2v5h5" /><path d="M9 13h6" /><path d="M9 17h6" /></Svg>,
  HardDrive: (p) => <Svg {...p}><path d="M22 12H2" /><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" /><path d="M6 16h.01" /><path d="M10 16h.01" /></Svg>,
  Home: (p) => <Svg {...p}><path d="M3 9.5 12 3l9 6.5" /><path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10" /></Svg>,
};

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
function iconFor(e, size = 20) {
  if (e.is_dir) return <Ic.Folder s={size} />;
  const x = (e.ext || '').toLowerCase();
  if (VIDEO_EXT.includes(x)) return <Ic.Film s={size} />;
  if (IMAGE_EXT.includes(x)) return <Ic.Image s={size} />;
  if (AUDIO_EXT.includes(x)) return <Ic.Music s={size} />;
  if (FONT_EXT.includes(x)) return <Ic.FileText s={size} />;
  return <Ic.FileText s={size} />;
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
// icon on error / non-previewable files. ``variant`` picks the design-comp box:
// 'list' = 40×40 rounded tile, 'grid' = 16:10 cover fill.
function Thumb({ entry, hostId, variant = 'list' }) {
  const x = (entry.ext || '').toLowerCase();
  const previewable = !entry.is_dir && (IMAGE_EXT.includes(x) || VIDEO_EXT.includes(x));
  const [failed, setFailed] = useState(false);
  const isGrid = variant === 'grid';
  const box = isGrid
    ? { position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--fb-elev)', color: 'var(--fb-tm)' }
    : { width: 40, height: 40, flexShrink: 0, borderRadius: 9, overflow: 'hidden',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: entry.is_dir ? 'transparent' : 'var(--fb-elev)',
        color: entry.is_dir ? 'var(--fb-accent)' : 'var(--fb-ts)' };
  if (previewable && !failed) {
    const src = `/api/providers/companion-files/thumb?host_id=${encodeURIComponent(hostId)}&path=${encodeURIComponent(entry.path)}&v=${entry.mtime_ms || ''}`;
    return (
      <div style={box}>
        <img src={src} loading="lazy" alt="" onError={() => setFailed(true)}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
      </div>
    );
  }
  return <div style={box}>{iconFor(entry, isGrid ? 30 : 20)}</div>;
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
  const [selectedPaths, setSelectedPaths] = useState(() => new Set());
  const [batchMsg, setBatchMsg] = useState('');
  const [view, setView] = useState('list'); // 'list' | 'grid'

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

  // Breadcrumb trail from the current path: "⌂ Shared" → root → sub-folders.
  // Each crumb carries the absolute path it navigates to (goTo). Robust to
  // both POSIX (/) and Windows (\\) separators.
  const crumbs = useMemo(() => {
    if (!cwd) return [];
    const root = roots.find((r) => cwd === r.path
      || cwd.startsWith(r.path + '/') || cwd.startsWith(r.path + '\\'));
    const out = [];
    let base = '';
    let rest = cwd;
    if (root) {
      out.push({ name: root.name, path: root.path });
      base = root.path;
      rest = cwd.slice(root.path.length).replace(/^[\\/]+/, '');
    }
    if (rest) {
      const sep = (base + cwd).includes('\\') && !(base + cwd).includes('/') ? '\\'
        : (cwd.includes('/') ? '/' : '\\');
      const parts = rest.split(/[\\/]+/).filter(Boolean);
      let acc = base;
      for (const part of parts) {
        acc = acc ? acc + sep + part : part;
        out.push({ name: part, path: acc });
      }
    }
    return out;
  }, [cwd, roots]);

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

  // Import one file end-to-end. Resolves to {job_id} (video), the media/font
  // result, or {error}. Drives importingPath/importPct for the active row.
  const importOne = (entry) => new Promise((resolve) => {
    setImportingPath(entry.path); setImportPct(null);
    (async () => {
      try {
        const res = await fetch('/api/providers/companion-files/import', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ host_id: hostId, path: entry.path, kind, size: entry.size || 0 }),
        });
        const data = await res.json();
        if (!res.ok) { resolve({ error: data.detail || res.status }); return; }
        if (kind === 'video' && data.import_id) {
          const poll = async () => {
            try {
              const r = await fetch(`/api/providers/companion-files/import-progress?import_id=${encodeURIComponent(data.import_id)}`);
              if (!r.ok) { resolve({ job_id: data.job_id }); return; }  // popped = done
              const p = await r.json();
              const total = p.total || entry.size || 0;
              setImportPct(total > 0 ? Math.min(100, Math.round((p.done / total) * 100)) : null);
              if (p.status === 'complete') { resolve({ job_id: p.job_id }); return; }
              if (p.status === 'error') { resolve({ error: p.error }); return; }
              setTimeout(poll, 500);
            } catch { setTimeout(poll, 900); }
          };
          poll();
        } else {
          resolve(data);
        }
      } catch (e) { resolve({ error: String(e) }); }
    })();
  });

  const doImport = async (entry) => {
    setImportMsg('');
    const r = await importOne(entry);
    setImportingPath('');
    if (r && r.error) { setImportMsg(`Import failed: ${r.error}`); return; }
    setImportMsg(`Imported “${entry.name}” ✓`);
    onImported && onImported(kind === 'video' ? { kind: 'video', ok: true, job_id: r.job_id } : r);
  };

  const toggleSel = (path) => setSelectedPaths((prev) => {
    const n = new Set(prev);
    if (n.has(path)) n.delete(path); else n.add(path);
    return n;
  });

  const importSelected = async () => {
    const picked = shown.filter((e) => selectedPaths.has(e.path) && importable(e));
    if (!picked.length) return;
    setImportMsg('');
    let ok = 0; let firstJob = null; const mediaResults = [];
    for (let i = 0; i < picked.length; i++) {
      setBatchMsg(`Importing ${i + 1} of ${picked.length}…`);
      const r = await importOne(picked[i]);        // sequential — steady on the LAN
      if (r && !r.error) {
        ok += 1;
        if (kind === 'video') { if (!firstJob) firstJob = r.job_id; }
        else mediaResults.push(r);
      }
    }
    setImportingPath(''); setBatchMsg(''); setSelectedPaths(new Set());
    setImportMsg(`Imported ${ok} of ${picked.length}`);
    if (kind === 'video' && firstJob) onImported && onImported({ kind: 'video', ok: true, job_id: firstJob });
    else mediaResults.forEach((r) => onImported && onImported(r));
  };
  const busy = !!importingPath || !!batchMsg;

  const onSearchKey = (e) => {
    if (e.key === 'Enter' && looksLikePath(query)) goTo(query.trim());
  };

  // ── Design tokens (mapped onto the app's theme vars so light + dark both
  //    look right; children read these via var(--fb-*)) ──
  const TT = {
    '--fb-panel': 'var(--bg-panel, rgba(28,28,30,0.92))',
    '--fb-elev': 'var(--bg-elevated, rgba(255,255,255,0.06))',
    '--fb-border': 'var(--border, rgba(255,255,255,0.1))',
    '--fb-border-strong': 'var(--border-strong, rgba(255,255,255,0.22))',
    '--fb-accent': 'var(--accent-cyan, #0A84FF)',
    '--fb-accent-dim': 'var(--accent-cyan-dim, rgba(10,132,255,0.16))',
    '--fb-tp': 'var(--text-primary, #F5F5F7)',
    '--fb-ts': 'var(--text-secondary, rgba(235,235,245,0.68))',
    '--fb-tm': 'var(--text-muted, rgba(235,235,245,0.42))',
    '--fb-sel': 'var(--accent-cyan-dim, rgba(10,132,255,0.2))',
    '--fb-mono': 'var(--font-mono, ui-monospace, monospace)',
    '--fb-track': 'var(--border, rgba(255,255,255,0.16))',
    '--fb-danger': 'var(--danger, #FF453A)',
    '--fb-danger-dim': 'var(--danger-dim, rgba(255,59,48,0.14))',
  };
  const onlineHosts = companions.filter((c) => c.online && (c.roots || []).length);
  const statusLine = activeHost
    ? `${roots.length} shared folder${roots.length === 1 ? '' : 's'} · ${activeHost.online ? 'online' : 'offline'}`
    : (loading ? 'Connecting…' : 'No Companion connected');
  const selCount = selectedPaths.size;

  const field = {
    display: 'flex', alignItems: 'center', gap: 9, height: 42, padding: '0 12px',
    borderRadius: 11, background: 'var(--fb-elev)', border: '1px solid var(--fb-border)',
  };
  const importPill = (active) => ({
    display: 'inline-flex', alignItems: 'center', gap: 5, flex: 'none', cursor: 'pointer',
    fontSize: 13, fontWeight: 600, padding: '7px 13px', borderRadius: 9, border: 'none',
    color: 'var(--fb-accent)', background: 'var(--fb-accent-dim)',
    opacity: active ? 0.6 : 1, transition: 'filter 120ms, transform 120ms',
  });

  const rootRow = (r) => (
    <button key={r.path} onClick={() => r.exists && goTo(r.path)} disabled={!r.exists}
      title={r.exists ? r.path : `${r.path} (folder not found)`}
      style={{
        display: 'flex', alignItems: 'center', gap: 12, minHeight: 56, padding: '0 10px',
        borderRadius: 11, border: 'none', background: 'transparent', cursor: r.exists ? 'pointer' : 'default',
        textAlign: 'left', width: '100%', color: 'var(--fb-tp)', opacity: r.exists ? 1 : 0.55,
      }}>
      <span style={{ width: 40, height: 40, flex: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fb-accent)' }}>
        <Ic.Folder s={22} />
      </span>
      <span style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <span style={{ fontSize: 15, fontWeight: 500, color: 'var(--fb-tp)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.name}</span>
        <span style={{ fontFamily: 'var(--fb-mono)', fontSize: 12, color: 'var(--fb-tm)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.path}</span>
      </span>
      {r.exists
        ? <span style={{ color: 'var(--fb-tm)', display: 'flex', flex: 'none' }}><Ic.ChevR s={17} /></span>
        : <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--fb-danger)', background: 'var(--fb-danger-dim)', padding: '2px 6px', borderRadius: 5, flex: 'none' }}>Missing</span>}
    </button>
  );

  const skeleton = (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: '4px 0' }}>
      {[0, 1, 2, 3, 4].map((i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, minHeight: 56, padding: '0 10px' }}>
          <div style={{ width: 40, height: 40, borderRadius: 9, flex: 'none', background: 'var(--fb-elev)', backgroundSize: '720px 100%', animation: `fbShimmer 1.4s ${i * 0.08}s infinite linear` }} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 7 }}>
            <div style={{ height: 11, width: '55%', borderRadius: 5, background: 'var(--fb-elev)', backgroundSize: '720px 100%', animation: `fbShimmer 1.4s ${i * 0.08}s infinite linear` }} />
            <div style={{ height: 9, width: '32%', borderRadius: 5, background: 'var(--fb-elev)', backgroundSize: '720px 100%', animation: `fbShimmer 1.4s ${i * 0.08 + 0.1}s infinite linear` }} />
          </div>
        </div>
      ))}
    </div>
  );

  const listRow = (e) => {
    const isImporting = importingPath === e.path;
    const sel = selectedPaths.has(e.path);
    const canImport = importable(e);
    return (
      <div key={e.path}
        onClick={() => (e.is_dir ? goTo(e.path) : canImport && toggleSel(e.path))}
        style={{
          display: 'flex', alignItems: 'center', gap: 12, minHeight: 56, padding: '0 10px',
          borderRadius: 11, cursor: (e.is_dir || canImport) ? 'pointer' : 'default',
          background: sel ? 'var(--fb-sel)' : 'transparent',
          boxShadow: sel ? 'inset 0 0 0 1.5px var(--fb-accent)' : 'none',
          transition: 'background 120ms',
        }}>
        {canImport && (
          <span style={{
            width: 22, height: 22, flex: 'none', borderRadius: '50%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: sel ? '0' : '1.8px solid var(--fb-border-strong)',
            background: sel ? 'var(--fb-accent)' : 'transparent', color: '#fff',
          }}>{sel && <Ic.Check s={14} sw={3} />}</span>
        )}
        <Thumb entry={e} hostId={hostId} />
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <span style={{ fontSize: 15, fontWeight: 500, color: 'var(--fb-tp)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.name}</span>
          <span style={{ fontFamily: 'var(--fb-mono)', fontSize: 12, color: 'var(--fb-tm)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {e.is_dir ? 'Folder' : fmtSize(e.size)}{!e.is_dir && (e.mtime_ms ? ` · ${fmtDate(e.mtime_ms)}` : '')}
          </span>
          {isImporting && (
            <div style={{ marginTop: 5 }}>
              <div style={{ height: 4, borderRadius: 2, background: 'var(--fb-track)', overflow: 'hidden', width: 120 }}>
                <div style={{
                  height: '100%', width: importPct == null ? '35%' : `${importPct}%`,
                  background: 'var(--fb-accent)', borderRadius: 2, transition: 'width 0.3s',
                  animation: importPct == null ? 'fbIndet 1.1s ease-in-out infinite' : 'none',
                }} />
              </div>
              <div style={{ fontFamily: 'var(--fb-mono)', fontSize: 11, color: 'var(--fb-ts)', marginTop: 3 }}>
                {importPct == null ? 'Importing…' : `Importing ${importPct}%`}
              </div>
            </div>
          )}
        </div>
        {e.is_dir ? (
          <span style={{ color: 'var(--fb-tm)', display: 'flex', flex: 'none' }}><Ic.ChevR s={17} /></span>
        ) : canImport ? (
          <button onClick={(ev) => { ev.stopPropagation(); doImport(e); }} disabled={busy} style={importPill(isImporting)}>
            <Ic.Download s={15} sw={2} />{isImporting ? '…' : 'Import'}
          </button>
        ) : (
          <span style={{ color: 'var(--fb-tm)', fontSize: 16, paddingRight: 6, flex: 'none' }}>—</span>
        )}
      </div>
    );
  };

  const gridCard = (e) => {
    const sel = selectedPaths.has(e.path);
    const canImport = importable(e);
    return (
      <div key={e.path}
        onClick={() => (e.is_dir ? goTo(e.path) : canImport && toggleSel(e.path))}
        style={{
          display: 'flex', flexDirection: 'column', borderRadius: 12, overflow: 'hidden',
          background: 'var(--fb-elev)', cursor: (e.is_dir || canImport) ? 'pointer' : 'default',
          border: '1px solid var(--fb-border)',
          boxShadow: sel ? 'inset 0 0 0 2px var(--fb-accent)' : 'none',
        }}>
        <div style={{ position: 'relative', width: '100%', aspectRatio: '16 / 10', overflow: 'hidden' }}>
          <Thumb entry={e} hostId={hostId} variant="grid" />
          {canImport && (
            <span style={{
              position: 'absolute', top: 8, right: 8, width: 22, height: 22, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: sel ? '0' : '1.8px solid rgba(255,255,255,0.7)',
              background: sel ? 'var(--fb-accent)' : 'rgba(0,0,0,0.35)', color: '#fff',
            }}>{sel && <Ic.Check s={13} sw={3} />}</span>
          )}
        </div>
        <div style={{ padding: '9px 11px 11px', display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--fb-tp)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.name}</span>
          <span style={{ fontFamily: 'var(--fb-mono)', fontSize: 11, color: 'var(--fb-tm)' }}>
            {e.is_dir ? 'Folder' : fmtSize(e.size)}
          </span>
        </div>
      </div>
    );
  };

  const tog = (on) => ({
    width: 34, height: 30, borderRadius: 7, display: 'flex', alignItems: 'center', justifyContent: 'center',
    color: on ? 'var(--fb-tp)' : 'var(--fb-tm)', background: on ? 'var(--fb-panel)' : 'transparent',
    boxShadow: on ? 'var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.2))' : 'none', border: 'none', cursor: 'pointer',
  });

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(3px)',
      WebkitBackdropFilter: 'blur(3px)', zIndex: 1000, display: 'flex', alignItems: 'center',
      justifyContent: 'center', padding: 20,
    }}>
      <style>{`
        @keyframes fbShimmer { 0% { background-position: -360px 0; } 100% { background-position: 360px 0; } }
        @keyframes fbIndet { 0% { transform: translateX(-120%); } 100% { transform: translateX(320%); } }
        @keyframes fbSpin { to { transform: rotate(360deg); } }
      `}</style>
      <div onClick={(e) => e.stopPropagation()} style={{
        ...TT, fontFamily: 'var(--font-sans, -apple-system, BlinkMacSystemFont, sans-serif)',
        background: 'var(--fb-panel)', backdropFilter: 'blur(30px) saturate(180%)',
        WebkitBackdropFilter: 'blur(30px) saturate(180%)', border: '1px solid var(--fb-border)',
        borderRadius: 18, width: 'min(840px, 96vw)', height: 'min(660px, 90vh)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
        boxShadow: 'var(--shadow-lg, 0 24px 64px rgba(0,0,0,0.5))', color: 'var(--fb-tp)',
      }}>
        {/* Header */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', borderBottom: '1px solid var(--fb-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 'none', minWidth: 0 }}>
            <span style={{ width: 30, height: 30, borderRadius: 8, background: 'var(--fb-accent)', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 'none' }}>
              <Ic.HardDrive s={18} />
            </span>
            <span style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--fb-tp)', lineHeight: 1.15, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{activeHost?.name || 'Companion'}</span>
              <span style={{ fontSize: 11, color: 'var(--fb-tm)', lineHeight: 1.2 }}>{statusLine}</span>
            </span>
          </div>
          <span style={{ fontSize: 17, fontWeight: 600, color: 'var(--fb-tp)', margin: '0 auto' }}>Import a file</span>
          {onlineHosts.length > 1 && (
            <select value={hostId} onChange={(ev) => { setHostId(ev.target.value); setCwd(''); }}
              style={{ flex: 'none', height: 30, borderRadius: 8, background: 'var(--fb-elev)', color: 'var(--fb-tp)', border: '1px solid var(--fb-border)', fontSize: 12, padding: '0 6px' }}>
              {onlineHosts.map((c) => <option key={c.host_id} value={c.host_id}>{c.name}</option>)}
            </select>
          )}
          <button onClick={onClose} aria-label="Close" style={{ width: 30, height: 30, borderRadius: '50%', background: 'var(--fb-elev)', border: 'none', color: 'var(--fb-ts)', display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 'none', cursor: 'pointer' }}>
            <Ic.X s={17} />
          </button>
        </div>

        {/* Search */}
        <div style={{ flex: 'none', padding: '12px 16px', borderBottom: '1px solid var(--fb-border)' }}>
          <div style={field}>
            <span style={{ display: 'flex', color: 'var(--fb-tm)', flex: 'none' }}><Ic.Search s={18} /></span>
            <input
              autoFocus type="text" value={query}
              onChange={(ev) => setQuery(ev.target.value)} onKeyDown={onSearchKey}
              placeholder={cwd ? 'Search this folder, or paste a path…' : 'Paste a full path and press Enter…'}
              style={{ flex: 1, minWidth: 0, background: 'transparent', border: 'none', outline: 'none', color: 'var(--fb-tp)', fontSize: 15 }}
            />
            {looksLikePath(query) && (
              <button onClick={() => goTo(query.trim())} style={{ display: 'flex', alignItems: 'center', gap: 5, background: 'var(--fb-accent)', color: '#fff', fontSize: 13, fontWeight: 600, padding: '6px 12px', borderRadius: 8, border: 'none', cursor: 'pointer' }}>Go</button>
            )}
          </div>
        </div>

        {/* Breadcrumb + view toggle */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: '9px 16px', borderBottom: '1px solid var(--fb-border)', minHeight: 46 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1, minWidth: 0, overflow: 'hidden' }}>
            <button onClick={() => goTo('')} title="Shared folders"
              style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '5px 8px', borderRadius: 7, color: cwd ? 'var(--fb-ts)' : 'var(--fb-tp)', background: cwd ? 'transparent' : 'var(--fb-elev)', border: 'none', cursor: 'pointer', flex: 'none' }}>
              <Ic.Home s={16} /><span style={{ fontSize: 13.5, fontWeight: 600 }}>Shared</span>
            </button>
            {crumbs.map((cr, i) => {
              const last = i === crumbs.length - 1;
              return (
                <React.Fragment key={cr.path}>
                  <span style={{ display: 'flex', color: 'var(--fb-tm)', flex: 'none' }}><Ic.ChevR s={15} /></span>
                  <button onClick={() => !last && goTo(cr.path)} disabled={last}
                    style={{ padding: '5px 9px', borderRadius: 7, fontSize: 13.5, fontWeight: last ? 600 : 500, color: last ? 'var(--fb-tp)' : 'var(--fb-ts)', background: last ? 'var(--fb-elev)' : 'transparent', border: 'none', cursor: last ? 'default' : 'pointer', whiteSpace: 'nowrap', flex: 'none', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {cr.name}
                  </button>
                </React.Fragment>
              );
            })}
          </div>
          {cwd && (
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--fb-tm)', flex: 'none' }}>
              <span style={{ whiteSpace: 'nowrap' }}>Sort</span>
              <select value={sortBy} onChange={(ev) => setSortBy(ev.target.value)}
                style={{ height: 30, borderRadius: 8, background: 'var(--fb-elev)', color: 'var(--fb-tp)', border: '1px solid var(--fb-border)', fontSize: 12, padding: '0 6px' }}>
                {SORTS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
              </select>
            </label>
          )}
          <div style={{ display: 'flex', background: 'var(--fb-elev)', border: '1px solid var(--fb-border)', borderRadius: 9, padding: 2, gap: 2, flex: 'none' }}>
            <button onClick={() => setView('list')} style={tog(view === 'list')} aria-label="List view"><Ic.List s={17} /></button>
            <button onClick={() => setView('grid')} style={tog(view === 'grid')} aria-label="Grid view"><Ic.Grid s={16} /></button>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 8px 8px', display: 'flex', flexDirection: 'column', gap: 1 }}>
          {loading ? skeleton
          : rootsError && !roots.length ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: '40px 16px', textAlign: 'center' }}>
              <div style={{ width: 52, height: 52, borderRadius: 16, background: 'var(--fb-danger-dim)', color: 'var(--fb-danger)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Ic.HardDrive s={24} />
              </div>
              <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--fb-tp)' }}>Companion not reachable</span>
              <span style={{ fontSize: 12.5, color: 'var(--fb-tm)', maxWidth: 340 }}>{rootsError}</span>
            </div>
          )
          : !cwd ? (
            roots.length
              ? <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>{roots.map(rootRow)}</div>
              : <div style={{ padding: 24, textAlign: 'center', color: 'var(--fb-tm)', fontSize: 13 }}>This Companion has no shared folders. Add one in the Companion app.</div>
          )
          : listing ? skeleton
          : listError ? (
            <div style={{ padding: 24, textAlign: 'center', color: 'var(--fb-tm)', fontSize: 13 }}>{listError}</div>
          )
          : shown.length === 0 ? (
            <div style={{ padding: 40, textAlign: 'center', color: 'var(--fb-tm)', fontSize: 13 }}>{query ? 'No matches.' : 'Empty folder.'}</div>
          )
          : view === 'grid' ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12, padding: '4px 2px' }}>
              {shown.map(gridCard)}
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>{shown.map(listRow)}</div>
          )}
        </div>

        {/* Footer */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', borderTop: '1px solid var(--fb-border)', background: 'var(--fb-elev)' }}>
          <span style={{ fontSize: 13, color: 'var(--fb-tm)', marginRight: 'auto', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {batchMsg || importMsg || (selCount > 0 ? `${selCount} file${selCount === 1 ? '' : 's'} selected` : statusLine)}
          </span>
          {selCount > 0 && !batchMsg && (
            <button onClick={() => setSelectedPaths(new Set())}
              style={{ fontSize: 15, fontWeight: 500, color: 'var(--fb-ts)', padding: '10px 16px', borderRadius: 11, background: 'transparent', border: 'none', cursor: 'pointer' }}>Clear</button>
          )}
          <button onClick={onClose}
            style={{ fontSize: 15, fontWeight: 500, color: 'var(--fb-ts)', padding: '10px 16px', borderRadius: 11, background: 'transparent', border: 'none', cursor: 'pointer' }}>Cancel</button>
          <button onClick={importSelected} disabled={busy || selCount === 0}
            style={{ display: 'flex', alignItems: 'center', gap: 7, fontSize: 15, fontWeight: 600, color: '#fff', background: 'var(--fb-accent)', padding: '11px 20px', borderRadius: 12, minHeight: 44, border: 'none', cursor: (busy || selCount === 0) ? 'default' : 'pointer', opacity: (busy || selCount === 0) ? 0.5 : 1, boxShadow: '0 3px 10px rgba(10,132,255,.32)' }}>
            <Ic.Download s={15} sw={2} />{batchMsg ? 'Importing…' : selCount > 1 ? `Import ${selCount}` : 'Import'}
          </button>
        </div>
      </div>
    </div>
  );
}
