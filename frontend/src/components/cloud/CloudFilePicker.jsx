/**
 * Modal file picker for a connected cloud provider — "Import a file".
 *
 * Behavior:
 *   - Shows a breadcrumb trail rooted at "root".
 *   - Debounced search box. Empty query -> folder browse; non-empty -> search.
 *   - Clicking a folder navigates into it. Clicking a file selects it
 *     (single-select radio). "Choose" in the footer calls onPick(file) and the
 *     caller decides what to do with the import.
 *   - Pagination is a "Load more" button that sends the provider's
 *     next_page_token back to the server.
 *
 * The component is provider-agnostic: every request goes to
 * `/api/cloud/{provider}/browse` or `/api/cloud/{provider}/search`,
 * which return the same uniform shape regardless of Google vs Box.
 *
 * The look is a 1:1 port of the file-browser design system (FileBrowser.css):
 * a frosted Apple-style modal on desktop, a bottom sheet on phones.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import useResponsive from '../../hooks/useResponsive';
import './FileBrowser.css';

const SEARCH_DEBOUNCE_MS = 300;

function formatBytes(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

// Defense-in-depth — the server already filters to video MIMEs/extensions,
// but we double-check so a bug on the server side can never surface docs
// in the picker.
const VIDEO_EXT_RE = /\.(mp4|mov|mkv|webm|avi|m4v|3gp|mpe?g)$/i;

function isVideo(file) {
  if (!file) return false;
  if (typeof file.mime_type === 'string' && file.mime_type.toLowerCase().startsWith('video/')) {
    return true;
  }
  if (typeof file.name === 'string' && VIDEO_EXT_RE.test(file.name)) {
    return true;
  }
  return false;
}

// ── Icons (lucide-style, matching the mockup) ────────────────────────────────
const P = {
  Search: [['circle', { cx: 11, cy: 11, r: 8 }], ['path', { d: 'm21 21-4.3-4.3' }]],
  X: [['path', { d: 'M18 6 6 18' }], ['path', { d: 'm6 6 12 12' }]],
  ChevronRight: [['path', { d: 'm9 18 6-6-6-6' }]],
  ChevronLeft: [['path', { d: 'm15 18-6-6 6-6' }]],
  Check: [['path', { d: 'M20 6 9 17l-5-5' }]],
  List: [['path', { d: 'M8 6h13' }], ['path', { d: 'M8 12h13' }], ['path', { d: 'M8 18h13' }], ['path', { d: 'M3 6h.01' }], ['path', { d: 'M3 12h.01' }], ['path', { d: 'M3 18h.01' }]],
  LayoutGrid: [['rect', { x: 3, y: 3, width: 7, height: 7, rx: 1 }], ['rect', { x: 14, y: 3, width: 7, height: 7, rx: 1 }], ['rect', { x: 14, y: 14, width: 7, height: 7, rx: 1 }], ['rect', { x: 3, y: 14, width: 7, height: 7, rx: 1 }]],
  Download: [['path', { d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4' }], ['path', { d: 'm7 10 5 5 5-5' }], ['path', { d: 'M12 15V3' }]],
  Folder: [['path', { d: 'M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z' }]],
  Film: [['rect', { x: 2, y: 3, width: 20, height: 18, rx: 2 }], ['path', { d: 'M7 3v18' }], ['path', { d: 'M17 3v18' }], ['path', { d: 'M2 12h20' }], ['path', { d: 'M2 7.5h5' }], ['path', { d: 'M2 16.5h5' }], ['path', { d: 'M17 7.5h5' }], ['path', { d: 'M17 16.5h5' }]],
  Image: [['rect', { x: 3, y: 3, width: 18, height: 18, rx: 2 }], ['circle', { cx: 9, cy: 9, r: 2 }], ['path', { d: 'm21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21' }]],
  FileText: [['path', { d: 'M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z' }], ['path', { d: 'M14 2v5h5' }], ['path', { d: 'M9 13h6' }], ['path', { d: 'M9 17h6' }]],
  Cloud: [['path', { d: 'M17.5 19a4.5 4.5 0 1 0-1.5-8.74A6 6 0 1 0 6.5 19z' }]],
  HardDrive: [['path', { d: 'M22 12H2' }], ['path', { d: 'M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z' }], ['path', { d: 'M6 16h.01' }], ['path', { d: 'M10 16h.01' }]],
};

function Icon({ name, size = 18, color = 'currentColor', stroke = 1.8, fill = 'none', style }) {
  const spec = P[name];
  if (!spec) return <span style={{ display: 'inline-block', width: size, height: size }} />;
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24"
      fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round"
      style={{ display: 'block', ...style }}
    >
      {spec.map((n, i) => React.createElement(n[0], { key: i, ...(n[1] || {}) }))}
    </svg>
  );
}

// Provider → header identity (icon + brand colour). Falls back to a generic
// cloud badge for any future provider.
const PROVIDER_META = {
  google_drive: { icon: 'Cloud', color: '#1a73e8', name: 'Google Drive' },
  box: { icon: 'HardDrive', color: '#0061d5', name: 'Box' },
};

function typeIcon(file) {
  if (isVideo(file)) return 'Film';
  const n = (file?.name || '').toLowerCase();
  if (/\.(png|jpe?g|gif|webp|heic|bmp|svg)$/.test(n)) return 'Image';
  return 'FileText';
}

export default function CloudFilePicker({ provider, providerLabel, account, onPick, onClose }) {
  const { isMobile } = useResponsive();
  const [path, setPath] = useState([{ id: null, name: provider === 'box' ? 'All Files' : 'My Drive' }]);
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [entries, setEntries] = useState({ folders: [], files: [] });
  const [nextPageToken, setNextPageToken] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [importing, setImporting] = useState(false);
  const [viewMode, setViewMode] = useState('list'); // 'list' | 'grid'
  const reqSeq = useRef(0);

  const meta = PROVIDER_META[provider] || { icon: 'Cloud', color: '#0A84FF', name: providerLabel || provider };
  const currentFolder = path[path.length - 1];

  // Debounce the search box
  useEffect(() => {
    const handle = setTimeout(() => setDebouncedQuery(query.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [query]);

  const load = useCallback(
    async ({ append = false } = {}) => {
      const mySeq = ++reqSeq.current;
      setLoading(true);
      setError(null);
      try {
        let url;
        if (debouncedQuery) {
          const params = new URLSearchParams({ q: debouncedQuery });
          if (append && nextPageToken) params.set('page_token', nextPageToken);
          url = `/api/cloud/${provider}/search?${params.toString()}`;
        } else {
          const params = new URLSearchParams();
          if (currentFolder.id) params.set('folder_id', currentFolder.id);
          if (append && nextPageToken) params.set('page_token', nextPageToken);
          const qs = params.toString();
          url = `/api/cloud/${provider}/browse${qs ? `?${qs}` : ''}`;
        }

        const resp = await fetch(url);
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        const page = await resp.json();
        if (mySeq !== reqSeq.current) return; // a newer request superseded this one

        const filteredFiles = (page.files || []).filter(isVideo);
        setEntries((prev) =>
          append
            ? {
                folders: [...prev.folders, ...(page.folders || [])],
                files: [...prev.files, ...filteredFiles],
              }
            : { folders: page.folders || [], files: filteredFiles }
        );
        setNextPageToken(page.next_page_token || null);
      } catch (err) {
        if (mySeq !== reqSeq.current) return;
        setError(err.message || 'Failed to load');
      } finally {
        if (mySeq === reqSeq.current) setLoading(false);
      }
    },
    [provider, currentFolder.id, debouncedQuery, nextPageToken]
  );

  // Reload when the folder or the debounced query changes.
  useEffect(() => {
    setEntries({ folders: [], files: [] });
    setNextPageToken(null);
    setSelected(null);
    load({ append: false });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, currentFolder.id, debouncedQuery]);

  const handleFolderClick = useCallback((folder) => {
    setQuery('');
    setSelected(null);
    setPath((prev) => [...prev, { id: folder.id, name: folder.name }]);
  }, []);

  const handleCrumbClick = useCallback((index) => {
    setQuery('');
    setSelected(null);
    setPath((prev) => prev.slice(0, index + 1));
  }, []);

  const handleChoose = useCallback(async () => {
    if (!selected || importing) return;
    setImporting(true);
    try {
      await onPick(selected);
    } finally {
      setImporting(false);
    }
  }, [selected, importing, onPick]);

  const fileMeta = (file) => {
    const size = formatBytes(file.size);
    const date = file.modified_at ? new Date(file.modified_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '';
    return [size, date].filter(Boolean).join(' · ');
  };

  const isEmpty = !loading && !error && entries.folders.length === 0 && entries.files.length === 0;

  // ── List row (folder or file) ──
  const renderListRow = (kind, item) => {
    const isFolder = kind === 'folder';
    const sel = !isFolder && selected?.id === item.id;
    return (
      <div
        key={`${kind}-${item.id}`}
        role="button"
        tabIndex={0}
        className={`fb-row${sel ? ' is-selected' : ''}`}
        onClick={() => (isFolder ? handleFolderClick(item) : setSelected(item))}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); isFolder ? handleFolderClick(item) : setSelected(item); } }}
      >
        {!isFolder && (
          <span className={`fb-radio${sel ? ' is-on' : ''}`}>
            {sel && <Icon name="Check" size={14} color="#fff" stroke={3} />}
          </span>
        )}
        <span
          className="fb-icontile"
          style={isFolder ? { background: 'var(--fb-accent-dim2)' } : undefined}
        >
          <Icon
            name={isFolder ? 'Folder' : typeIcon(item)}
            size={20}
            color={isFolder ? 'var(--fb-accent)' : 'var(--fb-ts)'}
            fill={isFolder ? 'var(--fb-accent-dim2)' : 'none'}
          />
        </span>
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <span style={{ fontSize: 15, fontWeight: 500, color: 'var(--fb-tp)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={item.name}>{item.name}</span>
          <span style={{ fontFamily: 'var(--fb-mono)', fontSize: 12, color: 'var(--fb-tm)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {isFolder ? (item.item_count != null ? `${item.item_count} items` : 'Folder') : fileMeta(item)}
          </span>
        </div>
        {isFolder && <span style={{ display: 'flex', color: 'var(--fb-tm)', flex: 'none' }}><Icon name="ChevronRight" size={17} /></span>}
      </div>
    );
  };

  // ── Grid card (files only; folders stay as list rows above) ──
  const renderCard = (file) => {
    const sel = selected?.id === file.id;
    return (
      <div
        key={`card-${file.id}`}
        role="button"
        tabIndex={0}
        className={`fb-card${sel ? ' is-selected' : ''}`}
        onClick={() => setSelected(file)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelected(file); } }}
      >
        <div className="fb-card__thumb">
          {file.thumbnail_url
            ? <img src={file.thumbnail_url} alt="" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }} />
            : <Icon name={typeIcon(file)} size={26} color="var(--fb-tm)" />}
          {sel && (
            <span style={{ position: 'absolute', top: 8, right: 8, width: 22, height: 22, borderRadius: '50%', background: 'var(--fb-accent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Icon name="Check" size={14} color="#fff" stroke={3} />
            </span>
          )}
        </div>
        <div style={{ padding: '9px 11px 11px', display: 'flex', flexDirection: 'column', gap: 3 }}>
          <span style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--fb-tp)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={file.name}>{file.name}</span>
          <span style={{ fontFamily: 'var(--fb-mono)', fontSize: 11, color: 'var(--fb-tm)' }}>{fileMeta(file)}</span>
        </div>
      </div>
    );
  };

  return (
    <div
      className="fb-scope fb-overlay"
      onClick={(e) => e.target === e.currentTarget && !importing && onClose?.()}
    >
      <div className="fb-modal" role="dialog" aria-label="Import a file">
        {/* Phone grab handle */}
        {isMobile && (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '9px 0 3px', flex: 'none' }}>
            <div style={{ width: 38, height: 5, borderRadius: 3, background: 'var(--fb-border-strong)' }} />
          </div>
        )}

        {/* Header */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 12, padding: isMobile ? '2px 16px 12px' : '14px 16px', borderBottom: '1px solid var(--fb-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 'none' }}>
            <span style={{ width: 30, height: 30, borderRadius: 8, background: meta.color, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', flex: 'none' }}>
              <Icon name={meta.icon} size={18} color="#fff" />
            </span>
            <span style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--fb-tp)', lineHeight: 1.15 }}>{providerLabel || meta.name}</span>
              <span style={{ fontSize: 11, color: 'var(--fb-tm)', lineHeight: 1.2 }}>{account || 'Videos only'}</span>
            </span>
          </div>
          {!isMobile && <span style={{ fontSize: 17, fontWeight: 600, color: 'var(--fb-tp)', margin: '0 auto' }}>Import a file</span>}
          <button type="button" className="fb-close" onClick={() => !importing && onClose?.()} aria-label="Close" style={{ marginLeft: isMobile ? 'auto' : 0 }}>
            <Icon name="X" size={17} />
          </button>
        </div>

        {/* Search */}
        <div style={{ flex: 'none', padding: '12px 16px', borderBottom: '1px solid var(--fb-border)' }}>
          <div className="fb-search">
            <span style={{ display: 'flex', color: 'var(--fb-tm)', flex: 'none' }}><Icon name="Search" size={18} color="var(--fb-tm)" /></span>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Search in ${meta.name}`}
              aria-label="Search"
            />
            {loading && query && (
              <span style={{ fontSize: 12, color: 'var(--fb-tm)', display: 'flex', alignItems: 'center', gap: 6, flex: 'none' }}>
                <span className="fb-spin" />Searching…
              </span>
            )}
          </div>
        </div>

        {/* Breadcrumbs + view toggle */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: isMobile ? '10px 12px' : '9px 16px', borderBottom: '1px solid var(--fb-border)', minHeight: isMobile ? 48 : 46 }}>
          {isMobile ? (
            <>
              {path.length > 1 ? (
                <button type="button" className="fb-crumb" style={{ display: 'flex', alignItems: 'center', gap: 2, color: 'var(--fb-accent)', fontSize: 15, fontWeight: 500, padding: 0 }} onClick={() => handleCrumbClick(path.length - 2)}>
                  <Icon name="ChevronLeft" size={22} color="var(--fb-accent)" />{path[path.length - 2].name}
                </button>
              ) : <span style={{ width: 36 }} />}
              <span style={{ fontSize: 15, fontWeight: 600, color: 'var(--fb-tp)', margin: '0 auto', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{currentFolder.name}</span>
              <span style={{ width: 36 }} />
            </>
          ) : (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1, minWidth: 0, overflow: 'hidden' }}>
                {path.map((crumb, idx) => (
                  <React.Fragment key={`${crumb.id || 'root'}-${idx}`}>
                    {idx > 0 && <span style={{ display: 'flex', color: 'var(--fb-tm)', flex: 'none' }}><Icon name="ChevronRight" size={15} /></span>}
                    <button
                      type="button"
                      className={`fb-crumb${idx === path.length - 1 ? ' is-current' : ''}`}
                      onClick={() => idx !== path.length - 1 && handleCrumbClick(idx)}
                      disabled={idx === path.length - 1}
                    >{crumb.name}</button>
                  </React.Fragment>
                ))}
              </div>
              <div style={{ display: 'flex', background: 'var(--fb-elev)', border: '1px solid var(--fb-border)', borderRadius: 9, padding: 2, gap: 2, flex: 'none' }}>
                <button type="button" className={`fb-toggle${viewMode === 'list' ? ' is-on' : ''}`} onClick={() => setViewMode('list')} aria-label="List view"><Icon name="List" size={17} /></button>
                <button type="button" className={`fb-toggle${viewMode === 'grid' ? ' is-on' : ''}`} onClick={() => setViewMode('grid')} aria-label="Grid view"><Icon name="LayoutGrid" size={16} /></button>
              </div>
            </>
          )}
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 8px 8px', display: 'flex', flexDirection: 'column', gap: 1 }}>
          {error && (
            <div style={{ padding: 14, color: 'var(--danger, #FF3B30)', fontSize: 13, textAlign: 'center' }}>{error}</div>
          )}
          {isEmpty && (
            <div style={{ padding: 28, textAlign: 'center', color: 'var(--fb-tm)', fontSize: 14 }}>
              {debouncedQuery ? 'No matching videos.' : 'This folder has no videos.'}
            </div>
          )}

          {viewMode === 'list' || debouncedQuery ? (
            <>
              {entries.folders.map((f) => renderListRow('folder', f))}
              {entries.files.map((f) => renderListRow('file', f))}
            </>
          ) : (
            <>
              {entries.folders.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 1, marginBottom: entries.files.length ? 8 : 0 }}>
                  {entries.folders.map((f) => renderListRow('folder', f))}
                </div>
              )}
              <div className="fb-grid" style={{ padding: '2px 2px 0' }}>
                {entries.files.map((f) => renderCard(f))}
              </div>
            </>
          )}

          {nextPageToken && !loading && (
            <div style={{ textAlign: 'center', margin: '12px 0 4px' }}>
              <button type="button" className="fb-cancel" style={{ border: '1px solid var(--fb-border)' }} onClick={() => load({ append: true })}>Load more</button>
            </div>
          )}
          {loading && !query && (
            <div style={{ padding: 16, textAlign: 'center', color: 'var(--fb-tm)', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <span className="fb-spin" />Loading…
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ flex: 'none', display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px', paddingBottom: isMobile ? 'calc(12px + env(safe-area-inset-bottom, 0px))' : 12, borderTop: '1px solid var(--fb-border)', background: 'var(--fb-elev)' }}>
          <span style={{ fontSize: 13, color: 'var(--fb-tm)', marginRight: 'auto' }}>
            {selected ? '1 file selected' : 'Select a video to import'}
          </span>
          <button type="button" className="fb-cancel" onClick={() => !importing && onClose?.()}>Cancel</button>
          <button type="button" className="fb-choose" onClick={handleChoose} disabled={!selected || importing}>
            {importing ? <span className="fb-spin" style={{ borderTopColor: '#fff', borderColor: 'rgba(255,255,255,0.4)' }} /> : <span style={{ display: 'flex' }}><Icon name="Download" size={15} stroke={2} color="#fff" /></span>}
            {importing ? 'Importing…' : 'Choose'}
          </button>
        </div>
      </div>
    </div>
  );
}
