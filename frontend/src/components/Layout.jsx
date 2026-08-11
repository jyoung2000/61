import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { showToast } from './Toast';
import ProviderStatus from './ProviderStatus';
import ConnectionStatus from './ConnectionStatus';
import UpdateNudge from './UpdateNudge';
import UserMenu from './UserMenu';
import useResponsive from '../hooks/useResponsive';
import useTheme from '../hooks/useTheme';
import useConnectionStatus from '../hooks/useConnectionStatus';
import useEncodingManager from '../hooks/useEncodingManager';
import { useClientGpuPreferences, useAutoDetectGpu } from '../hooks/useClientGpu';

const NAV_ITEMS = [
  { path: '/', label: 'Home', icon: 'D', mobileIcon: 'home' },
  { path: '/upload', label: 'Upload', icon: 'U', mobileIcon: 'upload' },
  { path: '/clips', label: 'Clips', icon: 'V', mobileIcon: 'clips' },
  { path: '/media', label: 'Media Library', icon: 'M', mobileIcon: 'media' },
  { path: '/logs', label: 'Exports', icon: 'L', mobileIcon: 'logs' },
  { path: '/settings', label: 'Settings', icon: 'S', mobileIcon: 'settings' },
];

// SF-style tab bar icons (simple SVG paths)
const MobileTabIcon = ({ type, active }) => {
  const color = active ? 'var(--accent-cyan)' : 'var(--text-muted)';
  const size = 22;
  const stroke = active ? 2 : 1.5;
  const fill = 'none';
  switch (type) {
    case 'home': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />{active && <rect x="9" y="14" width="6" height="8" fill={color} stroke="none" />}
      </svg>
    );
    case 'upload': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="17 8 12 3 7 8" /><line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    );
    case 'clips': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5 3 19 12 5 21 5 3" fill={active ? color : 'none'} />
      </svg>
    );
    case 'logs': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
      </svg>
    );
    case 'media': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" /><circle cx="8.5" cy="8.5" r="1.5" fill={active ? color : 'none'} /><polyline points="21 15 16 10 5 21" />
      </svg>
    );
    case 'settings': return (
      <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke={color} strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
      </svg>
    );
    default: return null;
  }
};

function shortModel(modelId) {
  if (!modelId) return '';
  let name = modelId;
  if (name.includes('/')) name = name.split('/').pop();
  name = name.replace(/:free$/, '');
  return name;
}

export default function Layout({ children }) {
  const [collapsed, setCollapsed] = useState(false);
  // The user's own expand/collapse choice on NORMAL pages. Editor pages force
  // a collapse for focus mode, but we restore this when leaving so the
  // auto-collapse never leaks onto Home/Upload/Settings/etc.
  const userCollapsePref = useRef(false);
  const [activeModel, setActiveModel] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const { isMobile } = useResponsive();
  const { isDark, toggleTheme } = useTheme();
  const connStatus = useConnectionStatus();
  const { activeCount, latestActivity } = useEncodingManager();
  const clientGpu = useClientGpuPreferences();
  useAutoDetectGpu(); // Auto-detect and enable WebGPU on first visit

  // GPU actually running the active *local* models — the paired Companion's
  // card (or a named Ollama host). Cloud providers have no local GPU.
  const hostGpu = activeModel && activeModel.provider === 'ollama'
    ? (activeModel.companion?.gpu_name
        ? `${activeModel.companion.gpu_name}${activeModel.companion?.name ? ` · ${activeModel.companion.name}` : ''}`
        : (activeModel.ollama_host_name || 'local'))
    : null;

  // Live Companion liveness: a cheap 10s poll (vs the 30s full status poll) so
  // ClipAI notices within ~10s when the Companion app is closed / the PC sleeps
  // / it drops off the network, and warns the moment it happens.
  const [companionLive, setCompanionLive] = useState(null);
  const prevOnlineRef = useRef(null);
  const prevPausedRef = useRef(null);
  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const res = await fetch('/api/providers/companion-status');
        if (!res.ok) return;
        const data = await res.json();
        if (!active) return;
        setCompanionLive(data);
        if (data.paired) {
          const prev = prevOnlineRef.current;
          if (prev === true && data.online === false) {
            showToast('GPU Companion went offline — AI falls back to another provider', 'error');
          } else if (prev === false && data.online === true) {
            showToast('GPU Companion reconnected', 'success');
          }
          prevOnlineRef.current = data.online;
          // Paused is distinct from offline: the Companion is reachable but
          // refusing work (503). Warn once when it flips on.
          const prevP = prevPausedRef.current;
          if (prevP === false && data.paused === true) {
            showToast('GPU Companion is PAUSED — resume sharing in the Companion app', 'error');
          }
          prevPausedRef.current = !!data.paused;
        }
      } catch { /* silent — the chip just holds its last state */ }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { active = false; clearInterval(id); };
  }, []);

  const agoText = (ms) => {
    if (!ms) return '';
    const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    return `${Math.floor(s / 3600)}h ago`;
  };

  const companionOffline = !!(companionLive?.paired && companionLive.online === false);
  const companionPaused = !!(companionLive?.paired && companionLive.paused === true);

  // Detect if this is a sub-page that should show a back button
  const isSubPage = location.pathname.startsWith('/analysis') || location.pathname.startsWith('/seo');

  // Focus mode: collapse the nav rail to icons on the editor pages (Analysis /
  // Clip Editor) so the preview + timeline get the full widescreen width the
  // user asked for — the 240px rail is the biggest side "bezel". On leaving an
  // editor route we RESTORE the user's own preference (tracked in the toggle
  // below), so the forced collapse never leaks onto other pages.
  useEffect(() => {
    if (isMobile) return;
    setCollapsed(isSubPage ? true : userCollapsePref.current);
  }, [isSubPage, isMobile]);

  // Poll /api/allocation to detect any active container activity
  // (analysis, transcription, clip detection, exports, etc.)
  const [containerActive, setContainerActive] = useState(false);
  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const res = await fetch('/api/allocation');
        if (res.ok && mounted) {
          const data = await res.json();
          setContainerActive((data.active_jobs || []).length > 0);
        }
      } catch {}
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, []);

  const pageName = location.pathname === '/' ? 'Dashboard'
    : location.pathname.startsWith('/upload') ? 'Upload'
    : location.pathname.startsWith('/clips') ? 'Viral Clips'
    : location.pathname.startsWith('/logs') ? 'Logs & Exports'
    : location.pathname.startsWith('/settings') ? 'Settings'
    : location.pathname.startsWith('/media') ? 'Media Library'
    : location.pathname.startsWith('/analysis') ? 'Analysis'
    : location.pathname.startsWith('/seo') ? 'Clip Editor'
    : '';

  const themeToggleBtn = (
    <button
      onClick={() => {
        document.documentElement.classList.add('theme-transition');
        toggleTheme();
        setTimeout(() => document.documentElement.classList.remove('theme-transition'), 350);
      }}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      style={{
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        color: 'var(--text-secondary)',
        fontSize: 16,
        width: 34,
        height: 34,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.2s ease',
        flexShrink: 0,
      }}
    >
      {isDark ? '\u2600' : '\u263D'}
    </button>
  );

  // System status: green pulsing = active/processing, grey = idle, red pulsing = stopped/disconnected
  const isDisconnected = connStatus.status === 'disconnected';
  const isProcessing = activeCount > 0 || containerActive;
  const systemDotColor = isDisconnected ? 'var(--danger)'
    : isProcessing ? 'var(--success)' : 'var(--text-muted)';
  const systemDotPulse = isDisconnected || isProcessing;

  // System status dot — green pulsing = processing, grey = idle, red pulsing = disconnected
  const activityTicker = (
    <span
      style={{
        width: 8, height: 8, borderRadius: '50%',
        background: systemDotColor,
        animation: systemDotPulse ? 'pulse 1.5s ease-in-out infinite' : 'none',
        flexShrink: 0,
        transition: 'background 0.3s ease',
        display: 'inline-block',
      }}
    />
  );

  return (
    <div style={{ display: 'flex', minHeight: '100vh', flexDirection: isMobile ? 'column' : 'row' }}>
      {/* Banner when the server is serving a newer UI than this tab runs */}
      <UpdateNudge />
      {/* ═══ Desktop Sidebar ═══ */}
      <aside
        className="desktop-sidebar"
        style={{
          width: collapsed ? 64 : 240,
          background: 'var(--bg-panel)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          transition: 'width 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          flexShrink: 0,
        }}
      >
        {/* Logo */}
        <div
          style={{
            padding: '16px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'space-between',
          }}
        >
          {!collapsed && (
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: 18,
                color: 'var(--accent-cyan)',
                letterSpacing: '0.02em',
              }}
            >
              CLIP<span style={{ color: 'var(--text-primary)' }}>AI</span>
            </span>
          )}
          <button
            onClick={() => {
              const next = !collapsed;
              setCollapsed(next);
              // Only a toggle on a NORMAL page updates the remembered
              // preference; expanding on an editor page is a one-off.
              if (!isSubPage) userCollapsePref.current = next;
            }}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 16,
              padding: 6,
              borderRadius: 'var(--radius-sm)',
              transition: 'background 0.15s',
            }}
          >
            {collapsed ? '\u276F' : '\u276E'}
          </button>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '8px 0' }}>
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path
              || (item.path === '/clips' && location.pathname.startsWith('/clips'));
            const isLogs = item.path === '/logs';
            return (
              <Link
                key={item.path}
                to={item.path}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: collapsed ? '12px 0' : '10px 16px',
                  margin: collapsed ? 0 : '2px 8px',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  color: isActive ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  background: isActive ? 'var(--accent-cyan-dim)' : 'transparent',
                  borderRadius: collapsed ? 0 : 'var(--radius-sm)',
                  textDecoration: 'none',
                  fontSize: 14,
                  fontWeight: isActive ? 600 : 400,
                  transition: 'background 0.15s ease, color 0.15s ease',
                  letterSpacing: '-0.01em',
                  position: 'relative',
                }}
                // Hover state matches the active background so nav
                // items get a light-blue tint on hover (without
                // graduating to the full active text color).
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'var(--accent-cyan-dim)';
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = 'transparent';
                }}
              >
                <span
                  style={{
                    width: 28,
                    height: 28,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    fontSize: 12,
                    background: isActive ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
                    color: isActive ? 'var(--nav-active-icon-text)' : 'var(--text-secondary)',
                    borderRadius: 'var(--radius-xs)',
                    position: 'relative',
                  }}
                >
                  {item.icon}
                  {/* Active encoding badge on Logs nav */}
                  {isLogs && activeCount > 0 && (
                    <span style={{
                      position: 'absolute', top: -4, right: -4,
                      width: 14, height: 14, borderRadius: '50%',
                      background: 'var(--accent-amber)', color: 'var(--bg-base)',
                      fontSize: 8, fontWeight: 700, fontFamily: 'var(--font-mono)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      animation: 'pulse 1.5s ease-in-out infinite',
                    }}>
                      {activeCount}
                    </span>
                  )}
                </span>
                {!collapsed && item.label}
              </Link>
            );
          })}
        </nav>

        {/* Theme toggle in sidebar */}
        {!collapsed && (
          <div style={{ padding: '8px 16px' }}>
            {themeToggleBtn}
          </div>
        )}

        {/* Provider Status */}
        <ProviderStatus collapsed={collapsed} onActiveChange={setActiveModel} />

        {/* User menu pinned to the sidebar footer — logout, switch user */}
        <UserMenu collapsed={collapsed} />
      </aside>

      {/* ═══ Main Content ═══ */}
      <main
        className="main-content"
        style={{
          flex: 1,
          overflow: 'auto',
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Connection status banner (visible only when disconnected/reconnecting) */}
        <ConnectionStatus status={connStatus.status} latency={connStatus.latency} />

        {/* Mobile Header — iOS-style navigation bar */}
        <header
          className="mobile-header"
          style={{
            display: 'none',
            flexDirection: 'column',
            position: 'sticky',
            top: 0,
            zIndex: 40,
            background: 'var(--bg-panel)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            borderBottom: '1px solid var(--border)',
          }}
        >
          {/* Nav bar row */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '8px var(--page-pad) 6px',
            minHeight: 44,
          }}>
            {/* Left: back button or logo */}
            {isSubPage ? (
              <button
                onClick={() => navigate(-1)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 2,
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: 'var(--accent-cyan)', fontSize: 15, fontWeight: 400,
                  padding: '4px 0', marginLeft: -4,
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="15 18 9 12 15 6" />
                </svg>
                Back
              </button>
            ) : (
              <span style={{
                fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 15,
                color: 'var(--accent-cyan)', letterSpacing: '0.02em',
              }}>
                CLIP<span style={{ color: 'var(--text-primary)' }}>AI</span>
              </span>
            )}

            {/* Right: status + theme */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span
                title={connStatus.status}
                style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: connStatus.status === 'connected' ? 'var(--success)'
                    : connStatus.status === 'reconnecting' ? 'var(--accent-amber)'
                    : 'var(--danger)',
                  flexShrink: 0,
                  animation: connStatus.status !== 'connected' ? 'pulse 1.5s ease-in-out infinite' : undefined,
                }}
              />
              {themeToggleBtn}
            </div>
          </div>

          {/* Large title (iOS-style) */}
          <div style={{
            padding: '0 var(--page-pad) 8px',
          }}>
            <span style={{
              fontSize: 28, fontWeight: 700, letterSpacing: '-0.03em',
              color: 'var(--text-primary)',
            }}>
              {pageName}
            </span>
          </div>

          {/* Activity ticker — compact pill */}
          {latestActivity && activeCount > 0 && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 10px', margin: '0 var(--page-pad) 8px',
              background: 'var(--bg-elevated)', borderRadius: 20,
              overflow: 'hidden',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: 'var(--accent-amber)',
                animation: 'pulse 1.5s ease-in-out infinite',
                flexShrink: 0,
              }} />
              <span style={{
                fontSize: 11, color: 'var(--text-muted)',
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {typeof latestActivity.message === 'string' ? latestActivity.message : String(latestActivity.message ?? '')}
              </span>
            </div>
          )}
        </header>

        {/* Desktop Header */}
        <header
          className="desktop-header"
          style={{
            padding: '12px 24px',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-panel)',
            backdropFilter: 'blur(20px) saturate(180%)',
            WebkitBackdropFilter: 'blur(20px) saturate(180%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: 13,
            color: 'var(--text-secondary)',
            gap: 12,
          }}
        >
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.1em', flexShrink: 0 }}>
            {location.pathname === '/' ? 'dashboard' : location.pathname.slice(1).replace(/\//g, ' / ')}
          </span>

          {/* Center activity ticker */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
            {activityTicker}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
            {activeModel && activeModel.provider !== 'none' && (
              <Link
                to="/settings"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '4px 12px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  textDecoration: 'none',
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  transition: 'border-color 0.2s',
                }}
                title="Click to change AI models"
              >
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success)', flexShrink: 0 }} />
                <span><span style={{ color: 'var(--accent-amber)' }}>T:</span> {String(activeModel.transcript_model || 'whisper-small')}</span>
                <span style={{ color: 'var(--border-strong)' }}>|</span>
                <span>
                  <span style={{ color: 'var(--accent-cyan)' }}>P:</span>{' '}
                  {activeModel.replicate_available
                    ? String(shortModel(activeModel.replicate_model || 'videollama3-7b'))
                    : activeModel.videollama2_available
                      ? 'VideoLLaMA2'
                      : ((activeModel.primary_model || activeModel.vision_model)
                          ? String(shortModel(activeModel.primary_model || activeModel.vision_model))
                          : '\u2014')}
                </span>
                <span style={{ color: 'var(--border-strong)' }}>|</span>
                <span>
                  <span style={{ color: 'var(--success)' }}>E:</span>{' '}
                  {(activeModel.editorial_model || activeModel.text_model)
                    ? String(shortModel(activeModel.editorial_model || activeModel.text_model))
                    : '\u2014'}
                </span>
              </Link>
            )}
            {companionOffline && (
              <Link
                to="/settings"
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 10px',
                  background: 'var(--danger-dim, rgba(239,68,68,0.12))',
                  border: '1px solid var(--danger)',
                  borderRadius: 'var(--radius-sm)', textDecoration: 'none',
                  fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--danger)',
                }}
                title={`GPU Companion unreachable${companionLive?.error ? ` — ${companionLive.error}` : ''}`
                  + `${companionLive?.last_seen_ms ? ` (last seen ${agoText(companionLive.last_seen_ms)})` : ''}.`
                  + ' AI has fallen back to another provider — click to check.'}
              >
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                  background: 'var(--danger)',
                  animation: 'pulse 1.5s ease-in-out infinite',
                }} />
                Companion offline
              </Link>
            )}
            {companionPaused && !companionOffline && (
              <Link
                to="/settings"
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '4px 10px',
                  background: 'var(--danger-dim, rgba(239,68,68,0.12))',
                  border: '1px solid var(--danger)',
                  borderRadius: 'var(--radius-sm)', textDecoration: 'none',
                  fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--danger)',
                }}
                title={'GPU Companion is PAUSED — it answers but refuses every job with 503.'
                  + ' Open the Companion app and click Resume sharing. AI has fallen back to another provider.'}
              >
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                  background: 'var(--danger)',
                }} />
                Companion paused
              </Link>
            )}
            {hostGpu && (
              <Link
                to="/settings"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 10px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  textDecoration: 'none',
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  transition: 'border-color 0.2s',
                }}
                title={activeModel.companion
                  ? `GPU running your local AI models (paired Companion)${activeModel.companion.whisper_remote ? ' — also transcribing here' : ''} — click to configure`
                  : 'GPU host running your local AI models — click to configure'}
              >
                {activeModel.companion && (
                  <span style={{
                    width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                    background: (companionLive?.paired ? !companionLive.online : !activeModel.companion.online)
                      ? 'var(--danger)'
                      : activeModel.companion.ready ? 'var(--success)'
                        : 'var(--accent-amber)',
                  }} />
                )}
                <span style={{ color: 'var(--accent-cyan)' }}>AI GPU:</span> {hostGpu}
              </Link>
            )}
            {clientGpu.enabled && clientGpu.selectedGpuName && (
              <Link
                to="/settings"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 10px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  textDecoration: 'none',
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  transition: 'border-color 0.2s',
                }}
                title="Client GPU (in-browser) — click to configure"
              >
                <span style={{ color: 'var(--accent-cyan)' }}>GPU:</span> {clientGpu.selectedGpuName}
              </Link>
            )}
            {/* Connection status is shown by the system dot in the center ticker */}
            {themeToggleBtn}
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-cyan)' }}>
              v1.0
            </span>
          </div>
        </header>

        {/* Editor pages run edge-to-edge (no page padding) so the preview +
            timeline get the full widescreen width; every other page keeps the
            comfortable --page-pad gutter. */}
        <div style={{ padding: isSubPage ? 0 : 'var(--page-pad)', flex: 1, minWidth: 0 }}>
          {children}
        </div>
      </main>

      {/* ═══ Mobile Bottom Tab Bar — iOS-style ═══ */}
      <nav
        className="mobile-bottom-nav"
        style={{
          display: 'none',
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0,
          height: 'calc(56px + var(--safe-bottom))',
          paddingBottom: 'var(--safe-bottom)',
          background: 'var(--bg-panel)',
          backdropFilter: 'blur(24px) saturate(180%)',
          WebkitBackdropFilter: 'blur(24px) saturate(180%)',
          borderTop: '1px solid var(--border)',
          alignItems: 'flex-start',
          justifyContent: 'space-around',
          paddingTop: 6,
          zIndex: 50,
        }}
      >
        {NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path
            || (item.path === '/clips' && location.pathname.startsWith('/clips'))
            || (item.path === '/logs' && location.pathname.startsWith('/logs'))
            || (item.path === '/' && location.pathname.startsWith('/analysis'))
            || (item.path === '/clips' && location.pathname.startsWith('/seo'));
          const isLogs = item.path === '/logs';
          return (
            <Link
              key={item.path}
              to={item.path}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                textDecoration: 'none',
                color: isActive ? 'var(--accent-cyan)' : 'var(--text-muted)',
                fontSize: 10,
                fontWeight: isActive ? 600 : 400,
                padding: '2px 12px',
                transition: 'color 0.15s',
                minWidth: 56,
                position: 'relative',
                letterSpacing: '-0.01em',
              }}
            >
              <span style={{ position: 'relative', lineHeight: 1 }}>
                <MobileTabIcon type={item.mobileIcon} active={isActive} />
                {isLogs && activeCount > 0 && (
                  <span style={{
                    position: 'absolute', top: -3, right: -8,
                    minWidth: 14, height: 14, borderRadius: 7,
                    background: 'var(--accent-amber)', color: 'var(--bg-base)',
                    fontSize: 9, fontWeight: 700, fontFamily: 'var(--font-mono)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    padding: '0 3px',
                    animation: 'pulse 1.5s ease-in-out infinite',
                  }}>
                    {activeCount}
                  </span>
                )}
              </span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
