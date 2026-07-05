import { useEffect, useState } from 'react';
import { showToast } from './Toast';

// GPU Companion — download card at the top of the Ollama section.
// Installers are served by THIS ClipAI container (baked into the image or
// cached under /config/companion-cache), with a labeled GitHub redirect
// when neither copy exists yet.

const fmtSize = (bytes) => {
  if (!bytes) return '';
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
};

const detectOS = () => {
  const ua = (navigator.userAgent || '').toLowerCase();
  if (ua.includes('mac')) return 'mac';
  return 'windows';
};

const btnStyle = (primary) => ({
  padding: '7px 14px',
  background: primary ? 'var(--accent-cyan)' : 'var(--bg-elevated)',
  color: primary ? 'var(--bg-base)' : 'var(--text-secondary)',
  border: primary ? 'none' : '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
  textDecoration: 'none', display: 'inline-block',
});

export default function CompanionDownloadCard({ isMobile = false }) {
  const [manifest, setManifest] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const os = detectOS();

  const load = () => {
    fetch('/api/downloads/companion/manifest')
      .then((r) => (r.ok ? r.json() : null))
      .then(setManifest)
      .catch(() => {});
  };
  useEffect(load, []);

  // While a cache refresh is downloading installers, keep polling so the
  // buttons flip from "From GitHub" to "Hosted by this server" on finish.
  useEffect(() => {
    if (!manifest?.refresh?.active) return undefined;
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [manifest?.refresh?.active]);

  const checkUpdates = async () => {
    setRefreshing(true);
    try {
      const res = await fetch('/api/downloads/companion/refresh', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'started') {
        showToast(`Checking for Companion updates (v${data.target_version || '?'})…`, 'success');
        setTimeout(load, 5000);
      } else {
        showToast(data.message || 'Nothing to fetch', 'error');
      }
    } catch {
      showToast('Update check failed', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  const platforms = manifest?.platforms || {};
  const win = platforms.windows;
  const mac = platforms.mac;
  const anyLocal = [win, mac].some((p) => p && p.source !== 'github-only');
  const anyAvailable = !!(win || mac);

  const buttons = [
    win && {
      key: 'windows',
      label: `Download for Windows (.exe)${win.size ? ` — ${fmtSize(win.size)}` : ''}`,
      href: '/api/downloads/companion/windows',
      primary: os === 'windows',
    },
    mac && {
      key: 'mac',
      label: `Download for macOS (.dmg)${mac.size ? ` — ${fmtSize(mac.size)}` : ''}`,
      href: '/api/downloads/companion/mac',
      primary: os === 'mac',
    },
  ].filter(Boolean).sort((a, b) => (b.primary ? 1 : 0) - (a.primary ? 1 : 0));

  return (
    <div style={{
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: isMobile ? '12px' : '12px 16px',
      boxShadow: 'var(--shadow-sm)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>GPU Companion</span>
        {manifest?.version && (
          <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            v{manifest.version}
          </span>
        )}
        {anyAvailable && (
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase',
            padding: '1px 6px', borderRadius: 8,
            background: anyLocal ? 'var(--success-dim)' : 'var(--amber-dim)',
            color: anyLocal ? 'var(--success)' : 'var(--accent-amber)',
          }}>
            {anyLocal ? 'Hosted by this server' : 'From GitHub'}
          </span>
        )}
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, margin: '0 0 10px' }}>
        Share a desktop GPU with ClipAI — a Windows/Mac app that runs Ollama and Whisper
        on your gaming PC's graphics card and lends them to this server over your network.
        {' '}
        <a href="/docs/remote-gpu.md" target="_blank" rel="noopener noreferrer"
          style={{ color: 'var(--accent-cyan)', textDecoration: 'none' }}>
          Setup guide →
        </a>
      </p>
      {anyAvailable ? (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
          {buttons.map((b) => (
            <a key={b.key} href={b.href} style={btnStyle(b.primary)}
              onClick={() => setDownloaded(true)}>
              {b.label}
            </a>
          ))}
          <button onClick={checkUpdates} disabled={refreshing}
            style={{ ...btnStyle(false), opacity: refreshing ? 0.5 : 1, cursor: 'pointer' }}>
            {refreshing ? 'Checking…' : 'Check for updates'}
          </button>
        </div>
      ) : (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
          No Companion installers are available yet — they're built by the
          <code style={{ margin: '0 4px' }}>companion-v*</code> release workflow.
          <button onClick={checkUpdates} disabled={refreshing}
            style={{ ...btnStyle(false), marginLeft: 8, cursor: 'pointer', opacity: refreshing ? 0.5 : 1 }}>
            {refreshing ? 'Checking…' : 'Check GitHub'}
          </button>
        </div>
      )}
      {downloaded && (
        <div style={{
          fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.6,
          paddingLeft: 8, borderLeft: '2px solid var(--border)',
        }}>
          <div>1. Run the installer on your desktop</div>
          <div>2. Open the GPU Companion — the setup wizard starts automatically</div>
          <div>
            3. Paste this server's address and your ClipAI API key to pair — the desktop
            GPU becomes the primary AI host
          </div>
        </div>
      )}
    </div>
  );
}
