import { useEffect, useState } from 'react';
import { showToast } from './Toast';

// GPU Companion — download card at the top of the Ollama section.
// BOTH platform buttons are ALWAYS shown. When an installer exists it
// downloads from this container (or 302-redirects to the GitHub asset);
// when it doesn't, the button links to the GitHub releases page so there
// is always a visible path — never a dead "check GitHub" state.

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

const btnStyle = (variant) => ({
  padding: '7px 14px',
  background: variant === 'primary' ? 'var(--accent-cyan)'
    : variant === 'ghost' ? 'transparent' : 'var(--bg-elevated)',
  color: variant === 'primary' ? 'var(--bg-base)' : 'var(--text-secondary)',
  border: variant === 'primary' ? 'none' : '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
  textDecoration: 'none', display: 'inline-block', cursor: 'pointer',
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
        showToast(`Fetching Companion installers (v${data.target_version || '?'})…`, 'success');
        setTimeout(load, 5000);
      } else {
        showToast(data.message || 'No installers to fetch yet', 'error');
      }
    } catch {
      showToast('Update check failed', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  const platforms = manifest?.platforms || {};
  const repo = manifest?.github_repo || '';
  const releasesUrl = repo ? `https://github.com/${repo}/releases` : '';
  const anyLocal = ['windows', 'mac'].some(
    (k) => platforms[k] && platforms[k].source !== 'github-only');
  const anyAvailable = !!(platforms.windows || platforms.mac);

  // One entry per platform, ALWAYS present.
  const specs = [
    { key: 'windows', os: 'windows', ext: '.exe', label: 'Windows' },
    { key: 'mac', os: 'mac', ext: '.dmg', label: 'macOS' },
  ].map((s) => {
    const p = platforms[s.key];
    return {
      ...s,
      available: !!p,
      size: p?.size,
      href: p ? `/api/downloads/companion/${s.key}` : releasesUrl,
      external: !p,
    };
  }).sort((a, b) => (b.os === os ? 1 : 0) - (a.os === os ? 1 : 0));

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

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        {specs.map((s) => (
          <a
            key={s.key}
            href={s.href || '#'}
            {...(s.external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
            onClick={(e) => {
              if (!s.href) { e.preventDefault(); return; }
              if (s.available) setDownloaded(true);
            }}
            title={s.available
              ? `Download the ${s.label} installer`
              : `Not built yet — opens the GitHub releases page`}
            style={{
              ...btnStyle(s.available && s.os === os ? 'primary' : (s.available ? 'secondary' : 'ghost')),
              opacity: s.available ? 1 : 0.75,
            }}
          >
            {s.available
              ? `Download for ${s.label} (${s.ext})${s.size ? ` — ${fmtSize(s.size)}` : ''}`
              : `${s.label} (${s.ext}) — on GitHub ↗`}
          </a>
        ))}
        <button onClick={checkUpdates} disabled={refreshing}
          style={{ ...btnStyle('secondary'), opacity: refreshing ? 0.5 : 1 }}
          title="Pull the latest installers from GitHub into this container so downloads are served locally">
          {refreshing ? 'Fetching…' : anyLocal ? 'Check for updates' : 'Fetch from GitHub'}
        </button>
      </div>

      {!anyAvailable && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.4 }}>
          No installer is published yet. The buttons above open the GitHub releases page;
          once a <code>companion-v*</code> release exists, click <strong>Fetch from
          GitHub</strong> and the installers will be served directly by this server.
        </div>
      )}
      {manifest?.built_from_source && (
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8, lineHeight: 1.4 }}>
          This Windows installer was built from source inside this server's Docker image.
          It shares your desktop GPU's Ollama fully; the Whisper sidecar ships with official
          <code>companion-v*</code> releases — transcription stays on this server until then.
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
