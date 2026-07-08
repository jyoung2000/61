/**
 * Segmented-control tab strip for the Upload page.
 *
 *   [ Local file ] [ Google Drive ] [ Box ]
 *
 * "Local file" is the default and leaves the existing drop zone untouched.
 * Clicking "Google Drive" or "Box" either opens the CloudFilePicker (if the
 * provider is connected) or prompts the user to connect it in Settings.
 *
 * When a file is picked, we POST to /api/cloud/{provider}/import, receive
 * a job_id, and navigate to the processing view via the supplied onJobStart
 * callback — which is wired in Upload.jsx to call the same navigate() the
 * local upload flow already uses.
 */
import React, { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import useCloudProviders from '../../hooks/useCloudProviders';
import CloudFilePicker from './CloudFilePicker';
import './FileBrowser.css';

const TABS = [
  { value: 'local', label: 'Local file', providerName: null, icon: 'local' },
  { value: 'google_drive', label: 'Google Drive', providerName: 'google_drive', icon: 'cloud' },
  { value: 'box', label: 'Box', providerName: 'box', icon: 'box' },
];

const PROVIDER_LABELS = {
  google_drive: 'Google Drive',
  box: 'Box',
};

// Small inline provider glyphs, matching the file-browser design system.
function SourceGlyph({ kind }) {
  const common = { width: 16, height: 16, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round', style: { display: 'block' } };
  if (kind === 'cloud') return (<svg {...common}><path d="M17.5 19a4.5 4.5 0 1 0-1.5-8.74A6 6 0 1 0 6.5 19z" /></svg>);
  if (kind === 'box') return (<svg {...common}><path d="M22 12H2" /><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" /><path d="M6 16h.01" /><path d="M10 16h.01" /></svg>);
  return (<svg {...common}><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8" /><path d="M12 17v4" /></svg>);
}

function tabButtonStyle(active) {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 7,
    padding: '8px 13px',
    fontSize: 13,
    fontWeight: 600,
    background: active ? 'var(--fb-sel)' : 'var(--fb-field)',
    color: active ? 'var(--fb-accent)' : 'var(--fb-ts)',
    border: `1px solid ${active ? 'var(--fb-accent)' : 'transparent'}`,
    borderRadius: 10,
    cursor: 'pointer',
    transition: 'all 120ms ease',
    whiteSpace: 'nowrap',
  };
}

export default function CloudSourceTabs({ uploadMetadata, onJobStart }) {
  const { enabled, providers } = useCloudProviders();
  const [active, setActive] = useState('local');
  const [pickerFor, setPickerFor] = useState(null);
  const [error, setError] = useState(null);

  const providerByName = Object.fromEntries(
    (providers || []).map((p) => [p.name, p])
  );

  // NOTE: every hook below must be declared unconditionally so React's
  // hook ordering stays stable across renders. The original version
  // early-returned on ``!enabled`` BEFORE these useCallbacks ran —
  // that flipped the hook count between the initial (loading) render
  // and the post-fetch render, triggering React error #310 ("rendered
  // more hooks than during the previous render") on any deployment
  // that doesn't have cloud storage configured.
  const handleTabClick = useCallback(
    (tab) => {
      setError(null);
      if (tab.value === 'local') {
        setActive('local');
        setPickerFor(null);
        return;
      }
      const provider = providerByName[tab.providerName];
      if (!provider || !provider.configured) {
        setError(
          `${PROVIDER_LABELS[tab.providerName]} is not configured on the server. See the setup guide.`
        );
        return;
      }
      if (!provider.connected) {
        setActive(tab.value);
        setPickerFor(null);
        return;
      }
      setActive(tab.value);
      setPickerFor(tab.providerName);
    },
    [providerByName]
  );

  const handlePick = useCallback(
    async (file) => {
      if (!pickerFor) return;
      setError(null);
      try {
        const body = {
          file_id: file.id,
          ...(uploadMetadata || {}),
        };
        const resp = await fetch(`/api/cloud/${pickerFor}/import`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || `HTTP ${resp.status}`);
        }
        const { job_id } = await resp.json();
        setPickerFor(null);
        if (onJobStart) onJobStart(job_id);
      } catch (err) {
        setError(err.message || 'Import failed');
      }
    },
    [pickerFor, uploadMetadata, onJobStart]
  );

  // If cloud storage is disabled on the server, render nothing — the
  // existing drop zone is still fully functional on its own. This
  // return MUST live after every hook above.
  if (!enabled) {
    return null;
  }

  // Also hide the tab strip entirely when zero cloud providers have
  // been configured. Showing a lone "Local file" tab with nothing
  // next to it is awkward, and the local drop zone below us is
  // already the default experience for users who don't want cloud.
  const anyCloudConfigured = (providers || []).some((p) => p?.configured);
  if (!anyCloudConfigured) {
    return null;
  }

  const notConnectedProvider = (() => {
    if (active === 'local') return null;
    const provider = providerByName[active];
    if (!provider) return null;
    if (!provider.configured || provider.connected) return null;
    return provider;
  })();

  return (
    <div className="fb-scope" style={{ marginBottom: 16 }}>
      <div
        role="tablist"
        aria-label="Video source"
        style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}
      >
        {TABS.map((tab) => {
          const isCloud = tab.value !== 'local';
          const provider = isCloud ? providerByName[tab.providerName] : null;
          // Always show Local. Only show cloud tabs for providers that
          // the server has actually configured.
          if (isCloud && (!provider || !provider.configured)) return null;
          return (
            <button
              key={tab.value}
              type="button"
              role="tab"
              aria-selected={active === tab.value}
              onClick={() => handleTabClick(tab)}
              style={tabButtonStyle(active === tab.value)}
            >
              <SourceGlyph kind={tab.icon} />
              {tab.label}
              {isCloud && provider?.connected && (
                <span
                  title="Connected"
                  style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--success, #34C759)', flex: 'none' }}
                />
              )}
            </button>
          );
        })}
      </div>

      {notConnectedProvider && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            border: '1px dashed var(--border)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--text-secondary)',
            fontSize: 13,
          }}
        >
          {PROVIDER_LABELS[notConnectedProvider.name]} is not connected yet.{' '}
          <Link to="/settings" style={{ color: 'var(--accent-cyan)' }}>
            Connect it in Settings &rarr;
          </Link>
        </div>
      )}

      {error && (
        <div
          style={{
            marginTop: 12,
            padding: 10,
            background: 'var(--bg-panel)',
            border: '1px solid var(--accent-red, #f66)',
            borderRadius: 'var(--radius-sm)',
            color: 'var(--accent-red, #f66)',
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      {pickerFor && (
        <CloudFilePicker
          provider={pickerFor}
          providerLabel={PROVIDER_LABELS[pickerFor]}
          account={providerByName[pickerFor]?.account?.display_name}
          onPick={handlePick}
          onClose={() => setPickerFor(null)}
        />
      )}
    </div>
  );
}
