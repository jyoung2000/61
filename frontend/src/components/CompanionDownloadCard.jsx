import { useEffect, useRef, useState } from 'react';
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

// Row for one capability in the test modal.
function CapRow({ ok, label, detail }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 12, margin: '4px 0' }}>
      <span style={{ color: ok ? 'var(--success)' : 'var(--danger)', fontWeight: 700, width: 14 }}>
        {ok ? '✓' : '✗'}
      </span>
      <span>
        <strong>{label}</strong>
        {detail && <span style={{ color: 'var(--text-muted)' }}> — {detail}</span>}
      </span>
    </div>
  );
}

// An unmissable modal that always shows the GPU test's live state and result.
function TestModal({ verifying, result, error, onClose, onRetry }) {
  const w = result?.whisper || {};
  const wOk = !!w.transcribed;
  const overlay = {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 9999,
    display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
  };
  const card = {
    background: 'var(--bg-panel)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)', padding: 20, maxWidth: 480, width: '100%',
    boxShadow: 'var(--shadow-lg, 0 20px 60px rgba(0,0,0,0.4))', color: 'var(--text-primary)',
  };
  return (
    <div style={overlay} onClick={onClose}>
      <div style={card} onClick={(e) => e.stopPropagation()}>
        <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 10 }}>GPU connection test</div>

        {verifying && (
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            <div style={{ marginBottom: 8 }}>Running a real job on the Companion…</div>
            {/* Indeterminate loading bar (not a spinner). */}
            <style>{`@keyframes clipaiIndet { 0% { left: -40%; } 100% { left: 100%; } }`}</style>
            <div style={{
              position: 'relative', height: 6, borderRadius: 3,
              background: 'var(--bg-elevated)', overflow: 'hidden',
            }}>
              <div style={{
                position: 'absolute', top: 0, bottom: 0, width: '40%', borderRadius: 3,
                background: 'var(--accent-cyan)', animation: 'clipaiIndet 1.1s ease-in-out infinite',
              }} />
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
              This can take up to ~2 minutes if the GPU is cold — it loads each model and
              transcribes a test clip. Leave this open.
            </div>
          </div>
        )}

        {!verifying && error && (
          <div style={{ fontSize: 13, color: 'var(--danger)', lineHeight: 1.5 }}>{error}</div>
        )}

        {!verifying && !error && result && (
          <>
            <div style={{
              fontSize: 14, fontWeight: 700, marginBottom: 10,
              color: result.verified ? 'var(--success)' : 'var(--accent-amber)',
            }}>
              {result.verified
                ? `PASS — everything AI runs on ${result.host?.gpu_name || result.host?.name || 'the Companion'}`
                : 'PARTIAL — some work will fall back to this server'}
            </div>

            <CapRow ok={result.ollama_ok} label="LLM + vision (Ollama)"
              detail={result.ollama_ok ? 'runs on the Companion GPU' : 'not fully working'} />
            {(result.models || []).map((m) => (
              <div key={m.model} style={{ paddingLeft: 22 }}>
                <CapRow ok={m.ok} label={m.model} detail={m.detail} />
              </div>
            ))}
            <CapRow ok={wOk} label="Whisper transcription"
              detail={wOk
                ? 'a test clip transcribed on the Companion GPU'
                : (w.configured
                  ? `test transcription failed${w.error ? `: ${w.error}` : ''} — runs on this server`
                  : `not offloaded${w.error ? ` (${w.error})` : ''} — runs on this server`)} />

            <div style={{
              fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5,
              marginTop: 12, paddingTop: 10, borderTop: '1px solid var(--border)',
            }}>
              Video decode, frame extraction and encoding always run on the ClipAI
              server's own GPU — the source video lives here, so those can't move to the
              Companion. The Companion GPU handles the heavy AI: vision, text and
              transcription.
            </div>
          </>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 16 }}>
          {!verifying && (
            <button type="button" onClick={onRetry} style={btnStyle('secondary')}>Re-test</button>
          )}
          <button type="button" onClick={onClose} style={btnStyle('primary')}>Close</button>
        </div>
      </div>
    </div>
  );
}

export default function CompanionDownloadCard({ isMobile = false }) {
  const [manifest, setManifest] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState(null);
  const [showTest, setShowTest] = useState(false);
  const [testError, setTestError] = useState('');
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
      } else if (data.status === 'local') {
        // From-source setup: the meaningful check is served-vs-paired-Companion,
        // not GitHub. "Update available" is good news, not an error.
        setRemote(data.companion || null);
        showToast(data.message,
          data.companion?.update_available ? 'success' : 'info');
      } else {
        showToast(data.message || 'No installers to fetch yet', 'error');
      }
    } catch {
      showToast('Update check failed', 'error');
    } finally {
      setRefreshing(false);
    }
  };

  // ── Remote Companion update: push the installer from here, watch progress.
  // The Companion downloads from THIS server, verifies, installs silently and
  // relaunches — nobody needs to be at the GPU PC. Expect a short offline
  // window ("restarting") while the installer swaps files.
  const [remote, setRemote] = useState(null);       // companion info from refresh
  const [push, setPush] = useState(null);           // {phase,pct,mb,totalMb,error,version}
  const pushTimer = useRef(null);
  const pushStarted = useRef(0);
  useEffect(() => () => clearInterval(pushTimer.current), []);

  // Quiet mount-time check so the "update available" banner appears without a
  // click. Uses the side-effect-free status endpoint (NOT /refresh, which can
  // kick off GitHub downloads).
  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        const res = await fetch('/api/downloads/companion/push-update/status');
        const st = await res.json();
        if ((st.state === 'idle' || st.state === 'done')
            && st.companion_version && st.up_to_date === false) {
          setRemote({ update_available: true, version: st.companion_version, reachable: true });
        }
      } catch { /* banner is best-effort */ }
    }, 1500);
    return () => clearTimeout(t);
  }, []);

  const pollPush = async () => {
    try {
      const res = await fetch('/api/downloads/companion/push-update/status');
      const st = await res.json();
      const phase = st.state || 'restarting';
      if (phase === 'done') {
        clearInterval(pushTimer.current);
        setPush({ phase: 'done', version: st.companion_version, build: st.companion_build });
        setRemote((r) => (r ? { ...r, update_available: false } : r));
        showToast(`Companion updated to v${st.companion_version}${st.companion_build ? ` (build ${st.companion_build})` : ''} ✓`, 'success');
        return;
      }
      if (phase === 'failed') {
        clearInterval(pushTimer.current);
        setPush({ phase: 'failed', error: st.error || 'update failed on the Companion' });
        return;
      }
      // Give the offline window a generous but finite budget (~6 min).
      if (Date.now() - pushStarted.current > 6 * 60 * 1000) {
        clearInterval(pushTimer.current);
        setPush({
          phase: 'failed',
          error: "The Companion hasn't come back after the installer ran — check the GPU PC (the app may need a manual launch once).",
        });
        return;
      }
      setPush({
        phase,
        pct: st.progress_pct,
        mb: st.downloaded_mb,
        totalMb: st.total_mb,
      });
    } catch {
      setPush({ phase: 'restarting' });
    }
  };

  const startRemoteUpdate = async () => {
    setPush({ phase: 'starting' });
    try {
      const res = await fetch('/api/downloads/companion/push-update', { method: 'POST' });
      const data = await res.json();
      if (data.status !== 'started') {
        setPush({ phase: 'failed', error: data.message || 'could not start the update' });
        showToast(data.message || 'Could not start the Companion update', data.status === 'busy' ? 'info' : 'error');
        return;
      }
      pushStarted.current = Date.now();
      setPush({ phase: 'downloading', pct: 0 });
      pushTimer.current = setInterval(pollPush, 2500);
    } catch (e) {
      setPush({ phase: 'failed', error: String(e) });
    }
  };

  // Remote "Force end all jobs": cancel every active analysis job on this
  // server AND make the paired Companion(s) drop everything — active-job
  // display cleared, whisper sidecar killed even mid-decode, all Ollama
  // models evicted. The remote sibling of the Companion GUI's local
  // "Force end" button, for when a job wedges the GPU PC.
  const [forceEnding, setForceEnding] = useState(false);
  const [forceEndResult, setForceEndResult] = useState(null);
  const forceEndAll = async () => {
    if (!window.confirm(
      'Force end ALL jobs?\n\nEvery running analysis on this server is '
      + 'cancelled, and the GPU Companion stops all work and frees its VRAM '
      + '(Whisper stopped, AI models unloaded). This cannot be undone.')) return;
    setForceEnding(true);
    setForceEndResult(null);
    try {
      const res = await fetch('/api/providers/companion/force-end-jobs', { method: 'POST' });
      if (!res.ok) {
        setForceEndResult({ error: `Server error (HTTP ${res.status})` });
        showToast(`Force end failed (HTTP ${res.status})`, 'error');
        return;
      }
      const data = await res.json();
      setForceEndResult(data);
      const nJobs = (data.jobs_cancelled || []).length;
      const comps = data.companions || [];
      const okComp = comps.filter((c) => c.ok).length;
      const unloaded = comps.reduce((s, c) => s + (c.ollama_unloaded || 0), 0);
      showToast(
        `Force-ended ${nJobs} job(s)`
        + (comps.length ? ` · ${okComp}/${comps.length} Companion(s) freed (${unloaded} model(s) unloaded)` : ''),
        okComp === comps.length ? 'success' : 'error');
    } catch (e) {
      setForceEndResult({ error: String(e) });
      showToast(`Force end failed: ${e}`, 'error');
    } finally {
      setForceEnding(false);
    }
  };

  // ── Remote vision-offload install: have the Companion set up the
  // face-detection sidecar FROM SOURCE (Python, torch, deps, and the model
  // weight it streams back from this server) — no GitHub, nobody at the GPU PC.
  const [vision, setVision] = useState(null); // {phase,percent,message,error} | {phase:'running'}
  const visionTimer = useRef(null);
  const visionStarted = useRef(0);
  useEffect(() => () => clearInterval(visionTimer.current), []);

  // Quiet mount check so the card shows "running" when it's already installed.
  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        const res = await fetch('/api/downloads/companion/vision-install/status');
        const st = await res.json();
        if (st.healthy && !st.active) setVision({ phase: 'running' });
        else if (st.active) { setVision({ phase: st.stage, percent: st.percent, message: st.message }); startVisionPoll(); }
      } catch { /* best-effort */ }
    }, 1600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pollVision = async () => {
    try {
      const res = await fetch('/api/downloads/companion/vision-install/status');
      const st = await res.json();
      if (st.state === 'unsupported') {
        clearInterval(visionTimer.current);
        setVision({ phase: 'failed', error: 'This Companion build is too old — update the Companion app, then retry.' });
        return;
      }
      if (st.error) {
        clearInterval(visionTimer.current);
        setVision({ phase: 'failed', error: st.error });
        return;
      }
      if (st.healthy && !st.active) {
        clearInterval(visionTimer.current);
        setVision({ phase: 'running' });
        showToast('Vision offload installed — face detection now runs on the Companion GPU ✓', 'success');
        return;
      }
      // Installs (esp. torch) can take many minutes; give a generous budget.
      if (Date.now() - visionStarted.current > 30 * 60 * 1000) {
        clearInterval(visionTimer.current);
        setVision({ phase: 'failed', error: 'Install is taking unusually long — check the Companion’s Processing Log.' });
        return;
      }
      setVision({ phase: st.stage || 'installing', percent: st.percent, message: st.message });
    } catch {
      setVision((v) => v || { phase: 'installing' });
    }
  };

  function startVisionPoll() {
    clearInterval(visionTimer.current);
    visionStarted.current = Date.now();
    visionTimer.current = setInterval(pollVision, 2500);
  }

  const startVisionInstall = async () => {
    setVision({ phase: 'starting', percent: 2 });
    try {
      const res = await fetch('/api/downloads/companion/vision-install', { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        setVision({ phase: 'failed', error: data.detail || `Could not start (HTTP ${res.status})` });
        showToast(data.detail || 'Could not start the vision install', 'error');
        return;
      }
      startVisionPoll();
    } catch (e) {
      setVision({ phase: 'failed', error: String(e) });
    }
  };

  const [removingVision, setRemovingVision] = useState(false);
  const removeVisionInstall = async () => {
    if (!window.confirm(
      'Remove the vision offload from the Companion?\n\nFace detection returns '
      + 'to running on the ClipAI server GPU. You can reinstall any time.')) return;
    setRemovingVision(true);
    clearInterval(visionTimer.current);
    try {
      const res = await fetch('/api/downloads/companion/vision-uninstall', { method: 'POST' });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        showToast(data.detail || `Could not remove (HTTP ${res.status})`, 'error');
        return;
      }
      setVision(null);
      showToast('Vision offload removed — face detection runs on the server again', 'success');
    } catch (e) {
      showToast(`Could not remove: ${e}`, 'error');
    } finally {
      setRemovingVision(false);
    }
  };

  // Round-trip proof that work actually runs on the paired Companion GPU —
  // not just that its token authenticates.
  const verifyOffload = async () => {
    setVerifying(true);
    setVerifyResult(null);
    setTestError('');
    setShowTest(true);   // open the modal immediately so there's always feedback
    try {
      const res = await fetch('/api/settings/companion-verify', { method: 'POST' });
      if (res.status === 404) {
        setTestError('No paired GPU Companion found — add/pair one first.');
        return;
      }
      if (!res.ok) {
        setTestError(`Server error (HTTP ${res.status}).`);
        return;
      }
      const data = await res.json();
      setVerifyResult(data);
      showToast(
        data.verified ? 'GPU test passed — inference + transcription run on the Companion ✓'
          : 'GPU test incomplete — some work falls back to this server (see details)',
        data.verified ? 'success' : 'error');
    } catch (e) {
      setTestError(`Could not reach the ClipAI backend: ${e}`);
    } finally {
      setVerifying(false);
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
        <button type="button" onClick={checkUpdates} disabled={refreshing}
          style={{ ...btnStyle('secondary'), opacity: refreshing ? 0.5 : 1 }}
          title="Pull the latest installers from GitHub into this container so downloads are served locally">
          {refreshing ? 'Fetching…' : anyLocal ? 'Check for updates' : 'Fetch from GitHub'}
        </button>
        <button type="button" onClick={verifyOffload} disabled={verifying}
          style={{ ...btnStyle('primary'), opacity: verifying ? 0.5 : 1 }}
          title="Run a real round-trip on the paired Companion — a 1-token generation on each model AND an actual test transcription — to confirm work truly runs on its GPU before you rely on it">
          {verifying ? 'Testing…' : 'Test GPU connection'}
        </button>
        <button type="button" onClick={forceEndAll} disabled={forceEnding}
          style={{
            ...btnStyle('secondary'), opacity: forceEnding ? 0.5 : 1,
            color: 'var(--danger, #ef4444)',
            borderColor: 'var(--danger, #ef4444)',
          }}
          title="Cancel every running analysis on this server AND make the Companion drop all work: whisper stopped (even mid-transcription), all AI models unloaded, its active-job display cleared">
          {forceEnding ? 'Ending…' : '⏹ Force end all jobs'}
        </button>
      </div>

      {forceEndResult && (
        <div style={{
          padding: '8px 10px', marginBottom: 8, borderRadius: 8, fontSize: 11,
          background: forceEndResult.error ? 'var(--amber-dim)' : 'var(--success-dim)',
          border: `1px solid ${forceEndResult.error ? 'var(--accent-amber)' : 'var(--success)'}`,
          color: 'var(--text-secondary)',
        }}>
          {forceEndResult.error ? (
            <>Force end failed: {forceEndResult.error}</>
          ) : (
            <>
              Force-ended {(forceEndResult.jobs_cancelled || []).length} job(s).
              {(forceEndResult.companions || []).map((c) => (
                <span key={c.host_id} style={{ display: 'block', marginTop: 2 }}>
                  {c.name}: {c.ok
                    ? `freed ✓ (whisper ${c.whisper_stopped ? 'stopped' : 'was idle'}, ${c.ollama_unloaded} model(s) unloaded${c.ended_job ? `, ended "${c.ended_job}"` : ''})`
                    : (c.error || 'unreachable')}
                </span>
              ))}
            </>
          )}
        </div>
      )}

      {remote?.update_available && (!push || push.phase === 'failed') && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
          padding: '8px 10px', marginBottom: 8, borderRadius: 8,
          background: 'var(--success-dim)', border: '1px solid var(--success)',
        }}>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1, minWidth: 180 }}>
            The Companion on your GPU PC (v{remote.version || '?'}) is older than the
            build this server hosts — it can be updated from here, no one needed at the PC.
          </span>
          <button type="button" onClick={startRemoteUpdate} style={btnStyle('primary')}
            title="The Companion downloads this server's installer, verifies it, installs silently and relaunches itself">
            ⬆ Update Companion now
          </button>
        </div>
      )}
      {push && push.phase !== 'done' && push.phase !== 'failed' && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: 'var(--text-muted)', marginBottom: 3 }}>
            <span>
              {push.phase === 'starting' && 'Contacting the Companion…'}
              {push.phase === 'downloading' && `Companion downloading the installer${push.totalMb ? ` — ${(push.mb || 0).toFixed(0)} / ${push.totalMb.toFixed(0)} MB` : '…'}`}
              {push.phase === 'verifying' && 'Verifying installer (sha256)…'}
              {push.phase === 'launching' && 'Installer starting on the GPU PC…'}
              {push.phase === 'restarting' && 'Installing + restarting the Companion… (it goes briefly offline — this is normal)'}
              {!['starting', 'downloading', 'verifying', 'launching', 'restarting'].includes(push.phase) && 'Updating…'}
            </span>
            {push.phase === 'downloading' && push.pct >= 0 && <span>{Math.round(push.pct)}%</span>}
          </div>
          <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-inset, var(--border))', overflow: 'hidden' }}>
            <div style={{
              height: '100%', borderRadius: 3, background: 'var(--accent-cyan)',
              transition: 'width .4s ease',
              width: push.phase === 'downloading' && push.pct >= 0 ? `${Math.max(3, push.pct)}%` : '100%',
              // Indeterminate phases pulse instead of pretending to know a %.
              animation: push.phase !== 'downloading' ? 'companionPulse 1.4s ease-in-out infinite' : 'none',
              opacity: push.phase !== 'downloading' ? 0.75 : 1,
            }} />
          </div>
          <style>{'@keyframes companionPulse { 0%,100% { opacity:.35 } 50% { opacity:.9 } }'}</style>
        </div>
      )}
      {push?.phase === 'done' && (
        <div style={{ fontSize: 11, color: 'var(--success)', marginBottom: 8 }}>
          ✓ Companion updated to v{push.version}{push.build ? ` (build ${push.build})` : ''} and back online.
        </div>
      )}
      {push?.phase === 'failed' && (
        <div style={{ fontSize: 11, color: 'var(--accent-red, #e5484d)', marginBottom: 8, lineHeight: 1.4 }}>
          Remote update didn't finish: {push.error}
        </div>
      )}

      {/* ── Vision (face-detection) offload install ─────────────────────── */}
      <div style={{
        padding: '8px 10px', marginBottom: 8, borderRadius: 8,
        background: 'var(--bg-inset, var(--bg-secondary))', border: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1, minWidth: 200, lineHeight: 1.4 }}>
            <strong style={{ color: 'var(--text-primary)' }}>Vision offload</strong>
            {vision?.phase === 'running'
              ? ' — installed. ClipAI’s heaviest stage (face/subject detection) runs on the Companion GPU.'
              : ' — run ClipAI’s heaviest stage (face/subject detection) on the Companion GPU. The Companion installs everything itself (Python, PyTorch, the model) — no GitHub, nobody at the GPU PC.'}
          </span>
          {vision?.phase !== 'running' && (
            <button type="button" onClick={startVisionInstall}
              disabled={!!vision && vision.phase !== 'failed'}
              style={btnStyle('primary')}
              title="The Companion downloads its prerequisites and the model from this server, then serves face detection on its GPU">
              {vision && vision.phase !== 'failed' ? 'Installing…' : '⬇ Install vision offload'}
            </button>
          )}
        </div>

        {vision && vision.phase !== 'running' && vision.phase !== 'failed' && (
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5, color: 'var(--text-muted)', marginBottom: 3 }}>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {vision.message || {
                  starting: 'Contacting the Companion…',
                  preparing: 'Preparing…',
                  python: 'Setting up Python…',
                  venv: 'Creating the Python environment…',
                  torch: 'Installing PyTorch (CUDA) — the big one…',
                  deps: 'Installing YOLO-World + dependencies…',
                  model: 'Downloading the model from this server…',
                  starting_sidecar: 'Starting the vision sidecar…',
                }[vision.phase] || 'Installing…'}
              </span>
              {typeof vision.percent === 'number' && vision.percent >= 0 && (
                <span style={{ marginLeft: 8 }}>{Math.round(vision.percent)}%</span>
              )}
            </div>
            <div style={{ height: 6, borderRadius: 3, background: 'var(--bg-inset, var(--border))', overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 3, background: 'var(--accent-cyan)',
                transition: 'width .4s ease',
                width: typeof vision.percent === 'number' && vision.percent >= 0 ? `${Math.max(3, vision.percent)}%` : '100%',
                animation: !(typeof vision.percent === 'number' && vision.percent >= 0) ? 'companionPulse 1.4s ease-in-out infinite' : 'none',
                opacity: typeof vision.percent === 'number' && vision.percent >= 0 ? 1 : 0.75,
              }} />
            </div>
            <style>{'@keyframes companionPulse { 0%,100% { opacity:.35 } 50% { opacity:.9 } }'}</style>
          </div>
        )}
        {vision?.phase === 'running' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginTop: 6 }}>
            <span style={{ fontSize: 11, color: 'var(--success)', flex: 1, minWidth: 180 }}>
              ⚡ Installed on the Companion GPU. Note: if Whisper also runs on that
              same Companion, ClipAI keeps face detection local so the two don't
              compete — set REMOTE_VISION_ALLOW_WITH_REMOTE_WHISPER=1 to override.
            </span>
            <button type="button" onClick={removeVisionInstall} disabled={removingVision}
              style={btnStyle('secondary')}
              title="Stop and delete the vision offload; face detection returns to the server GPU">
              {removingVision ? 'Removing…' : '✕ Remove'}
            </button>
          </div>
        )}
        {vision?.phase === 'failed' && (
          <div style={{ fontSize: 11, color: 'var(--accent-red, #e5484d)', marginTop: 6, lineHeight: 1.4 }}>
            Install didn’t finish: {vision.error}
          </div>
        )}
      </div>

      {showTest && (
        <TestModal
          verifying={verifying}
          result={verifyResult}
          error={testError}
          onClose={() => setShowTest(false)}
          onRetry={verifyOffload}
        />
      )}

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
