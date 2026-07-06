import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { openUrl } from '@tauri-apps/plugin-opener';
import { enable as enableAutostart, disable as disableAutostart, isEnabled as autostartEnabled } from '@tauri-apps/plugin-autostart';
import {
  CompanionStatus, getStatus, setConfig, regenerateToken,
  installOllama, startOllama, pullModel, pairClipai,
  listModels, deleteModel, InstalledModel,
  downloadWhisper, refreshSidecar,
} from './api';

const fmtMb = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`);
const fmtBytes = (b: number) => {
  if (!b) return '—';
  const gb = b / (1024 ** 3);
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(b / (1024 * 1024))} MB`;
};
const elapsed = (startMs: number, endMs?: number | null) => {
  const s = Math.max(0, Math.floor(((endMs ?? Date.now()) - startMs) / 1000));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

// Poll every 2 s while visible; stop entirely when the window is hidden
// (tray-minimized) so the idle Companion stays under 1% CPU.
function useStatus() {
  const [status, setStatus] = useState<CompanionStatus | null>(null);
  const timer = useRef<number | null>(null);
  const refresh = useCallback(async () => {
    try { setStatus(await getStatus()); } catch { /* backend booting */ }
  }, []);
  useEffect(() => {
    const arm = () => {
      if (timer.current !== null) return;
      refresh();
      timer.current = window.setInterval(refresh, 2000);
    };
    const disarm = () => {
      if (timer.current !== null) { clearInterval(timer.current); timer.current = null; }
    };
    const onVis = () => (document.hidden ? disarm() : arm());
    document.addEventListener('visibilitychange', onVis);
    arm();
    return () => { disarm(); document.removeEventListener('visibilitychange', onVis); };
  }, [refresh]);
  return { status, refresh };
}

function TokenBox({ token, onRegenerate }: { token: string; onRegenerate?: () => void }) {
  const [revealed, setRevealed] = useState(false);
  const copy = () => navigator.clipboard.writeText(token);
  return (
    <div className="token-box">
      <code className="mono">{revealed ? token : '•'.repeat(24)}</code>
      <button className="secondary" onClick={() => setRevealed((r) => !r)}>
        {revealed ? 'Hide' : 'Reveal'}
      </button>
      <button className="secondary" onClick={copy}>Copy</button>
      {onRegenerate && <button className="secondary" onClick={onRegenerate}>Regenerate</button>}
    </div>
  );
}

// Determinate or indeterminate progress bar (reuses the .meter styles).
function ProgressBar({ percent, label, indeterminate }: {
  percent?: number; label?: string; indeterminate?: boolean;
}) {
  const indet = indeterminate || percent == null || percent < 0;
  const pct = indet ? 100 : Math.max(3, Math.min(100, percent));
  return (
    <div style={{ margin: '6px 0' }}>
      {label && <div className="small muted" style={{ marginBottom: 3 }}>{label}</div>}
      <div className={`meter${indet ? ' indeterminate' : ''}`}>
        <div style={indet ? undefined : { width: `${pct}%` }} />
      </div>
    </div>
  );
}

// ── First-run wizard ─────────────────────────────────────────────────

function Wizard({ status, refresh, onDone }: {
  status: CompanionStatus; refresh: () => void; onDone: () => void;
}) {
  const [step, setStep] = useState(0);
  const [installing, setInstalling] = useState(false);
  const [installError, setInstallError] = useState('');
  const [installMsg, setInstallMsg] = useState('');
  const [pulling, setPulling] = useState<Record<string, 'pulling' | 'done' | 'error'>>({});
  const [pullProg, setPullProg] = useState<Record<string, { percent: number; status: string }>>({});
  const [clipaiUrl, setClipaiUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [pairBusy, setPairBusy] = useState(false);
  const [pairError, setPairError] = useState('');
  const [pairDone, setPairDone] = useState(false);

  const steps = ['Welcome', 'Ollama', 'Models', 'Access token', 'Connect', 'Done'];
  const ollamaReady = status.ollama.running;

  // Live progress from the backend (install transcript + model-pull bytes).
  useEffect(() => {
    const unInstall = listen<{ stage: string; message: string }>(
      'install-progress', (e) => setInstallMsg(e.payload.message || ''));
    const unPull = listen<{ model: string; status: string; percent: number }>(
      'pull-progress', (e) => {
        const { model, status: st, percent } = e.payload;
        setPullProg((p) => ({ ...p, [model]: { percent, status: st } }));
      });
    return () => { unInstall.then((f) => f()); unPull.then((f) => f()); };
  }, []);

  const doInstall = async () => {
    setInstalling(true);
    setInstallError('');
    try {
      await installOllama();
      await startOllama();
      refresh();
    } catch (e) {
      setInstallError(String(e));
    } finally {
      setInstalling(false);
    }
  };

  const doPull = async (model: string) => {
    setPulling((p) => ({ ...p, [model]: 'pulling' }));
    try {
      await pullModel(model);
      setPulling((p) => ({ ...p, [model]: 'done' }));
      refresh();
    } catch {
      setPulling((p) => ({ ...p, [model]: 'error' }));
    }
  };

  const doPair = async () => {
    setPairBusy(true);
    setPairError('');
    try {
      await pairClipai(clipaiUrl, apiKey);
      setPairDone(true);
      refresh();
    } catch (e) {
      setPairError(String(e));
    } finally {
      setPairBusy(false);
    }
  };

  // Hands-off setup: auto-install Ollama when its step opens, and auto-pull
  // the recommended models once Ollama is up — no clicks required. Manual
  // buttons remain as a fallback.
  const autoInstallTried = useRef(false);
  useEffect(() => {
    if (step === 1 && !ollamaReady && !installing && !autoInstallTried.current) {
      autoInstallTried.current = true;
      doInstall();
    }
  }, [step, ollamaReady, installing]);

  const autoPullTried = useRef(false);
  useEffect(() => {
    if (step === 2 && ollamaReady && !autoPullTried.current) {
      autoPullTried.current = true;
      status.recommended_models.forEach(({ model }) => {
        const have = status.ollama.models.some((m) => m.startsWith(model.split(':')[0]));
        if (!have && pulling[model] !== 'pulling') doPull(model);
      });
    }
  }, [step, ollamaReady]);

  return (
    <div className="wizard">
      <div className="steps">
        {steps.map((_, i) => <span key={i} className={i <= step ? 'done' : ''} />)}
      </div>

      {step === 0 && (
        <div className="panel">
          <h1>Share this GPU with ClipAI</h1>
          <p className="muted">
            This Companion runs Ollama and a Whisper transcription server on
            <strong> {status.gpu.gpu_name || 'your GPU'}</strong> and exposes them to your
            ClipAI server on one authenticated port ({status.config.port}). Your desktop
            stays usable — you control how much GPU memory ClipAI may borrow.
          </p>
          <button onClick={() => setStep(1)}>Get started</button>
        </div>
      )}

      {step === 1 && (
        <div className="panel">
          <h2>Ollama</h2>
          {ollamaReady ? (
            <p className="muted">
              <span className="dot ok" /> Ollama detected
              {status.ollama.version ? ` (${status.ollama.version})` : ''} and running.
            </p>
          ) : (
            <>
              <p className="muted">
                Ollama runs the vision/text models locally. Installing it automatically —
                this can take a couple of minutes. No action needed.
              </p>
              {installing && (
                <ProgressBar indeterminate label={installMsg || 'Installing Ollama…'} />
              )}
              <div className="row">
                <button onClick={doInstall} disabled={installing}>
                  {installing ? 'Installing…' : installError ? 'Retry install' : 'Install Ollama'}
                </button>
                <button className="secondary" onClick={() => openUrl('https://ollama.com/download')}>
                  Open ollama.com/download
                </button>
                <button className="secondary" onClick={() => startOllama().then(refresh).catch(() => {})}>
                  Re-detect
                </button>
              </div>
              {installError && (
                <p className="small" style={{ color: 'var(--danger)' }}>
                  Automatic install failed: {installError} — click “Open ollama.com/download”,
                  install it, then “Re-detect”.
                </p>
              )}
            </>
          )}
          <div className="row" style={{ marginTop: 10 }}>
            <button onClick={() => setStep(2)} disabled={!ollamaReady}>Continue</button>
            <button className="secondary" onClick={() => setStep(2)}>Skip for now</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="panel">
          <h2>Recommended models</h2>
          <p className="muted">
            Sized for your {fmtMb(status.gpu.vram_total_mb)} of GPU memory. Downloading
            these automatically — they're multi-GB, so this can take a while.
          </p>
          {status.recommended_models.map(({ model, why }) => {
            const installed = status.ollama.models.some((m) => m.startsWith(model.split(':')[0]));
            const st = pulling[model];
            const prog = pullProg[model];
            return (
              <div key={model} style={{ marginBottom: 8 }}>
                <div className="row spread">
                  <div>
                    <span className="mono">{model}</span>
                    <span className="muted" style={{ marginLeft: 8 }}>{why}</span>
                  </div>
                  {installed || st === 'done' ? (
                    <span className="badge live">Installed</span>
                  ) : (
                    <button className="secondary" disabled={st === 'pulling' || !ollamaReady}
                      onClick={() => doPull(model)}>
                      {st === 'pulling' ? 'Downloading…' : st === 'error' ? 'Retry' : 'Pull'}
                    </button>
                  )}
                </div>
                {st === 'pulling' && (
                  <ProgressBar
                    percent={prog?.percent}
                    label={prog
                      ? `${prog.status}${prog.percent >= 0 ? ` — ${Math.round(prog.percent)}%` : ''}`
                      : 'starting…'}
                  />
                )}
                {st === 'error' && (
                  <p className="small" style={{ color: 'var(--danger)' }}>
                    Download failed — click Retry.
                  </p>
                )}
              </div>
            );
          })}
          {!ollamaReady && (
            <p className="small" style={{ color: 'var(--warn)' }}>
              Ollama isn't running yet — go back a step so models can download.
            </p>
          )}
          <button style={{ marginTop: 8 }} onClick={() => setStep(3)}>Continue</button>
        </div>
      )}

      {step === 3 && (
        <div className="panel">
          <h2>Access token</h2>
          <p className="muted">
            Every request from ClipAI must carry this token. Pairing (next step) sends it
            automatically — you only need to copy it for manual setup.
          </p>
          <TokenBox token={status.config.token} />
          <button style={{ marginTop: 10 }} onClick={() => setStep(4)}>Continue</button>
        </div>
      )}

      {step === 4 && (
        <div className="panel">
          <h2>Connect to ClipAI</h2>
          <p className="muted">
            Paste your ClipAI server address and its API key (ClipAI Settings → API).
            ClipAI will add this machine as its primary AI host and use it for Whisper.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <input type="text" placeholder="http://tower.local:8000"
              value={clipaiUrl} onChange={(e) => setClipaiUrl(e.target.value)} />
            <input type="password" placeholder="ClipAI API key"
              value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
            <div className="row">
              <button onClick={doPair} disabled={pairBusy || !clipaiUrl.trim() || !apiKey.trim()}>
                {pairBusy ? 'Pairing…' : pairDone ? 'Paired ✓' : 'Pair'}
              </button>
              <button className="secondary" onClick={() => setStep(5)}>
                {pairDone ? 'Continue' : 'Skip — configure manually'}
              </button>
            </div>
            {pairError && <p className="small" style={{ color: 'var(--danger)' }}>{pairError}</p>}
            {pairDone && (
              <p className="small" style={{ color: 'var(--success)' }}>
                Paired — this machine is now ClipAI's primary AI host.
              </p>
            )}
          </div>
        </div>
      )}

      {step === 5 && (
        <div className="panel">
          <h2>All set</h2>
          <p className="muted">
            The Companion lives in your tray. Closing this window keeps it running;
            use the tray menu to pause sharing or quit. Allow inbound TCP&nbsp;
            {status.config.port} on Private networks if your firewall asks.
          </p>
          <button onClick={onDone}>Open dashboard</button>
        </div>
      )}
    </div>
  );
}

// ── Dashboard ────────────────────────────────────────────────────────

function Dashboard({ status, refresh }: { status: CompanionStatus; refresh: () => void }) {
  const [budget, setBudget] = useState<number | null>(null);
  const [autostart, setAutostart] = useState<boolean | null>(null);
  const [pairOpen, setPairOpen] = useState(false);
  const [clipaiUrl, setClipaiUrl] = useState(status.config.paired_clipai_url);
  const [apiKey, setApiKey] = useState('');
  const [pairMsg, setPairMsg] = useState('');
  const [models, setModels] = useState<InstalledModel[]>([]);
  const [confirmDel, setConfirmDel] = useState('');
  const [deleting, setDeleting] = useState('');
  const [whisperDl, setWhisperDl] = useState<{ active: boolean; percent: number; message: string } | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState('');

  useEffect(() => { autostartEnabled().then(setAutostart).catch(() => setAutostart(null)); }, []);

  // Live progress for the Whisper download + model re-sync.
  useEffect(() => {
    const uW = listen<{ stage: string; percent: number; message: string }>('whisper-progress', (e) => {
      const { stage, percent, message } = e.payload;
      setWhisperDl({ active: stage !== 'done' && stage !== 'error', percent, message });
      if (stage === 'done' || stage === 'error') setTimeout(() => setWhisperDl(null), 4000);
    });
    const uP = listen<{ model: string; status: string; percent: number }>('pull-progress', (e) => {
      const { model, status: st, percent } = e.payload;
      setSyncMsg(`${model}: ${st}${percent >= 0 ? ` ${Math.round(percent)}%` : ''}`);
    });
    return () => { uW.then((f) => f()); uP.then((f) => f()); };
  }, []);

  const loadModels = useCallback(() => {
    listModels().then(setModels).catch(() => {});
  }, []);
  // Refresh the installed-model list whenever Ollama's state/model-count changes.
  useEffect(() => { loadModels(); }, [loadModels, status.ollama.running, status.ollama.models.length]);
  const removeModel = async (name: string) => {
    setDeleting(name);
    try {
      await deleteModel(name);
      setConfirmDel('');
      loadModels();
      refresh();
    } catch { /* left installed */ } finally { setDeleting(''); }
  };

  const doDownloadWhisper = async () => {
    setWhisperDl({ active: true, percent: -1, message: 'Starting…' });
    try {
      await downloadWhisper();
      await refreshSidecar();
      refresh();
    } catch (e) {
      setWhisperDl({ active: false, percent: 0, message: `Failed: ${e}` });
      setTimeout(() => setWhisperDl(null), 6000);
    }
  };

  const doRefreshWhisper = async () => {
    try { await refreshSidecar(); } catch {}
    refresh();
  };

  // Re-sync: pull any recommended local models not yet installed (idempotent).
  const doReSync = async () => {
    setSyncing(true);
    setSyncMsg('');
    const missing = (status.recommended_models || [])
      .map((r) => r.model)
      .filter((m) => !status.ollama.models.some((im) => im.startsWith(m.split(':')[0])));
    if (missing.length === 0) {
      setSyncing(false);
      setSyncMsg('All recommended models already installed ✓');
      setTimeout(() => setSyncMsg(''), 5000);
      return;
    }
    const failed = [];
    try {
      for (const model of missing) {
        setSyncMsg(`Downloading ${model}…`);
        try { await pullModel(model); } catch { failed.push(model); }
      }
      loadModels();
      refresh();
    } finally {
      setSyncing(false);
      setSyncMsg(failed.length
        ? `Could not download: ${failed.join(', ')} — check the tag exists on ollama.com`
        : 'Recommended models downloaded ✓');
      setTimeout(() => setSyncMsg(''), 8000);
    }
  };

  const gpu = status.gpu;
  const totalGb = gpu.vram_total_mb / 1024;
  const usedMb = Math.max(0, gpu.vram_total_mb - gpu.vram_free_mb);
  const sliderValue = budget ?? (status.config.vram_budget_gb > 0
    ? status.config.vram_budget_gb : status.effective_budget_gb);
  const sliderMax = Math.max(2, Math.floor(totalGb) - 1);

  const commitBudget = async (v: number) => {
    setBudget(null);
    await setConfig({ vram_budget_gb: v });
    refresh();
  };

  const doPair = async () => {
    setPairMsg('');
    try {
      await pairClipai(clipaiUrl, apiKey);
      setPairMsg('Paired ✓');
      setApiKey('');
      refresh();
    } catch (e) {
      setPairMsg(String(e));
    }
  };

  const job = status.current_job;

  return (
    <div className="app">
      <div className="row spread">
        <h1>ClipAI GPU Companion</h1>
        <div className="row">
          {status.config.paused
            ? <span className="badge">Paused</span>
            : status.busy || job
              ? <span className="badge live">Active</span>
              : <span className="badge">Idle</span>}
          <button className="secondary" onClick={() =>
            setConfig({ paused: !status.config.paused }).then(refresh)}>
            {status.config.paused ? 'Resume sharing' : 'Pause sharing'}
          </button>
        </div>
      </div>

      {job && (
        <div className="panel" style={{ borderColor: 'var(--success)' }}>
          <div className="row spread">
            <div>
              <strong>
                {job.kind === 'whisper' ? 'Transcribing' : 'AI inference'}
                {job.job_title ? ` — ${job.job_title}` : ''}
              </strong>
              <div className="muted small">
                {job.stage ? `stage: ${job.stage} — ` : ''}{elapsed(job.started_at_ms)} elapsed
                {job.job_id ? ` — job ${job.job_id.slice(0, 8)}` : ''}
              </div>
            </div>
            <span className="badge live">Live</span>
          </div>
        </div>
      )}

      <div className="grid2">
        <div className="panel">
          <h2>{gpu.gpu_name || 'GPU'}</h2>
          {gpu.vram_total_mb > 0 ? (
            <>
              <div className="row spread small muted">
                <span>{gpu.unified_memory ? 'Unified memory' : 'VRAM'} used: {fmtMb(usedMb)}</span>
                <span>free: {fmtMb(gpu.vram_free_mb)} / {fmtMb(gpu.vram_total_mb)}</span>
              </div>
              <div className="meter" style={{ margin: '6px 0 12px' }}>
                <div style={{ width: `${Math.min(100, (usedMb / gpu.vram_total_mb) * 100)}%` }} />
              </div>
            </>
          ) : (
            <p className="muted">No GPU telemetry available.</p>
          )}
          <div className="row spread">
            <span className="small" title="A soft budget: the Companion sizes Whisper and reserves the rest of the card away from Ollama (OLLAMA_GPU_OVERHEAD). Individual allocations can still briefly exceed it.">
              GPU memory ClipAI may use: <strong>{sliderValue.toFixed(1)} GB</strong> ⓘ
            </span>
          </div>
          <input
            type="range" min={2} max={sliderMax} step={0.5} value={sliderValue}
            onChange={(e) => setBudget(parseFloat(e.target.value))}
            onMouseUp={() => budget !== null && commitBudget(budget)}
            onTouchEnd={() => budget !== null && commitBudget(budget)}
          />
          <div className="small muted">
            Whisper tier at this budget: <span className="mono">{status.whisper_tier.model}</span>
            {' '}({status.whisper_tier.compute}) — changing the slider restarts Ollama with the
            new reservation.
          </div>
        </div>

        <div className="panel">
          <h2>Connection</h2>
          <div className="row small" style={{ marginBottom: 6 }}>
            <span className={`dot ${status.ollama.running ? 'ok' : 'bad'}`} />
            Ollama {status.ollama.running
              ? `running (${status.ollama.models.length} models${status.ollama.managed ? ', managed' : ''})`
              : 'not running'}
          </div>
          <div className="row small" style={{ marginBottom: 6 }}>
            {/* Green once installed/ready (idle is healthy — it starts on
                demand); gray only when it isn't installed at all. */}
            <span className="dot" style={{
              background: (status.sidecar_running || status.sidecar_available)
                ? 'var(--success)' : 'var(--muted)',
            }} />
            Whisper sidecar {status.sidecar_running ? 'running'
              : status.sidecar_available ? 'ready (starts on demand)'
              : 'not installed (optional) — transcription runs on the ClipAI server instead; GPU sharing for Ollama is unaffected'}
          </div>
          {!status.sidecar_available && (
            <div style={{ margin: '0 0 8px 17px' }}>
              <div className="row" style={{ marginBottom: whisperDl ? 6 : 0 }}>
                <button className="secondary" onClick={doDownloadWhisper}
                  disabled={!!whisperDl?.active}
                  title="Download whisper.cpp so this GPU can also do transcription">
                  {whisperDl?.active ? 'Downloading…' : 'Download Whisper'}
                </button>
                <button className="secondary" onClick={doRefreshWhisper} disabled={!!whisperDl?.active}
                  title="Re-check whether Whisper is installed">
                  ↻ Refresh
                </button>
              </div>
              {whisperDl && (
                <div>
                  <div className="small muted" style={{ marginBottom: 3 }}>{whisperDl.message}</div>
                  <div className={`meter${whisperDl.percent < 0 ? ' indeterminate' : ''}`}>
                    <div style={whisperDl.percent < 0 ? undefined
                      : { width: `${Math.max(2, Math.min(100, whisperDl.percent))}%` }} />
                  </div>
                </div>
              )}
            </div>
          )}
          <div className="small muted" style={{ margin: '8px 0 4px' }}>
            Share this address with ClipAI:
          </div>
          <div className="mono small" style={{ marginBottom: 8 }}>
            http://{status.lan_ip || '<this-machine>'}:{status.config.port}
          </div>
          <TokenBox token={status.config.token}
            onRegenerate={() => { regenerateToken().then(refresh); }} />
          <div className="small muted" style={{ marginTop: 8 }}>
            {status.config.paired_clipai_url
              ? <>Paired with <span className="mono">{status.config.paired_clipai_url}</span></>
              : 'Not paired yet.'}
            {' '}
            <a href="#" onClick={(e) => { e.preventDefault(); setPairOpen((o) => !o); }}>
              {pairOpen ? 'Close' : status.config.paired_clipai_url ? 'Re-pair' : 'Pair now'}
            </a>
          </div>
          {pairOpen && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
              <input type="text" placeholder="http://tower.local:8000" value={clipaiUrl}
                onChange={(e) => setClipaiUrl(e.target.value)} />
              <input type="password" placeholder="ClipAI API key" value={apiKey}
                onChange={(e) => setApiKey(e.target.value)} />
              <div className="row">
                <button onClick={doPair} disabled={!clipaiUrl.trim() || !apiKey.trim()}>Pair</button>
                {pairMsg && <span className="small muted">{pairMsg}</span>}
              </div>
            </div>
          )}
          {autostart !== null && (
            <label className="checkbox" style={{ marginTop: 10 }}>
              <input type="checkbox" checked={autostart}
                onChange={async (e) => {
                  const on = e.target.checked;
                  setAutostart(on);
                  try { on ? await enableAutostart() : await disableAutostart(); }
                  catch { setAutostart(!on); }
                }} />
              Launch at login
            </label>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="row spread" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>Local AI models</h2>
          <div className="row">
            <span className="muted small">
              {models.length} installed{models.length ? ` · ${fmtBytes(models.reduce((s, m) => s + (m.size || 0), 0))}` : ''}
            </span>
            <button className="secondary" onClick={() => { loadModels(); refresh(); }}
              title="Reload the installed-model list">↻ Refresh</button>
            <button className="secondary" onClick={doReSync} disabled={syncing || !status.ollama.running}
              title="Download any recommended models not yet installed">
              {syncing ? 'Syncing…' : 'Re-sync models'}
            </button>
          </div>
        </div>
        {syncMsg && (
          <div className="small muted" style={{ marginBottom: 6 }}>{syncMsg}</div>
        )}
        {models.length === 0 ? (
          <p className="muted small">
            No models installed yet. Pick models in ClipAI and hit Save (or use the setup
            wizard) — they download here and appear in this list.
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {models.map((m) => (
              <div className="row spread" key={m.name}
                style={{ background: 'var(--elevated)', borderRadius: 6, padding: '6px 10px' }}>
                <div style={{ minWidth: 0 }}>
                  <span className="mono">{m.name}</span>
                  <span className="muted small" style={{ marginLeft: 8 }}>
                    {fmtBytes(m.size)}{m.parameter_size ? ` · ${m.parameter_size}` : ''}
                  </span>
                </div>
                {confirmDel === m.name ? (
                  <div className="row">
                    <button className="secondary" disabled={deleting === m.name}
                      onClick={() => removeModel(m.name)}
                      style={{ color: 'var(--danger)', borderColor: 'var(--danger)' }}>
                      {deleting === m.name ? 'Deleting…' : 'Confirm delete'}
                    </button>
                    <button className="secondary" onClick={() => setConfirmDel('')}>Cancel</button>
                  </div>
                ) : (
                  <button className="secondary" onClick={() => setConfirmDel(m.name)}
                    title="Delete this model from disk on this machine">
                    Delete
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
        <p className="muted small" style={{ marginTop: 8 }}>
          Deleting frees disk space; the model must be re-downloaded to use it again.
        </p>
      </div>

      <div className="panel">
        <h2>Recent activity</h2>
        {status.activity.length === 0 ? (
          <p className="muted small">
            Nothing yet — activity appears here when ClipAI sends work to this GPU.
          </p>
        ) : (
          <div className="feed">
            {status.activity.filter((a) => a.kind !== 'health').slice(0, 20).map((a) => (
              <div className="feed-item" key={a.id}>
                <span className={`dot ${a.finished_at_ms === null ? 'ok' : (a.status ?? 500) < 400 ? 'ok' : 'bad'}`} />
                <span style={{ minWidth: 66 }} className="badge">{a.kind}</span>
                <span style={{ flex: 1 }}>
                  {a.job_title || a.path}
                  {a.stage && <span className="muted"> — {a.stage}</span>}
                </span>
                <span className="muted mono">
                  {a.finished_at_ms === null
                    ? `${elapsed(a.started_at_ms)}…`
                    : elapsed(a.started_at_ms, a.finished_at_ms)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const { status, refresh } = useStatus();
  const [forceDashboard, setForceDashboard] = useState(false);

  if (!status) {
    return <div className="app"><p className="muted">Starting…</p></div>;
  }
  if (!status.config.setup_complete && !forceDashboard) {
    return (
      <Wizard status={status} refresh={refresh}
        onDone={() => { setConfig({ setup_complete: true }).then(refresh); setForceDashboard(true); }} />
    );
  }
  return <Dashboard status={status} refresh={refresh} />;
}
