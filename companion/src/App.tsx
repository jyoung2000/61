import { useCallback, useEffect, useRef, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { openUrl } from '@tauri-apps/plugin-opener';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { enable as enableAutostart, disable as disableAutostart, isEnabled as autostartEnabled } from '@tauri-apps/plugin-autostart';
import {
  CompanionStatus, getStatus, setConfig, regenerateToken,
  installOllama, startOllama, pullModel, pairClipai,
  listModels, deleteModel, InstalledModel,
  downloadWhisper, refreshSidecar, exportLogs, testClipai, ClipaiTest, freeVram, endActiveJob,
  checkAppUpdate, installAppUpdate, AppUpdateCheck,
} from './api';

// Common Ollama models offered as search suggestions on the Companion.
const OLLAMA_CATALOG = [
  'qwen2.5vl:7b', 'qwen2.5vl:3b', 'llava:13b', 'llava:7b', 'moondream:1.8b',
  'qwen2.5:14b', 'qwen2.5:7b-instruct', 'qwen2.5:3b-instruct',
  'qwen3:4b-instruct-2507-q4_K_M', 'qwen3:8b', 'llama3.1:8b', 'gemma2:9b',
  'mistral:7b', 'phi3:mini', 'nomic-embed-text',
];

// AdGuard Home "allow" rules for the hosts model downloads need. If a DNS
// blocker filters these, pulls stall — often near the end when Ollama fetches
// blobs from its Cloudflare R2 CDN (*.r2.cloudflarestorage.com).
const ADGUARD_RULES = [
  '@@||ollama.com^',
  '@@||registry.ollama.ai^',
  '@@||r2.cloudflarestorage.com^',
  '@@||huggingface.co^',
  '@@||hf.co^',
  '@@||github.com^',
  '@@||githubusercontent.com^',
];

const fmtMb = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`);
const fmtBytes = (b: number) => {
  if (!b) return '—';
  const gb = b / (1024 ** 3);
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(b / (1024 * 1024))} MB`;
};
const elapsed = (startMs: number, endMs?: number | null) => {
  // Guard against a missing/epoch start (would render millions of hours).
  if (!startMs || startMs < 1_000_000_000_000) return '—';
  const s = Math.max(0, Math.floor(((endMs ?? Date.now()) - startMs) / 1000));
  const h = Math.floor(s / 3600);
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
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

// Percent from an installer transcript line, when one is present. winget /
// brew print progress two ways — an explicit "45%" or a byte fraction
// ("12.5 MB / 205 MB") — and the first-run Ollama install used to render as
// an indeterminate shimmer for its whole multi-minute download because
// neither form was parsed. Returns null for lines with no usable number.
const _UNIT: Record<string, number> = { KB: 1e3, MB: 1e6, GB: 1e9 };
function parseInstallPercent(msg: string): number | null {
  // Installers overwrite their progress with carriage returns, so one
  // buffered line can hold several stale updates — the LAST number wins.
  const pcts = [...msg.matchAll(/(\d{1,3}(?:\.\d+)?)\s*%/g)];
  if (pcts.length) {
    const v = parseFloat(pcts[pcts.length - 1][1]);
    if (v >= 0 && v <= 100) return v;
  }
  const fracs = [...msg.matchAll(/([\d.]+)\s*(KB|MB|GB)\s*\/\s*([\d.]+)\s*(KB|MB|GB)/gi)];
  if (fracs.length) {
    const f = fracs[fracs.length - 1];
    const done = parseFloat(f[1]) * _UNIT[f[2].toUpperCase()];
    const total = parseFloat(f[3]) * _UNIT[f[4].toUpperCase()];
    if (total > 0 && done >= 0 && done <= total) return (done / total) * 100;
  }
  return null;
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
  const [installPct, setInstallPct] = useState<number | null>(null);
  const [pulling, setPulling] = useState<Record<string, 'pulling' | 'done' | 'error'>>({});
  const [pullProg, setPullProg] = useState<Record<string, { percent: number; status: string }>>({});
  const [clipaiUrl, setClipaiUrl] = useState('');
  const [pairBusy, setPairBusy] = useState(false);
  const [pairError, setPairError] = useState('');
  const [pairDone, setPairDone] = useState(false);

  const steps = ['Welcome', 'Ollama', 'Models', 'Access token', 'Connect', 'Done'];
  const ollamaReady = status.ollama.running;

  // Live progress from the backend (install transcript + model-pull bytes).
  useEffect(() => {
    const unInstall = listen<{ stage: string; message: string }>(
      'install-progress', (e) => {
        const msg = e.payload.message || '';
        setInstallMsg(msg);
        // Progress lines and status lines interleave; a line without a
        // number keeps the last known percentage instead of resetting the
        // bar to indeterminate.
        setInstallPct((prev) => parseInstallPercent(msg) ?? prev);
      });
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
    setInstallPct(null);
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
      await pairClipai(clipaiUrl);
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
              {installing && (installPct != null ? (
                <ProgressBar
                  percent={installPct}
                  label={`${installMsg || 'Installing Ollama…'} — ${Math.round(installPct)}%`}
                />
              ) : (
                <ProgressBar indeterminate label={installMsg || 'Installing Ollama…'} />
              ))}
              <div className="row">
                <button onClick={doInstall} disabled={installing}>
                  {installing ? 'Installing…' : installError ? 'Retry install' : 'Install Ollama'}
                </button>
                <button className="secondary"
                  onClick={() => openUrl('https://ollama.com/download')
                    .catch((e) => setInstallError(String(e)))}>
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
            Paste your ClipAI server address — no key needed. ClipAI will add
            this machine as its primary AI host and use it for Whisper.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <input type="text" placeholder="http://tower.local:8000"
              value={clipaiUrl} onChange={(e) => setClipaiUrl(e.target.value)} />
            <div className="row">
              <button onClick={doPair} disabled={pairBusy || !clipaiUrl.trim()}>
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
            The Companion lives in your tray and needs no terminal. Closing this
            window keeps it sharing in the background; use the tray menu to open
            it again, pause sharing, or quit. It also starts automatically at
            login — straight to the tray, no window — so the GPU is available
            after a reboot without anyone touching this machine. Allow inbound
            TCP&nbsp;{status.config.port} on Private networks if your firewall asks.
          </p>
          <button onClick={onDone}>Open dashboard</button>
        </div>
      )}
    </div>
  );
}

// ── Dashboard ────────────────────────────────────────────────────────

// Stable, distinct color per model name so each AI model reads as the same
// hue everywhere (VRAM bar segment + list swatch) — lets the user see at a
// glance how much VRAM each model is using. Deterministic hash → hue.
function modelColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  return `hsl(${h}, 68%, 55%)`;
}

function Dashboard({ status, refresh, theme, toggleTheme }: {
  status: CompanionStatus; refresh: () => void;
  theme: 'dark' | 'light'; toggleTheme: () => void;
}) {
  const [budget, setBudget] = useState<number | null>(null);
  const [bufferGb, setBufferGb] = useState<number | null>(null);
  const [newSharedPath, setNewSharedPath] = useState('');
  const [autostart, setAutostart] = useState<boolean | null>(null);
  const [pairOpen, setPairOpen] = useState(false);
  const [clipaiUrl, setClipaiUrl] = useState(status.config.paired_clipai_url);
  const [pairMsg, setPairMsg] = useState('');
  const [models, setModels] = useState<InstalledModel[]>([]);
  const [confirmDel, setConfirmDel] = useState('');
  const [deleting, setDeleting] = useState('');
  const [whisperDl, setWhisperDl] = useState<{ active: boolean; percent: number; message: string } | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncMsg, setSyncMsg] = useState('');
  const [modelQuery, setModelQuery] = useState('');
  const [pullingTag, setPullingTag] = useState('');
  const [addrCopied, setAddrCopied] = useState(false);
  const [adguardOpen, setAdguardOpen] = useState(false);
  const [adguardCopied, setAdguardCopied] = useState(false);

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

  const [clipaiTest, setClipaiTest] = useState<ClipaiTest | null>(null);
  const [testingClipai, setTestingClipai] = useState(false);
  const doTestClipai = async () => {
    setTestingClipai(true);
    setClipaiTest(null);
    try {
      setClipaiTest(await testClipai());
    } catch {
      setClipaiTest(null);
    } finally {
      setTestingClipai(false);
    }
  };

  const [freeing, setFreeing] = useState(false);
  const doFreeVram = async () => {
    setFreeing(true);
    setSyncMsg('Freeing GPU memory…');
    try {
      const r = await freeVram();
      const parts: string[] = [];
      if (r.unloaded > 0) parts.push(`${r.unloaded} model${r.unloaded === 1 ? '' : 's'}`);
      if (r.whisper_stopped) parts.push('whisper server');
      setSyncMsg(parts.length
        ? `Freed ${parts.join(' + ')} from VRAM ✓`
        : 'Nothing was resident — VRAM already free');
      setTimeout(() => { setSyncMsg(''); refresh(); }, 1500);
    } catch (e) {
      setSyncMsg(`Could not free VRAM: ${e}`);
      setTimeout(() => setSyncMsg(''), 6000);
    } finally {
      setFreeing(false);
    }
  };

  const [endingJob, setEndingJob] = useState(false);
  const doEndJob = async () => {
    setEndingJob(true);
    try {
      await endActiveJob();
      refresh();
    } catch (e) {
      setSyncMsg(`Could not end job: ${e}`);
      setTimeout(() => setSyncMsg(''), 5000);
    } finally {
      setEndingJob(false);
    }
  };

  // ── App self-update (installer served by the paired ClipAI container) ──
  const [updateInfo, setUpdateInfo] = useState<AppUpdateCheck | null>(null);
  const [updateMsg, setUpdateMsg] = useState('');
  const [checkingUpdate, setCheckingUpdate] = useState(false);
  const [installingUpdate, setInstallingUpdate] = useState(false);
  const doCheckUpdate = async (quiet = false) => {
    setCheckingUpdate(true);
    if (!quiet) setUpdateMsg('');
    try {
      const info = await checkAppUpdate();
      setUpdateInfo(info);
      if (!quiet && !info.update_available) {
        setUpdateMsg(info.installer_available
          ? `Up to date — v${info.current}${info.current_build ? ` (build ${info.current_build})` : ''} is the newest installer your ClipAI serves.`
          : 'Your ClipAI server has no Companion installer yet — on the server, run "Check for Companion updates" in Settings → GPU Companion (or rebuild the container with COMPANION_BUILD_FROM_SOURCE=1).');
      }
    } catch (e) {
      if (!quiet) setUpdateMsg(String(e));
    } finally {
      setCheckingUpdate(false);
    }
  };
  // One quiet check shortly after launch so the button shows a badge when an
  // update is already waiting on the paired server. Failures stay silent.
  useEffect(() => {
    const t = setTimeout(() => { doCheckUpdate(true); }, 4000);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const doInstallUpdate = async () => {
    setInstallingUpdate(true);
    setUpdateMsg('Downloading the installer from your ClipAI server…');
    try {
      const path = await installAppUpdate();
      setUpdateMsg(`Installer started (${path}) — this app will close so it can update. Reopen it when the installer finishes.`);
    } catch (e) {
      setUpdateMsg(String(e));
      setInstallingUpdate(false);
    }
  };

  const [exporting, setExporting] = useState(false);
  const doExportLogs = async () => {
    setExporting(true);
    setSyncMsg('Exporting logs…');
    try {
      const path = await exportLogs();
      setSyncMsg(`Logs saved to ${path}`);
      setTimeout(() => setSyncMsg(''), 12000);
    } catch (e) {
      setSyncMsg(`Export failed: ${e}`);
      setTimeout(() => setSyncMsg(''), 8000);
    } finally {
      setExporting(false);
    }
  };

  // Search + download any Ollama model by tag (with live pull-progress).
  const doPullTag = async (tag: string) => {
    const t = (tag || '').trim();
    if (!t) return;
    setPullingTag(t);
    setSyncMsg(`Downloading ${t}…`);
    try {
      await pullModel(t);
      loadModels();
      refresh();
      setModelQuery('');
      setSyncMsg(`${t} downloaded ✓`);
    } catch (e) {
      // Surface the REAL error (e.g. "requires a newer version of Ollama",
      // a 404, or a network failure) instead of a misleading canned message.
      setSyncMsg(`Could not download ${t}: ${e}`);
    } finally {
      setPullingTag('');
      setTimeout(() => setSyncMsg(''), 6000);
    }
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
    let lastErr = '';
    try {
      for (const model of missing) {
        setSyncMsg(`Downloading ${model}…`);
        try { await pullModel(model); } catch (e) { failed.push(model); lastErr = String(e); }
      }
      loadModels();
      refresh();
    } finally {
      setSyncing(false);
      setSyncMsg(failed.length
        ? `Could not download ${failed.join(', ')}: ${lastErr}`
        : 'Recommended models downloaded ✓');
      setTimeout(() => setSyncMsg(''), 10000);
    }
  };

  const gpu = status.gpu;
  const totalGb = gpu.vram_total_mb / 1024;
  const usedMb = Math.max(0, gpu.vram_total_mb - gpu.vram_free_mb);
  // Split the used VRAM: what ClipAI holds (Ollama models + Whisper) vs what
  // other apps (games/editors) are using, so each gets its own bar color.
  const clipaiMb = Math.min(usedMb, Math.max(0, status.clipai_vram_mb || 0));
  const otherMb = Math.max(0, usedMb - clipaiMb);
  const pct = (mb: number) => (gpu.vram_total_mb > 0 ? Math.min(100, (mb / gpu.vram_total_mb) * 100) : 0);
  const sliderValue = budget ?? (status.config.vram_budget_gb > 0
    ? status.config.vram_budget_gb : status.effective_budget_gb);
  const sliderMax = Math.max(2, Math.floor(totalGb) - 1);

  const commitBudget = async (v: number) => {
    setBudget(null);
    await setConfig({ vram_budget_gb: v });
    refresh();
  };

  const commitBuffer = async (v: number) => {
    setBufferGb(null);
    await setConfig({ vram_buffer_gb: v });
    refresh();
  };

  const doPair = async () => {
    setPairMsg('');
    try {
      await pairClipai(clipaiUrl);
      setPairMsg('Paired ✓');
      refresh();
    } catch (e) {
      setPairMsg(String(e));
    }
  };

  const job = status.current_job;
  const sharedPaths = status.config.shared_paths || [];
  const addSharedPath = async (pathArg?: string) => {
    const p = (pathArg ?? newSharedPath).trim();
    if (!p || sharedPaths.includes(p)) { setNewSharedPath(''); return; }
    await setConfig({ shared_paths: [...sharedPaths, p] });
    setNewSharedPath('');
    refresh();
  };
  const browseSharedPath = async () => {
    // Native OS folder picker (Tauri dialog plugin). null = user cancelled.
    const sel = await openDialog({
      directory: true, multiple: false,
      title: 'Share a folder with ClipAI',
      defaultPath: newSharedPath || undefined,
    });
    if (typeof sel === 'string') await addSharedPath(sel);
  };
  const removeSharedPath = async (p: string) => {
    await setConfig({ shared_paths: sharedPaths.filter((x) => x !== p) });
    refresh();
  };
  // Models actively resident in VRAM right now (largest first).
  const residentModels = [...(status.resident_models || [])].sort((a, b) => b.vram_mb - a.vram_mb);

  // ── Single "Performance" control ─────────────────────────────────────────
  // One knob for how much of this GPU ClipAI may use. It drives BOTH the Ollama
  // speed profile (pipeline parallelism) AND the Whisper transcription quality
  // (beam search + model), because both draw on the same VRAM/compute budget.
  // The two underlying config fields still exist (and the container reads
  // whisper quality back for sync) — the UI just sets them together.
  // Hover hints lead with CAPTION QUALITY — what the transcript will look
  // like is the thing users pick a level for; speed is the secondary trade.
  const PERF_LEVELS = [
    { key: 'auto', speed: 'auto', whisper: 'auto',
      hint: 'Caption accuracy ★★★☆ (adaptive) — picks the most accurate ' +
        'Whisper decode your allocated VRAM affords (usually large-v3-turbo ' +
        'with beam search: near-broadcast accuracy, occasional slips on names ' +
        'and whispered lines). Speed also auto-tunes. Recommended default.' },
    { key: 'eco', speed: 'eco', whisper: 'fast',
      hint: 'Caption accuracy ★★☆☆ (fastest, roughest) — small greedy decode: ' +
        'expect more misheard words, wrong names, and missed quiet lines. ' +
        'One AI job at a time; leaves the card free for games/other apps. ' +
        'Pick this when the PC is in use, not when subtitles matter.' },
    { key: 'balanced', speed: 'balanced', whisper: 'balanced',
      hint: 'Caption accuracy ★★★☆ (solid) — large-v3-turbo with beam search: ' +
        'accurate on clear dialogue, occasional slips on names and quiet or ' +
        'overlapped speech. Moderate parallelism for a steady pipeline.' },
    { key: 'turbo', speed: 'turbo', whisper: 'max',
      hint: 'Caption accuracy ★★★★ (best possible) — FULL large-v3 with beam ' +
        'search, the "Netflix-grade" tier: fewest mishearings and the best ' +
        'pickup of quiet/whispered lines. Needs ~8 GB allocated to engage ' +
        '(below that it safely falls back to the Balanced decode — the ' +
        '"Whisper tier" line below shows which engaged). Maximum pipeline ' +
        'parallelism; slowest to share the card with other apps.' },
  ] as const;
  // Which level is active. We always set both fields together, so the speed
  // profile identifies the level (they align 1:1). Falls back to 'auto' if a
  // pre-existing config has an unusual pairing.
  const perfLevel = PERF_LEVELS.find((l) => l.speed === status.config.speed_profile)?.key ?? 'auto';
  const wqe = status.whisper_quality_effective;
  const capDesc = wqe.beam_search ? `${wqe.model} beam ${wqe.beam_size}` : `${wqe.model} greedy`;
  const perfSubtitle =
    perfLevel === 'auto'
      ? `Auto — ${status.speed.num_parallel}× parallel and ${capDesc} captions for your ${gpu.gpu_name || 'GPU'} at ${status.effective_budget_gb.toFixed(1)} GB.`
      : perfLevel === 'eco'
      ? 'Eco — minimal GPU use: slowest pipeline and fast/greedy captions, but leaves the card free for other apps.'
      : perfLevel === 'balanced'
      ? `Balanced — ${status.speed.num_parallel}× parallel with beam-search captions (${wqe.model}). Solid speed and accuracy.`
      : `Turbo — pushing this GPU as hard as its VRAM allows: ${status.speed.num_parallel}× parallel and ${capDesc} captions for the fastest pipeline and best accuracy.`;

  return (
    <div className="app">
      <header className="row spread appbar">
        <h1>ClipAI GPU Companion</h1>
        <div className="row appbar-meta">
          {/* GPU currently in use */}
          <span className="chip" title={
            gpu.gpu_name
              ? `GPU: ${gpu.gpu_name}${gpu.vram_total_mb > 0
                  ? ` — ${fmtMb(usedMb)} of ${fmtMb(gpu.vram_total_mb)} in use` : ''}`
              : 'No GPU detected'}>
            <span className="dot" style={{
              background: gpu.available ? 'var(--success)' : 'var(--muted)',
            }} />
            <span className="chip-label">{gpu.gpu_name || 'No GPU'}</span>
            {gpu.vram_total_mb > 0 && (
              <span className="muted">{fmtMb(usedMb)}/{fmtMb(gpu.vram_total_mb)}</span>
            )}
          </span>

          {/* Installed AI models */}
          <span className="chip" title={
            status.ollama.models.length
              ? `Installed models:\n${status.ollama.models.join('\n')}`
              : 'No models installed yet'}>
            🧠 {status.ollama.models.length} model{status.ollama.models.length === 1 ? '' : 's'}
          </span>

          {/* Models ACTIVELY RESIDENT in VRAM right now (what's running). */}
          <span className={`chip ${residentModels.length ? 'live' : ''}`} title={
            residentModels.length
              ? `Running now (VRAM-resident):\n${residentModels.map((m) => `${m.name} — ${fmtMb(m.vram_mb)}`).join('\n')}`
              : 'No models resident in VRAM — idle'}>
            <span className="dot" style={{
              background: residentModels.length ? 'var(--success)' : 'var(--muted)',
            }} />
            <span className="chip-label">
              {residentModels.length ? `${residentModels.length} running · ${fmtMb(clipaiMb)}` : 'VRAM idle'}
            </span>
          </span>

          {/* Current job the companion is being used for */}
          <span className={`chip ${job || status.busy ? 'live' : ''}`} title={
            job
              ? `${job.kind === 'whisper' ? 'Transcribing' : 'AI inference'}`
                + `${job.job_title ? ` — ${job.job_title}` : ''}`
                + `${job.stage ? ` (stage: ${job.stage})` : ''}`
              : status.busy ? 'Working' : 'No active job'}>
            <span className="dot" style={{
              background: job || status.busy ? 'var(--success)' : 'var(--muted)',
            }} />
            <span className="chip-label">{job
              ? `${job.kind === 'whisper' ? 'Transcribing' : 'AI inference'}${job.job_title ? ` — ${job.job_title}` : ''}`
              : status.busy ? 'Working…' : 'No active job'}</span>
          </span>

          {/* Light / dark mode switcher */}
          <button className="secondary icon" title="Toggle light / dark mode"
            aria-label="Toggle light / dark mode" onClick={toggleTheme}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>

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
      </header>

      {status.config.paused && (
        <div className="panel" style={{ borderColor: 'var(--danger)', background: 'rgba(244,104,92,0.12)' }}>
          <div className="row spread">
            <strong style={{ color: 'var(--danger)', fontSize: 14 }}>
              ⏸ Sharing is PAUSED — ClipAI is getting 503 errors and can't use this GPU
            </strong>
            <button onClick={() => setConfig({ paused: false }).then(refresh)}>
              Resume sharing
            </button>
          </div>
          <div className="muted small" style={{ marginTop: 4 }}>
            While paused, every request from ClipAI (models, transcription, vision) is
            refused with 503. Click Resume to let ClipAI use this GPU again.
          </div>
        </div>
      )}

      {status.proxy_bound === false && (
        <div className="panel" style={{ borderColor: 'var(--danger)', background: 'rgba(244,104,92,0.10)' }}>
          <strong style={{ color: 'var(--danger)' }}>
            ⚠ Can't open port {status.config.port} — ClipAI can't reach this companion
          </strong>
          <div className="muted small" style={{ marginTop: 4 }}>
            {status.proxy_last_error || 'The port is in use.'} Another program (or a
            leftover Companion process) may be holding it. It will keep retrying — close the
            other program, or restart this app. Only one Companion should run at a time.
          </div>
        </div>
      )}

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
            <div className="row" style={{ gap: 8, flexShrink: 0 }}>
              <span className="badge live">
                {status.job_progress != null ? `${status.job_progress}%` : 'Live'}
              </span>
              <button className="secondary" style={{ padding: '3px 10px' }}
                onClick={doEndJob} disabled={endingJob}
                title="Force-end this job on the Companion: clears the display and unloads its models. Use if ClipAI stopped without telling the Companion.">
                {endingJob ? 'Ending…' : 'Force end'}
              </button>
            </div>
          </div>
          {/* Live progress bar for the pipeline ClipAI is running (from
              X-ClipAI-Progress). Indeterminate until ClipAI reports a %. */}
          <div className={`meter${status.job_progress == null ? ' indeterminate' : ''}`}
            style={{ marginTop: 8 }}>
            <div style={status.job_progress == null ? undefined
              : { width: `${Math.max(2, Math.min(100, status.job_progress))}%` }} />
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
              {/* Per-model VRAM bar: one colored segment per resident AI model
                  (so you can see how much each is using), then any remaining
                  ClipAI VRAM (green, e.g. Whisper), then other apps (blue). */}
              <div className="meter" style={{ margin: '6px 0 4px', display: 'flex', overflow: 'hidden' }}
                title={`ClipAI: ${fmtMb(clipaiMb)} · other apps: ${fmtMb(otherMb)} · free: ${fmtMb(gpu.vram_free_mb)}`}>
                {residentModels.map((m) => (
                  <div key={m.name} title={`${m.name} — ${fmtMb(m.vram_mb)}`}
                    style={{ width: `${pct(m.vram_mb)}%`, height: '100%', background: modelColor(m.name), transition: 'width 0.4s' }} />
                ))}
                {(() => {
                  const residentVram = residentModels.reduce((s, m) => s + m.vram_mb, 0);
                  const extra = Math.max(0, clipaiMb - residentVram);
                  return extra > 0
                    ? <div title={`ClipAI (other) — ${fmtMb(extra)}`}
                        style={{ width: `${pct(extra)}%`, height: '100%', background: 'var(--success)', transition: 'width 0.4s' }} />
                    : null;
                })()}
                <div style={{ width: `${pct(otherMb)}%`, height: '100%', background: '#3b82f6', transition: 'width 0.4s' }} />
              </div>
              <div className="row small muted" style={{ gap: 14, margin: '0 0 12px' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: 'var(--success, #22c55e)' }} />
                  ClipAI {fmtMb(clipaiMb)}
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, background: '#3b82f6' }} />
                  Other apps {fmtMb(otherMb)}
                </span>
              </div>
              {/* Resident-in-VRAM: the itemized expansion of the green ClipAI
                  segment — exactly which AI models are loaded right now. */}
              <div style={{ margin: '0 0 12px' }}>
                <div className="row small" style={{ gap: 6, marginBottom: 5 }}>
                  <span className="dot" style={{ background: residentModels.length ? 'var(--success)' : 'var(--muted)' }} />
                  <strong>Running in VRAM</strong>
                  <span className="muted">· {fmtMb(clipaiMb)}</span>
                </div>
                {residentModels.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {residentModels.map((m) => (
                      <div key={m.name} className="list-row small">
                        <span className="name mono" title={m.name} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 3, background: modelColor(m.name), flexShrink: 0 }} />
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{m.name}</span>
                        </span>
                        <span className="muted mono action">{fmtMb(m.vram_mb)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="list-row small">
                    <span className="name muted">No models loaded — VRAM idle. Models load on ClipAI's next request.</span>
                  </div>
                )}
              </div>
              <div className="row" style={{ marginBottom: 10 }}>
                <button className="secondary" onClick={doFreeVram} disabled={freeing || residentModels.length === 0}
                  title="Unload all resident Ollama models to free VRAM right now (e.g. before gaming)">
                  {freeing ? 'Freeing…' : 'Free GPU memory'}
                </button>
                <span className="muted small">
                  {residentModels.length > 0
                    ? 'Unloads models held in VRAM (they reload on the next request)'
                    : ''}
                </span>
              </div>
            </>
          ) : (
            <p className="muted">No GPU telemetry available.</p>
          )}

          {/* Performance — ONE control for how much of this GPU ClipAI may use.
              It drives both the pipeline speed (Ollama parallelism) and the
              transcription quality (Whisper beam search + model) together, since
              both draw on the same VRAM. Auto tunes to the card + allocated VRAM. */}
          <div style={{ margin: '4px 0 10px' }}>
            <div className="row spread small" style={{ marginBottom: 4 }}>
              <strong title="One control for how hard ClipAI pushes this GPU. Higher = more concurrent AI (faster pipeline) AND higher-accuracy transcription (beam search + bigger Whisper model). Both use this card's VRAM.">
                Performance ⚡🎯
              </strong>
              <span className="muted small mono">
                {status.speed.num_parallel}× · {capDesc}
              </span>
            </div>
            <div className="segmented">
              {PERF_LEVELS.map((lvl) => (
                <button
                  key={lvl.key}
                  className={perfLevel === lvl.key ? 'active' : ''}
                  onClick={() => setConfig({ speed_profile: lvl.speed, whisper_quality: lvl.whisper }).then(refresh)}
                  title={lvl.hint}
                >
                  {lvl.key}
                </button>
              ))}
            </div>
            <div className="small muted" style={{ marginTop: 4 }}>
              {perfSubtitle} Changing this may briefly restart the GPU engines.
            </div>
          </div>

          <label className="row small" style={{ gap: 8, margin: '4px 0 8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={status.config.vram_auto}
              onChange={(e) => setConfig({ vram_auto: e.target.checked }).then(refresh)}
              style={{ width: 'auto' }}
            />
            <span>
              <strong>Auto-allocate VRAM</strong> — dynamically use free memory,
              leaving a buffer so other apps (games, editors) can grow.
            </span>
          </label>

          {/* Shared folders — ClipAI can browse these and pull video/media/font
              files from them remotely (e.g. drop a clip here from your phone,
              then import it in ClipAI while away from the PC). Read-only + jailed
              to exactly these folders on the ClipAI side. */}
          <div style={{ margin: '6px 0 10px' }}>
            <div className="row spread small" style={{ marginBottom: 4 }}>
              <strong title="ClipAI can browse and pull files (video, media, fonts) from these folders over your LAN. Access is read-only and strictly limited to the folders you list here.">
                Shared folders 📂
              </strong>
              <span className="muted small">
                {status.config.share_all ? 'all drives' : `${sharedPaths.length} shared`}
              </span>
            </div>

            {/* Share EVERYTHING — all drives — instead of listing folders. */}
            <label className="row small" style={{ gap: 8, margin: '2px 0 8px', cursor: 'pointer', alignItems: 'flex-start' }}>
              <input
                type="checkbox"
                checked={!!status.config.share_all}
                onChange={(e) => setConfig({ share_all: e.target.checked }).then(refresh)}
                style={{ width: 'auto', marginTop: 3 }}
              />
              <span>
                <strong>Share entire computer</strong> — let ClipAI browse every
                drive (read-only). Convenient, but exposes all your files to the
                paired ClipAI over the LAN. Leave off to share only the folders below.
              </span>
            </label>
            {status.config.share_all ? (
              <div className="small muted" style={{ marginBottom: 2 }}>
                Sharing every drive. ClipAI can browse your whole computer (read-only).
                Turn this off to share only specific folders.
              </div>
            ) : (
              <>
                {sharedPaths.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 6 }}>
                    {sharedPaths.map((p) => (
                      <div key={p} className="list-row small">
                        <span className="name mono" title={p}>{p}</span>
                        <button className="secondary action" style={{ padding: '2px 8px' }}
                          onClick={() => removeSharedPath(p)} title="Stop sharing this folder">✕</button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="small muted" style={{ marginBottom: 6 }}>
                    No shared folders yet. Add one to let ClipAI import videos/media from it remotely.
                  </div>
                )}
                <div className="row" style={{ gap: 6 }}>
                  <input
                    type="text"
                    value={newSharedPath}
                    onChange={(e) => setNewSharedPath(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') addSharedPath(); }}
                    placeholder="e.g. C:\Users\you\Videos\ClipAI"
                    style={{ flex: '1 1 180px', minWidth: 0 }}
                  />
                  <button type="button" className="secondary" style={{ flexShrink: 0 }}
                    onClick={browseSharedPath} title="Pick a folder with the file browser">Browse…</button>
                  <button type="button" style={{ flexShrink: 0 }}
                    onClick={() => addSharedPath()} disabled={!newSharedPath.trim()}>Add</button>
                </div>
                <div className="small muted" style={{ marginTop: 4 }}>
                  Pick a folder with <strong>Browse…</strong> or paste a full path. ClipAI sees only
                  what's inside the folders you list — nothing else.
                </div>
              </>
            )}
          </div>

          {status.config.vram_auto ? (
            <>
              <div className="row spread small" style={{ marginBottom: 4 }}>
                <span title="The Companion measures how much VRAM other apps are using while no model is loaded, then gives Ollama the rest minus this buffer. When other apps need more (e.g. you start gaming), it re-shares and steps back — always keeping this much free on top of what they use.">
                  Free-VRAM buffer to keep for other apps ⓘ
                </span>
                <span className="mono">{status.config.vram_buffer_gb.toFixed(1)} GB</span>
              </div>
              <input
                type="range" min={0} max={Math.max(2, Math.floor(totalGb))} step={0.5}
                value={bufferGb ?? status.config.vram_buffer_gb}
                onChange={(e) => setBufferGb(parseFloat(e.target.value))}
                onMouseUp={() => bufferGb !== null && commitBuffer(bufferGb)}
                onTouchEnd={() => bufferGb !== null && commitBuffer(bufferGb)}
              />
              <div className="small muted">
                Currently giving Ollama <strong>{status.effective_budget_gb.toFixed(1)} GB</strong>
                {' '}(total {totalGb.toFixed(1)} GB − in-use by other apps −{' '}
                {(bufferGb ?? status.config.vram_buffer_gb).toFixed(1)} GB buffer).
                Whisper tier: <span className="mono">{status.whisper_tier.model}</span>
                {' '}({status.whisper_tier.compute}). Re-shares automatically when the card is idle
                and other apps' usage shifts by more than ~1 GB.
              </div>
            </>
          ) : (
            <>
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
            </>
          )}
        </div>

        <div className="panel">
          <h2>Connection</h2>
          <div className="row small" style={{ marginBottom: 6 }}>
            {/* Green + "serving jobs" when real inference/transcription ran in
                the last minute; green + "connected" when ClipAI is only probing;
                amber when paired but silent; gray when never seen. */}
            <span className="dot" style={{
              background: status.clipai_connected ? 'var(--success)'
                : status.config.paired_clipai_url ? 'var(--warn)' : 'var(--muted)',
            }} />
            {status.clipai_serving ? 'ClipAI connected — serving jobs'
              : status.clipai_connected ? 'ClipAI connected — reachable, idle'
              : status.config.paired_clipai_url ? 'ClipAI paired — waiting for requests'
              : 'No ClipAI connected yet'}
          </div>
          <div className="row" style={{ marginBottom: 6 }}>
            <button className="secondary" onClick={doTestClipai} disabled={testingClipai}
              title="Check whether a ClipAI container is properly connected to this companion">
              {testingClipai ? 'Testing…' : 'Test ClipAI connection'}
            </button>
          </div>
          {clipaiTest && (
            <div className="small" style={{
              margin: '0 0 8px', padding: '8px 10px', borderRadius: 6,
              background: 'var(--elevated)',
              borderLeft: `3px solid ${clipaiTest.connected ? 'var(--success)' : 'var(--warn)'}`,
            }}>
              {clipaiTest.connected ? (
                <div style={{ color: 'var(--success)', fontWeight: 600 }}>
                  ✓ ClipAI is connected{clipaiTest.serving ? ' and sending jobs' : ' (reachable, idle)'}
                </div>
              ) : clipaiTest.contacted ? (
                <div style={{ color: 'var(--warn)', fontWeight: 600 }}>
                  ⚠ A ClipAI last contacted this companion {clipaiTest.secs_ago}s ago, but not in the last minute
                </div>
              ) : (
                <div style={{ color: 'var(--warn)', fontWeight: 600 }}>
                  ✗ No ClipAI has contacted this companion yet
                </div>
              )}
              {!clipaiTest.contacted && (
                <div className="muted" style={{ marginTop: 4 }}>
                  In ClipAI → Settings → Ollama Hosts, add <span className="mono">http://{clipaiTest.lan_ip || '<this-PC-IP>'}:{clipaiTest.port}/ollama</span> with
                  the token above, and allow inbound TCP {clipaiTest.port} in the Windows firewall.
                </div>
              )}
              {clipaiTest.probe?.attempted && (
                <div className="muted" style={{ marginTop: 4 }}>
                  Back-probe to {clipaiTest.probe.url}:{' '}
                  {clipaiTest.probe.ok
                    ? 'reachable ✓'
                    : `unreachable — ${clipaiTest.probe.error || `HTTP ${clipaiTest.probe.status}`}`}
                </div>
              )}
            </div>
          )}
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
          {status.sidecar_available && status.whisper_build === 'cpu' && (
            <div className="panel" style={{
              margin: '0 0 8px 17px', padding: 8,
              borderColor: 'var(--accent-amber, #e0a52a)',
              background: 'rgba(224,165,42,0.12)',
            }}>
              <div className="small" style={{ color: 'var(--accent-amber, #e0a52a)', fontWeight: 600 }}>
                ⚠ Whisper is the CPU build — transcription runs on the CPU and is very slow
              </div>
              <div className="small muted" style={{ margin: '3px 0 6px' }}>
                Your GPU can do transcription too. Install the CUDA build to run Whisper
                on the {status.gpu.gpu_name || 'GPU'} — many times faster.
              </div>
              <div className="row" style={{ marginBottom: whisperDl ? 6 : 0 }}>
                <button onClick={doDownloadWhisper} disabled={!!whisperDl?.active}
                  title="Download the CUDA (GPU) whisper.cpp build and swap it in">
                  {whisperDl?.active ? 'Installing…' : '⚡ Install GPU build'}
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
          {status.sidecar_available && status.whisper_build === 'gpu' && (
            <div className="small muted" style={{ margin: '0 0 8px 17px' }}>
              ⚡ Whisper GPU build installed — transcription runs on the {status.gpu.gpu_name || 'GPU'}.
            </div>
          )}
          <div className="small muted" style={{ margin: '8px 0 4px' }}>
            Ollama endpoint for ClipAI (easiest: use “Pair now” below — it fills this in
            automatically):
          </div>
          <div className="row" style={{ gap: 6, marginBottom: 4 }}>
            <span className="mono small" style={{ flex: 1, wordBreak: 'break-all' }}>
              http://{status.lan_ip || '<this-machine>'}:{status.config.port}/ollama
            </span>
            <button className="secondary" style={{ fontSize: 11, padding: '4px 10px' }}
              onClick={() => {
                navigator.clipboard.writeText(`http://${status.lan_ip || ''}:${status.config.port}/ollama`);
                setAddrCopied(true); setTimeout(() => setAddrCopied(false), 1500);
              }}
              title="Copy the Ollama host URL to paste into ClipAI → Settings → Ollama Hosts">
              {addrCopied ? 'Copied ✓' : 'Copy'}
            </button>
          </div>
          <div className="small muted" style={{ marginBottom: 8 }}>
            Adding manually? Paste that URL (keep the <span className="mono">/ollama</span>)
            and the token below into ClipAI → Settings → Ollama Hosts, and allow inbound
            TCP&nbsp;{status.config.port} in the Windows firewall.
          </div>
          <div className="small muted" style={{ marginBottom: 8 }}>
            {status.config.paired_clipai_url
              ? <>ℹ️ The IP above is a DHCP lease and can change after a reboot or
                  router restart — that's fine: this Companion is paired, so it
                  re-announces its new address to ClipAI automatically within a
                  minute.</>
              : <>⚠️ The IP above is a DHCP lease — it can change after a reboot or
                  router restart, which breaks a manually pasted URL. Use
                  “Pair&nbsp;now” below and the Companion re-announces address
                  changes to ClipAI automatically; to pin the address itself,
                  give this PC a DHCP reservation (fixed IP) in your router.</>}
          </div>
          <TokenBox token={status.config.token}
            onRegenerate={() => { regenerateToken().then(refresh); }} />
          <div className="small muted" style={{ marginTop: 8 }}>
            {status.config.paired_clipai_url
              ? <>Paired with <span className="mono">{status.config.paired_clipai_url}</span></>
              : status.clipai_connected
                // Manual add in ClipAI (URL + token) never sets paired_clipai_url,
                // yet an authenticated ClipAI IS talking to us — don't claim
                // "not paired" while it's actively connected.
                ? <>In use by a ClipAI server (added manually — endpoint + token above).</>
                : 'Not paired yet.'}
            {' '}
            <a href="#" onClick={(e) => { e.preventDefault(); setPairOpen((o) => !o); }}>
              {pairOpen ? 'Close'
                : status.config.paired_clipai_url ? 'Re-pair'
                : status.clipai_connected ? 'Pair anyway' : 'Pair now'}
            </a>
          </div>
          {pairOpen && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
              <input type="text" placeholder="http://tower.local:8000" value={clipaiUrl}
                onChange={(e) => setClipaiUrl(e.target.value)} />
              <div className="row">
                <button onClick={doPair} disabled={!clipaiUrl.trim()}>Pair</button>
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
        {/* Search + download any Ollama model by tag. */}
        <div style={{ marginBottom: 10 }}>
          <div className="row" style={{ gap: 6 }}>
            <input type="text" style={{ flex: 1 }}
              placeholder="Search or type a model tag (e.g. llama3.1:8b)…"
              value={modelQuery} onChange={(e) => setModelQuery(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') doPullTag(modelQuery); }} />
            <button disabled={!modelQuery.trim() || !!pullingTag || !status.ollama.running}
              onClick={() => doPullTag(modelQuery)}
              title="Download this model from ollama.com onto this GPU">
              {pullingTag ? 'Downloading…' : '⤓ Download'}
            </button>
          </div>
          {modelQuery.trim() && (() => {
            const q = modelQuery.trim().toLowerCase();
            const sugg = OLLAMA_CATALOG.filter((t) =>
              t.toLowerCase().includes(q)
              && !status.ollama.models.some((im) => im.startsWith(t.split(':')[0])));
            if (!sugg.length) return null;
            return (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
                {sugg.slice(0, 8).map((t) => (
                  <button key={t} className="secondary" style={{ fontSize: 11 }}
                    disabled={!!pullingTag || !status.ollama.running}
                    onClick={() => doPullTag(t)}>⤓ {t}</button>
                ))}
              </div>
            );
          })()}
          {!status.ollama.running && (
            <div className="small muted" style={{ marginTop: 4 }}>Start Ollama to download models.</div>
          )}
        </div>
        {syncMsg && (
          <div className="small muted" style={{ marginBottom: 6 }}>{syncMsg}</div>
        )}
        {/* Downloads pushed from ClipAI through the proxy — shown here so the
            Companion reflects models ClipAI is syncing to this GPU. */}
        {(status.incoming_pulls || []).length > 0 && (
          <div style={{ marginBottom: 10 }}>
            <div className="small" style={{ marginBottom: 4, color: 'var(--accent)' }}>
              Downloading from ClipAI…
            </div>
            {status.incoming_pulls.map((p) => (
              <div key={p.model} style={{ marginBottom: 6 }}>
                <div className="row spread small" style={{ fontFamily: 'var(--font-mono)' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>{p.model}</span>
                  <span className="muted">{p.percent >= 0 ? `${Math.round(p.percent)}%` : '…'}</span>
                </div>
                <div className="meter">
                  <div style={{ width: `${Math.max(2, Math.min(100, p.percent))}%` }} />
                </div>
              </div>
            ))}
          </div>
        )}
        {models.length === 0 ? (
          <p className="muted small">
            No models installed yet. Pick models in ClipAI and hit Save (or use the setup
            wizard) — they download here and appear in this list.
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {models
              .filter((m) => !modelQuery.trim() || m.name.toLowerCase().includes(modelQuery.trim().toLowerCase()))
              .map((m) => (
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
        <div className="row spread">
          <h2 style={{ margin: 0 }}>Network / AdGuard allowlist</h2>
          <a href="#" onClick={(e) => { e.preventDefault(); setAdguardOpen((o) => !o); }}>
            {adguardOpen ? 'Hide' : 'Show'}
          </a>
        </div>
        {adguardOpen && (
          <>
            <p className="muted small" style={{ marginTop: 6 }}>
              If AdGuard Home (or any DNS blocker) filters these hosts, model downloads
              stall — often near the end, when Ollama fetches blobs from its Cloudflare
              CDN (<span className="mono">*.r2.cloudflarestorage.com</span>). Paste these
              into AdGuard Home → Filters → <strong>Custom filtering rules</strong>.
            </p>
            <pre style={{
              background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6,
              padding: '8px 10px', fontSize: 11, fontFamily: 'var(--font-mono)',
              overflowX: 'auto', margin: '4px 0', userSelect: 'text', color: 'var(--text)',
            }}>{ADGUARD_RULES.join('\n')}</pre>
            <button className="secondary" onClick={() => {
              navigator.clipboard.writeText(ADGUARD_RULES.join('\n'));
              setAdguardCopied(true); setTimeout(() => setAdguardCopied(false), 1500);
            }}>{adguardCopied ? 'Copied ✓' : 'Copy rules'}</button>
            <p className="muted small" style={{ marginTop: 8 }}>
              These are DNS-level allow rules for the internet hosts Ollama, Whisper models
              (Hugging Face) and the installer (GitHub) come from. Traffic between ClipAI
              and this Companion is on your LAN and isn't affected by AdGuard.
            </p>
          </>
        )}
      </div>

      <div className="panel">
        <div className="row spread">
          <h2>App updates</h2>
          <button className="secondary" onClick={() => doCheckUpdate()} disabled={checkingUpdate}
            title="Ask your paired ClipAI server which Companion installer it's serving and compare it to this app's version">
            {checkingUpdate ? 'Checking…' : 'Check for updates'}
          </button>
        </div>
        <p className="muted small" style={{ marginTop: 4 }}>
          This app is v{status.app_version || updateInfo?.current || '…'}
          {updateInfo?.latest ? ` — your ClipAI serves v${updateInfo.latest}` : ''}.
          Updates download from your ClipAI server over the LAN (no GitHub needed once
          the server has the installer).
        </p>
        {updateInfo?.update_available && (
          <div className="row" style={{ gap: 10, alignItems: 'center' }}>
            <button onClick={doInstallUpdate} disabled={installingUpdate}
              title="Downloads the installer from ClipAI, launches it, and closes this app so it can update in place">
              {installingUpdate
                ? 'Updating…'
                : (updateInfo.latest && updateInfo.latest !== updateInfo.current
                    ? `⬆ Update to v${updateInfo.latest}`
                    // Same version, newer build (from-source rebuild) — say so
                    // instead of "Update to v0.2.4" while already on 0.2.4.
                    : `⬆ Install the latest build${updateInfo.latest_build ? ` (${updateInfo.latest_build})` : ''}`)}
            </button>
            <span className="muted small">
              {updateInfo.filename}
              {updateInfo.size ? ` · ${(updateInfo.size / (1024 * 1024)).toFixed(0)} MB` : ''}
              {updateInfo.source ? ` · from the server's ${updateInfo.source} copy` : ''}
            </span>
          </div>
        )}
        {updateMsg && <p className="muted small" style={{ marginTop: 6 }}>{updateMsg}</p>}
      </div>

      <div className="panel">
        <div className="row spread">
          <h2>Pipeline activity</h2>
          <button className="secondary" onClick={doExportLogs} disabled={exporting}
            title="Save a full text report — config, connection, GPU, Ollama, every job/error since the app opened, plus the app log — to your Downloads folder">
            {exporting ? 'Exporting…' : '⤓ Export logs'}
          </button>
        </div>
        {status.job_logs.length === 0 ? (
          <p className="muted small">
            {status.clipai_connected
              ? 'Connected — no jobs yet. Each video analysis ClipAI runs on this GPU appears here as its own log.'
              : 'Nothing yet — activity appears here when ClipAI sends work to this GPU.'}
          </p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {status.job_logs.slice(0, 6).map((jl) => (
              <div key={jl.job_id || 'misc'}>
                <div className="row spread" style={{ marginBottom: 4 }}>
                  <strong style={{ fontSize: 13 }}>
                    {jl.job_title
                      || (jl.job_id ? `Job ${jl.job_id.slice(0, 8)}` : 'Ad-hoc requests')}
                  </strong>
                  {jl.active
                    ? <span className="badge live">
                        Running{jl.reported_stage ? ` · ${jl.reported_stage}` : ''}
                        {jl.reported_progress >= 0 ? ` ${jl.reported_progress}%` : ''}
                        {' · '}{elapsed(jl.started_at_ms)}
                      </span>
                    : <span className="badge">done · {elapsed(jl.started_at_ms, jl.last_activity_ms)}</span>}
                </div>
                {/* Active by heartbeat but no request in flight → ClipAI is on a
                    stage that runs on the SERVER (video decode/encode, offline
                    NMT translation) — this GPU isn't needed for it right now. */}
                {jl.active && !jl.entries.some((e) => e.finished_at_ms === null) && (
                  <div className="muted small" style={{ marginBottom: 4 }}>
                    Job still running on ClipAI — this stage
                    {jl.reported_stage ? ` (${jl.reported_stage})` : ''} runs on the server;
                    this GPU resumes when the next AI/transcription step starts.
                  </div>
                )}
                <div className="feed">
                  {jl.entries.slice(0, 12).map((a) => (
                    <div className="feed-item" key={a.id}>
                      <span className={`dot ${a.finished_at_ms === null ? 'ok' : (a.status ?? 500) < 400 ? 'ok' : 'bad'}`} />
                      <span style={{ minWidth: 66 }} className="badge">{a.kind}</span>
                      <span style={{ flex: 1 }}>
                        {a.stage || a.path}
                      </span>
                      <span className="muted mono">
                        {a.finished_at_ms === null
                          ? `${elapsed(a.started_at_ms)}…`
                          : elapsed(a.started_at_ms, a.finished_at_ms)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function useTheme(): ['dark' | 'light', () => void] {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('companion-theme');
    if (saved === 'light' || saved === 'dark') return saved;
    return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('companion-theme', theme);
  }, [theme]);
  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), []);
  return [theme, toggle];
}

export default function App() {
  const { status, refresh } = useStatus();
  const [forceDashboard, setForceDashboard] = useState(false);
  const [theme, toggleTheme] = useTheme();

  if (!status) {
    return <div className="app"><p className="muted">Starting…</p></div>;
  }
  if (!status.config.setup_complete && !forceDashboard) {
    return (
      <Wizard status={status} refresh={refresh}
        onDone={() => { setConfig({ setup_complete: true }).then(refresh); setForceDashboard(true); }} />
    );
  }
  return <Dashboard status={status} refresh={refresh} theme={theme} toggleTheme={toggleTheme} />;
}
