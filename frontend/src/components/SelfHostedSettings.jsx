import React, { useEffect, useState } from 'react';

// Settings → AI Provider section: the "Offline Mode" toggle. One switch flips
// the whole analysis pipeline onto the local GPU (the GTX 1650) — clip
// detection, transcription, translation and polishing all run locally with no
// cloud calls. A four-stage status grid shows where each stage actually runs,
// and a collapsible Advanced block keeps the per-engine cloud/local overrides.
// Loads + saves /api/self-hosted/settings (routing is read live, no restart).

const card = {
  background: 'var(--bg-panel)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
};
const selectStyle = {
  padding: '6px 10px', background: 'var(--bg-base)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 12,
};

// The four user-facing pipeline stages the toggle controls, with a one-liner
// describing what runs locally vs in the cloud. Keys match the backend's
// ``stages`` payload from /api/self-hosted/settings.
const STAGES = [
  {
    key: 'clip_detection', label: 'Clip Detection',
    local: 'Ollama vision model on the GPU',
    cloud: 'VideoLLaMA3 on Replicate (cloud GPU)',
  },
  {
    key: 'transcription', label: 'Transcription',
    local: 'Whisper on the GPU — always local',
    cloud: 'Whisper on the GPU — always local',
  },
  {
    key: 'translation', label: 'Translation',
    local: 'Local LLM, then offline Whisper/NMT',
    cloud: 'Cloud editorial LLM',
  },
  {
    key: 'polishing', label: 'Polishing',
    local: 'Local Ollama LLM',
    cloud: 'Cloud editorial LLM',
  },
];

function StageCell({ label, desc, resolved }) {
  const isLocal = resolved === 'local';
  return (
    <div style={{
      background: 'var(--bg-base)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)', padding: '10px 12px',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span>
        <span style={{
          fontSize: 9, fontWeight: 700, fontFamily: 'var(--font-mono)',
          letterSpacing: '0.06em', padding: '2px 6px', borderRadius: 'var(--radius-sm)',
          color: isLocal ? 'var(--success, #30D158)' : 'var(--accent-cyan)',
          background: isLocal ? 'rgba(48,209,88,0.12)' : 'rgba(34,211,238,0.12)',
          border: `1px solid ${isLocal ? 'rgba(48,209,88,0.3)' : 'rgba(34,211,238,0.3)'}`,
        }}>
          {isLocal ? 'LOCAL' : 'CLOUD'}
        </span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.4 }}>
        {desc}
      </div>
    </div>
  );
}

function EngineRow({ label, desc, value, resolved, onChange }) {
  return (
    <div style={{
      display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center',
      justifyContent: 'space-between', padding: '10px 0',
      borderTop: '1px solid var(--border)',
    }}>
      <div style={{ flex: '1 1 230px', minWidth: 0 }}>
        <div style={{ fontSize: 13, color: 'var(--text-primary)' }}>{label}</div>
        <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>{desc}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{
          fontSize: 10, fontWeight: 700, fontFamily: 'var(--font-mono)',
          color: resolved === 'local' ? 'var(--success, #30D158)' : 'var(--accent-cyan)',
        }}>
          {resolved === 'local' ? 'LOCAL' : 'CLOUD'}
        </span>
        <select style={selectStyle} value={value} onChange={(e) => onChange(e.target.value)}>
          <option value="auto">Auto</option>
          <option value="local">Local</option>
          <option value="cloud">Cloud</option>
        </select>
      </div>
    </div>
  );
}

export default function SelfHostedSettings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(0);
  const [error, setError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [ollamaHost, setOllamaHost] = useState('');
  const [stages, setStages] = useState({
    clip_detection: 'cloud', transcription: 'local',
    translation: 'cloud', polishing: 'cloud',
  });
  const [form, setForm] = useState({
    self_hosted_mode: false,
    clip_engine_source: 'auto',
    editorial_ai_source: 'auto',
  });

  const applyState = (d) => {
    setForm({
      self_hosted_mode: !!d.self_hosted_mode,
      clip_engine_source: d.clip_engine_source || 'auto',
      editorial_ai_source: d.editorial_ai_source || 'auto',
    });
    if (d.stages) setStages(d.stages);
    if (d.ollama_host !== undefined) setOllamaHost(d.ollama_host || '');
  };

  useEffect(() => {
    let alive = true;
    fetch('/api/self-hosted/settings')
      .then((r) => r.json())
      .then((d) => { if (alive) applyState(d); })
      .catch(() => { if (alive) setError('Could not load offline settings.'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  // Auto-clear the transient "Saved ✓" confirmation.
  useEffect(() => {
    if (!savedAt) return undefined;
    const t = setTimeout(() => setSavedAt(0), 2500);
    return () => clearTimeout(t);
  }, [savedAt]);

  // Persist a partial update immediately (the toggle + dropdowns auto-save).
  // The response echoes the resolved per-stage sources so the grid stays live.
  const persist = async (patch) => {
    const next = { ...form, ...patch };
    setForm(next);
    setSaving(true);
    setError('');
    try {
      const res = await fetch('/api/self-hosted/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next),
      });
      if (!res.ok) throw new Error('save failed');
      const d = await res.json();
      applyState(d);
      setSavedAt(Date.now());
    } catch {
      setError('Could not save — check the server logs.');
    } finally {
      setSaving(false);
    }
  };

  // Master switch. Turning it ON resets the per-engine overrides to Auto so
  // every stage genuinely follows the master onto the GPU; turning it OFF
  // returns Auto stages to the cloud. Advanced overrides can pin a stage after.
  const toggleOffline = () =>
    persist({ self_hosted_mode: !form.self_hosted_mode, clip_engine_source: 'auto', editorial_ai_source: 'auto' });

  const on = form.self_hosted_mode;
  const justSaved = savedAt > 0;

  if (loading) {
    return (
      <div style={card}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Loading offline settings…</div>
      </div>
    );
  }

  return (
    <div style={{
      ...card,
      border: `1px solid ${on ? 'rgba(48,209,88,0.45)' : 'var(--border)'}`,
    }}>
      {/* header + master switch */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ minWidth: 0 }}>
          <h4 style={{ fontSize: 14, margin: '0 0 4px', color: 'var(--text-primary)' }}>
            Offline Mode <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>(Local GPU)</span>
          </h4>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0, lineHeight: 1.5 }}>
            Run the whole pipeline on your local GPU — clip detection, transcription,
            translation and polishing — with no cloud calls.
          </p>
        </div>
        <div
          role="switch"
          aria-checked={on}
          tabIndex={0}
          onClick={toggleOffline}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleOffline(); } }}
          style={{
            position: 'relative', width: 46, height: 26, flexShrink: 0, cursor: 'pointer',
            borderRadius: 13, transition: 'background 0.2s', opacity: saving ? 0.6 : 1,
            background: on ? 'var(--success, #30D158)' : 'var(--border)',
          }}
        >
          <div style={{
            position: 'absolute', top: 3, left: on ? 23 : 3,
            width: 20, height: 20, borderRadius: '50%', background: '#fff',
            transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
          }} />
        </div>
      </div>

      {/* four-stage status grid */}
      <div style={{
        display: 'grid', gap: 8, marginTop: 14,
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
      }}>
        {STAGES.map((s) => (
          <StageCell
            key={s.key}
            label={s.label}
            resolved={stages[s.key] || 'cloud'}
            desc={(stages[s.key] === 'local') ? s.local : s.cloud}
          />
        ))}
      </div>

      {/* GPU sequencing note — the load/unload story on a 4 GB card */}
      <div style={{
        marginTop: 12, padding: '8px 10px', background: 'var(--bg-base)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
        fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.55,
      }}>
        Stages run <strong style={{ color: 'var(--text-secondary)' }}>one at a time</strong> and
        the pipeline unloads each model from VRAM before loading the next — so transcription,
        the editorial LLM and the clip-detection vision model never have to share the GPU at
        once. This is what lets everything fit on a small card like the GTX 1650 (4 GB).{' '}
        {on
          ? (ollamaHost
              ? <>Local AI uses your Ollama server (<code style={{ fontFamily: 'var(--font-mono)' }}>{ollamaHost}</code>); if it's unreachable a run fails loudly rather than silently using the cloud.</>
              : 'Set OLLAMA_HOST to point at your Ollama server.')
          : 'Turn it on to route every stage onto the GPU.'}
      </div>

      {/* advanced per-engine overrides */}
      <button
        onClick={() => setShowAdvanced((v) => !v)}
        style={{
          marginTop: 12, padding: 0, background: 'none', border: 'none',
          color: 'var(--accent-cyan)', fontSize: 11, cursor: 'pointer',
        }}
      >
        {showAdvanced ? '▾ Hide advanced overrides' : '▸ Advanced overrides (per stage)'}
      </button>
      {showAdvanced && (
        <div style={{ marginTop: 4 }}>
          <p style={{ fontSize: 10, color: 'var(--text-muted)', margin: '0 0 4px', lineHeight: 1.5 }}>
            Pin an individual engine to Local or Cloud. “Auto” follows the master toggle above.
          </p>
          <EngineRow
            label="Clip engine (Primary AI)"
            desc="Drives Clip Detection · Cloud = VideoLLaMA3 on Replicate, Local = Ollama vision"
            value={form.clip_engine_source}
            resolved={stages.clip_detection}
            onChange={(v) => persist({ clip_engine_source: v })}
          />
          <EngineRow
            label="Editorial AI"
            desc="Drives Translation + Polishing (also summaries, tags, SEO, viral scoring)"
            value={form.editorial_ai_source}
            resolved={stages.polishing}
            onChange={(v) => persist({ editorial_ai_source: v })}
          />
        </div>
      )}

      {/* save feedback */}
      <div style={{ minHeight: 16, marginTop: 8 }}>
        {saving && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Saving…</span>}
        {!saving && justSaved && <span style={{ fontSize: 11, color: 'var(--success, #30D158)' }}>Saved ✓</span>}
        {!saving && error && <span style={{ fontSize: 11, color: 'var(--danger, #FF375F)' }}>{error}</span>}
      </div>
    </div>
  );
}
