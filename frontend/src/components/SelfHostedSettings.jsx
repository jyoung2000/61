import React, { useEffect, useState } from 'react';

// Self-contained Settings section: a master "Self-Hosted Mode" toggle plus
// per-engine cloud/local overrides. Loads + saves /api/self-hosted/settings.

const card = {
  background: 'var(--bg-panel)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
};
const selectStyle = {
  padding: '6px 10px', background: 'var(--bg-base)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 12,
};
const saveBtn = {
  padding: '8px 18px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
  border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 13,
  fontWeight: 600, cursor: 'pointer',
};
const resetBtn = {
  padding: '6px 12px', background: 'transparent', color: 'var(--text-secondary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
  fontSize: 11, cursor: 'pointer',
};

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
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [ollamaHost, setOllamaHost] = useState('');
  const [form, setForm] = useState({
    self_hosted_mode: false,
    clip_engine_source: 'auto',
    editorial_ai_source: 'auto',
  });

  useEffect(() => {
    let alive = true;
    fetch('/api/self-hosted/settings')
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        setForm({
          self_hosted_mode: !!d.self_hosted_mode,
          clip_engine_source: d.clip_engine_source || 'auto',
          editorial_ai_source: d.editorial_ai_source || 'auto',
        });
        setOllamaHost(d.ollama_host || '');
      })
      .catch(() => { if (alive) setError('Could not load self-hosted settings.'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const set = (k, v) => { setForm((f) => ({ ...f, [k]: v })); setSaved(false); };

  // Resolve a source the same way the backend does, for live UI feedback.
  const resolve = (src) =>
    (src === 'local' || src === 'cloud')
      ? src
      : (form.self_hosted_mode ? 'local' : 'cloud');

  const isDefault = !form.self_hosted_mode
    && form.clip_engine_source === 'auto'
    && form.editorial_ai_source === 'auto';

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      const res = await fetch('/api/self-hosted/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error('save failed');
      setSaved(true);
    } catch {
      setError('Could not save — check the server logs.');
    } finally {
      setSaving(false);
    }
  };

  const reset = () => {
    setForm({
      self_hosted_mode: false,
      clip_engine_source: 'auto',
      editorial_ai_source: 'auto',
    });
    setSaved(false);
  };

  if (loading) {
    return (
      <div style={card}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Loading self-hosted settings…
        </div>
      </div>
    );
  }

  return (
    <div style={card}>
      <h4 style={{ fontSize: 13, margin: '0 0 8px', color: 'var(--text-primary)' }}>
        Self-Hosted Mode
      </h4>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
        Run the analysis pipeline on local AI — no cloud calls. The master toggle
        flips every engine set to “Auto”; a per-engine override wins over it.
        Local AI uses your Ollama server — if Ollama is unreachable the run fails
        loudly rather than silently falling back to the cloud.
      </p>

      {/* master toggle */}
      <label style={{
        display: 'flex', alignItems: 'center', gap: 10,
        cursor: 'pointer', padding: '6px 0 10px',
      }}>
        <input
          type="checkbox"
          checked={form.self_hosted_mode}
          onChange={(e) => set('self_hosted_mode', e.target.checked)}
          style={{ width: 16, height: 16, accentColor: 'var(--accent-cyan)' }}
        />
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
          Self-Hosted Mode — {form.self_hosted_mode ? 'on' : 'off'}
        </span>
      </label>

      {/* per-engine overrides */}
      <EngineRow
        label="Clip engine (Primary AI)"
        desc="Cloud = VideoLLaMA3 on Replicate · Local = Ollama vision model"
        value={form.clip_engine_source}
        resolved={resolve(form.clip_engine_source)}
        onChange={(v) => set('clip_engine_source', v)}
      />
      <EngineRow
        label="Editorial AI"
        desc="Summaries, tags, SEO, viral scoring, transcript translation"
        value={form.editorial_ai_source}
        resolved={resolve(form.editorial_ai_source)}
        onChange={(v) => set('editorial_ai_source', v)}
      />

      <div style={{
        marginTop: 12, padding: '8px 10px', background: 'var(--bg-base)',
        border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
        fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5,
      }}>
        Whisper transcription already runs locally, and transcript polishing is an
        inert pass-through since the engine swap — both stay local regardless of
        this setting.{' '}
        {ollamaHost ? `Ollama host: ${ollamaHost}` : 'Set OLLAMA_HOST to point at your Ollama server.'}
      </div>

      {/* actions */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 10,
        alignItems: 'center', marginTop: 14,
      }}>
        <button onClick={save} disabled={saving} style={saveBtn}>
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button
          onClick={reset}
          disabled={isDefault}
          style={{ ...resetBtn, opacity: isDefault ? 0.4 : 1 }}
        >
          Reset to defaults
        </button>
        {saved && (
          <span style={{ fontSize: 11, color: 'var(--success, #30D158)' }}>Saved ✓</span>
        )}
        {error && (
          <span style={{ fontSize: 11, color: 'var(--danger, #FF375F)' }}>{error}</span>
        )}
      </div>
    </div>
  );
}
