import React, { useEffect, useState } from 'react';

// Self-contained Settings section for the Primary AI (VideoLLaMA3) clip
// generation defaults: clip length, count, focus subjects and the editable
// discovery prompt. Loads + saves /api/clip-generation/settings on its own.

const card = {
  background: 'var(--bg-panel)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
};
const labelStyle = {
  fontSize: 11, color: 'var(--text-muted)', display: 'block', marginBottom: 4,
};
const inputStyle = {
  width: '100%', padding: '8px 10px', background: 'var(--bg-base)',
  color: 'var(--text-primary)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)', fontSize: 13, boxSizing: 'border-box',
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

const PLACEHOLDER_HINT =
  'Placeholders substituted at runtime: {start}, {end}, {transcript}, '
  + '{platforms}, {min_duration}, {max_duration}, {ideal_duration}, '
  + '{preferred}, {avoid}.';

export default function ClipGenerationSettings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [defaults, setDefaults] = useState(null);
  const [defaultPrompt, setDefaultPrompt] = useState('');
  const [form, setForm] = useState({
    min_duration: 60, max_duration: 300, clip_count: 0,
    preferred_subjects: '', avoid_subjects: '', discovery_prompt: '',
  });

  useEffect(() => {
    let alive = true;
    fetch('/api/clip-generation/settings')
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        setDefaults(d.defaults || null);
        setDefaultPrompt(d.default_discovery_prompt || '');
        setForm({
          min_duration: d.min_duration ?? 60,
          max_duration: d.max_duration ?? 300,
          clip_count: d.clip_count ?? 0,
          preferred_subjects: d.preferred_subjects || '',
          avoid_subjects: d.avoid_subjects || '',
          // Show the effective prompt — the saved custom one, or the
          // built-in default so the user can see exactly what is sent.
          discovery_prompt: d.discovery_prompt || d.default_discovery_prompt || '',
        });
      })
      .catch(() => { if (alive) setError('Could not load clip-generation settings.'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const set = (k, v) => { setForm((f) => ({ ...f, [k]: v })); setSaved(false); };

  const promptIsDefault =
    (form.discovery_prompt || '').trim() === (defaultPrompt || '').trim();

  const resetAll = () => {
    if (!defaults) return;
    setForm({
      min_duration: defaults.min_duration,
      max_duration: defaults.max_duration,
      clip_count: defaults.clip_count,
      preferred_subjects: defaults.preferred_subjects,
      avoid_subjects: defaults.avoid_subjects,
      discovery_prompt: defaultPrompt,
    });
    setSaved(false);
  };

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      // If the prompt still matches the built-in default, persist "" so a
      // future default change propagates instead of pinning the old text.
      const promptOut = promptIsDefault ? '' : form.discovery_prompt;
      const res = await fetch('/api/clip-generation/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          min_duration: Number(form.min_duration) || 0,
          max_duration: Number(form.max_duration) || 0,
          clip_count: Number(form.clip_count) || 0,
          preferred_subjects: form.preferred_subjects,
          avoid_subjects: form.avoid_subjects,
          discovery_prompt: promptOut,
        }),
      });
      if (!res.ok) throw new Error('save failed');
      const d = await res.json();
      setForm((f) => ({
        ...f,
        min_duration: d.min_duration,
        max_duration: d.max_duration,
        clip_count: d.clip_count,
      }));
      setSaved(true);
    } catch {
      setError('Could not save — check the server logs.');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div style={card}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Loading clip-generation settings…
        </div>
      </div>
    );
  }

  return (
    <div style={card}>
      <h4 style={{ fontSize: 13, margin: '0 0 8px', color: 'var(--text-primary)' }}>
        Clip Generation
      </h4>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Defaults the Primary AI (VideoLLaMA3) uses to choose viral clips on every
        analysis. The discovery prompt below is the exact instruction sent to the
        model — edit it to change how it picks clips. Saved settings persist
        across container restarts.
      </p>

      {/* clip length + count */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 12 }}>
        <div style={{ flex: '1 1 130px' }}>
          <label style={labelStyle}>Min clip length (s)</label>
          <input
            type="number" min={5} max={1800} style={inputStyle}
            value={form.min_duration}
            onChange={(e) => set('min_duration', e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 130px' }}>
          <label style={labelStyle}>Max clip length (s)</label>
          <input
            type="number" min={5} max={3600} style={inputStyle}
            value={form.max_duration}
            onChange={(e) => set('max_duration', e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 130px' }}>
          <label style={labelStyle}>Clip count (0 = auto)</label>
          <input
            type="number" min={0} max={100} style={inputStyle}
            value={form.clip_count}
            onChange={(e) => set('clip_count', e.target.value)}
          />
        </div>
      </div>

      {/* focus subjects */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
        <div style={{ flex: '1 1 220px' }}>
          <label style={labelStyle}>Preferred subjects</label>
          <input
            type="text" style={inputStyle} placeholder="funny moments, hot takes, drama"
            value={form.preferred_subjects}
            onChange={(e) => set('preferred_subjects', e.target.value)}
          />
        </div>
        <div style={{ flex: '1 1 220px' }}>
          <label style={labelStyle}>Avoid subjects</label>
          <input
            type="text" style={inputStyle} placeholder="sponsor segments, dead air"
            value={form.avoid_subjects}
            onChange={(e) => set('avoid_subjects', e.target.value)}
          />
        </div>
      </div>

      {/* editable discovery prompt */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 8,
        justifyContent: 'space-between', alignItems: 'baseline',
      }}>
        <label style={labelStyle}>
          VideoLLaMA3 discovery prompt
          {!promptIsDefault && (
            <span style={{ color: 'var(--accent-amber)', marginLeft: 6 }}>Modified</span>
          )}
        </label>
        <button
          onClick={() => set('discovery_prompt', defaultPrompt)}
          disabled={promptIsDefault}
          style={{ ...resetBtn, opacity: promptIsDefault ? 0.4 : 1 }}
        >
          Reset prompt to default
        </button>
      </div>
      <textarea
        value={form.discovery_prompt}
        onChange={(e) => set('discovery_prompt', e.target.value)}
        rows={12}
        style={{
          ...inputStyle, fontFamily: 'var(--font-mono)', fontSize: 11,
          lineHeight: 1.5, resize: 'vertical',
        }}
      />
      <p style={{ fontSize: 10, color: 'var(--text-muted)', margin: '6px 0 0', lineHeight: 1.5 }}>
        {PLACEHOLDER_HINT} Reset it to default any time to restore intended behavior.
      </p>

      {/* actions */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 10,
        alignItems: 'center', marginTop: 14,
      }}>
        <button onClick={save} disabled={saving} style={saveBtn}>
          {saving ? 'Saving…' : 'Save'}
        </button>
        <button onClick={resetAll} style={resetBtn}>
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
