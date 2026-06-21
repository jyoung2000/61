import React, { useEffect, useState } from 'react';

// Self-contained Settings section for subtitle readability rules,
// translation engine selection, and audio event detection. Loads + saves
// /api/subtitle-quality/settings on its own.

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
const selectStyle = {
  padding: '6px 10px', background: 'var(--bg-base)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
  fontSize: 12, width: '100%', boxSizing: 'border-box',
};
const saveBtn = {
  padding: '8px 18px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
  border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 13,
  fontWeight: 600, cursor: 'pointer',
};
const toggleLabel = {
  display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8,
  fontSize: 12, color: 'var(--text-primary)', cursor: 'pointer',
};

const PLATFORM_OPTIONS = [
  { value: '',           label: 'None (no safe-zone insets)' },
  { value: 'tiktok',     label: 'TikTok (1080×1920)' },
  { value: 'reels',      label: 'Instagram Reels (1080×1920)' },
  { value: 'shorts',     label: 'YouTube Shorts (1080×1920)' },
  { value: 'horizontal', label: 'Horizontal (16:9)' },
  { value: 'square',     label: 'Square (1:1)' },
];

const ENGINE_OPTIONS = [
  { value: 'auto',    label: 'Auto (DeepL → Google → Opus-MT → NLLB → LLM)' },
  { value: 'deepl',   label: 'DeepL (cloud, best fluency, requires key)' },
  { value: 'google',  label: 'Google Cloud Translation (requires key)' },
  { value: 'fugumt',  label: 'FuguMT (local, Japanese↔English specialist — best for JA)' },
  { value: 'opus-mt', label: 'Opus-MT (local, fastest, per-pair download)' },
  { value: 'nllb',    label: 'NLLB-200 (local, 200 languages, ~600 MB)' },
  { value: 'llm',     label: 'LLM via orchestrator (legacy)' },
  { value: 'whisper', label: 'Whisper translate (legacy, lower quality)' },
];

// ISO 639-1 codes shared with the Upload page language picker. Used to
// choose the source→target pair to pre-download (NLLB / Opus-MT).
const LANG_OPTIONS = [
  { value: 'en', label: 'English' },   { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },    { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },   { value: 'pt', label: 'Portuguese' },
  { value: 'ru', label: 'Russian' },   { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },    { value: 'zh', label: 'Chinese' },
  { value: 'ar', label: 'Arabic' },    { value: 'hi', label: 'Hindi' },
  { value: 'nl', label: 'Dutch' },     { value: 'pl', label: 'Polish' },
  { value: 'tr', label: 'Turkish' },   { value: 'vi', label: 'Vietnamese' },
  { value: 'th', label: 'Thai' },      { value: 'uk', label: 'Ukrainian' },
  { value: 'sv', label: 'Swedish' },   { value: 'id', label: 'Indonesian' },
  { value: 'ms', label: 'Malay' },     { value: 'tl', label: 'Filipino' },
];

// NMT device policy — cpu is the safe choice on 4 GB GPUs.
const NMT_DEVICE_OPTIONS = [
  { value: 'auto', label: 'Auto (CUDA if available, else CPU)' },
  { value: 'cpu',  label: 'CPU (safe on 4 GB GPUs, ~minutes/video)' },
  { value: 'cuda', label: 'CUDA (GPU — may OOM on 4 GB)' },
];

export default function SubtitleQualitySettings() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [downloadingModel, setDownloadingModel] = useState(false);
  const [downloadMsg, setDownloadMsg] = useState('');
  // Source→target pair to pre-download (Opus-MT needs a specific pair; NLLB is
  // a single multilingual model). Defaults: Japanese → English.
  const [dlSource, setDlSource] = useState('ja');
  const [dlTarget, setDlTarget] = useState('en');
  const [form, setForm] = useState({
    subtitle_cps_enforcement: true,
    subtitle_max_cps: 20,
    subtitle_max_chars_per_line: 42,
    subtitle_min_duration_ms: 833,
    subtitle_max_duration_ms: 7000,
    subtitle_smart_line_breaks: true,
    subtitle_platform_safe_zones: true,
    subtitle_platform_profile: '',
    transcript_polishing_enabled: true,
    transcript_filler_removal: true,
    transcript_sentence_repair: true,
    translation_engine: 'auto',
    translation_context_window: 5,
    translation_glossary_enabled: true,
    nmt_device: 'auto',
    audio_event_detection: true,
    audio_events_in_subtitles: false,
    audio_music_detection: true,
    google_translate_api_key: '',
    deepl_api_key: '',
    google_translate_configured: false,
    deepl_configured: false,
  });

  useEffect(() => {
    let alive = true;
    fetch('/api/subtitle-quality/settings')
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        setForm((prev) => ({ ...prev, ...d }));
      })
      .catch(() => { if (alive) setError('Could not load subtitle settings.'); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const set = (k, v) => { setForm((f) => ({ ...f, [k]: v })); setSaved(false); };

  const save = async () => {
    setSaving(true);
    setError('');
    try {
      const body = { ...form };
      // Don't send blank keys — backend ignores blanks but be explicit.
      if (!body.google_translate_api_key) delete body.google_translate_api_key;
      if (!body.deepl_api_key) delete body.deepl_api_key;
      delete body.google_translate_configured;
      delete body.deepl_configured;
      const res = await fetch('/api/subtitle-quality/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error('save failed');
      const d = await res.json();
      setForm((f) => ({
        ...f, ...d,
        google_translate_api_key: '', deepl_api_key: '',  // never echo back
      }));
      setSaved(true);
    } catch {
      setError('Could not save — check the server logs.');
    } finally {
      setSaving(false);
    }
  };

  // Engine-aware model download. NLLB is a single multilingual model;
  // Opus-MT is per-pair so we pass the selected source→target. For `auto`
  // we pre-download the Opus-MT pair (preferred local engine), falling the
  // user back to the explicit NLLB button if they want broad coverage.
  const downloadModel = async (engine) => {
    setDownloadingModel(true);
    setError('');
    setDownloadMsg('');
    const isOpus = engine === 'opus-mt';
    const body = isOpus
      ? { engine: 'opus-mt', source: dlSource, target: dlTarget }
      : { engine: 'nllb' };
    const label = isOpus
      ? `Opus-MT ${dlSource}→${dlTarget}`
      : 'NLLB-200';
    try {
      const res = await fetch('/api/translation/download-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await res.json();
      if (d.status === 'ok') {
        setDownloadMsg(`${label} ready ✓ (${d.path || 'cached'})`);
      } else {
        setError(`${label} download failed: ${d.message || 'unknown'}`);
      }
    } catch (e) {
      setError(`${label} download failed: ${e.message || e}`);
    } finally {
      setDownloadingModel(false);
    }
  };

  const uploadGlossary = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const json = JSON.parse(text);
      // We don't have a job context here — just validate the file shape.
      const terms = json.terms || json;
      const count = Object.keys(terms || {}).length;
      alert(`Glossary loaded: ${count} terms. Use the job-specific Glossary endpoint to apply it: POST /api/jobs/{job_id}/glossary`);
    } catch (err) {
      alert(`Invalid glossary JSON: ${err.message}`);
    } finally {
      e.target.value = '';
    }
  };

  if (loading) {
    return (
      <div style={card}>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Loading subtitle quality settings…
        </div>
      </div>
    );
  }

  return (
    <div style={card}>
      <h4 style={{ fontSize: 13, margin: '0 0 8px', color: 'var(--text-primary)' }}>
        Subtitle Quality
      </h4>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
        Netflix-style readability rules, per-platform safe zones, translation
        engine selection, and audio-event detection. Toggling any feature off
        reverts the pipeline to its legacy behavior for that step.
      </p>

      {/* Readability */}
      <div style={{
        borderTop: '1px solid var(--border)', paddingTop: 10, marginBottom: 14,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
          Readability
        </div>

        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.subtitle_cps_enforcement}
            onChange={(e) => set('subtitle_cps_enforcement', e.target.checked)}
          />
          <span>CPS / duration enforcement</span>
        </label>

        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.subtitle_smart_line_breaks}
            onChange={(e) => set('subtitle_smart_line_breaks', e.target.checked)}
            disabled={!form.subtitle_cps_enforcement}
          />
          <span>Smart line breaks (sentence + clause boundaries)</span>
        </label>

        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.subtitle_platform_safe_zones}
            onChange={(e) => set('subtitle_platform_safe_zones', e.target.checked)}
          />
          <span>Platform-aware safe-zone margins</span>
        </label>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 10 }}>
          <div style={{ flex: '1 1 160px' }}>
            <label style={labelStyle}>Max CPS ({form.subtitle_max_cps})</label>
            <input
              type="range" min={10} max={25} step={1}
              value={form.subtitle_max_cps}
              onChange={(e) => set('subtitle_max_cps', Number(e.target.value))}
              style={{ width: '100%' }}
              disabled={!form.subtitle_cps_enforcement}
            />
          </div>
          <div style={{ flex: '1 1 160px' }}>
            <label style={labelStyle}>Max chars/line</label>
            <input
              type="number" min={20} max={60} style={inputStyle}
              value={form.subtitle_max_chars_per_line}
              onChange={(e) => set('subtitle_max_chars_per_line', Number(e.target.value))}
              disabled={!form.subtitle_cps_enforcement}
            />
          </div>
          <div style={{ flex: '1 1 180px' }}>
            <label style={labelStyle}>Platform safe-zone profile</label>
            <select
              style={selectStyle} value={form.subtitle_platform_profile}
              onChange={(e) => set('subtitle_platform_profile', e.target.value)}
              disabled={!form.subtitle_platform_safe_zones}
            >
              {PLATFORM_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Transcript polishing */}
      <div style={{
        borderTop: '1px solid var(--border)', paddingTop: 10, marginBottom: 14,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
          Transcript Polishing
        </div>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.transcript_polishing_enabled}
            onChange={(e) => set('transcript_polishing_enabled', e.target.checked)}
          />
          <span>AI-powered polishing (proper nouns, punctuation, sentence repair)</span>
        </label>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.transcript_filler_removal}
            onChange={(e) => set('transcript_filler_removal', e.target.checked)}
            disabled={!form.transcript_polishing_enabled}
          />
          <span>Remove fillers (um, uh, like, you know)</span>
        </label>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.transcript_sentence_repair}
            onChange={(e) => set('transcript_sentence_repair', e.target.checked)}
            disabled={!form.transcript_polishing_enabled}
          />
          <span>Sentence boundary repair</span>
        </label>
      </div>

      {/* Translation engine */}
      <div style={{
        borderTop: '1px solid var(--border)', paddingTop: 10, marginBottom: 14,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
          Translation Engine
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 10 }}>
          <div style={{ flex: '1 1 260px' }}>
            <label style={labelStyle}>Engine</label>
            <select
              style={selectStyle} value={form.translation_engine}
              onChange={(e) => set('translation_engine', e.target.value)}
            >
              {ENGINE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <div style={{ flex: '1 1 200px' }}>
            <label style={labelStyle}>Local NMT device</label>
            <select
              style={selectStyle} value={form.nmt_device}
              onChange={(e) => set('nmt_device', e.target.value)}
            >
              {NMT_DEVICE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 10 }}>
          <div style={{ flex: '1 1 160px' }}>
            <label style={labelStyle}>Context window (segments)</label>
            <input
              type="number" min={0} max={20} style={inputStyle}
              value={form.translation_context_window}
              onChange={(e) => set('translation_context_window', Number(e.target.value))}
            />
          </div>
          <div style={{ flex: '1 1 230px', display: 'flex', alignItems: 'center', gap: 8 }}>
            <label style={{ ...toggleLabel, margin: 0 }}>
              <input
                type="checkbox" checked={!!form.translation_glossary_enabled}
                onChange={(e) => set('translation_glossary_enabled', e.target.checked)}
              />
              <span>Per-video glossary (KNP)</span>
            </label>
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 10 }}>
          <div style={{ flex: '1 1 220px' }}>
            <label style={labelStyle}>
              DeepL API key {form.deepl_configured && <span style={{ color: 'var(--success, #30D158)' }}>✓ configured</span>}
            </label>
            <input
              type="password" style={inputStyle} placeholder="leave blank to keep existing"
              value={form.deepl_api_key}
              onChange={(e) => set('deepl_api_key', e.target.value)}
            />
          </div>
          <div style={{ flex: '1 1 220px' }}>
            <label style={labelStyle}>
              Google Translate API key {form.google_translate_configured && <span style={{ color: 'var(--success, #30D158)' }}>✓ configured</span>}
            </label>
            <input
              type="password" style={inputStyle} placeholder="leave blank to keep existing"
              value={form.google_translate_api_key}
              onChange={(e) => set('google_translate_api_key', e.target.value)}
            />
          </div>
        </div>

        {/* Offline model download — NLLB (single multilingual) + Opus-MT (per-pair) */}
        <div style={{ marginTop: 6 }}>
          <label style={labelStyle}>Download offline model (one-time, on-demand)</label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 8 }}>
            <select
              style={{ ...selectStyle, width: 'auto', flex: '0 0 130px' }}
              value={dlSource} onChange={(e) => setDlSource(e.target.value)}
              title="Opus-MT source language"
            >
              {LANG_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>→</span>
            <select
              style={{ ...selectStyle, width: 'auto', flex: '0 0 130px' }}
              value={dlTarget} onChange={(e) => setDlTarget(e.target.value)}
              title="Opus-MT target language"
            >
              {LANG_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => downloadModel('opus-mt')}
              disabled={downloadingModel || dlSource === dlTarget}
              style={{
                padding: '6px 12px', background: 'var(--bg-elevated)',
                color: 'var(--text-secondary)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', fontSize: 11, cursor: 'pointer',
                opacity: (downloadingModel || dlSource === dlTarget) ? 0.5 : 1,
              }}
            >
              {downloadingModel ? 'Downloading…' : `Download Opus-MT (${dlSource}→${dlTarget}, ~300 MB)`}
            </button>
            <button
              type="button"
              onClick={() => downloadModel('nllb')}
              disabled={downloadingModel}
              style={{
                padding: '6px 12px', background: 'var(--bg-elevated)',
                color: 'var(--text-secondary)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', fontSize: 11, cursor: 'pointer',
                opacity: downloadingModel ? 0.5 : 1,
              }}
            >
              {downloadingModel ? 'Downloading…' : 'Download NLLB-200 (~600 MB)'}
            </button>
            <label
              style={{
                padding: '6px 12px', background: 'var(--bg-elevated)',
                color: 'var(--text-secondary)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)', fontSize: 11, cursor: 'pointer',
              }}
            >
              Validate glossary JSON
              <input
                type="file" accept=".json" onChange={uploadGlossary}
                style={{ display: 'none' }}
              />
            </label>
          </div>
          {downloadMsg && (
            <div style={{ fontSize: 11, color: 'var(--success, #30D158)', marginTop: 8 }}>
              {downloadMsg}
            </div>
          )}
        </div>
      </div>

      {/* Audio event detection */}
      <div style={{
        borderTop: '1px solid var(--border)', paddingTop: 10, marginBottom: 14,
      }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
          Audio Event Detection
        </div>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.audio_event_detection}
            onChange={(e) => set('audio_event_detection', e.target.checked)}
          />
          <span>Classify audio events (speech / music / laughter / applause / silence)</span>
        </label>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.audio_events_in_subtitles}
            onChange={(e) => set('audio_events_in_subtitles', e.target.checked)}
            disabled={!form.audio_event_detection}
          />
          <span>Inject [applause] / [music] / [laughter] into subtitle track</span>
        </label>
        <label style={toggleLabel}>
          <input
            type="checkbox" checked={!!form.audio_music_detection}
            onChange={(e) => set('audio_music_detection', e.target.checked)}
            disabled={!form.audio_event_detection}
          />
          <span>Music detection (sustained harmonic content)</span>
        </label>
      </div>

      {/* actions */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 10,
        alignItems: 'center', marginTop: 14,
      }}>
        <button onClick={save} disabled={saving} style={saveBtn}>
          {saving ? 'Saving…' : 'Save'}
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
