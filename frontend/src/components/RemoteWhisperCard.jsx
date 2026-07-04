import { useEffect, useState } from 'react';
import { showToast } from './Toast';

// Remote Whisper — point transcription at any OpenAI-compatible
// /v1/audio/transcriptions server (the GPU Companion, speaches,
// whisper-asr-webservice, whisper.cpp server). All probing is
// server-side; the API key is write-only client-side.

const inputStyle = {
  flex: 1, padding: '7px 10px', background: 'var(--bg-base)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
  color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)',
  boxSizing: 'border-box', minWidth: 0,
};

export default function RemoteWhisperCard({ isMobile = false }) {
  const [url, setUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [hasKey, setHasKey] = useState(false);
  const [clearKey, setClearKey] = useState(false);
  const [model, setModel] = useState('');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let alive = true;
    fetch('/api/settings/whisper-remote')
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (!alive || !data) return;
        setUrl(data.url || '');
        setHasKey(!!data.has_api_key);
        setModel(data.model || '');
      })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      const body = { url: url.trim(), model: model.trim() };
      if (clearKey) body.api_key = '';
      else if (apiKey.trim()) body.api_key = apiKey.trim();
      const res = await fetch('/api/settings/whisper-remote', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setHasKey(!!data.has_api_key);
      setApiKey('');
      setClearKey(false);
      setDirty(false);
      showToast('Remote Whisper settings saved', 'success');
    } catch {
      showToast('Failed to save Remote Whisper settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    setResult(null);
    try {
      const body = { url: url.trim(), model: model.trim() };
      if (apiKey.trim()) body.api_key = apiKey.trim();
      const res = await fetch('/api/settings/whisper-remote/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      setResult(await res.json());
    } catch {
      setResult({ online: false, error: 'Request failed — is the backend reachable?' });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)', padding: '12px 14px', marginBottom: 12,
    }}>
      <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
        Remote Whisper
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, marginBottom: 10 }}>
        Send transcription to a faster GPU on your LAN (the GPU Companion, or any
        OpenAI-compatible <code>/v1/audio/transcriptions</code> server). Only extracted
        audio is uploaded — never the video. If the server drops, jobs fall back to
        local Whisper automatically.
      </div>
      <div style={{ display: 'flex', gap: 6, flexDirection: isMobile ? 'column' : 'row', marginBottom: 6 }}>
        <input
          placeholder="Server URL (e.g. http://192.168.1.50:11500)"
          value={url}
          onChange={(e) => { setUrl(e.target.value); setDirty(true); setResult(null); }}
          style={inputStyle}
        />
        <input
          type="password"
          placeholder={hasKey && !clearKey ? 'API key saved — type to replace' : 'API key (optional)'}
          value={apiKey}
          disabled={clearKey}
          onChange={(e) => { setApiKey(e.target.value); setDirty(true); }}
          style={{ ...inputStyle, opacity: clearKey ? 0.5 : 1 }}
        />
      </div>
      <div style={{ display: 'flex', gap: 6, flexDirection: isMobile ? 'column' : 'row', marginBottom: 8 }}>
        <input
          placeholder="Model override (blank = auto: large-v3-turbo / large-v3)"
          value={model}
          onChange={(e) => { setModel(e.target.value); setDirty(true); }}
          style={inputStyle}
        />
        {hasKey && (
          <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', gap: 4, alignItems: 'center', whiteSpace: 'nowrap' }}>
            <input
              type="checkbox"
              checked={clearKey}
              onChange={(e) => { setClearKey(e.target.checked); setApiKey(''); setDirty(true); }}
              style={{ accentColor: 'var(--accent)' }}
            />
            Remove saved key
          </label>
        )}
      </div>
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          onClick={test}
          disabled={testing || !url.trim()}
          style={{
            padding: '6px 12px', background: 'var(--bg-panel)', color: 'var(--text-secondary)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
            opacity: testing || !url.trim() ? 0.5 : 1,
          }}
        >
          {testing ? 'Testing…' : 'Test'}
        </button>
        {dirty && (
          <button
            onClick={save}
            disabled={saving}
            style={{
              padding: '6px 14px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
              border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
              opacity: saving ? 0.5 : 1,
            }}
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        )}
      </div>
      {result && (
        <div style={{
          marginTop: 8, padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11, lineHeight: 1.5,
          background: result.online ? 'var(--success-dim)' : 'var(--danger-dim)',
          color: result.online ? 'var(--success)' : 'var(--danger)',
        }}>
          {result.online ? (
            <>
              Server online{result.detail ? ` — ${result.detail}` : ''}.
              Will transcribe with <strong>{result.model}</strong>
              {result.model_source === 'auto' ? ' (auto-selected)' : ''}.
            </>
          ) : (
            <>Unreachable{result.error ? ` — ${result.error}` : ''}</>
          )}
        </div>
      )}
    </div>
  );
}
