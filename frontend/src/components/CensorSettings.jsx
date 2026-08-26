import { useEffect, useRef, useState } from 'react';
import { showToast } from './Toast';

// Profanity Censor — Settings card.
// Configures the block list, the mask symbol (first + last letter of a
// blocked word stay visible: "shit" → "s**t"), and the beep sound used to
// cover the word's audio window on export. The editor's Export dialog has
// the per-export toggle; "censor by default" here sets its initial state.

const SYMBOLS = ['*', '#', '@', '!', '•', '█'];

export default function CensorSettings() {
  const [data, setData] = useState(null);
  const [wordsText, setWordsText] = useState('');
  const [maskChar, setMaskChar] = useState('*');
  const [enabledDefault, setEnabledDefault] = useState(false);
  const [sound, setSound] = useState('beep');
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef(null);

  const applyState = (d) => {
    setData(d);
    setWordsText((d.words || []).join('\n'));
    setMaskChar(d.mask_char || '*');
    setEnabledDefault(!!d.enabled_default);
    setSound(d.sound || 'beep');
  };

  useEffect(() => {
    let alive = true;
    fetch('/api/censor/settings')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d) applyState(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  const save = async (overrides = {}) => {
    setSaving(true);
    try {
      const words = wordsText.split(/[\n,]+/).map((w) => w.trim()).filter(Boolean);
      const res = await fetch('/api/censor/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled_default: enabledDefault,
          words,
          mask_char: maskChar,
          sound,
          ...overrides,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      applyState(await res.json());
      showToast('Censor settings saved', 'success');
    } catch {
      showToast('Failed to save censor settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const uploadSound = async (file) => {
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/censor/sound', { method: 'POST', body: form });
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail;
        throw new Error(detail || `HTTP ${res.status}`);
      }
      applyState(await res.json());
      showToast('Custom censor sound uploaded', 'success');
    } catch (e) {
      showToast(`Sound upload failed: ${e.message}`, 'error');
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  const removeSound = async () => {
    try {
      const res = await fetch('/api/censor/sound', { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      applyState(await res.json());
      showToast('Custom sound removed — using the beep tone', 'success');
    } catch {
      showToast('Failed to remove the custom sound', 'error');
    }
  };

  const inputStyle = {
    width: '100%', padding: '8px 12px', background: 'var(--bg-base)',
    border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
    color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)',
  };

  const sampleWord = 'curse';
  const sampleMasked = sampleWord[0] + maskChar.repeat(sampleWord.length - 2)
    + sampleWord[sampleWord.length - 1];

  return (
    <div id="censor" style={{
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: 24,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Profanity Censor</span>
        {saving && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Saving…</span>}
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5, margin: '0 0 10px' }}>
        When the censor toggle is on in the editor's Export dialog, blocked words are
        masked in burned subtitles (first and last letter kept — {sampleWord} becomes{' '}
        <strong style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
          {sampleMasked}
        </strong>) and their audio is muted under a beep.
      </p>

      {data === null ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading…</div>
      ) : (
        <>
          {/* Default state of the export toggle */}
          <label style={{
            display: 'flex', alignItems: 'center', gap: 8, fontSize: 12,
            color: 'var(--text-primary)', marginBottom: 10, cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={enabledDefault}
              onChange={(e) => setEnabledDefault(e.target.checked)}
            />
            Censor new exports by default (the Export dialog toggle starts ON)
          </label>

          {/* Block list */}
          <label style={{ display: 'block', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>
            Blocked words — one per line (or comma-separated).
            {data.using_default_words
              ? ' Currently using the built-in list; edit to customize.'
              : ' Custom list active — clear the box and Save to restore the built-in list.'}
          </label>
          <textarea
            value={wordsText}
            onChange={(e) => setWordsText(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder="one word per line"
            style={{ ...inputStyle, resize: 'vertical', minHeight: 90 }}
          />

          {/* Mask symbol */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '10px 0' }}>
            <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Mask symbol</span>
            {SYMBOLS.map((s) => (
              <button
                key={s}
                onClick={() => setMaskChar(s)}
                aria-pressed={maskChar === s}
                style={{
                  width: 30, height: 26, borderRadius: 'var(--radius-sm)',
                  border: `1px solid ${maskChar === s ? 'var(--accent-cyan)' : 'var(--border)'}`,
                  background: maskChar === s ? 'var(--cyan-dim, rgba(0,200,255,0.12))' : 'var(--bg-base)',
                  color: 'var(--text-primary)', fontSize: 13, cursor: 'pointer',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {s}
              </button>
            ))}
            <input
              value={maskChar}
              onChange={(e) => {
                const ch = e.target.value.slice(-1);
                if (ch && !/[a-zA-Z0-9]/.test(ch)) setMaskChar(ch);
              }}
              maxLength={2}
              aria-label="Custom mask symbol"
              style={{ ...inputStyle, width: 44, textAlign: 'center', padding: '4px 6px' }}
            />
          </div>

          {/* Beep sound */}
          <div style={{ margin: '10px 0' }}>
            <span style={{ display: 'block', fontSize: 11, color: 'var(--text-secondary)', marginBottom: 6 }}>
              Censor sound (played over the muted word)
            </span>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 4, cursor: 'pointer' }}>
              <input
                type="radio"
                name="censor-sound"
                checked={sound === 'beep'}
                onChange={() => setSound('beep')}
              />
              Beep tone (1 kHz — the classic broadcast bleep)
            </label>
            <label style={{
              display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 6,
              cursor: data.has_custom_sound ? 'pointer' : 'not-allowed',
              color: data.has_custom_sound ? 'var(--text-primary)' : 'var(--text-muted)',
            }}>
              <input
                type="radio"
                name="censor-sound"
                disabled={!data.has_custom_sound}
                checked={sound === 'custom'}
                onChange={() => setSound('custom')}
              />
              Custom sound{data.custom_sound_name ? ` (${data.custom_sound_name})` : ' (upload one below)'}
            </label>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <input
                ref={fileRef}
                type="file"
                accept=".mp3,.wav,.m4a,.aac,.ogg,.flac,audio/*"
                onChange={(e) => uploadSound(e.target.files?.[0])}
                style={{ fontSize: 11, color: 'var(--text-muted)' }}
              />
              {uploading && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Uploading…</span>}
              {data.has_custom_sound && (
                <button
                  onClick={removeSound}
                  style={{
                    padding: '4px 10px', fontSize: 11, borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)', background: 'var(--bg-base)',
                    color: 'var(--accent-amber)', cursor: 'pointer',
                  }}
                >
                  Remove custom sound
                </button>
              )}
            </div>
          </div>

          <button
            onClick={() => save()}
            disabled={saving}
            style={{
              padding: '8px 16px', fontSize: 12, fontWeight: 600,
              borderRadius: 'var(--radius-sm)', border: 'none',
              background: 'var(--accent-cyan)', color: 'var(--bg-base)',
              cursor: saving ? 'wait' : 'pointer',
            }}
          >
            Save Censor Settings
          </button>
        </>
      )}
    </div>
  );
}
