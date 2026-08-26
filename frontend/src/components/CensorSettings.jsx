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
  const [beepVolume, setBeepVolume] = useState(100); // percent; 100 = baseline
  const [separateTrack, setSeparateTrack] = useState(false);
  const [saving, setSaving] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const fileRef = useRef(null);

  const applyState = (d) => {
    setData(d);
    setWordsText((d.words || []).join('\n'));
    setMaskChar(d.mask_char || '*');
    setEnabledDefault(!!d.enabled_default);
    setSound(d.sound || 'beep');
    setBeepVolume(Math.round((d.beep_volume ?? 1.0) * 100));
    setSeparateTrack(!!d.separate_track);
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
          beep_volume: beepVolume / 100,
          separate_track: separateTrack,
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

  // Play the currently SELECTED sound at the currently SET volume — both
  // read from the unsaved UI state, so the user can dial it in by ear
  // before hitting Save. The default tone is synthesized locally (same
  // 1 kHz sine × 0.5 baseline the export uses); the custom file streams
  // from /api/censor/sound. WebAudio gain allows >100% unlike <audio>.
  const previewSound = async () => {
    if (previewing) return;
    setPreviewing(true);
    const done = () => setPreviewing(false);
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      const gainNode = ctx.createGain();
      gainNode.connect(ctx.destination);
      const v = beepVolume / 100;
      const cleanup = () => { ctx.close().catch(() => {}); done(); };
      if (sound === 'custom' && data?.has_custom_sound) {
        const res = await fetch(`/api/censor/sound?t=${Date.now()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await ctx.decodeAudioData(await res.arrayBuffer());
        const src = ctx.createBufferSource();
        src.buffer = buf;
        gainNode.gain.value = v;
        src.connect(gainNode);
        src.onended = cleanup;
        src.start();
        // Safety stop for long files — a preview is a taste, not a concert.
        src.stop(ctx.currentTime + Math.min(buf.duration, 3));
      } else {
        const osc = ctx.createOscillator();
        osc.type = 'sine';
        osc.frequency.value = 1000;
        gainNode.gain.value = 0.5 * v; // export baseline for the tone
        osc.connect(gainNode);
        osc.onended = cleanup;
        osc.start();
        osc.stop(ctx.currentTime + 0.6);
      }
    } catch {
      showToast('Preview failed — check the sound file', 'error');
      done();
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
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1 }}>
                Censor sound (played over the muted word)
              </span>
              <button
                onClick={previewSound}
                disabled={previewing}
                aria-label="Preview censor sound"
                style={{
                  padding: '4px 12px', fontSize: 11, borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--accent-cyan)', background: 'var(--bg-base)',
                  color: 'var(--accent-cyan)', cursor: previewing ? 'wait' : 'pointer',
                  fontWeight: 600,
                }}
              >
                {previewing ? 'Playing…' : '▶ Preview'}
              </button>
            </div>
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

          {/* Universal loudness for the censor sound */}
          <div style={{ margin: '10px 0' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', flex: 1 }}>
                Beep volume — applies to the tone and custom sounds alike
              </span>
              <span style={{
                fontSize: 11, fontFamily: 'var(--font-mono)', minWidth: 42,
                textAlign: 'right',
                color: beepVolume === 100 ? 'var(--text-muted)' : 'var(--accent-cyan)',
              }}>
                {beepVolume}%
              </span>
            </div>
            <input
              type="range"
              min={10}
              max={300}
              step={5}
              value={beepVolume}
              onChange={(e) => setBeepVolume(parseInt(e.target.value, 10))}
              aria-label="Beep volume percent"
              style={{ width: '100%' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--text-muted)' }}>
              <span>quieter</span>
              <span>100% = default</span>
              <span>louder</span>
            </div>
          </div>

          {/* Separate beep track */}
          <label style={{
            display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: 12,
            color: 'var(--text-primary)', margin: '10px 0', cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={separateTrack}
              onChange={(e) => setSeparateTrack(e.target.checked)}
              style={{ marginTop: 2 }}
            />
            <span>
              Also put the beeps on a <strong>separate audio track</strong> ("Censor beeps")
              <span style={{ display: 'block', fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                Track 1 stays the normal censored mix, so players sound identical — the extra
                track lets an editor grab or drop the beeps on their own.
              </span>
            </span>
          </label>

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
