import { useEffect, useState } from 'react';
import { showToast } from './Toast';

// Subtitle Polish Cloud Fallback — AI Providers card.
// One dropdown controls whether a failed LOCAL polish batch may retry via
// OpenRouter (which bills real money): "None" = strictly local, "Auto" =
// best efficient-tier model, or a pinned model from the curated shortlist.
// The choice persists via /api/settings/polish-fallback (user_settings.json).

const TIER_LABEL = { premium: 'Premium', efficient: 'Efficient', free: 'Free' };

export default function PolishFallbackCard({ isMobile = false }) {
  const [data, setData] = useState(null);
  const [choice, setChoice] = useState(null); // null until loaded
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let alive = true;
    fetch('/api/settings/polish-fallback')
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!alive || !d) return;
        setData(d);
        setChoice(d.choice);
      })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  const save = async (next) => {
    setSaving(true);
    try {
      const res = await fetch('/api/settings/polish-fallback', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ choice: next }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      setData(d);
      setChoice(d.choice);
      showToast(
        d.choice === 'none'
          ? 'Cloud polish fallback disabled — strictly local'
          : `Cloud polish fallback: ${d.choice === 'auto' ? `auto (${d.auto_resolves_to})` : d.choice}`,
        'success',
      );
    } catch {
      showToast('Failed to save polish fallback setting', 'error');
    } finally {
      setSaving(false);
    }
  };

  const dot = choice === 'none' ? 'var(--text-muted)'
    : data?.openrouter_key_set ? 'var(--accent-cyan)' : 'var(--accent-amber)';

  return (
    <div style={{
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: isMobile ? '12px' : '12px 16px',
      boxShadow: 'var(--shadow-sm)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: dot }} />
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Subtitle Polish Fallback</span>
        {saving && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Saving…</span>}
        <span style={{ fontSize: 10, color: choice === 'none' ? 'var(--text-muted)' : 'var(--accent-cyan)', fontWeight: 600 }}>
          {choice === 'none' ? 'Strictly local' : choice === 'auto' ? 'Auto (cloud)' : 'Pinned (cloud)'}
        </span>
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, margin: '0 0 8px' }}>
        When the local polish model fails a batch (timeout / overload), ClipAI can retry
        that batch via OpenRouter — <strong style={{ color: 'var(--text-secondary)' }}>this
        bills a small cloud cost</strong> on an otherwise-local job. Pick the model it may
        use, or <strong style={{ color: 'var(--text-secondary)' }}>None</strong> to never
        spend (failed batches then stay unpolished).
      </p>
      {choice === null ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading…</div>
      ) : (
        <select
          value={choice}
          disabled={saving}
          onChange={(e) => save(e.target.value)}
          style={{
            width: '100%', padding: '8px 12px', background: 'var(--bg-base)',
            border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            color: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)',
            cursor: 'pointer',
          }}
        >
          <option value="none">None — strictly local, never use a cloud model</option>
          <option value="auto">
            Auto — best efficient-tier model{data?.auto_resolves_to ? ` (${data.auto_resolves_to})` : ''}
          </option>
          {(data?.options || []).map((m) => (
            <option key={m.id} value={m.id} title={m.rationale}>
              {m.id}{m.tier ? ` — ${TIER_LABEL[m.tier] || m.tier}` : ''}
            </option>
          ))}
          {/* A previously saved model that fell off the shortlist stays selectable. */}
          {choice !== 'none' && choice !== 'auto'
            && !(data?.options || []).some((m) => m.id === choice) && (
              <option value={choice}>{choice} — (saved)</option>
          )}
        </select>
      )}
      {choice !== 'none' && data && !data.openrouter_key_set && (
        <div style={{
          marginTop: 8, padding: '6px 10px', borderRadius: 'var(--radius-sm)',
          fontSize: 11, background: 'var(--amber-dim)', color: 'var(--accent-amber)',
        }}>
          No OpenRouter API key is saved — the fallback can't actually run until one is
          added above, so behavior is currently strictly local either way.
        </div>
      )}
    </div>
  );
}
