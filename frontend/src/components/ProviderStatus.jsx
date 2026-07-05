import React, { useState, useEffect } from 'react';

const STATUS_COLORS = {
  connected: 'var(--success)',
  configured: 'var(--accent-cyan)',
  loading: 'var(--accent-amber)',
  offline: 'var(--danger)',
  not_configured: 'var(--text-muted)',
};

// Shorten model ID for display: "meta-llama/llama-4-scout:free" → "llama-4-scout"
function shortModel(modelId) {
  if (!modelId) return '';
  let name = modelId;
  // Remove provider prefix
  if (name.includes('/')) name = name.split('/').pop();
  // Remove :free suffix
  name = name.replace(/:free$/, '');
  return name;
}

export default function ProviderStatus({ collapsed, onActiveChange }) {
  const [statuses, setStatuses] = useState({});

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/providers/status');
        if (res.ok) {
          const data = await res.json();
          setStatuses(data);
          if (onActiveChange && data._active) {
            onActiveChange(data._active);
          }
        }
      } catch {
        // silent
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const providers = ['ollama', 'openrouter', 'anthropic', 'gemini', 'groq'];
  const active = statuses._active || {};

  return (
    <div
      style={{
        padding: collapsed ? '8px' : '12px 16px',
        borderTop: '1px solid var(--border)',
      }}
    >
      {collapsed ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          {/* Active model indicator */}
          {active.provider && active.provider !== 'none' && (
            <div
              title={`T: ${active.transcript_model || 'whisper-small'} | P: ${active.replicate_available ? shortModel(active.replicate_model || 'videollama3-7b') : active.videollama2_available ? 'VideoLLaMA2' : shortModel(active.primary_model || active.vision_model)} | E: ${shortModel(active.editorial_model || active.text_model)}${active.companion ? ` | GPU: ${active.companion.gpu_name || active.companion.name || 'Companion'} (${active.companion.online ? `${active.companion.models_ready}/${active.companion.models_total} ${active.companion.ready ? 'ready' : 'synced'}` : 'offline'})` : active.ollama_host_name ? ` | GPU: ${active.ollama_host_name}` : ''}`}
              style={{
                width: 20,
                height: 20,
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-cyan)',
                color: 'var(--bg-base)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 9,
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                marginBottom: 4,
              }}
            >
              {active.provider.charAt(0).toUpperCase()}
            </div>
          )}
          {providers.map((name) => {
            const status = statuses[name]?.status || 'not_configured';
            return (
              <div
                key={name}
                title={`${name}: ${status}`}
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: STATUS_COLORS[status] || STATUS_COLORS.not_configured,
                  animation: status === 'connected' ? 'pulse 2s infinite' : undefined,
                }}
              />
            );
          })}
        </div>
      ) : (
        <div>
          {/* Active Models Section */}
          {active.provider && active.provider !== 'none' && (
            <div style={{ marginBottom: 12 }}>
              <div
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-muted)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                  marginBottom: 6,
                }}
              >
                Active Models
              </div>
              <div
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 10px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 4,
                }}
              >
                {/* Transcript AI */}
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                  <span style={{ color: 'var(--accent-amber)', fontWeight: 600 }}>transcript:</span>{' '}
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {active.transcript_model || 'whisper-small'}
                  </span>
                </div>
                {/* Primary AI (Replicate / VideoLLaMA2 / Ollama / cloud) */}
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                  <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>primary:</span>{' '}
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {active.replicate_available
                      ? `${shortModel(active.replicate_model || 'videollama3-7b')} (Replicate)`
                      : active.videollama2_available
                        ? 'VideoLLaMA2'
                        : ((active.primary_model || active.vision_model)
                            ? shortModel(active.primary_model || active.vision_model)
                            : 'signal-only')}
                  </span>
                </div>
                {/* Editorial AI (scoring, summary, polish) */}
                {(active.editorial_model || active.text_model || active.primary_model) && (
                  <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', lineHeight: 1.5 }}>
                    <span style={{ color: 'var(--success)', fontWeight: 600 }}>editorial:</span>{' '}
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {shortModel(active.editorial_model || active.text_model || active.primary_model)}
                    </span>
                  </div>
                )}
                {/* GPU / host indicator — which GPU serves these models, and
                    (when a Companion is paired) whether all selected models
                    are downloaded to it and ready. */}
                {active.provider === 'ollama' && (active.companion || active.ollama_host_name) && (
                  <div style={{
                    fontSize: 9, fontFamily: 'var(--font-mono)', lineHeight: 1.5,
                    marginTop: 2, paddingTop: 4, borderTop: '1px solid var(--border)',
                    display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap',
                  }}>
                    <span style={{ color: 'var(--text-muted)' }}>gpu:</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {active.companion?.gpu_name
                        ? `${active.companion.gpu_name}${active.companion.name ? ` · ${active.companion.name}` : ''}`
                        : (active.ollama_host_name || 'local')}
                    </span>
                    {active.companion && (
                      <span style={{
                        marginLeft: 'auto', fontWeight: 600,
                        color: active.companion.ready ? 'var(--success)'
                          : active.companion.online ? 'var(--accent-amber)' : 'var(--danger)',
                      }}
                        title={active.companion.missing?.length
                          ? `Not yet on the Companion: ${active.companion.missing.join(', ')}`
                          : 'All selected models are downloaded to the Companion'}>
                        {!active.companion.online
                          ? '● companion offline'
                          : active.companion.ready
                            ? `● ${active.companion.models_ready}/${active.companion.models_total} ready`
                            : `● ${active.companion.models_ready}/${active.companion.models_total} synced`}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Provider list */}
          <div
            style={{
              fontSize: 10,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: 8,
            }}
          >
            Providers
          </div>
          {providers.map((name) => {
            const info = statuses[name] || {};
            const status = info.status || 'not_configured';
            const isActive = name === active.provider;
            return (
              <div
                key={name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '4px 0',
                  fontSize: 12,
                }}
              >
                <div
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: STATUS_COLORS[status] || STATUS_COLORS.not_configured,
                    flexShrink: 0,
                    animation: status === 'connected' ? 'pulse 2s infinite' : undefined,
                  }}
                />
                <span style={{
                  color: isActive ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  textTransform: 'capitalize',
                  fontWeight: isActive ? 600 : 400,
                }}>
                  {name}
                </span>
                {isActive && (
                  <span style={{ fontSize: 9, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
                    ACTIVE
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
