import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DndContext, PointerSensor, KeyboardSensor, TouchSensor,
  closestCenter, useSensor, useSensors,
} from '@dnd-kit/core';
import {
  SortableContext, arrayMove, sortableKeyboardCoordinates,
  useSortable, verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { showToast } from './Toast';

// Ollama Hosts — the multi-host registry card (AI Providers → Ollama).
// Row order IS priority: the top host is the primary, everything below is
// an ordered fallback with automatic failover. Reordering persists
// immediately on drop via PUT /api/settings/ollama-hosts. All probing is
// server-side (the browser can't reach LAN hosts across CORS/mixed-content).

const rowBtn = {
  padding: '4px 8px', background: 'var(--bg-elevated)', color: 'var(--text-secondary)',
  border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', fontSize: 11,
  whiteSpace: 'nowrap', cursor: 'pointer',
};

const inputStyle = {
  padding: '7px 10px', background: 'var(--bg-base)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontSize: 12,
  fontFamily: 'var(--font-mono)', boxSizing: 'border-box', width: '100%',
};

function StatusDot({ host }) {
  const color = !host.enabled ? 'var(--text-muted)'
    : host.online ? 'var(--success)'
    : host.in_cooldown ? 'var(--accent-amber)'
    : 'var(--danger)';
  const label = !host.enabled ? 'Disabled'
    : host.online ? `Online${host.latency_ms != null ? ` — ${host.latency_ms} ms` : ''}`
    : host.in_cooldown ? 'Cooling down after a failure'
    : `Offline${host.error ? ` — ${host.error}` : ''}`;
  return (
    <span
      title={label}
      aria-label={label}
      style={{ width: 8, height: 8, borderRadius: '50%', flexShrink: 0, background: color, display: 'inline-block' }}
    />
  );
}

function SortableHostRow({ host, index, isMobile, onToggle, onEdit, onDelete }) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: host.id });
  const modelsTip = host.online
    ? (host.models?.length ? `Models: ${host.models.join(', ')}` : 'Online — no models installed yet')
    : (host.error || 'Offline');
  return (
    <div
      ref={setNodeRef}
      style={{
        transform: CSS.Transform.toString(transform), transition,
        display: 'flex', alignItems: 'center', gap: 8, flexWrap: isMobile ? 'wrap' : 'nowrap',
        padding: '6px 8px', borderRadius: 'var(--radius-sm)',
        background: isDragging ? 'var(--bg-elevated)' : 'var(--bg-base)',
        border: '1px solid var(--border)', opacity: host.enabled ? 1 : 0.55,
        boxShadow: isDragging ? 'var(--shadow-md)' : 'none',
        position: 'relative', zIndex: isDragging ? 2 : 1,
      }}
    >
      <button
        {...attributes}
        {...listeners}
        aria-label={`Reorder ${host.name} (currently ${index === 0 ? 'primary' : `fallback #${index}`})`}
        title="Drag to reorder — top host is the primary"
        style={{
          // ≥44px touch target for mobile drag.
          minWidth: 44, minHeight: 44, display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'transparent', border: 'none', color: 'var(--text-muted)',
          cursor: isDragging ? 'grabbing' : 'grab', fontSize: 16, touchAction: 'none',
          margin: '-6px 0 -6px -8px',
        }}
      >
        ⠿
      </button>
      <StatusDot host={host} />
      <div style={{ flex: 1, minWidth: 140 }} title={modelsTip}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>{host.name}</span>
          {index === 0 ? (
            <span style={{
              fontSize: 9, fontWeight: 700, letterSpacing: 0.5, textTransform: 'uppercase',
              color: 'var(--success)', background: 'var(--success-dim)',
              padding: '1px 6px', borderRadius: 8,
            }}>Primary</span>
          ) : (
            <span style={{
              fontSize: 9, fontWeight: 600, letterSpacing: 0.5, textTransform: 'uppercase',
              color: 'var(--text-muted)', background: 'var(--bg-elevated)',
              padding: '1px 6px', borderRadius: 8,
            }}>Fallback #{index}</span>
          )}
          {host.has_token && <span title="Bearer token configured" style={{ fontSize: 10 }}>🔒</span>}
        </div>
        <div style={{
          fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: isMobile ? 200 : 320,
        }}>
          {host.url}
          {host.online && host.models?.length > 0 && ` — ${host.models.length} model${host.models.length === 1 ? '' : 's'}`}
        </div>
      </div>
      <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-secondary)', cursor: 'pointer' }}>
        <input
          type="checkbox"
          checked={host.enabled}
          onChange={(e) => onToggle(host.id, e.target.checked)}
          style={{ accentColor: 'var(--accent)' }}
        />
        {host.enabled ? 'On' : 'Off'}
      </label>
      <button style={rowBtn} onClick={() => onEdit(host)}>Edit</button>
      <button
        style={{ ...rowBtn, color: 'var(--danger)' }}
        onClick={() => onDelete(host)}
        aria-label={`Delete host ${host.name}`}
      >
        Delete
      </button>
    </div>
  );
}

export default function OllamaHostsCard({ isMobile = false }) {
  const [hosts, setHosts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  // Add/Edit form: null = closed, {} = adding, {id} = editing.
  const [form, setForm] = useState(null);
  const [testResult, setTestResult] = useState(null);
  const [testing, setTesting] = useState(false);
  const mounted = useRef(true);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 8 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  const load = useCallback(async () => {
    try {
      const res = await fetch('/api/settings/ollama-hosts');
      if (res.ok) {
        const data = await res.json();
        if (mounted.current) setHosts(data.hosts || []);
      }
    } catch { /* status refresh is best-effort */ }
    if (mounted.current) setLoading(false);
  }, []);

  useEffect(() => {
    mounted.current = true;
    load();
    const t = setInterval(load, 20000);
    return () => { mounted.current = false; clearInterval(t); };
  }, [load]);

  // Persist the given ordered list. Tokens are not held client-side for
  // existing hosts — token: null tells the backend "keep the stored one".
  const persist = useCallback(async (nextHosts, changed = null) => {
    setSaving(true);
    try {
      const res = await fetch('/api/settings/ollama-hosts', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hosts: nextHosts.map((h) => ({
            id: h.id, name: h.name, url: h.url, enabled: h.enabled,
            token: (changed && changed.id === h.id) ? changed.token : null,
          })),
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (mounted.current) setHosts(data.hosts || []);
      return true;
    } catch (e) {
      showToast('Failed to save Ollama hosts', 'error');
      load();
      return false;
    } finally {
      if (mounted.current) setSaving(false);
    }
  }, [load]);

  const onDragEnd = useCallback((event) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    setHosts((prev) => {
      const oldIdx = prev.findIndex((h) => h.id === active.id);
      const newIdx = prev.findIndex((h) => h.id === over.id);
      if (oldIdx < 0 || newIdx < 0) return prev;
      const next = arrayMove(prev, oldIdx, newIdx);
      persist(next);
      if (newIdx === 0) showToast(`${next[0].name} is now the primary host`, 'success');
      return next;
    });
  }, [persist]);

  const onToggle = (id, enabled) => {
    const next = hosts.map((h) => (h.id === id ? { ...h, enabled } : h));
    setHosts(next);
    persist(next);
  };

  const onDelete = (host) => {
    if (!window.confirm(`Remove Ollama host "${host.name}"?`)) return;
    const next = hosts.filter((h) => h.id !== host.id);
    setHosts(next);
    persist(next);
  };

  const openAdd = () => { setTestResult(null); setForm({ id: '', name: '', url: '', token: '' }); };
  const openEdit = (host) => {
    setTestResult(null);
    // token stays blank — blank means "keep stored token" unless the user
    // types a replacement (or checks "clear").
    setForm({ id: host.id, name: host.name, url: host.url, token: '', clearToken: false, hasToken: host.has_token });
  };

  const testHost = async () => {
    if (!form?.url?.trim()) return;
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch('/api/settings/ollama-hosts/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: form.url.trim(), token: form.token || '' }),
      });
      const data = await res.json();
      setTestResult(data);
    } catch {
      setTestResult({ online: false, error: 'Request failed — is the backend reachable?' });
    } finally {
      setTesting(false);
    }
  };

  const submitForm = async () => {
    const url = (form.url || '').trim();
    if (!url) { showToast('Host URL is required', 'error'); return; }
    let token = null; // keep stored
    if (form.clearToken) token = '';
    else if ((form.token || '').trim()) token = form.token.trim();
    const entry = {
      id: form.id || undefined,
      name: (form.name || '').trim() || url,
      url,
      enabled: true,
      token,
    };
    let next;
    if (form.id) {
      next = hosts.map((h) => (h.id === form.id ? { ...h, ...entry } : h));
    } else {
      entry.id = `h${Date.now().toString(36)}`;
      next = [...hosts, { ...entry, has_token: !!token }];
    }
    const changed = { id: entry.id, token };
    if (await persist(next, token === null ? null : changed)) {
      showToast(form.id ? 'Host updated' : 'Host added', 'success');
      setForm(null);
      setTestResult(null);
    }
  };

  return (
    <div style={{
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: isMobile ? '12px' : '12px 16px',
      boxShadow: 'var(--shadow-sm)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>Ollama Hosts</span>
        {saving && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Saving…</span>}
        <button style={rowBtn} onClick={load}>Refresh</button>
        <button
          onClick={openAdd}
          style={{
            padding: '4px 10px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
            border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
          }}
        >
          + Add host
        </button>
      </div>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4, margin: '0 0 10px' }}>
        Drag to reorder — the <strong style={{ color: 'var(--text-secondary)' }}>top host is the primary</strong>;
        hosts below are tried in order when it fails (automatic failover, no restart needed).
        Add a desktop GPU here via the GPU Companion, or any Ollama server on your LAN.
      </p>

      {loading ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Loading hosts…</div>
      ) : hosts.length === 0 ? (
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          No hosts configured — ClipAI uses the built-in default from <code>OLLAMA_HOST</code>.
          Add a host to enable multi-GPU failover.
        </div>
      ) : (
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
          <SortableContext items={hosts.map((h) => h.id)} strategy={verticalListSortingStrategy}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {hosts.map((host, index) => (
                <SortableHostRow
                  key={host.id}
                  host={host}
                  index={index}
                  isMobile={isMobile}
                  onToggle={onToggle}
                  onEdit={openEdit}
                  onDelete={onDelete}
                />
              ))}
            </div>
          </SortableContext>
        </DndContext>
      )}

      {form && (
        <div style={{
          marginTop: 10, padding: 10, border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)', background: 'var(--bg-base)',
          display: 'flex', flexDirection: 'column', gap: 8,
        }}>
          <div style={{ fontSize: 12, fontWeight: 600 }}>{form.id ? 'Edit host' : 'Add host'}</div>
          <input
            placeholder="Name (e.g. Desktop 4070)"
            value={form.name}
            onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
            style={inputStyle}
          />
          <input
            placeholder="URL (e.g. http://192.168.1.50:11500/ollama)"
            value={form.url}
            onChange={(e) => setForm((f) => ({ ...f, url: e.target.value }))}
            style={inputStyle}
          />
          <input
            type="password"
            placeholder={form.hasToken && !form.clearToken
              ? 'Token saved — type to replace'
              : 'Bearer token (optional — required for GPU Companion)'}
            value={form.token}
            disabled={form.clearToken}
            onChange={(e) => setForm((f) => ({ ...f, token: e.target.value }))}
            style={{ ...inputStyle, opacity: form.clearToken ? 0.5 : 1 }}
          />
          {form.hasToken && (
            <label style={{ fontSize: 10, color: 'var(--text-muted)', display: 'flex', gap: 6, alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={!!form.clearToken}
                onChange={(e) => setForm((f) => ({ ...f, clearToken: e.target.checked, token: '' }))}
                style={{ accentColor: 'var(--accent)' }}
              />
              Remove the saved token
            </label>
          )}
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            <button
              onClick={testHost}
              disabled={testing || !(form.url || '').trim()}
              style={{ ...rowBtn, opacity: testing || !(form.url || '').trim() ? 0.5 : 1 }}
            >
              {testing ? 'Testing…' : 'Test'}
            </button>
            <button
              onClick={submitForm}
              disabled={saving}
              style={{
                padding: '4px 12px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
                border: 'none', borderRadius: 'var(--radius-sm)', fontSize: 11, fontWeight: 600,
              }}
            >
              {form.id ? 'Save' : 'Add'}
            </button>
            <button style={rowBtn} onClick={() => { setForm(null); setTestResult(null); }}>Cancel</button>
          </div>
          {testResult && (
            <div style={{
              padding: '6px 10px', borderRadius: 'var(--radius-sm)', fontSize: 11, lineHeight: 1.5,
              background: testResult.online ? 'var(--success-dim)' : 'var(--danger-dim)',
              color: testResult.online ? 'var(--success)' : 'var(--danger)',
            }}>
              {testResult.online ? (
                <>
                  Online{testResult.version ? ` — Ollama ${testResult.version}` : ''}
                  {testResult.latency_ms != null ? ` (${testResult.latency_ms} ms)` : ''}
                  {testResult.models?.length
                    ? `. Models: ${testResult.models.slice(0, 8).join(', ')}${testResult.models.length > 8 ? '…' : ''}`
                    : '. No models installed yet.'}
                </>
              ) : (
                <>Unreachable{testResult.error ? ` — ${testResult.error}` : ''}</>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
