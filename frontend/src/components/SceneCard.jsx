import React, { useState } from 'react';
import { cropColorAt } from '../utils/cropColors';

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function SceneCard({ scene, sceneIndex, jobId, onClick, onUpdated }) {
  const [editing, setEditing] = useState(false);
  const [editDesc, setEditDesc] = useState('');
  const [editScore, setEditScore] = useState(5);
  const [saving, setSaving] = useState(false);

  const thumbnailUrl = scene.thumbnail_path
    ? `/api/files/${jobId}/frames/${scene.thumbnail_path.split('/').pop()}`
    : null;

  const startEdit = (e) => {
    e.stopPropagation();
    setEditDesc(scene.description);
    setEditScore(scene.importance_score);
    setEditing(true);
  };

  const cancelEdit = () => {
    setEditing(false);
  };

  const saveEdit = async () => {
    setSaving(true);
    try {
      const res = await fetch(`/api/jobs/${jobId}/scenes/${sceneIndex}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: editDesc, importance_score: editScore }),
      });
      if (res.ok) {
        setEditing(false);
        if (onUpdated) onUpdated();
      }
    } catch {
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete scene at ${formatTime(scene.timestamp)}?`)) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}/scenes/${sceneIndex}`, { method: 'DELETE' });
      if (res.ok && onUpdated) onUpdated();
    } catch {}
  };

  return (
    <div
      className="card-hover"
      onClick={() => !editing && onClick?.(scene.timestamp)}
      style={{
        background: 'var(--bg-panel)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-sm)',
        overflow: 'hidden',
        cursor: editing ? 'default' : 'pointer',
      }}
    >
      {/* Thumbnail */}
      <div
        style={{
          height: 120,
          background: 'var(--bg-elevated)',
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {thumbnailUrl && (
          <img
            src={thumbnailUrl}
            alt={`Scene at ${formatTime(scene.timestamp)}`}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        )}
        <span
          style={{
            position: 'absolute',
            bottom: 4,
            left: 4,
            background: 'var(--badge-overlay-bg)',
            color: 'var(--accent-cyan)',
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            padding: '2px 6px',
            borderRadius: 2,
          }}
        >
          {formatTime(scene.timestamp)}
        </span>
        {/* Subject count badge — reframer's face_positions per scene */}
        {((scene.face_count || 0) > 0 || (scene.face_positions?.length || 0) > 0) && (() => {
          const count = scene.face_count || scene.face_positions?.length || 0;
          const ids = (scene.face_positions || [])
            .map((f) => (f.identity_id != null ? f.identity_id : f.slot_id))
            .filter((v) => v != null);
          const speaking = (scene.face_positions || []).some((f) => f.is_speaking);
          const idStr = ids.length ? ` — #${ids.join(', #')}` : '';
          const tip = `${count} subject${count === 1 ? '' : 's'}${idStr}${speaking ? ' — someone speaking' : ''}`;
          return (
            <span
              style={{
                position: 'absolute',
                bottom: 4,
                right: 4,
                background: 'var(--badge-overlay-bg)',
                color: speaking ? 'var(--accent-amber)' : 'var(--accent-cyan)',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                padding: '2px 6px',
                borderRadius: 2,
              }}
              title={tip}
            >
              {count} subject{count === 1 ? '' : 's'}{speaking ? ' (speaking)' : ''}
            </span>
          );
        })()}
        {/* Edit/Delete buttons */}
        {onUpdated && !editing && (
          <div style={{ position: 'absolute', top: 4, right: 4, display: 'flex', gap: 4 }}>
            <button
              onClick={startEdit}
              title="Edit scene"
              style={{
                padding: '2px 6px',
                fontSize: 10,
                background: 'rgba(0,0,0,0.6)',
                color: 'var(--accent-cyan)',
                border: 'none',
                borderRadius: 3,
                cursor: 'pointer',
              }}
            >
              {'\u270E'}
            </button>
            <button
              onClick={handleDelete}
              title="Delete scene"
              style={{
                padding: '2px 6px',
                fontSize: 10,
                background: 'rgba(0,0,0,0.6)',
                color: 'var(--danger, #ff3b30)',
                border: 'none',
                borderRadius: 3,
                cursor: 'pointer',
              }}
            >
              {'\u2715'}
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div style={{ padding: 12 }}>
        {/* Importance score bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
          <div
            style={{
              flex: 1,
              height: 3,
              background: 'var(--bg-elevated)',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${scene.importance_score * 10}%`,
                background:
                  scene.importance_score >= 8
                    ? 'var(--accent-amber)'
                    : scene.importance_score >= 5
                      ? 'var(--accent-cyan)'
                      : 'var(--text-secondary)',
              }}
            />
          </div>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 11,
              color: 'var(--text-secondary)',
            }}
          >
            {Number(scene.importance_score) || 0}/10
          </span>
        </div>

        {editing ? (
          <div onClick={(e) => e.stopPropagation()}>
            <textarea
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              rows={3}
              style={{
                width: '100%',
                fontSize: 12,
                lineHeight: 1.4,
                color: 'var(--text-primary)',
                background: 'var(--bg-base)',
                border: '1px solid var(--accent-cyan)',
                borderRadius: 'var(--radius-sm)',
                padding: '4px 6px',
                resize: 'vertical',
                outline: 'none',
                fontFamily: 'inherit',
                marginBottom: 6,
              }}
            />
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Score:</span>
              <input
                type="range"
                min="1"
                max="10"
                value={editScore}
                onChange={(e) => setEditScore(parseInt(e.target.value))}
                style={{ width: 80, accentColor: 'var(--accent-cyan)' }}
              />
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                {editScore}/10
              </span>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                onClick={saveEdit}
                disabled={saving}
                style={{
                  padding: '3px 10px',
                  fontSize: 10,
                  fontWeight: 600,
                  background: 'var(--accent-cyan)',
                  color: 'var(--bg-base)',
                  border: 'none',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                }}
              >
                {saving ? '...' : 'Save'}
              </button>
              <button
                onClick={cancelEdit}
                style={{
                  padding: '3px 10px',
                  fontSize: 10,
                  background: 'none',
                  color: 'var(--text-muted)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {String(scene.description || '')}
            </p>
            {/* Detail strip — pulls every non-empty signal the
                reframer/perceiver dropped on the scene so the user can
                read it at a glance instead of digging into JSON. */}
            <div style={{
              marginTop: 8, display: 'flex', flexWrap: 'wrap',
              gap: 4, fontSize: 10, color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}>
              {(() => {
                const chips = [];
                if (scene.layout_mode && scene.layout_mode !== 'single') {
                  chips.push({ label: scene.layout_mode.replace(/_/g, ' '), color: 'var(--accent-cyan)' });
                }
                const fcount = scene.face_count || scene.face_positions?.length || 0;
                if (fcount > 0) {
                  chips.push({ label: `${fcount} face${fcount === 1 ? '' : 's'}`, color: 'var(--text-secondary)' });
                }
                const speaking = (scene.face_positions || []).some((f) => f.is_speaking);
                if (speaking) chips.push({ label: 'speaker active', color: 'var(--accent-amber)' });
                if (scene.has_screen_content) chips.push({ label: 'screen / slides', color: 'var(--accent-cyan)' });
                if (scene.primary_object_type) {
                  chips.push({ label: `obj: ${scene.primary_object_type}`, color: 'var(--text-secondary)' });
                }
                if (scene.fusion_source) {
                  chips.push({ label: scene.fusion_source.replace(/_/g, ' '), color: 'var(--text-muted)' });
                }
                if (typeof scene.subject_x === 'number') {
                  // Same crop%→hue as the timeline element + overview map, as a
                  // filled band (white label) so the chip reads as the same crop.
                  chips.push({
                    label: `crop ${Math.round(scene.subject_x)}%`,
                    color: '#fff',
                    bg: cropColorAt(scene.subject_x),
                  });
                }
                if (typeof scene.subject_confidence === 'number' && scene.subject_confidence > 0) {
                  chips.push({
                    label: `conf ${Math.round(scene.subject_confidence * 100)}%`,
                    color: scene.subject_confidence >= 0.7
                      ? 'var(--success)'
                      : scene.subject_confidence >= 0.4
                        ? 'var(--accent-amber)'
                        : 'var(--danger)',
                  });
                }
                if (scene.no_subject_reason) {
                  chips.push({ label: scene.no_subject_reason.replace(/_/g, ' '), color: 'var(--accent-amber)' });
                }
                return chips.map((c, i) => (
                  <span key={i} style={{
                    padding: '1px 6px',
                    borderRadius: 3,
                    background: c.bg || 'var(--badge-overlay-bg, rgba(127,127,127,0.12))',
                    color: c.color,
                    border: '1px solid var(--border)',
                  }}>{c.label}</span>
                ));
              })()}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
