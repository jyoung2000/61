/**
 * ShortcutCheatSheet (?) — generated from the same action registry the
 * key handler and command palette consume, so it can't go stale.
 */
import React, { useMemo } from 'react';
import { createPortal } from 'react-dom';
import { EDITOR_ACTIONS } from '../utils/editorActions';
import { CloseIcon } from './icons';

export default function ShortcutCheatSheet({ open, onClose }) {
  const groups = useMemo(() => {
    const byCategory = new Map();
    for (const a of EDITOR_ACTIONS) {
      if (a.hidden) continue;
      if (!byCategory.has(a.category)) byCategory.set(a.category, []);
      byCategory.get(a.category).push(a);
    }
    return [...byCategory.entries()];
  }, []);

  if (!open) return null;

  return createPortal(
    <div
      className="ve-palette-backdrop"
      onPointerDown={(e) => { if (e.target === e.currentTarget) onClose?.(); }}
      onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); onClose?.(); } }}
    >
      <div className="ve-cheatsheet" role="dialog" aria-label="Keyboard shortcuts" tabIndex={-1}>
        <div className="ve-cheatsheet__header">
          <span className="ve-cheatsheet__title">Keyboard shortcuts</span>
          <button className="ve-cheatsheet__close" onClick={onClose} aria-label="Close shortcuts"><CloseIcon /></button>
        </div>
        <div className="ve-cheatsheet__grid">
          {groups.map(([category, actions]) => (
            <div key={category} className="ve-cheatsheet__group">
              <div className="ve-cheatsheet__category">{category}</div>
              {actions.map((a) => (
                <div key={a.id} className="ve-cheatsheet__row">
                  <span className="ve-cheatsheet__label">{a.label}</span>
                  <kbd className="ve-cheatsheet__kbd">{a.kbd}</kbd>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
