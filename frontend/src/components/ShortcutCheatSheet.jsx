/**
 * ShortcutCheatSheet (?) — generated from the same action registry the
 * key handler and command palette consume, so it can't go stale.
 */
import React, { useEffect, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { EDITOR_ACTIONS } from '../utils/editorActions';
import { CloseIcon } from './icons';

export default function ShortcutCheatSheet({ open, onClose }) {
  const dialogRef = useRef(null);

  // Focus management: move focus into the dialog on open, trap Tab
  // inside it, and restore focus to the opener on close.
  useEffect(() => {
    if (!open) return undefined;
    const previouslyFocused = document.activeElement;
    dialogRef.current?.focus();
    return () => {
      if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
        previouslyFocused.focus();
      }
    };
  }, [open]);

  const trapTab = (e) => {
    if (e.key !== 'Tab') return;
    const focusables = dialogRef.current?.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    if (!focusables || focusables.length === 0) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

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
      <div
        ref={dialogRef}
        className="ve-cheatsheet"
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        tabIndex={-1}
        onKeyDown={trapTab}
      >
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
