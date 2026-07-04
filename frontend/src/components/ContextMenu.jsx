/**
 * ContextMenu — the editor's shared right-click menu.
 *
 * One component for timeline clips, track headers, and preview overlay
 * items (and, on touch, long-press action sheets) so behavior can't
 * drift per surface:
 *   • portal to <body>, glass --ve-dropdown-bg styling
 *   • position clamped to the viewport
 *   • keyboard navigable: ↑/↓ cycle, Enter/Space select, Esc closes
 *   • outside click / second contextmenu / scroll closes
 *   • `sheet` prop renders a bottom action sheet (mobile long-press)
 *
 * items: array of
 *   { id, label, kbd?, danger?, disabled?, checked?, onSelect }
 *   { separator: true }
 *   { heading: 'Section' }
 */
import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export default function ContextMenu({ x = 0, y = 0, items = [], onClose, sheet = false }) {
  const menuRef = useRef(null);
  const [pos, setPos] = useState({ left: x, top: y });
  const [focusIdx, setFocusIdx] = useState(-1);

  const actionable = items
    .map((it, i) => ({ it, i }))
    .filter(({ it }) => !it.separator && !it.heading && !it.disabled);

  // Clamp to viewport once we know the rendered size
  useLayoutEffect(() => {
    if (sheet) return;
    const el = menuRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    setPos({
      left: Math.max(4, Math.min(x, vw - r.width - 4)),
      top: Math.max(4, Math.min(y, vh - r.height - 4)),
    });
  }, [x, y, sheet, items.length]);

  useEffect(() => {
    menuRef.current?.focus();
  }, []);

  useEffect(() => {
    const close = (e) => {
      if (menuRef.current && menuRef.current.contains(e.target)) return;
      onClose?.();
    };
    // Capture phase so a click that opens something else still closes us
    window.addEventListener('pointerdown', close, true);
    window.addEventListener('contextmenu', close, true);
    window.addEventListener('blur', close);
    window.addEventListener('resize', close);
    return () => {
      window.removeEventListener('pointerdown', close, true);
      window.removeEventListener('contextmenu', close, true);
      window.removeEventListener('blur', close);
      window.removeEventListener('resize', close);
    };
  }, [onClose]);

  const onKeyDown = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); onClose?.(); return; }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (actionable.length === 0) return;
      const cur = actionable.findIndex(({ i }) => i === focusIdx);
      const next = e.key === 'ArrowDown'
        ? actionable[(cur + 1) % actionable.length]
        : actionable[(cur - 1 + actionable.length) % actionable.length];
      setFocusIdx(next.i);
      return;
    }
    if ((e.key === 'Enter' || e.key === ' ') && focusIdx >= 0) {
      e.preventDefault();
      const entry = items[focusIdx];
      if (entry && !entry.disabled) {
        entry.onSelect?.();
        onClose?.();
      }
    }
  };

  const select = (entry) => {
    if (entry.disabled) return;
    entry.onSelect?.();
    onClose?.();
  };

  const menu = (
    <div
      ref={menuRef}
      role="menu"
      tabIndex={-1}
      className={`ve-context-menu${sheet ? ' ve-context-menu--sheet' : ''}`}
      style={sheet ? undefined : { left: pos.left, top: pos.top }}
      onKeyDown={onKeyDown}
      onContextMenu={(e) => e.preventDefault()}
    >
      {sheet && <div className="ve-context-menu__grabber" aria-hidden="true" />}
      {items.map((entry, i) => {
        if (entry.separator) {
          return <div key={`sep-${i}`} className="ve-context-menu__separator" role="separator" />;
        }
        if (entry.heading) {
          return <div key={`h-${i}`} className="ve-context-menu__heading">{entry.heading}</div>;
        }
        return (
          <button
            key={entry.id || entry.label}
            role="menuitem"
            disabled={entry.disabled}
            className={[
              've-context-menu__item',
              entry.danger ? 've-context-menu__item--danger' : '',
              i === focusIdx ? 've-context-menu__item--focused' : '',
            ].filter(Boolean).join(' ')}
            onPointerEnter={() => setFocusIdx(i)}
            onClick={() => select(entry)}
          >
            <span className="ve-context-menu__check" aria-hidden="true">
              {entry.checked ? '✓' : ''}
            </span>
            <span className="ve-context-menu__label">{entry.label}</span>
            {entry.kbd && <kbd className="ve-context-menu__kbd">{entry.kbd}</kbd>}
          </button>
        );
      })}
    </div>
  );

  return createPortal(menu, document.body);
}
