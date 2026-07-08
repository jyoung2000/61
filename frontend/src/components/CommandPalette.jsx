/**
 * CommandPalette (⌘K) — fuzzy search over the editor action registry.
 * Runs the selected action with the same context the keyboard shortcuts
 * use, so palette and keys can never disagree.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { EDITOR_ACTIONS, fuzzyScore } from '../utils/editorActions';
import { formatKbd } from '../utils/platform';

export default function CommandPalette({ open, onClose, ctx }) {
  const [query, setQuery] = useState('');
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setQuery('');
      setCursor(0);
      // Focus after the portal paints
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  const results = useMemo(() => {
    const visible = EDITOR_ACTIONS.filter((a) => !a.hidden);
    if (!query.trim()) return visible;
    return visible
      .map((a) => ({ a, score: fuzzyScore(query, `${a.label} ${a.category}`) }))
      .filter((r) => r.score >= 0)
      .sort((x, y) => y.score - x.score)
      .map((r) => r.a);
  }, [query]);

  if (!open) return null;

  const run = (action) => {
    onClose?.();
    // Run after closing so actions that grab focus (dialogs) win
    setTimeout(() => action.run(ctx || {}, null), 0);
  };

  const onKeyDown = (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') onClose?.();
    else if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.min(c + 1, results.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === 'Enter' && results[cursor]) run(results[cursor]);
  };

  return createPortal(
    <div className="ve-palette-backdrop" onPointerDown={(e) => { if (e.target === e.currentTarget) onClose?.(); }}>
      <div className="ve-palette" role="dialog" aria-label="Command palette">
        <input
          ref={inputRef}
          className="ve-palette__input"
          placeholder="Type a command…"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setCursor(0); }}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded={results.length > 0}
          aria-controls="ve-palette-listbox"
          aria-activedescendant={results[cursor] ? `ve-palette-opt-${results[cursor].id}` : undefined}
          aria-autocomplete="list"
        />
        {/* Result count for screen readers — the visual list is aria-hidden
            from announcement duplication by virtue of the options below */}
        <div className="ve-visually-hidden" aria-live="polite">
          {results.length === 0 ? 'No matching commands'
            : `${results.length} command${results.length === 1 ? '' : 's'} available`}
        </div>
        <div className="ve-palette__list" role="listbox" id="ve-palette-listbox">
          {results.length === 0 && (
            <div className="ve-palette__empty">No matching commands</div>
          )}
          {results.map((a, i) => (
            <button
              key={a.id}
              id={`ve-palette-opt-${a.id}`}
              role="option"
              aria-selected={i === cursor}
              tabIndex={-1}
              className={`ve-palette__item${i === cursor ? ' ve-palette__item--active' : ''}`}
              onPointerEnter={() => setCursor(i)}
              onClick={() => run(a)}
            >
              <span className="ve-palette__category">{a.category}</span>
              <span className="ve-palette__label">{a.label}</span>
              {a.kbd && <kbd className="ve-palette__kbd">{formatKbd(a.kbd)}</kbd>}
            </button>
          ))}
        </div>
      </div>
    </div>,
    document.body,
  );
}
