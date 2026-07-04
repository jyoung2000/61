/**
 * ScrubInput + InspectorRow — the editor's pro numeric-input ergonomics
 * (After Effects style), shared by every inspector panel so behavior
 * can't drift per panel:
 *
 *   • drag horizontally on the value to adjust (pointer-captured)
 *   • Shift = 10× step, Alt/Option = 0.1× step
 *   • plain click = type a value; Enter commits, Esc reverts
 *   • arrow keys adjust by step when the field has focus
 *   • drags are undo-coalesced (one undo step per gesture)
 *   • a subtle dot marks values that differ from their default; the
 *     label (or ↺) resets that one control
 *
 * Bounds MUST come from shared range tables (SUBTITLE_RANGES etc. in
 * utils/defaultSettings.js) — never inline constants (the exact drift
 * bug class fixed in parity task 1.1).
 */
import React, { useEffect, useRef, useState } from 'react';
import { undoCoalesceHandlers } from '../utils/undoCoalesce';

/** Decimal places implied by a step (0.05 → 2, 1 → 0). */
export function stepPrecision(step) {
  if (!Number.isFinite(step) || step >= 1 || step <= 0) return 0;
  const s = String(step);
  const dot = s.indexOf('.');
  return dot === -1 ? 0 : s.length - dot - 1;
}

/**
 * Pure scrub math — 1 horizontal pixel = 1 step (× modifier).
 * Exported for unit tests.
 */
export function scrubValue(startValue, dxPixels, { step = 1, min, max, shiftKey = false, altKey = false } = {}) {
  const mod = shiftKey ? 10 : altKey ? 0.1 : 1;
  let v = startValue + dxPixels * step * mod;
  const prec = stepPrecision(step * mod);
  v = Number(v.toFixed(Math.min(6, prec + 1)));
  if (min != null && v < min) v = min;
  if (max != null && v > max) v = max;
  return v;
}

const DRAG_THRESHOLD_PX = 3;

export default function ScrubInput({
  value,
  min,
  max,
  step = 1,
  unit = '',
  defaultValue,
  onChange,
  ariaLabel,
  className = '',
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const gestureRef = useRef(null); // {startX, startValue, moved}
  const undoRef = useRef(undoCoalesceHandlers());

  const prec = stepPrecision(step);
  const display = Number.isFinite(value) ? Number(value).toFixed(prec) : '—';
  const changed = defaultValue != null && Number.isFinite(value)
    && Math.abs(value - defaultValue) > 1e-9;

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  const commitDraft = () => {
    const parsed = parseFloat(draft);
    if (Number.isFinite(parsed)) {
      onChange(scrubValue(parsed, 0, { step, min, max }));
    }
    setEditing(false);
  };

  const onPointerDown = (e) => {
    if (editing) return;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    gestureRef.current = { startX: e.clientX, startValue: value ?? 0, moved: false };
    undoRef.current.onPointerDown();
  };

  const onPointerMove = (e) => {
    const g = gestureRef.current;
    if (!g) return;
    const dx = e.clientX - g.startX;
    if (!g.moved && Math.abs(dx) < DRAG_THRESHOLD_PX) return;
    if (!g.moved) { g.moved = true; setDragging(true); }
    onChange(scrubValue(g.startValue, dx, {
      step, min, max, shiftKey: e.shiftKey, altKey: e.altKey,
    }));
  };

  const onPointerUp = () => {
    const g = gestureRef.current;
    gestureRef.current = null;
    undoRef.current.onPointerUp();
    setDragging(false);
    if (g && !g.moved) {
      // Plain click → type a value
      setDraft(display);
      setEditing(true);
    }
  };

  const onKeyDown = (e) => {
    if (editing) return;
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    const dir = e.key === 'ArrowUp' ? 1 : -1;
    onChange(scrubValue(value ?? 0, dir, {
      step, min, max, shiftKey: e.shiftKey, altKey: e.altKey,
    }));
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        type="text"
        inputMode="decimal"
        className={`ve-scrub ve-scrub--editing ${className}`}
        value={draft}
        aria-label={ariaLabel}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commitDraft}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commitDraft();
          else if (e.key === 'Escape') setEditing(false); // revert
          e.stopPropagation();
        }}
      />
    );
  }

  return (
    <span
      role="spinbutton"
      tabIndex={0}
      aria-label={ariaLabel}
      aria-valuenow={Number.isFinite(value) ? value : undefined}
      aria-valuemin={min}
      aria-valuemax={max}
      className={`ve-scrub${dragging ? ' ve-scrub--dragging' : ''}${changed ? ' ve-scrub--changed' : ''} ${className}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onKeyDown={onKeyDown}
    >
      {display}{unit && <span className="ve-scrub__unit">{unit}</span>}
    </span>
  );
}

/**
 * InspectorRow — label + control layout with the shared reset
 * affordance: a dot when `changed`, click the label (or ↺) to reset.
 */
export function InspectorRow({ label, changed = false, onReset, children }) {
  const canReset = changed && typeof onReset === 'function';
  return (
    <div className="ve-inspector-row">
      <button
        type="button"
        className={`ve-inspector-row__label${changed ? ' ve-inspector-row__label--changed' : ''}`}
        onClick={canReset ? onReset : undefined}
        disabled={!canReset}
        aria-label={canReset ? `Reset ${label} to default` : label}
      >
        <span className={`ve-inspector-row__dot${changed ? ' ve-inspector-row__dot--visible' : ''}`} aria-hidden="true" />
        {label}
        {canReset && <span className="ve-inspector-row__reset" aria-hidden="true">↺</span>}
      </button>
      <div className="ve-inspector-row__control">{children}</div>
    </div>
  );
}
