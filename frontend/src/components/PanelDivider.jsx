/**
 * PanelDivider + usePanelSize — resizable editor panels (2.5).
 *
 * Pure pointer events on a slim divider strip: drag to resize, double-
 * click to reset, keyboard arrows for accessibility. Sizes persist per
 * viewport class (desktop / tablet / mobile) so a phone layout never
 * inherits a desktop panel width.
 */
import React, { useCallback, useRef, useState } from 'react';

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

/** Persisted, clamped panel size. viewportClass keys the storage. */
export function usePanelSize({ key, viewportClass = 'desktop', defaultSize, min, max }) {
  const storageKey = `clipai_panel_${key}:${viewportClass}`;
  const [size, setSize] = useState(() => {
    try {
      const stored = parseFloat(localStorage.getItem(storageKey));
      return Number.isFinite(stored) ? clamp(stored, min, max) : defaultSize;
    } catch { return defaultSize; }
  });

  const set = useCallback((next) => {
    const clamped = clamp(next, min, max);
    setSize(clamped);
    try { localStorage.setItem(storageKey, String(clamped)); } catch { /* private mode */ }
  }, [storageKey, min, max]);

  const reset = useCallback(() => set(defaultSize), [set, defaultSize]);

  return [size, set, reset];
}

/**
 * orientation 'vertical'   → divider is a vertical strip, drags resize a WIDTH
 * orientation 'horizontal' → divider is a horizontal strip, drags resize a HEIGHT
 * sign +1: size grows as the pointer moves right/down; -1: inverse
 * (use -1 for a right/bottom panel whose divider sits on its leading edge).
 */
export default function PanelDivider({
  orientation = 'vertical',
  size,
  onResize,
  onReset,
  sign = 1,
  step = 16,
  ariaLabel = 'Resize panel',
}) {
  const gestureRef = useRef(null);

  const onPointerDown = (e) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    gestureRef.current = {
      start: orientation === 'vertical' ? e.clientX : e.clientY,
      startSize: size,
    };
  };

  const onPointerMove = (e) => {
    const g = gestureRef.current;
    if (!g) return;
    const cur = orientation === 'vertical' ? e.clientX : e.clientY;
    onResize(g.startSize + (cur - g.start) * sign);
  };

  const onPointerUp = () => { gestureRef.current = null; };

  const onKeyDown = (e) => {
    const grow = orientation === 'vertical' ? 'ArrowRight' : 'ArrowDown';
    const shrink = orientation === 'vertical' ? 'ArrowLeft' : 'ArrowUp';
    if (e.key === grow) { e.preventDefault(); onResize(size + step * sign); }
    else if (e.key === shrink) { e.preventDefault(); onResize(size - step * sign); }
    else if (e.key === 'Home') { e.preventDefault(); onReset?.(); }
  };

  return (
    <div
      role="separator"
      aria-label={ariaLabel}
      aria-orientation={orientation}
      tabIndex={0}
      className={`ve-panel-divider ve-panel-divider--${orientation}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
    >
      <div className="ve-panel-divider__grip" aria-hidden="true" />
    </div>
  );
}
