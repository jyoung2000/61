/**
 * BottomSheet — mobile inspector container (3.1).
 *
 * Drag-handle sheet with three detents (peek / half / full). The tabbed
 * PropertiesPanel renders inside unchanged, so phone and desktop share
 * one inspector implementation. Respects env(safe-area-inset-bottom).
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';

const DETENTS = { peek: 0.16, half: 0.45, full: 0.85 }; // × viewport height

export default function BottomSheet({ children, title = 'Inspector', onClose }) {
  const [detent, setDetent] = useState('half');
  const [dragY, setDragY] = useState(null); // live height while dragging (px)
  const gestureRef = useRef(null);

  // Detents measure against the VISIBLE viewport: on iOS Safari
  // window.innerHeight includes space under the collapsed URL bar and
  // above the keyboard, so a "half" sheet could bury its handle. The
  // visualViewport API tracks the live visible height; resubscribe on
  // its resize so the detents follow keyboard/toolbar changes.
  const readVh = () => (typeof window !== 'undefined'
    ? (window.visualViewport?.height || window.innerHeight)
    : 800);
  const [vh, setVh] = useState(readVh);
  useEffect(() => {
    const vv = typeof window !== 'undefined' ? window.visualViewport : null;
    const update = () => setVh(readVh());
    (vv || window).addEventListener('resize', update);
    return () => (vv || window).removeEventListener('resize', update);
  }, []);
  const heightPx = dragY ?? Math.round(DETENTS[detent] * vh);

  const snap = useCallback((px) => {
    let best = 'peek';
    let bestDelta = Infinity;
    for (const [name, frac] of Object.entries(DETENTS)) {
      const delta = Math.abs(px - frac * vh);
      if (delta < bestDelta) { bestDelta = delta; best = name; }
    }
    setDetent(best);
    setDragY(null);
  }, [vh]);

  const onHandleDown = (e) => {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    gestureRef.current = { startY: e.clientY, startH: heightPx };
  };

  const onHandleMove = (e) => {
    const g = gestureRef.current;
    if (!g) return;
    const next = Math.max(64, Math.min(vh * 0.92, g.startH + (g.startY - e.clientY)));
    setDragY(next);
  };

  const onHandleUp = () => {
    const g = gestureRef.current;
    gestureRef.current = null;
    if (g && dragY != null) snap(dragY);
  };

  return (
    <div
      className="ve-bottom-sheet"
      style={{ height: heightPx }}
      role="dialog"
      aria-label={title}
    >
      <div
        className="ve-bottom-sheet__handle-row"
        onPointerDown={onHandleDown}
        onPointerMove={onHandleMove}
        onPointerUp={onHandleUp}
        onPointerCancel={onHandleUp}
        onDoubleClick={() => setDetent(detent === 'full' ? 'half' : 'full')}
      >
        <div className="ve-bottom-sheet__grabber" aria-hidden="true" />
        <span className="ve-bottom-sheet__title">{title}</span>
        {onClose && (
          <button className="ve-bottom-sheet__close" onClick={onClose} aria-label={`Close ${title}`}>
            ✕
          </button>
        )}
      </div>
      <div className="ve-bottom-sheet__content">{children}</div>
    </div>
  );
}
