/**
 * Tooltip — styled replacement for native title= on editor chrome.
 *
 * Apple-style behavior: 400 ms delay on first hover, then INSTANT
 * re-show while the pointer moves between sibling controls (a shared
 * module-scope timestamp tracks the last dismissal; re-entering within
 * 500 ms skips the delay). Renders a label plus an optional ⌘-chip.
 *
 *   <Tooltip label="Split" kbd="⌘⇧S"><button …/></Tooltip>
 *   <Tooltip actionId="split-at-playhead"><button …/></Tooltip>
 */
import React, { cloneElement, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getActionById } from '../utils/editorActions';
import { formatKbd } from '../utils/platform';

const SHOW_DELAY_MS = 400;
const CHAIN_WINDOW_MS = 500;

// Shared across all tooltips: when did the last one hide?
let lastHideAt = 0;

export default function Tooltip({ label, kbd, actionId, side = 'top', children }) {
  const action = actionId ? getActionById(actionId) : null;
  const text = label ?? action?.label;
  const chip = kbd ?? action?.kbd;

  const timerRef = useRef(0);
  const anchorRef = useRef(null);
  const [pos, setPos] = useState(null);

  const show = () => {
    const el = anchorRef.current;
    if (!el || !text) return;
    const r = el.getBoundingClientRect();
    setPos({
      x: r.left + r.width / 2,
      y: side === 'bottom' ? r.bottom + 6 : r.top - 6,
      side,
    });
  };

  const onEnter = () => {
    clearTimeout(timerRef.current);
    const chained = Date.now() - lastHideAt < CHAIN_WINDOW_MS;
    if (chained) show();
    else timerRef.current = setTimeout(show, SHOW_DELAY_MS);
  };

  const onLeave = () => {
    clearTimeout(timerRef.current);
    setPos((p) => {
      if (p) lastHideAt = Date.now();
      return null;
    });
  };

  const child = React.Children.only(children);
  const anchored = cloneElement(child, {
    ref: (node) => {
      anchorRef.current = node;
      const { ref } = child;
      if (typeof ref === 'function') ref(node);
      else if (ref && typeof ref === 'object') ref.current = node;
    },
    onPointerEnter: (e) => { child.props.onPointerEnter?.(e); onEnter(); },
    onPointerLeave: (e) => { child.props.onPointerLeave?.(e); onLeave(); },
    onPointerDown: (e) => { child.props.onPointerDown?.(e); onLeave(); },
    'aria-label': child.props['aria-label'] ?? text,
  });

  return (
    <>
      {anchored}
      {pos && createPortal(
        <div
          className={`ve-tooltip ve-tooltip--${pos.side}`}
          style={{ left: pos.x, top: pos.y }}
          role="tooltip"
        >
          <span className="ve-tooltip__label">{text}</span>
          {chip && <kbd className="ve-tooltip__kbd">{formatKbd(chip)}</kbd>}
        </div>,
        document.body,
      )}
    </>
  );
}
