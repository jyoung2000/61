import { useEffect, useCallback, useRef } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { matchAction, EDITOR_ACTIONS } from '../utils/editorActions';

/**
 * Centralized keyboard shortcut handling for the video editor.
 *
 * Dispatch is registry-driven: every simple action lives in
 * utils/editorActions.js (the same registry the command palette and the
 * shortcut cheat sheet render), so keys, palette and help can't drift.
 * The only logic kept here is press-and-hold repeat for the arrow keys,
 * which needs keyup pairing and an interval.
 */
export default function useKeyboardShortcuts({
  enabled = true,
  onTogglePlay,
  onSeek,
  onSkipTime,
  onToggleMute,
  onShuttleSpeed,
  onOpenPalette,
  onOpenHelp,
  // Wall-clock seek bounds for Home / End. When the editor is showing
  // a clip slice (clipStart > 0), Home / End must seek to the **wall
  // clock** boundaries of the clip, not the timeline-relative 0 /
  // duration. Defaulting to ``null`` preserves the legacy behavior.
  clipRange = null,
} = {}) {
  // Stable refs for all callback props — prevents effect re-registration
  // from killing arrow hold intervals when parent re-renders
  const arrowHoldRef = useRef({ key: null, interval: null });
  const ctxRef = useRef({});
  ctxRef.current = {
    onTogglePlay, onSeek, onSkipTime, onToggleMute, onShuttleSpeed,
    onOpenPalette, onOpenHelp, clipRange,
  };

  const handleKeyDown = useCallback((e) => {
    if (!enabled) return;

    // Don't capture when typing in inputs/textareas/contenteditable
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.target.contentEditable === 'true') return;
    if (e.target.closest && e.target.closest('[contenteditable="true"]')) return;

    // ── Nudge selected WYSIWYG element(s) with arrow keys ──
    // When a positionable overlay (text / shape / image / subtitle) is
    // selected, arrows move it in the preview (Shift = larger step, Alt =
    // fine) instead of seeking. Non-positionable selections (video / audio
    // clips) fall through to the seek handler below.
    if (e.code === 'ArrowLeft' || e.code === 'ArrowRight'
        || e.code === 'ArrowUp' || e.code === 'ArrowDown') {
      const st = useTimelineStore.getState();
      const ids = (st.selectedItemIds && st.selectedItemIds.length)
        ? st.selectedItemIds
        : (st.selectedItemId ? [st.selectedItemId] : []);
      const POSITIONABLE = new Set(['text', 'shape', 'image', 'overlay', 'subtitle']);
      const targets = ids
        .map((id) => st.items.find((it) => it.id === id))
        .filter((it) => it && POSITIONABLE.has(it.type)
          && !(st.tracks.find((t) => t.id === it.trackId)?.locked));
      if (targets.length) {
        e.preventDefault();
        const stepPct = e.shiftKey ? 3 : e.altKey ? 0.1 : 0.5;
        const dx = e.code === 'ArrowLeft' ? -stepPct : e.code === 'ArrowRight' ? stepPct : 0;
        const dy = e.code === 'ArrowUp' ? -stepPct : e.code === 'ArrowDown' ? stepPct : 0;
        const updates = {};
        for (const it of targets) {
          const p = it.position || { x: 50, y: 50 };
          updates[it.id] = { x: p.x + dx, y: p.y + dy };
        }
        if (typeof st.setItemPositions === 'function') st.setItemPositions(updates);
        else for (const it of targets) st.updateItem(it.id, { position: updates[it.id] });
        return;
      }
      // No positionable selection — fall through (Left/Right seek below).
    }

    // ── Arrow nudge with press-and-hold repeat ──
    if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
      e.preventDefault();
      const dir = e.code === 'ArrowLeft' ? -1 : 1;
      const delta = e.shiftKey ? dir : dir / 30;

      const hold = arrowHoldRef.current;
      if (e.repeat) {
        // If our interval is still running, let it handle stepping
        if (hold.interval) return;
        // Otherwise effect cleanup killed the interval; restart below
      } else {
        // First press: immediate single step
        ctxRef.current.onSkipTime?.(delta);
      }
      // Start (or restart) hold-to-repeat interval
      if (hold.interval) clearInterval(hold.interval);
      hold.key = e.code;
      hold.interval = setInterval(() => {
        ctxRef.current.onSkipTime?.(delta);
      }, 1000 / 15); // 15 steps per second while held
      return;
    }

    // ── Registry dispatch ──
    const action = matchAction(e);
    if (action) {
      e.preventDefault();
      action.run(ctxRef.current, e);
    }
  }, [enabled]);

  const handleKeyUp = useCallback((e) => {
    if (e.code === 'ArrowLeft' || e.code === 'ArrowRight') {
      const hold = arrowHoldRef.current;
      if (hold.key === e.code && hold.interval) {
        clearInterval(hold.interval);
        hold.interval = null;
        hold.key = null;
      }
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      const hold = arrowHoldRef.current;
      if (hold.interval) {
        clearInterval(hold.interval);
        hold.interval = null;
        hold.key = null;
      }
    };
  }, [enabled, handleKeyDown, handleKeyUp]);
}

// Re-export for components that want the registry (palette, cheat sheet)
export { EDITOR_ACTIONS };
