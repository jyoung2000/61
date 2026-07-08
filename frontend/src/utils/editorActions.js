/**
 * Editor action registry — SINGLE SOURCE OF TRUTH for every keyboard-
 * reachable editor action. Consumed by:
 *   • useKeyboardShortcuts (key dispatch)
 *   • CommandPalette (⌘K fuzzy search + run)
 *   • ShortcutCheatSheet (? overlay)
 *   • Tooltip kbd chips (via getActionById)
 *
 * Adding an action here gives it a shortcut, a palette entry and a
 * cheat-sheet row at once — the pre-registry switch statement meant the
 * palette and help could silently drift from what keys actually did.
 *
 * Each def:
 *   id        — stable slug
 *   label     — human name (palette / cheat sheet / tooltips)
 *   category  — cheat-sheet grouping
 *   kbd       — display string ('⌘⇧Z', 'Space', …)
 *   code      — KeyboardEvent.code to match (or `codes` array)
 *   mod       — true: requires Ctrl/Cmd; false/absent: requires NEITHER
 *   shift/alt — required modifier state (absent = must be off)
 *   holdRepeat— matched by the hook's press-and-hold path, not dispatch
 *   run(ctx, e) — perform the action; must tolerate a missing event
 *                 (palette invokes with no KeyboardEvent)
 */
import useTimelineStore from '../stores/timelineStore';

const st = () => useTimelineStore.getState();

export const EDITOR_ACTIONS = [
  // ── Transport ─────────────────────────────────────────────────────
  { id: 'play-pause', label: 'Play / pause', category: 'Transport', kbd: 'Space', code: 'Space',
    run: (ctx) => ctx.onTogglePlay?.() },
  { id: 'shuttle-reverse', label: 'Shuttle reverse', category: 'Transport', kbd: 'J', code: 'KeyJ',
    run: (ctx) => ctx.onShuttleSpeed?.('reverse') },
  { id: 'shuttle-stop', label: 'Shuttle stop', category: 'Transport', kbd: 'K', code: 'KeyK',
    run: (ctx) => ctx.onShuttleSpeed?.('stop') },
  { id: 'shuttle-forward', label: 'Shuttle forward', category: 'Transport', kbd: 'L', code: 'KeyL',
    run: (ctx) => ctx.onShuttleSpeed?.('forward') },
  { id: 'loop-toggle', label: 'Toggle loop playback', category: 'Transport', kbd: '⇧L', code: 'KeyL', shift: true,
    run: () => { if (typeof st().toggleLoop === 'function') st().toggleLoop(); } },
  { id: 'go-start', label: 'Go to clip start', category: 'Transport', kbd: 'Home', code: 'Home',
    run: (ctx) => {
      const r = ctx.clipRange;
      ctx.onSeek?.(r && Number.isFinite(r.start) ? r.start : 0);
    } },
  { id: 'go-end', label: 'Go to clip end', category: 'Transport', kbd: 'End', code: 'End',
    run: (ctx) => {
      const r = ctx.clipRange;
      ctx.onSeek?.(r && Number.isFinite(r.end) ? r.end : st().duration);
    } },
  { id: 'frame-back', label: 'Step back one frame', category: 'Transport', kbd: ',', code: 'Comma',
    run: (ctx) => ctx.onSkipTime?.(-1 / ((st().project && st().project.fps) || 30)) },
  { id: 'frame-forward', label: 'Step forward one frame', category: 'Transport', kbd: '.', code: 'Period',
    run: (ctx) => ctx.onSkipTime?.(1 / ((st().project && st().project.fps) || 30)) },
  { id: 'frame-back-5', label: 'Step back five frames', category: 'Transport', kbd: '⇧,', code: 'Comma', shift: true,
    run: (ctx) => ctx.onSkipTime?.(-5 / ((st().project && st().project.fps) || 30)) },
  { id: 'frame-forward-5', label: 'Step forward five frames', category: 'Transport', kbd: '⇧.', code: 'Period', shift: true,
    run: (ctx) => ctx.onSkipTime?.(5 / ((st().project && st().project.fps) || 30)) },
  { id: 'nudge-back', label: 'Nudge back (hold to scrub)', category: 'Transport', kbd: '←', code: 'ArrowLeft', holdRepeat: true,
    run: (ctx, e) => ctx.onSkipTime?.(e?.shiftKey ? -1 : -1 / 30) },
  { id: 'nudge-forward', label: 'Nudge forward (hold to scrub)', category: 'Transport', kbd: '→', code: 'ArrowRight', holdRepeat: true,
    run: (ctx, e) => ctx.onSkipTime?.(e?.shiftKey ? 1 : 1 / 30) },
  { id: 'mute', label: 'Mute audio', category: 'Transport', kbd: 'M', code: 'KeyM',
    run: (ctx) => ctx.onToggleMute?.() },

  // ── Edit ──────────────────────────────────────────────────────────
  { id: 'undo', label: 'Undo', category: 'Edit', kbd: '⌘Z', code: 'KeyZ', mod: true,
    run: () => useTimelineStore.temporal.getState().undo() },
  { id: 'redo', label: 'Redo', category: 'Edit', kbd: '⌘⇧Z', code: 'KeyZ', mod: true, shift: true,
    run: () => useTimelineStore.temporal.getState().redo() },
  { id: 'redo-y', label: 'Redo', category: 'Edit', kbd: '⌘Y', code: 'KeyY', mod: true, hidden: true,
    run: () => useTimelineStore.temporal.getState().redo() },
  { id: 'select-all', label: 'Select all items', category: 'Edit', kbd: '⌘A', code: 'KeyA', mod: true,
    run: () => st().setSelectedItemIds(st().items.map((i) => i.id)) },
  { id: 'deselect', label: 'Deselect', category: 'Edit', kbd: 'Esc', code: 'Escape',
    run: () => st().setSelectedItemId(null) },
  { id: 'delete-selection', label: 'Delete selection', category: 'Edit', kbd: '⌫', codes: ['Delete', 'Backspace'],
    run: () => {
      const s = st();
      const ids = s.selectedItemIds.length > 0
        ? s.selectedItemIds
        : (s.selectedItemId ? [s.selectedItemId] : []);
      if (ids.length === 1) s.removeItem(ids[0]);
      else if (ids.length > 1) s.removeItems(ids);
    } },
  { id: 'split-at-playhead', label: 'Split clip at playhead', category: 'Edit', kbd: 'S', code: 'KeyS',
    run: () => {
      const s = st();
      const target = s.items.find((i) =>
        s.playhead > i.start + 0.1 && s.playhead < i.end - 0.1
        && (i.type === 'video' || i.type === 'audio'));
      if (target) s.splitItem(target.id, s.playhead);
    } },
  { id: 'duplicate', label: 'Duplicate selection', category: 'Edit', kbd: '⌘D', code: 'KeyD', mod: true,
    run: () => {
      const s = st();
      const ids = s.selectedItemIds.length > 0
        ? s.selectedItemIds
        : (s.selectedItemId ? [s.selectedItemId] : []);
      for (const id of ids) s.duplicateItem(id);
    } },
  { id: 'group-toggle', label: 'Group / ungroup selection', category: 'Edit', kbd: '⌘G', code: 'KeyG', mod: true,
    run: () => {
      const s = st();
      const selIds = s.selectedItemIds;
      if (selIds.length < 2) return;
      const selItems = selIds.map((id) => s.items.find((i) => i.id === id)).filter(Boolean);
      const firstGroupId = selItems[0]?.groupId;
      const allSameGroup = firstGroupId && selItems.every((i) => i.groupId === firstGroupId);
      if (allSameGroup) s.ungroupItems(selIds);
      else s.groupItems(selIds);
    } },
  { id: 'ungroup', label: 'Ungroup selection', category: 'Edit', kbd: '⌘⇧G', code: 'KeyG', mod: true, shift: true,
    run: () => {
      const s = st();
      if (s.selectedItemIds.length > 0) s.ungroupItems(s.selectedItemIds);
    } },

  // ── Tools ─────────────────────────────────────────────────────────
  { id: 'tool-select', label: 'Select tool', category: 'Tools', kbd: 'V', code: 'KeyV',
    run: () => st().setActiveTool('select') },
  { id: 'tool-razor', label: 'Razor tool', category: 'Tools', kbd: 'C', code: 'KeyC',
    run: () => st().setActiveTool('razor') },
  { id: 'tool-text', label: 'Text tool', category: 'Tools', kbd: 'T', code: 'KeyT',
    run: () => st().setActiveTool('text') },
  { id: 'tool-shape', label: 'Shape tool', category: 'Tools', kbd: 'R', code: 'KeyR',
    run: () => st().setActiveTool('shape') },
  { id: 'toggle-snap', label: 'Toggle snapping', category: 'Tools', kbd: 'N', code: 'KeyN',
    run: () => st().toggleSnap() },
  { id: 'toggle-golden-grid', label: 'Toggle golden-ratio grid', category: 'Tools', kbd: 'G', code: 'KeyG',
    run: () => st().toggleGoldenGrid() },
  { id: 'toggle-ripple', label: 'Toggle ripple editing', category: 'Tools', kbd: '\\', code: 'Backslash',
    run: () => st().toggleRipple() },

  // ── Loop range ────────────────────────────────────────────────────
  { id: 'loop-in', label: 'Set loop in point', category: 'Loop', kbd: 'I', code: 'KeyI',
    run: () => { const s = st(); if (typeof s.setLoopRange === 'function') s.setLoopRange({ start: s.playhead || 0 }); } },
  { id: 'loop-out', label: 'Set loop out point', category: 'Loop', kbd: 'O', code: 'KeyO',
    run: () => { const s = st(); if (typeof s.setLoopRange === 'function') s.setLoopRange({ end: s.playhead || 0 }); } },
  { id: 'loop-in-clear', label: 'Clear loop in point', category: 'Loop', kbd: '⌥I', code: 'KeyI', alt: true,
    run: () => { const s = st(); if (typeof s.setLoopRange === 'function') s.setLoopRange({ start: null }); } },
  { id: 'loop-out-clear', label: 'Clear loop out point', category: 'Loop', kbd: '⌥O', code: 'KeyO', alt: true,
    run: () => { const s = st(); if (typeof s.setLoopRange === 'function') s.setLoopRange({ end: null }); } },

  // ── View ──────────────────────────────────────────────────────────
  { id: 'zoom-to-fit', label: 'Zoom timeline to fit', category: 'View', kbd: '⇧Z', code: 'KeyZ', shift: true,
    run: () => window.dispatchEvent(new CustomEvent('ve:zoom-fit')) },

  // ── App ───────────────────────────────────────────────────────────
  // Swallow ⌘S so the browser's save dialog never appears mid-edit
  // (project state autosaves via useTimelinePersistence).
  { id: 'save', label: 'Save project (autosaves)', category: 'App', kbd: '⌘S', code: 'KeyS', mod: true, hidden: true,
    run: () => {} },
  { id: 'command-palette', label: 'Command palette', category: 'App', kbd: '⌘K', code: 'KeyK', mod: true,
    run: (ctx) => ctx.onOpenPalette?.() },
  { id: 'shortcut-help', label: 'Keyboard shortcuts', category: 'App', kbd: '?', code: 'Slash', shift: true,
    run: (ctx) => ctx.onOpenHelp?.() },
];

export function getActionById(id) {
  return EDITOR_ACTIONS.find((a) => a.id === id) || null;
}

/**
 * Match a keydown against the registry. Modifier semantics: `mod`
 * requires Ctrl or Cmd (absent = both must be up), same for shift/alt.
 */
export function matchAction(e) {
  const mod = !!(e.ctrlKey || e.metaKey);
  return EDITOR_ACTIONS.find((a) => {
    if (a.holdRepeat) return false; // hook handles press-and-hold itself
    const codes = a.codes || [a.code];
    if (!codes.includes(e.code)) return false;
    if (!!a.mod !== mod) return false;
    if (!!a.shift !== !!e.shiftKey) return false;
    if (!!a.alt !== !!e.altKey) return false;
    return true;
  }) || null;
}

/** Simple subsequence fuzzy match; returns a score (higher = better) or -1. */
export function fuzzyScore(query, text) {
  const q = query.toLowerCase().trim();
  const t = text.toLowerCase();
  if (!q) return 0;
  let qi = 0;
  let score = 0;
  let streak = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      qi += 1;
      streak += 1;
      score += streak * 2 + (ti === 0 || t[ti - 1] === ' ' ? 3 : 0);
    } else {
      streak = 0;
    }
  }
  return qi === q.length ? score : -1;
}
