import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import useTimelineStore from '../stores/timelineStore';
import ContextMenu from './ContextMenu';
import { snapToGuides } from '../utils/goldenGrid';

/**
 * InteractiveOverlay renders selectable, draggable, resizable, and rotatable
 * handles over overlay items (text, shape, image) in the video preview viewport.
 *
 * Props:
 *   currentTime  – playhead position (seconds, relative to clip start)
 *   clipStart    – absolute start of the clip
 *   containerRef – ref to the viewport container div
 *   onInteraction – callback(bool) to pause/resume play-on-click behavior
 */
export default function InteractiveOverlay({ currentTime = 0, clipStart = 0, containerRef, onInteraction }) {
  const items = useTimelineStore((s) => s.items);
  const tracks = useTimelineStore((s) => s.tracks);
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const selectedItemIds = useTimelineStore((s) => s.selectedItemIds);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const toggleSelectedItem = useTimelineStore((s) => s.toggleSelectedItem);
  const updateItem = useTimelineStore((s) => s.updateItem);
  const setItemPositions = useTimelineStore((s) => s.setItemPositions);

  // currentTime is already relative to clipStart (passed as currentTime - clipStart)
  const absTime = currentTime;

  // Filter to visible interactive items at current time.
  // Audio and subtitle items are excluded — subtitle items are rendered by SubtitleOverlay.
  // Respects track visibility — items on hidden tracks are not interactive.
  const visible = useMemo(() => {
    return items
      .filter((it) => {
        if (it.type === 'audio' || it.type === 'subtitle') return false;
        if (!(absTime >= it.start && absTime < it.end)) return false;
        // Check track visibility
        const track = tracks.find((t) => t.id === it.trackId);
        if (track && track.visible === false) return false;
        return true;
      })
      // Sort by track position: items on higher tracks (lower index) render later (on top)
      .sort((a, b) => {
        const idxA = tracks.findIndex((t) => t.id === a.trackId);
        const idxB = tracks.findIndex((t) => t.id === b.trackId);
        return idxB - idxA;
      });
  }, [items, absTime, tracks]);

  // Build a set of locked item IDs for quick lookup
  const lockedItemIds = useMemo(() => {
    const lockedTrackIds = new Set(tracks.filter((t) => t.locked).map((t) => t.id));
    return new Set(items.filter((it) => lockedTrackIds.has(it.trackId)).map((it) => it.id));
  }, [items, tracks]);

  // Right-click menu on preview overlay items (shared ContextMenu)
  const [overlayMenu, setOverlayMenu] = useState(null);

  const overlayMenuItems = useMemo(() => {
    if (!overlayMenu) return [];
    const store = useTimelineStore.getState();
    const item = store.items.find((i) => i.id === overlayMenu.itemId);
    if (!item) return [];
    const trackIdx = store.tracks.findIndex((t) => t.id === item.trackId);
    const track = store.tracks[trackIdx];
    const locked = !!track?.locked;
    // Z-order in the preview follows track order — bring forward/backward
    // swaps this item's track with the adjacent overlay track so preview,
    // client export and server compositing all agree on the new order.
    const canSwap = (dir) => {
      const target = store.tracks[trackIdx + dir];
      return !!target && target.type === 'overlay' && track?.type === 'overlay';
    };
    const swap = (dir) => store.reorderTracks(trackIdx, trackIdx + dir);
    return [
      { id: 'forward', label: 'Bring forward', disabled: locked || !canSwap(-1), onSelect: () => swap(-1) },
      { id: 'backward', label: 'Send backward', disabled: locked || !canSwap(1), onSelect: () => swap(1) },
      { separator: true },
      { id: 'duplicate', label: 'Duplicate', kbd: '⌘D', disabled: locked, onSelect: () => store.duplicateItem(item.id) },
      {
        id: 'reset-transform', label: 'Reset transform', disabled: locked,
        onSelect: () => {
          const defSize = item.type === 'video' ? { w: 100, h: 100 }
            : item.type === 'text' ? { w: 35, h: 12 }
            : item.type === 'shape' ? { w: 40, h: 30 }
            : { w: 30, h: 30 };
          store.updateItem(item.id, {
            position: { x: 50, y: 50 },
            size: defSize,
            transform: { ...(item.transform || {}), rotation: 0 },
          });
        },
      },
      { separator: true },
      { id: 'delete', label: 'Delete', kbd: '⌫', danger: true, disabled: locked, onSelect: () => store.removeItem(item.id) },
    ];
  }, [overlayMenu]);

  if (visible.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 10,
        // Container is transparent to clicks — only individual elements capture.
        // This lets clicks pass through to SubtitleOverlay and viewport below.
        pointerEvents: 'none',
      }}
    >
      {visible.map((item) => (
        <InteractiveElement
          key={item.id}
          item={item}
          isSelected={selectedItemIds.includes(item.id)}
          isPrimarySelection={item.id === selectedItemId}
          selectedItemIds={selectedItemIds}
          allItems={items}
          isLocked={lockedItemIds.has(item.id)}
          containerRef={containerRef}
          onSelect={(e) => {
            // Shift / Cmd / Ctrl click extends the selection instead of replacing it.
            if (e && (e.shiftKey || e.metaKey || e.ctrlKey)) {
              toggleSelectedItem(item.id);
            } else if (!selectedItemIds.includes(item.id)) {
              // Plain click on an unselected element resets the selection.
              // Plain click on an already-selected element keeps the group
              // intact so the user can start a multi-item drag.
              setSelectedItemId(item.id);
            }
          }}
          onUpdate={(updates) => updateItem(item.id, updates)}
          setItemPositions={setItemPositions}
          onInteraction={onInteraction}
          onContextMenu={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setSelectedItemId(item.id);
            setOverlayMenu({ itemId: item.id, x: e.clientX, y: e.clientY });
          }}
        />
      ))}
      {overlayMenu && overlayMenuItems.length > 0 && (
        <ContextMenu
          x={overlayMenu.x}
          y={overlayMenu.y}
          items={overlayMenuItems}
          onClose={() => setOverlayMenu(null)}
        />
      )}
    </div>
  );
}


// Touch devices get fat, finger-sized handle hit areas (the visible dot stays
// small). Detected once — pointer type is stable for a session.
const IO_COARSE_POINTER = typeof window !== 'undefined' && typeof window.matchMedia === 'function'
  ? window.matchMedia('(pointer: coarse)').matches
  : false;
const IO_HANDLE_HIT = IO_COARSE_POINTER ? 30 : 14;   // transparent grab area
const IO_HANDLE_DOT = IO_COARSE_POINTER ? 15 : 10;   // visible marker
const IO_ROTATE_HIT = IO_COARSE_POINTER ? 34 : 18;
const IO_ROTATE_DOT = IO_COARSE_POINTER ? 22 : 16;
const IO_ACCENT = '#6E7BFF';

// ── Corner / edge handle positions ────────────────────────────────
const HANDLES = [
  { id: 'nw', cursor: 'nwse-resize', x: 0, y: 0 },
  { id: 'ne', cursor: 'nesw-resize', x: 1, y: 0 },
  { id: 'sw', cursor: 'nesw-resize', x: 0, y: 1 },
  { id: 'se', cursor: 'nwse-resize', x: 1, y: 1 },
  { id: 'n',  cursor: 'ns-resize',   x: 0.5, y: 0 },
  { id: 's',  cursor: 'ns-resize',   x: 0.5, y: 1 },
  { id: 'w',  cursor: 'ew-resize',   x: 0, y: 0.5 },
  { id: 'e',  cursor: 'ew-resize',   x: 1, y: 0.5 },
];


function InteractiveElement({
  item,
  isSelected,
  isPrimarySelection = isSelected,
  selectedItemIds = [],
  allItems = [],
  isLocked,
  containerRef,
  onSelect,
  onUpdate,
  setItemPositions,
  onInteraction,
  onContextMenu,
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [isRotating, setIsRotating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const editRef = useRef(null);
  const dragState = useRef(null);
  // Long-press-to-edit (touch): a held press on a text/subtitle item opens
  // inline edit — the reliable touch equivalent of desktop double-click.
  const longPressRef = useRef(0);
  const longPressStartRef = useRef(null);

  // Use refs for callbacks/item to keep the useEffect stable during drag operations.
  // Without this, the mousemove/mouseup listeners get torn down and re-added on every
  // render (because inline onUpdate/onInteraction change identity each render), which
  // can drop mouse events mid-drag.
  const onUpdateRef = useRef(onUpdate);
  const onInteractionRef = useRef(onInteraction);
  const setItemPositionsRef = useRef(setItemPositions);
  const itemRef = useRef(item);
  onUpdateRef.current = onUpdate;
  onInteractionRef.current = onInteraction;
  setItemPositionsRef.current = setItemPositions;
  itemRef.current = item;

  const isVideo = item.type === 'video';
  const isText = item.type === 'text';
  const isShape = item.type === 'shape';

  // Video items default to centered full-size; overlays default to smaller centered
  const defaultPos = isVideo ? { x: 50, y: 50 } : { x: 50, y: 50 };
  const defaultSize = isVideo ? { w: 100, h: 100 } : { w: 30, h: 30 };
  const pos = item.position || defaultPos;
  // For items with legacy {x:0,y:0}, treat as centered (viewport uses % from center)
  const effectivePos = (pos.x === 0 && pos.y === 0) ? { x: 50, y: 50 } : pos;

  // For text items, auto-measure the text to compute a tight bounding box
  const measuredSize = useMemo(() => {
    if (!isText || !containerRef?.current) return null;
    const text = item.textContent || '';
    if (!text) return null;
    const style = item.textStyle || {};
    const fontSize = style.fontSize || 48;
    const fontFamily = style.fontFamily || 'DM Sans';
    const fontWeight = style.fontWeight || 400;
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext && canvas.getContext('2d');
      // ``getContext`` can return null in restricted browser contexts
      // (private mode quirks, very old browsers, certain mobile WebViews).
      // Without this guard the next line throws and the entire overlay
      // fails to render.
      if (!ctx) return null;
      ctx.font = `${fontWeight} ${fontSize}px "${fontFamily}", sans-serif`;
      const lines = text.split('\n');
      const textW = Math.max(...lines.map((l) => ctx.measureText(l).width));
      const textH = fontSize * 1.4 * Math.max(1, lines.length);
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        const pad = 20; // generous padding for easier click targeting
        const wPct = ((textW + pad * 2) / rect.width) * 100;
        const hPct = ((textH + pad * 2) / rect.height) * 100;
        return { w: Math.max(12, Math.min(95, wPct)), h: Math.max(8, Math.min(80, hPct)) };
      }
    } catch { /* fallback to stored size */ }
    return null;
  }, [isText, item.textContent, item.textStyle?.fontSize, item.textStyle?.fontFamily, item.textStyle?.fontWeight, containerRef]);

  const size = measuredSize || item.size || defaultSize;
  const rotation = item.transform?.rotation || 0;

  // ── Get container dimensions ──
  const getContainerRect = useCallback(() => {
    if (containerRef?.current) {
      return containerRef.current.getBoundingClientRect();
    }
    return { width: 1, height: 1, left: 0, top: 0 };
  }, [containerRef]);

  // ── DRAG ──────────────────────────────────────────────
  const handleDragStart = useCallback((e) => {
    // Right-button goes to the context menu, not a drag
    if (e.button === 2) return;
    e.stopPropagation();
    e.preventDefault();
    onSelect(e);
    if (isLocked) return; // Locked items can be selected but not moved
    onInteraction?.(true);

    const rect = getContainerRect();
    // Pause undo history during drag so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();

    // Capture the starting position of EVERY selected item (including this
    // one). On move we apply the same dx/dy in viewport % to all of them so
    // the marquee-grouped selection translates as a rigid block.
    const dragIds = (selectedItemIds && selectedItemIds.length > 0
      && selectedItemIds.includes(item.id))
      ? selectedItemIds
      : [item.id];
    const startPositions = {};
    for (const id of dragIds) {
      const it = allItems.find((x) => x.id === id);
      if (!it) continue;
      const p = it.position || { x: 50, y: 50 };
      const ep = (p.x === 0 && p.y === 0) ? { x: 50, y: 50 } : p;
      startPositions[id] = { x: ep.x, y: ep.y };
    }

    dragState.current = {
      type: 'drag',
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPosX: effectivePos.x,
      startPosY: effectivePos.y,
      containerW: rect.width,
      containerH: rect.height,
      startPositions,
      dragIds,
    };
    setIsDragging(true);

    // Touch long-press → inline edit for text/subtitle items (single selection).
    // A finger held still for 500ms aborts the pending drag and opens the
    // editor; any movement past 8px cancels it (that's a drag, not an edit).
    if ((item.type === 'text' || item.type === 'subtitle') && !isLocked
        && !(selectedItemIds && selectedItemIds.length > 1)) {
      longPressStartRef.current = { x: e.clientX, y: e.clientY };
      clearTimeout(longPressRef.current);
      longPressRef.current = setTimeout(() => {
        longPressRef.current = 0;
        longPressStartRef.current = null;
        dragState.current = null;
        setIsDragging(false);
        try { useTimelineStore.temporal.getState().resume(); } catch { /* noop */ }
        onInteractionRef.current?.(false);
        setIsEditing(true);
      }, 500);
    }
  }, [effectivePos, getContainerRect, onSelect, onInteraction, isLocked,
      selectedItemIds, allItems, item.id, item.type]);

  // ── RESIZE ────────────────────────────────────────────
  const handleResizeStart = useCallback((e, handleId) => {
    e.stopPropagation();
    e.preventDefault();
    if (isLocked) return; // Locked items cannot be resized
    onInteraction?.(true);

    const rect = getContainerRect();
    // Pause undo history during resize so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();

    // Capture every selected item's geometry. When a group is selected
    // we resize them all relative to the GROUP'S bounding box, so each
    // item scales proportionally instead of every item snapping to the
    // exact same size — that preserves their layout relative to each
    // other (the "...and relative to each other" requirement).
    const groupIds = (selectedItemIds && selectedItemIds.length > 1
      && selectedItemIds.includes(item.id))
      ? selectedItemIds
      : null;
    let groupBounds = null;
    let startGeoms = null;
    if (groupIds) {
      startGeoms = {};
      let gLeft = Infinity, gRight = -Infinity, gTop = Infinity, gBottom = -Infinity;
      for (const id of groupIds) {
        const it = allItems.find((x) => x.id === id);
        if (!it) continue;
        const p = it.position || { x: 50, y: 50 };
        const ep = (p.x === 0 && p.y === 0) ? { x: 50, y: 50 } : p;
        const sz = it.size || { w: 30, h: 30 };
        startGeoms[id] = {
          x: ep.x, y: ep.y, w: sz.w, h: sz.h,
          fontSize: it.textStyle?.fontSize,
          isText: it.type === 'text' || it.type === 'subtitle',
        };
        gLeft = Math.min(gLeft, ep.x - sz.w / 2);
        gRight = Math.max(gRight, ep.x + sz.w / 2);
        gTop = Math.min(gTop, ep.y - sz.h / 2);
        gBottom = Math.max(gBottom, ep.y + sz.h / 2);
      }
      if (Number.isFinite(gLeft)) {
        groupBounds = { left: gLeft, right: gRight, top: gTop, bottom: gBottom };
      }
    }

    dragState.current = {
      type: 'resize',
      handle: handleId,
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPosX: effectivePos.x,
      startPosY: effectivePos.y,
      startW: size.w,
      startH: size.h,
      containerW: rect.width,
      containerH: rect.height,
      startFontSize: item.textStyle?.fontSize || 48,
      isTextItem: item.type === 'text',
      groupIds,
      groupBounds,
      startGeoms,
    };
    setIsResizing(true);
  }, [effectivePos, size, item, getContainerRect, onInteraction, isLocked,
      selectedItemIds, allItems]);

  // ── ROTATE ────────────────────────────────────────────
  const handleRotateStart = useCallback((e) => {
    e.stopPropagation();
    e.preventDefault();
    if (isLocked) return; // Locked items cannot be rotated
    onInteraction?.(true);

    const rect = getContainerRect();
    // Group rotation pivots around the centre of the GROUP'S bounding box
    // (and orbits each item's position around that pivot) so the selection
    // rotates as one rigid block. Single-item rotation pivots around the
    // item itself, same as before.
    const groupIds = (selectedItemIds && selectedItemIds.length > 1
      && selectedItemIds.includes(item.id))
      ? selectedItemIds
      : null;
    let pivotX = effectivePos.x;
    let pivotY = effectivePos.y;
    let startGeoms = null;
    if (groupIds) {
      startGeoms = {};
      let gLeft = Infinity, gRight = -Infinity, gTop = Infinity, gBottom = -Infinity;
      for (const id of groupIds) {
        const it = allItems.find((x) => x.id === id);
        if (!it) continue;
        const p = it.position || { x: 50, y: 50 };
        const ep = (p.x === 0 && p.y === 0) ? { x: 50, y: 50 } : p;
        const sz = it.size || { w: 30, h: 30 };
        startGeoms[id] = {
          x: ep.x, y: ep.y,
          rotation: it.transform?.rotation || 0,
        };
        gLeft = Math.min(gLeft, ep.x - sz.w / 2);
        gRight = Math.max(gRight, ep.x + sz.w / 2);
        gTop = Math.min(gTop, ep.y - sz.h / 2);
        gBottom = Math.max(gBottom, ep.y + sz.h / 2);
      }
      if (Number.isFinite(gLeft)) {
        pivotX = (gLeft + gRight) / 2;
        pivotY = (gTop + gBottom) / 2;
      }
    }
    const centerX = rect.left + (pivotX / 100) * rect.width;
    const centerY = rect.top + (pivotY / 100) * rect.height;

    // Pause undo history during rotation so intermediate frames don't flood it
    useTimelineStore.temporal.getState().pause();
    dragState.current = {
      type: 'rotate',
      centerX,
      centerY,
      startAngle: Math.atan2(e.clientY - centerY, e.clientX - centerX) * (180 / Math.PI),
      startRotation: rotation,
      groupIds,
      pivotX,
      pivotY,
      containerW: rect.width,
      containerH: rect.height,
      startGeoms,
    };
    setIsRotating(true);
  }, [effectivePos, rotation, getContainerRect, onInteraction, isLocked,
      selectedItemIds, allItems, item.id]);

  // ── Pointer move / up handlers ────────────────────────
  // Pointer events unify mouse + touch + pen so the same drag, resize
  // and rotate paths work on tablets / phones. Only depends on the
  // boolean flags so listeners stay stable during the gesture; accesses
  // latest ``item``/``onUpdate``/``onInteraction`` via refs to avoid
  // listener churn.
  useEffect(() => {
    if (!isDragging && !isResizing && !isRotating) return;

    const handleMouseMove = (e) => {
      // Any real movement cancels a pending long-press-to-edit (it's a drag).
      if (longPressRef.current && longPressStartRef.current) {
        const lp = longPressStartRef.current;
        if (Math.hypot(e.clientX - lp.x, e.clientY - lp.y) > 8) {
          clearTimeout(longPressRef.current);
          longPressRef.current = 0;
          longPressStartRef.current = null;
        }
      }
      const ds = dragState.current;
      if (!ds) return;

      if (ds.type === 'drag') {
        const dx = e.clientX - ds.startMouseX;
        const dy = e.clientY - ds.startMouseY;
        const dxPct = (dx / ds.containerW) * 100;
        const dyPct = (dy / ds.containerH) * 100;

        // Multi-item drag: shift every selected item's start position by
        // the same dx/dy so they translate as a rigid block.
        if (ds.startPositions && ds.dragIds && ds.dragIds.length > 1
            && setItemPositionsRef.current) {
          const updates = {};
          for (const id of ds.dragIds) {
            const sp = ds.startPositions[id];
            if (!sp) continue;
            updates[id] = { x: sp.x + dxPct, y: sp.y + dyPct };
          }
          setItemPositionsRef.current(updates);
        } else {
          let newX = ds.startPosX + dxPct;
          let newY = ds.startPosY + dyPct;
          // Magnetic snap to the golden grid, frame centre/edges, and other
          // elements (Photoshop-style) while the grid is on. Holding Alt
          // temporarily disables it for free, un-snapped placement.
          const store = useTimelineStore.getState();
          if (store.goldenGrid && !e.altKey) {
            const it = itemRef.current;
            const sz = it.size || { w: 20, h: 20 };
            const others = store.items
              .filter((o) => o.id !== it.id && o.position
                && o.type !== 'audio' && o.type !== 'subtitle' && o.type !== 'video')
              .map((o) => ({ x: o.position.x, y: o.position.y, w: o.size?.w || 0, h: o.size?.h || 0 }));
            // ~12px pull on each axis, converted to the % space of each
            // dimension, so vertical and horizontal snapping feel identical.
            const thX = (12 / (ds.containerW || 1)) * 100;
            const thY = (12 / (ds.containerH || 1)) * 100;
            const snapped = snapToGuides({ x: newX, y: newY, w: sz.w, h: sz.h, others, thX, thY });
            newX = snapped.x;
            newY = snapped.y;
            store.setSnapGuides(snapped.guides);
          } else if (store.snapGuides.length) {
            store.setSnapGuides([]);
          }
          onUpdateRef.current({
            position: {
              x: Math.max(0, Math.min(100, Math.round(newX * 10) / 10)),
              y: Math.max(0, Math.min(100, Math.round(newY * 10) / 10)),
            },
          });
        }
      }

      if (ds.type === 'resize') {
        const dx = ((e.clientX - ds.startMouseX) / ds.containerW) * 100;
        const dy = ((e.clientY - ds.startMouseY) / ds.containerH) * 100;
        const h = ds.handle;

        let newX = ds.startPosX;
        let newY = ds.startPosY;
        let newW = ds.startW;
        let newH = ds.startH;

        if (h.includes('e')) {
          newW = Math.max(2, ds.startW + dx);
          newX = ds.startPosX + dx / 2;
        }
        if (h.includes('w')) {
          newW = Math.max(2, ds.startW - dx);
          newX = ds.startPosX + dx / 2;
        }
        if (h.includes('s')) {
          newH = Math.max(2, ds.startH + dy);
          newY = ds.startPosY + dy / 2;
        }
        if (h.includes('n')) {
          newH = Math.max(2, ds.startH - dy);
          newY = ds.startPosY + dy / 2;
        }

        // Shift key = lock aspect ratio
        if (e.shiftKey && ds.startW > 0 && ds.startH > 0) {
          const aspect = ds.startW / ds.startH;
          if (h === 'e' || h === 'w') {
            newH = newW / aspect;
          } else if (h === 'n' || h === 's') {
            newW = newH * aspect;
          } else {
            const dw = Math.abs(newW - ds.startW);
            const dh = Math.abs(newH - ds.startH);
            if (dw > dh) {
              newH = newW / aspect;
            } else {
              newW = newH * aspect;
            }
          }
        }

        const updates = {
          position: {
            x: Math.round(newX * 10) / 10,
            y: Math.round(newY * 10) / 10,
          },
          size: {
            w: Math.round(Math.max(2, newW) * 10) / 10,
            h: Math.round(Math.max(2, newH) * 10) / 10,
          },
        };

        // Scale font size for text items on corner (diagonal) drags
        const currentItem = itemRef.current;
        if (ds.isTextItem && h.length === 2 && ds.startW > 0) {
          const scale = newW / ds.startW;
          const newFontSize = Math.max(8, Math.min(400, Math.round(ds.startFontSize * scale)));
          updates.textStyle = { ...(currentItem.textStyle || {}), fontSize: newFontSize };
        }

        onUpdateRef.current(updates);

        // Group resize: scale every selected item's position + size around
        // the group's starting bounds. The handle being dragged controls
        // the active item; the GROUP'S bounds get the same proportional
        // change, and each other item is repositioned/resized to stay in
        // the same relative spot inside those bounds.
        if (ds.groupBounds && ds.groupIds && ds.groupIds.length > 1
            && ds.startGeoms && setItemPositionsRef.current) {
          const gb = ds.groupBounds;
          const startW = Math.max(0.001, gb.right - gb.left);
          const startH = Math.max(0.001, gb.bottom - gb.top);

          let gLeft = gb.left, gRight = gb.right, gTop = gb.top, gBottom = gb.bottom;
          if (h.includes('e')) gRight = gb.right + dx;
          if (h.includes('w')) gLeft = gb.left + dx;
          if (h.includes('s')) gBottom = gb.bottom + dy;
          if (h.includes('n')) gTop = gb.top + dy;
          // Aspect lock for the group (Shift)
          if (e.shiftKey) {
            const newGW = gRight - gLeft;
            const newGH = gBottom - gTop;
            const aspect = startW / startH;
            if (h === 'e' || h === 'w') {
              const targetH = newGW / aspect;
              const cy = (gTop + gBottom) / 2;
              gTop = cy - targetH / 2;
              gBottom = cy + targetH / 2;
            } else if (h === 'n' || h === 's') {
              const targetW = newGH * aspect;
              const cx = (gLeft + gRight) / 2;
              gLeft = cx - targetW / 2;
              gRight = cx + targetW / 2;
            } else {
              const dw = Math.abs(newGW - startW);
              const dh = Math.abs(newGH - startH);
              if (dw > dh) {
                const targetH = newGW / aspect;
                if (h.includes('s')) gBottom = gTop + targetH;
                else gTop = gBottom - targetH;
              } else {
                const targetW = newGH * aspect;
                if (h.includes('e')) gRight = gLeft + targetW;
                else gLeft = gRight - targetW;
              }
            }
          }
          const scaleX = (gRight - gLeft) / startW;
          const scaleY = (gBottom - gTop) / startH;
          const updatesAll = {};
          for (const id of ds.groupIds) {
            if (id === item.id) continue; // already updated above
            const sg = ds.startGeoms[id];
            if (!sg) continue;
            const relX = (sg.x - gb.left) / startW;
            const relY = (sg.y - gb.top) / startH;
            const nx = gLeft + relX * (gRight - gLeft);
            const ny = gTop + relY * (gBottom - gTop);
            const nw = sg.w * scaleX;
            const nh = sg.h * scaleY;
            const u = { x: nx, y: ny, w: nw, h: nh };
            updatesAll[id] = u;
            // Text items also scale their font size with the group so
            // the rendered text grows/shrinks with the bounding box.
            if (sg.isText && sg.fontSize) {
              const fontScale = Math.min(scaleX, scaleY);
              const newFontSize = Math.max(
                8, Math.min(400, Math.round(sg.fontSize * fontScale)),
              );
              const it = allItems.find((x) => x.id === id);
              if (it) {
                useTimelineStore.getState().updateItem(id, {
                  textStyle: { ...(it.textStyle || {}), fontSize: newFontSize },
                });
              }
            }
          }
          setItemPositionsRef.current(updatesAll);
        }
      }

      if (ds.type === 'rotate') {
        const angle = Math.atan2(e.clientY - ds.centerY, e.clientX - ds.centerX) * (180 / Math.PI);
        const delta = angle - ds.startAngle;
        let newRotation = ds.startRotation + delta;

        if (e.shiftKey) {
          newRotation = Math.round(newRotation / 15) * 15;
        }
        newRotation = ((newRotation % 360) + 360) % 360;
        if (newRotation > 180) newRotation -= 360;

        const currentItem = itemRef.current;
        onUpdateRef.current({
          transform: { ...(currentItem.transform || {}), rotation: Math.round(newRotation) },
        });

        // Group rotation: orbit every other selected item's position
        // around the group pivot, and rotate each item by the same delta
        // so the whole selection rotates rigidly.
        if (ds.groupIds && ds.groupIds.length > 1 && ds.startGeoms
            && setItemPositionsRef.current) {
          let dRad = (delta * Math.PI) / 180;
          if (e.shiftKey) {
            // Snap the delta to 15° so positions and rotations stay aligned.
            const snapped = Math.round(delta / 15) * 15;
            dRad = (snapped * Math.PI) / 180;
          }
          const cosD = Math.cos(dRad);
          const sinD = Math.sin(dRad);
          // Convert % to a uniform px-like space using container aspect so
          // orbits stay circular regardless of viewport shape.
          const ar = (ds.containerW || 1) / (ds.containerH || 1);
          const updatesAll = {};
          for (const id of ds.groupIds) {
            if (id === item.id) continue;
            const sg = ds.startGeoms[id];
            if (!sg) continue;
            const rx = (sg.x - ds.pivotX) * ar;
            const ry = sg.y - ds.pivotY;
            const nx = ds.pivotX + (rx * cosD - ry * sinD) / ar;
            const ny = ds.pivotY + (rx * sinD + ry * cosD);
            updatesAll[id] = { x: nx, y: ny };
            // Also rotate each item individually
            const it = allItems.find((x) => x.id === id);
            if (it) {
              let r = sg.rotation + (e.shiftKey
                ? Math.round(delta / 15) * 15
                : delta);
              r = ((r % 360) + 360) % 360;
              if (r > 180) r -= 360;
              useTimelineStore.getState().updateItem(id, {
                transform: { ...(it.transform || {}), rotation: Math.round(r) },
              });
            }
          }
          setItemPositionsRef.current(updatesAll);
        }
      }
    };

    const handleMouseUp = () => {
      // A quick tap-release cancels any pending long-press (it was a tap/drag).
      if (longPressRef.current) { clearTimeout(longPressRef.current); longPressRef.current = 0; longPressStartRef.current = null; }
      // Resume undo history so the final state is recorded as one snapshot
      useTimelineStore.temporal.getState().resume();
      // Clear any live snap guide lines.
      if (useTimelineStore.getState().snapGuides.length) useTimelineStore.getState().setSnapGuides([]);
      dragState.current = null;
      setIsDragging(false);
      setIsResizing(false);
      setIsRotating(false);
      setTimeout(() => onInteractionRef.current?.(false), 100);
    };

    // Pointer events are a superset of mouse/touch/pen — registering
    // ``pointermove`` + ``pointerup`` (with ``pointercancel`` for the
    // touch-interrupted case) is enough to drive every input modality.
    window.addEventListener('pointermove', handleMouseMove);
    window.addEventListener('pointerup', handleMouseUp);
    window.addEventListener('pointercancel', handleMouseUp);
    return () => {
      window.removeEventListener('pointermove', handleMouseMove);
      window.removeEventListener('pointerup', handleMouseUp);
      window.removeEventListener('pointercancel', handleMouseUp);
      // Safety: resume undo history if component unmounts during drag
      if (isDragging || isResizing || isRotating) {
        useTimelineStore.temporal.getState().resume();
      }
    };
  }, [isDragging, isResizing, isRotating]);

  // ── Click to select ───────────────────────────────────
  const handleClick = useCallback((e) => {
    e.stopPropagation();
    onSelect(e);
  }, [onSelect]);

  // ── Double-click to edit text/subtitle ────────────────
  const handleDoubleClick = useCallback((e) => {
    e.stopPropagation();
    if (isLocked) return; // Locked items cannot be edited
    if (item.type === 'text' || item.type === 'subtitle') {
      setIsEditing(true);
      onInteraction?.(true);
      // Focus the input after render
      setTimeout(() => editRef.current?.focus(), 50);
    }
  }, [item.type, onInteraction, isLocked]);

  const handleEditBlur = useCallback(() => {
    setIsEditing(false);
    onInteraction?.(false);
  }, [onInteraction]);

  const handleEditChange = useCallback((e) => {
    const val = e.target.value;
    if (item.type === 'text') {
      onUpdate({ textContent: val });
    } else if (item.type === 'subtitle') {
      onUpdate({ subtitleText: val });
    }
  }, [item.type, onUpdate]);

  const handleEditKeyDown = useCallback((e) => {
    e.stopPropagation(); // prevent keyboard shortcuts
    if (e.key === 'Escape') {
      setIsEditing(false);
      onInteraction?.(false);
    }
  }, [onInteraction]);

  // Determine the bounding box style.
  // Elements use percentage-based position (center) and size.
  // Text items get extra padding so the bounding box isn't cramped
  const boxPad = 0;
  const boxStyle = {
    position: 'absolute',
    left: `${effectivePos.x}%`,
    top: `${effectivePos.y}%`,
    width: `${size.w}%`,
    height: `${size.h}%`,
    padding: boxPad > 0 ? boxPad : undefined,
    transform: `translate(-50%, -50%) ${rotation ? `rotate(${rotation}deg)` : ''}`,
    cursor: isLocked ? 'not-allowed' : isDragging ? 'grabbing' : (isVideo ? 'default' : 'grab'),
    // Video items: pointer-events none so they don't block overlays.
    // Exception: during active manipulation, allow events for smooth tracking.
    pointerEvents: isVideo
      ? (isDragging || isResizing || isRotating ? 'auto' : 'none')
      : 'auto',
    minWidth: 20,
    minHeight: 20,
    // Video items always sit behind overlays (z:1). Other items at z:10, selected at z:15.
    // This ensures text/shape/image overlays are always clickable above the video.
    zIndex: isVideo ? 1 : (isSelected ? 15 : 10),
  };

  const isActive = isDragging || isResizing || isRotating;

  return (
    <div
      data-item-id={item.id}
      data-item-type={item.type}
      style={{
        ...boxStyle,
        // ``touch-action: none`` lets us own the touch sequence so the
        // browser doesn't preempt the drag with scroll/zoom gestures.
        touchAction: isLocked ? 'auto' : 'none',
      }}
      onPointerDown={isEditing ? undefined : handleDragStart}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onContextMenu={onContextMenu}
      onPointerEnter={() => setIsHovered(true)}
      onPointerLeave={() => setIsHovered(false)}
    >
      {/* Invisible hit area matching the element */}
      <div style={{
        position: 'absolute',
        inset: -4,
        borderRadius: 2,
      }} />

      {/* Inline text editing */}
      {isEditing && (item.type === 'text' || item.type === 'subtitle') && (
        <textarea
          ref={editRef}
          value={item.type === 'text' ? (item.textContent || '') : (item.subtitleText || '')}
          onChange={handleEditChange}
          onBlur={handleEditBlur}
          onKeyDown={handleEditKeyDown}
          onPointerDown={(e) => e.stopPropagation()}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            minHeight: 40,
            background: 'rgba(0,0,0,0.6)',
            color: '#fff',
            border: '2px solid #6E7BFF',
            borderRadius: 4,
            padding: '6px 8px',
            fontSize: 14,
            fontFamily: 'inherit',
            resize: 'none',
            outline: 'none',
            zIndex: 50,
            backdropFilter: 'blur(4px)',
          }}
        />
      )}

      {/* Hover outline (when not selected) */}
      {isHovered && !isSelected && (
        <div style={{
          position: 'absolute',
          inset: -1,
          border: '1px dashed rgba(110, 123, 255, 0.5)',
          borderRadius: 2,
          pointerEvents: 'none',
        }} />
      )}

      {/* Selection outline — drawn on every selected member of the group.
          Cyan = primary (carries the handles), dashed cyan = secondary
          members (selected but no handles, drag still moves them). */}
      {isSelected && (
        <>
          {/* Selection border — interactive: enables drag-from-border for
              every selected item, including secondary group members, so
              the whole group can be moved from any of its outlines. */}
          <div
            style={{
              position: 'absolute',
              inset: -2,
              border: isPrimarySelection
                ? '2px solid #6E7BFF'
                : '2px dashed rgba(110, 123, 255, 0.85)',
              borderRadius: 2,
              pointerEvents: 'auto',
              boxShadow: isPrimarySelection
                ? '0 0 0 1px rgba(110, 123, 255, 0.3)'
                : undefined,
              cursor: isLocked ? 'not-allowed' : (isDragging ? 'grabbing' : 'grab'),
              background: 'transparent',
              touchAction: 'none',
            }}
            onPointerDown={isEditing ? undefined : handleDragStart}
          />
        </>
      )}

      {/* Resize / rotate handles only on the PRIMARY selection. For a
          group selection the handles operate on the primary item; resize
          scales every other selected item proportionally around the
          group's bounding box, and rotation orbits them around its
          centre — that's the "relative to each other" behaviour. */}
      {isPrimarySelection && (
        <>
          {/* Resize handles — a large transparent grab area (finger-sized on
              touch) centred on each corner/edge, with a small visible dot. */}
          {HANDLES.map((h) => (
            <div
              key={h.id}
              style={{
                position: 'absolute',
                left: `${h.x * 100}%`,
                top: `${h.y * 100}%`,
                width: IO_HANDLE_HIT,
                height: IO_HANDLE_HIT,
                transform: 'translate(-50%, -50%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: h.cursor,
                zIndex: 30,
                pointerEvents: 'auto',
                touchAction: 'none',
              }}
              onPointerDown={(e) => handleResizeStart(e, h.id)}
            >
              <div style={{
                width: IO_HANDLE_DOT,
                height: IO_HANDLE_DOT,
                background: '#fff',
                border: `2px solid ${IO_ACCENT}`,
                borderRadius: h.id.length === 2 ? 2 : '50%', // corners = square, edges = round
                boxShadow: '0 1px 3px rgba(0,0,0,0.3)',
                pointerEvents: 'none',
              }} />
            </div>
          ))}

          {/* Rotation handle (above the element) */}
          <div style={{
            position: 'absolute',
            left: '50%',
            top: -24,
            transform: 'translateX(-50%)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            zIndex: 30,
            pointerEvents: 'auto',
          }}>
            {/* Stem line connecting to element */}
            <div style={{
              width: 1,
              height: 8,
              background: IO_ACCENT,
              position: 'absolute',
              bottom: -8,
            }} />
            {/* Rotation handle — large transparent grab area, small visible circle */}
            <div
              style={{
                width: IO_ROTATE_HIT,
                height: IO_ROTATE_HIT,
                cursor: 'grab',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'auto',
                touchAction: 'none',
              }}
              onPointerDown={handleRotateStart}
              title="Rotate"
            >
              <div style={{
                width: IO_ROTATE_DOT,
                height: IO_ROTATE_DOT,
                borderRadius: '50%',
                background: '#fff',
                border: `2px solid ${IO_ACCENT}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                pointerEvents: 'none',
              }}>
                {/* Rotation icon */}
                <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke={IO_ACCENT} strokeWidth="2">
                  <path d="M14 8A6 6 0 1 1 8 2" strokeLinecap="round" />
                  <path d="M8 0l3 2-3 2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
            </div>
          </div>

          {/* Info badge showing position/size while dragging */}
          {isActive && (
            <div style={{
              position: 'absolute',
              left: '50%',
              bottom: -24,
              transform: 'translateX(-50%)',
              background: 'rgba(0,0,0,0.8)',
              color: '#fff',
              fontSize: 10,
              fontFamily: 'var(--ve-font-mono, monospace)',
              padding: '2px 6px',
              borderRadius: 3,
              whiteSpace: 'nowrap',
              pointerEvents: 'none',
              zIndex: 40,
              backdropFilter: 'blur(4px)',
            }}>
              {isDragging && (selectedItemIds.length > 1
                ? `${selectedItemIds.length} items`
                : `${effectivePos.x.toFixed(1)}%, ${effectivePos.y.toFixed(1)}%`)}
              {isResizing && (selectedItemIds.length > 1
                ? `${selectedItemIds.length} items`
                : `${size.w.toFixed(1)}% x ${size.h.toFixed(1)}%`)}
              {isRotating && `${rotation}°`}
            </div>
          )}
        </>
      )}
    </div>
  );
}
