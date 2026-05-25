import React, { useEffect, useRef, useState, useCallback } from 'react';
import useTimelineStore from '../stores/timelineStore';

/**
 * MarqueeSelection — drag-out box on the preview that selects every
 * overlay element whose on-screen bounding rect intersects it.
 *
 * Strategy: listen for ``pointerdown`` on the stage container. If the
 * mousedown target is the stage itself (i.e. an empty area, not an
 * element — every interactive element calls ``stopPropagation`` on its
 * own ``pointerdown``), start a marquee. Move events update the box;
 * mouseup commits the selection.
 *
 * Hit-testing reads ``data-item-id`` attributes from the actual rendered
 * elements (InteractiveOverlay items + the current subtitle span). Using
 * ``getBoundingClientRect`` keeps the geometry honest regardless of how
 * each overlay computes its own position (settings-based subtitle layout,
 * letterboxed video-content rects, rotation, etc.).
 *
 * Props:
 *   containerRef   – ref to the stage div the marquee should draw inside
 *   onInteraction  – callback(bool) to pause the click-to-toggle-play
 *                    behaviour while the marquee is active
 *   disabled       – skip wiring (e.g. in fullscreen / mobile)
 */
export default function MarqueeSelection({ containerRef, onInteraction, disabled = false }) {
  const setSelectedItemIds = useTimelineStore((s) => s.setSelectedItemIds);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const items = useTimelineStore((s) => s.items);
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const onInteractionRef = useRef(onInteraction);
  onInteractionRef.current = onInteraction;

  // ``rect`` is in CSS coordinates relative to the stage container.
  const [rect, setRect] = useState(null);
  const stateRef = useRef(null);

  const startMarquee = useCallback((e) => {
    const stage = containerRef?.current;
    if (!stage) return;
    // Only the LEFT mouse button starts a marquee.
    if (e.button !== 0 && e.button !== undefined) return;
    // Ignore pointerdowns that came FROM an interactive element — they
    // stop propagation on the React layer, but native ``addEventListener``
    // still receives the bubbling event. ``closest`` walks the DOM up
    // from the actual target to catch every nested case.
    if (e.target && e.target.closest && e.target.closest('[data-item-id]')) return;
    // Skip if the user is interacting with form controls (e.g. inline
    // textarea on a subtitle being edited).
    if (e.target && e.target.tagName === 'TEXTAREA') return;
    if (e.target && e.target.tagName === 'INPUT') return;

    const stageRect = stage.getBoundingClientRect();
    const x0 = e.clientX - stageRect.left;
    const y0 = e.clientY - stageRect.top;
    // Stash state and start tracking, but DON'T render the box or claim
    // ``overlayInteracting`` yet — that happens on first real movement
    // (see ``handleMove`` below). A pure click on empty space therefore
    // falls through to the viewport's click-to-toggle-play handler,
    // preserving existing behaviour.
    stateRef.current = {
      stage,
      stageRect,
      startX: x0,
      startY: y0,
      additive: e.shiftKey || e.metaKey || e.ctrlKey,
      moved: false,
      // Snapshot the existing selection so additive marquees can layer
      // onto it without clobbering Shift+click work.
      baseSelection: useTimelineStore.getState().selectedItemIds.slice(),
    };
    setRect({ left: x0, top: y0, width: 0, height: 0, dormant: true });
  }, [containerRef]);

  // Wire pointerdown on the stage. The handler is added once per stage
  // change; pointermove/pointerup are added globally only while a
  // marquee is in flight (see the next effect).
  useEffect(() => {
    if (disabled) return undefined;
    const stage = containerRef?.current;
    if (!stage) return undefined;
    stage.addEventListener('pointerdown', startMarquee);
    return () => stage.removeEventListener('pointerdown', startMarquee);
  }, [containerRef, disabled, startMarquee]);

  // Global move/up listeners while marquee is dragging.
  useEffect(() => {
    if (!rect) return undefined;

    const handleMove = (e) => {
      const st = stateRef.current;
      if (!st) return;
      const x = e.clientX - st.stageRect.left;
      const y = e.clientY - st.stageRect.top;
      const dx = x - st.startX;
      const dy = y - st.startY;
      if (!st.moved && Math.abs(dx) + Math.abs(dy) > 3) {
        st.moved = true;
        // First real motion — claim the interaction so the viewport's
        // click handler doesn't toggle play when the gesture ends.
        onInteractionRef.current?.(true);
      }
      setRect({
        left: Math.min(st.startX, x),
        top: Math.min(st.startY, y),
        width: Math.abs(dx),
        height: Math.abs(dy),
        dormant: !st.moved,
      });
    };

    const handleUp = () => {
      const st = stateRef.current;
      const finalRect = rect;
      const didMove = !!(st && st.moved);
      stateRef.current = null;
      setRect(null);
      if (didMove) {
        // Small delay before clearing onInteraction so the viewport's
        // play-on-click handler still sees the gesture and skips.
        setTimeout(() => onInteractionRef.current?.(false), 80);
      }
      if (!st) return;

      // Pure click (no movement) — don't interfere. The viewport's
      // own onClick handles play-toggle / video-item-select.
      if (!didMove || !finalRect || finalRect.width < 3 || finalRect.height < 3) {
        return;
      }

      // Compute marquee rect in client coords for intersection tests.
      const stageRect = st.stage.getBoundingClientRect();
      const marqueeLeft = stageRect.left + finalRect.left;
      const marqueeTop = stageRect.top + finalRect.top;
      const marqueeRight = marqueeLeft + finalRect.width;
      const marqueeBottom = marqueeTop + finalRect.height;

      const els = st.stage.querySelectorAll('[data-item-id]');
      const hits = [];
      for (const el of els) {
        const id = el.getAttribute('data-item-id');
        if (!id) continue;
        const r = el.getBoundingClientRect();
        // Standard AABB intersection. ``getBoundingClientRect`` already
        // accounts for CSS rotation by returning the rotated bbox.
        if (
          r.right > marqueeLeft &&
          r.left < marqueeRight &&
          r.bottom > marqueeTop &&
          r.top < marqueeBottom
        ) {
          // Don't include the background video item in marquees — a
          // full-stage drag would otherwise always grab it.
          const type = el.getAttribute('data-item-type');
          if (type === 'video') continue;
          if (!hits.includes(id)) hits.push(id);
        }
      }

      let next = hits;
      if (st.additive) {
        // Toggle behaviour: items already in the base selection but inside
        // the marquee get removed; items not in the base get added.
        const base = new Set(st.baseSelection);
        for (const id of hits) {
          if (base.has(id)) base.delete(id);
          else base.add(id);
        }
        next = Array.from(base);
      }
      // Keep only IDs that still exist in the timeline (avoids stale
      // selections if items were removed during the drag).
      const live = new Set(itemsRef.current.map((it) => it.id));
      next = next.filter((id) => live.has(id));
      setSelectedItemIds(next);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    window.addEventListener('pointercancel', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
      window.removeEventListener('pointercancel', handleUp);
    };
  }, [rect, setSelectedItemIds, setSelectedItemId]);

  // Hide the box during the dormant phase (mousedown happened, no movement
  // yet) so a plain click on empty space looks identical to before.
  if (!rect || rect.dormant) return null;

  return (
    <div
      style={{
        position: 'absolute',
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        pointerEvents: 'none',
        border: '1px solid rgba(10, 132, 255, 0.9)',
        background: 'rgba(10, 132, 255, 0.12)',
        zIndex: 25,
        boxSizing: 'border-box',
      }}
    />
  );
}
