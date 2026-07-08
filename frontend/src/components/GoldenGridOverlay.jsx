import React from 'react';
import useTimelineStore from '../stores/timelineStore';
import { GOLDEN_LINES } from '../utils/goldenGrid';

/**
 * GoldenGridOverlay — a togglable golden-ratio ("golden canon") grid drawn over
 * the preview to help align subtitles, images, videos, and shapes. The φ lines
 * sit at ≈38.2% / 61.8% of each axis; because they're percentages of the stage
 * (which is sized to the selected aspect ratio) the grid is correct and
 * responsive for every ratio. It also renders the transient snap guide lines
 * that light up while an element magnetically snaps into place (Photoshop-style).
 *
 * Fills its positioned parent (the stage) and never captures pointer events.
 */
const GOLD = 'rgba(233, 184, 75, 0.55)';       // golden-section lines
const GOLD_SOFT = 'rgba(233, 184, 75, 0.26)';  // centre cross
const SNAP = '#6E7BFF';                          // active snap guide (accent)

export default function GoldenGridOverlay() {
  const goldenGrid = useTimelineStore((s) => s.goldenGrid);
  const snapGuides = useTimelineStore((s) => s.snapGuides);

  if (!goldenGrid && (!snapGuides || snapGuides.length === 0)) return null;

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 13, overflow: 'hidden' }}>
      {goldenGrid && (
        <>
          {GOLDEN_LINES.map((p, i) => (
            <div key={`gv${i}`} style={{ position: 'absolute', left: `${p}%`, top: 0, bottom: 0, width: 1, background: GOLD, transform: 'translateX(-0.5px)' }} />
          ))}
          {GOLDEN_LINES.map((p, i) => (
            <div key={`gh${i}`} style={{ position: 'absolute', top: `${p}%`, left: 0, right: 0, height: 1, background: GOLD, transform: 'translateY(-0.5px)' }} />
          ))}
          {/* Faint centre cross */}
          <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: GOLD_SOFT, transform: 'translateX(-0.5px)' }} />
          <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: 1, background: GOLD_SOFT, transform: 'translateY(-0.5px)' }} />
          {/* Golden "power points" at the line intersections */}
          {GOLDEN_LINES.map((px) => GOLDEN_LINES.map((py) => (
            <div key={`pp${px}-${py}`} style={{ position: 'absolute', left: `${px}%`, top: `${py}%`, width: 6, height: 6, marginLeft: -3, marginTop: -3, borderRadius: '50%', background: GOLD, boxShadow: '0 0 0 1px rgba(0,0,0,0.25)' }} />
          )))}
        </>
      )}
      {/* Active snap guides — brighter, only while dragging into place */}
      {(snapGuides || []).map((g, i) => (g.axis === 'x' ? (
        <div key={`sx${i}`} style={{ position: 'absolute', left: `${g.pos}%`, top: 0, bottom: 0, width: 1, background: SNAP, boxShadow: '0 0 3px rgba(110,123,255,0.9)', transform: 'translateX(-0.5px)' }} />
      ) : (
        <div key={`sy${i}`} style={{ position: 'absolute', top: `${g.pos}%`, left: 0, right: 0, height: 1, background: SNAP, boxShadow: '0 0 3px rgba(110,123,255,0.9)', transform: 'translateY(-0.5px)' }} />
      )))}
    </div>
  );
}
