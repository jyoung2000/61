import React from 'react';
import useTimelineStore from '../stores/timelineStore';
import { GOLDEN_LINES } from '../utils/goldenGrid';

/**
 * GoldenGridOverlay — a togglable golden-ratio ("golden canon") grid drawn over
 * the preview to help align subtitles, images, videos, and shapes.
 *
 * The complete golden canon armature is drawn:
 *   • the four φ lines at ≈38.2% / 61.8% of each axis (1-1/φ, 1/φ),
 *   • the faint centre cross,
 *   • the frame diagonals + the golden "reciprocal" diagonals (the classic
 *     harmonic armature that locates the golden points), and
 *   • the four golden power points at the φ-line intersections.
 * Everything is expressed in percentages of the stage, so the grid is correct
 * and responsive for every aspect ratio automatically.
 *
 * It also renders the transient snap guide lines that light up while an element
 * magnetically snaps into place (Photoshop-style). Fills its positioned parent
 * and never captures pointer events.
 */
const GOLD = 'rgba(233, 184, 75, 0.60)';       // φ lines
const GOLD_SOFT = 'rgba(233, 184, 75, 0.30)';  // centre cross
const GOLD_DIAG = 'rgba(233, 184, 75, 0.22)';  // armature diagonals
const SNAP = '#6E7BFF';                          // active snap guide (accent)

const A = GOLDEN_LINES[0]; // ≈38.197
const B = GOLDEN_LINES[1]; // ≈61.803

export default function GoldenGridOverlay() {
  const goldenGrid = useTimelineStore((s) => s.goldenGrid);
  const snapGuides = useTimelineStore((s) => s.snapGuides);

  if (!goldenGrid && (!snapGuides || snapGuides.length === 0)) return null;

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 13, overflow: 'hidden' }}>
      {goldenGrid && (
        <>
          {/* Diagonals + reciprocals (harmonic armature) via a non-distorting SVG.
              preserveAspectRatio=none maps 0–100 user units to the box; the
              non-scaling stroke keeps lines hairline at any size. */}
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
          >
            {[
              // Frame diagonals (corner ↔ corner)
              [0, 0, 100, 100], [100, 0, 0, 100],
              // Golden reciprocals: each corner to the two far golden points,
              // whose crossings sit on the φ lines (the "armature of the rectangle").
              [0, 0, 100, B], [0, 0, B, 100],
              [100, 0, 0, B], [100, 0, A, 100],
              [0, 100, 100, A], [0, 100, B, 0],
              [100, 100, 0, A], [100, 100, A, 0],
            ].map(([x1, y1, x2, y2], i) => (
              <line
                key={i}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke={i < 2 ? GOLD_SOFT : GOLD_DIAG}
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>

          {/* φ lines */}
          {GOLDEN_LINES.map((p, i) => (
            <div key={`gv${i}`} style={{ position: 'absolute', left: `${p}%`, top: 0, bottom: 0, width: 1, background: GOLD, transform: 'translateX(-0.5px)' }} />
          ))}
          {GOLDEN_LINES.map((p, i) => (
            <div key={`gh${i}`} style={{ position: 'absolute', top: `${p}%`, left: 0, right: 0, height: 1, background: GOLD, transform: 'translateY(-0.5px)' }} />
          ))}
          {/* Faint centre cross */}
          <div style={{ position: 'absolute', left: '50%', top: 0, bottom: 0, width: 1, background: GOLD_SOFT, transform: 'translateX(-0.5px)' }} />
          <div style={{ position: 'absolute', top: '50%', left: 0, right: 0, height: 1, background: GOLD_SOFT, transform: 'translateY(-0.5px)' }} />
          {/* Golden "power points" at the φ-line intersections */}
          {GOLDEN_LINES.map((px) => GOLDEN_LINES.map((py) => (
            <div key={`pp${px}-${py}`} style={{ position: 'absolute', left: `${px}%`, top: `${py}%`, width: 7, height: 7, marginLeft: -3.5, marginTop: -3.5, borderRadius: '50%', background: GOLD, boxShadow: '0 0 0 1px rgba(0,0,0,0.3)' }} />
          )))}
        </>
      )}
      {/* Active snap guides — brighter, only while dragging into place */}
      {(snapGuides || []).map((g, i) => (g.axis === 'x' ? (
        <div key={`sx${i}`} style={{ position: 'absolute', left: `${g.pos}%`, top: 0, bottom: 0, width: 2, background: SNAP, boxShadow: '0 0 4px rgba(110,123,255,0.95)', transform: 'translateX(-1px)' }} />
      ) : (
        <div key={`sy${i}`} style={{ position: 'absolute', top: `${g.pos}%`, left: 0, right: 0, height: 2, background: SNAP, boxShadow: '0 0 4px rgba(110,123,255,0.95)', transform: 'translateY(-1px)' }} />
      )))}
    </div>
  );
}
