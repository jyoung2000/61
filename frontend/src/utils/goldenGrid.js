/**
 * Golden-ratio ("golden canon") grid geometry + magnetic snapping.
 *
 * The golden ratio φ = (1+√5)/2 ≈ 1.618. The "phi grid" (a.k.a. golden-ratio
 * grid / golden canon) divides a frame at the golden-section fractions of each
 * axis — 1 − 1/φ ≈ 0.382 and 1/φ ≈ 0.618 — giving a 3×3 grid whose centre cell
 * is φ-proportioned to the outer cells (tighter than the rule-of-thirds, whose
 * lines sit at 0.333 / 0.667). Because the fractions are relative to the frame,
 * the SAME grid is correct for every aspect ratio; rendered as percentages of
 * the preview box it stays responsive and per-aspect-ratio automatically.
 *
 * Elements in the editor are positioned by their CENTRE as a percentage of the
 * preview (0–100 on each axis), with a size also in percent, so all snapping
 * math below is in that same percent space.
 */

export const PHI = 1.61803398875;

// Golden-section line positions as PERCENT of the axis (0–100).
export const GOLDEN_LINES = [100 * (1 - 1 / PHI), 100 * (1 / PHI)]; // ≈ [38.197, 61.803]

// The frame guides every drag can snap its CENTRE to (percent): the two golden
// lines plus the exact centre. Edges (0 / 100) are handled separately using the
// element's half-size so an element's EDGE — not its centre — meets the frame.
const CENTER_GUIDES = [GOLDEN_LINES[0], 50, GOLDEN_LINES[1]];

/**
 * Snap a dragged element to the golden grid, the frame centre/edges, and other
 * elements — Photoshop-guide style. All inputs/outputs are in percent.
 *
 * @param {object}   p
 * @param {number}   p.x        proposed element centre X (%)
 * @param {number}   p.y        proposed element centre Y (%)
 * @param {number}   p.w        element width (%)
 * @param {number}   p.h        element height (%)
 * @param {Array}    p.others   other elements: [{ x, y, w, h }] (centres+sizes %)
 * @param {number}   p.thX      snap threshold on X (%) — usually px→% converted
 * @param {number}   p.thY      snap threshold on Y (%)
 * @returns {{ x:number, y:number, guides:Array<{axis:'x'|'y',pos:number}> }}
 */
export function snapToGuides({ x, y, w, h, others = [], thX = 1.5, thY = 1.5 }) {
  const halfW = (w || 0) / 2;
  const halfH = (h || 0) / 2;

  const bestX = _bestSnap(x, halfW, others, 'x', thX);
  const bestY = _bestSnap(y, halfH, others, 'y', thY);

  const guides = [];
  const outX = bestX ? bestX.value : x;
  const outY = bestY ? bestY.value : y;
  if (bestX) guides.push({ axis: 'x', pos: bestX.guide });
  if (bestY) guides.push({ axis: 'y', pos: bestY.guide });
  return { x: outX, y: outY, guides };
}

// Find the closest snap for one axis. `center` is the element centre on that
// axis; `half` is its half-extent; returns { value:newCenter, guide:linePos }
// or null when nothing is within `threshold`.
function _bestSnap(center, half, others, axis, threshold) {
  // Candidate = { c: desired centre value, g: guide-line position to draw }.
  const cands = [];

  // Frame centre + golden lines (centre-to-line).
  for (const g of CENTER_GUIDES) cands.push({ c: g, g });
  // Frame edges (element edge-to-frame-edge).
  cands.push({ c: half, g: 0 });          // leading edge → 0
  cands.push({ c: 100 - half, g: 100 });  // trailing edge → 100

  // Alignment to other elements (centre↔centre, edge↔edge).
  for (const o of others) {
    const oc = axis === 'x' ? o.x : o.y;
    const oh = ((axis === 'x' ? o.w : o.h) || 0) / 2;
    const oLead = oc - oh;   // their leading edge
    const oTrail = oc + oh;  // their trailing edge
    cands.push({ c: oc, g: oc });                    // centres align
    cands.push({ c: oLead + half, g: oLead });       // our lead → their lead
    cands.push({ c: oTrail - half, g: oTrail });     // our trail → their trail
    cands.push({ c: oTrail + half, g: oTrail });     // our lead → their trail (abut)
    cands.push({ c: oLead - half, g: oLead });       // our trail → their lead (abut)
  }

  let best = null;
  for (const cand of cands) {
    const d = Math.abs(center - cand.c);
    if (d <= threshold && (!best || d < best.d)) best = { value: cand.c, guide: cand.g, d };
  }
  return best;
}
