// ── Crop-element colour helpers (shared across the NLE) ──────────────────────
// A crop element's colour encodes its FRAMING POSITION: crop X (0–100 %) mapped
// across a full ~280° hue sweep — a left-biased crop is warm (red/orange), a
// centred crop is green, a right-biased crop is cool (blue/violet). So a human-
// operator pan from 20 %→80 % reads as a rainbow wipe, and the SAME crop % is
// the SAME colour everywhere it is drawn (main track, overview minimap,
// properties panel, scene card). This module is the single source of truth for
// that mapping so those surfaces can never diverge again.

export const CROP_HUE_SPAN = 280; // 0° red (left) → 280° violet-blue (right); no wrap back to red

// Discrete speaker/cluster palette (manual override = index 4). Kept here so
// both crop palettes live in one module; the position-hue above is what colours
// the crop bands, this is retained for callers that key off cluster identity.
export const CROP_CLUSTER_COLORS = [
  '#3B82F6', // blue — speaker 0
  '#10B981', // green — speaker 1
  '#F59E0B', // amber — speaker 2
  '#EC4899', // pink — speaker 3
  '#8B5CF6', // purple — manual override / unknown
];

// Map a crop X (0–100 %) to a full-spectrum hue. Deterministic and monotonic in
// ``cropX`` — the same percentage always returns the same colour. Deliberately
// DARK (low lightness): crop elements carry a white % label, and a bright fill
// washes the text out; toning lightness down keeps every hue legible.
export function cropColorAt(cropX, alpha = 1) {
  const x = Math.max(0, Math.min(100, Number.isFinite(cropX) ? cropX : 50)) / 100; // 0..1
  const hue = Math.round(x * CROP_HUE_SPAN);
  return `hsla(${hue}, 60%, 38%, ${alpha})`;
}
