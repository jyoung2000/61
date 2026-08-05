// ── Crop-element colour helpers (shared across the NLE) ──────────────────────
// A crop element's colour encodes its crop PERCENTAGE on a fixed red→blue
// thermometer: 0 % is pure red, 100 % is pure blue, and every percentage in
// between sits proportionally BETWEEN those two anchors — through magenta and
// purple (25 % warm red-pink, 50 % magenta, 75 % purple), NEVER through
// yellow/green. Each element is ONE solid colour, and the SAME percentage is
// the SAME colour everywhere it is drawn (main track, overview minimap,
// properties panel, scene card). This module is the single source of truth for
// that mapping so those surfaces can never diverge.
//
// Why the magenta path and not the rainbow: hue 0°→240° walks the colour wheel
// the LONG way, so mid percentages rendered olive (35 % ≈ 84°) and green
// (62 % ≈ 149°) — the track read as "red and green", not "red to blue". Going
// the SHORT way around the wheel (360° down to 240°) blends red directly into
// blue: every step is warmer-vs-cooler on one axis, with no third colour
// family hijacking the middle.

export const CROP_HUE_SPAN = 120; // degrees travelled: 360° red (0 %) → 240° blue (100 %)

// Discrete speaker/cluster palette (manual override = index 4). Kept here so
// both crop palettes live in one module; the percentage-hue above is what
// colours the crop bands, this is retained for callers that key off cluster
// identity.
export const CROP_CLUSTER_COLORS = [
  '#3B82F6', // blue — speaker 0
  '#10B981', // green — speaker 1
  '#F59E0B', // amber — speaker 2
  '#EC4899', // pink — speaker 3
  '#8B5CF6', // purple — manual override / unknown
];

// Map a crop percentage (0–100) to its single spectrum colour. Deterministic
// and monotonic — the same percentage always returns the same colour, and a
// higher percentage is always bluer (hue descends 360°→240°, red→magenta→
// purple→blue). Deliberately DARK (low lightness): crop elements carry a
// white % label, and a bright fill washes the text out; toning lightness down
// keeps every hue legible.
export function cropColorAt(cropPct, alpha = 1) {
  const x = Math.max(0, Math.min(100, Number.isFinite(cropPct) ? cropPct : 50)) / 100; // 0..1
  const hue = Math.round(360 - x * CROP_HUE_SPAN) % 360; // 0 % → 0° red … 100 % → 240° blue
  return `hsla(${hue}, 60%, 38%, ${alpha})`;
}
