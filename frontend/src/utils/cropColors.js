// ── Crop-element colour helpers (shared across the NLE) ──────────────────────
// A crop element's colour encodes its crop PERCENTAGE on a fixed red→blue
// thermometer: 0 % is pure red (hue 0°), 100 % is pure blue (hue 240°), and
// every percentage in between sits proportionally on that spectrum (25 %
// orange, 50 % yellow-green, 75 % cyan-ish). Each element is ONE solid colour,
// and the SAME percentage is the SAME colour everywhere it is drawn (main
// track, overview minimap, properties panel, scene card). This module is the
// single source of truth for that mapping so those surfaces can never diverge.
//
// Why red→blue and not the previous ~280° full sweep: the sweep ended in
// violet, which put "high %" (blue) and "low-ish %" (red/orange) at BOTH ends
// of a wheel that visually circles back — blue-vs-green mid-tones carried no
// readable ordering. A thermometer with fixed anchors does: red always means
// "near 0 %", blue always means "near 100 %".

export const CROP_HUE_SPAN = 240; // 0° red (0 %) → 240° blue (100 %); no wrap past blue

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
// higher percentage is always bluer. Deliberately DARK (low lightness): crop
// elements carry a white % label, and a bright fill washes the text out;
// toning lightness down keeps every hue legible.
export function cropColorAt(cropPct, alpha = 1) {
  const x = Math.max(0, Math.min(100, Number.isFinite(cropPct) ? cropPct : 50)) / 100; // 0..1
  const hue = Math.round(x * CROP_HUE_SPAN);
  return `hsla(${hue}, 60%, 38%, ${alpha})`;
}
