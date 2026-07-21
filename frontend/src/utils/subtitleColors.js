/**
 * Active-word highlight color resolution — keep the karaoke highlight
 * visually distinct from the speaker's caption color.
 *
 * The default highlight (#FFD700 gold) collides with the amber/gold
 * speaker palette entries (#F59E0B, #FBBF24, #F97316...): the active word
 * became invisible against those speakers' text. When the configured
 * highlight is too close in hue to the speaker color, fall through a
 * candidate list until one clears the distance check.
 *
 * MIRRORED in backend/services/ass_generator.py (_resolve_aw_color) so the
 * exported burn-in picks the identical color — keep the two in sync.
 */

const FALLBACKS = ['#FFD700', '#3DFF8C', '#FF5CF0', '#00E5FF', '#FFFFFF'];

function hexToHsl(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec((hex || '').trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  const r = ((n >> 16) & 255) / 255;
  const g = ((n >> 8) & 255) / 255;
  const b = (n & 255) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  if (d === 0) return { h: 0, s: 0, l };
  const s = d / (1 - Math.abs(2 * l - 1));
  let h;
  if (max === r) h = 60 * (((g - b) / d) % 6);
  else if (max === g) h = 60 * ((b - r) / d + 2);
  else h = 60 * ((r - g) / d + 4);
  if (h < 0) h += 360;
  return { h, s, l };
}

/** True when two colors would visually merge in a caption line. */
export function colorsClash(a, b) {
  const ha = hexToHsl(a);
  const hb = hexToHsl(b);
  if (!ha || !hb) return false;
  // Two saturated colors clash when their hues are close; a desaturated /
  // near-white pair clashes when both are bright.
  const hueDiff = Math.min(Math.abs(ha.h - hb.h), 360 - Math.abs(ha.h - hb.h));
  if (ha.s > 0.35 && hb.s > 0.35 && hueDiff < 40) return true;
  if (ha.s <= 0.2 && hb.s <= 0.2 && Math.abs(ha.l - hb.l) < 0.25) return true;
  return false;
}

/**
 * The highlight color to use for a word spoken by a speaker whose caption
 * color is `speakerColor`. Returns `preferred` unless it clashes; then the
 * first non-clashing fallback.
 */
export function resolveActiveWordColor(preferred, speakerColor) {
  const pref = preferred || FALLBACKS[0];
  if (!speakerColor || !colorsClash(pref, speakerColor)) return pref;
  for (const c of FALLBACKS) {
    if (c.toLowerCase() !== pref.toLowerCase() && !colorsClash(c, speakerColor)) {
      return c;
    }
  }
  return '#FFFFFF';
}
