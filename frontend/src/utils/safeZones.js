/**
 * Platform safe zones — mirror of backend
 * subtitle_formatter._PLATFORM_PROFILES / get_safe_zone_margins().
 * Keep the two tables in sync (values are px at the 1080×1920 reference
 * and scale proportionally with the actual video size).
 */
export const PLATFORM_PROFILES = {
  tiktok:     { top: 140, bottom: 324, left: 60, right: 164, recommendedPosition: 'middle' },
  reels:      { top: 120, bottom: 350, left: 60, right: 120, recommendedPosition: 'middle' },
  shorts:     { top: 100, bottom: 280, left: 60, right: 100, recommendedPosition: 'middle' },
  horizontal: { top: 0,   bottom: 80,  left: 40, right: 40,  recommendedPosition: 'bottom' },
  square:     { top: 60,  bottom: 100, left: 40, right: 40,  recommendedPosition: 'bottom' },
};

const REF_W = 1080;
const REF_H = 1920;

/**
 * Export-dialog platform chips: one click sets aspect + quality and
 * names the safe-zone profile to preview.
 */
export const PLATFORM_PRESETS = [
  { id: 'tiktok', label: 'TikTok', aspect: '9:16', quality: '1080p', profile: 'tiktok' },
  { id: 'reels', label: 'Reels', aspect: '9:16', quality: '1080p', profile: 'reels' },
  { id: 'shorts', label: 'Shorts', aspect: '9:16', quality: '1080p', profile: 'shorts' },
  { id: 'youtube', label: 'YouTube', aspect: '16:9', quality: '1080p', profile: 'horizontal' },
];

/** Per-side pixel margins for a platform at the given output size. */
export function getSafeZoneMargins(profileName, videoWidth, videoHeight) {
  const profile = PLATFORM_PROFILES[(profileName || '').trim().toLowerCase()];
  if (!profile) {
    return { topPx: 0, bottomPx: 0, leftPx: 0, rightPx: 0, recommendedPosition: 'bottom' };
  }
  const scaleW = Math.max(0.1, videoWidth / REF_W);
  const scaleH = Math.max(0.1, videoHeight / REF_H);
  return {
    topPx: Math.round(profile.top * scaleH),
    bottomPx: Math.round(profile.bottom * scaleH),
    leftPx: Math.round(profile.left * scaleW),
    rightPx: Math.round(profile.right * scaleW),
    recommendedPosition: profile.recommendedPosition,
  };
}

/** Same margins expressed as % of the frame — what an overlay div needs. */
export function getSafeZonePercents(profileName, videoWidth, videoHeight) {
  const m = getSafeZoneMargins(profileName, videoWidth, videoHeight);
  const w = Math.max(1, videoWidth);
  const h = Math.max(1, videoHeight);
  return {
    top: (m.topPx / h) * 100,
    bottom: (m.bottomPx / h) * 100,
    left: (m.leftPx / w) * 100,
    right: (m.rightPx / w) * 100,
    recommendedPosition: m.recommendedPosition,
  };
}
