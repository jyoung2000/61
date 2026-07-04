/**
 * SafeZoneOverlay — shades the regions a platform's UI (TikTok action
 * rail, Reels caption bar, …) will cover, so users can see exactly
 * where their subtitles and overlays are safe. Toggled from the export
 * dialog's platform chips ("Show safe zones").
 */
import React from 'react';
import { getSafeZonePercents } from '../utils/safeZones';

export default function SafeZoneOverlay({ profile, videoWidth = 1080, videoHeight = 1920 }) {
  if (!profile) return null;
  const z = getSafeZonePercents(profile, videoWidth, videoHeight);

  const shade = { position: 'absolute', background: 'var(--overlay-heavy, rgba(0,0,0,0.55))', opacity: 0.55, pointerEvents: 'none' };

  return (
    <div className="ve-safezone" aria-hidden="true">
      {z.top > 0 && <div style={{ ...shade, top: 0, left: 0, right: 0, height: `${z.top}%` }} />}
      {z.bottom > 0 && <div style={{ ...shade, bottom: 0, left: 0, right: 0, height: `${z.bottom}%` }} />}
      {z.left > 0 && <div style={{ ...shade, top: `${z.top}%`, bottom: `${z.bottom}%`, left: 0, width: `${z.left}%` }} />}
      {z.right > 0 && <div style={{ ...shade, top: `${z.top}%`, bottom: `${z.bottom}%`, right: 0, width: `${z.right}%` }} />}
      <div
        className="ve-safezone__frame"
        style={{
          position: 'absolute',
          top: `${z.top}%`,
          bottom: `${z.bottom}%`,
          left: `${z.left}%`,
          right: `${z.right}%`,
        }}
      >
        <span className="ve-safezone__label">{profile} safe zone</span>
      </div>
    </div>
  );
}
