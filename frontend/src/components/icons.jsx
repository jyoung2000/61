/**
 * Shared editor icons — SVGs used by components OUTSIDE VideoEditor's
 * module graph (dialogs, overlays). Keeping them here avoids the
 * VideoEditor ↔ dialog import cycle that re-exporting VideoEditor's
 * local Icon set would create.
 */
import React from 'react';

export function CloseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}
