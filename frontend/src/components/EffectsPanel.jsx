import React, { useCallback } from 'react';
import useTimelineStore from '../stores/timelineStore';
import { undoCoalesceHandlers } from '../utils/undoCoalesce';
import ScrubInput, { InspectorRow } from './ScrubInput';

const EFFECT_CONTROLS = [
  { key: 'brightness', label: 'Brightness', min: -100, max: 100, step: 1, unit: '' },
  { key: 'contrast', label: 'Contrast', min: -100, max: 100, step: 1, unit: '' },
  { key: 'saturation', label: 'Saturation', min: -100, max: 100, step: 1, unit: '' },
  { key: 'blur', label: 'Blur', min: 0, max: 20, step: 0.5, unit: 'px' },
  { key: 'hueRotate', label: 'Hue rotate', min: 0, max: 360, step: 1, unit: '°' },
  { key: 'sepia', label: 'Sepia', min: 0, max: 100, step: 1, unit: '%' },
];

// Item-type defaults the reset dot compares against (an effect at its
// default renders identically to no effect on every path).
const EFFECT_DEFAULTS = {
  brightness: 0, contrast: 0, saturation: 0, blur: 0, hueRotate: 0, sepia: 0,
};

export default function EffectsPanel({ compact = false }) {
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const items = useTimelineStore((s) => s.items);
  const updateItem = useTimelineStore((s) => s.updateItem);

  const item = items.find((i) => i.id === selectedItemId) || null;

  const updateEffect = useCallback((key, value) => {
    if (!item) return;
    const effects = { ...(item.effects || {}), [key]: value };
    updateItem(item.id, { effects });
  }, [item, updateItem]);

  const resetAll = useCallback(() => {
    if (!item) return;
    updateItem(item.id, { effects: { ...EFFECT_DEFAULTS } });
  }, [item, updateItem]);

  if (!item || (item.type !== 'video' && item.type !== 'image' && item.type !== 'overlay')) {
    return (
      <div className="ve-effects ve-effects--empty">
        <span className="ve-effects__placeholder">
          Select a video or image clip to adjust effects
        </span>
      </div>
    );
  }

  const effects = item.effects || {};

  // Live slider fill — a computed percentage, not a styling constant
  // (documented dynamic-value exception to the no-inline-style rule).
  const fillStyle = (value, min, max) => ({
    background: `linear-gradient(to right, var(--ve-accent) ${
      ((value - min) / (max - min)) * 100
    }%, var(--ve-slider-track) ${
      ((value - min) / (max - min)) * 100
    }%)`,
  });

  return (
    <div className={`ve-effects${compact ? ' ve-effects--compact' : ''}`}>
      <div className="ve-effects__header">
        <span className="ve-effects__title">Effects</span>
        <button className="ve-effects__reset" onClick={resetAll}>
          Reset all
        </button>
      </div>

      {EFFECT_CONTROLS.map((ctrl) => {
        const value = effects[ctrl.key] ?? EFFECT_DEFAULTS[ctrl.key];
        return (
          <div key={ctrl.key} className="ve-effects__control">
            <InspectorRow
              label={ctrl.label}
              changed={value !== EFFECT_DEFAULTS[ctrl.key]}
              onReset={() => updateEffect(ctrl.key, EFFECT_DEFAULTS[ctrl.key])}
            >
              <ScrubInput
                value={value}
                min={ctrl.min}
                max={ctrl.max}
                step={ctrl.step}
                unit={ctrl.unit}
                defaultValue={EFFECT_DEFAULTS[ctrl.key]}
                onChange={(v) => updateEffect(ctrl.key, v)}
                ariaLabel={ctrl.label}
              />
            </InspectorRow>
            <input
              type="range"
              min={ctrl.min}
              max={ctrl.max}
              step={ctrl.step}
              value={value}
              onChange={(e) => updateEffect(ctrl.key, parseFloat(e.target.value))}
              {...undoCoalesceHandlers()}
              className="ve-effects__slider"
              aria-label={ctrl.label}
              style={fillStyle(value, ctrl.min, ctrl.max)}
            />
          </div>
        );
      })}

      {/* Opacity (applies to all visual items) */}
      <div className="ve-effects__control">
        <InspectorRow
          label="Opacity"
          changed={(item.opacity ?? 1) !== 1}
          onReset={() => updateItem(item.id, { opacity: 1 })}
        >
          <ScrubInput
            value={item.opacity ?? 1}
            min={0}
            max={1}
            step={0.01}
            defaultValue={1}
            onChange={(v) => updateItem(item.id, { opacity: v })}
            ariaLabel="Opacity"
          />
        </InspectorRow>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={item.opacity ?? 1}
          onChange={(e) => updateItem(item.id, { opacity: parseFloat(e.target.value) })}
          {...undoCoalesceHandlers()}
          className="ve-effects__slider"
          aria-label="Opacity"
          style={fillStyle(item.opacity ?? 1, 0, 1)}
        />
      </div>
    </div>
  );
}
