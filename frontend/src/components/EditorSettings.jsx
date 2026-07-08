import React, { useMemo, useCallback } from 'react';
import { DEFAULT_CLIP_SETTINGS, SUBTITLE_RANGES, offsetVFromPosition } from '../utils/defaultSettings';

/**
 * EditorSettings — the ClipAI Editor pane's Subtitles / Highlights / Layout
 * tabs. A 1:1 port of the mockup's collapsible-section settings UI, wired
 * directly to the live clip `settings` object (not a private copy) so every
 * change reflects in the preview immediately. Uses the shared `.cae-*`
 * control primitives so hover/active/focus feel snappy.
 *
 * Every setting key in DEFAULT_CLIP_SETTINGS is represented here — the
 * redesign adds settings *into* the editor without removing any.
 */

const BUILTIN_FONTS = [
  'DM Sans', 'Inter', 'Montserrat', 'Poppins', 'Open Sans', 'Roboto',
  'Nunito', 'Lato', 'Oswald', 'Bebas Neue', 'Playfair Display',
  'Liberation Sans', 'Liberation Serif', 'DejaVu Sans Mono',
];

const SWATCH_COLORS = ['#FFFFFF', '#FFC53D', '#6E7BFF', '#4DD0A6', '#FF7AA2', '#101013'];
const HL_COLORS = ['#FFC53D', '#6E7BFF', '#4DD0A6', '#FF7AA2', '#FFFFFF'];

const DEFAULT_SPEAKER_PALETTE = [
  '#00D9FF', '#F59E0B', '#10B981', '#A78BFA', '#EF4444', '#EC4899',
  '#06B6D4', '#8B5CF6', '#F97316', '#14B8A6', '#E879F9', '#84CC16',
];

// ── Small pointer-driven slider (matches the mockup's dragSlider) ──
function Slider({ value, min, max, step = 1, onChange }) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
  const onDown = useCallback((e) => {
    const rail = e.currentTarget;
    const rect = rail.getBoundingClientRect();
    const apply = (x) => {
      const p = Math.max(0, Math.min(1, (x - rect.left) / rect.width));
      let v = min + p * (max - min);
      v = Math.round(v / step) * step;
      v = Math.round(v * 1000) / 1000;
      onChange(v);
    };
    apply(e.clientX);
    const mv = (ev) => apply(ev.clientX);
    const up = () => {
      window.removeEventListener('pointermove', mv);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
  }, [min, max, step, onChange]);
  return (
    <div className="cae-slider" style={{ height: 22 }} onPointerDown={onDown}>
      <div className="cae-slider__rail" />
      <div className="cae-slider__fill" style={{ width: pct + '%' }} />
      <div className="cae-slider__knob" style={{ width: 16, height: 16, marginTop: -8, left: `calc(${pct}% - 8px)` }} />
    </div>
  );
}

function Swatch({ colors, value, onPick }) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {colors.map((c) => (
        <button
          key={c}
          type="button"
          aria-label={c}
          className={`cae-swatch${String(value).toUpperCase() === c.toUpperCase() ? ' is-on' : ''}`}
          style={{ background: c }}
          onClick={() => onPick(c)}
        />
      ))}
    </div>
  );
}

function Toggle({ checked, onToggle, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
      <span style={{ fontSize: 12, color: 'var(--cae-text-2)' }}>{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={checked ? 'true' : 'false'}
        aria-label={label}
        className={`cae-switch${checked ? ' is-on' : ''}`}
        onClick={onToggle}
      >
        <span className="cae-switch__knob" />
      </button>
    </div>
  );
}

function Pills({ options, value, onPick }) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {options.map((o) => {
        const v = typeof o === 'object' ? o.v : o;
        const t = typeof o === 'object' ? o.label : o;
        return (
          <button
            key={String(v)}
            type="button"
            className={`cae-pill${value === v ? ' is-on' : ''}`}
            onClick={() => onPick(v)}
          >
            {t}
          </button>
        );
      })}
    </div>
  );
}

function CtlRow({ label, value, children }) {
  return (
    <div>
      {(label != null) && (
        <div className="cae-ctl-label">
          <span className="cae-ctl-label__name">{label}</span>
          {value != null && <span className="cae-ctl-label__val">{value}</span>}
        </div>
      )}
      {children}
    </div>
  );
}

export default function EditorSettings({
  tab, settings, onChange, onBulkChange, speakers, speakerNames,
  search, collapsed, onToggleSection, onAspectRatioChange,
  onApply, onReset, onSetDefault,
}) {
  const s = { ...DEFAULT_CLIP_SETTINGS, ...(settings || {}) };
  const set = (k, v) => onChange(k, v);

  // Live caption sample — mirrors the preview compositor closely enough to
  // preview font / size / color / background / outline choices.
  const sample = useMemo(() => {
    const outline = s.subtitleOutlineWidth || 0;
    const shadow = outline > 0
      ? `0 0 ${outline}px rgba(0,0,0,0.95), 0 0 ${outline}px rgba(0,0,0,0.95), 0 ${Math.max(1, Math.round(outline / 2))}px ${outline * 2}px rgba(0,0,0,0.6)`
      : 'none';
    const box = {
      display: 'inline-flex', gap: '0.3em', padding: '0.32em 0.55em',
      borderRadius: Math.round((s.subtitleBgRadius || 0) * 0.5),
      background: s.subtitleBgEnabled ? `rgba(6,6,9,${(s.subtitleBgOpacity ?? 55) / 100})` : 'transparent',
      textShadow: shadow, fontWeight: s.subtitleFontWeight || 700,
      fontFamily: `"${s.subtitleFont}", -apple-system, sans-serif`,
      fontSize: Math.max(12, Math.round((s.subtitleSize || 30) * 0.5)), lineHeight: 1.2,
      letterSpacing: '0.01em', whiteSpace: 'nowrap',
    };
    const words = ['Great', 'clips', 'start', 'with', 'captions'];
    const active = 2;
    return { box, words, active };
  }, [s.subtitleFont, s.subtitleSize, s.subtitleFontWeight, s.subtitleBgEnabled, s.subtitleBgOpacity, s.subtitleBgRadius, s.subtitleOutlineWidth]);

  // ── Section definitions per tab ──
  const R = SUBTITLE_RANGES;
  const sections = useMemo(() => {
    if (tab === 'subtitles') {
      return [
        {
          id: 'text', title: 'Text', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <CtlRow label="Font">
                <select
                  className="cae-input"
                  value={s.subtitleFont}
                  onChange={(e) => set('subtitleFont', e.target.value)}
                  style={{ cursor: 'pointer' }}
                >
                  {BUILTIN_FONTS.map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </CtlRow>
              <CtlRow label="Size" value={`${Math.round(s.subtitleSize)}px`}>
                <Slider value={s.subtitleSize} min={R.size.min} max={R.size.max} step={R.size.step} onChange={(v) => set('subtitleSize', v)} />
              </CtlRow>
              <CtlRow label="Weight">
                <Pills options={[{ v: 400, label: 'Regular' }, { v: 700, label: 'Bold' }, { v: 900, label: 'Black' }]} value={s.subtitleFontWeight} onPick={(v) => set('subtitleFontWeight', v)} />
              </CtlRow>
              <CtlRow label="Color">
                <Swatch colors={SWATCH_COLORS} value={s.subtitleFontColor} onPick={(c) => set('subtitleFontColor', c)} />
              </CtlRow>
            </div>
          ),
        },
        {
          id: 'position', title: 'Position', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <CtlRow label="Placement">
                <Pills
                  options={[{ v: 'top', label: 'Top' }, { v: 'center', label: 'Middle' }, { v: 'bottom', label: 'Bottom' }]}
                  value={s.subtitlePosition}
                  onPick={(v) => { onBulkChange ? onBulkChange({ subtitlePosition: v, subtitleOffsetV: offsetVFromPosition(v) }) : set('subtitlePosition', v); }}
                />
              </CtlRow>
              <CtlRow label="Vertical offset" value={`${Math.round(s.subtitleOffsetV)}%`}>
                <Slider value={s.subtitleOffsetV} min={R.offsetV.min} max={R.offsetV.max} step={R.offsetV.step} onChange={(v) => set('subtitleOffsetV', v)} />
              </CtlRow>
              <CtlRow label="Max width" value={`${Math.round(s.subtitleMaxWidth)}%`}>
                <Slider value={s.subtitleMaxWidth} min={R.maxWidth.min} max={R.maxWidth.max} step={R.maxWidth.step} onChange={(v) => set('subtitleMaxWidth', v)} />
              </CtlRow>
              <CtlRow label="Max words / line" value={s.subtitleMaxWords === 0 ? 'Auto' : String(s.subtitleMaxWords)}>
                <Slider value={s.subtitleMaxWords} min={R.maxWords.min} max={R.maxWords.max} step={R.maxWords.step} onChange={(v) => set('subtitleMaxWords', v)} />
              </CtlRow>
            </div>
          ),
        },
        {
          id: 'background', title: 'Background', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <Toggle checked={!!s.subtitleBgEnabled} onToggle={() => set('subtitleBgEnabled', !s.subtitleBgEnabled)} label="Show background box" />
              <CtlRow label="Color"><Swatch colors={['#000000', '#101013', '#1D1D21', '#6E7BFF']} value={s.subtitleBgColor} onPick={(c) => set('subtitleBgColor', c)} /></CtlRow>
              <CtlRow label="Opacity" value={`${Math.round(s.subtitleBgOpacity)}%`}>
                <Slider value={s.subtitleBgOpacity} min={R.bgOpacity.min} max={R.bgOpacity.max} step={R.bgOpacity.step} onChange={(v) => set('subtitleBgOpacity', v)} />
              </CtlRow>
              <CtlRow label="Corner radius" value={`${Math.round(s.subtitleBgRadius)}px`}>
                <Slider value={s.subtitleBgRadius} min={R.bgRadius.min} max={R.bgRadius.max} step={R.bgRadius.step} onChange={(v) => set('subtitleBgRadius', v)} />
              </CtlRow>
            </div>
          ),
        },
        {
          id: 'outline', title: 'Outline', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <CtlRow label="Width" value={`${Math.round(s.subtitleOutlineWidth)}px`}>
                <Slider value={s.subtitleOutlineWidth} min={R.outlineWidth.min} max={R.outlineWidth.max} step={R.outlineWidth.step} onChange={(v) => set('subtitleOutlineWidth', v)} />
              </CtlRow>
              <CtlRow label="Color"><Swatch colors={['#000000', '#101013', '#6E7BFF', '#FFFFFF']} value={s.subtitleOutlineColor} onPick={(c) => set('subtitleOutlineColor', c)} /></CtlRow>
              <CtlRow label="Opacity" value={`${Math.round(s.subtitleOutlineOpacity)}%`}>
                <Slider value={s.subtitleOutlineOpacity} min={R.outlineOpacity.min} max={R.outlineOpacity.max} step={R.outlineOpacity.step} onChange={(v) => set('subtitleOutlineOpacity', v)} />
              </CtlRow>
            </div>
          ),
        },
      ];
    }
    if (tab === 'highlights') {
      return [
        {
          id: 'active', title: 'Active word', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <Toggle checked={!!s.activeWordEnabled} onToggle={() => set('activeWordEnabled', !s.activeWordEnabled)} label="Highlight active word" />
              <CtlRow label="Highlight color"><Swatch colors={HL_COLORS} value={s.activeWordColor} onPick={(c) => set('activeWordColor', c)} /></CtlRow>
              <CtlRow label="Background color"><Swatch colors={['#000000', '#6E7BFF', '#FFC53D', '#101013']} value={s.activeWordBgColor} onPick={(c) => set('activeWordBgColor', c)} /></CtlRow>
              <CtlRow label="Background opacity" value={`${Math.round(s.activeWordBgOpacity)}%`}>
                <Slider value={s.activeWordBgOpacity} min={0} max={100} step={1} onChange={(v) => set('activeWordBgOpacity', v)} />
              </CtlRow>
              <CtlRow label="Background radius" value={`${Math.round(s.activeWordBgRadius)}px`}>
                <Slider value={s.activeWordBgRadius} min={0} max={24} step={1} onChange={(v) => set('activeWordBgRadius', v)} />
              </CtlRow>
              <CtlRow label="Outline color"><Swatch colors={['#000000', '#101013', '#6E7BFF', '#FFFFFF']} value={s.activeWordOutlineColor} onPick={(c) => set('activeWordOutlineColor', c)} /></CtlRow>
            </div>
          ),
        },
        {
          id: 'labels', title: 'Speaker labels', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
              <Toggle checked={!!s.showSpeakerLabels} onToggle={() => set('showSpeakerLabels', !s.showSpeakerLabels)} label="Show speaker name above cue" />
              <Toggle checked={!!s.useSpeakerColors} onToggle={() => set('useSpeakerColors', !s.useSpeakerColors)} label="Color captions per speaker" />
            </div>
          ),
        },
        {
          id: 'speakers', title: 'Speakers', render: () => (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(speakers || []).length === 0 && (
                <div style={{ fontSize: 11, color: 'var(--cae-text-3)' }}>No speakers detected for this clip.</div>
              )}
              {(speakers || []).map((spk, i) => {
                const color = s.speakerColors?.[spk] || DEFAULT_SPEAKER_PALETTE[i % DEFAULT_SPEAKER_PALETTE.length];
                const name = String(speakerNames?.[spk] || spk || `Speaker ${i + 1}`);
                return (
                  <div key={spk} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'var(--cae-surface)', border: '1px solid var(--cae-border-soft)', borderRadius: 8, padding: '7px 10px' }}>
                    <label style={{ width: 22, height: 22, borderRadius: '50%', background: color, border: '2px solid rgba(255,255,255,0.2)', cursor: 'pointer', flex: 'none', position: 'relative', overflow: 'hidden' }}>
                      <input type="color" value={color} onChange={(e) => set('speakerColors', { ...(s.speakerColors || {}), [spk]: e.target.value })} style={{ opacity: 0, position: 'absolute', inset: 0, cursor: 'pointer' }} />
                    </label>
                    <span style={{ fontSize: 12, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
                    <span style={{ fontSize: 10.5, color: 'var(--cae-text-3)', fontVariantNumeric: 'tabular-nums' }}>{String(color).toUpperCase()}</span>
                  </div>
                );
              })}
              <div style={{ fontSize: 10.5, color: 'var(--cae-text-3)' }}>Speaker colors apply to cue segments, labels, and highlights.</div>
            </div>
          ),
        },
      ];
    }
    // layout
    return [
      {
        id: 'aspect', title: 'Aspect ratio', render: () => (
          <Pills
            options={[{ v: null, label: 'Original' }, '16:9', '9:16', '1:1', '4:5']}
            value={s.aspectRatio ?? null}
            onPick={(v) => { set('aspectRatio', v); if (onAspectRatioChange) onAspectRatioChange(v); }}
          />
        ),
      },
      {
        id: 'quality', title: 'Quality', render: () => (
          <Pills options={['720p', '1080p', '4k']} value={s.exportQuality} onPick={(v) => set('exportQuality', v)} />
        ),
      },
      {
        id: 'speed', title: 'Speed', render: () => (
          <CtlRow label="Playback speed" value={`${s.playbackSpeed}×`}>
            <Slider value={s.playbackSpeed} min={0.25} max={4} step={0.05} onChange={(v) => set('playbackSpeed', v)} />
          </CtlRow>
        ),
      },
    ];
  }, [tab, s, speakers, speakerNames, onAspectRatioChange, onBulkChange]); // eslint-disable-line react-hooks/exhaustive-deps

  const q = (search || '').trim().toLowerCase();
  const visibleSections = q
    ? sections.filter((sec) => sec.title.toLowerCase().includes(q) || sec.id.includes(q))
    : sections;

  const showSample = tab === 'subtitles' || tab === 'highlights';

  return (
    <div>
      {showSample && (
        <div style={{ position: 'sticky', top: 0, zIndex: 2, background: 'var(--cae-panel)', padding: '2px 0 10px' }}>
          <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', background: 'repeating-linear-gradient(45deg,#101014 0 10px,#15151A 10px 20px)', height: 76, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--cae-border)' }}>
            <div style={sample.box}>
              {sample.words.map((w, i) => (
                <span key={i} style={{
                  color: s.activeWordEnabled && i === sample.active ? s.activeWordColor : s.subtitleFontColor,
                  transform: s.activeWordEnabled && i === sample.active ? 'scale(1.07)' : 'none',
                  display: 'inline-block',
                }}>{w}</span>
              ))}
            </div>
          </div>
          <div style={{ fontSize: 10, color: 'var(--cae-text-3)', marginTop: 5, textAlign: 'center' }}>Live sample — matches export</div>
        </div>
      )}

      {visibleSections.map((sec) => {
        const open = q ? true : !collapsed[sec.id];
        return (
          <div key={sec.id} className="cae-section">
            <button type="button" className="cae-section__head" aria-expanded={open ? 'true' : 'false'} onClick={() => onToggleSection(sec.id)}>
              {sec.title}
              <span className={`cae-section__chev${open ? ' is-open' : ''}`}>›</span>
            </button>
            {open && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 13, padding: '2px 0 13px' }}>
                {sec.render()}
              </div>
            )}
          </div>
        );
      })}

      {q && visibleSections.length === 0 && (
        <div style={{ padding: '24px 0', textAlign: 'center', fontSize: 12, color: 'var(--cae-text-3)' }}>No settings match “{search}”</div>
      )}

      <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
        <button type="button" className="cae-btn-primary" style={{ flex: 1, fontSize: 12, padding: '8px 0' }} onClick={onApply}>Apply settings</button>
        <button type="button" className="cae-btn-secondary" style={{ flex: 1, fontSize: 12, padding: '8px 0' }} onClick={onReset}>Reset</button>
      </div>
    </div>
  );
}
