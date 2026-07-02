/**
 * Feature parity matrix — every user-settable NLE property and its
 * render status across the three paths:
 *
 *   preview       — RenderEngine.renderFrame() / SubtitleOverlay (DOM)
 *   clientExport  — ExportEngine (WebCodecs, same renderFrame) + offline audio
 *   serverExport  — backend clip_exporter / ffmpeg_filter_builder (FFmpeg)
 *
 * Statuses:
 *   'ok'      — renders/applies on this path
 *   'partial' — applies with a documented caveat (see notes)
 *   'gap'     — does NOT apply on this path (known divergence)
 *   'na'      — not applicable to this path
 *
 * This file is DATA, consumed by:
 *   - featureParityMatrix.test.js  (structure + coverage assertions)
 *   - scripts/generate-parity-checklist.mjs → docs/parity-checklist.md
 *
 * When you wire a property into a new path, update the status here and
 * regenerate the checklist. The test fails if a panel exposes a property
 * that has no row here.
 */

export const FEATURE_PARITY = [
  // ── Clip basics (PropertiesPanel) ────────────────────────────────
  { key: 'start', label: 'Clip start (timeline in-point)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'end', label: 'Clip end (timeline out-point)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'opacity', label: 'Clip opacity', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'volume', label: 'Clip volume', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'muted', label: 'Clip / track audio mute', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Client export honors clip.muted and track.audioMuted since the decoded-buffer audio rewrite.' },
  { key: 'speed', label: 'Per-clip playback speed', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Client export maps time as trimStart+(t-start)*speed for frames and uses AudioBufferSourceNode.playbackRate for audio (varispeed: pitch shifts, matching preview element.playbackRate). Server uses setpts+atempo (pitch-preserving) — audible pitch differs from preview at speeds far from 1x.' },
  { key: 'fadeIn', label: 'Fade in (video opacity + audio gain)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Client export schedules matching gain automation on the offline audio context. Server applies visual fade via video_effects.fade_in; audio fade on the server path is not yet implemented.' },
  { key: 'fadeOut', label: 'Fade out (video opacity + audio gain)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Same as fadeIn.' },
  { key: 'position', label: 'Position (x/y %)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'size', label: 'Size (w/h %)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'transform.rotation', label: 'Rotation', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'transition', label: 'Clip transition (type + duration)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Canvas supports dissolve/fade/wipe-left/wipe-right/slide-left/slide-right/zoom; the FFmpeg path inserts xfade=transition=fade only — other transition types render as fade on server export. Easing also differs (canvas cubic-bezier vs xfade linear).' },

  // ── Effects (EffectsPanel) ───────────────────────────────────────
  { key: 'effects.brightness', label: 'Brightness', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Canvas paths use ctx.filter, unsupported in Safari — RenderEngine.supportsCanvasFilter() detects this and warns instead of silently rendering unfiltered.' },
  { key: 'effects.contrast', label: 'Contrast', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Same Safari caveat.' },
  { key: 'effects.saturation', label: 'Saturation', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Same Safari caveat.' },
  { key: 'effects.blur', label: 'Blur', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Same Safari caveat.' },
  { key: 'effects.hueRotate', label: 'Hue rotate', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Same Safari caveat.' },
  { key: 'effects.sepia', label: 'Sepia', panel: 'EffectsPanel',
    preview: 'partial', clientExport: 'partial', serverExport: 'ok',
    notes: 'Same Safari caveat.' },

  // ── Text / shapes ────────────────────────────────────────────────
  { key: 'textContent', label: 'Text overlay content', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'textStyle', label: 'Text style (font/size/weight/color/outline/shadow/bg/align/animation)', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Server drawtext font metrics and line breaking differ slightly from canvas; keep text inside the safe area. Fonts must be uploaded (not blob:) to reach server export.' },
  { key: 'shapeType', label: 'Shape overlays', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleText', label: 'Edited subtitle text/timing', panel: 'PropertiesPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Server export receives edited_subtitle_segments from the timeline store.' },

  // ── Subtitle styling (ClipSettingsPanel) ─────────────────────────
  { key: 'subtitlesEnabled', label: 'Subtitles on/off (global + per-segment)', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleFont', label: 'Subtitle font', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'libass font metrics / line-break points can differ from canvas by a few px; active-word timing constants are shared (utils/activeWordTiming.js ↔ ass_generator.py net offset).' },
  { key: 'subtitleSize', label: 'Subtitle size', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleFontWeight', label: 'Subtitle weight', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleFontColor', label: 'Subtitle color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitlePosition', label: 'Subtitle position preset', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleOffsetV', label: 'Subtitle vertical offset', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleMaxWidth', label: 'Subtitle max width', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleMaxWords', label: 'Max words per cue', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleOutlineWidth', label: 'Subtitle outline width', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleOutlineColor', label: 'Subtitle outline color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleOutlineOpacity', label: 'Subtitle outline opacity', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleBgEnabled', label: 'Subtitle background', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleBgColor', label: 'Subtitle background color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleBgOpacity', label: 'Subtitle background opacity', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'subtitleBgRadius', label: 'Subtitle background radius', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'ASS BorderStyle=4 boxes are rectangular; radius is approximated server-side.' },
  { key: 'useSpeakerColors', label: 'Per-speaker colors', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Palette shared via utils/activeWordTiming.js DEFAULT_SPEAKER_PALETTE; explicit speakerColors map forwarded in subtitle_settings.' },
  { key: 'speakerColors', label: 'Explicit speaker color map', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'activeWordEnabled', label: 'Active-word highlighting', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Client export boosts FPS (computeOptimalFPS) for smooth word transitions; word-index algorithm is the shared module.' },
  { key: 'activeWordColor', label: 'Active word color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'activeWordBgColor', label: 'Active word background color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'activeWordBgOpacity', label: 'Active word background opacity', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'activeWordBgRadius', label: 'Active word background radius', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'partial',
    notes: 'Same ASS box-radius approximation as subtitleBgRadius.' },
  { key: 'activeWordOutlineColor', label: 'Active word outline color', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },

  // ── Layout / reframing ───────────────────────────────────────────
  { key: 'layoutMode', label: 'Layout mode (auto/single/split)', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok' },
  { key: 'gamingLayoutMode', label: 'Gaming layouts (fullscreen/blurfill/wide_zoom)', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Blurfill blur now matches server exactly: CSS blur(50px) brightness(0.9) ↔ gblur=sigma=50,eq=brightness=-0.1.' },
  { key: 'subjectTracking', label: 'Subject tracking crop (keyframes)', panel: 'ClipSettingsPanel',
    preview: 'ok', clientExport: 'ok', serverExport: 'ok',
    notes: 'Same keyframe pipeline injected via RenderEngine.setSubjectTrackingFns; server renders the RenderPlan.' },
];

/**
 * Panel-exposed keys that MUST have a row above. The parity test
 * cross-checks this list against the matrix so newly added panel
 * properties can't silently skip the checklist.
 */
export const REQUIRED_KEYS = [
  'start', 'end', 'opacity', 'volume', 'muted', 'speed', 'fadeIn', 'fadeOut',
  'position', 'size', 'transform.rotation', 'transition',
  'effects.brightness', 'effects.contrast', 'effects.saturation',
  'effects.blur', 'effects.hueRotate', 'effects.sepia',
  'textContent', 'textStyle', 'shapeType', 'subtitleText',
  'subtitlesEnabled', 'subtitleFont', 'subtitleSize', 'subtitleFontWeight',
  'subtitleFontColor', 'subtitlePosition', 'subtitleOffsetV',
  'subtitleMaxWidth', 'subtitleMaxWords',
  'subtitleOutlineWidth', 'subtitleOutlineColor', 'subtitleOutlineOpacity',
  'subtitleBgEnabled', 'subtitleBgColor', 'subtitleBgOpacity', 'subtitleBgRadius',
  'useSpeakerColors', 'speakerColors',
  'activeWordEnabled', 'activeWordColor', 'activeWordBgColor',
  'activeWordBgOpacity', 'activeWordBgRadius', 'activeWordOutlineColor',
  'layoutMode', 'gamingLayoutMode', 'subjectTracking',
];

export const VALID_STATUSES = new Set(['ok', 'partial', 'gap', 'na']);
