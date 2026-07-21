import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import ExportEngine from '../engine/ExportEngine';
import useTimelineStore from '../stores/timelineStore';
import { runSubtitleQA } from '../utils/subtitleQA';
import { buildOverlayPayload, buildVideoEffectsPayload, buildPlaybackPayload, mapSubtitleSettings } from '../utils/buildExportPayload';
import { EXPORT_QUALITIES, EXPORT_QUALITY_PRESETS, getExportDims } from '../utils/defaultSettings';
import { PLATFORM_PRESETS } from '../utils/safeZones';
import { CloseIcon } from './icons';

// Derived from the shared quality tables in defaultSettings.js — the
// dialog can't drift from the panel or the backend dims again.
// Exported so exportQuality.test.js can assert the dialog offers exactly
// EXPORT_QUALITIES.
export const QUALITY_PRESETS = EXPORT_QUALITIES.map((id) => ({ id, ...EXPORT_QUALITY_PRESETS[id] }));

// Tiny labelled-number tile used by the cost/quota preview row.
function CostStat({ label, value, sub }) {
  return (
    <div className="ve-export-dialog__stat">
      <span className="ve-export-dialog__stat-label">{label}</span>
      <span className="ve-export-dialog__stat-value">{value}</span>
      {sub && <span className="ve-export-dialog__stat-sub">{sub}</span>}
    </div>
  );
}

export default function ExportDialog({
  onClose,
  onServerExport,
  renderEngine,
  tracks,
  clips,
  settings,
  mediaElements,
  startTime = 0,
  endTime = 0,
  aspectRatio,
  jobId,
  clipId,
  clipTitle,
  transcript,
  scenes,
  sourceWidth = 1920,
  sourceHeight = 1080,
  subjectX = 50,
  // The editor preview's subject-tracking keyframes ([{t, x}]) — shipped
  // with server exports so the render follows the exact camera path the
  // preview showed (no server-side re-derivation).
  subjectKeyframes = null,
  // Platform safe-zone preview: called with a profile name (or null) so
  // the parent viewport can shade the regions platform UI will cover.
  onSafeZonePreview,
}) {
  // Initialize from the parent's clip settings so the dialog and the
  // Settings panel stay in agreement. Default to ``1080p`` only when
  // the parent hasn't supplied a value.
  const [quality, setQuality] = useState(() => settings?.exportQuality || '1080p');
  // Re-sync if settings.exportQuality changes after mount (e.g. user
  // tweaked it in the side panel without closing the dialog).
  useEffect(() => {
    if (settings?.exportQuality && settings.exportQuality !== quality) {
      setQuality(settings.exportQuality);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings?.exportQuality]);
  const [exportMode, setExportMode] = useState('server'); // 'server' | 'client'
  const [isExporting, setIsExporting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [qaReport, setQaReport] = useState(null);
  const exportEngineRef = useRef(null);

  // Platform preset chips (TikTok / Reels / Shorts / YouTube): one click
  // sets aspect + quality for THIS export and offers the safe-zone
  // preview. The aspect override is export-local — it never mutates the
  // editor settings.
  const [platformId, setPlatformId] = useState(null);
  const [aspectOverride, setAspectOverride] = useState(null);
  const [showSafeZones, setShowSafeZones] = useState(false);
  const effectiveAspect = aspectOverride ?? aspectRatio;
  const activePlatform = PLATFORM_PRESETS.find((p) => p.id === platformId) || null;

  const selectPlatform = (preset) => {
    if (platformId === preset.id) {
      setPlatformId(null);
      setAspectOverride(null);
      setShowSafeZones(false);
      onSafeZonePreview?.(null);
      return;
    }
    setPlatformId(preset.id);
    setAspectOverride(preset.aspect);
    setQuality(preset.quality);
    if (showSafeZones) onSafeZonePreview?.(preset.profile);
  };

  const toggleSafeZones = () => {
    const next = !showSafeZones;
    setShowSafeZones(next);
    onSafeZonePreview?.(next && activePlatform ? activePlatform.profile : null);
  };

  // Clear the viewport overlay when the dialog unmounts
  useEffect(() => () => onSafeZonePreview?.(null), []); // eslint-disable-line react-hooks/exhaustive-deps

  const canClientExport = ExportEngine.isWebCodecsAvailable();

  // Run subtitle QA validation
  const timelineItems = useTimelineStore((s) => s.items);
  const timelineMediaLibrary = useTimelineStore((s) => s.mediaLibrary);

  // Preserve-pitch speed changes can't be rendered faithfully by the
  // browser path (AudioBufferSourceNode.playbackRate is varispeed-only,
  // and a phase-vocoder time-stretch in JS is out of scope) — those
  // exports are routed to the server.
  const needsServerForPitch = useMemo(
    () => !!buildPlaybackPayload(useTimelineStore.getState()).preserve_pitch,
    [timelineItems],
  );

  // Compute optimal FPS (may be boosted for active word highlighting)
  const exportFPS = useMemo(() => {
    return ExportEngine.computeOptimalFPS(timelineItems, settings, 30);
  }, [timelineItems, settings]);

  // ── Cost / quota preview ─────────────────────────────────────────
  // Pure-client estimate so the user knows what they're about to spend
  // before pressing "Export". Three signals:
  //   * Output file size (bitrate × duration)
  //   * Estimated render time (resolution + complexity multiplier)
  //   * Server-credit usage when the share/quota API exposes a budget
  // The resulting widget is informational; it never blocks the
  // export. ``costEstimate`` returns ``null`` when we can't form a
  // confident estimate (e.g. zero-length clip).
  const costEstimate = useMemo(() => {
    const preset = QUALITY_PRESETS.find((p) => p.id === quality) || QUALITY_PRESETS[1];
    const durationSec = Math.max(0, (endTime || 0) - (startTime || 0));
    if (!durationSec || durationSec < 0.1) return null;

    // File size: bitrate (bits/sec) × duration / 8  → bytes.
    const bytes = (preset.bitrate * durationSec) / 8;

    // Render-time heuristic. Server export ~= 0.5x realtime for 1080p,
    // scales linearly with pixel count and roughly 1.6x slower per
    // tier above 1080p. Browser export is ~2x slower because
    // WebCodecs has to draw + encode every frame in JS.
    const pixelScale = (preset.w * preset.h) / (1920 * 1080);
    const baseFactor = exportMode === 'client' ? 1.0 : 0.5;
    const renderSec = durationSec * baseFactor * pixelScale * (exportFPS / 30);

    // Credits — only meaningful for server export. Treats every minute
    // of exported footage as 1 credit at 1080p, scaling linearly with
    // pixel count. The server may rate-limit further.
    const credits = exportMode === 'server'
      ? Math.max(1, Math.ceil((durationSec / 60) * pixelScale))
      : 0;

    return {
      bytes,
      renderSec,
      credits,
      preset,
      durationSec,
    };
  }, [quality, exportMode, startTime, endTime, exportFPS]);

  const formatBytes = (n) => {
    if (n == null) return '—';
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${Math.round(n)} B`;
  };
  const formatTime = (s) => {
    if (s == null || !Number.isFinite(s)) return '—';
    if (s < 60) return `${Math.max(1, Math.round(s))}s`;
    const m = Math.floor(s / 60);
    const r = Math.round(s - m * 60);
    return r === 0 ? `${m}m` : `${m}m ${r}s`;
  };

  const subtitleQA = useMemo(() => {
    const { w: exportW, h: exportH } = getExportDims(quality, effectiveAspect);
    const syncInfo = transcript ? { transcript, clipStart: startTime, clipEnd: endTime } : undefined;
    const trackingInfo = scenes?.length ? {
      scenes, clipStart: startTime, clipEnd: endTime,
      srcW: sourceWidth, srcH: sourceHeight, subjectX,
    } : null;
    return runSubtitleQA(timelineItems, settings, { w: exportW, h: exportH }, syncInfo, exportFPS, trackingInfo, effectiveAspect);
  }, [timelineItems, settings, quality, effectiveAspect, transcript, startTime, endTime, exportFPS, scenes, sourceWidth, sourceHeight, subjectX]);

  // Pre-export checklist: overlay payload warnings (media not uploaded,
  // blob URLs, …) surfaced as checklist rows instead of a toast at
  // export time.
  const overlayWarnings = useMemo(() => {
    try {
      return buildOverlayPayload({
        timelineItems,
        mediaLibrary: timelineMediaLibrary,
        clipStart: startTime,
        tracks: useTimelineStore.getState().tracks,
      }).warnings;
    } catch { return []; }
  }, [timelineItems, timelineMediaLibrary, startTime]);

  const handleExport = useCallback(async () => {
    setError(null);
    setQaReport(subtitleQA);

    // Block export if subtitle QA has errors
    if (!subtitleQA.valid) {
      setError(`Export blocked: ${subtitleQA.errors.join('; ')}`);
      return;
    }

    if (exportMode === 'server') {
      const preset = QUALITY_PRESETS.find(p => p.id === quality) || QUALITY_PRESETS[1];
      const exportPayload = {
        start: startTime,
        end: endTime,
        clip_id: parseInt(clipId) || 0,
        export_quality: preset.id,
        clip_title: clipTitle || undefined,
      };

      // Map camelCase clipSettings → snake_case backend fields (not a blind spread)
      if (settings) {
        if (aspectOverride || settings.aspectRatio) {
          // Platform chip override wins for this export only
          exportPayload.aspect_ratio = aspectOverride || settings.aspectRatio;
        }
        const globalSubsOn = settings.subtitlesEnabled || false;
        // Check if any segment has per-segment subtitle overrides
        const storeStateForSubs = useTimelineStore.getState();
        const anySegmentSubsOn = storeStateForSubs.segments?.some(s => s.subtitlesEnabled !== false) || false;
        const needsSubtitleSettings = globalSubsOn || anySegmentSubsOn;

        exportPayload.subtitles_enabled = needsSubtitleSettings;
        exportPayload.global_subtitles_enabled = globalSubsOn;
        // ALWAYS send subtitle_settings when ANY subtitle rendering is needed
        // (global ON, or any per-segment override ON). Without settings, the
        // backend uses hardcoded defaults that won't match the preview.
        if (needsSubtitleSettings) {
          exportPayload.subtitle_settings = mapSubtitleSettings(settings);
        }
      }

      // Pull volume, speed (+ preserve_pitch), trim, segments from the
      // timeline store via the shared payload builder.
      const storeState = useTimelineStore.getState();
      Object.assign(exportPayload, buildPlaybackPayload(storeState));
      if (storeState.trimStartOffset > 0) exportPayload.trim_start_offset = storeState.trimStartOffset;
      if (storeState.trimEndOffset > 0) exportPayload.trim_end_offset = storeState.trimEndOffset;
      if (storeState.segments?.length > 0) {
        exportPayload.segments = storeState.segments.map(s => ({
          start: s.start, end: s.end,
          volume: (s.muted ? 0 : s.volume) / 100,
          muted: s.muted,
          subtitles_enabled: s.subtitlesEnabled,
          subject_tracking_enabled: s.subjectTrackingEnabled !== false,
          speed: s.speed || 1.0,
        }));
      }

      // Send user-edited subtitle timing from the timeline store so the
      // export uses the actual item durations (which may have been resized).
      const subtitleItemsForExport = timelineItems
        .filter(it => it.type === 'subtitle')
        .sort((a, b) => a.start - b.start)
        .map(it => ({
          start: it.start + startTime,
          end: it.end + startTime,
          text: it.subtitleText || '',
          speaker: it.speaker || '',
          words: it.words ? it.words.map(w => ({
            start: (w.start || 0) + startTime,
            end: (w.end || 0) + startTime,
            word: w.text || w.word || '',
          })) : null,
        }));
      if (subtitleItemsForExport.length > 0) {
        exportPayload.edited_subtitle_segments = subtitleItemsForExport;
      }

      // Video effects + transform from multi-track editor
      const videoEffects = buildVideoEffectsPayload(timelineItems);
      if (videoEffects) exportPayload.video_effects = videoEffects;

      // Build overlay arrays via shared utility (consistent filtering + validation)
      const overlays = buildOverlayPayload({
        timelineItems,
        mediaLibrary: timelineMediaLibrary,
        clipStart: startTime,
        tracks: useTimelineStore.getState().tracks,
      });
      if (overlays.textOverlays.length > 0) exportPayload.text_overlays = overlays.textOverlays;
      if (overlays.imageOverlays.length > 0) exportPayload.image_overlays = overlays.imageOverlays;
      if (overlays.shapeOverlays.length > 0) exportPayload.shape_overlays = overlays.shapeOverlays;
      if (overlays.audioOverlays.length > 0) exportPayload.audio_overlays = overlays.audioOverlays;
      if (overlays.compositingOrder?.length > 0) {
        exportPayload.overlay_compositing_order = overlays.compositingOrder;
        console.log('[ExportDialog] Compositing order:',
          overlays.compositingOrder.map(e =>
            `${e.type}(${String(e.id).slice(-8)}) prio=${e.compositing_priority}`
          ).join(' → ')
        );
      }
      if (exportPayload.image_overlays?.length) {
        console.log('[ExportDialog] image_overlays order:',
          exportPayload.image_overlays.map(i => `img(${String(i.item_id).slice(-8)})`).join(' → ')
        );
      }
      if (exportPayload.shape_overlays?.length) {
        console.log('[ExportDialog] shape_overlays order:',
          exportPayload.shape_overlays.map(s => `${s.shape_type}(${String(s.item_id).slice(-8)})`).join(' → ')
        );
      }

      if (overlays.warnings.length > 0) {
        for (const w of overlays.warnings) console.warn(`[Export] ${w}`);
        setError(`Warning: ${overlays.warnings.length} overlay(s) skipped from export — media files not yet uploaded. The export will proceed without them.`);
      }

      // Layout mode for multi-speaker reframing
      if (settings?.layoutMode && settings.layoutMode !== 'auto') {
        exportPayload.layout_mode = settings.layoutMode;
      } else {
        exportPayload.layout_mode = 'auto';
      }

      // Preview-export parity: ship the EXACT keyframes the editor preview
      // interpolates (hold-until-next) so the server renders the same camera
      // path instead of re-deriving its own (which panned/swayed where the
      // preview held still).
      if (exportPayload.aspect_ratio && subjectKeyframes?.length > 0) {
        exportPayload.subject_keyframes = subjectKeyframes.map(
          (kf) => ({ time: +(+kf.t).toFixed(3), x: kf.x,
                     ...(kf.snap ? { snap: true } : {}) }));
      }

      // Diagnostic logging: full export payload for debugging overlay/settings issues
      console.log('[ExportDialog] Export payload:', JSON.stringify({
        clip_id: exportPayload.clip_id,
        start: exportPayload.start,
        end: exportPayload.end,
        aspect_ratio: exportPayload.aspect_ratio,
        subtitles_enabled: exportPayload.subtitles_enabled,
        subtitle_settings: exportPayload.subtitle_settings ? 'YES' : 'NO',
        video_effects: exportPayload.video_effects ? 'YES' : 'NO',
        volume: exportPayload.volume,
        speed: exportPayload.speed,
        trim: [exportPayload.trim_start_offset || 0, exportPayload.trim_end_offset || 0],
        segments: exportPayload.segments?.length || 0,
        text_overlays: exportPayload.text_overlays?.length || 0,
        image_overlays: exportPayload.image_overlays?.length || 0,
        shape_overlays: exportPayload.shape_overlays?.length || 0,
        audio_overlays: exportPayload.audio_overlays?.length || 0,
        timelineItems_total: timelineItems.length,
        timelineItems_types: [...new Set(timelineItems.map(it => it.type))],
      }));
      if (exportPayload.text_overlays?.length) {
        for (const t of exportPayload.text_overlays) {
          console.log(`[ExportDialog] Text overlay: "${(t.text || '').slice(0, 30)}" at (${t.x}%, ${t.y}%) t=${t.start_time}-${t.end_time}s font=${t.font_family}`);
        }
      }

      if (onServerExport) {
        onServerExport(exportPayload);
      } else if (jobId && clipId) {
        fetch(`/api/jobs/${jobId}/export-clip`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(exportPayload),
        }).catch(() => {});
      }
      onClose?.();
      return;
    }

    // Client-side export
    if (needsServerForPitch) {
      setError('This clip uses a preserve-pitch speed change, which browser export cannot render faithfully. Switched to server export — press Export again.');
      setExportMode('server');
      return;
    }
    if (!renderEngine) {
      setError('Client-side export is not available in this browser. Use server export.');
      return;
    }

    setIsExporting(true);
    setProgress(0);

    const preset = QUALITY_PRESETS.find(p => p.id === quality) || QUALITY_PRESETS[1];
    // Shared quality × aspect table — identical pixel dims to server export.
    const { w: exportW, h: exportH } = getExportDims(quality, effectiveAspect);

    const engine = new ExportEngine(renderEngine, {
      fps: exportFPS,
      videoBitrate: preset.bitrate,
      width: exportW,
      height: exportH,
      onProgress: setProgress,
      onError: (msg) => { setError(msg); setIsExporting(false); },
      // WebCodecs missing → MediaRecorder records in real time with
      // possible dropped frames / word-highlight drift. Ask first.
      onFallbackRequired: () => window.confirm(
        'WebCodecs is not available in this browser. Continue with a ' +
        'reduced-fidelity real-time export? (May drop frames; audio is ' +
        'not included. Server export is recommended.)'
      ),
      onComplete: (blob, meta) => {
        if (meta) console.info('[ExportDialog] Export completed via', meta.path, meta);
        setIsExporting(false);
        setProgress(100);
        // Download the blob
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `export-${Date.now()}.${blob.type.includes('mp4') ? 'mp4' : 'webm'}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
      },
    });

    exportEngineRef.current = engine;

    // Ensure render engine is at export resolution
    renderEngine.setResolution(exportW, exportH);

    await engine.export(startTime, endTime, tracks, clips, settings, mediaElements);
  // Stale-closure fix: ``handleExport`` reads ``jobId`` / ``clipId``
  // / ``clipTitle`` / ``timelineMediaLibrary`` / ``transcript`` /
  // ``exportFPS`` / ``scenes`` / ``sourceWidth`` / ``sourceHeight`` /
  // ``subjectX`` inside its body. Without these in the dep list,
  // navigating to a different clip while the dialog is mounted would
  // export the OLD clip's data.
  }, [
    exportMode, quality, renderEngine, tracks, clips, settings, mediaElements,
    startTime, endTime, aspectRatio, onServerExport, onClose, subtitleQA,
    jobId, clipId, clipTitle, timelineMediaLibrary, transcript, exportFPS,
    scenes, sourceWidth, sourceHeight, subjectX, needsServerForPitch,
    aspectOverride, effectiveAspect, subjectKeyframes,
  ]);

  const handleCancel = useCallback(() => {
    if (exportEngineRef.current) {
      exportEngineRef.current.cancel();
    }
    setIsExporting(false);
    setProgress(0);
  }, []);

  return (
    <div
      className="ve-export-backdrop"
      onPointerDown={(e) => { if (e.target === e.currentTarget && onClose) onClose(); }}
    >
    <div className="ve-export-dialog" onClick={(e) => e.stopPropagation()}>
      <div className="ve-export-dialog__header">
        <span className="ve-export-dialog__title">Export video</span>
        <button className="ve-export-dialog__close" onClick={onClose} aria-label="Close export dialog"><CloseIcon /></button>
      </div>

      {/* Export mode toggle */}
      <div className="ve-export-dialog__mode">
        <button
          className={`ve-export-dialog__mode-btn${exportMode === 'server' ? ' ve-export-dialog__mode-btn--active' : ''}`}
          onClick={() => setExportMode('server')}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="2" width="20" height="8" rx="2" />
            <rect x="2" y="14" width="20" height="8" rx="2" />
            <circle cx="6" cy="6" r="1" fill="currentColor" />
            <circle cx="6" cy="18" r="1" fill="currentColor" />
          </svg>
          Server export
        </button>
        <button
          className={`ve-export-dialog__mode-btn${exportMode === 'client' ? ' ve-export-dialog__mode-btn--active' : ''}`}
          onClick={() => setExportMode('client')}
          disabled={!canClientExport}
          aria-label={canClientExport ? 'Export in browser' : 'WebCodecs not available in this browser'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          Browser export {!canClientExport && '(unavailable)'}
        </button>
      </div>

      {/* Quality picker */}
      <div className="ve-export-dialog__quality">
        <label className="ve-export-dialog__label">Quality</label>
        <div className="ve-export-dialog__quality-pills">
          {QUALITY_PRESETS.map((p) => (
            <button
              key={p.id}
              className={`ve-export-dialog__quality-pill${quality === p.id ? ' ve-export-dialog__quality-pill--active' : ''}`}
              onClick={() => setQuality(p.id)}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Platform presets: aspect + quality + safe-zone preview */}
        <div className="ve-export-dialog__platforms">
          {PLATFORM_PRESETS.map((p) => (
            <button
              key={p.id}
              className={`ve-export-dialog__platform-chip${platformId === p.id ? ' ve-export-dialog__platform-chip--active' : ''}`}
              onClick={() => selectPlatform(p)}
              aria-pressed={platformId === p.id}
            >
              {p.label}
            </button>
          ))}
          {activePlatform && (
            <button
              className={`ve-export-dialog__safezone-toggle${showSafeZones ? ' ve-export-dialog__safezone-toggle--active' : ''}`}
              onClick={toggleSafeZones}
              aria-pressed={showSafeZones}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              Show safe zones
            </button>
          )}
        </div>
        {activePlatform && aspectOverride && aspectOverride !== aspectRatio && (
          <p className="ve-export-dialog__notice ve-export-dialog__notice--small">
            Exporting as {aspectOverride} for {activePlatform.label} (editor aspect unchanged).
          </p>
        )}
      </div>

      {/* Cost / quota preview ─ informational only ─────────────── */}
      {costEstimate && (
        <div
          className={`ve-export-dialog__cost${exportMode === 'server' ? ' ve-export-dialog__cost--server' : ''}`}
        >
          <CostStat
            label="Estimated size"
            value={formatBytes(costEstimate.bytes)}
            sub={`${costEstimate.preset.label} \u00b7 ${formatTime(costEstimate.durationSec)}`}
          />
          <CostStat
            label={exportMode === 'client' ? 'Render time (browser)' : 'Render time (server)'}
            value={formatTime(costEstimate.renderSec)}
            sub={`@ ${exportFPS}fps`}
          />
          {exportMode === 'server' && (
            <CostStat
              label="Quota"
              value={`${costEstimate.credits} credit${costEstimate.credits === 1 ? '' : 's'}`}
              sub="server time"
            />
          )}
        </div>
      )}

      {/* Info */}
      <div className="ve-export-dialog__info">
        {exportMode === 'server' ? (
          <p>Export will be processed on the server using FFmpeg with full quality encoding.</p>
        ) : (
          <p>Export directly in your browser using WebCodecs. Faster for short clips, no server needed.</p>
        )}
        {needsServerForPitch && exportMode === 'client' && (
          <p className="ve-export-dialog__notice">
            This clip uses a preserve-pitch speed change — browser export renders audio varispeed only. Server export will be used for faithful pitch.
          </p>
        )}
        {exportFPS > 30 && (
          <p className="ve-export-dialog__fps-note">
            FPS boosted to {exportFPS}fps for smooth active word highlighting.
          </p>
        )}
      </div>

      {/* Progress */}
      {isExporting && (
        <div className="ve-export-dialog__progress">
          <div className="ve-export-dialog__progress-bar">
            <div
              className="ve-export-dialog__progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="ve-export-dialog__progress-text">{progress}%</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="ve-export-dialog__error">{error}</div>
      )}

      {/* Subtitle QA Report */}
      <div className={`ve-export-dialog__qa ${!subtitleQA.valid
        ? 've-export-dialog__qa--error'
        : subtitleQA.warnings.length > 0
          ? 've-export-dialog__qa--warn'
          : 've-export-dialog__qa--ok'}`}
      >
        <div className={`ve-export-dialog__qa-head${subtitleQA.errors.length + subtitleQA.warnings.length > 0 ? ' ve-export-dialog__qa-head--spaced' : ''}`}>
          <span className="ve-export-dialog__qa-icon">{subtitleQA.valid ? (subtitleQA.warnings.length > 0 ? '⚠' : '✓') : '✕'}</span>
          <span className="ve-export-dialog__qa-title">Export QA: {subtitleQA.summary}</span>
        </div>
        {subtitleQA.confidence && (
          <div className={`ve-export-dialog__qa-confidence ve-export-dialog__qa-confidence--${subtitleQA.confidence}`}>
            Preview-to-export match confidence: {subtitleQA.confidence}
            {subtitleQA.confidence === 'high' && ' — exported video will look exactly like preview'}
          </div>
        )}
        {/* Overlay payload pre-flight — media/upload issues that would
            otherwise only appear as a toast at export time */}
        {overlayWarnings.length > 0 && (
          <div className="ve-export-dialog__qa-check">
            <div className="ve-export-dialog__qa-check-head">
              <span className="ve-export-dialog__qa-check-icon ve-export-dialog__qa-check-icon--warn">⚠</span>
              <span className="ve-export-dialog__qa-check-name">Overlays</span>
            </div>
            {overlayWarnings.map((w, i) => (
              <div key={`ow-${i}`} className="ve-export-dialog__qa-issue ve-export-dialog__qa-issue--warn">• {w}</div>
            ))}
          </div>
        )}
        {/* Per-check breakdown */}
        {subtitleQA.checks && subtitleQA.checks.map((check, ci) => {
          const level = check.errors.length > 0 ? 'error' : check.warnings.length > 0 ? 'warn' : 'ok';
          const icon = check.errors.length > 0 ? '✕' : check.warnings.length > 0 ? '⚠' : '✓';
          return (
            <div key={ci} className="ve-export-dialog__qa-check">
              <div className="ve-export-dialog__qa-check-head">
                <span className={`ve-export-dialog__qa-check-icon ve-export-dialog__qa-check-icon--${level}`}>{icon}</span>
                <span className="ve-export-dialog__qa-check-name">{check.name}</span>
              </div>
              {check.errors.map((e, i) => (
                <div key={`ce-${ci}-${i}`} className="ve-export-dialog__qa-issue ve-export-dialog__qa-issue--error">• {e}</div>
              ))}
              {check.warnings.map((w, i) => (
                <div key={`cw-${ci}-${i}`} className="ve-export-dialog__qa-issue ve-export-dialog__qa-issue--warn">• {w}</div>
              ))}
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div className="ve-export-dialog__actions">
        {isExporting ? (
          <button className="ve-export-dialog__btn ve-export-dialog__btn--cancel" onClick={handleCancel}>
            Cancel
          </button>
        ) : (
          <>
            <button className="ve-export-dialog__btn ve-export-dialog__btn--secondary" onClick={onClose}>
              Cancel
            </button>
            <button className="ve-export-dialog__btn ve-export-dialog__btn--primary" onClick={handleExport}>
              Export
            </button>
          </>
        )}
      </div>
    </div>
    </div>
  );
}
