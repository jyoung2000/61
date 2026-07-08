import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { outlineTextShadow } from '../utils/textOutline';
import { spokenWindow, isSpokenAt } from '../utils/subtitleTiming';
import { DEFAULT_SPEAKER_PALETTE, computeSpeakerRates, getCurrentWordIndex } from '../utils/activeWordTiming';
import useTimelineStore from '../stores/timelineStore';
import { snapToGuides } from '../utils/goldenGrid';

// ── Backend-matching constants (ass_generator.py / clip_exporter.py) ──────
const ASPECT_RATIO_DIMS = {
  '16:9': [1920, 1080],
  '9:16': [1080, 1920],
  '1:1': [1080, 1080],
  '4:5': [1080, 1350],
};

const FONT_SIZE_MAP = { small: 22, medium: 30, large: 40 };
const REF_W = 1920;
const REF_H = 1080;

// ── Builtin font URL map (mirrors ClipSettingsPanel) ─────────────────────
const BUILTIN_FONT_FILES = {
  'DM Sans': '/api/fonts/builtin/DMSans.ttf',
  'Montserrat': '/api/fonts/builtin/Montserrat.ttf',
  'Open Sans': '/api/fonts/builtin/OpenSans.ttf',
  'Roboto': '/api/fonts/builtin/Roboto.ttf',
  'Poppins': '/api/fonts/builtin/Poppins-Regular.ttf',
  'Inter': '/api/fonts/builtin/Inter.ttf',
  'Nunito': '/api/fonts/builtin/Nunito.ttf',
  'Lato': '/api/fonts/builtin/Lato-Regular.ttf',
  'Oswald': '/api/fonts/builtin/Oswald.ttf',
  'Playfair Display': '/api/fonts/builtin/PlayfairDisplay.ttf',
  'Bebas Neue': '/api/fonts/builtin/BebasNeue-Regular.ttf',
  'Liberation Sans': '/api/fonts/builtin/LiberationSans-Regular.ttf',
  'Liberation Serif': '/api/fonts/builtin/LiberationSerif-Regular.ttf',
  'Liberation Mono': '/api/fonts/builtin/LiberationMono-Regular.ttf',
  'DejaVu Sans': '/api/fonts/builtin/DejaVuSans.ttf',
  'DejaVu Serif': '/api/fonts/builtin/DejaVuSerif.ttf',
  'DejaVu Sans Mono': '/api/fonts/builtin/DejaVuSansMono.ttf',
  'FreeSans': '/api/fonts/builtin/FreeSans.ttf',
};

function registerFontFace(fontName, url) {
  const existingId = `custom-font-${fontName.replace(/\s+/g, '-')}`;
  if (document.getElementById(existingId)) return;
  const style = document.createElement('style');
  style.id = existingId;
  style.textContent = `@font-face { font-family: '${fontName}'; src: url('${url}'); font-weight: 100 900; font-display: swap; }`;
  document.head.appendChild(style);
}

// Active-word timing constants, speaker palette, per-speaker rates and
// the word-index algorithm are shared with RenderEngine (preview AND
// client export) via ONE module — see utils/activeWordTiming.js.

function splitSegmentsByMaxWords(segments, maxWords) {
  if (!maxWords || maxWords <= 0) return segments;
  const result = [];
  for (const seg of segments) {
    const text = seg.subtitleText || seg.text || '';
    const words = text.split(/\s+/).filter(Boolean);
    if (words.length <= maxWords) { result.push(seg); continue; }
    const totalWords = words.length;
    const duration = seg.end - seg.start;
    const hasWordTs = seg.words && Array.isArray(seg.words) && seg.words.length === totalWords;
    let ct = seg.start;
    for (let i = 0; i < totalWords; i += maxWords) {
      const chunkWords = words.slice(i, i + maxWords);
      let chunkEnd;
      let chunkWordTs = null;

      if (hasWordTs) {
        // Use actual word timestamps for accurate chunk boundaries
        chunkWordTs = seg.words.slice(i, i + maxWords);
        if (chunkWordTs.length > 0) {
          const lastWordInChunk = chunkWordTs[chunkWordTs.length - 1];
          chunkEnd = (lastWordInChunk.end || lastWordInChunk.endTime) + 0.02;
        } else {
          chunkWordTs = null;
          chunkEnd = ct + duration * (chunkWords.length / totalWords);
        }
      } else {
        // Fallback: proportional splitting
        chunkEnd = ct + duration * (chunkWords.length / totalWords);
        if (seg.words && Array.isArray(seg.words)) {
          chunkWordTs = seg.words.slice(i, i + maxWords);
          if (!chunkWordTs.length) chunkWordTs = null;
        }
      }

      // Last chunk always ends at segment end
      if (i + maxWords >= totalWords) chunkEnd = seg.end;
      // Never exceed segment end
      chunkEnd = Math.min(chunkEnd, seg.end);

      if (chunkEnd - ct >= 0.1) {
        result.push({ ...seg, start: ct, end: chunkEnd, subtitleText: chunkWords.join(' '), text: chunkWords.join(' '), speaker: seg.speaker, words: chunkWordTs });
      }
      ct = chunkEnd;
    }
  }
  return result;
}

function getSpeakerColor(speaker, speakersOrdered, settings) {
  const fontColor = settings?.subtitleFontColor;
  const useSpeaker = settings?.useSpeakerColors ?? true;
  // When speaker colors are enabled, they override the font color picker
  if (useSpeaker) {
    const speakerColors = settings?.speakerColors || {};
    if (speakerColors[speaker]) return speakerColors[speaker];
    const idx = speakersOrdered.indexOf(speaker);
    return DEFAULT_SPEAKER_PALETTE[(idx >= 0 ? idx : 0) % DEFAULT_SPEAKER_PALETTE.length];
  }
  // Speaker colors off — use explicit font color or default white
  return fontColor || '#FFFFFF';
}

function hexToRgba(hex, opacity) {
  hex = (hex || '#000000').replace('#', '');
  if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${opacity})`;
}

// ── Component ───────────────────────────────────────────────────────────
/**
 * Self-contained subtitle overlay for use inside VideoEditor's viewport.
 * Renders subtitle items from the timeline store as the SINGLE SOURCE OF TRUTH.
 * Falls back to transcript prop only when no timeline subtitle items exist.
 *
 * Props:
 *  - currentTime: number (absolute video time)
 *  - transcript: array of { start, end, text, speaker, words? } (fallback only)
 *  - clipStart, clipEnd: clip time boundaries
 *  - settings: full clip settings object (subtitlesEnabled, subtitleFont, etc.)
 *  - aspectRatio, sourceWidth, sourceHeight: for font scaling
 */
export default function SubtitleOverlay({
  currentTime = 0,
  transcript = [],
  clipStart = 0,
  clipEnd = 0,
  settings = {},
  aspectRatio,
  sourceWidth = 1920,
  sourceHeight = 1080,
  segments = [],
}) {
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  const [currentWordIdx, setCurrentWordIdx] = useState(-1);
  const [isEditing, setIsEditing] = useState(false);
  const editRef = useRef(null);

  // Timeline store — SINGLE SOURCE OF TRUTH for subtitle items
  const timelineItems = useTimelineStore((s) => s.items);
  const tracks = useTimelineStore((s) => s.tracks);
  const selectedItemId = useTimelineStore((s) => s.selectedItemId);
  const selectedItemIds = useTimelineStore((s) => s.selectedItemIds);
  const setSelectedItemId = useTimelineStore((s) => s.setSelectedItemId);
  const toggleSelectedItem = useTimelineStore((s) => s.toggleSelectedItem);
  const updateItem = useTimelineStore((s) => s.updateItem);
  const setItemPositions = useTimelineStore((s) => s.setItemPositions);

  // Drag state for repositioning the subtitle in the preview. Mirrors the
  // pattern in InteractiveOverlay — start positions for every selected
  // item are captured on pointerdown so a multi-item drag translates the
  // whole selection as a rigid block.
  const [isDraggingSub, setIsDraggingSub] = useState(false);
  const dragRef = useRef(null);
  // Long-press-to-edit (touch): held press opens inline caption edit.
  const subLongPressRef = useRef(0);
  const subLongPressStartRef = useRef(null);

  // Check if subtitle track is hidden via the eye icon toggle
  // This IS the single source of truth — settings.subtitlesEnabled syncs TO this
  // via the useEffect in VideoEditor (FIX 2)
  const subtitleTrackVisible = useMemo(() => {
    const subTrack = tracks.find((t) => t.type === 'subtitle');
    return subTrack ? subTrack.visible !== false : true;
  }, [tracks]);

  // Per-segment subtitle override: segment's subtitlesEnabled takes precedence.
  //
  // COORDINATE BASE — deliberate split (keep in sync):
  //   • This gate compares ABSOLUTE `currentTime` against editor `segments`,
  //     whose `.start`/`.end` are absolute (full-video) seconds.
  //   • The active-line finder below compares CLIP-RELATIVE
  //     `relTime = currentTime - clipStart` against `clipSegments`, whose
  //     timings are clip-relative.
  // Each is internally correct, but they intentionally use different bases.
  // If either collection's base ever changes, both call sites must change
  // together — this is the one spot a clip-vs-absolute mix-up would surface
  // (only when clipStart > 0).
  const perSegmentEnabled = useMemo(() => {
    if (!segments || segments.length === 0) return true; // no segments = always on
    const absTime = currentTime;
    for (const seg of segments) {
      if (absTime >= seg.start && absTime < seg.end) {
        return seg.subtitlesEnabled !== false;
      }
    }
    return true; // not in any segment = default on
  }, [segments, currentTime]);

  // SINGLE GATE: track visible AND per-segment enabled
  // settings.subtitlesEnabled is NOT checked here because it's already
  // synced to track.visible via the useEffect in VideoEditor (FIX 2)
  const subtitlesEnabled = subtitleTrackVisible && perSegmentEnabled;

  // Track container size for font scaling
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerSize({ w: entry.contentRect.width, h: entry.contentRect.height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Register @font-face and preload subtitle font.
  // The /api/fonts request is aborted on unmount so we never call
  // ``registerFontFace`` after the component is gone (which would
  // dirty the global document.fonts set with no consumer).
  useEffect(() => {
    const font = settings.subtitleFont;
    if (!font || typeof document === 'undefined') return undefined;

    if (BUILTIN_FONT_FILES[font]) {
      registerFontFace(font, BUILTIN_FONT_FILES[font]);
      document.fonts.load(`400 16px "${font}"`).catch(() => {});
      document.fonts.load(`700 16px "${font}"`).catch(() => {});
      return undefined;
    }

    const ctrl = new AbortController();
    fetch('/api/fonts', { signal: ctrl.signal })
      .then((r) => (r.ok ? r.json() : []))
      .then((fonts) => {
        if (ctrl.signal.aborted) return;
        const match = fonts.find((f) => f.name === font);
        if (match) {
          registerFontFace(match.name, match.url);
          document.fonts.load(`400 16px "${font}"`).catch(() => {});
          document.fonts.load(`700 16px "${font}"`).catch(() => {});
        }
      })
      .catch((err) => {
        // AbortError is the expected unmount path — ignore silently.
        if (err && err.name !== 'AbortError') {
          // Anything else is a real fetch failure; swallow per legacy behavior.
        }
      });

    document.fonts.load(`400 16px "${font}"`).catch(() => {});
    document.fonts.load(`700 16px "${font}"`).catch(() => {});

    return () => ctrl.abort();
  }, [settings.subtitleFont]);

  // ── SINGLE SOURCE: Use timeline store subtitle items ──
  // Timeline items are the source of truth. They are populated from transcript
  // during initFromClip() in VideoEditor, so they always exist when subtitles
  // are available. This eliminates the dual-source problem.
  const subtitleItems = useMemo(() => {
    if (!subtitlesEnabled || !subtitleTrackVisible) return [];
    return timelineItems.filter((it) => it.type === 'subtitle');
  }, [subtitlesEnabled, subtitleTrackVisible, timelineItems]);

  // Apply maxWords splitting to subtitle items for display
  const clipSegments = useMemo(() => {
    if (subtitleItems.length === 0) return [];
    const mapped = subtitleItems.map((it) => ({
      ...it,
      text: it.subtitleText || '',
      speaker: it.speaker || '',
    }));
    const maxWords = settings.subtitleMaxWords || 0;
    return maxWords > 0 ? splitSegmentsByMaxWords(mapped, maxWords) : mapped;
  }, [subtitleItems, settings.subtitleMaxWords]);

  const speakersOrdered = useMemo(() => {
    const seen = [];
    for (const seg of clipSegments) {
      if (seg.speaker && !seen.includes(seg.speaker)) seen.push(seg.speaker);
    }
    return seen;
  }, [clipSegments]);

  const speakerRates = useMemo(() => computeSpeakerRates(clipSegments), [clipSegments]);

  // Find current subtitle based on currentTime (clip-relative)
  const relTime = currentTime - clipStart;
  const activeWordEnabled = settings.activeWordEnabled || false;
  const currentSubtitle = useMemo(() => {
    if (!subtitlesEnabled || clipSegments.length === 0) return null;

    // Direct hit against the spoken window (prefers per-word
    // timestamps, falls back to segment-level — see
    // ``utils/subtitleTiming.js`` for the math). Using the same
    // predicate here and in ``TranscriptViewer`` guarantees the overlay
    // and the active-line highlight agree on when a line is active,
    // and removes the timing inconsistency the old ``effectiveEnd``
    // introduced between ``activeWordEnabled`` on vs off.
    const direct = clipSegments.find((seg) => isSpokenAt(seg, relTime));
    if (direct) return direct;

    // Gap bridging: when the playhead sits in a small gap between two
    // segments, hold the *previous* segment so the overlay doesn't
    // flash off/on. Tightened from 0.5 s → 0.35 s — anything longer
    // reads as lingering. We always bound the bridge by the next
    // segment's spoken start, so we can't carry a segment past when
    // the next speaker actually begins.
    const MAX_GAP_FILL = 0.45;
    for (let i = 0; i < clipSegments.length - 1; i++) {
      const cur = clipSegments[i];
      const nxt = clipSegments[i + 1];
      const curWin = spokenWindow(cur);
      const nxtWin = spokenWindow(nxt);
      if (
        relTime >= curWin.end &&
        relTime < nxtWin.start &&
        nxtWin.start - curWin.end < MAX_GAP_FILL
      ) {
        return cur;
      }
    }
    // Tail: after the last segment ends, don't hold — let it clear.
    return null;
  }, [subtitlesEnabled, clipSegments, relTime]);

  // Find the original timeline item for the current subtitle (for selection)
  const currentTimelineItem = useMemo(() => {
    if (!currentSubtitle) return null;
    // If the subtitle came directly from timeline (has id), use it
    if (currentSubtitle.id) {
      const direct = subtitleItems.find((it) => it.id === currentSubtitle.id);
      if (direct) return direct;
    }
    // Split-chunk fallback: match by *overlap* rather than endpoint
    // proximity. ``splitSegmentsByMaxWords`` can move chunk boundaries
    // further than the old 0.15 s endpoint tolerance, so endpoint
    // matching silently dropped chunks. Overlap matching also
    // correctly picks the parent item even when word-timestamp
    // boundary nudging pulls the chunk start slightly before the
    // parent item's nominal start.
    return (
      subtitleItems.find(
        (it) =>
          it.start <= currentSubtitle.start + 0.05 &&
          it.end >= currentSubtitle.end - 0.05,
      ) || null
    );
  }, [currentSubtitle, subtitleItems]);

  // Click-to-select: select the subtitle timeline item.
  // Guard: don't steal selection from overlay items (text/image/shape) that are
  // visible at the same time — clicking near the bottom of the frame likely
  // intends to hit the subtitle, but elsewhere the user may want the overlay item.
  // Also guard if a non-subtitle is already selected.
  const hasVisibleOverlayItems = useMemo(() => {
    const absTime = currentTime;
    return timelineItems.some((it) => {
      if (it.type === 'subtitle' || it.type === 'video' || it.type === 'audio') return false;
      if (!(absTime >= it.start && absTime < it.end)) return false;
      const track = tracks.find((t) => t.id === it.trackId);
      return track && track.visible !== false;
    });
  }, [timelineItems, tracks, currentTime]);

  const handleSubtitleClick = useCallback((e) => {
    e.stopPropagation();
    if (!currentTimelineItem) return;
    // Shift / Cmd / Ctrl click extends the current selection so the
    // subtitle can join the marquee group.
    if (e.shiftKey || e.metaKey || e.ctrlKey) {
      toggleSelectedItem(currentTimelineItem.id);
      return;
    }
    // Plain click on an unselected element selects only the subtitle.
    if (!selectedItemIds.includes(currentTimelineItem.id)) {
      setSelectedItemId(currentTimelineItem.id);
    }
  }, [currentTimelineItem, setSelectedItemId, toggleSelectedItem, selectedItemIds]);

  // ── DRAG: reposition the subtitle (and any other selected items) ──
  const handleSubtitlePointerDown = useCallback((e) => {
    if (!currentTimelineItem) return;
    // Ignore secondary buttons and event from the inline editor textarea.
    if (e.button !== 0 && e.button !== undefined) return;
    e.stopPropagation();

    // Promote the subtitle to (part of) the selection. Shift/Cmd/Ctrl adds
    // to the existing selection; plain pointerdown either selects it alone
    // or keeps the existing group if the subtitle is already in it.
    if (e.shiftKey || e.metaKey || e.ctrlKey) {
      if (!selectedItemIds.includes(currentTimelineItem.id)) {
        toggleSelectedItem(currentTimelineItem.id);
      }
    } else if (!selectedItemIds.includes(currentTimelineItem.id)) {
      setSelectedItemId(currentTimelineItem.id);
    }

    // Container rect = the actual video-content area inside the viewport
    // (handles letterboxing). Drag deltas are normalised against this so
    // dragging tracks the cursor 1:1 regardless of letterbox bars.
    const containerEl = containerRef.current;
    const rect = (containerEl && typeof containerEl.getBoundingClientRect === 'function')
      ? containerEl.getBoundingClientRect()
      : null;
    if (!rect || rect.width <= 0 || rect.height <= 0) return;

    // Latest selection at the moment of pointerdown (including the
    // subtitle we just toggled in).
    const liveSel = useTimelineStore.getState().selectedItemIds;
    const dragIds = (liveSel && liveSel.length > 0 && liveSel.includes(currentTimelineItem.id))
      ? liveSel
      : [currentTimelineItem.id];
    const liveItems = useTimelineStore.getState().items;
    const startPositions = {};
    for (const id of dragIds) {
      const it = liveItems.find((x) => x.id === id);
      if (!it) continue;
      const p = it.position || (it.type === 'subtitle' ? { x: 50, y: 90 } : { x: 50, y: 50 });
      startPositions[id] = { x: p.x, y: p.y };
    }

    useTimelineStore.temporal.getState().pause();
    dragRef.current = {
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      containerW: rect.width,
      containerH: rect.height,
      startPositions,
      dragIds,
      moved: false,
    };
    setIsDraggingSub(true);

    // Touch long-press → inline caption edit (reliable double-click substitute).
    subLongPressStartRef.current = { x: e.clientX, y: e.clientY };
    clearTimeout(subLongPressRef.current);
    subLongPressRef.current = setTimeout(() => {
      subLongPressRef.current = 0;
      subLongPressStartRef.current = null;
      dragRef.current = null;
      setIsDraggingSub(false);
      try { useTimelineStore.temporal.getState().resume(); } catch { /* noop */ }
      setSelectedItemId(currentTimelineItem.id);
      setIsEditing(true);
      setTimeout(() => editRef.current?.focus(), 50);
    }, 500);
  }, [currentTimelineItem, selectedItemIds, setSelectedItemId,
      toggleSelectedItem, containerRef]);

  // Global pointermove/up while a subtitle drag is active.
  useEffect(() => {
    if (!isDraggingSub) return;

    const handleMove = (e) => {
      // Movement cancels a pending long-press-to-edit (it's a drag).
      if (subLongPressRef.current && subLongPressStartRef.current) {
        const lp = subLongPressStartRef.current;
        if (Math.hypot(e.clientX - lp.x, e.clientY - lp.y) > 8) {
          clearTimeout(subLongPressRef.current);
          subLongPressRef.current = 0;
          subLongPressStartRef.current = null;
        }
      }
      const ds = dragRef.current;
      if (!ds) return;
      const dx = e.clientX - ds.startMouseX;
      const dy = e.clientY - ds.startMouseY;
      if (!ds.moved && Math.abs(dx) + Math.abs(dy) > 2) {
        ds.moved = true;
      }
      let dxPct = (dx / ds.containerW) * 100;
      let dyPct = (dy / ds.containerH) * 100;

      // Magnetic snap to the golden grid / frame / other elements while the
      // grid is on (Alt disables). Snap the PRIMARY subtitle's centre, then
      // shift the whole dragged group by the same correction. Subtitles snap
      // by centre only (size 0) — their item box is the full frame.
      const store = useTimelineStore.getState();
      let guides = [];
      if (store.goldenGrid && !e.altKey && ds.dragIds.length) {
        const primaryId = ds.dragIds[0];
        const sp = ds.startPositions[primaryId];
        if (sp) {
          const rawX = sp.x + dxPct;
          const rawY = sp.y + dyPct;
          const others = store.items
            .filter((o) => !ds.dragIds.includes(o.id) && o.position
              && o.type !== 'audio' && o.type !== 'video')
            .map((o) => ({ x: o.position.x, y: o.position.y, w: o.size?.w || 0, h: o.size?.h || 0 }));
          const thX = (12 / ds.containerW) * 100;
          const thY = (12 / ds.containerH) * 100;
          const snapped = snapToGuides({ x: rawX, y: rawY, w: 0, h: 0, others, thX, thY });
          dxPct += snapped.x - rawX;
          dyPct += snapped.y - rawY;
          guides = snapped.guides;
        }
      }
      store.setSnapGuides(guides);

      const updates = {};
      for (const id of ds.dragIds) {
        const sp = ds.startPositions[id];
        if (!sp) continue;
        updates[id] = { x: sp.x + dxPct, y: sp.y + dyPct };
      }
      setItemPositions(updates);
    };

    const handleUp = () => {
      if (subLongPressRef.current) { clearTimeout(subLongPressRef.current); subLongPressRef.current = 0; subLongPressStartRef.current = null; }
      useTimelineStore.temporal.getState().resume();
      if (useTimelineStore.getState().snapGuides.length) useTimelineStore.getState().setSnapGuides([]);
      dragRef.current = null;
      setIsDraggingSub(false);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    window.addEventListener('pointercancel', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
      window.removeEventListener('pointercancel', handleUp);
      if (isDraggingSub) {
        useTimelineStore.temporal.getState().resume();
      }
    };
  }, [isDraggingSub, setItemPositions]);

  const handleSubtitleDoubleClick = useCallback((e) => {
    e.stopPropagation();
    if (!currentTimelineItem) return;
    // Don't steal selection from overlay items
    if (selectedItemId) {
      const sel = timelineItems.find((it) => it.id === selectedItemId);
      if (sel && sel.type !== 'subtitle') return;
    }
    setSelectedItemId(currentTimelineItem.id);
    setIsEditing(true);
    setTimeout(() => editRef.current?.focus(), 50);
  }, [currentTimelineItem, setSelectedItemId, selectedItemId, timelineItems]);

  const handleEditBlur = useCallback(() => {
    setIsEditing(false);
  }, []);

  const handleEditChange = useCallback((e) => {
    if (currentTimelineItem) {
      updateItem(currentTimelineItem.id, { subtitleText: e.target.value });
    }
  }, [currentTimelineItem, updateItem]);

  const handleEditKeyDown = useCallback((e) => {
    e.stopPropagation();
    if (e.key === 'Escape') setIsEditing(false);
  }, []);

  // Active word tracking
  useEffect(() => {
    if (!activeWordEnabled || !currentSubtitle) {
      setCurrentWordIdx(-1);
      return;
    }
    const idx = getCurrentWordIndex(currentSubtitle, relTime, speakerRates);
    setCurrentWordIdx(idx);
  }, [activeWordEnabled, currentSubtitle, relTime, speakerRates]);

  // Output dims for font scaling
  const outputDims = useMemo(() => {
    if (aspectRatio && ASPECT_RATIO_DIMS[aspectRatio]) {
      return { w: ASPECT_RATIO_DIMS[aspectRatio][0], h: ASPECT_RATIO_DIMS[aspectRatio][1] };
    }
    return { w: sourceWidth, h: sourceHeight };
  }, [aspectRatio, sourceWidth, sourceHeight]);

  const subtitleScale = useMemo(() => {
    if (containerSize.w === 0) return 0;
    const scaleW = containerSize.w / outputDims.w;
    const scaleH = containerSize.h / outputDims.h;
    return Math.min(scaleW, scaleH);
  }, [containerSize, outputDims]);

  const backendFontScale = useMemo(
    () => Math.min(outputDims.w, outputDims.h) / Math.min(REF_W, REF_H),
    [outputDims],
  );

  const subtitleFontSize = useMemo(() => {
    if (!subtitlesEnabled || subtitleScale === 0) return 14;
    const size = settings.subtitleSize || 'medium';
    const basePx = typeof size === 'number' ? size : (FONT_SIZE_MAP[size] || 30);
    const backendPx = Math.max(16, Math.round(basePx * backendFontScale));
    return Math.max(8, backendPx * subtitleScale);
  }, [subtitlesEnabled, settings.subtitleSize, subtitleScale, backendFontScale]);

  // Compute the actual video content area within the viewport
  const videoContentRect = useMemo(() => {
    if (containerSize.w === 0 || containerSize.h === 0) {
      return { left: 0, top: 0, width: containerSize.w, height: containerSize.h };
    }
    const videoAR = outputDims.w / outputDims.h;
    const containerAR = containerSize.w / containerSize.h;

    if (Math.abs(videoAR - containerAR) < 0.02) {
      return { left: 0, top: 0, width: containerSize.w, height: containerSize.h };
    }

    if (videoAR > containerAR) {
      const h = containerSize.w / videoAR;
      return { left: 0, top: (containerSize.h - h) / 2, width: containerSize.w, height: h };
    } else {
      const w = containerSize.h * videoAR;
      return { left: (containerSize.w - w) / 2, top: 0, width: w, height: containerSize.h };
    }
  }, [containerSize, outputDims]);

  // Check if the current subtitle's timeline item is selected — primary
  // selection draws a solid outline, group members draw a dashed one so
  // the user can tell which item handles act on.
  const isSubtitlePrimary = useMemo(() => {
    if (!currentTimelineItem || !selectedItemId) return false;
    return currentTimelineItem.id === selectedItemId;
  }, [currentTimelineItem, selectedItemId]);
  const isSubtitleInSelection = useMemo(() => {
    if (!currentTimelineItem) return false;
    return selectedItemIds.includes(currentTimelineItem.id);
  }, [currentTimelineItem, selectedItemIds]);
  const isSubtitleSelected = isSubtitleInSelection;

  // The resolved text comes directly from the timeline item (single source)
  const resolvedSubtitleText = currentSubtitle?.subtitleText || currentSubtitle?.text || '';

  // Container wrapper — fills parent, used for ResizeObserver
  if (!subtitlesEnabled) {
    return <div ref={containerRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />;
  }

  if (!currentSubtitle) {
    return <div ref={containerRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />;
  }

  // ── Render subtitle ─────────────────────────────────────────────────
  // Use per-item position if set (from InteractiveOverlay drag), else use settings
  const itemPos = currentTimelineItem?.position || { x: 50, y: 90 };
  const itemRotation = currentTimelineItem?.transform?.rotation || 0;

  const position = settings.subtitlePosition || 'bottom';
  const maxWidthPct = settings.subtitleMaxWidth ?? 90;
  const offsetVPct = settings.subtitleOffsetV ?? 4;
  const bgEnabled = settings.subtitleBgEnabled || false;
  const bgColor = settings.subtitleBgColor || '#000000';
  const bgOpacity = settings.subtitleBgOpacity ?? 75;
  const fontWeight = typeof settings.subtitleFontWeight === 'number' ? settings.subtitleFontWeight : settings.subtitleFontWeight === 'bold' ? 700 : settings.subtitleFontWeight === 'black' ? 900 : 400;
  const rawFont = settings.subtitleFont || 'DM Sans';
  const fontFamily = `"${rawFont}", sans-serif`;
  const showLabels = settings.showSpeakerLabels ?? false;
  const color = getSpeakerColor(currentSubtitle.speaker, speakersOrdered, settings);

  // Outline
  const olColorHex = (settings.subtitleOutlineColor || '#000000').replace('#', '');
  const olR = parseInt(olColorHex.substring(0, 2), 16) || 0;
  const olG = parseInt(olColorHex.substring(2, 4), 16) || 0;
  const olB = parseInt(olColorHex.substring(4, 6), 16) || 0;
  const olOpacity = Math.max(0, Math.min(100, settings.subtitleOutlineOpacity ?? 100)) / 100;
  const olWidth = Math.max(0, Math.min(10, settings.subtitleOutlineWidth ?? 2));
  const backendOlWidth = Math.max(0, Math.round(olWidth * backendFontScale));
  const scaledOlWidth = backendOlWidth * subtitleScale;

  let outlineStyle;
  if (bgEnabled && scaledOlWidth > 0) {
    const olColorStr = `rgba(${olR},${olG},${olB},${olOpacity})`;
    outlineStyle = {
      WebkitTextStroke: `${scaledOlWidth * 2}px ${olColorStr}`,
      paintOrder: 'stroke fill',
    };
  } else if (bgEnabled) {
    outlineStyle = {};
  } else if (scaledOlWidth > 0) {
    const shadowDepth = Math.max(1, Math.min(4, Math.round(backendOlWidth * 0.75)));
    const scaledShadow = shadowDepth * subtitleScale;
    const olColorStr = `rgba(${olR},${olG},${olB},${olOpacity})`;
    const dropShadow = `${scaledShadow}px ${scaledShadow}px 0px rgba(0,0,0,0.5)`;
    outlineStyle = {
      WebkitTextStroke: `${scaledOlWidth * 2}px ${olColorStr}`,
      paintOrder: 'stroke fill',
      textShadow: outlineTextShadow(scaledOlWidth, olColorStr, dropShadow),
    };
  } else {
    outlineStyle = { textShadow: '1px 1px 2px rgba(0,0,0,0.8)' };
  }

  // Margins
  const clampedMaxWidth = Math.max(20, Math.min(100, maxWidthPct));
  const clampedOffsetV = Math.max(0, Math.min(100, offsetVPct));
  const marginH_px = Math.max(20, Math.floor(outputDims.w * (100 - clampedMaxWidth) / 100 / 2));
  const maxMarginH = Math.floor(outputDims.w * 0.40);
  const effectiveMarginH = Math.min(marginH_px, maxMarginH) / outputDims.w * 100;

  // Position: use per-item position if dragged, otherwise use settings-based position
  const hasCustomPosition = itemPos.x !== 50 || itemPos.y !== 90;
  let positionStyle;
  if (hasCustomPosition) {
    // Per-item position from InteractiveOverlay drag (percentage-based)
    positionStyle = {
      left: `${itemPos.x}%`,
      top: `${itemPos.y}%`,
      transform: `translate(-50%, -50%)${itemRotation ? ` rotate(${itemRotation}deg)` : ''}`,
    };
  } else if (position === 'top') {
    positionStyle = {
      top: `${clampedOffsetV}%`,
      ...(itemRotation ? { transform: `rotate(${itemRotation}deg)` } : {}),
    };
  } else if (position === 'center') {
    positionStyle = {
      top: '50%',
      transform: `translateY(-50%)${itemRotation ? ` rotate(${itemRotation}deg)` : ''}`,
    };
  } else {
    positionStyle = {
      bottom: `${clampedOffsetV}%`,
      ...(itemRotation ? { transform: `rotate(${itemRotation}deg)` } : {}),
    };
  }

  const text = showLabels && currentSubtitle.speaker
    ? `${currentSubtitle.speaker}: ${resolvedSubtitleText}`
    : resolvedSubtitleText;

  // Active word highlighting
  const awColor = settings.activeWordColor || '#FFD700';
  const awOutlineColor = settings.activeWordOutlineColor || '#000000';
  const awBgColor = settings.activeWordBgColor || '#000000';
  const awBgOpacity = settings.activeWordBgOpacity ?? 0;

  let textContent;
  if (activeWordEnabled && currentWordIdx >= 0) {
    const words = resolvedSubtitleText.split(/\s+/).filter(Boolean);
    const prefix = showLabels && currentSubtitle.speaker ? `${currentSubtitle.speaker}: ` : '';
    const awOlHex = awOutlineColor.replace('#', '');
    const awOlR = parseInt(awOlHex.substring(0, 2), 16) || 0;
    const awOlG = parseInt(awOlHex.substring(2, 4), 16) || 0;
    const awOlB = parseInt(awOlHex.substring(4, 6), 16) || 0;
    textContent = (
      <>
        {prefix}
        {words.map((word, idx) => {
          const isActive = idx === currentWordIdx;
          const wordStyle = isActive ? {
            color: awColor,
            ...(!bgEnabled && scaledOlWidth > 0 ? {
              WebkitTextStroke: `${scaledOlWidth * 2}px rgba(${awOlR},${awOlG},${awOlB},${olOpacity})`,
              paintOrder: 'stroke fill',
              textShadow: outlineTextShadow(scaledOlWidth, `rgba(${awOlR},${awOlG},${awOlB},${olOpacity})`),
            } : {}),
            ...(awBgOpacity > 0 ? {
              backgroundColor: hexToRgba(awBgColor, awBgOpacity / 100),
              padding: `${Math.max(1, 1 * subtitleScale)}px ${Math.max(1, 2 * subtitleScale)}px`,
              borderRadius: `${(settings.activeWordBgRadius ?? 4) * subtitleScale}px`,
            } : {}),
          } : {};
          return (
            <span key={idx} style={wordStyle}>
              {word}{idx < words.length - 1 ? ' ' : ''}
            </span>
          );
        })}
      </>
    );
  } else {
    textContent = text;
  }

  // Editing text uses the resolved text from timeline store
  const editingText = resolvedSubtitleText;

  return (
    /* z-index 12 keeps subtitles ABOVE the interactive overlay layer
       (InteractiveOverlay root is z-index 10), so subtitles always composite
       on top of shapes / images / text overlays — the subtitle track sits at
       the top of the compositing order (see getCompositingOrder). */
    <div ref={containerRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 12 }}>
      {/* Constrain subtitles to the actual video content area (handles letterboxing) */}
      <div style={{
        position: 'absolute',
        left: videoContentRect.left,
        top: videoContentRect.top,
        width: videoContentRect.width,
        height: videoContentRect.height,
        overflow: 'hidden',
        pointerEvents: 'none',
      }}>
        <div style={{
          position: 'absolute',
          ...(hasCustomPosition ? {
            inset: 0,
          } : {
            left: `${effectiveMarginH}%`,
            right: `${effectiveMarginH}%`,
          }),
          textAlign: 'center',
          pointerEvents: 'none',
          ...positionStyle,
        }}>
          {/* Clickable subtitle text */}
          <span
            data-item-id={currentTimelineItem?.id}
            data-item-type="subtitle"
            style={{
              display: 'inline-block',
              fontFamily,
              fontSize: subtitleFontSize,
              fontWeight,
              color,
              lineHeight: 1.4,
              wordWrap: 'break-word',
              overflowWrap: 'break-word',
              whiteSpace: 'pre-wrap',
              cursor: isDraggingSub ? 'grabbing' : 'grab',
              pointerEvents: 'auto',
              touchAction: 'none',
              ...outlineStyle,
              ...(bgEnabled ? {
                background: hexToRgba(bgColor, bgOpacity / 100),
                padding: `${Math.max(1, Math.max(Math.floor(4 * backendFontScale), 2) * subtitleScale)}px`,
                borderRadius: `${(settings.subtitleBgRadius || 0) * subtitleScale}px`,
              } : {}),
              ...(isSubtitleSelected ? {
                outline: isSubtitlePrimary
                  ? '2px solid #0A84FF'
                  : '2px dashed rgba(10, 132, 255, 0.85)',
                outlineOffset: 4,
                borderRadius: 4,
              } : {}),
            }}
            onPointerDown={isEditing ? undefined : handleSubtitlePointerDown}
            onClick={handleSubtitleClick}
            onDoubleClick={handleSubtitleDoubleClick}
            title="Click to select, drag to reposition, double-click to edit"
          >
            {textContent}
          </span>

          {/* Inline editing overlay (double-click to edit) */}
          {isEditing && (
            <textarea
              ref={editRef}
              value={editingText}
              onChange={handleEditChange}
              onBlur={handleEditBlur}
              onKeyDown={handleEditKeyDown}
              onMouseDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              style={{
                display: 'block',
                width: '100%',
                minHeight: 40,
                marginTop: 4,
                background: 'rgba(0,0,0,0.75)',
                color: '#fff',
                border: '2px solid #0A84FF',
                borderRadius: 6,
                padding: '8px 10px',
                fontSize: Math.max(12, subtitleFontSize * 0.7),
                fontFamily,
                resize: 'vertical',
                outline: 'none',
                pointerEvents: 'auto',
                zIndex: 50,
                backdropFilter: 'blur(4px)',
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
