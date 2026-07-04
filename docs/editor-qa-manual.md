# Editor manual QA script

Run this after any change to the render paths, the inspector, or the
export dialog. Automated coverage lives in `frontend` vitest (83 tests)
and the backend pytest modules; this script covers what only eyes can
verify. The feature-level contract is `docs/parity-checklist.md`
(generated from `frontend/src/utils/featureParityMatrix.js`).

Pixel-level verification: `make parity CLIENT=… SERVER=…` (see
Makefile; SSIM ≥ 0.90 at 3 timestamps on the Unraid host).

## 1. Subtitle settings live-update the preview

Open a clip with a transcript, turn subtitles on, then change each
control in the Subtitles panel and confirm the preview updates within
one frame (no export needed):

- [ ] Font family, size (S/M/L + numeric), weight
- [ ] Font color; per-speaker colors on/off; explicit speaker color map
- [ ] Position pills (top / center / bottom) and vertical offset — the
      pill highlight must always agree with the numeric offset
- [ ] Max width %, max words per cue
- [ ] Outline width / color / opacity
- [ ] Background on/off, color, opacity, corner radius
- [ ] Active-word highlighting on/off, its color, bg color/opacity/radius,
      outline color — the highlighted word must track playback
- [ ] Speaker labels on/off

`settingsCoverage.test.js` proves every one of these keys reaches both
`SubtitleOverlay` (preview) and `mapSubtitleSettings` (export payload);
this manual pass verifies they *look* right.

## 2. Export parity — every quality × aspect

For each quality (720p, 1080p, 1440p, 4K) × aspect (16:9, 9:16, 1:1,
4:5) worth spot-checking (at minimum 1080p and 1440p at 9:16 and 16:9):

- [ ] Exported dimensions match `EXPORT_DIMS_BY_QUALITY` (e.g. 9:16
      @1440p = 1440×2560); server and browser exports have identical
      pixel dimensions
- [ ] Framing matches the preview (subject crop, gaming layout)
- [ ] Subtitle position and size match the preview at the same frame
- [ ] The export QA block in the dialog reports "high" confidence

## 3. Fades (video + audio together)

- [ ] Set fade in = 1s and fade out = 1s on the main clip; preview shows
      both the image and the AUDIO ramping (listen)
- [ ] Server export: audio ramps with the picture (afade parity, task
      1.2); no abrupt audio at either edge
- [ ] Fade in + fade out longer than the clip: both ramps multiply
      (quieter middle), never a hard cut

## 4. Transitions

For each type (dissolve, fade, wipe left/right, slide left/right, zoom):

- [ ] Preview plays the right style
- [ ] Server export renders the matching xfade (dissolve intentionally
      exports as crossfade — canvas dissolve IS a crossfade)
- [ ] Mid-transition frames may differ slightly (easing caveat in the
      parity matrix) but start/end frames must match

## 5. Speed and pitch

- [ ] 2× speed, Preserve pitch OFF (default): preview audio chipmunks;
      browser AND server exports sound identical to the preview
- [ ] 2× speed, Preserve pitch ON: preview keeps pitch; browser export
      routes to server with the dialog notice; server export keeps pitch
- [ ] 0.5× spot check in both modes

## 6. Editor UX spot checks

- [ ] ⌘K palette runs actions; ? shows the cheat sheet; both lists agree
      with actual key behavior (same registry)
- [ ] Right-click menus: timeline clip, track header, preview overlay
      item; Esc and outside-click close; keyboard navigation works
- [ ] Ctrl/⌘+wheel zooms centered on the cursor; double-click a clip
      zooms to it; ⇧Z fits the timeline
- [ ] Drag the inspector and timeline dividers; double-click resets;
      sizes survive a reload
- [ ] Scrub a numeric field (drag), Shift = coarse, Alt = fine, click to
      type, Esc reverts; one ⌘Z undoes the whole drag
- [ ] Export dialog platform chips set aspect+quality; "Show safe zones"
      shades the preview; overlay warnings appear in the QA block
- [ ] OS "reduce motion" enabled: no dialog scale-in, no playhead glide

## 7. Phone / tablet (390×844 and 820×1180)

- [ ] Timeline renders compact lanes; tapping a track header expands
      that lane
- [ ] Pinch zooms the timeline; two-finger drag pans; long-press a clip
      opens the action sheet
- [ ] Properties opens as a bottom sheet; drag between peek/half/full
- [ ] Tablet: inspector is a 320px rail; chevron collapses it to icons
- [ ] Nothing scrolls horizontally at the page level
