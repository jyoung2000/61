/**
 * ExportEngine — Client-side export via WebCodecs + FFmpeg.wasm
 *
 * Uses the same RenderEngine.renderFrame() function as preview,
 * ensuring pixel-perfect export parity. Steps through time at
 * 1/fps intervals (not real-time).
 *
 * Pipeline:
 *   Canvas frames → WebCodecs VideoEncoder → webm-muxer → .webm
 *   Audio tracks → OfflineAudioContext → WAV
 *   FFmpeg.wasm: WebM video + WAV audio → MP4 (H.264 + AAC)
 *
 * Active word highlighting:
 *   When activeWordEnabled is on, the export FPS is automatically boosted
 *   to ensure smooth, natural word-by-word highlighting that matches
 *   the preview player exactly. The RenderEngine computes word indices
 *   internally using the same algorithm as SubtitleOverlay.
 *
 * Fallback: If WebCodecs unavailable, uses canvas.captureStream() + MediaRecorder.
 * The fallback records in REAL TIME (dropped frames and word-highlight
 * timing can diverge from the stepped WebCodecs path), so it only runs
 * after the ``onFallbackRequired`` callback confirms the reduced-fidelity
 * export with the user. Which path actually ran is recorded in
 * ``lastExportMeta`` and passed to ``onComplete``.
 * If FFmpeg.wasm fails, falls back to server-side export.
 */

/**
 * Map timeline time → source (media) time for a clip, honoring both
 * trim and per-clip speed. Preview applies speed via
 * ``element.playbackRate``; export must apply the same mapping when
 * seeking frames and scheduling audio, or a 2× clip exports at 1×.
 *
 * source_time = trimStart + (timeline_time - clip.start) * speed
 */
export function timelineToSourceTime(clip, timelineTime) {
  const speed = clip.speed || 1;
  return (clip.trimStart || 0) + (timelineTime - clip.start) * speed;
}

/**
 * Gain envelope for a clip at a timeline time — the audio twin of the
 * opacity fade math in RenderEngine._renderClip (elapsed/fadeIn,
 * remaining/fadeOut), so audible fades match the visual ones.
 * Returns the multiplier applied on top of clip.volume.
 */
export function fadeGainAt(clip, timelineTime) {
  let v = 1;
  const clipDur = clip.end - clip.start;
  const elapsed = timelineTime - clip.start;
  if (clip.fadeIn > 0 && elapsed < clip.fadeIn) {
    v *= Math.max(0, elapsed / clip.fadeIn);
  }
  if (clip.fadeOut > 0 && (clipDur - elapsed) < clip.fadeOut) {
    v *= Math.max(0, (clipDur - elapsed) / clip.fadeOut);
  }
  return v;
}

/**
 * Build the piecewise-linear gain automation points for a clip within
 * an export range. Returns [{time, value}] in OFFLINE-CONTEXT seconds
 * (0 = export start), ready for setValueAtTime/linearRampToValueAtTime.
 * Where fadeIn and fadeOut overlap the product is quadratic, so the
 * overlap is sampled densely (20 ms) to stay faithful.
 */
export function buildGainAutomation(clip, exportStart, exportEnd) {
  const base = clip.muted ? 0 : (clip.volume ?? 1.0);
  const t0 = Math.max(clip.start, exportStart);
  const t1 = Math.min(clip.end, exportEnd);
  if (t1 <= t0) return [];

  const fadeIn = Math.max(0, clip.fadeIn || 0);
  const fadeOut = Math.max(0, clip.fadeOut || 0);
  const points = new Set([t0, t1]);
  if (fadeIn > 0) points.add(Math.min(t1, Math.max(t0, clip.start + fadeIn)));
  if (fadeOut > 0) points.add(Math.min(t1, Math.max(t0, clip.end - fadeOut)));

  // Dense sampling only where the two fades overlap (non-linear region)
  const overlapStart = clip.end - fadeOut;
  const overlapEnd = clip.start + fadeIn;
  if (fadeIn > 0 && fadeOut > 0 && overlapStart < overlapEnd) {
    const s = Math.max(t0, overlapStart);
    const e = Math.min(t1, overlapEnd);
    for (let t = s; t < e; t += 0.02) points.add(t);
  }

  return [...points]
    .sort((a, b) => a - b)
    .map((t) => ({ time: t - exportStart, value: base * fadeGainAt(clip, t) }));
}

export default class ExportEngine {
  constructor(renderEngine, options = {}) {
    this.renderEngine = renderEngine;
    this.fps = options.fps || 30;
    this.videoBitrate = options.videoBitrate || 8_000_000;
    this.audioBitrate = options.audioBitrate || 128_000;
    this.width = options.width || 1920;
    this.height = options.height || 1080;
    this._cancelled = false;
    this._ffmpeg = null;
    this.onProgress = options.onProgress || null;
    this.onError = options.onError || null;
    this.onComplete = options.onComplete || null;
    // Called before falling back to the real-time MediaRecorder path.
    // Must return (or resolve to) true to proceed — the fallback is
    // reduced-fidelity (dropped frames, word-highlight drift) and needs
    // explicit user confirmation. No callback = fallback refused.
    this.onFallbackRequired = options.onFallbackRequired || null;
    // Metadata about the last export: which encode path ran, fps, timing.
    this.lastExportMeta = null;
  }

  /**
   * Check if WebCodecs API is available
   */
  static isWebCodecsAvailable() {
    return typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';
  }

  /**
   * Check if FFmpeg.wasm can be loaded (SharedArrayBuffer required)
   */
  static isFFmpegAvailable() {
    return typeof SharedArrayBuffer !== 'undefined';
  }

  /**
   * Compute the optimal export FPS based on content.
   * When active word highlighting is enabled, boost FPS to ensure
   * smooth, lag-free word transitions that look human-edited.
   *
   * @param {Array} clips - All clip items
   * @param {Object} settings - Project settings
   * @returns {number} - Optimal FPS for export
   */
  static computeOptimalFPS(clips, settings, baseFPS = 30) {
    const subSettings = settings?.subtitle || settings || {};
    if (!subSettings.activeWordEnabled) return baseFPS;

    // Find the fastest word transition rate across all subtitle clips
    const subtitles = (clips || []).filter(c => c.type === 'subtitle');
    if (subtitles.length === 0) return baseFPS;

    let minWordDuration = Infinity;
    for (const sub of subtitles) {
      const text = sub.subtitleText || '';
      const words = text.split(/\s+/).filter(Boolean);
      if (words.length <= 1) continue;
      const segDuration = sub.end - sub.start;
      const avgWordDuration = segDuration / words.length;
      minWordDuration = Math.min(minWordDuration, avgWordDuration);
    }

    if (minWordDuration === Infinity) return baseFPS;

    // We want at least 3 frames per word transition for smooth highlighting.
    // This prevents the jarring "skip" effect where a word appears to jump.
    const neededFPS = Math.ceil(3 / minWordDuration);

    // Clamp to reasonable range: at least baseFPS, at most 60fps
    // 60fps gives ~16.7ms per frame, which covers even fast speakers
    // (5+ words/sec = ~200ms/word, 60fps = ~12 frames per word)
    return Math.min(60, Math.max(baseFPS, neededFPS));
  }

  /**
   * Lazy-load FFmpeg.wasm
   */
  async _loadFFmpeg() {
    if (this._ffmpeg) return this._ffmpeg;

    try {
      const { FFmpeg } = await import('@ffmpeg/ffmpeg');
      const { toBlobURL } = await import('@ffmpeg/util');

      const ffmpeg = new FFmpeg();

      // Use multi-threaded core if SharedArrayBuffer is available
      const coreURL = await toBlobURL(
        'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm/ffmpeg-core.js',
        'text/javascript'
      );
      const wasmURL = await toBlobURL(
        'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm/ffmpeg-core.wasm',
        'application/wasm'
      );
      const workerURL = await toBlobURL(
        'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm/ffmpeg-core.worker.js',
        'text/javascript'
      );

      await ffmpeg.load({ coreURL, wasmURL, workerURL });
      this._ffmpeg = ffmpeg;
      return ffmpeg;
    } catch (err) {
      this.onError?.(`FFmpeg.wasm load failed: ${err.message}. Falling back to server export.`);
      return null;
    }
  }

  /**
   * Extract audio by decoding each clip's media and scheduling
   * AudioBufferSourceNodes into an OfflineAudioContext.
   *
   * The previous implementation called createMediaElementSource() on an
   * OfflineAudioContext — MediaElement sources are NOT supported on
   * offline contexts (browsers throw or render silence), so every
   * client export was silent. Decoded buffers are the only reliable
   * offline path, and they let us honor per-clip speed
   * (AudioBufferSourceNode.playbackRate), trim, volume, mute and
   * fadeIn/fadeOut gain automation exactly like the preview.
   *
   * @param {number} startTime - Start of export range
   * @param {number} endTime - End of export range
   * @param {Array} clips - Clip items
   * @param {Map} mediaElements - Media element map
   * @param {Array} [tracks] - Track definitions (for audioMuted)
   * @returns {Blob|null} - WAV blob or null if no audio
   */
  async _extractAudio(startTime, endTime, clips, mediaElements, tracks) {
    try {
      const duration = endTime - startTime;
      const sampleRate = 44100;
      const numSamples = Math.max(1, Math.ceil(duration * sampleRate));

      const trackAudioMuted = (trackId) => {
        const track = (tracks || []).find(t => t.id === trackId);
        if (!track) return false;
        // audioMuted is the explicit field; legacy ``muted`` falls back
        // to it (same resolution RenderEngine uses for visibility).
        return track.audioMuted !== undefined ? !!track.audioMuted : !!track.muted;
      };

      // Audio-producing clips overlapping the range. Muted in preview
      // must mean muted in export — drop clip-muted and track-muted here.
      const audioClips = clips.filter(c =>
        (c.type === 'video' || c.type === 'audio') &&
        c.end > startTime && c.start < endTime &&
        !c.muted && !trackAudioMuted(c.trackId)
      );

      if (audioClips.length === 0) return null;

      const offlineCtx = new OfflineAudioContext(2, numSamples, sampleRate);

      let scheduled = 0;
      for (const clip of audioClips) {
        try {
          const buffer = await this._getDecodedAudio(clip, mediaElements, offlineCtx);
          if (!buffer) continue;

          const speed = clip.speed || 1;
          const tlStart = Math.max(clip.start, startTime);
          const tlEnd = Math.min(clip.end, endTime);
          if (tlEnd <= tlStart) continue;

          const source = offlineCtx.createBufferSource();
          source.buffer = buffer;
          source.playbackRate.value = speed;

          const gainNode = offlineCtx.createGain();
          const automation = buildGainAutomation(clip, startTime, endTime);
          if (automation.length > 0) {
            gainNode.gain.setValueAtTime(automation[0].value, Math.max(0, automation[0].time));
            for (let i = 1; i < automation.length; i++) {
              gainNode.gain.linearRampToValueAtTime(
                automation[i].value, Math.max(0, automation[i].time));
            }
          } else {
            gainNode.gain.value = clip.muted ? 0 : (clip.volume ?? 1.0);
          }

          source.connect(gainNode);
          gainNode.connect(offlineCtx.destination);

          // start(when, offset, duration): offset/duration are in
          // BUFFER seconds (unscaled by playbackRate), when is in
          // context seconds — so source time uses the speed-mapped
          // helper and duration is timeline duration × speed.
          const when = tlStart - startTime;
          const sourceOffset = Math.max(0, timelineToSourceTime(clip, tlStart));
          const sourceDur = (tlEnd - tlStart) * speed;
          source.start(when, sourceOffset, sourceDur);
          scheduled++;
        } catch {
          // Skip clips whose media can't be fetched/decoded
        }
      }

      if (scheduled === 0) return null;

      const audioBuffer = await offlineCtx.startRendering();

      // Convert AudioBuffer to WAV
      return this._audioBufferToWav(audioBuffer);
    } catch {
      // Audio extraction failed — export will be silent
      return null;
    }
  }

  /**
   * Fetch + decode a clip's media into an AudioBuffer, cached per
   * mediaRef so multiple clips of the same asset decode once.
   */
  async _getDecodedAudio(clip, mediaElements, ctx) {
    const key = clip.mediaRef || clip.id;
    if (!this._audioBufferCache) this._audioBufferCache = new Map();
    if (this._audioBufferCache.has(key)) return this._audioBufferCache.get(key);

    const mediaEl = mediaElements?.get(key);
    const url = clip.src || mediaEl?.currentSrc || mediaEl?.src;
    let buffer = null;
    if (url) {
      try {
        const res = await fetch(url);
        if (res.ok) {
          const data = await res.arrayBuffer();
          buffer = await ctx.decodeAudioData(data);
        }
      } catch {
        buffer = null;
      }
    }
    this._audioBufferCache.set(key, buffer);
    return buffer;
  }

  /**
   * Convert an AudioBuffer to a WAV Blob.
   */
  _audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    const numSamples = audioBuffer.length;
    const dataSize = numSamples * blockAlign;
    const headerSize = 44;
    const buffer = new ArrayBuffer(headerSize + dataSize);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave channels and write PCM data
    const channels = [];
    for (let ch = 0; ch < numChannels; ch++) {
      channels.push(audioBuffer.getChannelData(ch));
    }

    let offset = headerSize;
    for (let i = 0; i < numSamples; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = Math.max(-1, Math.min(1, channels[ch][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }

  /**
   * Export using WebCodecs + FFmpeg.wasm
   *
   * @param {number} startTime - Start of export range
   * @param {number} endTime - End of export range
   * @param {Array} tracks - Track definitions
   * @param {Array} clips - Clip/item definitions
   * @param {Object} settings - Project settings
   * @param {Map} mediaElements - Media element map
   * @returns {Blob|null} - MP4 blob or null on failure
   */
  async export(startTime, endTime, tracks, clips, settings, mediaElements) {
    this._cancelled = false;

    // Pre-compute subtitle context for active word highlighting
    this.renderEngine.prepareSubtitleContext(clips);

    // Enable export mode — track visibility is preview-only, all tracks
    // are included in the exported video regardless of visibility state
    this.renderEngine._exportMode = true;

    // Boost FPS if active word highlighting is enabled for smooth transitions
    const exportFPS = ExportEngine.computeOptimalFPS(clips, settings, this.fps);

    if (!ExportEngine.isWebCodecsAvailable()) {
      // MediaRecorder fallback records in real time — reduced fidelity.
      // Require explicit confirmation instead of silently degrading.
      let confirmed = false;
      try {
        confirmed = this.onFallbackRequired ? await this.onFallbackRequired() : false;
      } catch {
        confirmed = false;
      }
      if (!confirmed) {
        this.renderEngine._exportMode = false;
        this.onError?.(
          'WebCodecs is not available and the reduced-fidelity real-time ' +
          'fallback was not confirmed. Use Server Export instead.');
        return null;
      }
      console.info('[ExportEngine] Export path: mediarecorder (reduced fidelity, user-confirmed)');
      return this._exportWithMediaRecorder(startTime, endTime, tracks, clips, settings, mediaElements, exportFPS);
    }
    console.info('[ExportEngine] Export path: webcodecs (frame-stepped)');

    const totalFrames = Math.ceil((endTime - startTime) * exportFPS);
    const frameDuration = 1_000_000 / exportFPS; // microseconds

    try {
      // Dynamically import webm-muxer
      const { Muxer, ArrayBufferTarget } = await import('webm-muxer');

      const target = new ArrayBufferTarget();
      const muxer = new Muxer({
        target,
        video: {
          codec: 'V_VP9',
          width: this.width,
          height: this.height,
        },
        firstTimestampBehavior: 'offset',
      });

      const encoder = new VideoEncoder({
        output: (chunk, meta) => {
          muxer.addVideoChunk(chunk, meta);
        },
        error: (err) => {
          this.onError?.(`VideoEncoder error: ${err.message}`);
        },
      });

      encoder.configure({
        codec: 'vp09.00.10.08',
        width: this.width,
        height: this.height,
        bitrate: this.videoBitrate,
        framerate: exportFPS,
        hardwareAcceleration: 'prefer-hardware',
      });

      // Pre-load all fonts at correct weights before rendering begins
      await this.renderEngine.ensureFontsLoaded(clips, settings);

      // Verify each font is usable at the weight we need
      for (const clip of clips) {
        if (clip.type === 'text') {
          const fn = clip.textStyle?.fontFamily;
          const fw = clip.textStyle?.fontWeight || 400;
          if (fn) {
            const fontString = `${fw} 48px "${fn}"`;
            if (!document.fonts.check(fontString)) {
              console.warn(`[ExportEngine] Font not available for canvas: ${fontString}, retrying...`);
              try {
                await this.renderEngine.loadFont(fn, null, fw);
                await document.fonts.ready;
              } catch { /* best effort */ }
            }
          }
        }
      }

      // Step through time and encode each frame
      for (let i = 0; i < totalFrames; i++) {
        if (this._cancelled) break;

        const time = startTime + i / exportFPS;

        // Seek media elements to correct time
        for (const [assetId, mediaEl] of mediaElements) {
          if (mediaEl instanceof HTMLVideoElement) {
            // Find the clip that uses this asset at this time
            const clip = clips.find(c =>
              (c.mediaRef === assetId || c.id === assetId) &&
              time >= c.start && time < c.end
            );
            if (clip) {
              // Speed-aware mapping — a 2× clip must step through source
              // frames twice as fast (same formula as audio scheduling).
              const clipTime = timelineToSourceTime(clip, time);
              if (Math.abs(mediaEl.currentTime - clipTime) > 0.02) {
                mediaEl.currentTime = clipTime;
                // Wait for seek
                await new Promise(resolve => {
                  const onSeeked = () => { mediaEl.removeEventListener('seeked', onSeeked); resolve(); };
                  mediaEl.addEventListener('seeked', onSeeked);
                  setTimeout(resolve, 100);
                });
              }
            }
          }
        }

        // Render frame to canvas (RenderEngine now computes active word indices internally)
        this.renderEngine.renderFrame(time, tracks, clips, settings, mediaElements);

        // Encode
        const frame = new VideoFrame(this.renderEngine.canvas, {
          timestamp: i * frameDuration,
          duration: frameDuration,
        });

        const keyFrame = i % (exportFPS * 2) === 0; // keyframe every 2 seconds
        encoder.encode(frame, { keyFrame });
        frame.close();

        // Report progress
        if (i % 10 === 0) {
          this.onProgress?.(Math.round((i / totalFrames) * 75)); // 75% for video encoding
        }
      }

      await encoder.flush();
      encoder.close();
      muxer.finalize();

      if (this._cancelled) return null;

      const webmBlob = new Blob([target.buffer], { type: 'video/webm' });

      // Extract audio
      this.onProgress?.(78);
      const audioBlob = await this._extractAudio(startTime, endTime, clips, mediaElements, tracks);
      this.lastExportMeta = {
        path: 'webcodecs',
        fps: exportFPS,
        frames: totalFrames,
        hasAudio: !!audioBlob,
      };

      // Try to mux to MP4 via FFmpeg.wasm
      this.onProgress?.(85);

      if (ExportEngine.isFFmpegAvailable()) {
        const ffmpeg = await this._loadFFmpeg();
        if (ffmpeg) {
          try {
            const webmData = new Uint8Array(await webmBlob.arrayBuffer());
            await ffmpeg.writeFile('input.webm', webmData);

            const ffmpegArgs = ['-i', 'input.webm'];

            if (audioBlob) {
              const audioData = new Uint8Array(await audioBlob.arrayBuffer());
              await ffmpeg.writeFile('audio.wav', audioData);
              ffmpegArgs.push('-i', 'audio.wav');
            }

            ffmpegArgs.push(
              '-c:v', 'copy',
              '-c:a', 'aac',
              '-b:a', `${this.audioBitrate}`,
              '-movflags', '+faststart',
              '-shortest',
              'output.mp4'
            );

            await ffmpeg.exec(ffmpegArgs);

            const mp4Data = await ffmpeg.readFile('output.mp4');
            await ffmpeg.deleteFile('input.webm');
            if (audioBlob) await ffmpeg.deleteFile('audio.wav');
            await ffmpeg.deleteFile('output.mp4');

            this.onProgress?.(100);
            const mp4Blob = new Blob([mp4Data.buffer], { type: 'video/mp4' });
            this.lastExportMeta.container = 'mp4';
            this.onComplete?.(mp4Blob, this.lastExportMeta);
            return mp4Blob;
          } catch {
            // FFmpeg failed, return WebM
          }
        }
      }

      // Return WebM if MP4 conversion failed
      this.onProgress?.(100);
      this.lastExportMeta.container = 'webm';
      this.onComplete?.(webmBlob, this.lastExportMeta);
      return webmBlob;

    } catch (err) {
      this.onError?.(`Export failed: ${err.message}`);
      return null;
    } finally {
      // Restore preview-only visibility behavior after export completes
      if (this.renderEngine) this.renderEngine._exportMode = false;
    }
  }

  /**
   * Fallback export using canvas.captureStream() + MediaRecorder
   */
  async _exportWithMediaRecorder(startTime, endTime, tracks, clips, settings, mediaElements, exportFPS) {
    const fps = exportFPS || this.fps;
    // Export mode: include all tracks regardless of visibility state
    if (this.renderEngine) this.renderEngine._exportMode = true;

    try {
      const stream = this.renderEngine.canvas.captureStream(fps);
      const recorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9',
        videoBitsPerSecond: this.videoBitrate,
      });

      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      return new Promise((resolve) => {
        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'video/webm' });
          this.onProgress?.(100);
          this.lastExportMeta = {
            path: 'mediarecorder',
            fps,
            container: 'webm',
            hasAudio: false,
            reducedFidelity: true,
          };
          this.onComplete?.(blob, this.lastExportMeta);
          resolve(blob);
        };

        recorder.start(100); // Collect data every 100ms

        const totalFrames = Math.ceil((endTime - startTime) * fps);
        const frameTime = 1000 / fps;
        let frame = 0;

        const renderNext = () => {
          if (this._cancelled || frame >= totalFrames) {
            recorder.stop();
            return;
          }

          const time = startTime + frame / fps;
          this.renderEngine.renderFrame(time, tracks, clips, settings, mediaElements);
          frame++;

          if (frame % 10 === 0) {
            this.onProgress?.(Math.round((frame / totalFrames) * 90));
          }

          setTimeout(renderNext, frameTime);
        };

        renderNext();
      });
    } catch (err) {
      this.onError?.(`MediaRecorder export failed: ${err.message}`);
      return null;
    } finally {
      if (this.renderEngine) this.renderEngine._exportMode = false;
    }
  }

  /**
   * Cancel an in-progress export
   */
  cancel() {
    this._cancelled = true;
    // Restore preview-only visibility behavior
    if (this.renderEngine) this.renderEngine._exportMode = false;
  }

  /**
   * Clean up FFmpeg.wasm resources
   */
  async destroy() {
    this.cancel();
    if (this._audioBufferCache) this._audioBufferCache.clear();
    if (this._ffmpeg) {
      try {
        this._ffmpeg.terminate();
      } catch {}
      this._ffmpeg = null;
    }
  }
}
