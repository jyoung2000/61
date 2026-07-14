//! Whisper sidecar lifecycle.
//!
//! Windows/NVIDIA: a PyInstaller-packaged faster-whisper server
//! (``whisper-server.exe``, built from companion/sidecars/whisper-server).
//! macOS (Apple Silicon): whisper.cpp's ``whisper-server`` with Metal,
//! launched with ``--inference-path /v1/audio/transcriptions``.
//!
//! Both serve the OpenAI transcription schema (verbose_json + word
//! timestamps) on 127.0.0.1 behind the proxy. The sidecar starts lazily on
//! the first transcription request and exits after a configurable idle
//! period so an overnight-idle desktop holds no models and near-zero CPU.

use crate::state::{quiet_command, AppState, WHISPER_SIDECAR_PORT};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::process::Child;

pub struct SidecarHandle {
    pub child: Child,
    pub model: String,
    /// Kept for diagnostics/log correlation (not read on every platform).
    #[allow(dead_code)]
    pub started_ms: u64,
}

pub fn sidecar_url() -> String {
    format!("http://127.0.0.1:{WHISPER_SIDECAR_PORT}")
}

/// Where sidecar binaries live inside the installed app bundle
/// (declared as `resources` in tauri.conf.json).
fn sidecar_dir(resource_dir: &PathBuf) -> PathBuf {
    resource_dir.join("sidecar")
}

fn sidecar_binary(resource_dir: &PathBuf) -> PathBuf {
    let dir = sidecar_dir(resource_dir);
    if cfg!(target_os = "windows") {
        dir.join("whisper-server.exe")
    } else {
        dir.join("whisper-server")
    }
}

/// Where a runtime-DOWNLOADED whisper.cpp server lives (app data). This is how
/// a from-source build (which can't bundle the sidecar) gains local Whisper:
/// the GUI's "Download Whisper" fetches the whisper.cpp release here.
pub fn downloaded_whisper_dir(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("whisper-bin")
}

fn downloaded_whisper_server(data_dir: &PathBuf) -> Option<PathBuf> {
    let dir = downloaded_whisper_dir(data_dir);
    let names = ["whisper-server.exe", "whisper-server", "server.exe", "server"];
    for name in names {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    // whisper-bin-x64.zip may nest the binaries one folder deep — its DLLs sit
    // beside the .exe there, so running it in place still resolves them.
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            if entry.path().is_dir() {
                for name in names {
                    let p = entry.path().join(name);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

/// Resolve the whisper server to run: prefer a downloaded whisper.cpp build
/// (works for from-source installs), else the bundled sidecar. Returns
/// (binary_path, is_whispercpp) — whisper.cpp and faster-whisper take
/// different launch args.
fn resolve_sidecar(resource_dir: &PathBuf, data_dir: &PathBuf) -> Option<(PathBuf, bool)> {
    if let Some(p) = downloaded_whisper_server(data_dir) {
        return Some((p, true)); // downloaded == whisper.cpp on every platform
    }
    let bundled = sidecar_binary(resource_dir);
    if bundled.is_file() {
        // Bundled: macOS ships whisper.cpp, Windows ships faster-whisper.
        return Some((bundled, cfg!(target_os = "macos")));
    }
    None
}

/// Whether a whisper sidecar is usable — bundled OR downloaded. From-source /
/// cross-compiled installers don't bundle one; the user can add it via the
/// GUI's "Download Whisper". Ollama sharing works regardless; health + pairing
/// report whisper honestly so ClipAI keeps transcription local until it's here.
pub fn available(resource_dir: &PathBuf, data_dir: &PathBuf) -> bool {
    resolve_sidecar(resource_dir, data_dir).is_some()
}

/// Which whisper build is installed, so the GUI can warn when transcription
/// would run on the CPU (unusably slow for large models) and offer the GPU one.
/// Returns "gpu", "cpu", "bundled", or "none".
///   - "gpu"     — a downloaded whisper.cpp CUDA (cuBLAS) build: CUDA runtime
///                 DLLs sit beside the server, so it offloads to the NVIDIA GPU.
///   - "cpu"     — a downloaded whisper.cpp build with no CUDA DLLs.
///   - "bundled" — the sidecar shipped with an official release (GPU-capable).
///   - "none"    — no sidecar available.
pub fn build_kind(resource_dir: &PathBuf, data_dir: &PathBuf) -> &'static str {
    if let Some(server) = downloaded_whisper_server(data_dir) {
        // CUDA builds ship cudart/cublas/ggml-cuda DLLs next to the server exe.
        let dir = server.parent().map(|p| p.to_path_buf())
            .unwrap_or_else(|| downloaded_whisper_dir(data_dir));
        let has_cuda = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.flatten().any(|e| {
                    let n = e.file_name().to_string_lossy().to_lowercase();
                    n.ends_with(".dll")
                        && (n.contains("cudart") || n.contains("cublas") || n.contains("cuda"))
                })
            })
            .unwrap_or(false);
        return if has_cuda { "gpu" } else { "cpu" };
    }
    if sidecar_binary(resource_dir).is_file() {
        return "bundled";
    }
    "none"
}

/// Model directory in app data — models download on demand, never bundled.
fn models_dir(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("whisper-models")
}

/// macOS whisper.cpp quant for a tier (int8 tiers map to q5 GGUFs).
fn whispercpp_model_file(model: &str) -> String {
    match model {
        // Full large-v3 (highest accuracy — the "max" quality tier).
        "large-v3" => "ggml-large-v3.bin".into(),
        "large-v3-turbo" => "ggml-large-v3-turbo.bin".into(),
        "medium" => "ggml-medium-q5_0.bin".into(),
        _ => "ggml-small-q5_1.bin".into(),
    }
}

/// whisper.cpp has no on-demand model download (unlike faster-whisper) —
/// fetch the GGML weights from the official Hugging Face mirror on first
/// use. Streams to a `.part` file and renames atomically so a killed
/// download never leaves a truncated model in place.
async fn ensure_whispercpp_model(
    models_dir: &std::path::Path,
    model: &str,
) -> Result<PathBuf, String> {
    let file = whispercpp_model_file(model);
    let dest = models_dir.join(&file);
    if dest.exists() {
        return Ok(dest);
    }
    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}");
    log::info!("downloading whisper.cpp model {file} (first use) from {url}");
    let part = models_dir.join(format!("{file}.part"));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("model download failed to start: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("model download failed: HTTP {}", resp.status()));
    }
    let mut out = tokio::fs::File::create(&part)
        .await
        .map_err(|e| format!("could not create {}: {e}", part.display()))?;
    let mut stream = resp.bytes_stream();
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("model download interrupted: {e}"))?;
        out.write_all(&chunk)
            .await
            .map_err(|e| format!("model write failed: {e}"))?;
    }
    out.flush().await.map_err(|e| e.to_string())?;
    drop(out);
    tokio::fs::rename(&part, &dest)
        .await
        .map_err(|e| format!("could not finalize model file: {e}"))?;
    log::info!("whisper.cpp model ready: {}", dest.display());
    Ok(dest)
}

pub async fn healthy() -> bool {
    reqwest::Client::new()
        .get(format!("{}/health", sidecar_url()))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .is_ok()
}

/// Ensure the sidecar is running with the model tier the current VRAM
/// budget allows. Restarts it when the tier changed. Returns the model
/// in service.
pub async fn ensure_running(
    state: &Arc<AppState>,
    resource_dir: PathBuf,
    data_dir: PathBuf,
    requested_model: &str,
) -> Result<String, String> {
    let budget = state.effective_budget_gb();
    // Honor the model ClipAI selected (synced per request) but cap by budget so
    // it always fits on the GPU. Empty request ⇒ the budget-default tier.
    let (mut model, compute) = crate::state::whisper_tier_for_request(requested_model, budget);
    // Transcription quality: beam search (accuracy) + optionally the full
    // large-v3 model, scaled to the VRAM budget. This is where the extra VRAM
    // buys Netflix/YouTube-grade captions.
    let quality = state.config.lock().unwrap().whisper_quality.clone();
    let (beam_size, prefer_full) = crate::state::whisper_quality_params(&quality, budget);
    if prefer_full && model == "large-v3-turbo" {
        model = "large-v3";
    }
    let is_gpu_build = build_kind(&resource_dir, &data_dir) == "gpu";
    // Tuned decode parity with the ClipAI backend's local faster-whisper path
    // (see backend/services/reframer_audio.py: _vad_parameters /
    // _decoding_kwargs / WHISPER_NO_SPEECH_THRESHOLD). Passed to the
    // faster-whisper sidecar as env vars — zero protocol risk since we launch
    // that binary ourselves; server.py feature-detects each value against the
    // installed faster-whisper and a per-request form field still overrides.
    // whisper.cpp ignores these (it has its own CLI flags above).
    let fw_tuning: [(&str, String); 8] = [
        ("WHISPER_BEAM", beam_size.to_string()),
        ("WHISPER_VAD_ONSET", "0.10".into()),
        ("WHISPER_VAD_MIN_SILENCE_MS", "300".into()),
        ("WHISPER_VAD_SPEECH_PAD_MS", "150".into()),
        ("WHISPER_NO_SPEECH_THRESHOLD", "0.4".into()),
        ("WHISPER_COND_PREV", "0".into()),
        ("WHISPER_NO_REPEAT_NGRAM", "3".into()),
        ("WHISPER_HALLUCINATION_SILENCE_S", "2.0".into()),
    ];
    // A restart is needed when the model OR the decode settings change, so the
    // service key folds both in — including the env tuning, so editing the
    // values above (or a Companion upgrade that introduces them) retires a
    // sidecar still running the old decode parameters.
    let tuning_key = fw_tuning
        .iter()
        .map(|(_, v)| v.as_str())
        .collect::<Vec<_>>()
        .join(",");
    // mc0 = whisper.cpp --max-context 0 (folded in so upgraded Companions
    // retire a sidecar still running the loop-prone contextual decode).
    let service_key = format!("{model}|bs{beam_size}|mc0|tune:{tuning_key}");

    let mut guard = state.sidecar.lock().await;
    if let Some(handle) = guard.as_mut() {
        let alive = handle.child.try_wait().map(|s| s.is_none()).unwrap_or(false);
        if alive && handle.model == service_key && healthy().await {
            return Ok(model.to_string());
        }
        // Say WHY the old sidecar is being retired — this restart used to be
        // silent, which made an unexplained "whisper sidecar stopped" in the
        // log impossible to attribute.
        if !alive {
            log::warn!(
                "whisper sidecar process had EXITED on its own (crash?) — starting a fresh one"
            );
        } else if handle.model != service_key {
            log::info!(
                "restarting whisper sidecar: decode settings changed ({} → {service_key})",
                handle.model
            );
        } else {
            log::warn!("whisper sidecar unresponsive to /health — restarting it");
        }
        let _ = handle.child.kill().await;
        *guard = None;
    }

    let (binary, is_whispercpp) = resolve_sidecar(&resource_dir, &data_dir).ok_or_else(|| {
        "no whisper sidecar available — use \"Download Whisper\" in the Companion, \
         or install a build that bundles it".to_string()
    })?;
    let models = models_dir(&data_dir);
    let _ = std::fs::create_dir_all(&models);

    log::info!(
        "starting whisper sidecar: {} model={model} compute={compute} \
         quality={quality} beam_size={beam_size} flash_attn={is_gpu_build} (budget {budget:.1} GB)",
        if is_whispercpp { "whisper.cpp" } else { "faster-whisper" }
    );
    let mut cmd = quiet_command(&binary);
    if is_whispercpp {
        // whisper.cpp server CLI — weights must exist before launch.
        let model_file = ensure_whispercpp_model(&models, model).await?;
        cmd.args([
            "--host",
            "127.0.0.1",
            "--port",
            &WHISPER_SIDECAR_PORT.to_string(),
            "--inference-path",
            "/v1/audio/transcriptions",
            "--model",
            model_file.to_string_lossy().as_ref(),
        ]);
        // Beam search — the main accuracy lever over greedy decoding. Costs more
        // compute/VRAM, which is exactly what the extra card affords.
        if beam_size > 1 {
            cmd.arg("--beam-size").arg(beam_size.to_string());
        }
        // Anti-loop parity with the backend's tuned decode
        // (condition_on_previous_text=False): whisper.cpp conditions each
        // window on the previous windows' text (prompt_past), which drives the
        // classic repeated-line hallucination on music-heavy material — a
        // 128-min anime run returned 257 segments of which 254 were the SAME
        // line, and the server filtered the transcript to zero.
        // --max-context 0 zeroes that conditioning. NOTE: --no-context is NOT
        // a whisper-server CLI flag (unknown args make it print usage and
        // exit 0, silently killing the sidecar) and the server's no_context
        // param wouldn't stop within-file conditioning anyway — -mc 0 is the
        // real lever, and it parses on every shipped build (verified against
        // v1.7.4, the bundled macOS build, and v1.9.1, the Windows download).
        cmd.arg("--max-context").arg("0");
        // Flash attention on the CUDA build: faster + lower memory, numerically
        // exact (no quality trade-off) — lets the bigger model + beam fit.
        if is_gpu_build {
            cmd.arg("--flash-attn");
        }
    } else {
        // faster-whisper PyInstaller server — configured via env.
        cmd.env("WHISPER_MODEL", model)
            .env("WHISPER_COMPUTE", compute)
            .env("WHISPER_PORT", WHISPER_SIDECAR_PORT.to_string())
            .env("WHISPER_MODELS_DIR", models.to_string_lossy().as_ref())
            .env("HF_HOME", models.to_string_lossy().as_ref());
        // Decode parity with the backend's tuned local path (folded into
        // service_key above so a change here restarts the sidecar).
        for (key, value) in &fw_tuning {
            cmd.env(key, value);
        }
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start whisper sidecar: {e}"))?;
    // Die with the Companion — no orphaned sidecar after quit/kill.
    crate::state::bind_child_to_lifetime(&child);
    *guard = Some(SidecarHandle {
        child,
        // Store the model+decode key so a quality change (beam size / full
        // model) is detected and triggers a restart on the next request.
        model: service_key,
        started_ms: crate::state::now_ms(),
    });
    drop(guard);

    // First start may download the model — allow a generous warmup.
    for _ in 0..600 {
        if healthy().await {
            return Ok(model.to_string());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err("whisper sidecar started but never became healthy (model download may have failed)".into())
}

/// Download the latest whisper.cpp prebuilt server into app data so a
/// from-source Companion (which can't bundle the sidecar) gains local Whisper.
/// Windows only — macOS builds bundle whisper.cpp. Emits `whisper-progress`.
#[cfg(target_os = "windows")]
pub async fn download_whispercpp(
    app: &tauri::AppHandle,
    data_dir: &PathBuf,
) -> Result<String, String> {
    use futures_util::StreamExt;
    use tauri::Emitter;
    use tokio::io::AsyncWriteExt;

    let emit = |stage: &str, percent: f64, message: &str| {
        let _ = app.emit(
            "whisper-progress",
            serde_json::json!({"stage": stage, "percent": percent, "message": message}),
        );
    };
    emit("resolving", -1.0, "Finding the latest whisper.cpp release…");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .user_agent("clipai-companion")
        .build()
        .map_err(|e| e.to_string())?;

    let rel: serde_json::Value = client
        .get("https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest")
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("release lookup failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("release parse failed: {e}"))?;
    let tag = rel.get("tag_name").and_then(|t| t.as_str()).unwrap_or("").to_string();
    let assets = rel.get("assets").and_then(|a| a.as_array()).cloned().unwrap_or_default();
    let asset_name = |a: &serde_json::Value| {
        a.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string()
    };
    let asset_url = |a: &serde_json::Value| {
        a.get("browser_download_url").and_then(|u| u.as_str()).unwrap_or("").to_string()
    };

    // Prefer a CUDA (cuBLAS) build when an NVIDIA GPU is present so transcription
    // runs ON THE GPU. A CPU build makes large-v3-turbo unusably slow — minutes
    // per clip, so even the tiny verify clip times out. whisper.cpp bundles the
    // CUDA runtime DLLs inside the cublas zip, so it runs in place without a
    // separate CUDA toolkit install. Falls back to the CPU build otherwise.
    let is_nvidia = crate::gpu::snapshot().gpu_name.to_lowercase().contains("nvidia");
    let mut url = String::new();
    let mut chosen = String::new();
    let mut gpu_build = false;
    if is_nvidia {
        // Newest cuBLAS x64 asset (e.g. whisper-cublas-12.4.0-bin-x64.zip).
        // Lexicographic max over the name is a good-enough "highest CUDA build".
        let mut best: Option<&serde_json::Value> = None;
        for a in &assets {
            let n = asset_name(a);
            let nl = n.to_lowercase();
            if nl.contains("cublas") && nl.contains("x64") && nl.ends_with(".zip") {
                if best.map(|b| asset_name(b) < n).unwrap_or(true) {
                    best = Some(a);
                }
            }
        }
        if let Some(a) = best {
            url = asset_url(a);
            chosen = asset_name(a);
            gpu_build = true;
        }
    }
    if url.is_empty() {
        // CPU fallback — plain build first (most compatible), then BLAS.
        'outer: for want in ["whisper-bin-x64.zip", "whisper-blas-bin-x64.zip"] {
            for a in &assets {
                if asset_name(a) == want {
                    url = asset_url(a);
                    chosen = want.to_string();
                    break 'outer;
                }
            }
        }
    }
    if url.is_empty() {
        return Err("no whisper x64 build in the latest whisper.cpp release".into());
    }
    log::info!(
        "selected whisper.cpp asset '{chosen}' ({}) from {tag}",
        if gpu_build { "CUDA/GPU" } else { "CPU" }
    );

    let dl_dir = downloaded_whisper_dir(data_dir);
    let _ = std::fs::create_dir_all(&dl_dir);
    let zip_path = dl_dir.join("_download.zip");

    emit("downloading", 0.0, &format!(
        "Downloading whisper.cpp {tag} ({} build)…",
        if gpu_build { "GPU / CUDA" } else { "CPU" }
    ));
    let resp = client.get(&url).send().await.map_err(|e| format!("download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&zip_path).await.map_err(|e| e.to_string())?;
    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("download interrupted: {e}"))?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        downloaded += chunk.len() as u64;
        if total > 0 {
            emit("downloading", downloaded as f64 / total as f64 * 100.0, "Downloading…");
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);

    emit("extracting", -1.0, "Extracting…");
    let status = quiet_command("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!(
                "Expand-Archive -Force -LiteralPath '{}' -DestinationPath '{}'",
                zip_path.display(),
                dl_dir.display()
            ),
        ])
        .status()
        .await
        .map_err(|e| format!("extraction failed to start: {e}"))?;
    let _ = std::fs::remove_file(&zip_path);
    if !status.success() {
        return Err("extraction failed (Expand-Archive)".into());
    }

    match downloaded_whisper_server(data_dir) {
        Some(_) => {
            let kind = if gpu_build { "GPU / CUDA" } else { "CPU" };
            emit("done", 100.0, &format!("Whisper installed ({kind} build)"));
            Ok(format!("whisper.cpp {tag} installed ({kind} build)"))
        }
        None => Err("whisper-server.exe not found after extraction".into()),
    }
}

#[cfg(not(target_os = "windows"))]
pub async fn download_whispercpp(
    _app: &tauri::AppHandle,
    _data_dir: &PathBuf,
) -> Result<String, String> {
    Err("Runtime Whisper download is Windows-only — macOS builds bundle whisper.cpp.".into())
}

pub async fn shutdown(state: &AppState, reason: &str) {
    if let Some(mut handle) = state.sidecar.lock().await.take() {
        let _ = handle.child.kill().await;
        let _ = handle.child.wait().await;
        // Every stop carries its cause: an unattributed bare "whisper sidecar
        // stopped" in a live log proved undiagnosable (2026-07-11 04:10:57Z).
        log::info!("whisper sidecar stopped ({reason})");
    }
}

/// Free everything ClipAI-related on the GPU: stop the whisper sidecar (only
/// when no decode holds the slot — an in-flight transcription always wins)
/// and evict every resident Ollama model (keep_alive=0 per model). Safe to
/// call any time; both halves are no-ops when there's nothing to free.
/// Returns ``(whisper_stopped, ollama_models_unloaded)``.
pub async fn free_gpu(state: &Arc<AppState>, reason: &str) -> (bool, usize) {
    let mut whisper_stopped = false;
    if let Ok(permit) = state.whisper_slot.try_acquire() {
        whisper_stopped = state.sidecar.lock().await.is_some();
        if whisper_stopped {
            shutdown(state, reason).await;
        }
        drop(permit);
    }
    crate::vision::shutdown(state, reason).await;
    let (n, names) = crate::ollama::unload_all().await;
    if whisper_stopped || n > 0 {
        log::info!(
            "GPU freed ({reason}): whisper_stopped={whisper_stopped}, ollama_unloaded={n}{}",
            if n > 0 {
                format!(" [{}]", names.join(", "))
            } else {
                String::new()
            }
        );
    }
    (whisper_stopped, n)
}

/// Whole-GPU idle-free window in milliseconds from the two config knobs. The
/// seconds knob is primary (fast, default 45 s); the minutes knob is the coarse
/// fallback used only when seconds is 0. Both 0 → 0 (auto-free disabled).
pub(crate) fn effective_free_ms(free_sec: u32, free_min: u32) -> u64 {
    if free_sec > 0 {
        free_sec as u64 * 1_000
    } else {
        free_min as u64 * 60_000
    }
}

/// Background reaper: return the GPU to the desktop when ClipAI goes quiet.
///
/// Two timers, both keyed on REAL work (``real_work_snapshot``) — the old
/// implementation keyed on ``last_request_ms``, which every /v1/health and
/// /api/tags PROBE refreshes; with the GUI open or ClipAI polling its host
/// registry, "idle" never elapsed and the whisper sidecar sat on VRAM
/// indefinitely. That is the "models linger on the GPU after the test" bug.
///
///   * ``gpu_idle_free_sec`` (default 45, primary) / ``gpu_idle_free_min``
///     (coarse fallback, default 3): after that idle window with no
///     transcription/inference running or finishing, free the WHOLE GPU —
///     whisper sidecar stopped AND all resident Ollama models evicted. The
///     seconds knob wins when > 0 (see ``effective_free_ms``); both 0 disables
///     it. Fires once per idle period (marker), so it never churns /api/ps.
///   * ``sidecar_idle_min`` (default 15): whisper-only backstop for setups
///     that disable the full auto-free.
pub fn spawn_idle_reaper(state: Arc<AppState>) {
    // Use Tauri's runtime handle, not `tokio::spawn`: this is called from the
    // synchronous `setup` hook (main thread, no Tokio runtime in context), so
    // a bare `tokio::spawn` panics with "there is no reactor running".
    tauri::async_runtime::spawn(async move {
        loop {
            // Poll every 10 s so a sub-minute idle-free window is honored
            // promptly — while idle these ticks are cheap in-memory snapshot
            // reads, and the free itself fires once per idle period (marker).
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            let (idle_min, free_ms) = {
                let cfg = state.config.lock().unwrap();
                (
                    cfg.sidecar_idle_min.max(1) as u64,
                    effective_free_ms(cfg.gpu_idle_free_sec, cfg.gpu_idle_free_min),
                )
            };
            let (inflight, last_real) = state.real_work_snapshot();
            if inflight {
                continue;
            }
            // An ACTIVE ClipAI job (fresh /v1/progress heartbeats, sent every
            // ~1.5 s while a job runs) means the GPU is about to be needed —
            // do NOT idle-free mid-job. The observed failure: the 3-min free
            // fired during the container's local extraction stage, evicting
            // the models (and the pre-warmed whisper sidecar) minutes before
            // the pipeline needed them, forcing cold reloads mid-job. ClipAI
            // still hands VRAM around explicitly (/v1/sidecar/release after
            // transcription), and the job-ended grace free + these reapers
            // (heartbeats go stale 45 s after a dead container) remain the
            // cleanup for every other case.
            if state.reported_job_fresh().is_some() {
                continue;
            }
            let now = crate::state::now_ms();

            // Whole-GPU auto-free (whisper + Ollama), once per idle period.
            if free_ms > 0
                && last_real > 0
                && now.saturating_sub(last_real) > free_ms
                && state.last_gpu_free_marker.load(Ordering::Relaxed) != last_real
            {
                state
                    .last_gpu_free_marker
                    .store(last_real, Ordering::Relaxed);
                let secs = free_ms / 1_000;
                free_gpu(&state, &format!("idle {secs}s")).await;
                continue;
            }

            // Whisper-only backstop (kept for gpu_idle_free_min=0 setups).
            let has_sidecar = state.sidecar.lock().await.is_some();
            if has_sidecar
                && last_real > 0
                && now.saturating_sub(last_real) > idle_min * 60_000
            {
                shutdown(&state, &format!("idle {idle_min} min backstop")).await;
            }
        }
    });
}

#[cfg(test)]
mod free_window_tests {
    use super::effective_free_ms;

    #[test]
    fn seconds_knob_is_primary() {
        // Default: 45 s free window (prompt free once no job/test is running).
        assert_eq!(effective_free_ms(45, 3), 45_000);
        assert_eq!(effective_free_ms(10, 3), 10_000);
    }

    #[test]
    fn falls_back_to_minutes_when_seconds_zero() {
        assert_eq!(effective_free_ms(0, 3), 180_000);
        assert_eq!(effective_free_ms(0, 1), 60_000);
    }

    #[test]
    fn both_zero_disables_auto_free() {
        assert_eq!(effective_free_ms(0, 0), 0);
    }
}
