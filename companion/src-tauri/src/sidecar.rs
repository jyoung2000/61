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

use crate::state::{quiet_command, whisper_tier_for_budget, AppState, WHISPER_SIDECAR_PORT};
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

/// Model directory in app data — models download on demand, never bundled.
fn models_dir(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("whisper-models")
}

/// macOS whisper.cpp quant for a tier (int8 tiers map to q5 GGUFs).
fn whispercpp_model_file(model: &str) -> String {
    match model {
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
) -> Result<String, String> {
    let budget = state.effective_budget_gb();
    let (model, compute) = whisper_tier_for_budget(budget);

    let mut guard = state.sidecar.lock().await;
    if let Some(handle) = guard.as_mut() {
        let alive = handle.child.try_wait().map(|s| s.is_none()).unwrap_or(false);
        if alive && handle.model == model && healthy().await {
            return Ok(model.to_string());
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
        "starting whisper sidecar: {} model={model} compute={compute} (budget {budget:.1} GB)",
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
    } else {
        // faster-whisper PyInstaller server — configured via env.
        cmd.env("WHISPER_MODEL", model)
            .env("WHISPER_COMPUTE", compute)
            .env("WHISPER_PORT", WHISPER_SIDECAR_PORT.to_string())
            .env("WHISPER_MODELS_DIR", models.to_string_lossy().as_ref())
            .env("HF_HOME", models.to_string_lossy().as_ref());
    }
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start whisper sidecar: {e}"))?;
    // Die with the Companion — no orphaned sidecar after quit/kill.
    crate::state::bind_child_to_lifetime(&child);
    *guard = Some(SidecarHandle {
        child,
        model: model.to_string(),
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
    // Plain CPU x64 build first (most compatible); BLAS as a fallback.
    let mut url = String::new();
    'outer: for want in ["whisper-bin-x64.zip", "whisper-blas-bin-x64.zip"] {
        for a in &assets {
            if a.get("name").and_then(|n| n.as_str()) == Some(want) {
                url = a.get("browser_download_url").and_then(|u| u.as_str()).unwrap_or("").to_string();
                break 'outer;
            }
        }
    }
    if url.is_empty() {
        return Err("no whisper-bin-x64.zip in the latest whisper.cpp release".into());
    }

    let dl_dir = downloaded_whisper_dir(data_dir);
    let _ = std::fs::create_dir_all(&dl_dir);
    let zip_path = dl_dir.join("_download.zip");

    emit("downloading", 0.0, &format!("Downloading whisper.cpp {tag}…"));
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
            emit("done", 100.0, "Whisper installed");
            Ok(format!("whisper.cpp {tag} installed"))
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

pub async fn shutdown(state: &AppState) {
    if let Some(mut handle) = state.sidecar.lock().await.take() {
        let _ = handle.child.kill().await;
        let _ = handle.child.wait().await;
        log::info!("whisper sidecar stopped");
    }
}

/// Background reaper: stop the sidecar after the configured idle period.
pub fn spawn_idle_reaper(state: Arc<AppState>) {
    // Use Tauri's runtime handle, not `tokio::spawn`: this is called from the
    // synchronous `setup` hook (main thread, no Tokio runtime in context), so
    // a bare `tokio::spawn` panics with "there is no reactor running".
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let idle_min = state.config.lock().unwrap().sidecar_idle_min.max(1) as u64;
            let last = state.last_request_ms.load(Ordering::Relaxed);
            let busy = state.whisper_busy.load(Ordering::Relaxed);
            let has_sidecar = state.sidecar.lock().await.is_some();
            if has_sidecar
                && !busy
                && crate::state::now_ms().saturating_sub(last) > idle_min * 60_000
            {
                log::info!("whisper sidecar idle for {idle_min} min — shutting it down");
                shutdown(&state).await;
            }
        }
    });
}
