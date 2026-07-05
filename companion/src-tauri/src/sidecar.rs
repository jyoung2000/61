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

use crate::state::{whisper_tier_for_budget, AppState, WHISPER_SIDECAR_PORT};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::process::{Child, Command};

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

/// Whether this build shipped a whisper sidecar at all. From-source /
/// cross-compiled installers may not bundle one — Ollama sharing still
/// works; health + pairing report whisper honestly so ClipAI keeps its
/// transcription local instead of probing a dead endpoint.
pub fn available(resource_dir: &PathBuf) -> bool {
    sidecar_binary(resource_dir).exists()
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

    let binary = sidecar_binary(&resource_dir);
    if !binary.exists() {
        return Err(format!(
            "whisper sidecar binary not found at {} — reinstall the Companion",
            binary.display()
        ));
    }
    let models = models_dir(&data_dir);
    let _ = std::fs::create_dir_all(&models);

    log::info!("starting whisper sidecar: model={model} compute={compute} (budget {budget:.1} GB)");
    let mut cmd = Command::new(&binary);
    if cfg!(target_os = "macos") {
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
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start whisper sidecar: {e}"))?;
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

pub async fn shutdown(state: &AppState) {
    if let Some(mut handle) = state.sidecar.lock().await.take() {
        let _ = handle.child.kill().await;
        let _ = handle.child.wait().await;
        log::info!("whisper sidecar stopped");
    }
}

/// Background reaper: stop the sidecar after the configured idle period.
pub fn spawn_idle_reaper(state: Arc<AppState>) {
    tokio::spawn(async move {
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
