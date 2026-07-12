//! Vision sidecar lifecycle (face-detection offload).
//!
//! A PyInstaller-packaged YOLO-World server (`vision-server[.exe]`) serving
//! `/v1/vision/detect` on localhost behind the proxy. Same contract as the
//! whisper sidecar: starts lazily on the first request, dies with the app,
//! is stopped by `free_gpu`. When no binary is installed (older installers,
//! from-source builds) every route answers 404 and ClipAI silently keeps
//! face detection local — the offload is strictly additive.

use crate::state::{quiet_command, AppState};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Child;

pub const VISION_SIDECAR_PORT: u16 = 11511;

pub fn vision_url() -> String {
    format!("http://127.0.0.1:{VISION_SIDECAR_PORT}")
}

fn resolve_binary(resource_dir: &PathBuf, data_dir: &PathBuf) -> Option<PathBuf> {
    let names: [&str; 2] = if cfg!(target_os = "windows") {
        ["vision-server.exe", "vision-server.exe"]
    } else {
        ["vision-server", "vision-server"]
    };
    let downloaded = data_dir.join("vision-bin").join(names[0]);
    if downloaded.is_file() {
        return Some(downloaded);
    }
    let bundled = resource_dir.join("sidecar").join(names[1]);
    if bundled.is_file() {
        return Some(bundled);
    }
    None
}

pub fn available(resource_dir: &PathBuf, data_dir: &PathBuf) -> bool {
    resolve_binary(resource_dir, data_dir).is_some()
}

pub async fn healthy() -> bool {
    reqwest::Client::new()
        .get(format!("{}/health", vision_url()))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Start (or confirm) the vision sidecar. Returns Err when no binary exists.
pub async fn ensure_running(
    state: &Arc<AppState>,
    resource_dir: PathBuf,
    data_dir: PathBuf,
) -> Result<(), String> {
    let mut guard = state.vision_sidecar.lock().await;
    if let Some(child) = guard.as_mut() {
        let alive = child.try_wait().map(|s| s.is_none()).unwrap_or(false);
        if alive && healthy().await {
            return Ok(());
        }
        if !alive {
            log::warn!("vision sidecar process had EXITED on its own — starting a fresh one");
        }
        let _ = child.kill().await;
        *guard = None;
    }
    let binary = resolve_binary(&resource_dir, &data_dir)
        .ok_or_else(|| "no vision sidecar installed".to_string())?;
    log::info!("starting vision sidecar: {} (port {VISION_SIDECAR_PORT})", binary.display());
    let mut cmd = quiet_command(&binary);
    cmd.env("VISION_PORT", VISION_SIDECAR_PORT.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child: Child = cmd
        .spawn()
        .map_err(|e| format!("could not start vision sidecar: {e}"))?;
    crate::state::bind_child_to_lifetime(&child);
    *guard = Some(child);
    drop(guard);
    // First start loads YOLO-World weights — allow a generous warmup.
    for _ in 0..240 {
        if healthy().await {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err("vision sidecar started but never became healthy".into())
}

pub async fn shutdown(state: &AppState, reason: &str) {
    if let Some(mut child) = state.vision_sidecar.lock().await.take() {
        let _ = child.kill().await;
        let _ = child.wait().await;
        log::info!("vision sidecar stopped ({reason})");
    }
}

/// Whether the vision offload is allowed under the user's speed settings.
///
/// * "eco" reserves the card for other apps — vision stays on the ClipAI
///   server (exactly the pre-offload behavior).
/// * The VRAM budget must leave room for YOLO-World (~1 GB) NEXT TO a
///   concurrent whisper decode (~3 GB fp16 turbo) — below 5 GB the offload
///   would fight the transcription for memory, so it stays local.
pub fn allowed(profile: &str, budget_gb: f32) -> bool {
    profile != "eco" && budget_gb >= 5.0
}

#[cfg(test)]
mod tests {
    use super::allowed;

    #[test]
    fn eco_profile_keeps_vision_local() {
        assert!(!allowed("eco", 12.0));
    }

    #[test]
    fn small_budget_keeps_vision_local() {
        assert!(!allowed("turbo", 4.0));
    }

    #[test]
    fn auto_and_turbo_allow_vision_with_room() {
        assert!(allowed("auto", 9.5));
        assert!(allowed("turbo", 9.5));
        assert!(allowed("balanced", 6.0));
    }
}
