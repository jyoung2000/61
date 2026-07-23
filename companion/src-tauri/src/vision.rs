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

/// Where a runtime-downloaded vision sidecar lands (mirrors the whisper
/// `whisper-bin` layout). `resolve_binary` checks this dir first, so a
/// downloaded build wins over any bundled one.
pub fn downloaded_vision_dir(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("vision-bin")
}

// The ClipAI repo whose companion release carries the vision-server asset.
// The installers + this asset are published by companion-release.yml on the
// same repo, so the download target is stable (mirrors the whisper.cpp
// download hard-coding its upstream repo).
#[cfg(target_os = "windows")]
const VISION_ASSET: &str = "vision-server-x64.zip";
#[cfg(target_os = "windows")]
const RELEASE_REPO: &str = "jyoung2000/61";

/// Download the vision sidecar (`vision-server-x64.zip`) from the latest
/// companion release and extract it into `app-data/vision-bin`. Windows-only:
/// the asset is a PyInstaller CUDA build. Progress is emitted on the
/// `vision-progress` event, mirroring the whisper download.
#[cfg(target_os = "windows")]
pub async fn download_vision(
    app: &tauri::AppHandle,
    data_dir: &PathBuf,
) -> Result<String, String> {
    use crate::state::quiet_command;
    use futures_util::StreamExt;
    use tauri::Emitter;
    use tokio::io::AsyncWriteExt;

    let emit = |stage: &str, percent: f64, message: &str| {
        let _ = app.emit(
            "vision-progress",
            serde_json::json!({"stage": stage, "percent": percent, "message": message}),
        );
    };
    emit("resolving", -1.0, "Finding the latest vision sidecar…");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .user_agent("clipai-companion")
        .build()
        .map_err(|e| e.to_string())?;

    // Find the newest release that actually carries the vision asset — the
    // companion asset set trails the installers on a fresh release, so scan a
    // page of releases rather than assuming /latest has it.
    let rels: serde_json::Value = client
        .get(format!(
            "https://api.github.com/repos/{RELEASE_REPO}/releases?per_page=20"
        ))
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("release lookup failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("release parse failed: {e}"))?;

    let mut url = String::new();
    let mut tag = String::new();
    if let Some(list) = rels.as_array() {
        'find: for rel in list {
            let this_tag = rel.get("tag_name").and_then(|t| t.as_str()).unwrap_or("");
            for a in rel.get("assets").and_then(|a| a.as_array()).into_iter().flatten() {
                if a.get("name").and_then(|n| n.as_str()) == Some(VISION_ASSET) {
                    url = a
                        .get("browser_download_url")
                        .and_then(|u| u.as_str())
                        .unwrap_or("")
                        .to_string();
                    tag = this_tag.to_string();
                    break 'find;
                }
            }
        }
    }
    if url.is_empty() {
        return Err(format!(
            "no {VISION_ASSET} in the recent {RELEASE_REPO} releases yet — \
             the companion build that publishes it may still be running"
        ));
    }
    log::info!("selected vision asset '{VISION_ASSET}' from {tag}");

    let dl_dir = downloaded_vision_dir(data_dir);
    let _ = std::fs::create_dir_all(&dl_dir);
    let zip_path = dl_dir.join("_download.zip");

    emit("downloading", 0.0, &format!("Downloading vision sidecar {tag}…"));
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;
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

    // The zip may nest the exe one folder deep (dist/vision-server/*). If the
    // top-level exe is missing but a single subdir holds it, flatten it up so
    // resolve_binary finds vision-bin/vision-server.exe.
    if !dl_dir.join("vision-server.exe").is_file() {
        if let Ok(entries) = std::fs::read_dir(&dl_dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() && p.join("vision-server.exe").is_file() {
                    for inner in std::fs::read_dir(&p).into_iter().flatten().flatten() {
                        let dest = dl_dir.join(inner.file_name());
                        let _ = std::fs::rename(inner.path(), dest);
                    }
                    let _ = std::fs::remove_dir_all(&p);
                    break;
                }
            }
        }
    }

    if dl_dir.join("vision-server.exe").is_file() {
        emit("done", 100.0, "Vision sidecar installed");
        Ok(format!("vision sidecar {tag} installed"))
    } else {
        Err("vision-server.exe not found after extraction".into())
    }
}

#[cfg(not(target_os = "windows"))]
pub async fn download_vision(
    _app: &tauri::AppHandle,
    _data_dir: &PathBuf,
) -> Result<String, String> {
    Err("Runtime vision-sidecar download is Windows-only.".into())
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
    // CI-free path: a vision server the user started from source (or that we
    // launched earlier) is already answering on the sidecar port — use it
    // as-is, no packaged binary required. This is how the offload works when
    // the GitHub-release asset can't be built (e.g. Actions disabled) — the
    // user runs sidecars/vision-server/run.{ps1,sh} on this machine.
    if healthy().await {
        return Ok(());
    }
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
