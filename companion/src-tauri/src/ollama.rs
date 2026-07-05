//! Ollama lifecycle: detect an existing install, drive a GUI install
//! (winget / brew), and run a MANAGED `ollama serve` bound to localhost
//! only — the Companion proxy is the single LAN-exposed surface.
//!
//! Ollama itself is never bundled or reimplemented: it installs from the
//! official channels (https://ollama.com/download).

use crate::state::{AppState, OLLAMA_LOCAL};
use serde::Serialize;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Command;

#[derive(Serialize, Clone, Default)]
pub struct OllamaStatus {
    pub installed: bool,
    pub version: String,
    pub running: bool,
    pub managed: bool,
    pub models: Vec<String>,
}

fn ollama_binary() -> &'static str {
    if cfg!(target_os = "windows") {
        "ollama.exe"
    } else {
        "ollama"
    }
}

pub async fn detect_version() -> Option<String> {
    let out = Command::new(ollama_binary())
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub async fn daemon_running() -> bool {
    reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/version"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .is_ok()
}

pub async fn list_models() -> Vec<String> {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/tags"))
        .timeout(std::time::Duration::from_secs(4))
        .send()
        .await
    else {
        return vec![];
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return vec![];
    };
    json["models"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["name"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

pub async fn status(state: &AppState) -> OllamaStatus {
    let version = detect_version().await;
    let running = daemon_running().await;
    let managed = state.ollama_child.lock().await.is_some();
    OllamaStatus {
        installed: version.is_some() || running,
        version: version.unwrap_or_default(),
        running,
        managed,
        models: if running { list_models().await } else { vec![] },
    }
}

/// One-click install. Windows: winget (official Ollama package).
/// macOS: Homebrew when present. Returns a human-readable transcript;
/// the GUI falls back to an "open ollama.com/download" button when this
/// reports failure.
pub async fn install() -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        let out = Command::new("winget")
            .args([
                "install",
                "--id",
                "Ollama.Ollama",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--silent",
            ])
            .output()
            .await
            .map_err(|e| format!("winget not available: {e}"))?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if out.status.success() {
            return Ok(text);
        }
        return Err(format!("winget install failed:\n{text}"));
    }
    #[cfg(target_os = "macos")]
    {
        // brew when available; otherwise the caller opens the download page.
        let brew = Command::new("brew").arg("--version").output().await;
        if brew.map(|o| o.status.success()).unwrap_or(false) {
            let out = Command::new("brew")
                .args(["install", "ollama"])
                .output()
                .await
                .map_err(|e| format!("brew failed to start: {e}"))?;
            let text = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            if out.status.success() {
                return Ok(text);
            }
            return Err(format!("brew install failed:\n{text}"));
        }
        return Err("Homebrew not found — use the download page instead.".into());
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        Err("Automated install is only wired for Windows/macOS — see https://ollama.com/download".into())
    }
}

/// GPU-overhead reservation: everything ABOVE the user's budget is held
/// back from Ollama so the rest of the card stays free for the desktop.
fn gpu_overhead_bytes(state: &AppState) -> u64 {
    let total_mb = state.gpu.lock().unwrap().vram_total_mb;
    if total_mb == 0 {
        return 0;
    }
    let budget_mb = (state.effective_budget_gb() * 1024.0) as u64;
    (total_mb.saturating_sub(budget_mb)) * 1024 * 1024
}

/// Start a managed `ollama serve` bound to 127.0.0.1 with the Companion's
/// resource policy. No-op when a daemon is already answering (external
/// install, e.g. the Ollama tray app) — we use it as-is.
pub async fn ensure_running(state: &Arc<AppState>) -> Result<bool, String> {
    if daemon_running().await {
        return Ok(false);
    }
    let keep_alive = state.config.lock().unwrap().ollama_keep_alive.clone();
    let overhead = gpu_overhead_bytes(state);
    log::info!(
        "starting managed ollama (keep_alive={keep_alive}, gpu_overhead={} MB)",
        overhead / 1024 / 1024
    );
    let mut cmd = Command::new(ollama_binary());
    cmd.arg("serve")
        .env("OLLAMA_HOST", OLLAMA_LOCAL) // localhost ONLY — proxy is the LAN surface
        .env("OLLAMA_MAX_LOADED_MODELS", "1")
        .env("OLLAMA_NUM_PARALLEL", "1")
        .env("OLLAMA_KEEP_ALIVE", keep_alive)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if overhead > 0 {
        cmd.env("OLLAMA_GPU_OVERHEAD", overhead.to_string());
    }
    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start ollama serve: {e}"))?;
    *state.ollama_child.lock().await = Some(child);

    // Wait for the daemon to come up (up to ~15 s).
    for _ in 0..30 {
        if daemon_running().await {
            return Ok(true);
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err("ollama serve started but the daemon never answered".into())
}

/// Restart the MANAGED daemon (budget/keep-alive change). An external
/// daemon is left alone — we can't know what else depends on it.
pub async fn restart(state: &Arc<AppState>) -> Result<bool, String> {
    let mut guard = state.ollama_child.lock().await;
    if let Some(mut child) = guard.take() {
        let _ = child.kill().await;
        let _ = child.wait().await;
        drop(guard);
        // Give the port a moment to free up.
        tokio::time::sleep(std::time::Duration::from_millis(750)).await;
        ensure_running(state).await
    } else {
        Ok(false) // external daemon — settings apply to the next managed start
    }
}

pub async fn shutdown(state: &AppState) {
    if let Some(mut child) = state.ollama_child.lock().await.take() {
        let _ = child.kill().await;
        let _ = child.wait().await;
    }
}

/// Recommended model pulls by VRAM budget, offered by the setup wizard.
pub fn recommended_models(budget_gb: f32) -> Vec<(&'static str, &'static str)> {
    if budget_gb >= 10.0 {
        vec![
            ("llava:13b", "Vision (Primary AI)"),
            ("qwen2.5:14b", "Editorial / translation"),
        ]
    } else if budget_gb >= 6.0 {
        vec![
            ("llava:7b", "Vision (Primary AI)"),
            ("qwen2.5:7b-instruct", "Editorial / translation"),
        ]
    } else {
        vec![
            ("moondream:1.8b", "Vision (Primary AI)"),
            ("qwen2.5:3b-instruct", "Editorial / translation"),
        ]
    }
}
