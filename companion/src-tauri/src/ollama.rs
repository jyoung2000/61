//! Ollama lifecycle: detect an existing install, drive a GUI install
//! (winget / brew), and run a MANAGED `ollama serve` bound to localhost
//! only — the Companion proxy is the single LAN-exposed surface.
//!
//! Ollama itself is never bundled or reimplemented: it installs from the
//! official channels (https://ollama.com/download).

use crate::state::{quiet_command, AppState, OLLAMA_LOCAL};
use serde::Serialize;
use std::process::Stdio;
use std::sync::Arc;

#[derive(Serialize, Clone, Default)]
pub struct OllamaStatus {
    pub installed: bool,
    pub version: String,
    pub running: bool,
    pub managed: bool,
    pub models: Vec<String>,
}

/// Resolve the Ollama executable. Prefer a KNOWN INSTALL PATH over the bare
/// name so we still find it right after a fresh install — the installer adds
/// Ollama to PATH, but our already-running process has a stale PATH and would
/// otherwise fail with "program not found" until the app is restarted.
fn ollama_binary() -> String {
    #[cfg(target_os = "windows")]
    {
        let mut candidates: Vec<String> = Vec::new();
        // Ollama's Windows installer is per-user → %LOCALAPPDATA%\Programs\Ollama.
        for var in ["LOCALAPPDATA", "ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
            if let Ok(base) = std::env::var(var) {
                candidates.push(format!("{base}\\Programs\\Ollama\\ollama.exe"));
                candidates.push(format!("{base}\\Ollama\\ollama.exe"));
            }
        }
        for c in candidates {
            if std::path::Path::new(&c).exists() {
                return c;
            }
        }
        "ollama.exe".to_string()
    }
    #[cfg(not(target_os = "windows"))]
    {
        for c in [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
        ] {
            if std::path::Path::new(c).exists() {
                return c.to_string();
            }
        }
        "ollama".to_string()
    }
}

pub async fn detect_version() -> Option<String> {
    let out = quiet_command(ollama_binary())
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

/// Number of models currently resident in VRAM (via /api/ps). 0 means Ollama
/// holds no GPU memory, so free VRAM reflects only other apps — the moment to
/// measure the auto-VRAM baseline. Fail-open to 0.
pub async fn loaded_model_count() -> usize {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/ps"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    else {
        return 0;
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return 0;
    };
    json["models"].as_array().map(|a| a.len()).unwrap_or(0)
}

/// VRAM (bytes) held by resident Ollama models right now — the ClipAI portion
/// of GPU memory, via /api/ps `size_vram`. Fail-open to 0 so the bar just shows
/// no ClipAI segment if Ollama is briefly unreachable.
pub async fn loaded_vram_bytes() -> u64 {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/ps"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    else {
        return 0;
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return 0;
    };
    json["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("size_vram").and_then(|v| v.as_u64()))
                .sum()
        })
        .unwrap_or(0)
}

/// The models CURRENTLY resident in VRAM (actively loaded/running), one entry
/// per model from /api/ps: `{name, vram_mb, expires_at}`. Powers the "what AI
/// is running right now" readout. Fail-open to an empty list on any error so a
/// brief Ollama blip just shows "idle" rather than breaking the status call.
pub async fn resident_models() -> Vec<serde_json::Value> {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/ps"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    json["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| {
                    serde_json::json!({
                        "name": m.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "vram_mb": m.get("size_vram").and_then(|v| v.as_u64()).unwrap_or(0)
                            / (1024 * 1024),
                        "expires_at": m.get("expires_at").and_then(|v| v.as_str()).unwrap_or(""),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Running models that spilled to system RAM even though the card could hold
/// them right now: `size_vram` is less than half the model's footprint while
/// `free VRAM + what it already holds` covers the full footprint. These are
/// eviction candidates — Ollama pins a model to whatever placement it chose at
/// LOAD time for as long as keep_alive lasts, so a model scheduled onto the
/// CPU during a moment of VRAM pressure (e.g. whisper transcribing) stays on
/// the CPU long after the pressure clears. Unloading it lets the scheduler
/// re-place the next request on the GPU. Fail-open to an empty list.
pub async fn cpu_spilled_models(free_vram_mb: u64) -> Vec<String> {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/ps"))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
    else {
        return Vec::new();
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return Vec::new();
    };
    json["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let name = m.get("name")?.as_str()?;
                    let size_mb = m.get("size")?.as_u64()? / (1024 * 1024);
                    let vram_mb = m
                        .get("size_vram")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0)
                        / (1024 * 1024);
                    if size_mb == 0 {
                        return None;
                    }
                    let mostly_cpu = vram_mb * 2 < size_mb;
                    // Evicting frees its current VRAM share too, so the
                    // budget for a full-GPU reload is free + already-held.
                    let would_fit = free_vram_mb + vram_mb >= size_mb;
                    if mostly_cpu && would_fit {
                        Some(name.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Unload ONE resident model immediately (POST /api/generate keep_alive=0).
/// The next request for it reloads from scratch — with placement re-decided
/// against CURRENT free VRAM. Returns whether Ollama accepted the request.
pub async fn unload_model(name: &str) -> bool {
    reqwest::Client::new()
        .post(format!("http://{OLLAMA_LOCAL}/api/generate"))
        .json(&serde_json::json!({ "model": name, "keep_alive": 0 }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Unload every resident model (POST /api/generate keep_alive=0 per model) to
/// free VRAM immediately — for the "Free GPU memory" button and idle auto-free.
/// Returns (models_unloaded, names). Works whether Ollama is managed or external.
pub async fn unload_all() -> (usize, Vec<String>) {
    let client = reqwest::Client::new();
    let Ok(resp) = client
        .get(format!("http://{OLLAMA_LOCAL}/api/ps"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    else {
        return (0, vec![]);
    };
    let Ok(json) = resp.json::<serde_json::Value>().await else {
        return (0, vec![]);
    };
    let names: Vec<String> = json["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    for name in &names {
        let _ = client
            .post(format!("http://{OLLAMA_LOCAL}/api/generate"))
            .json(&serde_json::json!({ "model": name, "keep_alive": 0 }))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;
    }
    (names.len(), names)
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

/// Installed models with size + family, for the dashboard's model manager.
pub async fn list_models_detailed() -> Vec<serde_json::Value> {
    let Ok(resp) = reqwest::Client::new()
        .get(format!("http://{OLLAMA_LOCAL}/api/tags"))
        .timeout(std::time::Duration::from_secs(5))
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
                .map(|m| {
                    serde_json::json!({
                        "name": m["name"].as_str().unwrap_or(""),
                        "size": m["size"].as_u64().unwrap_or(0),
                        "family": m["details"]["family"].as_str().unwrap_or(""),
                        "parameter_size": m["details"]["parameter_size"].as_str().unwrap_or(""),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Delete an installed model from the local Ollama (frees disk).
pub async fn delete_model(model: &str) -> Result<(), String> {
    let resp = reqwest::Client::new()
        .request(
            reqwest::Method::DELETE,
            format!("http://{OLLAMA_LOCAL}/api/delete"),
        )
        .json(&serde_json::json!({ "name": model }))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("delete request failed: {e}"))?;
    if resp.status().is_success() {
        Ok(())
    } else {
        Err(format!("Ollama returned HTTP {} deleting {model}", resp.status()))
    }
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

/// One-click install with live progress. Windows: winget (official Ollama
/// package). macOS: Homebrew when present. `on_line` is called with each
/// output line so the GUI can show a live status/progress bar. Returns the
/// transcript; the GUI falls back to an "open ollama.com/download" button
/// when this reports failure.
#[allow(unused_variables, unused_mut)]
pub async fn install_streaming<F: FnMut(&str)>(mut on_line: F) -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        on_line("Locating Ollama package (winget)…");
        let mut child = quiet_command("winget")
            .args([
                "install",
                "--id",
                "Ollama.Ollama",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--silent",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("winget not available: {e}"))?;
        let transcript = stream_lines(&mut child, &mut on_line).await;
        let status = child.wait().await.map_err(|e| format!("winget wait failed: {e}"))?;
        if status.success() {
            return Ok(transcript);
        }
        return Err(format!("winget install failed:\n{transcript}"));
    }
    #[cfg(target_os = "macos")]
    {
        let brew = quiet_command("brew").arg("--version").output().await;
        if brew.map(|o| o.status.success()).unwrap_or(false) {
            on_line("Installing Ollama via Homebrew…");
            let mut child = quiet_command("brew")
                .args(["install", "ollama"])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("brew failed to start: {e}"))?;
            let transcript = stream_lines(&mut child, &mut on_line).await;
            let status = child.wait().await.map_err(|e| format!("brew wait failed: {e}"))?;
            if status.success() {
                return Ok(transcript);
            }
            return Err(format!("brew install failed:\n{transcript}"));
        }
        return Err("Homebrew not found — install from https://ollama.com/download".into());
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let _ = on_line;
        Err("Automated install is only wired for Windows/macOS — see https://ollama.com/download".into())
    }
}

/// Drain a child's stdout line-by-line, forwarding each to `on_line` and
/// accumulating a transcript.
#[cfg(any(target_os = "windows", target_os = "macos"))]
async fn stream_lines<F: FnMut(&str)>(
    child: &mut tokio::process::Child,
    on_line: &mut F,
) -> String {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let mut transcript = String::new();
    if let Some(stdout) = child.stdout.take() {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let t = line.trim();
            if !t.is_empty() {
                on_line(t);
                transcript.push_str(t);
                transcript.push('\n');
            }
        }
    }
    transcript
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
    // Only one start attempt at a time — the supervisor loop and the UI's
    // start button both call this and would otherwise spawn duplicate daemons.
    if state
        .ollama_starting
        .swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        for _ in 0..40 {
            if daemon_running().await {
                return Ok(true);
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
        return Ok(false);
    }
    let result = ensure_running_inner(state).await;
    state
        .ollama_starting
        .store(false, std::sync::atomic::Ordering::SeqCst);
    result
}

async fn ensure_running_inner(state: &Arc<AppState>) -> Result<bool, String> {
    if daemon_running().await {
        return Ok(false);
    }
    let keep_alive = state.config.lock().unwrap().ollama_keep_alive.clone();
    let overhead = gpu_overhead_bytes(state);
    // Concurrency + resident models come from the user's Speed profile (auto by
    // default, sized to the VRAM budget). Parallel slots let ClipAI's per-frame
    // vision requests run concurrently; keeping >1 model resident avoids a
    // llava↔qwen reload every stage switch. Quality is unchanged — same models,
    // same outputs, just no artificial serialization.
    let budget_gb = state.effective_budget_gb();
    let (num_parallel, max_loaded) = state.resolve_speed_settings();
    let profile = state.config.lock().unwrap().speed_profile.clone();
    log::info!(
        "starting managed ollama (keep_alive={keep_alive}, gpu_overhead={} MB, \
         speed={profile}, num_parallel={num_parallel}, max_loaded={max_loaded}, budget={budget_gb:.1} GB)",
        overhead / 1024 / 1024
    );
    let mut cmd = quiet_command(ollama_binary());
    cmd.arg("serve")
        .env("OLLAMA_HOST", OLLAMA_LOCAL) // localhost ONLY — proxy is the LAN surface
        .env("OLLAMA_MAX_LOADED_MODELS", max_loaded.to_string())
        .env("OLLAMA_NUM_PARALLEL", num_parallel.to_string())
        .env("OLLAMA_KEEP_ALIVE", keep_alive)
        // Parity with the ClipAI container's Ollama service: flash attention
        // is numerically exact and speeds up prefill — the polish/MTPE
        // prompts are long, so prefill is most of each request — and q8_0 KV
        // halves the cache, which is what lets num_parallel slots fit. The
        // container has shipped both for weeks; the managed daemon simply
        // never set them.
        .env("OLLAMA_FLASH_ATTENTION", "1")
        .env("OLLAMA_KV_CACHE_TYPE", "q8_0")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if overhead > 0 {
        cmd.env("OLLAMA_GPU_OVERHEAD", overhead.to_string());
    }
    let child = cmd
        .spawn()
        .map_err(|e| format!("could not start ollama serve: {e}"))?;
    // Die with the Companion — no orphaned daemon after quit/kill.
    crate::state::bind_child_to_lifetime(&child);
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
/// `qwen2.5vl` is the Ollama-native image/video-frame understanding model
/// (VideoLLaMA is not an Ollama/GGUF model and can't run here).
pub fn recommended_models(budget_gb: f32) -> Vec<(&'static str, &'static str)> {
    if budget_gb >= 10.0 {
        vec![
            ("llava:13b", "Vision (Primary AI)"),
            ("qwen2.5vl:7b", "Video/vision understanding"),
            ("qwen2.5:14b", "Editorial (SEO / summaries)"),
            ("qwen3:4b-instruct-2507-q4_K_M", "Subtitle translation"),
        ]
    } else if budget_gb >= 6.0 {
        vec![
            ("llava:7b", "Vision (Primary AI)"),
            ("qwen2.5vl:7b", "Video/vision understanding"),
            ("qwen2.5:7b-instruct", "Editorial (SEO / summaries)"),
            ("qwen3:4b-instruct-2507-q4_K_M", "Subtitle translation"),
        ]
    } else {
        vec![
            ("moondream:1.8b", "Vision (Primary AI)"),
            ("qwen2.5vl:3b", "Video/vision understanding"),
            ("qwen2.5:3b-instruct", "Editorial (SEO / summaries)"),
            ("qwen3:4b-instruct-2507-q4_K_M", "Subtitle translation"),
        ]
    }
}
