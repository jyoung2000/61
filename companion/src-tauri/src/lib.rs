//! ClipAI GPU Companion — Tauri shell.
//!
//! Responsibilities: run the authenticated LAN proxy, manage Ollama and
//! the whisper sidecar, poll GPU telemetry, expose commands to the
//! React dashboard, and live in the tray (closing the window hides it).

// Public so the tests/proxy_e2e.rs integration test can exercise the
// proxy + state contract without the Tauri shell.
pub mod gpu;
pub mod ollama;
pub mod pairing;
pub mod proxy;
pub mod sidecar;
pub mod state;

use serde::Deserialize;
use state::AppState;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tauri::menu::{CheckMenuItem, Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{Emitter, Manager, WindowEvent};

type SharedState = Arc<AppState>;

// ── Commands (invoked from the React UI) ────────────────────────────

#[tauri::command]
async fn get_status(state: tauri::State<'_, SharedState>) -> Result<serde_json::Value, String> {
    let config = state.config_snapshot();
    let gpu = state.gpu.lock().unwrap().clone();
    let ollama_status = ollama::status(&state).await;
    let budget = state.effective_budget_gb();
    let (whisper_model, whisper_compute) = state::whisper_tier_for_budget(budget);
    let sidecar_running = state.sidecar.lock().await.is_some();
    let activity: Vec<state::ActivityEntry> =
        state.activity.lock().unwrap().iter().cloned().collect();
    Ok(serde_json::json!({
        "config": {
            "token": config.token,
            "port": config.port,
            "vram_budget_gb": config.vram_budget_gb,
            "ollama_keep_alive": config.ollama_keep_alive,
            "sidecar_idle_min": config.sidecar_idle_min,
            "paused": config.paused,
            "paired_clipai_url": config.paired_clipai_url,
            "name": config.name,
            "setup_complete": config.setup_complete,
        },
        "gpu": gpu,
        "effective_budget_gb": budget,
        "whisper_tier": { "model": whisper_model, "compute": whisper_compute },
        "ollama": ollama_status,
        "sidecar_running": sidecar_running,
        "busy": state.whisper_busy.load(Ordering::Relaxed),
        "current_job": state.current_job(),
        "activity": activity,
        "lan_ip": pairing::detect_lan_ip(),
        "recommended_models": ollama::recommended_models(budget)
            .into_iter()
            .map(|(m, why)| serde_json::json!({"model": m, "why": why}))
            .collect::<Vec<_>>(),
    }))
}

#[derive(Deserialize)]
struct ConfigPatch {
    vram_budget_gb: Option<f32>,
    ollama_keep_alive: Option<String>,
    sidecar_idle_min: Option<u32>,
    paused: Option<bool>,
    name: Option<String>,
    setup_complete: Option<bool>,
}

#[tauri::command]
async fn set_config(
    state: tauri::State<'_, SharedState>,
    patch: ConfigPatch,
) -> Result<serde_json::Value, String> {
    let mut ollama_restart_needed = false;
    let mut sidecar_restart_needed = false;
    {
        let mut cfg = state.config.lock().unwrap();
        if let Some(v) = patch.vram_budget_gb {
            if (v - cfg.vram_budget_gb).abs() > 0.01 {
                cfg.vram_budget_gb = v.max(0.0);
                ollama_restart_needed = true;
                sidecar_restart_needed = true;
            }
        }
        if let Some(v) = patch.ollama_keep_alive {
            if v != cfg.ollama_keep_alive {
                cfg.ollama_keep_alive = v;
                ollama_restart_needed = true;
            }
        }
        if let Some(v) = patch.sidecar_idle_min {
            cfg.sidecar_idle_min = v.clamp(1, 24 * 60);
        }
        if let Some(v) = patch.paused {
            cfg.paused = v;
        }
        if let Some(v) = patch.name {
            if !v.trim().is_empty() {
                cfg.name = v.trim().to_string();
            }
        }
        if let Some(v) = patch.setup_complete {
            cfg.setup_complete = v;
        }
    }
    state.save();
    if ollama_restart_needed {
        // Apply the new VRAM reservation: the managed daemon restarts with
        // OLLAMA_GPU_OVERHEAD = (total − budget).
        let _ = ollama::restart(&state).await;
    }
    if sidecar_restart_needed {
        // The whisper tier may have changed — drop the sidecar; the next
        // transcription lazily starts the right one.
        sidecar::shutdown(&state).await;
    }
    Ok(serde_json::json!({"ok": true, "ollama_restarted": ollama_restart_needed}))
}

#[tauri::command]
fn regenerate_token(state: tauri::State<'_, SharedState>) -> String {
    let token = state::generate_token();
    state.config.lock().unwrap().token = token.clone();
    state.save();
    token
}

#[tauri::command]
async fn install_ollama() -> Result<String, String> {
    ollama::install().await
}

#[tauri::command]
async fn start_ollama(state: tauri::State<'_, SharedState>) -> Result<bool, String> {
    ollama::ensure_running(&state).await
}

#[tauri::command]
async fn pull_model(model: String) -> Result<(), String> {
    // Stream the pull so the request survives multi-GB downloads.
    let resp = reqwest::Client::new()
        .post(format!("http://{}/api/pull", state::OLLAMA_LOCAL))
        .json(&serde_json::json!({"name": model}))
        .timeout(std::time::Duration::from_secs(3600))
        .send()
        .await
        .map_err(|e| format!("pull failed to start: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("pull failed: HTTP {}", resp.status()));
    }
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        chunk.map_err(|e| format!("pull interrupted: {e}"))?;
    }
    Ok(())
}

#[tauri::command]
async fn pair_clipai(
    state: tauri::State<'_, SharedState>,
    clipai_url: String,
    api_key: String,
) -> Result<serde_json::Value, String> {
    pairing::pair(&state, &clipai_url, &api_key).await
}

// ── Tray ────────────────────────────────────────────────────────────

fn build_tray(app: &tauri::App, state: SharedState) -> tauri::Result<()> {
    let open_item = MenuItem::with_id(app, "open", "Open Dashboard", true, None::<&str>)?;
    let pause_item = CheckMenuItem::with_id(
        app,
        "pause",
        "Pause sharing",
        true,
        state.config.lock().unwrap().paused,
        None::<&str>,
    )?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open_item, &pause_item, &quit_item])?;

    let pause_for_menu = pause_item.clone();
    let state_for_menu = state.clone();
    TrayIconBuilder::with_id("main-tray")
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("ClipAI GPU Companion — idle")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "open" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "pause" => {
                let paused = {
                    let mut cfg = state_for_menu.config.lock().unwrap();
                    cfg.paused = !cfg.paused;
                    cfg.paused
                };
                state_for_menu.save();
                let _ = pause_for_menu.set_checked(paused);
                let _ = app.emit("config-changed", paused);
            }
            "quit" => {
                let state = state_for_menu.clone();
                let app = app.clone();
                tauri::async_runtime::spawn(async move {
                    sidecar::shutdown(&state).await;
                    ollama::shutdown(&state).await;
                    app.exit(0);
                });
            }
            _ => {}
        })
        .build(app)?;
    Ok(())
}

// ── App entry ───────────────────────────────────────────────────────

pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            let config_dir = app
                .path()
                .app_config_dir()
                .expect("no app config dir available");
            let data_dir = app.path().app_data_dir().unwrap_or_else(|_| config_dir.clone());
            let resource_dir = app
                .path()
                .resource_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));

            let state: SharedState = Arc::new(AppState::load(config_dir));
            app.manage(state.clone());

            // GPU telemetry poll (5 s) + tray tooltip/busy indicator.
            {
                let state = state.clone();
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        let snap = tokio::task::spawn_blocking(gpu::snapshot)
                            .await
                            .unwrap_or_default();
                        *state.gpu.lock().unwrap() = snap.clone();
                        if let Some(tray) = app_handle.tray_by_id("main-tray") {
                            let busy = state.whisper_busy.load(Ordering::Relaxed)
                                || state.current_job().is_some();
                            let paused = state.config.lock().unwrap().paused;
                            let tip = if paused {
                                "ClipAI GPU Companion — sharing paused".to_string()
                            } else if busy {
                                let job = state
                                    .current_job()
                                    .map(|j| j.job_title)
                                    .filter(|t| !t.is_empty())
                                    .unwrap_or_else(|| "working".into());
                                format!("ClipAI GPU Companion — active: {job}")
                            } else {
                                format!(
                                    "ClipAI GPU Companion — idle ({} MB VRAM free)",
                                    snap.vram_free_mb
                                )
                            };
                            let _ = tray.set_tooltip(Some(tip));
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                });
            }

            // Start the managed Ollama (no-op if one is already running)
            // and the LAN proxy.
            {
                let state = state.clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = ollama::ensure_running(&state).await {
                        log::warn!("ollama not started yet: {e}");
                    }
                });
            }
            {
                let ctx = proxy::ProxyCtx {
                    state: state.clone(),
                    client: reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(3600))
                        .connect_timeout(std::time::Duration::from_secs(10))
                        .build()
                        .expect("reqwest client"),
                    resource_dir,
                    data_dir,
                };
                tauri::async_runtime::spawn(proxy::serve(ctx));
            }
            sidecar::spawn_idle_reaper(state.clone());
            build_tray(app, state)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window minimizes to the tray; Quit lives in the
            // tray menu.
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            set_config,
            regenerate_token,
            install_ollama,
            start_ollama,
            pull_model,
            pair_clipai,
        ])
        .run(tauri::generate_context!())
        .expect("error while running the ClipAI GPU Companion");
}
