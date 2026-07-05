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

// ── Crash diagnostics ───────────────────────────────────────────────
// The GUI has no console (windows_subsystem = "windows"), so logs and
// panics would otherwise vanish. Everything is mirrored to a file the
// user can paste back when the app misbehaves.

/// `%LOCALAPPDATA%\app.clipai.companion\companion.log` on Windows,
/// `~/Library/Application Support/…` on macOS, `~/.local/share/…` on Linux.
pub fn log_file_path() -> std::path::PathBuf {
    dirs::data_local_dir()
        .or_else(dirs::config_dir)
        .unwrap_or_else(std::env::temp_dir)
        .join("app.clipai.companion")
        .join("companion.log")
}

/// Append a clearly-delimited crash record straight to the log file — used
/// by the panic hook and the top-level error handler so it survives even if
/// the normal logger never initialised.
fn append_crash(msg: &str) {
    use std::io::Write;
    let path = log_file_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(f, "\n===== CRASH {stamp} =====\n{msg}\n========================");
    }
}

/// Route `log::*` to the log file (fresh per launch) and install a panic
/// hook that records any thread's panic before it unwinds. Returns the log
/// path so it can be surfaced to the user.
fn init_diagnostics() -> std::path::PathBuf {
    let path = log_file_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // Append (don't truncate): a blocked second instance must not wipe the
    // running instance's log. Each launch writes a "starting" separator.
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok();
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));
    builder.format_timestamp_secs();
    if let Some(file) = file {
        builder.target(env_logger::Target::Pipe(Box::new(file)));
    }
    let _ = builder.try_init();

    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "<unknown location>".into());
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "<non-string panic payload>".into());
        let bt = std::backtrace::Backtrace::force_capture();
        let msg = format!("panic at {loc}: {payload}\nbacktrace:\n{bt}");
        log::error!("{msg}");
        append_crash(&msg);
        default(info);
    }));
    path
}

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
        "sidecar_available": state.sidecar_available.load(Ordering::Relaxed),
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
async fn install_ollama(app: tauri::AppHandle) -> Result<String, String> {
    let progress = app.clone();
    let result = ollama::install_streaming(move |line| {
        let _ = progress.emit(
            "install-progress",
            serde_json::json!({"stage": "installing", "message": line}),
        );
    })
    .await;
    match &result {
        Ok(_) => {
            let _ = app.emit(
                "install-progress",
                serde_json::json!({"stage": "done", "message": "Ollama installed"}),
            );
        }
        Err(e) => {
            let _ = app.emit(
                "install-progress",
                serde_json::json!({"stage": "error", "message": e}),
            );
        }
    }
    result
}

#[tauri::command]
async fn start_ollama(state: tauri::State<'_, SharedState>) -> Result<bool, String> {
    ollama::ensure_running(&state).await
}

#[tauri::command]
async fn list_models() -> Vec<serde_json::Value> {
    ollama::list_models_detailed().await
}

#[tauri::command]
async fn delete_model(model: String) -> Result<(), String> {
    ollama::delete_model(&model).await
}

#[tauri::command]
async fn pull_model(app: tauri::AppHandle, model: String) -> Result<(), String> {
    // Stream the pull so the request survives multi-GB downloads AND so we can
    // relay real byte-level progress to the UI's progress bar. Ollama's
    // /api/pull emits NDJSON lines: {status, digest?, total?, completed?}.
    let resp = reqwest::Client::new()
        .post(format!("http://{}/api/pull", state::OLLAMA_LOCAL))
        .json(&serde_json::json!({"name": model, "stream": true}))
        .timeout(std::time::Duration::from_secs(3600))
        .send()
        .await
        .map_err(|e| format!("pull failed to start: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("pull failed: HTTP {}", resp.status()));
    }
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("pull interrupted: {e}"))?;
        buf.extend_from_slice(&chunk);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = &line[..line.len().saturating_sub(1)];
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(format!("pull error: {err}"));
                }
                let status = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
                let completed = v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
                // -1 = indeterminate (manifest/verify phases have no byte total).
                let percent = if total > 0 {
                    completed as f64 / total as f64 * 100.0
                } else {
                    -1.0
                };
                let _ = app.emit(
                    "pull-progress",
                    serde_json::json!({
                        "model": &model, "status": status,
                        "percent": percent, "completed": completed, "total": total,
                    }),
                );
            }
        }
    }
    let _ = app.emit(
        "pull-progress",
        serde_json::json!({"model": &model, "status": "success", "percent": 100.0}),
    );
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
    let mut tray = TrayIconBuilder::with_id("main-tray")
        .tooltip("ClipAI GPU Companion — idle")
        .menu(&menu)
        .show_menu_on_left_click(true);
    // Don't unwrap the icon — a missing default icon must not abort startup.
    match app.default_window_icon() {
        Some(icon) => tray = tray.icon(icon.clone()),
        None => log::warn!("tray: no default window icon available"),
    }
    tray
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
    let log_path = init_diagnostics();
    log::info!(
        "=== ClipAI GPU Companion v{} starting ===",
        env!("CARGO_PKG_VERSION")
    );
    log::info!("log file: {}", log_path.display());

    let builder = tauri::Builder::default()
        // Single instance MUST be the first plugin: a second launch focuses
        // the running window and exits instead of starting a duplicate.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            log::info!("second instance launched — focusing the existing window");
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.show();
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            log::info!("setup: begin");
            let config_dir = app
                .path()
                .app_config_dir()
                .unwrap_or_else(|e| {
                    log::warn!("no app config dir ({e}); falling back to temp dir");
                    std::env::temp_dir().join("app.clipai.companion")
                });
            let data_dir = app.path().app_data_dir().unwrap_or_else(|_| config_dir.clone());
            let resource_dir = app
                .path()
                .resource_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            log::info!(
                "setup: config_dir={} resource_dir={}",
                config_dir.display(),
                resource_dir.display()
            );

            let state: SharedState = Arc::new(AppState::load(config_dir));
            state.sidecar_available.store(
                sidecar::available(&resource_dir),
                Ordering::Relaxed,
            );
            app.manage(state.clone());
            log::info!(
                "setup: state loaded (sidecar_available={})",
                state.sidecar_available.load(Ordering::Relaxed)
            );

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

            // Self-healing Ollama supervisor: keep probing/starting so Ollama
            // is detected automatically once it's installed or started — e.g.
            // right after the in-app install, WITHOUT needing an app restart
            // (the freshly-installed binary is found by known-path resolution
            // even though this process's PATH is stale).
            {
                let state = state.clone();
                tauri::async_runtime::spawn(async move {
                    loop {
                        if !ollama::daemon_running().await {
                            if let Err(e) = ollama::ensure_running(&state).await {
                                log::warn!("ollama not up yet: {e}");
                            }
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(8)).await;
                    }
                });
            }
            {
                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(3600))
                    .connect_timeout(std::time::Duration::from_secs(10))
                    .build()
                {
                    Ok(c) => Some(c),
                    Err(e) => {
                        log::error!("failed to build HTTP client, proxy disabled: {e}");
                        None
                    }
                };
                if let Some(client) = client {
                    let ctx = proxy::ProxyCtx {
                        state: state.clone(),
                        client,
                        resource_dir,
                        data_dir,
                    };
                    tauri::async_runtime::spawn(proxy::serve(ctx));
                }
            }
            sidecar::spawn_idle_reaper(state.clone());
            // Tray failure must not abort startup — the dashboard window is
            // the primary surface; log and continue without the tray.
            if let Err(e) = build_tray(app, state) {
                log::error!("tray build failed (continuing without tray): {e}");
            }
            log::info!("setup: complete");
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
            list_models,
            delete_model,
            pair_clipai,
        ]);

    // Build first (so a build failure — e.g. missing WebView2 — is logged, not
    // a silent .expect()), then run with an exit handler that stops the managed
    // Ollama + whisper children on the way out. On Windows the kill-on-close job
    // object also guarantees this even on a hard kill; this covers clean quits
    // and other platforms.
    match builder.build(tauri::generate_context!()) {
        Ok(app) => app.run(|handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = handle.try_state::<SharedState>() {
                    let state = state.inner().clone();
                    tauri::async_runtime::block_on(async move {
                        sidecar::shutdown(&state).await;
                        ollama::shutdown(&state).await;
                    });
                    log::info!("shutdown: stopped managed ollama + whisper sidecar");
                }
            }
        }),
        Err(e) => {
            let msg = format!("tauri runtime failed to build/start: {e:?}");
            log::error!("{msg}");
            append_crash(&msg);
            std::process::exit(1);
        }
    }
    log::info!("app exited");
}
