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

/// Bring the main window to the foreground — recreating it if it was somehow
/// destroyed (a WebView crash, or a close path that bypassed prevent_close),
/// so the GUI can never end up permanently invisible with the process alive.
fn show_or_create_main(app: &tauri::AppHandle) {
    use tauri::Manager;
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
        return;
    }
    match tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::default())
        .title("ClipAI GPU Companion")
        .inner_size(920.0, 720.0)
        .build()
    {
        Ok(w) => {
            let _ = w.show();
            let _ = w.set_focus();
            log::info!("recreated missing main window");
        }
        Err(e) => log::error!("could not recreate main window: {e}"),
    }
}

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
async fn get_status(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
) -> Result<serde_json::Value, String> {
    let config = state.config_snapshot();
    let gpu = state.gpu.lock().unwrap().clone();
    let ollama_status = ollama::status(&state).await;
    // Models actively resident in VRAM right now (skip the probe when Ollama
    // isn't even running so we don't wait on a doomed request).
    let resident_models = if ollama_status.running {
        ollama::resident_models().await
    } else {
        Vec::new()
    };
    let budget = state.effective_budget_gb();
    let (whisper_model, whisper_compute) = state::whisper_tier_for_budget(budget);
    // Effective transcription quality for the current profile + VRAM.
    let (whisper_beam, whisper_full) =
        state::whisper_quality_params(&config.whisper_quality, budget);
    let whisper_model_eff = if whisper_full && whisper_model == "large-v3-turbo" {
        "large-v3"
    } else {
        whisper_model
    };
    let sidecar_running = state.sidecar.lock().await.is_some();
    // VRAM ClipAI is actively holding: resident Ollama models (/api/ps) plus a
    // rough whisper-model footprint while transcribing (whisper.cpp VRAM isn't
    // in /api/ps). Lets the GUI color ClipAI's use vs unrelated apps (games).
    let clipai_vram_mb: u64 = {
        let ollama_bytes = ollama::loaded_vram_bytes().await;
        let whisper_mb: u64 = if state.whisper_busy.load(Ordering::Relaxed) {
            // Tier footprint: turbo/large ~1.6 GB, medium ~0.9 GB, else ~0.4 GB.
            match whisper_model {
                "large-v3-turbo" => 1600,
                "medium" => 900,
                _ => 400,
            }
        } else {
            0
        };
        ollama_bytes / (1024 * 1024) + whisper_mb
    };
    // Which whisper build is installed (gpu/cpu/bundled/none) so the GUI can warn
    // when transcription would fall back to the CPU and offer the GPU build.
    let rd = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let dd = app.path().app_data_dir().unwrap_or_else(|_| rd.clone());
    let whisper_build = sidecar::build_kind(&rd, &dd);
    let (speed_parallel, speed_loaded) = state.resolve_speed_settings();
    let activity: Vec<state::ActivityEntry> =
        state.activity.lock().unwrap().iter().cloned().collect();
    // Prefer the progress ClipAI EXPLICITLY reported (POST /v1/progress) — it
    // covers local-only pipeline stages where no AI request reaches us, so the
    // bar tracks the container instead of freezing on the last in-flight call.
    let reported = state.reported_job_fresh();
    let in_flight = state.current_job();
    // Headline: a fresh heartbeat wins; else the in-flight request; else null.
    let current_job: serde_json::Value = if let Some(r) = &reported {
        serde_json::json!({
            "job_id": r.job_id,
            "job_title": r.job_title,
            "stage": r.stage,
            "kind": in_flight.as_ref().map(|e| e.kind.clone()).unwrap_or_default(),
            // The job's real start (first heartbeat) — never 0, so "elapsed"
            // isn't measured from the Unix epoch during a local-only stage.
            "started_at_ms": r.started_ms,
        })
    } else if let Some(e) = &in_flight {
        serde_json::to_value(e).unwrap_or(serde_json::Value::Null)
    } else {
        serde_json::Value::Null
    };
    // Live job progress (0-100) — from the heartbeat, else an in-flight request.
    let job_progress: serde_json::Value = if let Some(r) = &reported {
        serde_json::json!(r.progress)
    } else {
        let p = state.job_progress.load(Ordering::Relaxed);
        if p != u64::MAX && in_flight.is_some() {
            serde_json::json!(p)
        } else {
            serde_json::Value::Null
        }
    };
    Ok(serde_json::json!({
        "app_version": env!("CARGO_PKG_VERSION"),
        "app_build": env!("CLIPAI_BUILD_ID"),
        "config": {
            "token": config.token,
            "port": config.port,
            "vram_budget_gb": config.vram_budget_gb,
            "vram_auto": config.vram_auto,
            "vram_buffer_gb": config.vram_buffer_gb,
            "ollama_keep_alive": config.ollama_keep_alive,
            "sidecar_idle_min": config.sidecar_idle_min,
            "gpu_idle_free_min": config.gpu_idle_free_min,
            "gpu_idle_free_sec": config.gpu_idle_free_sec,
            "paused": config.paused,
            "paired_clipai_url": config.paired_clipai_url,
            "name": config.name,
            "setup_complete": config.setup_complete,
            "speed_profile": config.speed_profile,
            "whisper_quality": config.whisper_quality,
            "shared_paths": config.shared_paths,
            "share_all": config.share_all,
        },
        "speed": {
            "profile": config.speed_profile,
            "num_parallel": speed_parallel,
            "max_loaded_models": speed_loaded,
        },
        "gpu": gpu,
        "clipai_vram_mb": clipai_vram_mb,
        "effective_budget_gb": budget,
        "whisper_tier": { "model": whisper_model, "compute": whisper_compute },
        "whisper_quality_effective": {
            "profile": config.whisper_quality,
            "model": whisper_model_eff,
            "beam_size": whisper_beam,
            "beam_search": whisper_beam > 1,
        },
        "ollama": ollama_status,
        "resident_models": resident_models,
        "sidecar_available": state.sidecar_available.load(Ordering::Relaxed),
        "sidecar_running": sidecar_running,
        "whisper_build": whisper_build,
        "busy": state.whisper_busy.load(Ordering::Relaxed),
        "current_job": current_job,
        "job_progress": job_progress,
        "proxy_bound": state.proxy_bound.load(Ordering::Relaxed),
        "proxy_last_error": state.proxy_last_error.lock().unwrap().clone(),
        "clipai_connected": state.clipai_connected(),
        "clipai_serving": state.serving_jobs(),
        "clipai_last_contact_ms": state.last_clipai_contact_ms(),
        "activity": activity,
        "job_logs": state.job_logs(),
        "incoming_pulls": state.incoming_pulls_snapshot().into_iter()
            .map(|(m, p)| serde_json::json!({"model": m, "percent": p}))
            .collect::<Vec<_>>(),
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
    gpu_idle_free_min: Option<u32>,
    gpu_idle_free_sec: Option<u32>,
    paused: Option<bool>,
    name: Option<String>,
    setup_complete: Option<bool>,
    vram_auto: Option<bool>,
    vram_buffer_gb: Option<f32>,
    speed_profile: Option<String>,
    whisper_quality: Option<String>,
    shared_paths: Option<Vec<String>>,
    share_all: Option<bool>,
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
        if let Some(v) = patch.gpu_idle_free_min {
            // 0 = auto-free off (whisper-only backstop still applies).
            cfg.gpu_idle_free_min = v.clamp(0, 24 * 60);
        }
        if let Some(v) = patch.gpu_idle_free_sec {
            // Fast idle-free window in seconds (primary knob). 0 = fall back to
            // the minutes knob; capped at 1 h.
            cfg.gpu_idle_free_sec = v.clamp(0, 3600);
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
        if let Some(v) = patch.vram_auto {
            if v != cfg.vram_auto {
                cfg.vram_auto = v;
                ollama_restart_needed = true;
                sidecar_restart_needed = true;
            }
        }
        if let Some(v) = patch.vram_buffer_gb {
            let v = v.clamp(0.0, 64.0);
            if (v - cfg.vram_buffer_gb).abs() > 0.01 {
                cfg.vram_buffer_gb = v;
                if cfg.vram_auto {
                    ollama_restart_needed = true;
                    sidecar_restart_needed = true;
                }
            }
        }
        if let Some(v) = patch.speed_profile {
            let v = v.trim().to_lowercase();
            if matches!(v.as_str(), "auto" | "eco" | "balanced" | "turbo") && v != cfg.speed_profile {
                cfg.speed_profile = v;
                // Restart Ollama so the new NUM_PARALLEL / MAX_LOADED take effect.
                ollama_restart_needed = true;
            }
        }
        if let Some(v) = patch.whisper_quality {
            let v = v.trim().to_lowercase();
            if matches!(v.as_str(), "auto" | "fast" | "balanced" | "max") && v != cfg.whisper_quality {
                cfg.whisper_quality = v;
                // Drop the whisper sidecar so the next request restarts it with
                // the new beam-search / model settings.
                sidecar_restart_needed = true;
            }
        }
        if let Some(v) = patch.shared_paths {
            // Normalize: trim, drop empties, de-dupe (order preserved). These are
            // the folders ClipAI may browse + pull files from; no restart needed.
            let mut seen = std::collections::HashSet::new();
            let cleaned: Vec<String> = v
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && seen.insert(s.clone()))
                .collect();
            cfg.shared_paths = cleaned;
        }
        if let Some(v) = patch.share_all {
            cfg.share_all = v;
        }
    }
    // Force the auto-VRAM loop to re-apply immediately after a settings change.
    state.last_auto_baseline_mb.store(u64::MAX, Ordering::Relaxed);
    state.last_auto_apply_ms.store(0, Ordering::Relaxed);
    state.save();
    if ollama_restart_needed {
        // Apply the new VRAM reservation: the managed daemon restarts with
        // OLLAMA_GPU_OVERHEAD = (total − budget).
        let _ = ollama::restart(&state).await;
    }
    if sidecar_restart_needed {
        // The whisper tier may have changed — drop the sidecar; the next
        // transcription lazily starts the right one.
        sidecar::shutdown(&state, "whisper settings changed").await;
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

/// Re-probe whether a whisper sidecar is present (bundled or downloaded) and
/// update the cached flag — the "Refresh" button after a Download/install.
#[tauri::command]
fn refresh_sidecar(app: tauri::AppHandle, state: tauri::State<'_, SharedState>) -> bool {
    let rd = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let dd = app.path().app_data_dir().unwrap_or_else(|_| rd.clone());
    let ok = sidecar::available(&rd, &dd);
    state.sidecar_available.store(ok, Ordering::Relaxed);
    ok
}

/// Download the latest whisper.cpp server into app data (Windows), then
/// re-probe availability. Progress arrives via `whisper-progress` events.
#[tauri::command]
async fn download_whisper(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    let dd = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("no app data dir: {e}"))?;
    let rd = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    // Stop any running sidecar first: on Windows a live whisper-server.exe holds
    // a file lock, so re-downloading (e.g. swapping a CPU build for the GPU one)
    // would fail to overwrite it. It restarts lazily on the next request.
    sidecar::shutdown(&state, "whisper download requested").await;
    let msg = sidecar::download_whispercpp(&app, &dd).await?;
    state
        .sidecar_available
        .store(sidecar::available(&rd, &dd), Ordering::Relaxed);
    Ok(msg)
}

/// Format a ms-epoch as local time, or "-" for 0/unset.
fn fmt_ms(ms: u64) -> String {
    if ms == 0 {
        return "-".into();
    }
    use chrono::TimeZone;
    chrono::Local
        .timestamp_millis_opt(ms as i64)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| ms.to_string())
}

/// Human duration for a millisecond span.
fn fmt_dur_ms(ms: u64) -> String {
    let s = ms / 1000;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}h{m}m{sec}s")
    } else if m > 0 {
        format!("{m}m{sec}s")
    } else {
        format!("{sec}s")
    }
}

/// Compose a full human-readable diagnostics report: environment, config
/// (token redacted), connection/pairing, GPU, Ollama, Whisper, every job/
/// activity entry since launch, and the on-disk application log.
pub(crate) async fn build_diagnostics_report(state: &AppState, whisper_build: &str) -> String {
    use std::fmt::Write as _;
    let now = state::now_ms();
    let cfg = state.config_snapshot();
    let gpu = state.gpu.lock().unwrap().clone();
    let budget = state.effective_budget_gb();
    let ollama = ollama::status(state).await;
    let sidecar_running = state.sidecar.lock().await.is_some();
    let pulls = state.incoming_pulls_snapshot();
    let jobs = state.job_logs();
    let activity: Vec<state::ActivityEntry> =
        state.activity.lock().unwrap().iter().cloned().collect();

    let mut r = String::new();
    let _ = writeln!(r, "================ ClipAI GPU Companion — Diagnostics Report ================");
    let _ = writeln!(r, "Generated:     {}", fmt_ms(now));
    let _ = writeln!(r, "App version:   {} ({})", env!("CARGO_PKG_VERSION"), env!("CLIPAI_BUILD_ID"));
    let _ = writeln!(r, "Platform:      {} / {}", std::env::consts::OS, std::env::consts::ARCH);
    let _ = writeln!(r, "Log file:      {}", log_file_path().display());
    let _ = writeln!(r, "App started:   {}  (uptime {})",
        fmt_ms(state.app_started_ms), fmt_dur_ms(now.saturating_sub(state.app_started_ms)));

    let _ = writeln!(r, "\n---- Configuration ----");
    let _ = writeln!(r, "Companion name:       {}", cfg.name);
    let _ = writeln!(r, "Proxy port:           {}", cfg.port);
    let _ = writeln!(r, "Access token:         present ({} chars) [redacted]", cfg.token.len());
    let _ = writeln!(r, "Paused:               {}", cfg.paused);
    let _ = writeln!(r, "Setup complete:       {}", cfg.setup_complete);
    let _ = writeln!(r, "Auto-allocate VRAM:   {}", cfg.vram_auto);
    let _ = writeln!(r, "VRAM budget (manual): {} GB", cfg.vram_budget_gb);
    let _ = writeln!(r, "VRAM free buffer:     {} GB", cfg.vram_buffer_gb);
    let _ = writeln!(r, "Effective VRAM given: {:.1} GB", budget);
    let _ = writeln!(r, "Ollama keep-alive:    {}", cfg.ollama_keep_alive);
    let _ = writeln!(r, "Sidecar idle (min):   {}", cfg.sidecar_idle_min);
    let _ = writeln!(r, "GPU auto-free (min):  {}", cfg.gpu_idle_free_min);
    let _ = writeln!(r, "GPU auto-free (sec):  {}", cfg.gpu_idle_free_sec);
    let _ = writeln!(r, "Paired ClipAI URL:    {}",
        if cfg.paired_clipai_url.is_empty() { "(none — added manually in ClipAI, or not paired)".into() }
        else { cfg.paired_clipai_url.clone() });

    let _ = writeln!(r, "\n---- Connection ----");
    let _ = writeln!(r, "LAN IP:               {}", pairing::detect_lan_ip().unwrap_or_else(|| "(unknown)".into()));
    let last = state.last_clipai_contact_ms();
    let _ = writeln!(r, "ClipAI connected:     {} (last contact {}{})",
        state.clipai_connected(), fmt_ms(last),
        if last > 0 { format!(", {} ago", fmt_dur_ms(now.saturating_sub(last))) } else { String::new() });
    let _ = writeln!(r, "Serving real jobs:    {}", state.serving_jobs());

    let _ = writeln!(r, "\n---- GPU ----");
    let _ = writeln!(r, "Name:                 {}", if gpu.gpu_name.is_empty() { "(none)".into() } else { gpu.gpu_name.clone() });
    let _ = writeln!(r, "VRAM used/free/total: {} / {} / {} MB",
        gpu.vram_total_mb.saturating_sub(gpu.vram_free_mb), gpu.vram_free_mb, gpu.vram_total_mb);
    let _ = writeln!(r, "Unified memory:       {}", gpu.unified_memory);
    let _ = writeln!(r, "Available:            {}", gpu.available);

    let _ = writeln!(r, "\n---- Ollama ----");
    let _ = writeln!(r, "Installed: {}   Version: {}   Running: {}   Managed: {}",
        ollama.installed, if ollama.version.is_empty() { "-" } else { &ollama.version },
        ollama.running, ollama.managed);
    let _ = writeln!(r, "Models ({}):", ollama.models.len());
    for m in &ollama.models {
        let _ = writeln!(r, "  - {m}");
    }

    let _ = writeln!(r, "\n---- Whisper sidecar ----");
    let _ = writeln!(r, "Bundled/available:    {}", state.sidecar_available.load(Ordering::Relaxed));
    let _ = writeln!(r, "Running now:          {}", sidecar_running);
    let _ = writeln!(r, "Build:                {}  ({})", whisper_build, match whisper_build {
        "gpu" => "CUDA — runs on the GPU",
        "cpu" => "CPU only — SLOW; install the GPU build",
        "bundled" => "official release sidecar",
        _ => "not installed",
    });

    let _ = writeln!(r, "\n---- Incoming model pulls (from ClipAI) ----");
    if pulls.is_empty() {
        let _ = writeln!(r, "  (none)");
    } else {
        for (m, pct) in &pulls {
            let _ = writeln!(r, "  {m}: {pct:.0}%");
        }
    }

    let _ = writeln!(r, "\n---- Pipeline activity (grouped by ClipAI job) ----");
    if jobs.is_empty() {
        let _ = writeln!(r, "  (no jobs served this session)");
    }
    for j in &jobs {
        let _ = writeln!(r, "\n[JOB {}] {}  ({})  started {}  last {}",
            if j.job_id.is_empty() { "ad-hoc".into() } else { j.job_id.clone() },
            if j.job_title.is_empty() { "(untitled)".into() } else { j.job_title.clone() },
            if j.active { "ACTIVE" } else { "done" },
            fmt_ms(j.started_at_ms), fmt_ms(j.last_activity_ms));
        for e in &j.entries {
            let dur = match e.finished_at_ms {
                Some(f) => fmt_dur_ms(f.saturating_sub(e.started_at_ms)),
                None => format!("{}…", fmt_dur_ms(now.saturating_sub(e.started_at_ms))),
            };
            let _ = writeln!(r, "  {}  {:<8} {:<28} {}  {}{}",
                fmt_ms(e.started_at_ms), e.kind, e.path,
                e.status.map(|s| s.to_string()).unwrap_or_else(|| "…".into()),
                dur,
                if e.stage.is_empty() { String::new() } else { format!("  stage={}", e.stage) });
        }
    }

    let _ = writeln!(r, "\n---- Raw activity log (newest first, {} entries) ----", activity.len());
    for e in &activity {
        let _ = writeln!(r, "  {}  {:<8} {:<28} {}  job={} {}",
            fmt_ms(e.started_at_ms), e.kind, e.path,
            e.status.map(|s| s.to_string()).unwrap_or_else(|| "…".into()),
            if e.job_id.is_empty() { "-" } else { &e.job_id },
            if e.stage.is_empty() { String::new() } else { format!("stage={}", e.stage) });
    }

    let _ = writeln!(r, "\n======================= Application log (companion.log) =======================");
    match std::fs::read_to_string(log_file_path()) {
        Ok(contents) => {
            r.push_str(&contents);
        }
        Err(e) => {
            let _ = writeln!(r, "(could not read log file: {e})");
        }
    }
    r
}

/// Write a full diagnostics report to the Downloads folder and reveal it.
/// Returns the saved path. Powers the GUI "Export logs" button.
#[tauri::command]
async fn export_logs(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    let rd = app.path().resource_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let dd = app.path().app_data_dir().unwrap_or_else(|_| rd.clone());
    let report = build_diagnostics_report(&state, sidecar::build_kind(&rd, &dd)).await;
    let dir = dirs::download_dir()
        .or_else(dirs::desktop_dir)
        .or_else(dirs::home_dir)
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&dir);
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = dir.join(format!("clipai-companion-logs_{stamp}.txt"));
    std::fs::write(&path, report).map_err(|e| format!("could not write report: {e}"))?;
    // Best-effort reveal in the OS file manager so the user can grab it.
    {
        use tauri_plugin_opener::OpenerExt;
        let _ = app.opener().reveal_item_in_dir(&path);
    }
    log::info!("exported diagnostics report to {}", path.display());
    Ok(path.to_string_lossy().to_string())
}

/// Test whether a ClipAI container is properly connected to THIS companion.
/// The real data flow is ClipAI → companion (ClipAI is the client hitting our
/// proxy on :11500), so the authoritative signal is "did an authenticated
/// ClipAI request arrive recently". When we also know ClipAI's URL (paired),
/// probe it back to confirm the LAN link is healthy both directions.
#[tauri::command]
async fn test_clipai(state: tauri::State<'_, SharedState>) -> Result<serde_json::Value, String> {
    let cfg = state.config_snapshot();
    let now = state::now_ms();
    let last = state.last_clipai_contact_ms();
    let contacted = last > 0;
    let secs_ago = if contacted { now.saturating_sub(last) / 1000 } else { 0 };

    // Active back-probe (companion → ClipAI) when we know the URL. Uses an
    // unauthenticated ClipAI endpoint (the companion doesn't store ClipAI's key).
    let mut probe = serde_json::json!({ "attempted": false });
    let url = cfg.paired_clipai_url.trim().trim_end_matches('/').to_string();
    if !url.is_empty() {
        let target = format!("{url}/api/providers/status");
        match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(client) => match client.get(&target).send().await {
                Ok(r) => {
                    probe = serde_json::json!({
                        "attempted": true, "ok": r.status().is_success(),
                        "status": r.status().as_u16(), "url": url,
                    });
                }
                Err(e) => {
                    probe = serde_json::json!({
                        "attempted": true, "ok": false,
                        "error": format!("{e}"), "url": url,
                    });
                }
            },
            Err(e) => probe = serde_json::json!({ "attempted": true, "ok": false, "error": format!("{e}") }),
        }
    }

    Ok(serde_json::json!({
        "contacted": contacted,
        "connected": state.clipai_connected(),
        "serving": state.serving_jobs(),
        "secs_ago": secs_ago,
        "paired_url": url,
        "probe": probe,
        "port": cfg.port,
        "lan_ip": pairing::detect_lan_ip(),
    }))
}

/// Free the WHOLE GPU right now (e.g. before gaming): unload every resident
/// Ollama model AND stop the whisper sidecar (skipped only when a decode is
/// mid-flight). Previously this button evicted Ollama models but left the
/// whisper server holding its VRAM until the idle reaper.
#[tauri::command]
async fn free_vram(state: tauri::State<'_, SharedState>) -> Result<serde_json::Value, String> {
    let st: SharedState = state.inner().clone();
    let (whisper_stopped, n) = sidecar::free_gpu(&st, "Free GPU memory button").await;
    Ok(serde_json::json!({ "unloaded": n, "whisper_stopped": whisper_stopped }))
}

/// Force-clear a stuck active-job display AND free the GPU (Ollama models
/// unloaded + whisper sidecar stopped). Used by the GUI "Force end" button
/// when ClipAI reports a job the Companion never heard finish (e.g. the
/// container was stopped mid-job). Local-only — it does not command ClipAI
/// (no reverse channel); it just stops the Companion showing a phantom job
/// and frees the VRAM it was holding.
#[tauri::command]
async fn end_active_job(state: tauri::State<'_, SharedState>) -> Result<serde_json::Value, String> {
    // Sticky force-end: clear + SUPPRESS the job id so a still-heartbeating
    // ClipAI can't resurrect the card, and finish its in-flight activity.
    let ended = state.force_end_job();
    state.job_progress.store(0, std::sync::atomic::Ordering::Relaxed);
    let st: SharedState = state.inner().clone();
    let (whisper_stopped, n) = sidecar::free_gpu(&st, "Force end button").await;
    log::info!(
        "end_active_job: force-ended {} + unloaded {n} model(s), whisper_stopped={whisper_stopped}",
        ended.as_deref().unwrap_or("(none)")
    );
    Ok(serde_json::json!({
        "unloaded": n, "whisper_stopped": whisper_stopped, "ended_job": ended,
    }))
}

/// ── App self-update (served by the paired ClipAI container) ─────────────
/// The container's /api/downloads/companion/* routes hold the newest
/// installer (GitHub-release cache or an image-baked from-source build), so
/// the Companion can update ITSELF over the LAN with no GitHub dependency:
/// check compares the manifest version against this build; install downloads
/// the platform installer to a temp file, launches it, and exits the app so
/// the installer can replace files. Both fail-soft with actionable errors.

fn paired_base(state: &SharedState) -> Result<String, String> {
    let url = state
        .config_snapshot()
        .paired_clipai_url
        .trim()
        .trim_end_matches('/')
        .to_string();
    if !url.is_empty() {
        return Ok(url);
    }
    // Manually-added setups (endpoint+token pasted into ClipAI) never pair
    // from this side — fall back to the address learned from inbound traffic.
    let seen = state.seen_clipai_url();
    if !seen.is_empty() {
        return Ok(seen.trim_end_matches('/').to_string());
    }
    Err("This Companion doesn't know your ClipAI server's address yet. \
         Either use \"Pair now\" (paste the ClipAI URL + API key), or update \
         the ClipAI container — new builds identify themselves on every \
         request and the address is learned automatically."
        .into())
}

#[cfg(target_os = "macos")]
const UPDATE_PLATFORM: &str = "mac";
#[cfg(not(target_os = "macos"))]
const UPDATE_PLATFORM: &str = "windows";

/// Best-effort fetch of the published installer SHA-256 for this platform from
/// ClipAI's manifest, used to integrity-check the download before running it.
/// Returns None if the manifest can't be fetched/parsed or carries no hash —
/// the caller then falls back to the size-only gate.
async fn fetch_installer_sha256(client: &reqwest::Client, base: &str) -> Option<String> {
    let url = format!("{base}/api/downloads/companion/manifest");
    let resp = client.get(&url).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let manifest: serde_json::Value = resp.json().await.ok()?;
    manifest["platforms"][UPDATE_PLATFORM]["sha256"]
        .as_str()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[tauri::command]
async fn check_app_update(
    state: tauri::State<'_, SharedState>,
) -> Result<serde_json::Value, String> {
    let base = paired_base(&state.inner().clone())?;
    let url = format!("{base}/api/downloads/companion/manifest");
    log::info!("self-update: checking manifest at {url}");
    let resp = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("{e}"))?
        .get(&url)
        .send()
        .await
        // Name the exact URL tried so a wrong learned address is obvious. The
        // usual cause on an odd network: the address the container reaches us
        // from ({base}) isn't the address WE reach ClipAI at — use "Pair now"
        // to set the ClipAI URL explicitly and override the inference.
        .map_err(|e| format!(
            "Couldn't reach ClipAI at {base} ({e}). If that address is wrong, \
             click \"Pair now\" and paste your ClipAI URL + key to set it \
             explicitly."))?;
    if !resp.status().is_success() {
        return Err(format!(
            "ClipAI at {base} answered HTTP {} for the installer manifest.",
            resp.status()));
    }
    let manifest: serde_json::Value = resp.json().await.map_err(|e| format!("{e}"))?;
    let latest = manifest["version"].as_str().unwrap_or("").to_string();
    let entry = &manifest["platforms"][UPDATE_PLATFORM];
    let has_installer = entry.is_object();
    let current = env!("CARGO_PKG_VERSION");
    let current_build = env!("CLIPAI_BUILD_ID");
    // Build identity of the served installer (the ClipAI repo's git SHA baked
    // into the manifest). Comparing it lets a from-source rebuild that reused
    // the same semver still count as an update — otherwise the button says
    // "up to date" forever and the new binary never lands.
    let latest_build = manifest["build_id"].as_str().unwrap_or("").to_string();
    let update_available = has_installer
        && crate::proxy::should_update(current, current_build, &latest, &latest_build);
    Ok(serde_json::json!({
        "current": current,
        "latest": latest,
        "current_build": current_build,
        "latest_build": latest_build,
        "update_available": update_available,
        "installer_available": has_installer,
        "platform": UPDATE_PLATFORM,
        "filename": entry["filename"].as_str().unwrap_or(""),
        "size": entry["size"].as_u64().unwrap_or(0),
        "source": entry["source"].as_str().unwrap_or(""),
        "sha256": entry["sha256"].as_str().unwrap_or(""),
    }))
}

// ── Self-update core — shared by the GUI Update button and the REMOTE
//    /v1/update/* proxy routes (so ClipAI's web UI can push the update
//    without anyone at the GPU PC). Progress lives in a process-global so
//    both the Tauri command and the proxy can report it. ──────────────────

#[derive(Clone, Default)]
pub(crate) struct UpdateStatus {
    pub state: String,        // downloading | verifying | launching | failed
    pub progress_pct: f64,    // download progress (0-100; -1 = size unknown)
    pub downloaded_mb: f64,
    pub total_mb: f64,
    pub error: String,
    pub started_ms: u64,
}

static UPDATE_STATUS: std::sync::Mutex<Option<UpdateStatus>> =
    std::sync::Mutex::new(None);
static UPDATE_RUNNING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// After this long, an "in progress" update is presumed dead and a new attempt
/// takes the lock. The happy path never reaches it — a successful update exits
/// the process within seconds of reaching "launching" — but several unhappy
/// paths leave the flag set forever in a still-live app: the installer's UAC
/// prompt is dismissed, the launch succeeds but the swap does not, or a
/// download stalls under a timeout longer than a user will wait. Without an
/// escape those states brick updating until someone restarts the app by hand,
/// which on a headless GPU box means walking to it. Ten minutes is longer than
/// any real download on a LAN and shorter than anyone's patience.
const UPDATE_STALE_MS: u64 = 600_000;

/// Whether a new update attempt may take a lock another attempt still holds.
/// `prev_started_ms == 0` means we hold the flag but never recorded a start —
/// nothing to defer to, so take it.
pub(crate) fn update_lock_is_stale(prev_started_ms: u64, now_ms: u64) -> bool {
    prev_started_ms == 0 || now_ms.saturating_sub(prev_started_ms) >= UPDATE_STALE_MS
}

fn update_status_set(f: impl FnOnce(&mut UpdateStatus)) {
    if let Ok(mut g) = UPDATE_STATUS.lock() {
        let mut s = g.take().unwrap_or_default();
        f(&mut s);
        *g = Some(s);
    }
}

pub(crate) fn update_status_json() -> serde_json::Value {
    let snap = UPDATE_STATUS.lock().ok().and_then(|g| g.clone());
    match snap {
        None => serde_json::json!({"state": "idle", "running": false}),
        Some(s) => serde_json::json!({
            "state": s.state,
            "running": UPDATE_RUNNING.load(std::sync::atomic::Ordering::Relaxed),
            "progress_pct": s.progress_pct,
            "downloaded_mb": s.downloaded_mb,
            "total_mb": s.total_mb,
            "error": s.error,
            "started_ms": s.started_ms,
            "app_version": env!("CARGO_PKG_VERSION"),
            "app_build": env!("CLIPAI_BUILD_ID"),
        }),
    }
}

/// Download → verify → launch the installer. `silent` is the REMOTE path:
/// the NSIS installer runs with /S and the app is relaunched afterwards, so
/// the whole cycle needs nobody at the desktop (Windows only — a .dmg can't
/// be installed unattended). Returns the installer path; the CALLER decides
/// how to exit the app (GUI: AppHandle.exit; remote: process::exit after the
/// HTTP response is flushed — on Windows the kill-on-close job object still
/// reaps the managed Ollama/whisper children on a hard exit).
pub(crate) async fn perform_self_update(
    state: &SharedState, silent: bool,
) -> Result<String, String> {
    if silent && UPDATE_PLATFORM != "windows" {
        return Err("remote (unattended) update is Windows-only — a macOS .dmg \
                    needs a user at the machine; use the Companion app's own \
                    Update button".into());
    }
    let started = state::now_ms();
    if UPDATE_RUNNING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        // Held. Concede only to an attempt that is still plausibly alive; a
        // stale flag must never be the reason an update cannot be started.
        let prev = UPDATE_STATUS
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.started_ms))
            .unwrap_or(0);
        if !update_lock_is_stale(prev, started) {
            return Err("an update is already in progress".into());
        }
        log::warn!(
            "self-update: taking over a stale in-progress update (started {} ms \
             ago) — an update is never blocked by a previous attempt",
            started.saturating_sub(prev)
        );
    }
    update_status_set(|s| {
        *s = UpdateStatus {
            state: "downloading".into(), progress_pct: 0.0,
            downloaded_mb: 0.0, total_mb: 0.0, error: String::new(),
            started_ms: started,
        }
    });
    let res = perform_self_update_inner(state, silent).await;
    match &res {
        Ok(_) => update_status_set(|s| s.state = "launching".into()),
        Err(e) => {
            update_status_set(|s| { s.state = "failed".into(); s.error = e.clone(); });
            UPDATE_RUNNING.store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }
    res
}

async fn perform_self_update_inner(
    state: &SharedState, silent: bool,
) -> Result<String, String> {
    let base = paired_base(state)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("{e}"))?;
    // First, learn the installer's AUTHORITATIVE sha256 from the manifest. In
    // the honest path ClipAI computed this by verifying the installer against
    // the GitHub release manifest before serving it, so a byte-for-byte match
    // means we're about to run exactly what GitHub released. Best-effort: a
    // manifest hiccup or a hand-dropped installer with no published hash falls
    // back to the size-only gate (logged), rather than blocking the update.
    let expected_sha256 = fetch_installer_sha256(&client, &base).await;
    let url = format!("{base}/api/downloads/companion/{UPDATE_PLATFORM}");
    log::info!("self-update: downloading installer from {url} (silent={silent})");
    let mut resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "no installer available (HTTP {}) — on the ClipAI side, run \
             \"Check for Companion updates\" in Settings → GPU Companion, or \
             rebuild the container with COMPANION_BUILD_FROM_SOURCE=1",
            resp.status()
        ));
    }
    // On-disk name: a FIXED, safe filename per platform. The server's
    // Content-Disposition ("ClipAI GPU Companion_0.2.6_x64-setup.exe") carries
    // spaces AND an RFC-5987 ``filename*=utf-8''…`` part; the old
    // ``split("filename=").nth(1)`` swallowed that tail, producing an on-disk
    // path with embedded quotes/`;`/`%` that shattered the launch script's
    // quoting and popped a "Windows cannot find '\\'" dialog while the app was
    // already exiting. We don't need the server's name for anything — the file
    // is a throwaway we execute once — so never trust it for the FS path.
    let filename = if UPDATE_PLATFORM == "mac" {
        "ClipAI-GPU-Companion-update.dmg"
    } else {
        "ClipAI-GPU-Companion-update.exe"
    };
    // Stream the body so the progress bar (GUI + ClipAI's remote card) tracks
    // the real download instead of jumping 0 → done.
    let total = resp.content_length().unwrap_or(0);
    let mut bytes: Vec<u8> = Vec::with_capacity(total as usize);
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| format!("download interrupted: {e}"))?
    {
        bytes.extend_from_slice(&chunk);
        let done = bytes.len() as f64;
        update_status_set(|s| {
            s.downloaded_mb = done / (1024.0 * 1024.0);
            s.total_mb = total as f64 / (1024.0 * 1024.0);
            s.progress_pct = if total > 0 {
                (done / total as f64 * 100.0).min(100.0)
            } else {
                -1.0
            };
        });
    }
    if bytes.len() < 1_000_000 {
        // A real installer is tens of MB; a tiny body is an error page.
        return Err(format!(
            "downloaded file is implausibly small ({} bytes) — not running it",
            bytes.len()
        ));
    }
    update_status_set(|s| s.state = "verifying".into());
    // Integrity gate: the installer is about to be EXECUTED, so when the server
    // publishes a hash we refuse to run bytes that don't match it. This closes
    // the "corrupted download or on-path rewrite over the plaintext LAN hop
    // slips through a size-only check" gap. No published hash → size gate only.
    if let Some(expected) = expected_sha256.as_deref().filter(|h| !h.is_empty()) {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got = hex::encode(hasher.finalize());
        if !got.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "installer integrity check FAILED — the download does not match the \
                 hash ClipAI published (expected {expected}, got {got}). Refusing to \
                 run it. This usually means a corrupted download or a tampered network \
                 path; try again, or use \"Pair now\" to set your ClipAI URL explicitly."
            ));
        }
        log::info!("self-update: installer sha256 verified ({got})");
    } else {
        log::warn!(
            "self-update: ClipAI published no installer sha256 — running on the size \
             check alone (hand-dropped installer or older manifest)"
        );
    }
    let path = std::env::temp_dir().join(&filename);
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| format!("could not save installer: {e}"))?;
    log::info!(
        "self-update: launching installer {} ({} MB, silent={}) and exiting so \
         it can replace the app",
        path.display(),
        bytes.len() / (1024 * 1024),
        silent
    );
    if silent {
        // Unattended (remote-triggered): wait for this app to exit, run the
        // NSIS installer silently, then relaunch the (replaced) app so the
        // Companion comes back online with nobody at the desktop.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let exe = std::env::current_exe()
                .map_err(|e| format!("could not resolve app path: {e}"))?;
            let exe_str = exe.display().to_string();
            let path_str = path.display().to_string();
            // Guard the exact failure that stranded a user: NEVER build a
            // launch line around an empty/degenerate path (that is what
            // produced `start "" ""` → the "cannot find '\\'" dialog).
            if exe_str.trim().is_empty() || path_str.trim().is_empty() {
                return Err(format!(
                    "refusing to launch silent update — degenerate path \
                     (installer={path_str:?}, app={exe_str:?})"
                ));
            }
            // Relaunch must be RESILIENT — nobody is at the desk to restart a
            // Companion that fails to come back. The image name lets the script
            // confirm the app is actually RUNNING (not just that its .exe file
            // exists) and keep retrying the launch until it is.
            let exe_name = exe
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("ClipAI GPU Companion.exe")
                .to_string();
            // Write a real .cmd file instead of an inline `cmd /C "…"` string:
            // the app's own path has spaces, and interpolating it into a
            // single command line is what let quoting break. A script file
            // sidesteps all shell-escaping of the interpolated paths.
            //   * `ping -n N 127.0.0.1` waits WITHOUT a console — `timeout`
            //     needs console input and silently errors under CREATE_NO_WINDOW,
            //     so the app could still hold files when the installer started.
            //   * after the installer runs, LOOP: launch the app, then poll
            //     tasklist for its image; only stop once it is actually up.
            //     If the installer failed and left the OLD exe in place, this
            //     still brings the (old) Companion back online rather than
            //     leaving it dead — a broken update must never take the app
            //     offline until someone walks over to the desktop.
            //   * a log next to the script records the outcome for diagnosis.
            let script_path = std::env::temp_dir().join("clipai-companion-update.cmd");
            let log_path = std::env::temp_dir()
                .join("clipai-companion-update.log");
            let log_str = log_path.display().to_string();
            let script = format!(
                "@echo off\r\n\
                 echo [%date% %time%] update script start >\"{log_str}\"\r\n\
                 ping -n 4 127.0.0.1 >nul\r\n\
                 echo [%date% %time%] running installer >>\"{log_str}\"\r\n\
                 \"{path_str}\" /S\r\n\
                 echo [%date% %time%] installer exit=%errorlevel% >>\"{log_str}\"\r\n\
                 ping -n 3 127.0.0.1 >nul\r\n\
                 set _started=0\r\n\
                 for /l %%i in (1,1,40) do (\r\n\
                 \x20 tasklist /fi \"imagename eq {exe_name}\" 2>nul | find /i \"{exe_name}\" >nul && ( set _started=1 & goto up )\r\n\
                 \x20 if exist \"{exe_str}\" start \"\" \"{exe_str}\"\r\n\
                 \x20 ping -n 3 127.0.0.1 >nul\r\n\
                 )\r\n\
                 :up\r\n\
                 echo [%date% %time%] app running=%_started% >>\"{log_str}\"\r\n\
                 del \"%~f0\"\r\n"
            );
            std::fs::write(&script_path, script)
                .map_err(|e| format!("could not write update script: {e}"))?;
            log::info!(
                "self-update: silent update script written to {} (installer={}, app={})",
                script_path.display(), path_str, exe_str
            );
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            const DETACHED_PROCESS: u32 = 0x0000_0008;
            std::process::Command::new("cmd")
                // Pass the script path as a real argv entry (not concatenated
                // into a command string) so its own spaces can't break parsing.
                .args(["/C", &script_path.display().to_string()])
                .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
                .spawn()
                .map_err(|e| format!("could not launch silent installer: {e}"))?;
        }
        #[cfg(not(windows))]
        return Err("silent update is Windows-only".into());
    } else {
        #[cfg(target_os = "macos")]
        std::process::Command::new("open")
            .arg(&path)
            .spawn()
            .map_err(|e| format!("could not open installer: {e}"))?;
        #[cfg(not(target_os = "macos"))]
        std::process::Command::new(&path)
            .spawn()
            .map_err(|e| format!("could not launch installer: {e}"))?;
    }
    #[allow(unreachable_code)]
    Ok(path.display().to_string())
}

#[tauri::command]
async fn install_app_update(
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedState>,
) -> Result<String, String> {
    let shared = state.inner().clone();
    // Same rule as the remote path: clicking Update ends whatever this GPU is
    // doing rather than the click doing nothing. The app is seconds from being
    // replaced under any in-flight work, so cancel it cleanly and hand the
    // installer a quiet machine — a whisper sidecar still holding a file is a
    // file the installer cannot swap.
    if let Some(job) = shared.current_job() {
        log::warn!(
            "self-update: ending in-flight job {} before installing",
            job.job_title
        );
    }
    proxy::force_end_everything(&shared, "self-update (app Update button)").await;
    let path = perform_self_update(&shared, false).await?;
    // Give the GUI a moment to render the "installer started" state, then
    // exit — the RunEvent::Exit handler stops Ollama/whisper children so the
    // installer can replace every file.
    let handle = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(1500));
        handle.exit(0);
    });
    Ok(path)
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
        let code = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let detail: String = body.trim().chars().take(200).collect();
        return Err(format!("Ollama HTTP {code}{}", if detail.is_empty() { String::new() } else { format!(": {detail}") }));
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
                show_or_create_main(app);
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
                    sidecar::shutdown(&state, "app quit").await;
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
            show_or_create_main(app);
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .setup(|app| {
            log::info!("setup: begin");
            // Reboot backstop for unattended updates: register the app to run
            // at login so that even if a self-update's relaunch fails, or the
            // GPU box simply reboots, the Companion comes back online without
            // anyone at the desk. Idempotent; best-effort (a locked-down
            // machine may refuse the registration — never fatal).
            {
                use tauri_plugin_autostart::ManagerExt;
                let al = app.autolaunch();
                match al.is_enabled() {
                    Ok(true) => log::info!("autostart: already enabled"),
                    _ => match al.enable() {
                        Ok(()) => log::info!("autostart: enabled (login relaunch backstop)"),
                        Err(e) => log::warn!("autostart: could not enable ({e})"),
                    },
                }
            }
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
                sidecar::available(&resource_dir, &data_dir),
                Ordering::Relaxed,
            );
            app.manage(state.clone());
            log::info!(
                "setup: state loaded (sidecar_available={})",
                state.sidecar_available.load(Ordering::Relaxed)
            );

            // GPU Whisper by default: on a from-source install (which ships no
            // bundled sidecar) with an NVIDIA GPU, install the CUDA whisper.cpp
            // build in the background so transcription runs ON THE GPU out of the
            // box — the user never has to click "Download Whisper", and a CPU
            // build (unusably slow for large models) is upgraded once. Latched in
            // config so it attempts at most once per machine (no re-download
            // loop); a manual "Install GPU build" button remains in the GUI.
            #[cfg(target_os = "windows")]
            {
                let state = state.clone();
                let app_handle = app.handle().clone();
                let rd = resource_dir.clone();
                let dd = data_dir.clone();
                tauri::async_runtime::spawn(async move {
                    let already = state.config.lock().unwrap().whisper_autoinstalled;
                    let kind = sidecar::build_kind(&rd, &dd);
                    // Nothing to do if a GPU/bundled build is already in place, or
                    // we've already tried once on this machine.
                    if already || kind == "gpu" || kind == "bundled" {
                        return;
                    }
                    let is_nvidia = tokio::task::spawn_blocking(gpu::snapshot)
                        .await
                        .map(|s| s.gpu_name.to_lowercase().contains("nvidia"))
                        .unwrap_or(false);
                    if !is_nvidia {
                        return; // CPU-only host: leave transcription on the ClipAI server.
                    }
                    log::info!(
                        "auto-installing GPU whisper build (current='{kind}', NVIDIA GPU present)"
                    );
                    // Release any file lock on an existing whisper-server.exe so
                    // the overwrite (CPU→GPU swap) can't fail. No-op if unstarted.
                    sidecar::shutdown(&state, "GPU whisper auto-install").await;
                    match sidecar::download_whispercpp(&app_handle, &dd).await {
                        Ok(msg) => log::info!("auto-install whisper: {msg}"),
                        Err(e) => log::warn!(
                            "auto-install whisper failed (retry via the GUI button): {e}"
                        ),
                    }
                    state.sidecar_available.store(
                        sidecar::available(&rd, &dd), Ordering::Relaxed);
                    // Latch regardless of outcome so a transient failure can't loop.
                    {
                        let mut c = state.config.lock().unwrap();
                        c.whisper_autoinstalled = true;
                    }
                    state.save();
                });
            }

            // GPU telemetry poll (5 s) + tray tooltip/busy indicator.
            {
                let state = state.clone();
                let app_handle = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    // GPU-residency watchdog state: consecutive ticks a model
                    // has been seen spilled to CPU, and when it was last
                    // evicted (per-model cooldown against eviction storms).
                    let mut spill_ticks: std::collections::HashMap<String, u32> =
                        std::collections::HashMap::new();
                    let mut spill_last_evict_ms: std::collections::HashMap<String, u64> =
                        std::collections::HashMap::new();
                    loop {
                        let snap = tokio::task::spawn_blocking(gpu::snapshot)
                            .await
                            .unwrap_or_default();
                        *state.gpu.lock().unwrap() = snap.clone();

                        // ── Auto-VRAM: track what other apps use and re-share ──
                        if snap.vram_total_mb > 0 {
                            // Baseline (non-companion VRAM) is only meaningful
                            // when Ollama holds no model — else "used" includes
                            // our own model. Measured here so it reflects games.
                            if ollama::loaded_model_count().await == 0 {
                                let baseline =
                                    snap.vram_total_mb.saturating_sub(snap.vram_free_mb);
                                state.gpu_baseline_used_mb.store(baseline, Ordering::Relaxed);
                            }
                            let auto = state.config.lock().unwrap().vram_auto;
                            if auto {
                                let busy = state.whisper_busy.load(Ordering::Relaxed)
                                    || state.current_job().is_some();
                                let managed = state.ollama_child.lock().await.is_some();
                                let now = state::now_ms();
                                let since = now.saturating_sub(
                                    state.last_auto_apply_ms.load(Ordering::Relaxed));
                                let baseline =
                                    state.gpu_baseline_used_mb.load(Ordering::Relaxed);
                                let last = state.last_auto_baseline_mb.load(Ordering::Relaxed);
                                // Re-apply only when idle and free VRAM has
                                // drifted >1 GB since we last set the budget, at
                                // most every 30 s (a restart briefly unloads).
                                if managed && !busy && since > 30_000 && baseline.abs_diff(last) > 1024 {
                                    log::info!(
                                        "auto-vram: free VRAM changed (baseline {last}→{baseline} MB) — re-sharing"
                                    );
                                    if ollama::restart(&state).await.is_ok() {
                                        state.last_auto_baseline_mb.store(baseline, Ordering::Relaxed);
                                        state.last_auto_apply_ms.store(now, Ordering::Relaxed);
                                    }
                                }
                            }
                        }

                        // ── GPU-residency watchdog: a model that Ollama
                        // scheduled onto the CPU during a moment of VRAM
                        // pressure stays there for its whole keep_alive, even
                        // after the pressure clears (measured: qwen2.5:14b
                        // translating on CPU beside 10.3 GB of free VRAM).
                        // Evict it so the next request reloads ON the GPU.
                        if snap.vram_total_mb > 0 {
                            let spilled =
                                ollama::cpu_spilled_models(snap.vram_free_mb).await;
                            // A model no longer spilled (or gone) resets its streak.
                            spill_ticks.retain(|k, _| spilled.contains(k));
                            for name in spilled {
                                let ticks = spill_ticks.entry(name.clone()).or_insert(0);
                                *ticks += 1;
                                if *ticks < 2 {
                                    continue; // could be a load in flight — confirm next tick
                                }
                                let now = state::now_ms();
                                let last =
                                    spill_last_evict_ms.get(&name).copied().unwrap_or(0);
                                if now.saturating_sub(last) < 180_000 {
                                    continue; // per-model cooldown
                                }
                                log::info!(
                                    "gpu watchdog: '{name}' is running on the CPU while \
                                     {} MB of VRAM is free — unloading it so the next \
                                     request reloads on the GPU",
                                    snap.vram_free_mb
                                );
                                if ollama::unload_model(&name).await {
                                    spill_last_evict_ms.insert(name.clone(), now);
                                    spill_ticks.remove(&name);
                                }
                            }
                        }

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

            // Address-change self-healing: this Companion's IP is a DHCP
            // lease, and every renewal used to strand ClipAI on the dead
            // address until the user deleted and re-added the pairing by
            // hand. ClipAI rebinds registrations by our persistent pairing
            // token, so re-announcing from the new address heals the
            // registry, Whisper routing, and the GPU metadata in place.
            // Re-announce IMMEDIATELY when the LAN IP changes, and as a
            // low-cost heartbeat every 15 minutes so a ClipAI-side restart
            // that lost its settings also re-learns us.
            {
                let state = state.clone();
                tauri::async_runtime::spawn(async move {
                    let mut last_ip = pairing::detect_lan_ip().unwrap_or_default();
                    let mut last_announce = std::time::Instant::now()
                        - std::time::Duration::from_secs(3600);
                    let mut last_fail_log = std::time::Instant::now()
                        - std::time::Duration::from_secs(3600);
                    loop {
                        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                        let ip = pairing::detect_lan_ip().unwrap_or_default();
                        let ip_changed = !ip.is_empty() && ip != last_ip;
                        let heartbeat_due =
                            last_announce.elapsed().as_secs() >= 900;
                        if !ip_changed && !heartbeat_due {
                            continue;
                        }
                        match pairing::reannounce(&state).await {
                            Ok(true) => {
                                if ip_changed {
                                    log::info!(
                                        "re-announced to ClipAI after LAN IP \
                                         change {last_ip} → {ip}"
                                    );
                                }
                                last_ip = ip;
                                last_announce = std::time::Instant::now();
                            }
                            Ok(false) => {
                                // Never paired (or paired before the key was
                                // stored) — nothing to heal.
                                last_ip = ip;
                                last_announce = std::time::Instant::now();
                            }
                            Err(e) => {
                                // ClipAI may simply be off; don't spam.
                                if last_fail_log.elapsed().as_secs() >= 3600 {
                                    log::warn!(
                                        "re-announce to ClipAI failed \
                                         (will keep retrying): {e}"
                                    );
                                    last_fail_log = std::time::Instant::now();
                                }
                            }
                        }
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
            // Guarantee the window is visible on launch regardless of config
            // defaults — a hidden-but-alive window reads as "GUI unreachable".
            show_or_create_main(&app.handle().clone());
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
            refresh_sidecar,
            download_whisper,
            export_logs,
            test_clipai,
            free_vram,
            end_active_job,
            pair_clipai,
            check_app_update,
            install_app_update,
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
                    // Bound the graceful shutdown so a hung child.wait() can't
                    // hold the proxy port + single-instance lock into the next
                    // launch (which showed up as the Companion being
                    // unreachable after a quick restart). On Windows the
                    // kill-on-job-close object reaps children regardless.
                    tauri::async_runtime::block_on(async move {
                        let _ = tokio::time::timeout(
                            std::time::Duration::from_secs(3),
                            async {
                                sidecar::shutdown(&state, "app exit").await;
                                ollama::shutdown(&state).await;
                            },
                        )
                        .await;
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

#[cfg(test)]
mod update_lock_tests {
    use super::{update_lock_is_stale, UPDATE_STALE_MS};

    #[test]
    fn a_live_attempt_still_holds_the_lock() {
        let now = 1_000_000u64;
        assert!(!update_lock_is_stale(now - 1_000, now));
        assert!(!update_lock_is_stale(now - (UPDATE_STALE_MS - 1), now));
    }

    #[test]
    fn a_dead_attempt_never_blocks_a_new_one() {
        // The failure that stranded a user: the installer launched, the swap
        // never happened, the app stayed alive with the flag set. Without this
        // escape, updating is bricked until someone restarts the app by hand.
        let now = 10_000_000u64;
        assert!(update_lock_is_stale(now - UPDATE_STALE_MS, now));
        assert!(update_lock_is_stale(now - UPDATE_STALE_MS * 10, now));
    }

    #[test]
    fn a_flag_with_no_recorded_start_is_stale() {
        assert!(update_lock_is_stale(0, 1_000_000));
        assert!(update_lock_is_stale(0, 0));
    }

    #[test]
    fn a_clock_that_went_backwards_does_not_wedge_the_lock() {
        // saturating_sub yields 0, which reads as "just started" — the one
        // case where we defer. It self-heals once the clock passes the mark.
        assert!(!update_lock_is_stale(2_000_000, 1_000_000));
    }
}
