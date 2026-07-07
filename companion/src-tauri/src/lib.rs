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
        "config": {
            "token": config.token,
            "port": config.port,
            "vram_budget_gb": config.vram_budget_gb,
            "vram_auto": config.vram_auto,
            "vram_buffer_gb": config.vram_buffer_gb,
            "ollama_keep_alive": config.ollama_keep_alive,
            "sidecar_idle_min": config.sidecar_idle_min,
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
    sidecar::shutdown(&state).await;
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
    let _ = writeln!(r, "App version:   {}", env!("CARGO_PKG_VERSION"));
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

/// Unload all resident Ollama models to free GPU VRAM right now (e.g. before
/// gaming). Returns how many were unloaded.
#[tauri::command]
async fn free_vram() -> Result<serde_json::Value, String> {
    let (n, names) = ollama::unload_all().await;
    log::info!("free_vram: unloaded {n} model(s): {}", names.join(", "));
    Ok(serde_json::json!({ "unloaded": n, "models": names }))
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
                    sidecar::shutdown(&state).await;
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
                    // Bound the graceful shutdown so a hung child.wait() can't
                    // hold the proxy port + single-instance lock into the next
                    // launch (which showed up as the Companion being
                    // unreachable after a quick restart). On Windows the
                    // kill-on-job-close object reaps children regardless.
                    tauri::async_runtime::block_on(async move {
                        let _ = tokio::time::timeout(
                            std::time::Duration::from_secs(3),
                            async {
                                sidecar::shutdown(&state).await;
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
