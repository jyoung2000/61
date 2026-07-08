//! Shared Companion state: persisted config (token, port, VRAM budget),
//! the live activity feed the dashboard renders, and handles to the
//! managed Ollama / whisper-sidecar child processes.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROXY_PORT_DEFAULT: u16 = 11500;
pub const OLLAMA_LOCAL: &str = "127.0.0.1:11434";
pub const WHISPER_SIDECAR_PORT: u16 = 11510;
/// Request history kept in memory. Big enough that "Export logs" can dump
/// everything the app served since it opened; the dashboard and get_status
/// only ever render/serialize a small slice of it.
const ACTIVITY_CAP: usize = 1000;

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Bearer token every proxy route requires. Generated on first run;
    /// shown in the GUI with a copy button. NEVER logged.
    pub token: String,
    pub port: u16,
    /// Soft VRAM budget (GB) ClipAI may use on this GPU. Drives the
    /// whisper model tier and Ollama's OLLAMA_GPU_OVERHEAD reservation.
    pub vram_budget_gb: f32,
    /// Ollama keep-alive for loaded models (Ollama duration syntax).
    pub ollama_keep_alive: String,
    /// Minutes of inactivity before the whisper sidecar exits.
    pub sidecar_idle_min: u32,
    /// "Pause sharing" tray toggle — proxy answers 503 to everything.
    pub paused: bool,
    /// The ClipAI server this Companion is paired with (display only).
    pub paired_clipai_url: String,
    /// Friendly name announced to ClipAI when pairing.
    pub name: String,
    /// Whether the first-run wizard has completed.
    pub setup_complete: bool,
    /// Auto-allocate VRAM: budget tracks free VRAM (measured while Ollama is
    /// idle) so the card is shared dynamically with games/other apps — it uses
    /// what's free minus `vram_buffer_gb`, and backs off when other apps need
    /// more. When true, `vram_budget_gb` (the manual slider) is ignored.
    pub vram_auto: bool,
    /// GB always kept free for the desktop in auto mode.
    pub vram_buffer_gb: f32,
    /// One-shot latch: the GPU (CUDA) whisper build auto-install has been
    /// attempted once on this machine. Prevents re-download loops when a build
    /// is already present — a manual retry stays available in the GUI.
    pub whisper_autoinstalled: bool,
    /// How hard to push the GPU for ClipAI: "auto" (pick from VRAM + card),
    /// "eco" (1 job at a time — leave the card for games), "balanced", or
    /// "turbo" (max concurrency the VRAM allows). Drives Ollama parallelism and
    /// is advertised to ClipAI so it parallelizes the pipeline to match.
    pub speed_profile: String,
    /// Transcription accuracy vs speed: "auto" (pick from VRAM), "fast" (turbo,
    /// greedy), "balanced" (turbo + beam search), "max" (full large-v3 + beam
    /// search — best/"Netflix-grade", needs the VRAM). More VRAM ⇒ beam search
    /// and a bigger model, which is where the accuracy gains come from.
    pub whisper_quality: String,
    /// Folders the user has shared with ClipAI. ClipAI can browse these paths
    /// and pull video/media/font files from them remotely (e.g. drop a clip on
    /// a shared folder from your phone, then import it in ClipAI while away from
    /// the PC). Every file API is strictly JAILED to these roots — a request for
    /// any path that doesn't canonicalize to inside one of them is refused.
    pub shared_paths: Vec<String>,
    /// When true, share the ENTIRE computer (all drives) instead of just the
    /// folders in ``shared_paths`` — the file browser starts at the drive roots
    /// and can reach any path. Opt-in and off by default; a deliberate, powerful
    /// choice for a trusted LAN.
    pub share_all: bool,
}

/// Resolve transcription quality + VRAM budget to whisper.cpp decode settings:
/// (beam_size, prefer_full_large_v3). beam_size>1 enables beam search (the main
/// accuracy lever over greedy); prefer_full swaps large-v3-turbo for the full
/// large-v3 model (highest accuracy) when the VRAM affords it.
pub fn whisper_quality_params(quality: &str, budget_gb: f32) -> (u32, bool) {
    match quality {
        "fast" => (1, false),
        "balanced" => (5, false),
        // Full large-v3 needs ~3 GB just for weights + beam KV — gate on VRAM.
        "max" => (5, budget_gb >= 8.0),
        // "auto": turn on beam search once the card can afford it (the 4070
        // easily can); keep greedy only on tiny budgets.
        _ => {
            if budget_gb >= 6.0 {
                (5, false)
            } else if budget_gb >= 3.0 {
                (2, false)
            } else {
                (1, false)
            }
        }
    }
}

/// Resolve a speed profile + VRAM budget to (num_parallel, max_loaded_models).
/// The ceiling always fits the budget (each parallel slot costs KV-cache VRAM);
/// profiles pick how much of that ceiling to use. "auto"/"turbo" maximize speed
/// (the user's goal); "eco" keeps the card free for other apps.
pub fn resolve_speed(profile: &str, budget_gb: f32) -> (u32, u32) {
    let ceil_parallel: u32 = if budget_gb >= 16.0 {
        6
    } else if budget_gb >= 10.0 {
        4
    } else if budget_gb >= 7.0 {
        3
    } else if budget_gb >= 4.0 {
        2
    } else {
        1
    };
    let ceil_loaded: u32 = if budget_gb >= 10.0 {
        3
    } else if budget_gb >= 5.0 {
        2
    } else {
        1
    };
    match profile {
        "eco" => (1, 1),
        "balanced" => (ceil_parallel.min(2).max(1), ceil_loaded.min(2).max(1)),
        // "turbo" and "auto" both use the full safe ceiling for the VRAM.
        _ => (ceil_parallel.max(1), ceil_loaded.max(1)),
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            token: generate_token(),
            port: PROXY_PORT_DEFAULT,
            vram_budget_gb: 0.0, // 0 = auto (total minus ~1 GB headroom)
            ollama_keep_alive: "10m".into(),
            sidecar_idle_min: 15,
            paused: false,
            paired_clipai_url: String::new(),
            name: default_name(),
            setup_complete: false,
            vram_auto: false,
            vram_buffer_gb: 1.0,
            whisper_autoinstalled: false,
            speed_profile: "auto".into(),
            whisper_quality: "auto".into(),
            shared_paths: Vec::new(),
            share_all: false,
        }
    }
}

/// The storage-drive roots to expose when ``share_all`` is on: every existing
/// drive letter on Windows (C:\, D:\, …), or ``/`` on Unix. Lets the file
/// browser start at the top of each drive.
pub fn list_drive_roots() -> Vec<String> {
    if cfg!(windows) {
        let mut out = Vec::new();
        for c in b'A'..=b'Z' {
            let p = format!("{}:\\", c as char);
            if std::path::Path::new(&p).is_dir() {
                out.push(p);
            }
        }
        out
    } else {
        vec!["/".to_string()]
    }
}

/// Resolve a user-supplied path against the shared roots, enforcing the jail.
/// Returns the canonicalized absolute path ONLY when it exists and (a)
/// ``share_all`` is on — the whole computer is shared — or (b) it canonicalizes
/// to inside one of the (canonicalized) shared roots. Otherwise `None`.
/// Canonicalizing both sides defeats `..` traversal and symlink escapes: a
/// symlink inside a shared folder that points outside it resolves to a real path
/// that fails the prefix check.
pub fn resolve_shared_path(shared_roots: &[String], share_all: bool, requested: &str) -> Option<PathBuf> {
    let req = std::fs::canonicalize(requested).ok()?;
    if share_all {
        // The user opted to share the entire computer — any real path is allowed.
        return Some(req);
    }
    for root in shared_roots {
        if root.trim().is_empty() {
            continue;
        }
        if let Ok(root_c) = std::fs::canonicalize(root) {
            if req == root_c || req.starts_with(&root_c) {
                return Some(req);
            }
        }
    }
    None
}

fn default_name() -> String {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "GPU Companion".into())
}

pub fn generate_token() -> String {
    use rand::Rng;
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    (0..40)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}

#[derive(Clone, Serialize)]
pub struct ActivityEntry {
    pub id: u64,
    pub kind: String, // "ollama" | "whisper" | "health"
    pub path: String,
    pub job_id: String,
    pub job_title: String,
    pub stage: String,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub status: Option<u16>,
}

/// One video-analysis pipeline's worth of proxy activity, grouped by the
/// X-ClipAI-Job-Id header so the GUI can show a separate log per job.
#[derive(Clone, Serialize)]
pub struct JobLog {
    pub job_id: String,
    pub job_title: String,
    pub active: bool,
    pub started_at_ms: u64,
    pub last_activity_ms: u64,
    pub entries: Vec<ActivityEntry>,
    /// Stage ClipAI last reported for this job (heartbeat) — shown so the GUI
    /// can say WHICH stage is live even when it isn't hitting our proxy (a
    /// local-only stage like frame extraction or offline NMT translation).
    pub reported_stage: String,
    /// Overall job progress (0-100) from the heartbeat, or -1 when unknown.
    pub reported_progress: i64,
}

#[derive(Clone, Serialize, Default)]
pub struct GpuSnapshot {
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub vram_free_mb: u64,
    pub unified_memory: bool,
    pub available: bool,
}

pub struct AppState {
    pub config: Mutex<Config>,
    pub config_path: PathBuf,
    pub activity: Mutex<VecDeque<ActivityEntry>>,
    next_activity_id: AtomicU64,
    pub gpu: Mutex<GpuSnapshot>,
    /// Whisper sidecar child (spawned lazily, reaped on idle).
    pub sidecar: tokio::sync::Mutex<Option<crate::sidecar::SidecarHandle>>,
    /// Managed Ollama child (None when an external Ollama is used).
    pub ollama_child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    /// Guard so concurrent ensure_running() calls don't spawn `ollama serve`
    /// more than once (the supervisor loop + the UI's start button can race).
    pub ollama_starting: AtomicBool,
    /// Model pulls flowing THROUGH the proxy (initiated by ClipAI): model →
    /// (percent, last_update_ms). Lets the Companion GUI show downloads that
    /// ClipAI pushed, not just ones started from the Companion itself.
    pub incoming_pulls: Mutex<HashMap<String, (f64, u64)>>,
    /// ms epoch of when this app instance started — for uptime in exports.
    pub app_started_ms: u64,
    /// ms epoch of the last authenticated request from a ClipAI server (any
    /// proxy route). Drives the "ClipAI connected" indicator.
    pub last_clipai_contact: AtomicU64,
    /// ms epoch of the last REAL job request (inference/transcription) — i.e.
    /// not a /api/tags, /api/ps or /v1/health probe. Distinguishes "serving
    /// jobs" from merely "reachable" so the UI can say which is happening.
    pub last_job_ms: AtomicU64,
    /// Latest overall job progress (0-100) from ClipAI's X-ClipAI-Progress
    /// header, or u64::MAX when unknown — drives the live progress bar.
    pub job_progress: AtomicU64,
    /// VRAM (MB) used by NON-companion apps, measured while Ollama holds no
    /// resident model. Auto-VRAM sizes the budget as total − this − buffer.
    pub gpu_baseline_used_mb: AtomicU64,
    /// ms epoch of the last auto-VRAM Ollama restart (debounces re-applies).
    pub last_auto_apply_ms: AtomicU64,
    /// Baseline (MB) in effect at the last auto-VRAM apply, so we only restart
    /// Ollama when free VRAM has drifted materially (game started/stopped).
    pub last_auto_baseline_mb: AtomicU64,
    /// Serializes GPU-heavy whisper work: one transcription at a time.
    pub whisper_slot: tokio::sync::Semaphore,
    /// Last time any proxied request finished (ms epoch) — idle shutdown.
    pub last_request_ms: AtomicU64,
    /// True while a transcription request is in flight.
    pub whisper_busy: AtomicBool,
    /// Whether the whisper sidecar binary shipped with this build (set once
    /// at startup). From-source/cross-compiled builds may lack it — Ollama
    /// sharing still works; /v1/health + pairing report whisper honestly.
    pub sidecar_available: AtomicBool,
    /// True while the LAN proxy is bound and listening. False if the port bind
    /// failed (another process on the port) — surfaced to the GUI so an
    /// unreachable Companion is loud, not a silently dead proxy.
    pub proxy_bound: AtomicBool,
    /// Last proxy bind error (empty when bound) for the GUI banner.
    pub proxy_last_error: Mutex<String>,
    /// Progress ClipAI EXPLICITLY reported via POST /v1/progress. During local-
    /// only pipeline stages (video decode/frame-extract on the SERVER GPU) no AI
    /// request reaches us, so the in-flight-request view would freeze; this
    /// heartbeat keeps the GUI bar tracking the container. None until reported.
    pub reported_job: Mutex<Option<ReportedJob>>,
}

/// A job-progress heartbeat pushed by ClipAI (POST /v1/progress) so the GUI
/// tracks the container even when we're not actively serving an AI request.
#[derive(Clone, Serialize)]
pub struct ReportedJob {
    pub job_id: String,
    pub job_title: String,
    pub stage: String,
    pub progress: u64,
    pub updated_ms: u64,
    /// ms epoch of the FIRST heartbeat for this job_id — so the GUI's "elapsed"
    /// counts from when the job started here, not from the Unix epoch.
    pub started_ms: u64,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl AppState {
    pub fn load(config_dir: PathBuf) -> Self {
        let config_path = config_dir.join("companion.json");
        let config = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Config>(&raw).ok())
            .unwrap_or_default();
        let state = Self {
            config: Mutex::new(config),
            config_path,
            activity: Mutex::new(VecDeque::new()),
            next_activity_id: AtomicU64::new(1),
            gpu: Mutex::new(GpuSnapshot::default()),
            sidecar: tokio::sync::Mutex::new(None),
            ollama_child: tokio::sync::Mutex::new(None),
            ollama_starting: AtomicBool::new(false),
            incoming_pulls: Mutex::new(HashMap::new()),
            app_started_ms: now_ms(),
            last_clipai_contact: AtomicU64::new(0),
            last_job_ms: AtomicU64::new(0),
            job_progress: AtomicU64::new(u64::MAX),
            gpu_baseline_used_mb: AtomicU64::new(0),
            last_auto_apply_ms: AtomicU64::new(0),
            last_auto_baseline_mb: AtomicU64::new(u64::MAX),
            whisper_slot: tokio::sync::Semaphore::new(1),
            last_request_ms: AtomicU64::new(now_ms()),
            whisper_busy: AtomicBool::new(false),
            sidecar_available: AtomicBool::new(false),
            proxy_bound: AtomicBool::new(false),
            proxy_last_error: Mutex::new(String::new()),
            reported_job: Mutex::new(None),
        };
        state.save(); // persist the generated token on first run
        state
    }

    pub fn save(&self) {
        let config = self.config.lock().unwrap().clone();
        if let Some(parent) = self.config_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&config) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.config_path, json) {
                    log::warn!("could not persist companion config: {e}");
                }
            }
            Err(e) => log::warn!("could not serialize companion config: {e}"),
        }
    }

    pub fn config_snapshot(&self) -> Config {
        self.config.lock().unwrap().clone()
    }

    /// True when a ClipAI server has made an authenticated request recently.
    pub fn clipai_connected(&self) -> bool {
        let last = self.last_clipai_contact.load(Ordering::Relaxed);
        last > 0 && now_ms().saturating_sub(last) < 60_000
    }

    pub fn last_clipai_contact_ms(&self) -> u64 {
        self.last_clipai_contact.load(Ordering::Relaxed)
    }

    /// Record progress of a model pull flowing through the proxy.
    pub fn note_incoming_pull(&self, model: &str, percent: f64) {
        self.incoming_pulls
            .lock()
            .unwrap()
            .insert(model.to_string(), (percent, now_ms()));
    }

    pub fn clear_incoming_pull(&self, model: &str) {
        self.incoming_pulls.lock().unwrap().remove(model);
    }

    /// Snapshot of in-flight incoming pulls, dropping entries not updated in
    /// the last 2 minutes (a client that vanished mid-pull).
    pub fn incoming_pulls_snapshot(&self) -> Vec<(String, f64)> {
        let now = now_ms();
        let mut map = self.incoming_pulls.lock().unwrap();
        map.retain(|_, (_, ts)| now.saturating_sub(*ts) < 120_000);
        map.iter().map(|(m, (p, _))| (m.clone(), *p)).collect()
    }

    pub fn begin_activity(
        &self,
        kind: &str,
        path: &str,
        job_id: &str,
        job_title: &str,
        stage: &str,
    ) -> u64 {
        // Every proxy route calls this only after the bearer check passes, so
        // it's a reliable "a ClipAI server is talking to us" signal.
        self.last_clipai_contact.store(now_ms(), Ordering::Relaxed);
        // Real work vs. a liveness probe: /api/tags, /api/ps and /v1/health are
        // polled constantly and shouldn't read as "serving jobs".
        let is_probe = kind == "health"
            || path.ends_with("/api/tags")
            || path.ends_with("/api/ps")
            || path.ends_with("/api/version");
        if !is_probe {
            self.last_job_ms.store(now_ms(), Ordering::Relaxed);
        }
        let id = self.next_activity_id.fetch_add(1, Ordering::Relaxed);
        let entry = ActivityEntry {
            id,
            kind: kind.into(),
            path: path.into(),
            job_id: job_id.into(),
            job_title: job_title.into(),
            stage: stage.into(),
            started_at_ms: now_ms(),
            finished_at_ms: None,
            status: None,
        };
        let mut feed = self.activity.lock().unwrap();
        feed.push_front(entry);
        feed.truncate(ACTIVITY_CAP);
        id
    }

    pub fn end_activity(&self, id: u64, status: u16) {
        let mut feed = self.activity.lock().unwrap();
        if let Some(entry) = feed.iter_mut().find(|e| e.id == id) {
            entry.finished_at_ms = Some(now_ms());
            entry.status = Some(status);
        }
        drop(feed);
        self.last_request_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// The most recent still-running non-health request, for /v1/health's
    /// `current_job` field and the dashboard headline.
    pub fn current_job(&self) -> Option<ActivityEntry> {
        self.activity
            .lock()
            .unwrap()
            .iter()
            .find(|e| e.finished_at_ms.is_none() && e.kind != "health")
            .cloned()
    }

    /// Record a progress heartbeat from ClipAI (POST /v1/progress). Also counts
    /// as ClipAI contact so the connection indicator stays live during long
    /// local stages.
    pub fn set_reported_progress(
        &self,
        job_id: &str,
        job_title: &str,
        stage: &str,
        progress: u64,
    ) {
        self.last_clipai_contact.store(now_ms(), Ordering::Relaxed);
        let now = now_ms();
        let mut slot = self.reported_job.lock().unwrap();
        // Preserve the original start time across heartbeats for the same job;
        // a new job_id resets it. This keeps the GUI "elapsed" sane.
        let started_ms = match slot.as_ref() {
            Some(prev) if prev.job_id == job_id => prev.started_ms,
            _ => now,
        };
        *slot = Some(ReportedJob {
            job_id: job_id.into(),
            job_title: job_title.into(),
            stage: stage.into(),
            progress: progress.min(100),
            updated_ms: now,
            started_ms,
        });
    }

    /// Clear the reported active job. When ``job_id`` is non-empty only clears
    /// it if it matches (so a stale end signal can't wipe a newer job); an empty
    /// ``job_id`` force-clears whatever is there (the GUI "Force end" button).
    pub fn clear_reported_job(&self, job_id: &str) {
        let mut slot = self.reported_job.lock().unwrap();
        let matches = match slot.as_ref() {
            Some(j) => job_id.is_empty() || j.job_id == job_id,
            None => false,
        };
        if matches {
            *slot = None;
        }
    }

    /// The reported job if a heartbeat arrived recently (< 45s) — else None so a
    /// finished/abandoned job stops driving the bar.
    pub fn reported_job_fresh(&self) -> Option<ReportedJob> {
        let r = self.reported_job.lock().unwrap().clone();
        r.filter(|j| now_ms().saturating_sub(j.updated_ms) < 45_000)
    }

    /// True when a real inference/transcription request (not a probe) has been
    /// served in the last 60s — "serving jobs" vs merely "reachable".
    pub fn serving_jobs(&self) -> bool {
        let last = self.last_job_ms.load(Ordering::Relaxed);
        last > 0 && now_ms().saturating_sub(last) < 60_000
    }

    /// Activity grouped by ClipAI job id, newest job first, so the GUI can show
    /// one log stream per video-analysis pipeline. Health probes are excluded.
    pub fn job_logs(&self) -> Vec<JobLog> {
        // A fresh progress heartbeat means ClipAI is STILL working this job even
        // if it isn't hitting us right now (local-only stage) — so it must not
        // read as "done". Captured before locking activity (separate mutex).
        let reported = self.reported_job_fresh();
        let feed = self.activity.lock().unwrap();
        let mut order: Vec<String> = Vec::new();
        let mut groups: HashMap<String, Vec<ActivityEntry>> = HashMap::new();
        // feed is newest-first (push_front), so first-seen key order = newest job first.
        for e in feed.iter() {
            if e.kind == "health" {
                continue;
            }
            let key = e.job_id.clone();
            if !groups.contains_key(&key) {
                order.push(key.clone());
            }
            groups.entry(key).or_default().push(e.clone());
        }
        order
            .into_iter()
            .map(|key| {
                let entries = groups.remove(&key).unwrap_or_default();
                let job_title = entries
                    .iter()
                    .map(|e| e.job_title.clone())
                    .find(|t| !t.is_empty())
                    .unwrap_or_default();
                // Active if a request is in-flight OR ClipAI is still reporting
                // progress for this job (covers local-only stages like frame
                // extraction / offline translation where nothing hits our proxy).
                let reported_active = reported.as_ref().map(|r| r.job_id == key).unwrap_or(false);
                let active = reported_active
                    || entries.iter().any(|e| e.finished_at_ms.is_none());
                let started_at_ms = entries.iter().map(|e| e.started_at_ms).min().unwrap_or(0);
                let mut last_activity_ms = entries
                    .iter()
                    .map(|e| e.finished_at_ms.unwrap_or(e.started_at_ms))
                    .max()
                    .unwrap_or(0);
                if reported_active {
                    if let Some(r) = &reported {
                        last_activity_ms = last_activity_ms.max(r.updated_ms);
                    }
                }
                let (reported_stage, reported_progress) = match (reported_active, reported.as_ref()) {
                    (true, Some(r)) => (r.stage.clone(), r.progress as i64),
                    _ => (String::new(), -1),
                };
                JobLog {
                    job_id: key,
                    job_title,
                    active,
                    started_at_ms,
                    last_activity_ms,
                    entries,
                    reported_stage,
                    reported_progress,
                }
            })
            .collect()
    }

    /// Effective VRAM budget in GB. Auto mode: total − (non-companion usage) −
    /// buffer, so the card is shared dynamically with games/other apps. Manual
    /// mode: the user's slider, or (total − 1 GB) when the slider is 0/auto.
    pub fn effective_budget_gb(&self) -> f32 {
        let (auto, buffer, configured) = {
            let c = self.config.lock().unwrap();
            (c.vram_auto, c.vram_buffer_gb, c.vram_budget_gb)
        };
        let total_gb = self.gpu.lock().unwrap().vram_total_mb as f32 / 1024.0;

        let budget = if auto {
            if total_gb > 0.5 {
                let baseline_gb =
                    self.gpu_baseline_used_mb.load(Ordering::Relaxed) as f32 / 1024.0;
                (total_gb - baseline_gb - buffer.max(0.0)).max(1.0)
            } else {
                4.0
            }
        } else if configured > 0.1 {
            configured
        } else if total_gb > 0.5 {
            (total_gb - 1.0).max(1.0)
        } else {
            4.0
        };
        if total_gb > 0.5 {
            budget.clamp(1.0, total_gb)
        } else {
            budget.max(1.0)
        }
    }

    /// Effective (num_parallel, max_loaded) for the current speed profile + VRAM
    /// budget — the single source of truth for Ollama concurrency and what we
    /// advertise to ClipAI.
    pub fn resolve_speed_settings(&self) -> (u32, u32) {
        let profile = self.config.lock().unwrap().speed_profile.clone();
        resolve_speed(&profile, self.effective_budget_gb())
    }
}

// ── Process helpers ─────────────────────────────────────────────────
// The GUI runs with no console (windows_subsystem = "windows"); by default
// every child process (nvidia-smi, ollama, winget, whisper) would flash its
// own console window. CREATE_NO_WINDOW suppresses that.

#[cfg(target_os = "windows")]
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// A tokio `Command` that never flashes a console window on Windows.
pub fn quiet_command(program: impl AsRef<std::ffi::OsStr>) -> tokio::process::Command {
    #[allow(unused_mut)]
    let mut cmd = tokio::process::Command::new(program);
    #[cfg(target_os = "windows")]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// A blocking `std` `Command` that never flashes a console window on Windows
/// (used for the every-5s nvidia-smi telemetry poll).
pub fn quiet_std_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    #[allow(unused_mut)]
    let mut cmd = std::process::Command::new(program);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Tie a freshly spawned child process to the Companion's lifetime so it is
/// killed when the Companion exits for ANY reason (quit, crash, or kill).
/// Windows: assign it to a KILL_ON_JOB_CLOSE job object. Other platforms:
/// no-op here — graceful shutdown (tray Quit / RunEvent::Exit) stops them.
pub fn bind_child_to_lifetime(_child: &tokio::process::Child) {
    #[cfg(target_os = "windows")]
    {
        if let Some(handle) = _child.raw_handle() {
            killjob::assign(handle as isize);
        }
    }
}

#[cfg(target_os = "windows")]
mod killjob {
    //! A process-wide Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
    //! Children assigned to it are terminated automatically when the last
    //! handle to the job closes — which happens when our process dies, for
    //! any reason. This guarantees no orphaned `ollama serve` after the
    //! Companion is killed from Task Manager.
    use std::sync::OnceLock;
    use windows::core::PCWSTR;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    // Stored as isize (HANDLE isn't Send/Sync) — created once, intentionally
    // never closed: its lifetime IS the process lifetime.
    static JOB: OnceLock<isize> = OnceLock::new();

    fn job_handle() -> isize {
        *JOB.get_or_init(|| unsafe {
            match CreateJobObjectW(None, PCWSTR::null()) {
                Ok(job) => {
                    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                    let _ = SetInformationJobObject(
                        job,
                        JobObjectExtendedLimitInformation,
                        &info as *const _ as *const core::ffi::c_void,
                        std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                    );
                    job.0 as isize
                }
                Err(e) => {
                    log::warn!("could not create kill-on-close job object: {e}");
                    0
                }
            }
        })
    }

    pub fn assign(raw_handle: isize) {
        let job = job_handle();
        if job == 0 || raw_handle == 0 {
            return;
        }
        unsafe {
            if let Err(e) = AssignProcessToJobObject(
                HANDLE(job as *mut core::ffi::c_void),
                HANDLE(raw_handle as *mut core::ffi::c_void),
            ) {
                log::warn!("could not assign child to kill-on-close job: {e}");
            }
        }
    }
}

/// Whisper model tier for a VRAM budget — mirrored in the dashboard UI.
/// macOS uses the corresponding whisper.cpp quants (resolved in sidecar.rs).
pub fn whisper_tier_for_budget(budget_gb: f32) -> (&'static str, &'static str) {
    if budget_gb >= 6.0 {
        ("large-v3-turbo", "float16")
    } else if budget_gb >= 3.0 {
        ("medium", "int8_float16")
    } else {
        ("small", "int8_float16")
    }
}

/// Rank a whisper model family: 3=large/turbo, 2=medium, 1=small/base/tiny.
fn whisper_rank(name: &str) -> u8 {
    let n = name.to_lowercase();
    if n.contains("large") || n.contains("turbo") {
        3
    } else if n.contains("medium") {
        2
    } else {
        1
    }
}

/// Whisper tier honoring the model ClipAI SELECTED (synced per request via the
/// `X-ClipAI-Whisper-Model` header) while never exceeding what the VRAM budget
/// can fit on this GPU. Keeps the Companion loading the same model family the
/// ClipAI server picked — so transcription quality matches — but caps it so a
/// big model can never OOM a small budget. Empty `requested` ⇒ budget default.
pub fn whisper_tier_for_request(requested: &str, budget_gb: f32) -> (&'static str, &'static str) {
    let (cap_model, cap_compute) = whisper_tier_for_budget(budget_gb);
    if requested.trim().is_empty() {
        return (cap_model, cap_compute);
    }
    let rank = whisper_rank(requested).min(whisper_rank(cap_model));
    match rank {
        3 => ("large-v3-turbo", "float16"),
        2 => ("medium", "int8_float16"),
        _ => ("small", "int8_float16"),
    }
}
