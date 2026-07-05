//! Shared Companion state: persisted config (token, port, VRAM budget),
//! the live activity feed the dashboard renders, and handles to the
//! managed Ollama / whisper-sidecar child processes.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROXY_PORT_DEFAULT: u16 = 11500;
pub const OLLAMA_LOCAL: &str = "127.0.0.1:11434";
pub const WHISPER_SIDECAR_PORT: u16 = 11510;
/// Recent-request history kept for the dashboard.
const ACTIVITY_CAP: usize = 50;

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
        }
    }
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
            whisper_slot: tokio::sync::Semaphore::new(1),
            last_request_ms: AtomicU64::new(now_ms()),
            whisper_busy: AtomicBool::new(false),
            sidecar_available: AtomicBool::new(false),
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

    pub fn begin_activity(
        &self,
        kind: &str,
        path: &str,
        job_id: &str,
        job_title: &str,
        stage: &str,
    ) -> u64 {
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

    /// Effective VRAM budget in GB: the user's slider, or (total − 1 GB)
    /// when set to auto/0, clamped to sane bounds.
    pub fn effective_budget_gb(&self) -> f32 {
        let configured = self.config.lock().unwrap().vram_budget_gb;
        let total_gb = self.gpu.lock().unwrap().vram_total_mb as f32 / 1024.0;
        let budget = if configured > 0.1 {
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
