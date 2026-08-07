import { invoke } from '@tauri-apps/api/core';

export interface GpuSnapshot {
  gpu_name: string;
  vram_total_mb: number;
  vram_free_mb: number;
  unified_memory: boolean;
  available: boolean;
}

export interface ActivityEntry {
  id: number;
  kind: string;
  path: string;
  job_id: string;
  job_title: string;
  stage: string;
  started_at_ms: number;
  finished_at_ms: number | null;
  status: number | null;
}

export interface JobLog {
  job_id: string;
  job_title: string;
  active: boolean;
  started_at_ms: number;
  last_activity_ms: number;
  entries: ActivityEntry[];
  /** Stage ClipAI last reported (heartbeat) — live even during local-only stages. */
  reported_stage: string;
  /** Overall job progress 0-100 from the heartbeat, or -1 when unknown. */
  reported_progress: number;
}

export interface CompanionStatus {
  /** This app's own version (CARGO_PKG_VERSION). */
  app_version: string;
  config: {
    token: string;
    port: number;
    vram_budget_gb: number;
    vram_auto: boolean;
    vram_buffer_gb: number;
    ollama_keep_alive: string;
    sidecar_idle_min: number;
    /** Minutes of real-work idleness before the WHOLE GPU is auto-freed
     *  (whisper sidecar stopped + Ollama models evicted). 0 = off. */
    gpu_idle_free_min: number;
    /** SECONDS of real-work idleness before the whole-GPU auto-free fires —
     *  the fast, primary knob (default 300). When > 0 it wins over
     *  gpu_idle_free_min; both 0 disables the auto-free. */
    gpu_idle_free_sec: number;
    paused: boolean;
    paired_clipai_url: string;
    name: string;
    setup_complete: boolean;
    speed_profile: 'auto' | 'eco' | 'balanced' | 'turbo';
    whisper_quality: 'auto' | 'fast' | 'balanced' | 'max';
    /** Folders shared with ClipAI (it can browse + pull files from these). */
    shared_paths: string[];
    /** Share the ENTIRE computer (all drives) instead of just shared_paths. */
    share_all: boolean;
  };
  /** Effective speed: resolved concurrency for the chosen profile + VRAM. */
  speed: { profile: string; num_parallel: number; max_loaded_models: number };
  /** Effective transcription quality (beam search + model) for the profile + VRAM. */
  whisper_quality_effective: { profile: string; model: string; beam_size: number; beam_search: boolean };
  gpu: GpuSnapshot;
  /** VRAM (MB) ClipAI is actively holding (Ollama models + Whisper while busy). */
  clipai_vram_mb: number;
  effective_budget_gb: number;
  whisper_tier: { model: string; compute: string };
  ollama: {
    installed: boolean;
    version: string;
    running: boolean;
    managed: boolean;
    models: string[];
  };
  /** Models actively resident in VRAM right now (name + VRAM MB), via /api/ps. */
  resident_models: { name: string; vram_mb: number; expires_at: string }[];
  sidecar_available: boolean;
  sidecar_running: boolean;
  /** Which whisper build is installed: "gpu" (CUDA), "cpu", "bundled", "none". */
  whisper_build: 'gpu' | 'cpu' | 'bundled' | 'none';
  busy: boolean;
  current_job: ActivityEntry | null;
  job_progress: number | null;
  proxy_bound: boolean;
  proxy_last_error: string;
  clipai_connected: boolean;
  clipai_serving: boolean;
  clipai_last_contact_ms: number;
  activity: ActivityEntry[];
  job_logs: JobLog[];
  incoming_pulls: { model: string; percent: number }[];
  lan_ip: string | null;
  recommended_models: { model: string; why: string }[];
}

export const getStatus = () => invoke<CompanionStatus>('get_status');

export const setConfig = (patch: Partial<CompanionStatus['config']>) =>
  invoke('set_config', { patch });

export const regenerateToken = () => invoke<string>('regenerate_token');
export const installOllama = () => invoke<string>('install_ollama');
export const startOllama = () => invoke<boolean>('start_ollama');
export const pullModel = (model: string) => invoke('pull_model', { model });
export interface InstalledModel { name: string; size: number; family: string; parameter_size: string; }
export const listModels = () => invoke<InstalledModel[]>('list_models');
export const deleteModel = (model: string) => invoke('delete_model', { model });
export const downloadWhisper = () => invoke<string>('download_whisper');
export const refreshSidecar = () => invoke<boolean>('refresh_sidecar');
/** Write a full diagnostics report to Downloads and reveal it; returns the path. */
export const exportLogs = () => invoke<string>('export_logs');
export interface ClipaiTest {
  contacted: boolean;
  connected: boolean;
  serving: boolean;
  secs_ago: number;
  paired_url: string;
  probe: { attempted: boolean; ok?: boolean; status?: number; error?: string; url?: string };
  port: number;
  lan_ip: string | null;
}
/** Test whether a ClipAI container is properly connected to this companion. */
export const testClipai = () => invoke<ClipaiTest>('test_clipai');
/** Unload all resident Ollama models to free GPU VRAM now. */
export const freeVram = () =>
  invoke<{ unloaded: number; whisper_stopped: boolean }>('free_vram');
export const endActiveJob = () =>
  invoke<{ unloaded: number; whisper_stopped: boolean; ended_job: string | null }>('end_active_job');
// No API key: ClipAI's register endpoint is LAN-trust (same model as the
// rest of its settings API). The empty string keeps the Rust command's
// signature, which still forwards a key if one is ever configured.
export const pairClipai = (clipaiUrl: string, apiKey: string = '') =>
  invoke('pair_clipai', { clipaiUrl, apiKey });

/** Self-update: what installer the paired ClipAI container is serving. */
export interface AppUpdateCheck {
  current: string;
  latest: string;
  /** Build id (git SHA) of the running app and of the served installer. Lets a
   *  from-source rebuild at the same version still be offered as an update. */
  current_build?: string;
  latest_build?: string;
  update_available: boolean;
  installer_available: boolean;
  platform: string;
  filename: string;
  size: number;
  source: string;
}
export const checkAppUpdate = () => invoke<AppUpdateCheck>('check_app_update');
/** Download the installer from ClipAI, launch it, and exit this app. */
export const installAppUpdate = () => invoke<string>('install_app_update');
