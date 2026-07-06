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
}

export interface CompanionStatus {
  config: {
    token: string;
    port: number;
    vram_budget_gb: number;
    vram_auto: boolean;
    vram_buffer_gb: number;
    ollama_keep_alive: string;
    sidecar_idle_min: number;
    paused: boolean;
    paired_clipai_url: string;
    name: string;
    setup_complete: boolean;
  };
  gpu: GpuSnapshot;
  effective_budget_gb: number;
  whisper_tier: { model: string; compute: string };
  ollama: {
    installed: boolean;
    version: string;
    running: boolean;
    managed: boolean;
    models: string[];
  };
  sidecar_available: boolean;
  sidecar_running: boolean;
  busy: boolean;
  current_job: ActivityEntry | null;
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
export const pairClipai = (clipaiUrl: string, apiKey: string) =>
  invoke('pair_clipai', { clipaiUrl, apiKey });
