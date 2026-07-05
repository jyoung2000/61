//! Pairing with a ClipAI server: POST /api/settings/companion-register
//! with the user's ClipAI API key. On success ClipAI adds this
//! Companion's proxy as its PRIMARY Ollama host and points remote
//! Whisper at it. Manual copy-paste configuration always remains
//! possible without this.

use crate::state::AppState;
use serde::Serialize;
use std::net::UdpSocket;
use std::sync::Arc;

/// Best-effort LAN IP: the address a UDP socket would source from.
/// No packets are sent.
pub fn detect_lan_ip() -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.168.255.255:80").ok().or_else(|| socket.connect("8.8.8.8:80").ok())?;
    Some(socket.local_addr().ok()?.ip().to_string())
}

#[derive(Serialize)]
struct RegisterBody {
    name: String,
    url: String,
    token: String,
    gpu_name: String,
    vram_total_mb: u64,
    version: String,
    register_whisper: bool,
}

pub async fn pair(
    state: &Arc<AppState>,
    clipai_url: &str,
    api_key: &str,
) -> Result<serde_json::Value, String> {
    let clipai_url = clipai_url.trim().trim_end_matches('/').to_string();
    if clipai_url.is_empty() {
        return Err("ClipAI URL is required".into());
    }
    if api_key.trim().is_empty() {
        return Err("ClipAI API key is required".into());
    }
    let lan_ip =
        detect_lan_ip().ok_or_else(|| "could not detect this machine's LAN IP".to_string())?;
    let (name, token, port) = {
        let cfg = state.config.lock().unwrap();
        (cfg.name.clone(), cfg.token.clone(), cfg.port)
    };
    let gpu = state.gpu.lock().unwrap().clone();
    let body = RegisterBody {
        name,
        url: format!("http://{lan_ip}:{port}"),
        token,
        gpu_name: gpu.gpu_name,
        vram_total_mb: gpu.vram_total_mb,
        version: env!("CARGO_PKG_VERSION").into(),
        // Only point ClipAI's remote Whisper here when this build actually
        // shipped a sidecar — otherwise transcription stays on the server.
        register_whisper: state
            .sidecar_available
            .load(std::sync::atomic::Ordering::Relaxed),
    };
    let resp = reqwest::Client::new()
        .post(format!("{clipai_url}/api/settings/companion-register"))
        .bearer_auth(api_key.trim())
        .json(&body)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .map_err(|e| format!("could not reach ClipAI at {clipai_url}: {e}"))?;
    let status = resp.status();
    let payload: serde_json::Value = resp
        .json()
        .await
        .unwrap_or_else(|_| serde_json::json!({"detail": "non-JSON response"}));
    if !status.is_success() {
        let detail = payload["detail"].as_str().unwrap_or("pairing rejected");
        return Err(format!("ClipAI answered HTTP {status}: {detail}"));
    }
    {
        let mut cfg = state.config.lock().unwrap();
        cfg.paired_clipai_url = clipai_url;
    }
    state.save();
    Ok(payload)
}
