//! Pairing with a ClipAI server: POST /api/settings/companion-register.
//! On success ClipAI adds this Companion's proxy as its PRIMARY Ollama
//! host and points remote Whisper at it. No API key is required (ClipAI's
//! settings surface is LAN-trust); one is forwarded only if configured,
//! for older servers. Manual copy-paste configuration always remains
//! possible without this.

use crate::state::AppState;
use serde::Serialize;
use std::net::UdpSocket;
use std::sync::Arc;

/// Best-effort LAN IP: the address a UDP socket would source from.
/// No packets are sent.
pub fn detect_lan_ip() -> Option<String> {
    detect_lan_ip_toward(None)
}

/// LAN IP as seen on the route TOWARD a specific host (the ClipAI server).
///
/// On a machine with several 192.168.x adapters (Ethernet + Wi-Fi both up,
/// Hyper-V/WSL switches, a VPN), the generic probe below lets Windows pick
/// whichever interface it likes for a broadcast-ish target — and that pick
/// can flip between boots, so the advertised endpoint "changes" with no
/// config change at all. Anchoring the probe at the ClipAI server's own
/// address selects the source IP of the actual route to that server, i.e.
/// the one address the server is guaranteed to be able to reach back.
pub fn detect_lan_ip_toward(clipai_host: Option<&str>) -> Option<String> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    if let Some(h) = clipai_host.filter(|h| !h.is_empty()) {
        // Port is irrelevant for route selection; 80 avoids resolving the
        // proxy port. A hostname here resolves via DNS but sends nothing.
        if socket.connect((h, 80)).is_ok() {
            if let Ok(addr) = socket.local_addr() {
                let ip = addr.ip();
                if !ip.is_loopback() && !ip.is_unspecified() {
                    return Some(ip.to_string());
                }
            }
        }
    }
    socket.connect("192.168.255.255:80").ok().or_else(|| socket.connect("8.8.8.8:80").ok())?;
    Some(socket.local_addr().ok()?.ip().to_string())
}

/// LAN IP anchored at whatever ClipAI server this Companion knows about:
/// the paired URL first, else the server address learned from inbound
/// proxy traffic (covers manual, ClipAI-side pairing), else the generic
/// default-route probe.
pub fn detect_lan_ip_for(state: &AppState) -> Option<String> {
    let anchor = {
        let cfg = state.config.lock().unwrap();
        cfg.paired_clipai_url.clone()
    };
    let anchor = if anchor.is_empty() { state.seen_clipai_url() } else { anchor };
    detect_lan_ip_toward(url_host(&anchor).as_deref())
}

/// Extract the bare host out of a URL-ish string ("http://192.168.8.14:3000/x"
/// → "192.168.8.14"). No external deps; LAN URLs only (no IPv6 brackets).
pub fn url_host(url: &str) -> Option<String> {
    let u = url.trim();
    if u.is_empty() {
        return None;
    }
    let u = u
        .strip_prefix("http://")
        .or_else(|| u.strip_prefix("https://"))
        .unwrap_or(u);
    let host_port = u.split(['/', '?', '#']).next().unwrap_or("");
    let host = match host_port.rsplit_once(':') {
        Some((h, p)) if !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => h,
        _ => host_port,
    };
    let host = host.trim();
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
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
    // Anchor detection at the server we're pairing WITH so multi-adapter
    // machines always advertise the address that server can route back to.
    let lan_ip = detect_lan_ip_toward(url_host(&clipai_url).as_deref())
        .ok_or_else(|| "could not detect this machine's LAN IP".to_string())?;
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
    // No API key is needed: ClipAI's settings surface is LAN-trust and the
    // register endpoint accepts unauthenticated pairing. A key is still
    // FORWARDED when the user has one configured, so pairing keeps working
    // against older ClipAI servers that still demand the bearer.
    let mut rb = reqwest::Client::new()
        .post(format!("{clipai_url}/api/settings/companion-register"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(15));
    if !api_key.trim().is_empty() {
        rb = rb.bearer_auth(api_key.trim());
    }
    let resp = rb
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
        // Keep the key so the re-announce loop can rebind this Companion
        // after a DHCP change without the user re-pairing by hand.
        cfg.paired_clipai_api_key = api_key.trim().to_string();
    }
    state.save();
    Ok(payload)
}

/// Re-announce this Companion to its paired ClipAI server.
///
/// ClipAI matches the registration by this Companion's persistent `token`,
/// so calling this from a NEW address rebinds the existing host entry in
/// place — Whisper routing, the Ollama registry, and the GPU metadata all
/// follow the move with nothing to re-add. A no-op (Ok(false)) only when the
/// Companion has never been paired; a keyless pairing re-announces fine
/// (the register endpoint no longer requires the bearer).
pub async fn reannounce(state: &Arc<AppState>) -> Result<bool, String> {
    let (url, key) = {
        let cfg = state.config.lock().unwrap();
        (cfg.paired_clipai_url.clone(), cfg.paired_clipai_api_key.clone())
    };
    if url.is_empty() {
        return Ok(false);
    }
    pair(state, &url, &key).await.map(|_| true)
}

#[cfg(test)]
mod tests {
    use super::url_host;

    #[test]
    fn url_host_strips_scheme_port_and_path() {
        assert_eq!(url_host("http://192.168.8.14:3000/api"), Some("192.168.8.14".into()));
        assert_eq!(url_host("https://tower.local/"), Some("tower.local".into()));
        assert_eq!(url_host("192.168.8.14:8043"), Some("192.168.8.14".into()));
        assert_eq!(url_host("http://192.168.8.14"), Some("192.168.8.14".into()));
        assert_eq!(url_host("tower"), Some("tower".into()));
        assert_eq!(url_host("http://192.168.8.14:3000?x=1"), Some("192.168.8.14".into()));
    }

    #[test]
    fn url_host_rejects_empty() {
        assert_eq!(url_host(""), None);
        assert_eq!(url_host("   "), None);
        assert_eq!(url_host("http://"), None);
    }
}
