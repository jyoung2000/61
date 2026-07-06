//! The Companion's single LAN surface: an authenticated reverse proxy on
//! 0.0.0.0:<port> (default 11500).
//!
//!   /ollama/*                 → the localhost-only Ollama daemon
//!   /v1/audio/transcriptions  → the whisper sidecar (lazily started,
//!                               one job at a time; 503 + Retry-After
//!                               when saturated)
//!   /v1/health                → GPU + backend status JSON
//!
//! EVERY route requires `Authorization: Bearer <token>`. The token is
//! generated on first run, shown in the GUI, and never logged. In-flight
//! requests are recorded (via the X-ClipAI-* headers ClipAI attaches) to
//! power the live activity feed.

use crate::state::{AppState, OLLAMA_LOCAL};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use futures_util::{StreamExt, TryStreamExt};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone)]
pub struct ProxyCtx {
    pub state: Arc<AppState>,
    pub client: reqwest::Client,
    pub resource_dir: PathBuf,
    pub data_dir: PathBuf,
}

fn header_str(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}

fn authorized(ctx: &ProxyCtx, headers: &HeaderMap) -> bool {
    let expected = ctx.state.config.lock().unwrap().token.clone();
    if expected.is_empty() {
        return false; // never run open
    }
    let presented = header_str(headers, "authorization");
    let presented = presented.strip_prefix("Bearer ").unwrap_or("").trim();
    // Constant-time-ish comparison; the token is high-entropy anyway.
    presented.len() == expected.len()
        && presented
            .bytes()
            .zip(expected.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [("WWW-Authenticate", "Bearer")],
        "missing or invalid bearer token",
    )
        .into_response()
}

fn paused() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [("Retry-After", "60")],
        "GPU sharing is paused on this Companion",
    )
        .into_response()
}

/// Copy a reqwest upstream response back to the axum client, streaming.
fn relay(upstream: reqwest::Response) -> Response {
    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        // hop-by-hop headers are managed by the servers themselves
        let lower = name.as_str().to_ascii_lowercase();
        if ["connection", "transfer-encoding", "keep-alive"].contains(&lower.as_str()) {
            continue;
        }
        builder = builder.header(name, value);
    }
    let stream = upstream
        .bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

fn bad_gateway(err: impl std::fmt::Display) -> Response {
    (StatusCode::BAD_GATEWAY, format!("upstream error: {err}")).into_response()
}

/// /ollama/* → strip the prefix, forward to the local daemon.
async fn ollama_proxy(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    if ctx.state.config.lock().unwrap().paused {
        return paused();
    }
    let path_q = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let stripped = path_q.strip_prefix("/ollama").unwrap_or(path_q).to_string();
    let stripped = if stripped.is_empty() { "/".to_string() } else { stripped };
    let target = format!("http://{OLLAMA_LOCAL}{stripped}");
    let is_pull = stripped.starts_with("/api/pull");

    let activity = ctx.state.begin_activity(
        "ollama",
        &stripped,
        &header_str(req.headers(), "x-clipai-job-id"),
        &header_str(req.headers(), "x-clipai-job-title"),
        &header_str(req.headers(), "x-clipai-stage"),
    );

    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);
    // Forwardable headers (drop hop-by-hop + auth).
    let mut fwd: Vec<(String, String)> = Vec::new();
    for (name, value) in req.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        if ["host", "authorization", "connection", "content-length"].contains(&lower.as_str()) {
            continue;
        }
        if let Ok(v) = value.to_str() {
            fwd.push((name.as_str().to_string(), v.to_string()));
        }
    }

    // A model pull ClipAI pushes through here: record its progress so the
    // Companion GUI can SHOW the download, not just silently proxy it.
    if is_pull {
        return proxy_pull(ctx, req, target, method, fwd, activity).await;
    }

    let mut upstream = ctx.client.request(method, &target);
    for (n, v) in fwd {
        upstream = upstream.header(n, v);
    }
    let body_stream = req
        .into_body()
        .into_data_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
    upstream = upstream.body(reqwest::Body::wrap_stream(body_stream));

    match upstream.send().await {
        Ok(resp) => {
            ctx.state.end_activity(activity, resp.status().as_u16());
            relay(resp)
        }
        Err(e) => {
            ctx.state.end_activity(activity, 502);
            bad_gateway(e)
        }
    }
}

/// Forward an /api/pull, teeing the NDJSON progress into AppState so the
/// Companion GUI shows the download ClipAI initiated.
async fn proxy_pull(
    ctx: ProxyCtx,
    req: Request<Body>,
    target: String,
    method: reqwest::Method,
    fwd: Vec<(String, String)>,
    activity: u64,
) -> Response {
    // The pull body is tiny ({"name":"…"}) — buffer it to read the model.
    let body_bytes = match axum::body::to_bytes(req.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            ctx.state.end_activity(activity, 400);
            return bad_gateway(e);
        }
    };
    let model = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| {
            v.get("name")
                .or_else(|| v.get("model"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_default();
    if !model.is_empty() {
        ctx.state.note_incoming_pull(&model, 0.0);
    }

    let mut upstream = ctx.client.request(method, &target);
    for (n, v) in fwd {
        upstream = upstream.header(n, v);
    }
    upstream = upstream.body(body_bytes.to_vec());

    match upstream.send().await {
        Ok(resp) => {
            let status = resp.status();
            ctx.state.end_activity(activity, status.as_u16());
            if !status.is_success() {
                if !model.is_empty() {
                    ctx.state.clear_incoming_pull(&model);
                }
                return relay(resp);
            }
            let out_status =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
            let mut builder = Response::builder().status(out_status);
            for (name, value) in resp.headers() {
                let lower = name.as_str().to_ascii_lowercase();
                if ["connection", "transfer-encoding", "keep-alive"].contains(&lower.as_str()) {
                    continue;
                }
                builder = builder.header(name, value);
            }
            let state = ctx.state.clone();
            let model_c = model.clone();
            let mut buf: Vec<u8> = Vec::new();
            let stream = resp.bytes_stream().map(move |chunk| {
                if let (Ok(bytes), false) = (&chunk, model_c.is_empty()) {
                    buf.extend_from_slice(bytes);
                    while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = buf.drain(..=nl).collect();
                        let line = &line[..line.len().saturating_sub(1)];
                        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) {
                            let st = v.get("status").and_then(|s| s.as_str()).unwrap_or("");
                            let total = v.get("total").and_then(|t| t.as_u64()).unwrap_or(0);
                            let completed = v.get("completed").and_then(|c| c.as_u64()).unwrap_or(0);
                            if total > 0 {
                                state.note_incoming_pull(
                                    &model_c,
                                    completed as f64 / total as f64 * 100.0,
                                );
                            }
                            if st == "success" || v.get("error").is_some() {
                                state.clear_incoming_pull(&model_c);
                            }
                        }
                    }
                }
                chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            });
            builder
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
        }
        Err(e) => {
            if !model.is_empty() {
                ctx.state.clear_incoming_pull(&model);
            }
            ctx.state.end_activity(activity, 502);
            bad_gateway(e)
        }
    }
}

/// /v1/audio/transcriptions → whisper sidecar, one job at a time.
async fn whisper_proxy(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    if ctx.state.config.lock().unwrap().paused {
        return paused();
    }
    // Serialize GPU-heavy transcription: a second concurrent job gets an
    // honest 503 + Retry-After instead of an OOM or a silent queue.
    let Ok(permit) = ctx.state.whisper_slot.try_acquire() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [("Retry-After", "20")],
            "a transcription is already running on this GPU",
        )
            .into_response();
    };
    ctx.state.whisper_busy.store(true, Ordering::Relaxed);

    let activity = ctx.state.begin_activity(
        "whisper",
        "/v1/audio/transcriptions",
        &header_str(req.headers(), "x-clipai-job-id"),
        &header_str(req.headers(), "x-clipai-job-title"),
        &header_str(req.headers(), "x-clipai-stage"),
    );

    let result = async {
        crate::sidecar::ensure_running(
            &ctx.state,
            ctx.resource_dir.clone(),
            ctx.data_dir.clone(),
        )
        .await
        .map_err(|e| format!("sidecar unavailable: {e}"))?;

        let mut upstream = ctx.client.post(format!(
            "{}/v1/audio/transcriptions",
            crate::sidecar::sidecar_url()
        ));
        for (name, value) in req.headers() {
            let lower = name.as_str().to_ascii_lowercase();
            if ["host", "authorization", "connection", "content-length"].contains(&lower.as_str())
            {
                continue;
            }
            if let Ok(v) = value.to_str() {
                upstream = upstream.header(name.as_str(), v);
            }
        }
        let body_stream = req
            .into_body()
            .into_data_stream()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
        upstream
            .body(reqwest::Body::wrap_stream(body_stream))
            .send()
            .await
            .map_err(|e| format!("sidecar request failed: {e}"))
    }
    .await;

    ctx.state.whisper_busy.store(false, Ordering::Relaxed);
    drop(permit);

    match result {
        Ok(resp) => {
            ctx.state.end_activity(activity, resp.status().as_u16());
            relay(resp)
        }
        Err(e) => {
            ctx.state.end_activity(activity, 502);
            (StatusCode::BAD_GATEWAY, e).into_response()
        }
    }
}

/// /v1/health → status JSON for ClipAI's probes + the pairing handshake.
async fn health(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let activity = ctx.state.begin_activity("health", "/v1/health", "", "", "");
    let gpu = ctx.state.gpu.lock().unwrap().clone();
    let config = ctx.state.config_snapshot();
    let (whisper_model, _) = crate::state::whisper_tier_for_budget(ctx.state.effective_budget_gb());
    let busy = ctx.state.whisper_busy.load(Ordering::Relaxed);
    let current = ctx.state.current_job().map(|e| {
        serde_json::json!({
            "job_id": e.job_id,
            "title": e.job_title,
            "stage": e.stage,
            "kind": e.kind,
            "started_at_ms": e.started_at_ms,
        })
    });
    let ollama_up = crate::ollama::daemon_running().await;
    let whisper_shipped = ctx.state.sidecar_available.load(Ordering::Relaxed);
    let body = serde_json::json!({
        "service": "clipai-gpu-companion",
        "version": env!("CARGO_PKG_VERSION"),
        "gpu_name": gpu.gpu_name,
        "vram_total_mb": gpu.vram_total_mb,
        "vram_free_mb": gpu.vram_free_mb,
        "unified_memory": gpu.unified_memory,
        "vram_budget_gb": ctx.state.effective_budget_gb(),
        "backends": {
            "ollama": ollama_up,
            "whisper": whisper_shipped,
            "whisper_model": whisper_model,
        },
        "busy": busy,
        "paused": config.paused,
        "current_job": current,
    });
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

pub async fn serve(ctx: ProxyCtx) {
    let port = ctx.state.config.lock().unwrap().port;
    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/audio/transcriptions", post(whisper_proxy))
        .route("/ollama", any(ollama_proxy))
        .route("/ollama/", any(ollama_proxy))
        .route("/ollama/*path", any(ollama_proxy))
        .with_state(ctx);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    log::info!("companion proxy listening on {addr}");
    match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => {
            if let Err(e) = axum::serve(listener, app).await {
                log::error!("proxy server exited: {e}");
            }
        }
        Err(e) => log::error!("could not bind {addr}: {e} (port in use?)"),
    }
}
