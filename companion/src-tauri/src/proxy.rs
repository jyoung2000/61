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
use axum::extract::{Query, State};
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router};
use futures_util::{StreamExt, TryStreamExt};
use std::collections::HashMap;
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

/// `own < expected`, comparing tolerant semver triples ("0.2.0", "v0.2").
/// Unparseable/empty `expected` never reports an update (a plain probe with
/// no handshake header must not flag anything). Also used by the GUI's
/// self-update check (lib.rs) against the paired ClipAI's installer manifest.
pub(crate) fn version_lt(own: &str, expected: &str) -> bool {
    fn triple(s: &str) -> Option<(u32, u32, u32)> {
        let digits: Vec<u32> = s
            .split(|c: char| !c.is_ascii_digit())
            .filter(|p| !p.is_empty())
            .filter_map(|p| p.parse().ok())
            .collect();
        match digits.as_slice() {
            [] => None,
            [a] => Some((*a, 0, 0)),
            [a, b] => Some((*a, *b, 0)),
            [a, b, c, ..] => Some((*a, *b, *c)),
        }
    }
    match (triple(own), triple(expected)) {
        (Some(o), Some(e)) => o < e,
        _ => false,
    }
}

/// Decide whether the paired ClipAI is serving a build we should install.
///
/// The Update button used to gate purely on semver (`version_lt`), so a
/// from-source rebuild that reused the SAME version (0.2.4 → 0.2.4) reported
/// "up to date" forever and the freshly-built binary — editorial fix and all —
/// never reached the app. Now the decision is:
///   * a strictly-NEWER semver always updates (normal release path);
///   * a LOWER semver never updates (never downgrade);
///   * at an EQUAL or incomparable semver, a DIFFERENT non-empty build id
///     updates too — a from-source rebuild carries a fresh git SHA, so the
///     same-version rebuild is still installable.
/// Converges: once installed, the running id equals the served id → no re-offer.
pub(crate) fn should_update(
    current_ver: &str,
    current_id: &str,
    latest_ver: &str,
    latest_id: &str,
) -> bool {
    if version_lt(current_ver, latest_ver) {
        return true; // strictly newer release
    }
    if version_lt(latest_ver, current_ver) {
        return false; // older release — never downgrade
    }
    // Equal (or incomparable) semver → fall back to build identity. An empty or
    // "unknown"/"source" served id can't prove a newer build, so it never flags.
    let latest_id = latest_id.trim();
    let current_id = current_id.trim();
    !latest_id.is_empty()
        && !latest_id.eq_ignore_ascii_case("unknown")
        && !latest_id.eq_ignore_ascii_case("source")
        && !latest_id.eq_ignore_ascii_case(current_id)
}

/// Record ClipAI's overall job progress (X-ClipAI-Progress, 0-100) so the GUI
/// can show a live bar for the work it's serving.
fn note_progress(state: &AppState, headers: &HeaderMap) {
    if let Some(p) = headers
        .get("x-clipai-progress")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
    {
        state.job_progress.store(p.min(100), Ordering::Relaxed);
    }
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

/// Map a reqwest error from the LOCAL Ollama/Whisper to a client response. A
/// connect/timeout usually means the upstream is mid-start or reloading, so we
/// answer 503 + Retry-After (RETRYABLE) instead of 502 — that keeps ClipAI
/// retrying THIS GPU rather than failing the work over to its weak local card.
/// Returns (status_code, response) so the activity log records the real code.
fn upstream_error(err: reqwest::Error) -> (u16, Response) {
    if err.is_connect() || err.is_timeout() {
        return (
            503,
            (
                StatusCode::SERVICE_UNAVAILABLE,
                [("Retry-After", "3")],
                format!("upstream starting/unavailable: {err}"),
            )
                .into_response(),
        );
    }
    (502, bad_gateway(err))
}

/// Marks an activity finished exactly once when dropped — used to keep a
/// streamed response "current" until its body actually completes (or the
/// client aborts), instead of finishing it the moment the headers arrive.
struct EndActivityGuard {
    state: Arc<AppState>,
    activity: u64,
    code: u16,
}

impl Drop for EndActivityGuard {
    fn drop(&mut self) {
        self.state.end_activity(self.activity, self.code);
    }
}

/// Like `relay`, but ends `activity` when the response BODY finishes streaming
/// (via the guard's Drop), so a long streamed generation reads as an active job
/// the whole time it's running.
fn relay_tracked(upstream: reqwest::Response, state: Arc<AppState>, activity: u64) -> Response {
    let code = upstream.status().as_u16();
    let status = StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        if ["connection", "transfer-encoding", "keep-alive"].contains(&lower.as_str()) {
            continue;
        }
        builder = builder.header(name, value);
    }
    let guard = EndActivityGuard { state, activity, code };
    let stream = upstream.bytes_stream().map(move |chunk| {
        // Hold the guard for the life of the stream; when the body completes or
        // the client disconnects, the stream drops and end_activity fires once.
        let _hold = &guard;
        chunk.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    });
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// Receiver-side validation of an ``X-ClipAI-Origin`` value before it can
/// become the self-update download base. The header is attacker-influenceable
/// (any token-holder can send it), so beyond the scheme we cap the length and
/// reject anything that isn't a clean bare origin: no whitespace/control chars,
/// no embedded credentials (``user:pass@host`` userinfo), no path/query/fragment
/// — the value is only ever used as ``{base}/api/downloads/...`` so a trailing
/// path or userinfo has no legitimate purpose and only serves to obscure the
/// real host that a downloaded installer would be fetched (and run) from.
fn plausible_clipai_origin(s: &str) -> bool {
    if s.len() > 255 || !(s.starts_with("http://") || s.starts_with("https://")) {
        return false;
    }
    let rest = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or("");
    // A single trailing slash is legitimate (paired_base trims it); anything
    // beyond the authority is not.
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    // Authority only — reject once we hit a path/query/fragment delimiter, and
    // reject userinfo (`@`) so the shown host can't be spoofed.
    if rest.is_empty() || rest.contains('@') {
        return false;
    }
    !rest
        .chars()
        .any(|c| c.is_whitespace() || c.is_control() || matches!(c, '/' | '?' | '#' | '\\'))
}

/// Learn the calling ClipAI's reachable base URL from an inbound request, so
/// self-update works even for MANUALLY-added pairings (no GUI pair step).
/// Priority:
///   1. ``X-ClipAI-Origin`` — the container's own base URL when it knows it
///      (most reliable; immune to Docker NAT rewriting the source IP).
///   2. peer IP (ConnectInfo) + ``X-ClipAI-Port`` — the common Unraid case
///      where the container's outbound traffic is SNAT'd to the host LAN IP.
/// No usable signal → no-op.
fn note_clipai_origin(state: &AppState, headers: &HeaderMap, addr: Option<SocketAddr>) {
    let origin = headers
        .get("x-clipai-origin")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| plausible_clipai_origin(s));
    if let Some(url) = origin {
        state.note_clipai_origin(url);
        return;
    }
    let Some(port) = headers
        .get("x-clipai-port")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u16>().ok())
    else {
        return;
    };
    if let Some(sa) = addr {
        let ip = sa.ip();
        let url = if ip.is_ipv6() {
            format!("http://[{ip}]:{port}")
        } else {
            format!("http://{ip}:{port}")
        };
        state.note_clipai_origin(url);
    }
}

async fn ollama_proxy(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let _peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0);
    note_clipai_origin(&ctx.state, req.headers(), _peer);
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
    note_progress(&ctx.state, req.headers());

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
        Ok(resp) => relay_tracked(resp, ctx.state.clone(), activity),
        Err(e) => {
            let (code, resp) = upstream_error(e);
            ctx.state.end_activity(activity, code);
            resp
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
    note_progress(&ctx.state, req.headers());

    // The whisper model ClipAI selected, synced per request so the Companion
    // loads the same family on its GPU (capped by VRAM budget in ensure_running).
    let requested_model = header_str(req.headers(), "x-clipai-whisper-model");
    let result = async {
        crate::sidecar::ensure_running(
            &ctx.state,
            ctx.resource_dir.clone(),
            ctx.data_dir.clone(),
            &requested_model,
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
            // Tell ClipAI which model ACTUALLY ran. The sidecar's own quality
            // selector caps the requested model against the VRAM budget, and
            // that budget is read at start-up — so a Companion that restarted
            // for a self-update while its GPU was still occupied silently ran
            // a smaller model for a whole job while ClipAI went on logging the
            // name it had asked for. Without this header the substitution is
            // invisible from the ClipAI side; the transcript just gets worse.
            let served = ctx
                .state
                .sidecar
                .lock()
                .await
                .as_ref()
                .map(|s| s.model_name.clone())
                .unwrap_or_default();
            let mut out = relay(resp);
            if !served.is_empty() {
                if let Ok(v) = axum::http::HeaderValue::from_str(&served) {
                    out.headers_mut().insert("x-clipai-served-whisper-model", v);
                }
            }
            out
        }
        Err(e) => {
            ctx.state.end_activity(activity, 502);
            (StatusCode::BAD_GATEWAY, e).into_response()
        }
    }
}

/// /v1/sidecar/release → shut the whisper sidecar down NOW, freeing its VRAM.
///
/// ClipAI calls this the moment a job's transcription stage completes: the
/// very next pipeline phase (translation / polish / SEO) hammers Ollama on
/// this same GPU, and on an 8 GB card a resident whisper server (~3-4 GB)
/// forces Ollama to spill layers to CPU — the LLM phase runs several times
/// slower until the 15-minute idle reaper finally fires. Releasing eagerly
/// costs nothing: the next transcription request cold-starts the sidecar
/// automatically via ensure_running (bounded model reload, self-healing).
///
/// Never kills a decode in flight — the whisper slot is acquired first, so a
/// concurrent transcription simply wins and the caller gets a 409 (it's a
/// best-effort optimization; ClipAI ignores the result).
async fn sidecar_release(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let Ok(permit) = ctx.state.whisper_slot.try_acquire() else {
        return (
            StatusCode::CONFLICT,
            "a transcription is running — not releasing",
        )
            .into_response();
    };
    crate::sidecar::shutdown(&ctx.state, "ClipAI /v1/sidecar/release").await;
    drop(permit);
    log::info!("whisper sidecar released on ClipAI's request (VRAM freed for LLM phase)");
    (StatusCode::OK, "released").into_response()
}

/// /v1/sidecar/warm → start the whisper sidecar NOW, in the background.
///
/// ClipAI calls this the moment it SELECTS remote Whisper for a job — several
/// minutes before the WAV upload (frame extraction + language ID run first).
/// The observed cold path cost the decode ~35 s of server spawn + model load
/// that this hides entirely inside those earlier stages. Non-blocking: replies
/// 202 immediately and ensure_running runs on a background task (idempotent —
/// a healthy sidecar with the same decode settings is left untouched, so
/// repeated warms are free). Honors pause; never preempts a running decode.
async fn sidecar_warm(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    if ctx.state.config.lock().unwrap().paused {
        return paused();
    }
    let requested_model = header_str(req.headers(), "x-clipai-whisper-model");
    // Record the warm as REAL work (kind "whisper", not a probe): it stamps
    // the idle clock so the reaper backstop on a long-idle machine can't
    // shut the freshly warmed sidecar down before the job's first heartbeat
    // arrives — and it shows up in the activity feed.
    let activity = ctx.state.begin_activity(
        "whisper",
        "/v1/sidecar/warm",
        &header_str(req.headers(), "x-clipai-job-id"),
        &header_str(req.headers(), "x-clipai-job-title"),
        &header_str(req.headers(), "x-clipai-stage"),
    );
    let state = ctx.state.clone();
    let rd = ctx.resource_dir.clone();
    let dd = ctx.data_dir.clone();
    tauri::async_runtime::spawn(async move {
        match crate::sidecar::ensure_running(&state, rd, dd, &requested_model).await {
            Ok(model) => {
                state.end_activity(activity, 200);
                log::info!(
                    "whisper sidecar pre-warmed (model={model}) — ready before the audio arrives"
                );
            }
            Err(e) => {
                state.end_activity(activity, 502);
                log::warn!("whisper sidecar pre-warm failed (non-fatal): {e}");
            }
        }
    });
    (StatusCode::ACCEPTED, "warming").into_response()
}


/// /v1/gpu/release → free the WHOLE GPU on request: whisper sidecar stopped
/// (skipped, never killed, when a decode holds the slot) AND every resident
/// Ollama model evicted. The full-scope sibling of /v1/sidecar/release, for
/// ClipAI's end-of-work hooks and tests. Best-effort by design — the response
/// reports what was actually freed.
async fn gpu_release(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let (inflight, _) = ctx.state.real_work_snapshot();
    if inflight {
        return (
            StatusCode::CONFLICT,
            "work is in flight — not releasing",
        )
            .into_response();
    }
    let (whisper_stopped, unloaded) =
        crate::sidecar::free_gpu(&ctx.state, "clipai /v1/gpu/release").await;
    Json(serde_json::json!({
        "whisper_stopped": whisper_stopped,
        "ollama_unloaded": unloaded,
    }))
    .into_response()
}

/// /v1/jobs/force-end → remotely end EVERYTHING the Companion is doing for
/// ClipAI: clear + suppress the active-job display, kill the whisper sidecar
/// EVEN MID-DECODE (that decode belongs to the job being ended), and evict
/// every resident Ollama model.
///
/// The remote sibling of the GUI "Force end" button (``end_active_job`` in
/// lib.rs), driven by ClipAI's Settings → GPU Companion card. Unlike
/// /v1/gpu/release this deliberately does NOT refuse while work is in
/// flight — the container cancels its jobs before calling, so any in-flight
/// request that errors back lands in an already-cancelled pipeline.
async fn jobs_force_end(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let (ended, whisper_was_running, unloaded) =
        force_end_everything(&ctx.state, "ClipAI /v1/jobs/force-end").await;
    Json(serde_json::json!({
        "ended_job": ended,
        "whisper_stopped": whisper_was_running,
        "ollama_unloaded": unloaded,
    }))
    .into_response()
}

/// End every piece of work this Companion is doing and free the GPU.
///
/// Clears + suppresses the active-job display, kills the whisper sidecar EVEN
/// MID-DECODE, and evicts every resident Ollama model. Shared by
/// ``/v1/jobs/force-end`` and by both update paths, which call it instead of
/// refusing: an update that waits for the GPU to go quiet is an update that
/// never happens on a busy machine. Returns
/// ``(ended_job, whisper_was_running, ollama_unloaded)``.
pub(crate) async fn force_end_everything(
    state: &crate::SharedState, reason: &str,
) -> (Option<String>, bool, usize) {
    // End EVERY job showing, not just the headline — ClipAI can run several
    // pipelines at once (Concurrent Analyses), each with its own card. Each
    // force_end_job("") ends the current headline and suppresses it, so the
    // next iteration sees the next job; the cap is a safety net only.
    let mut ended: Option<String> = None;
    for _ in 0..100 {
        match state.force_end_job("") {
            Some(jid) => {
                ended.get_or_insert(jid);
            }
            None => break,
        }
    }
    state.job_progress.store(0, Ordering::Relaxed);
    // Whisper first, unconditionally: free_gpu skips it while a decode holds
    // the slot, but a force-end exists precisely to stop that decode.
    let whisper_was_running = state.sidecar.lock().await.is_some();
    crate::sidecar::shutdown(state, reason).await;
    let (_stopped, unloaded) = crate::sidecar::free_gpu(state, reason).await;
    log::info!(
        "force_end_everything ({reason}): ended {} — whisper_stopped={whisper_was_running}, ollama_unloaded={unloaded}",
        ended.as_deref().unwrap_or("(no active job)")
    );
    (ended, whisper_was_running, unloaded)
}

/// /v1/progress → a lightweight job-progress heartbeat from ClipAI (same
/// X-ClipAI-* headers the AI routes carry). During local-only pipeline stages
/// (video decode/frame extraction on the SERVER GPU) no AI request reaches us,
/// so the GUI bar would freeze; this keeps it tracking the container live.
async fn progress_report(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    // A job-ended signal (X-ClipAI-Job-Ended) clears the active-job display at
    // once — ClipAI sends this when a job completes / fails / is cancelled or
    // deleted, so the GUI never shows a phantom job for the 300s staleness window.
    let ended = header_str(&headers, "x-clipai-job-ended");
    if matches!(ended.trim(), "1" | "true" | "yes") {
        ctx.state.clear_reported_job(&header_str(&headers, "x-clipai-job-id"));
        ctx.state.job_progress.store(0, Ordering::Relaxed);
        // The job is OVER — free the GPU promptly instead of letting models
        // sit out the keep-alive / sidecar windows. A short grace absorbs
        // back-to-back queued jobs: if ANY new real request (or a fresh
        // heartbeat from another job) lands during it, skip — the idle
        // reaper remains the backstop.
        let st = ctx.state.clone();
        tauri::async_runtime::spawn(async move {
            let job_mark = st.last_job_ms.load(Ordering::Relaxed);
            tokio::time::sleep(std::time::Duration::from_secs(20)).await;
            let (inflight, _) = st.real_work_snapshot();
            let moved = st.last_job_ms.load(Ordering::Relaxed) != job_mark;
            let other_job = st.reported_job_fresh().is_some();
            if !inflight && !moved && !other_job {
                crate::sidecar::free_gpu(&st, "job ended").await;
            }
        });
        return (StatusCode::OK, "ok").into_response();
    }
    let progress = headers
        .get("x-clipai-progress")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    ctx.state.set_reported_progress(
        &header_str(&headers, "x-clipai-job-id"),
        &header_str(&headers, "x-clipai-job-title"),
        &header_str(&headers, "x-clipai-stage"),
        progress,
    );
    // Mirror onto the atomic too so consumers reading job_progress stay in sync.
    ctx.state.job_progress.store(progress.min(100), Ordering::Relaxed);
    (StatusCode::OK, "ok").into_response()
}

// ── Shared-folder file access ───────────────────────────────────────────────
// ClipAI can browse the folders the user shared and pull files from them (video
// to analyze, media/fonts for the library). EVERY handler is bearer-authed AND
// jailed to the configured shared roots via resolve_shared_path (canonicalized,
// so `..` traversal and symlink escapes are refused). Read-only: no write/delete.

/// /v1/files/roots → the shared folders the user configured (name + exists).
async fn files_roots(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let (roots, share_all) = {
        let c = ctx.state.config.lock().unwrap();
        (c.shared_paths.clone(), c.share_all)
    };
    // Share-all → the storage-drive roots; otherwise the configured folders.
    let listed: Vec<String> = if share_all {
        crate::state::list_drive_roots()
    } else {
        roots
    };
    let items: Vec<_> = listed
        .iter()
        .filter(|r| !r.trim().is_empty())
        .map(|r| {
            let p = std::path::Path::new(r);
            serde_json::json!({
                "path": r,
                "name": p.file_name().and_then(|s| s.to_str()).unwrap_or(r.as_str()),
                "exists": p.is_dir(),
            })
        })
        .collect();
    (StatusCode::OK, Json(serde_json::json!({ "roots": items, "share_all": share_all }))).into_response()
}

/// Strip Windows "verbatim" path prefixes (\\?\C:\… and \\?\UNC\server\…)
/// that std::fs::canonicalize produces on Windows. resolve_shared_path
/// canonicalizes, so every listed entry inherited the prefix — which then
/// leaked into ClipAI's breadcrumbs, bookmarks, and pasted paths as line
/// noise ("Shared > ? > C:"). The plain form resolves identically.
fn strip_verbatim(s: String) -> String {
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        rest.to_string()
    } else {
        s
    }
}

#[cfg(test)]
mod strip_verbatim_tests {
    use super::strip_verbatim;
    #[test]
    fn strips_verbatim_prefixes() {
        assert_eq!(
            strip_verbatim(r"\\?\C:\Users\jalon\Videos".into()),
            r"C:\Users\jalon\Videos"
        );
        assert_eq!(strip_verbatim(r"\\?\UNC\nas\media".into()), r"\\nas\media");
        assert_eq!(strip_verbatim(r"C:\plain".into()), r"C:\plain");
        assert_eq!(strip_verbatim("/mnt/media".into()), "/mnt/media");
    }
}

/// /v1/files/list?path=… → directory entries inside a shared root (dirs first).
async fn files_list(
    State(ctx): State<ProxyCtx>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let (roots, share_all) = {
        let c = ctx.state.config.lock().unwrap();
        (c.shared_paths.clone(), c.share_all)
    };
    let requested = q.get("path").cloned().unwrap_or_default();
    let dir = match crate::state::resolve_shared_path(&roots, share_all, &requested) {
        Some(p) => p,
        None => {
            return (StatusCode::FORBIDDEN, "path is not inside a shared folder").into_response()
        }
    };
    if !dir.is_dir() {
        return (StatusCode::BAD_REQUEST, "not a directory").into_response();
    }
    let mut entries = vec![];
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            let md = e.metadata().ok();
            let is_dir = md.as_ref().map(|m| m.is_dir()).unwrap_or(false);
            let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime_ms = md
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            // Creation time — available on Windows/macOS; Err on most Linux FS
            // (falls back to 0, and the UI just sorts those together).
            let created_ms = md
                .as_ref()
                .and_then(|m| m.created().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let name = e.file_name().to_string_lossy().to_string();
            let ext = p
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            entries.push(serde_json::json!({
                "name": name,
                "path": strip_verbatim(p.to_string_lossy().into_owned()),
                "is_dir": is_dir,
                "size": size,
                "ext": ext,
                "mtime_ms": mtime_ms,
                "created_ms": created_ms,
            }));
        }
    }
    entries.sort_by(|a, b| {
        let ad = a["is_dir"].as_bool().unwrap_or(false);
        let bd = b["is_dir"].as_bool().unwrap_or(false);
        bd.cmp(&ad).then(
            a["name"]
                .as_str()
                .unwrap_or("")
                .to_lowercase()
                .cmp(&b["name"].as_str().unwrap_or("").to_lowercase()),
        )
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "path": strip_verbatim(dir.to_string_lossy().into_owned()),
            "entries": entries,
        })),
    )
        .into_response()
}

/// Parse a single-range `Range: bytes=START-END` header against a known total.
/// Supports `start-end`, `start-` (to EOF) and `-suffix` (last N). Returns the
/// inclusive (start, end) clamped to the file, or None when unsatisfiable.
fn parse_range(h: &str, total: u64) -> Option<(u64, u64)> {
    let s = h.trim().strip_prefix("bytes=")?;
    let (a, b) = s.split_once('-')?;
    let (a, b) = (a.trim(), b.trim());
    if a.is_empty() {
        let n: u64 = b.parse().ok()?;
        if n == 0 || total == 0 {
            return None;
        }
        return Some((total.saturating_sub(n), total - 1));
    }
    let start: u64 = a.parse().ok()?;
    let end: u64 = if b.is_empty() { total.saturating_sub(1) } else { b.parse().ok()? };
    if total == 0 || start > end || start >= total {
        return None;
    }
    Some((start, end.min(total - 1)))
}

/// /v1/files/read?path=… → stream a shared file's bytes in 1 MB chunks. Supports
/// HTTP Range (206) so ClipAI can pull big videos with PARALLEL segments (to
/// match multi-connection upload speed) and so browsers can seek.
async fn files_read(
    State(ctx): State<ProxyCtx>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let (roots, share_all) = {
        let c = ctx.state.config.lock().unwrap();
        (c.shared_paths.clone(), c.share_all)
    };
    let requested = q.get("path").cloned().unwrap_or_default();
    let path = match crate::state::resolve_shared_path(&roots, share_all, &requested) {
        Some(p) => p,
        None => {
            return (StatusCode::FORBIDDEN, "path is not inside a shared folder").into_response()
        }
    };
    if !path.is_file() {
        return (StatusCode::BAD_REQUEST, "not a file").into_response();
    }
    let total = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let range = headers
        .get("range")
        .and_then(|v| v.to_str().ok())
        .and_then(|h| parse_range(h, total));

    let mut file = match tokio::fs::File::open(&path).await {
        Ok(f) => f,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("open failed: {e}")).into_response()
        }
    };
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let (status, start, len) = match range {
        Some((s, e)) => (StatusCode::PARTIAL_CONTENT, s, e - s + 1),
        None => (StatusCode::OK, 0u64, total),
    };
    if start > 0 {
        if let Err(e) = file.seek(std::io::SeekFrom::Start(start)).await {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("seek failed: {e}")).into_response();
        }
    }
    // Stream EXACTLY `len` bytes in 1 MB chunks (16x fewer round-trips than the
    // old 64 KB reads → a single stream can saturate a gigabit LAN; parallel
    // Range segments push it further, matching multi-connection upload speed).
    let stream = futures_util::stream::try_unfold((file, len), |(mut f, remaining)| async move {
        if remaining == 0 {
            return Ok::<_, std::io::Error>(None);
        }
        let want = remaining.min(1024 * 1024) as usize;
        let mut buf = vec![0u8; want];
        let n = f.read(&mut buf).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.truncate(n);
        Ok(Some((axum::body::Bytes::from(buf), (f, remaining - n as u64))))
    });
    let fname = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", "application/octet-stream")
        .header("accept-ranges", "bytes")
        .header("content-length", len.to_string())
        .header("content-disposition", format!("attachment; filename=\"{fname}\""));
    if status == StatusCode::PARTIAL_CONTENT {
        builder = builder.header(
            "content-range",
            format!("bytes {}-{}/{}", start, start + len - 1, total),
        );
    }
    builder.body(Body::from_stream(stream)).unwrap()
}

/// /v1/logs → the full diagnostics report as plain text, so the ClipAI web app
/// can pull this Companion's logs remotely (export menu) without the user being
/// at the Companion PC. Bearer-authed like every other proxy route.
async fn logs_export(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let build = crate::sidecar::build_kind(&ctx.resource_dir, &ctx.data_dir);
    let report = crate::build_diagnostics_report(&ctx.state, build).await;
    (
        StatusCode::OK,
        [("content-type", "text/plain; charset=utf-8")],
        report,
    )
        .into_response()
}

/// POST /v1/update/install → REMOTE self-update, triggered from ClipAI's web
/// UI so the user never has to be at the GPU PC. Kicks off the same download →
/// sha256-verify → install cycle as the app's own Update button, but with a
/// SILENT (NSIS /S) install + automatic relaunch. Returns immediately; the
/// caller polls /v1/update/status, and must treat the short offline window
/// while the installer swaps files as "installing", not an error.
async fn update_install(
    State(ctx): State<ProxyCtx>,
    peer: Option<axum::extract::ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    // Learn/refresh the ClipAI origin so paired_base() resolves even on
    // manually-paired setups (mirrors the health handler).
    note_clipai_origin(&ctx.state, &headers, peer.map(|ci| ci.0));
    // An update is never refused. This used to answer 409 while a job was in
    // flight, on the theory that killing the app mid-transcription fails that
    // job confusingly — but the GPU PC is busy most of the time it is on, so
    // "retry when idle" meant the Companion drifted versions behind and the
    // user had to walk to the machine. Ending the job is the lesser cost, and
    // it is the honest one: the app is about to be replaced underneath any
    // work anyway, so the choice is between a clean cancel and a confusing
    // mid-flight death. ClipAI cancels its side before calling, so anything
    // that errors back lands in an already-cancelled pipeline.
    let ended_job = ctx.state.current_job().map(|j| j.job_title);
    if ended_job.is_some() {
        log::warn!(
            "remote self-update: ending in-flight job {} — an update is never \
             blocked by running work",
            ended_job.as_deref().unwrap_or("")
        );
    }
    force_end_everything(&ctx.state, "self-update (remote)").await;
    let state = ctx.state.clone();
    tauri::async_runtime::spawn(async move {
        match crate::perform_self_update(&state, true).await {
            Ok(path) => {
                log::info!(
                    "remote self-update: installer launched ({path}) — exiting \
                     so it can replace the app (relaunch is scripted)"
                );
                // Let the status endpoint serve "launching" briefly, then exit.
                // process::exit skips the Tauri exit handler, but the Windows
                // kill-on-close job object still reaps the managed Ollama /
                // whisper children (remote update is Windows-only).
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
                std::process::exit(0);
            }
            Err(e) => log::error!("remote self-update failed: {e}"),
        }
    });
    axum::Json(serde_json::json!({"started": true, "ended_job": ended_job}))
        .into_response()
}

/// GET /v1/update/status → progress of a remote-triggered (or GUI) update.
async fn update_status(
    State(ctx): State<ProxyCtx>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    axum::Json(crate::update_status_json()).into_response()
}

/// /v1/health → status JSON for ClipAI's probes + the pairing handshake.
async fn health(
    State(ctx): State<ProxyCtx>,
    // Read the peer address SOFTLY (Option), mirroring `ollama_proxy`. A hard
    // `ConnectInfo` extractor would 500 the whole route — before auth — if the
    // router were ever served without `into_make_service_with_connect_info`
    // (e.g. a future unix-socket bind or an in-process test harness). The
    // origin-learning below is best-effort, so absence just means "don't learn."
    peer: Option<axum::extract::ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    // Health probes fire constantly even when no job is running, so learning
    // the ClipAI origin here means self-update knows the server address as
    // soon as the two are talking — no job required.
    note_clipai_origin(&ctx.state, &headers, peer.map(|ci| ci.0));
    let activity = ctx.state.begin_activity("health", "/v1/health", "", "", "");
    let gpu = ctx.state.gpu.lock().unwrap().clone();
    let config = ctx.state.config_snapshot();
    let budget = ctx.state.effective_budget_gb();
    let (whisper_model, _) = crate::state::whisper_tier_for_budget(budget);
    // Effective transcription quality (beam search + model) so ClipAI can sync
    // its own whisper settings + display to what this GPU is actually doing.
    let (whisper_beam, whisper_full) =
        crate::state::whisper_quality_params(&config.whisper_quality, budget);
    let whisper_model_eff = if whisper_full && whisper_model == "large-v3-turbo" {
        "large-v3"
    } else {
        whisper_model
    };
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
    // Advertise the audio→English translate task ONLY for the CUDA build: it
    // runs whisper.cpp with --translate + word timestamps on the GPU, so ClipAI
    // can route the tier-A timing pass here instead of its slow local card. The
    // CPU build would be slower than the ClipAI server, so we don't advertise it.
    let whisper_translate_cap =
        whisper_shipped && crate::sidecar::build_kind(&ctx.resource_dir, &ctx.data_dir) == "gpu";
    // Advertise the concurrency this GPU is willing to run so ClipAI can
    // parallelize the pipeline to match the user's Speed profile.
    let (num_parallel, max_loaded) = ctx.state.resolve_speed_settings();
    // VRAM ClipAI is actively holding (Ollama models + Whisper while busy) so
    // ClipAI's diagnostics can split this GPU's usage into ClipAI vs other apps.
    let clipai_vram_mb: u64 = {
        let ollama_bytes = crate::ollama::loaded_vram_bytes().await;
        let whisper_mb: u64 = if busy {
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
    // Version handshake: ClipAI sends the Companion version its build was
    // released alongside. When that's NEWER than this install, say so in the
    // response (ClipAI's Processing Log warns the user) and log it here once
    // per process so the desktop-side logs explain degraded offloads too.
    let expected = header_str(&headers, "x-clipai-expected-companion");
    let update_available = version_lt(env!("CARGO_PKG_VERSION"), &expected);
    if update_available {
        use std::sync::atomic::AtomicBool;
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "ClipAI expects Companion v{expected} but this install is v{} — \
                 update the Companion app (newer offload routes 404 until then)",
                env!("CARGO_PKG_VERSION")
            );
        }
    }
    let body = serde_json::json!({
        "service": "clipai-gpu-companion",
        "version": env!("CARGO_PKG_VERSION"),
        // Build identity (git SHA) — lets ClipAI's "Check for updates" tell a
        // same-version from-source rebuild apart from the running install.
        "build": env!("CLIPAI_BUILD_ID"),
        "update_available": update_available,
        "expected_version": expected,
        "gpu_name": gpu.gpu_name,
        "vram_total_mb": gpu.vram_total_mb,
        "clipai_vram_mb": clipai_vram_mb,
        "vram_free_mb": gpu.vram_free_mb,
        "unified_memory": gpu.unified_memory,
        "vram_budget_gb": ctx.state.effective_budget_gb(),
        // Raw VRAM controls so ClipAI can render + drive the auto-allocate toggle
        // and the manual budget slider remotely (POST /v1/config/vram writes them).
        "vram_auto": config.vram_auto,
        "vram_budget_manual_gb": config.vram_budget_gb,
        "vram_buffer_gb": config.vram_buffer_gb,
        "speed_profile": config.speed_profile,
        "num_parallel": num_parallel,
        "max_loaded_models": max_loaded,
        // Transcription quality this GPU is configured for — ClipAI syncs its
        // own whisper settings + display to match.
        "whisper_quality": config.whisper_quality,
        "whisper_model_effective": whisper_model_eff,
        "whisper_beam_size": whisper_beam,
        // Capability flag: this GPU can run the audio→English translate task
        // (ClipAI routes the tier-A timing pass here when true).
        "whisper_translate": whisper_translate_cap,
        "backends": {
            "ollama": ollama_up,
            "whisper": whisper_shipped,
            "whisper_model": whisper_model_eff,
        },
        "busy": busy,
        "paused": config.paused,
        "current_job": current,
    });
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// Shared JSON body for the VRAM config read + write paths.
fn vram_config_json(ctx: &ProxyCtx) -> serde_json::Value {
    let gpu = ctx.state.gpu.lock().unwrap().clone();
    let cfg = ctx.state.config_snapshot();
    serde_json::json!({
        "vram_auto": cfg.vram_auto,
        "vram_budget_manual_gb": cfg.vram_budget_gb,
        "vram_buffer_gb": cfg.vram_buffer_gb,
        "vram_total_mb": gpu.vram_total_mb,
        "vram_free_mb": gpu.vram_free_mb,
        "effective_budget_gb": ctx.state.effective_budget_gb(),
    })
}

/// GET /v1/config/vram → the current VRAM controls + card total, so ClipAI can
/// seed its auto-allocate toggle + budget slider to what the Companion is
/// actually running (the same values `/v1/health` now carries, standalone).
async fn vram_config_get(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let activity = ctx.state.begin_activity("config", "/v1/config/vram", "", "", "");
    let body = vram_config_json(&ctx);
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// POST /v1/config/vram → set the VRAM budget + auto-allocate REMOTELY (the
/// desktop slider, mirrored over the LAN). Body may carry any of
/// {vram_auto, vram_budget_gb, vram_buffer_gb}; each is applied with the same
/// clamps + Ollama/sidecar restart as the local GUI's `set_config`, then the
/// fresh effective budget is returned so ClipAI can immediately re-check whether
/// the requested model now fits (and say so instead of hanging on "pulling").
async fn set_vram_config(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let activity = ctx.state.begin_activity("config", "/v1/config/vram", "", "", "");
    let bytes = match axum::body::to_bytes(req.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            ctx.state.end_activity(activity, 400);
            return (StatusCode::BAD_REQUEST, format!("bad body: {e}")).into_response();
        }
    };
    let patch: serde_json::Value = if bytes.is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                ctx.state.end_activity(activity, 400);
                return (StatusCode::BAD_REQUEST, format!("bad json: {e}")).into_response();
            }
        }
    };
    let mut ollama_restart = false;
    let mut sidecar_restart = false;
    {
        let mut cfg = ctx.state.config.lock().unwrap();
        if let Some(v) = patch.get("vram_auto").and_then(|x| x.as_bool()) {
            if v != cfg.vram_auto {
                cfg.vram_auto = v;
                ollama_restart = true;
                sidecar_restart = true;
            }
        }
        if let Some(v) = patch.get("vram_budget_gb").and_then(|x| x.as_f64()) {
            let v = (v as f32).max(0.0);
            if (v - cfg.vram_budget_gb).abs() > 0.01 {
                cfg.vram_budget_gb = v;
                ollama_restart = true;
                sidecar_restart = true;
            }
        }
        if let Some(v) = patch.get("vram_buffer_gb").and_then(|x| x.as_f64()) {
            let v = (v as f32).clamp(0.0, 64.0);
            if (v - cfg.vram_buffer_gb).abs() > 0.01 {
                cfg.vram_buffer_gb = v;
                if cfg.vram_auto {
                    ollama_restart = true;
                    sidecar_restart = true;
                }
            }
        }
    }
    // Force the auto-VRAM loop to re-apply immediately (mirrors set_config).
    ctx.state
        .last_auto_baseline_mb
        .store(u64::MAX, Ordering::Relaxed);
    ctx.state.last_auto_apply_ms.store(0, Ordering::Relaxed);
    ctx.state.save();
    if ollama_restart {
        let _ = crate::ollama::restart(&ctx.state).await;
    }
    if sidecar_restart {
        crate::sidecar::shutdown(&ctx.state, "VRAM settings changed remotely").await;
    }
    let mut body = vram_config_json(&ctx);
    if let Some(obj) = body.as_object_mut() {
        obj.insert("ok".into(), serde_json::json!(true));
        obj.insert("ollama_restarted".into(), serde_json::json!(ollama_restart));
    }
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// Shared JSON body for the quality config read + write paths: the two
/// performance fields (Ollama speed profile + Whisper transcription quality)
/// plus what they RESOLVE to on this GPU right now, so ClipAI can render the
/// same "Performance" control the desktop GUI shows — including the honest
/// "which decode actually engages at this VRAM budget" line.
fn quality_config_json(ctx: &ProxyCtx) -> serde_json::Value {
    let cfg = ctx.state.config_snapshot();
    let budget = ctx.state.effective_budget_gb();
    let (whisper_model, _) = crate::state::whisper_tier_for_budget(budget);
    let (beam, prefer_full) = crate::state::whisper_quality_params(&cfg.whisper_quality, budget);
    let model_eff = if prefer_full && whisper_model == "large-v3-turbo" {
        "large-v3"
    } else {
        whisper_model
    };
    let (num_parallel, max_loaded) = ctx.state.resolve_speed_settings();
    serde_json::json!({
        "speed_profile": cfg.speed_profile,
        "whisper_quality": cfg.whisper_quality,
        "effective_budget_gb": budget,
        "num_parallel": num_parallel,
        "max_loaded_models": max_loaded,
        "whisper_effective": {
            "model": model_eff,
            "beam_size": beam,
            "beam_search": beam > 1,
        },
    })
}

/// GET /v1/config/quality → the current performance/quality settings + their
/// effective resolution, so ClipAI can seed its remote control to what this
/// Companion is actually doing.
async fn quality_config_get(State(ctx): State<ProxyCtx>, headers: HeaderMap) -> Response {
    if !authorized(&ctx, &headers) {
        return unauthorized();
    }
    let activity = ctx.state.begin_activity("config", "/v1/config/quality", "", "", "");
    let body = quality_config_json(&ctx);
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

/// POST /v1/config/quality → set the speed profile and/or Whisper quality
/// REMOTELY (the desktop "Performance" control, mirrored over the LAN). Body
/// may carry any of {speed_profile, whisper_quality}; values are validated
/// against the same allow-lists as the local GUI's `set_config`, and invalid
/// values 400 (instead of silently ignoring like the trusted local path — a
/// remote typo should be loud). Applies with the same side effects: a speed
/// change restarts the managed Ollama (new NUM_PARALLEL / MAX_LOADED), a
/// quality change drops the whisper sidecar so the next transcription starts
/// with the new beam-search / model settings.
async fn set_quality_config(State(ctx): State<ProxyCtx>, req: Request<Body>) -> Response {
    if !authorized(&ctx, req.headers()) {
        return unauthorized();
    }
    let activity = ctx.state.begin_activity("config", "/v1/config/quality", "", "", "");
    let bytes = match axum::body::to_bytes(req.into_body(), 64 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            ctx.state.end_activity(activity, 400);
            return (StatusCode::BAD_REQUEST, format!("bad body: {e}")).into_response();
        }
    };
    let patch: serde_json::Value = if bytes.is_empty() {
        serde_json::json!({})
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                ctx.state.end_activity(activity, 400);
                return (StatusCode::BAD_REQUEST, format!("bad json: {e}")).into_response();
            }
        }
    };
    // Validate BEFORE applying anything, so a request with one bad field
    // doesn't half-apply.
    let speed = patch
        .get("speed_profile")
        .and_then(|x| x.as_str())
        .map(|s| s.trim().to_lowercase());
    if let Some(v) = &speed {
        if !matches!(v.as_str(), "auto" | "eco" | "balanced" | "turbo") {
            ctx.state.end_activity(activity, 400);
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid speed_profile {v:?} (auto|eco|balanced|turbo)"),
            )
                .into_response();
        }
    }
    let quality = patch
        .get("whisper_quality")
        .and_then(|x| x.as_str())
        .map(|s| s.trim().to_lowercase());
    if let Some(v) = &quality {
        if !matches!(v.as_str(), "auto" | "fast" | "balanced" | "max") {
            ctx.state.end_activity(activity, 400);
            return (
                StatusCode::BAD_REQUEST,
                format!("invalid whisper_quality {v:?} (auto|fast|balanced|max)"),
            )
                .into_response();
        }
    }
    let mut ollama_restart = false;
    let mut sidecar_restart = false;
    {
        let mut cfg = ctx.state.config.lock().unwrap();
        if let Some(v) = speed {
            if v != cfg.speed_profile {
                cfg.speed_profile = v;
                ollama_restart = true;
            }
        }
        if let Some(v) = quality {
            if v != cfg.whisper_quality {
                cfg.whisper_quality = v;
                sidecar_restart = true;
            }
        }
    }
    ctx.state.save();
    if ollama_restart {
        let _ = crate::ollama::restart(&ctx.state).await;
    }
    if sidecar_restart {
        crate::sidecar::shutdown(&ctx.state, "quality settings changed remotely").await;
    }
    let mut body = quality_config_json(&ctx);
    if let Some(obj) = body.as_object_mut() {
        obj.insert("ok".into(), serde_json::json!(true));
        obj.insert("ollama_restarted".into(), serde_json::json!(ollama_restart));
    }
    ctx.state.end_activity(activity, 200);
    (StatusCode::OK, axum::Json(body)).into_response()
}

fn build_router(ctx: ProxyCtx) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/logs", get(logs_export))
        .route("/v1/files/roots", get(files_roots))
        .route("/v1/files/list", get(files_list))
        .route("/v1/files/read", get(files_read))
        .route("/v1/progress", post(progress_report))
        .route("/v1/sidecar/release", post(sidecar_release))
        .route("/v1/sidecar/warm", post(sidecar_warm))
        .route("/v1/update/install", post(update_install))
        .route("/v1/update/status", get(update_status))
        .route("/v1/gpu/release", post(gpu_release))
        .route("/v1/config/vram", get(vram_config_get).post(set_vram_config))
        .route("/v1/config/quality", get(quality_config_get).post(set_quality_config))
        .route("/v1/jobs/force-end", post(jobs_force_end))
        .route("/v1/audio/transcriptions", post(whisper_proxy))
        .route("/ollama", any(ollama_proxy))
        .route("/ollama/", any(ollama_proxy))
        .route("/ollama/*path", any(ollama_proxy))
        .with_state(ctx)
}

/// Bind with SO_REUSEADDR so a rebind can succeed while a prior instance's
/// socket is still draining (mio doesn't set it on Windows, our primary
/// target). Returns a tokio listener.
fn bind_reuse(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP))?;
    socket.set_reuse_address(true)?;
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    socket.set_nonblocking(true)?;
    let std_listener: std::net::TcpListener = socket.into();
    tokio::net::TcpListener::from_std(std_listener)
}

pub async fn serve(ctx: ProxyCtx) {
    let port = ctx.state.config.lock().unwrap().port;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let mut attempt: u32 = 0;
    // Supervise the listener forever: a transient port conflict on a rapid
    // quit→relaunch must NOT kill the proxy for the whole session (that's the
    // silent "Companion unreachable"). Retry fast while a prior socket drains,
    // then slowly self-heal, always publishing bound/error state to the GUI.
    loop {
        match bind_reuse(addr) {
            Ok(listener) => {
                attempt = 0;
                ctx.state.proxy_bound.store(true, Ordering::Relaxed);
                *ctx.state.proxy_last_error.lock().unwrap() = String::new();
                log::info!("companion proxy listening on {addr}");
                // ConnectInfo gives handlers the peer address, which (with the
                // X-ClipAI-Port header) lets the Companion LEARN the ClipAI
                // server's URL from inbound traffic — self-update then works
                // even for manually-added (never GUI-paired) setups.
                if let Err(e) = axum::serve(
                    listener,
                    build_router(ctx.clone())
                        .into_make_service_with_connect_info::<SocketAddr>(),
                )
                .await
                {
                    log::error!("proxy server exited: {e} — rebinding");
                }
                ctx.state.proxy_bound.store(false, Ordering::Relaxed);
                // Loop to rebind.
            }
            Err(e) => {
                ctx.state.proxy_bound.store(false, Ordering::Relaxed);
                *ctx.state.proxy_last_error.lock().unwrap() =
                    format!("cannot open port {port}: {e}");
                if attempt < 3 {
                    log::error!("could not bind {addr}: {e} (port in use? another program \
                                 or a lingering Companion may hold it — retrying)");
                }
                attempt = attempt.saturating_add(1);
                // ~500ms for the first 5s (ride out a draining prior socket),
                // then back off to every 3s so it recovers the moment it frees.
                let wait_ms = if attempt <= 10 { 500 } else { 3000 };
                tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
            }
        }
    }
}

#[cfg(test)]
mod version_tests {
    use super::version_lt;

    #[test]
    fn older_install_flags_update() {
        assert!(version_lt("0.1.0", "0.2.0"));
        assert!(version_lt("0.1.9", "0.2.0"));
    }

    #[test]
    fn same_or_newer_does_not_flag() {
        assert!(!version_lt("0.2.0", "0.2.0"));
        assert!(!version_lt("0.3.0", "0.2.0"));
        assert!(!version_lt("1.0.0", "0.9.9"));
    }

    #[test]
    fn missing_or_garbage_expected_never_flags() {
        assert!(!version_lt("0.1.0", ""));
        assert!(!version_lt("0.1.0", "latest"));
    }

    #[test]
    fn tolerant_formats() {
        assert!(version_lt("v0.1.0", "companion-v0.2.0"));
        assert!(version_lt("0.1", "0.1.1"));
    }
}

#[cfg(test)]
mod update_decision_tests {
    use super::should_update;

    #[test]
    fn newer_semver_updates_regardless_of_id() {
        assert!(should_update("0.2.4", "aaaaaaa", "0.2.5", "aaaaaaa"));
        assert!(should_update("0.2.4", "aaaaaaa", "0.3.0", ""));
    }

    #[test]
    fn lower_semver_never_downgrades() {
        assert!(!should_update("0.3.0", "aaaaaaa", "0.2.9", "bbbbbbb"));
    }

    #[test]
    fn same_semver_different_build_updates() {
        // The from-source-rebuild case: version unchanged, git SHA changed.
        assert!(should_update("0.2.4", "aaaaaaa", "0.2.4", "bbbbbbb"));
    }

    #[test]
    fn same_semver_same_build_converges() {
        // After installing, ids match → no perpetual re-offer.
        assert!(!should_update("0.2.4", "abcdef1", "0.2.4", "abcdef1"));
        assert!(!should_update("0.2.4", "ABCDEF1", "0.2.4", "abcdef1")); // case-insensitive
    }

    #[test]
    fn placeholder_or_empty_served_id_never_flags() {
        // A build with no real identity can't prove it's newer.
        assert!(!should_update("0.2.4", "abcdef1", "0.2.4", ""));
        assert!(!should_update("0.2.4", "abcdef1", "0.2.4", "unknown"));
        assert!(!should_update("0.2.4", "abcdef1", "0.2.4", "source"));
    }
}

#[cfg(test)]
mod origin_tests {
    use super::plausible_clipai_origin;

    #[test]
    fn accepts_clean_origins() {
        assert!(plausible_clipai_origin("http://192.168.8.5:1353"));
        assert!(plausible_clipai_origin("https://clipai.lan:1353"));
        // A single trailing slash is fine (paired_base trims it).
        assert!(plausible_clipai_origin("http://192.168.8.5:1353/"));
    }

    #[test]
    fn rejects_userinfo_and_paths() {
        // Userinfo could disguise the real host a downloaded installer runs from.
        assert!(!plausible_clipai_origin("http://evil.lan@192.168.8.5:1353"));
        // Anything beyond the authority is not a base origin.
        assert!(!plausible_clipai_origin("http://192.168.8.5:1353/api/x"));
        assert!(!plausible_clipai_origin("http://192.168.8.5:1353?q=1"));
        assert!(!plausible_clipai_origin("http://host\\evil"));
    }

    #[test]
    fn rejects_bad_scheme_whitespace_and_overlong() {
        assert!(!plausible_clipai_origin("ftp://192.168.8.5"));
        assert!(!plausible_clipai_origin("192.168.8.5:1353"));
        assert!(!plausible_clipai_origin("http://host with space"));
        assert!(!plausible_clipai_origin("http://")); // no authority
        let long = format!("http://{}", "a".repeat(300));
        assert!(!plausible_clipai_origin(&long));
    }
}
