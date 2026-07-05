//! Companion proxy end-to-end test: real sockets, real axum server, a fake
//! localhost Ollama. Verifies the security and routing contract ClipAI
//! depends on: bearer auth on every route, /ollama prefix stripping,
//! X-ClipAI header passthrough (without leaking the bearer token
//! upstream), the activity ledger, whisper saturation → honest 503 +
//! Retry-After, missing-sidecar → clear 502, and pause → 503.
//!
//! One test function on purpose — the fake Ollama binds the fixed
//! 127.0.0.1:11434 the proxy targets, so parallel tests would collide.

use clipai_companion_lib::{proxy, state};
use std::sync::Arc;

#[tokio::test]
async fn proxy_auth_routing_and_saturation() {
    // ── fake Ollama on the proxy's fixed upstream address ──
    use axum::{http::HeaderMap, routing::get, Json, Router};
    async fn tags(headers: HeaderMap) -> Json<serde_json::Value> {
        let job = headers
            .get("x-clipai-job-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let auth_leaked = headers.contains_key("authorization");
        Json(serde_json::json!({
            "models": [{"name": "moondream:1.8b"}],
            "seen_job_header": job,
            "auth_header_leaked": auth_leaked,
        }))
    }
    let fake = Router::new().route("/api/tags", get(tags));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:11434")
        .await
        .expect("bind fake ollama on 127.0.0.1:11434");
    tokio::spawn(async move { axum::serve(listener, fake).await.unwrap() });

    // ── Companion state + proxy ──
    let dir = std::env::temp_dir().join("companion-proxy-e2e-cfg");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let st: Arc<state::AppState> = Arc::new(state::AppState::load(dir.clone()));
    st.config.lock().unwrap().token = "test-token-0123456789".into();
    st.config.lock().unwrap().port = 11599;
    let ctx = proxy::ProxyCtx {
        state: st.clone(),
        client: reqwest::Client::new(),
        resource_dir: dir.clone(), // deliberately holds no sidecar binary
        data_dir: dir.clone(),
    };
    tokio::spawn(proxy::serve(ctx));
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;

    let c = reqwest::Client::new();
    let base = "http://127.0.0.1:11599";
    let token = "test-token-0123456789";

    // 1. every route rejects missing/wrong tokens
    let r = c.get(format!("{base}/v1/health")).send().await.unwrap();
    assert_eq!(r.status(), 401, "health without token");
    let r = c
        .get(format!("{base}/ollama/api/tags"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401, "ollama with wrong token");
    let r = c
        .post(format!("{base}/v1/audio/transcriptions"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401, "whisper without token");

    // 2. health with token reports GPU + backend status
    let r = c
        .get(format!("{base}/v1/health"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let health: serde_json::Value = r.json().await.unwrap();
    assert_eq!(health["backends"]["ollama"], true);
    assert!(health.get("vram_total_mb").is_some());

    // 3. /ollama prefix strip, X-ClipAI passthrough, no bearer leak upstream
    let r = c
        .get(format!("{base}/ollama/api/tags"))
        .bearer_auth(token)
        .header("X-ClipAI-Job-Id", "job-42")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["models"][0]["name"], "moondream:1.8b");
    assert_eq!(body["seen_job_header"], "job-42");
    assert_eq!(body["auth_header_leaked"], false, "bearer token must not reach upstream");
    assert!(
        st.activity
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.job_id == "job-42" && e.finished_at_ms.is_some()),
        "activity ledger records the proxied job"
    );

    // 4. whisper saturation: slot held → honest 503 + Retry-After
    let permit = st.whisper_slot.try_acquire().unwrap();
    let r = c
        .post(format!("{base}/v1/audio/transcriptions"))
        .bearer_auth(token)
        .body("x")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503, "busy whisper answers 503");
    assert!(r.headers().get("retry-after").is_some(), "503 carries Retry-After");
    drop(permit);

    // 5. missing sidecar binary → clear 502, never a hang
    let r = c
        .post(format!("{base}/v1/audio/transcriptions"))
        .bearer_auth(token)
        .body("x")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    assert!(r.text().await.unwrap_or_default().contains("sidecar"));

    // 6. pause sharing → 503 on proxied work
    st.config.lock().unwrap().paused = true;
    let r = c
        .get(format!("{base}/ollama/api/tags"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503, "paused Companion refuses work");
}
