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

    // 7. remote quality config: authed, readable, writable, validated — and a
    //    CONFIG route, so it must keep working even while sharing is paused.
    //    (Only whisper_quality is changed here: a speed_profile change would
    //    restart the managed Ollama, which this harness doesn't run.)
    let r = c.get(format!("{base}/v1/config/quality")).send().await.unwrap();
    assert_eq!(r.status(), 401, "quality config without token");
    let r = c
        .get(format!("{base}/v1/config/quality"))
        .bearer_auth(token)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let q: serde_json::Value = r.json().await.unwrap();
    assert_eq!(q["speed_profile"], "auto", "fresh config defaults to auto");
    assert_eq!(q["whisper_quality"], "auto");
    assert!(q["whisper_effective"]["beam_size"].as_u64().is_some());
    let r = c
        .post(format!("{base}/v1/config/quality"))
        .bearer_auth(token)
        .json(&serde_json::json!({"whisper_quality": "MAX "}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "write normalizes case/whitespace");
    let q: serde_json::Value = r.json().await.unwrap();
    assert_eq!(q["ok"], true);
    assert_eq!(q["whisper_quality"], "max");
    assert_eq!(q["ollama_restarted"], false, "whisper-only change must not restart Ollama");
    assert_eq!(
        st.config.lock().unwrap().whisper_quality,
        "max",
        "remote write landed in the persisted config"
    );
    let r = c
        .post(format!("{base}/v1/config/quality"))
        .bearer_auth(token)
        .json(&serde_json::json!({"speed_profile": "ludicrous"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 400, "invalid profile is rejected loudly");
    assert_eq!(
        st.config.lock().unwrap().speed_profile,
        "auto",
        "rejected write must not half-apply"
    );

    // 8. multi-job progress: when ClipAI runs several pipelines at once
    //    (Concurrent Analyses > 1), each job keeps its OWN tracked entry —
    //    heartbeats never overwrite each other, one job ending never wipes
    //    another, and a heartbeat-only job (all its stages so far ran on the
    //    server) still shows up as its own job log.
    let hb = |id: &str, title: &str, stage: &str, pct: &str| {
        c.post(format!("{base}/v1/progress"))
            .bearer_auth(token)
            .header("X-ClipAI-Job-Id", id.to_string())
            .header("X-ClipAI-Job-Title", title.to_string())
            .header("X-ClipAI-Stage", stage.to_string())
            .header("X-ClipAI-Progress", pct.to_string())
            .send()
    };
    let r = hb("job-a", "first video.mp4", "transcribing", "40").await.unwrap();
    assert_eq!(r.status(), 200, "progress heartbeat accepted (even while paused)");
    tokio::time::sleep(std::time::Duration::from_millis(5)).await; // distinct started_ms
    let r = hb("job-b", "second video.mp4", "extracting frames", "10").await.unwrap();
    assert_eq!(r.status(), 200);
    let jobs = st.reported_jobs_fresh();
    assert_eq!(jobs.len(), 2, "both pipelines tracked separately");
    assert_eq!(jobs[0].job_id, "job-a", "oldest started first");
    assert_eq!(jobs[0].progress, 40);
    assert_eq!(jobs[1].job_id, "job-b");
    assert_eq!(jobs[1].stage, "extracting frames");
    let logs = st.job_logs();
    assert!(
        logs.iter().any(|j| j.job_id == "job-a" && j.active && j.reported_progress == 40),
        "job-a keeps its own stage/progress in the job logs"
    );
    assert!(
        logs.iter().any(|j| j.job_id == "job-b" && j.active
            && j.job_title == "second video.mp4"),
        "heartbeat-only job-b (no proxy traffic) still gets its own job log"
    );
    // one job ending clears ONLY that job…
    let r = c
        .post(format!("{base}/v1/progress"))
        .bearer_auth(token)
        .header("X-ClipAI-Job-Id", "job-a")
        .header("X-ClipAI-Job-Ended", "1")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let jobs = st.reported_jobs_fresh();
    assert_eq!(jobs.len(), 1, "ending job-a must not clear job-b");
    assert_eq!(jobs[0].job_id, "job-b");
    // …and a late heartbeat for the ended job must NOT resurrect it.
    let r = hb("job-a", "first video.mp4", "transcribing", "41").await.unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(st.reported_jobs_fresh().len(), 1, "suppressed job stays gone");
    // Per-id force end (each GUI card has its own button) ends exactly job-b.
    let ended = st.force_end_job("job-b");
    assert_eq!(ended.as_deref(), Some("job-b"));
    assert!(st.reported_jobs_fresh().is_empty(), "all jobs ended");
}
