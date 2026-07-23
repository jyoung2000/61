//! Vision sidecar lifecycle (face-detection offload).
//!
//! A PyInstaller-packaged YOLO-World server (`vision-server[.exe]`) serving
//! `/v1/vision/detect` on localhost behind the proxy. Same contract as the
//! whisper sidecar: starts lazily on the first request, dies with the app,
//! is stopped by `free_gpu`. When no binary is installed (older installers,
//! from-source builds) every route answers 404 and ClipAI silently keeps
//! face detection local — the offload is strictly additive.

use crate::state::{quiet_command, AppState};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::process::Child;

pub const VISION_SIDECAR_PORT: u16 = 11511;

pub fn vision_url() -> String {
    format!("http://127.0.0.1:{VISION_SIDECAR_PORT}")
}

fn resolve_binary(resource_dir: &PathBuf, data_dir: &PathBuf) -> Option<PathBuf> {
    let names: [&str; 2] = if cfg!(target_os = "windows") {
        ["vision-server.exe", "vision-server.exe"]
    } else {
        ["vision-server", "vision-server"]
    };
    let downloaded = data_dir.join("vision-bin").join(names[0]);
    if downloaded.is_file() {
        return Some(downloaded);
    }
    let bundled = resource_dir.join("sidecar").join(names[1]);
    if bundled.is_file() {
        return Some(bundled);
    }
    None
}

pub fn available(resource_dir: &PathBuf, data_dir: &PathBuf) -> bool {
    resolve_binary(resource_dir, data_dir).is_some() || resolve_source_launch(data_dir).is_some()
}

/// Working dir for a from-source install: venv + server.py + the local
/// YOLO-World weight all live here, under app-data so they survive updates.
pub fn source_dir(data_dir: &PathBuf) -> PathBuf {
    data_dir.join("vision-src")
}

/// A from-source install that's ready to (re)launch: the embedded server.py
/// plus a venv python. Lets the offload survive a Companion restart without
/// any packaged binary — the CI-free path.
pub fn resolve_source_launch(data_dir: &PathBuf) -> Option<(PathBuf, PathBuf)> {
    let work = source_dir(data_dir);
    let script = work.join("server.py");
    if !script.is_file() {
        return None;
    }
    let py = if cfg!(target_os = "windows") {
        work.join(".venv").join("Scripts").join("python.exe")
    } else {
        work.join(".venv").join("bin").join("python")
    };
    if py.is_file() {
        Some((py, script))
    } else {
        None
    }
}

/// The locally-cached YOLO-World weight, if the install fetched it from the
/// ClipAI container. Pointing the sidecar at this via VISION_MODEL means
/// ultralytics never reaches out to GitHub for the weight.
fn local_model_path(data_dir: &PathBuf) -> Option<PathBuf> {
    let p = source_dir(data_dir).join("yolov8s-worldv2.pt");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Live progress of a from-source vision-offload install. Serialized to the
/// local GUI (get_status) and the remote container (/v1/vision/install/status).
#[derive(Clone, Default, serde::Serialize)]
pub struct InstallProgress {
    /// An install is running right now.
    pub active: bool,
    /// Machine-ish phase: idle|preparing|python|venv|torch|deps|model|starting|running|error.
    pub stage: String,
    /// 0..100, or -1 for indeterminate.
    pub percent: f64,
    /// Human-readable line, live-updated (e.g. the latest pip output line).
    pub message: String,
    /// Set on failure.
    pub error: Option<String>,
    /// ms epoch of the last update.
    pub updated_ms: u64,
}


// ── From-source install (NO GitHub) ───────────────────────────────────────
// The vision offload used to depend on a GitHub-release asset; that path is
// gone. Instead the Companion bootstraps everything itself, entirely off
// GitHub: Python (system, else python.org), torch (pytorch.org), the rest of
// the deps (PyPI), and the YOLO-World weight (streamed from the paired ClipAI
// container over the LAN). The install can be triggered locally (the GUI
// button) or remotely (the container's button, via /v1/vision/install), and
// both watch the same InstallProgress.

/// The embedded sidecar source — written into app-data at install time so the
/// venv has something to run without shipping a separate file.
const SERVER_PY: &str = include_str!("../../sidecars/vision-server/server.py");

fn now_ms() -> u64 {
    crate::state::now_ms()
}

/// Set the shared install-progress state (and log the stage).
fn set_progress(state: &AppState, active: bool, stage: &str, percent: f64, message: impl Into<String>) {
    let message = message.into();
    log::info!("vision install [{stage} {percent:.0}%] {message}");
    if let Ok(mut g) = state.vision_install.lock() {
        g.active = active;
        g.stage = stage.to_string();
        g.percent = percent;
        g.message = message;
        g.error = None;
        g.updated_ms = now_ms();
    }
}

fn set_error(state: &AppState, message: impl Into<String>) {
    let message = message.into();
    log::warn!("vision install failed: {message}");
    if let Ok(mut g) = state.vision_install.lock() {
        g.active = false;
        g.stage = "error".to_string();
        g.message = format!("Failed: {message}");
        g.error = Some(message);
        g.updated_ms = now_ms();
    }
}

/// Run a child process, streaming its combined output into the progress
/// `message` so the UI shows live updates, and failing on a non-zero exit.
async fn run_streamed(
    state: &AppState,
    stage: &str,
    percent: f64,
    mut cmd: tokio::process::Command,
) -> Result<(), String> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("could not start step: {e}"))?;
    let mut lines: tokio::io::Lines<BufReader<tokio::process::ChildStdout>> =
        BufReader::new(child.stdout.take().expect("piped stdout")).lines();
    let mut err_lines =
        BufReader::new(child.stderr.take().expect("piped stderr")).lines();
    loop {
        tokio::select! {
            l = lines.next_line() => match l {
                Ok(Some(line)) => { let t = line.trim(); if !t.is_empty() { set_progress(state, true, stage, percent, t); } }
                _ => break,
            },
            e = err_lines.next_line() => match e {
                Ok(Some(line)) => { let t = line.trim(); if !t.is_empty() { set_progress(state, true, stage, percent, t); } }
                _ => {}
            },
        }
    }
    // Drain any remaining stderr.
    while let Ok(Some(line)) = err_lines.next_line().await {
        let t = line.trim();
        if !t.is_empty() {
            set_progress(state, true, stage, percent, t);
        }
    }
    let status = child.wait().await.map_err(|e| format!("step wait failed: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("step exited with {status}"))
    }
}

/// Probe a candidate Python: returns its real executable path when it's a
/// usable 3.9–3.12 with the `venv` module available.
async fn probe_python(cand: &str) -> Option<PathBuf> {
    let out = quiet_command(cand)
        .arg("-c")
        .arg("import sys,venv;print(sys.executable);print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.lines();
    let exe = it.next()?.trim().to_string();
    let ver = it.next()?.trim();
    let mut parts = ver.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    if major == 3 && (9..=12).contains(&minor) && !exe.is_empty() {
        Some(PathBuf::from(exe))
    } else {
        None
    }
}

/// Ensure a usable Python: prefer one already on the machine; on Windows,
/// fall back to a per-user install from python.org (NOT GitHub).
async fn ensure_python(state: &AppState, data_dir: &PathBuf) -> Result<PathBuf, String> {
    set_progress(state, true, "python", 6.0, "Looking for Python 3.11…");
    for cand in ["python3.11", "py -3.11", "python3", "python", "py"] {
        // `py -3.11` must be split into program + arg.
        let found = if let Some(rest) = cand.strip_prefix("py ") {
            let out = quiet_command("py")
                .arg(rest)
                .arg("-c")
                .arg("import sys,venv;print(sys.executable);print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
                .output()
                .await
                .ok();
            match out {
                Some(o) if o.status.success() => {
                    let text = String::from_utf8_lossy(&o.stdout);
                    text.lines().next().map(|s| PathBuf::from(s.trim())).filter(|p| p.exists())
                }
                _ => None,
            }
        } else {
            probe_python(cand).await
        };
        if let Some(p) = found {
            set_progress(state, true, "python", 10.0, format!("Using Python at {}", p.display()));
            return Ok(p);
        }
    }

    #[cfg(target_os = "windows")]
    {
        return download_python_windows(state, data_dir).await;
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = data_dir;
        Err("Python 3.9–3.12 was not found. Install it from python.org and run the install again.".into())
    }
}

/// Download + silently install a private, per-user Python 3.11 from python.org
/// (no admin, no GitHub) when the machine has none.
#[cfg(target_os = "windows")]
async fn download_python_windows(state: &AppState, data_dir: &PathBuf) -> Result<PathBuf, String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    const PY_VER: &str = "3.11.9";
    let work = source_dir(data_dir);
    let _ = std::fs::create_dir_all(&work);
    let target = work.join("py311");
    let py_exe = target.join("python.exe");
    if py_exe.is_file() {
        return Ok(py_exe);
    }
    set_progress(state, true, "python", 7.0, "Downloading Python 3.11 from python.org…");
    let url = format!("https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-amd64.exe");
    let installer = work.join("python-setup.exe");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client.get(&url).send().await.map_err(|e| format!("python download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("python download failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let mut file = tokio::fs::File::create(&installer).await.map_err(|e| e.to_string())?;
    let mut done: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("python download interrupted: {e}"))?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        done += chunk.len() as u64;
        if total > 0 {
            set_progress(state, true, "python", 7.0 + (done as f64 / total as f64) * 2.0, "Downloading Python 3.11…");
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);
    set_progress(state, true, "python", 9.0, "Installing Python 3.11 (per-user)…");
    // Per-user, silent, with pip; no PATH changes, no admin prompt.
    let mut cmd = quiet_command(&installer);
    cmd.args([
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_pip=1",
        "Include_launcher=0",
        "Include_test=0",
        "Shortcuts=0",
    ]);
    cmd.arg(format!("TargetDir={}", target.display()));
    let status = cmd.status().await.map_err(|e| format!("python install failed to start: {e}"))?;
    let _ = std::fs::remove_file(&installer);
    if !status.success() {
        return Err(format!("python installer exited with {status}"));
    }
    if py_exe.is_file() {
        Ok(py_exe)
    } else {
        Err("Python installed but python.exe was not found where expected".into())
    }
}

/// Download the YOLO-World weight from the paired ClipAI container (LAN, no
/// GitHub) into the source dir, so ultralytics never fetches it itself.
async fn fetch_model(
    state: &AppState,
    data_dir: &PathBuf,
    model_url: &str,
    token: Option<&str>,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    let dest = source_dir(data_dir).join("yolov8s-worldv2.pt");
    if dest.is_file() && std::fs::metadata(&dest).map(|m| m.len() > 1_000_000).unwrap_or(false) {
        return Ok(()); // already have a plausible weight
    }
    set_progress(state, true, "model", 86.0, "Downloading the YOLO-World model from ClipAI…");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(900))
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client.get(model_url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req.send().await.map_err(|e| format!("model download failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("model download failed: HTTP {} (is ClipAI reachable?)", resp.status()));
    }
    let total = resp.content_length().unwrap_or(0);
    let tmp = dest.with_extension("part");
    let mut file = tokio::fs::File::create(&tmp).await.map_err(|e| e.to_string())?;
    let mut done: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("model download interrupted: {e}"))?;
        file.write_all(&chunk).await.map_err(|e| e.to_string())?;
        done += chunk.len() as u64;
        if total > 0 {
            set_progress(state, true, "model", 86.0 + (done as f64 / total as f64) * 3.0, "Downloading the YOLO-World model…");
        }
    }
    file.flush().await.map_err(|e| e.to_string())?;
    drop(file);
    std::fs::rename(&tmp, &dest).map_err(|e| format!("could not save model: {e}"))?;
    Ok(())
}

/// The end-to-end from-source install. Single-flight (a second call while one
/// runs is a no-op). Drives `state.vision_install` the whole way.
pub async fn install_from_source(
    state: Arc<AppState>,
    resource_dir: PathBuf,
    data_dir: PathBuf,
    model_url: Option<String>,
    model_token: Option<String>,
) {
    // Never run the heavy install (torch download is ~2.5 GB and pegs the box)
    // while this GPU is transcribing — that starved Whisper for a whole job and
    // made the Companion unresponsive. Make the user run it when idle.
    if state.whisper_busy.load(std::sync::atomic::Ordering::Relaxed) {
        set_error(&state, "A transcription is running on this GPU — wait for the current job to finish, then install.");
        return;
    }
    {
        let mut g = match state.vision_install.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if g.active {
            return;
        }
        *g = InstallProgress {
            active: true,
            stage: "preparing".into(),
            percent: 2.0,
            message: "Preparing…".into(),
            error: None,
            updated_ms: now_ms(),
        };
    }
    match install_inner(&state, &resource_dir, &data_dir, model_url, model_token).await {
        Ok(()) => set_progress(&state, false, "running", 100.0, "Vision offload running on this GPU"),
        Err(e) => set_error(&state, e),
    }
}

async fn install_inner(
    state: &Arc<AppState>,
    resource_dir: &PathBuf,
    data_dir: &PathBuf,
    model_url: Option<String>,
    model_token: Option<String>,
) -> Result<(), String> {
    let work = source_dir(data_dir);
    std::fs::create_dir_all(&work).map_err(|e| format!("could not create {}: {e}", work.display()))?;
    // Write the embedded server.py (idempotent — always refresh to the shipped
    // version so an app update carries fixes into an existing install).
    std::fs::write(work.join("server.py"), SERVER_PY)
        .map_err(|e| format!("could not write server.py: {e}"))?;

    // 1) Python.
    let py = ensure_python(state, data_dir).await?;

    // 2) venv.
    set_progress(state, true, "venv", 12.0, "Creating the Python environment…");
    let venv_dir = work.join(".venv");
    let mut venv_cmd = quiet_command(&py);
    venv_cmd.arg("-m").arg("venv").arg(&venv_dir);
    let st = venv_cmd.status().await.map_err(|e| format!("venv failed to start: {e}"))?;
    if !st.success() {
        return Err(format!("creating the virtual environment failed ({st})"));
    }
    let venv_py = if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    };
    if !venv_py.is_file() {
        return Err("virtual environment python not found after venv".into());
    }

    // 3) pip upgrade.
    set_progress(state, true, "deps", 15.0, "Upgrading pip…");
    let mut up = quiet_command(&venv_py);
    up.args(["-m", "pip", "install", "--upgrade", "pip"]);
    run_streamed(state, "deps", 15.0, up).await?;

    // 4) torch (CUDA from pytorch.org — NOT GitHub, NOT the CPU PyPI wheel).
    set_progress(state, true, "torch", 20.0, "Installing PyTorch (CUDA) — this is the big one…");
    let mut torch = quiet_command(&venv_py);
    if cfg!(target_os = "windows") {
        torch.args([
            "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ]);
    } else {
        torch.args(["-m", "pip", "install", "torch", "torchvision"]);
    }
    run_streamed(state, "torch", 20.0, torch).await?;

    // 5) the rest of the sidecar deps (PyPI).
    set_progress(state, true, "deps", 68.0, "Installing YOLO-World + server dependencies…");
    let mut deps = quiet_command(&venv_py);
    deps.args([
        "-m", "pip", "install",
        "fastapi==0.111.0", "uvicorn==0.30.0", "ultralytics>=8.2,<8.4",
        "opencv-python-headless", "numpy",
    ]);
    run_streamed(state, "deps", 68.0, deps).await?;

    // 6) model weight from ClipAI (LAN). Without it ultralytics would fetch
    //    from GitHub on first run — which is exactly what we're avoiding.
    if let Some(url) = model_url.as_deref() {
        fetch_model(state, data_dir, url, model_token.as_deref()).await?;
    } else if local_model_path(data_dir).is_none() {
        return Err("no model source given — ClipAI must provide the YOLO-World weight (pair the Companion with ClipAI, then retry)".into());
    }

    // 7) launch + wait for healthy.
    set_progress(state, true, "starting", 92.0, "Starting the vision sidecar…");
    // Drop any stale child so ensure_running relaunches from the fresh venv.
    {
        let mut guard = state.vision_sidecar.lock().await;
        if let Some(mut child) = guard.take() {
            let _ = child.kill().await;
        }
    }
    ensure_running(state, resource_dir.clone(), data_dir.clone()).await?;
    Ok(())
}

/// Remove the from-source vision offload: stop the sidecar and delete the
/// venv + server + weight so `available()` goes false and ClipAI stops trying
/// to offload — a clean revert to the pre-install (faces-local) behaviour.
/// Does NOT touch a packaged/downloaded binary (there's nothing to clean up
/// there); a user who wants the offload back just installs again.
pub async fn uninstall(state: &AppState, data_dir: &PathBuf) -> Result<(), String> {
    shutdown(state, "vision offload removed").await;
    let work = source_dir(data_dir);
    if work.exists() {
        std::fs::remove_dir_all(&work)
            .map_err(|e| format!("could not remove {}: {e}", work.display()))?;
    }
    if let Ok(mut g) = state.vision_install.lock() {
        *g = InstallProgress::default();
    }
    Ok(())
}

pub async fn healthy() -> bool {
    reqwest::Client::new()
        .get(format!("{}/health", vision_url()))
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

/// Start (or confirm) the vision sidecar. Returns Err when no binary exists.
pub async fn ensure_running(
    state: &Arc<AppState>,
    resource_dir: PathBuf,
    data_dir: PathBuf,
) -> Result<(), String> {
    // CI-free path: a vision server the user started from source (or that we
    // launched earlier) is already answering on the sidecar port — use it
    // as-is, no packaged binary required. This is how the offload works when
    // the GitHub-release asset can't be built (e.g. Actions disabled) — the
    // user runs sidecars/vision-server/run.{ps1,sh} on this machine.
    if healthy().await {
        return Ok(());
    }
    let mut guard = state.vision_sidecar.lock().await;
    if let Some(child) = guard.as_mut() {
        let alive = child.try_wait().map(|s| s.is_none()).unwrap_or(false);
        if alive && healthy().await {
            return Ok(());
        }
        if !alive {
            log::warn!("vision sidecar process had EXITED on its own — starting a fresh one");
        }
        let _ = child.kill().await;
        *guard = None;
    }
    // Prefer a packaged/downloaded binary; else launch the from-source venv
    // (server.py under app-data). Either way, when a LOCAL weight is present
    // point the sidecar at it via VISION_MODEL so ultralytics never fetches
    // from GitHub.
    let mut cmd = if let Some(binary) = resolve_binary(&resource_dir, &data_dir) {
        log::info!("starting vision sidecar (packaged): {} (port {VISION_SIDECAR_PORT})", binary.display());
        quiet_command(&binary)
    } else if let Some((venv_py, script)) = resolve_source_launch(&data_dir) {
        log::info!("starting vision sidecar (from source): {} {} (port {VISION_SIDECAR_PORT})", venv_py.display(), script.display());
        let mut c = quiet_command(&venv_py);
        c.arg(&script);
        c
    } else {
        return Err("no vision sidecar installed".to_string());
    };
    cmd.env("VISION_PORT", VISION_SIDECAR_PORT.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(model) = local_model_path(&data_dir) {
        cmd.env("VISION_MODEL", model);
    }
    let child: Child = cmd
        .spawn()
        .map_err(|e| format!("could not start vision sidecar: {e}"))?;
    crate::state::bind_child_to_lifetime(&child);
    *guard = Some(child);
    drop(guard);
    // First start loads YOLO-World weights — allow a generous warmup.
    for _ in 0..240 {
        if healthy().await {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    Err("vision sidecar started but never became healthy".into())
}

pub async fn shutdown(state: &AppState, reason: &str) {
    if let Some(mut child) = state.vision_sidecar.lock().await.take() {
        let _ = child.kill().await;
        let _ = child.wait().await;
        log::info!("vision sidecar stopped ({reason})");
    }
}

/// Whether the vision offload is allowed under the user's speed settings.
///
/// * "eco" reserves the card for other apps — vision stays on the ClipAI
///   server (exactly the pre-offload behavior).
/// * The VRAM budget must leave room for YOLO-World (~1 GB) NEXT TO a
///   concurrent whisper decode (~3 GB fp16 turbo) — below 5 GB the offload
///   would fight the transcription for memory, so it stays local.
pub fn allowed(profile: &str, budget_gb: f32) -> bool {
    profile != "eco" && budget_gb >= 5.0
}

#[cfg(test)]
mod tests {
    use super::allowed;

    #[test]
    fn eco_profile_keeps_vision_local() {
        assert!(!allowed("eco", 12.0));
    }

    #[test]
    fn small_budget_keeps_vision_local() {
        assert!(!allowed("turbo", 4.0));
    }

    #[test]
    fn auto_and_turbo_allow_vision_with_room() {
        assert!(allowed("auto", 9.5));
        assert!(allowed("turbo", 9.5));
        assert!(allowed("balanced", 6.0));
    }
}
