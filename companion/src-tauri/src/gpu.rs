//! GPU telemetry. Windows/Linux: nvidia-smi polling. macOS (Apple
//! Silicon): unified memory via sysctl — free VRAM is not a meaningful
//! number there, so we degrade gracefully and report totals only.

use crate::state::{quiet_std_command, GpuSnapshot};

#[cfg(target_os = "windows")]
fn nvidia_smi_candidates() -> Vec<String> {
    let mut v = vec!["nvidia-smi".to_string()];
    if let Ok(windir) = std::env::var("SystemRoot") {
        v.push(format!("{windir}\\System32\\nvidia-smi.exe"));
    }
    v.push("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe".into());
    v
}

#[cfg(not(target_os = "windows"))]
fn nvidia_smi_candidates() -> Vec<String> {
    vec!["nvidia-smi".into(), "/usr/bin/nvidia-smi".into()]
}

fn query_nvidia() -> Option<GpuSnapshot> {
    for bin in nvidia_smi_candidates() {
        let out = quiet_std_command(&bin)
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output();
        let Ok(out) = out else { continue };
        if !out.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&out.stdout);
        // First GPU only — the Companion shares one card.
        let line = text.lines().next()?;
        let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
        if parts.len() < 3 {
            continue;
        }
        return Some(GpuSnapshot {
            gpu_name: parts[0].to_string(),
            vram_total_mb: parts[1].parse().unwrap_or(0),
            vram_free_mb: parts[2].parse().unwrap_or(0),
            unified_memory: false,
            available: true,
        });
    }
    None
}

#[cfg(target_os = "macos")]
fn query_macos() -> Option<GpuSnapshot> {
    // Apple Silicon: GPU shares unified memory. Report the machine total;
    // "free" is approximated from vm_stat's free+inactive pages and is
    // best-effort (blank when unavailable).
    let total_bytes: u64 = quiet_std_command("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())?;
    let chip = quiet_std_command("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "Apple GPU".into());
    let mut free_mb = 0u64;
    if let Ok(out) = quiet_std_command("vm_stat").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        let mut page_size = 16384u64;
        if let Some(first) = text.lines().next() {
            if let Some(idx) = first.find("page size of ") {
                page_size = first[idx + 13..]
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(16384);
            }
        }
        let mut pages = 0u64;
        for line in text.lines() {
            if line.starts_with("Pages free:") || line.starts_with("Pages inactive:") {
                if let Some(n) = line
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().trim_end_matches('.').parse::<u64>().ok())
                {
                    pages += n;
                }
            }
        }
        free_mb = pages * page_size / 1024 / 1024;
    }
    Some(GpuSnapshot {
        gpu_name: format!("{chip} (unified memory)"),
        vram_total_mb: total_bytes / 1024 / 1024,
        vram_free_mb: free_mb,
        unified_memory: true,
        available: true,
    })
}

pub fn snapshot() -> GpuSnapshot {
    if let Some(s) = query_nvidia() {
        return s;
    }
    #[cfg(target_os = "macos")]
    if let Some(s) = query_macos() {
        return s;
    }
    GpuSnapshot {
        gpu_name: "No GPU detected".into(),
        ..Default::default()
    }
}
