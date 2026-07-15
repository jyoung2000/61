use std::process::Command;

fn main() {
    // Build identity for the self-updater. The Update button compares this id
    // against the one ClipAI advertises in its installer manifest, so a
    // from-source rebuild that DIDN'T bump the semver is still recognised as a
    // newer build (a fresh git SHA differs from the running one). Preference:
    //   1. CLIPAI_COMPANION_BUILD_ID — the repo's short git SHA that ClipAI's
    //      Docker build passes in (authoritative when built in the container);
    //   2. a local `git rev-parse --short HEAD` for dev builds off a checkout;
    //   3. "source" when neither is available (never flags an update — an
    //      unidentifiable build can't prove it is newer).
    println!("cargo:rerun-if-env-changed=CLIPAI_COMPANION_BUILD_ID");
    let id = std::env::var("CLIPAI_COMPANION_BUILD_ID")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "unknown")
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "source".to_string());
    println!("cargo:rustc-env=CLIPAI_BUILD_ID={id}");

    tauri_build::build()
}
