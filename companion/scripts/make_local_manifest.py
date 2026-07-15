#!/usr/bin/env python3
"""Write a downloads-router manifest for locally (from-source) built Companion
installers.

Used by the Dockerfile's companion-builder stage (Windows .exe) and by
``build-macos.sh`` (macOS .dmg): given a directory containing freshly built
installers, emit a ``manifest.json`` in the same shape the release workflow
publishes.

The ClipAI downloads router serves installers by FILE PRESENCE, so this
manifest is optional metadata — it supplies the version and the
``built_from_source`` flag (which drives the UI's "no bundled whisper
sidecar" note). It scans for every installer present (.exe / .msi / .dmg)
so it is correct whether one platform or several live in the directory.
"""
import hashlib
import json
import os
import sys

_EXT_PLATFORM = {".exe": "windows", ".msi": "windows_msi", ".dmg": "mac"}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(out_dir: str) -> int:
    platforms = {}
    for name in sorted(os.listdir(out_dir)):
        ext = os.path.splitext(name)[1].lower()
        platform = _EXT_PLATFORM.get(ext)
        if not platform or platform in platforms:
            continue
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        platforms[platform] = {
            "filename": name,
            "size": os.path.getsize(path),
            "sha256": _sha256(path),
            "url": "",
        }
    if not platforms:
        print("make_local_manifest: no .exe/.msi/.dmg found in", out_dir,
              file=sys.stderr)
        return 1

    # Version from the tauri config (single source of truth for the app).
    version = "0.0.0"
    conf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src-tauri", "tauri.conf.json")
    try:
        with open(conf) as f:
            version = json.load(f).get("version", version)
    except Exception:
        pass

    # Build identity: the ClipAI repo's short git SHA the Docker build passed in
    # (CLIPAI_COMPANION_BUILD_ID). The Companion's Update button compares it to
    # the running binary's own build id, so a from-source rebuild that reused
    # the same semver is still recognised as a newer build. Empty/"unknown"
    # (a build with no SHA) simply never proves a same-version update.
    build_id = (os.environ.get("CLIPAI_COMPANION_BUILD_ID", "") or "").strip()
    manifest = {
        "version": version,
        "tag": f"companion-v{version}",
        "published_at": "",
        "build_id": build_id,
        # Marks these as image/local-build fallbacks. Official companion-v*
        # release assets (fetched by the companion-fetch stage or the
        # "Fetch from GitHub" refresh) overwrite this manifest when present.
        "built_from_source": True,
        "platforms": platforms,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    built = ", ".join(f"{k} ({v['filename']})" for k, v in platforms.items())
    print(f"make_local_manifest: v{version} — {built}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
