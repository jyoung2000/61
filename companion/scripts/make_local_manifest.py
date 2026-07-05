#!/usr/bin/env python3
"""Write a downloads-router manifest for a locally (from-source) built
Companion installer.

Used by the Dockerfile's companion-builder stage: given a directory
containing the freshly cross-built ``*-setup.exe``, emit the same
``manifest.json`` shape the release workflow publishes so
``backend/routers/downloads.py`` can serve the file and the Settings
card can render a real download button — no GitHub release required.
"""
import hashlib
import json
import os
import sys


def main(out_dir: str) -> int:
    exes = sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".exe"))
    if not exes:
        print("make_local_manifest: no .exe found in", out_dir, file=sys.stderr)
        return 1
    filename = exes[0]
    path = os.path.join(out_dir, filename)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    # Version from the tauri config (single source of truth for the app).
    version = "0.0.0"
    conf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src-tauri", "tauri.conf.json")
    try:
        with open(conf) as f:
            version = json.load(f).get("version", version)
    except Exception:
        pass

    manifest = {
        "version": version,
        "tag": f"companion-v{version}",
        "published_at": "",
        # Marks this as an image-build fallback (no whisper sidecar bundled;
        # see the Companion dashboard note). GitHub-release assets fetched by
        # the companion-fetch stage overwrite this manifest when they exist.
        "built_from_source": True,
        "platforms": {
            "windows": {
                "filename": filename,
                "size": os.path.getsize(path),
                "sha256": digest.hexdigest(),
                "url": "",
            },
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"make_local_manifest: {filename} "
          f"({manifest['platforms']['windows']['size']} bytes, v{version})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
