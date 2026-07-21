#!/usr/bin/env python3
"""Stamp a monotonically increasing version into the Companion sources.

Called by the Dockerfile's companion-builder stage before ``tauri build``:
``set_build_version.py <build_number>`` rewrites the PATCH component of the
version to the build number (0.3.0 → 0.3.<n>) in every file that feeds the
built app:

  * ``src-tauri/tauri.conf.json`` — the bundle/installer version
  * ``src-tauri/Cargo.toml``      — env!("CARGO_PKG_VERSION"), the version
                                    the running app reports in /v1/health and
                                    compares against the update manifest
  * ``src-tauri/Cargo.lock``      — keeps the pinned self-entry consistent
  * ``package.json``              — the web UI's package version

``update-all.sh`` passes the repo's git commit count as the build number, so
every deployed from-source build carries a version STRICTLY HIGHER than the
previous one — the observed alternative was five self-updates in one day all
announcing "v0.2.9", indistinguishable in every log and update prompt.

``0`` (or a non-number) is a no-op: the repo's base version ships unchanged.
Exit code is always 0 — a stamping hiccup must never abort the image build
(the fallback is simply today's behavior: a same-version build).
"""
import json
import re
import sys


def main() -> int:
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    except (ValueError, TypeError):
        n = 0
    if n <= 0:
        print("set_build_version: no build number — keeping the repo version")
        return 0
    try:
        with open("src-tauri/tauri.conf.json") as f:
            conf = json.load(f)
        base = str(conf.get("version", "0.0.0"))
        majmin = ".".join(base.split(".")[:2])
        newv = f"{majmin}.{n}"

        conf["version"] = newv
        with open("src-tauri/tauri.conf.json", "w") as f:
            json.dump(conf, f, indent=2)
            f.write("\n")

        with open("package.json") as f:
            pkg = json.load(f)
        pkg["version"] = newv
        with open("package.json", "w") as f:
            json.dump(pkg, f, indent=2)
            f.write("\n")

        with open("src-tauri/Cargo.toml") as f:
            toml = f.read()
        toml = re.sub(r'^version = "[^"]+"', f'version = "{newv}"',
                      toml, count=1, flags=re.M)
        with open("src-tauri/Cargo.toml", "w") as f:
            f.write(toml)

        with open("src-tauri/Cargo.lock") as f:
            lock = f.read()
        lock = lock.replace(
            f'name = "clipai-companion"\nversion = "{base}"',
            f'name = "clipai-companion"\nversion = "{newv}"')
        with open("src-tauri/Cargo.lock", "w") as f:
            f.write(lock)

        print(f"set_build_version: {base} → {newv}")
    except Exception as e:  # never abort the image build over version stamping
        print(f"set_build_version: FAILED ({e}) — keeping the repo version",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
