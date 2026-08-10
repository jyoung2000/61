#!/usr/bin/env bash
#
# One-shot updater: rebuild the ClipAI container AND cross-build the GPU
# Companion installer from source, restart the app (Ollama + models stay up),
# then verify the served installer version.
#
# Living in the repo so it never has to be pasted into a terminal (long
# multi-line pastes drop lines and corrupt inline here-scripts). Run it with:
#
#   cd /mnt/user/appdata/clipai
#   nohup bash update-all.sh > /mnt/user/appdata/clipai-update.log 2>&1 &
#   tail -f /mnt/user/appdata/clipai-update.log
#
# The script fetches + hard-resets to DEFAULT_BRANCH below (override with
# `bash update-all.sh <branch>` or CLIPAI_UPDATE_BRANCH=<branch>), so any
# checkout done before invoking it is replaced — keep DEFAULT_BRANCH pointed
# at the branch you actually want deployed. (This bit a real update: the
# caller reset to the new work branch, then this script's stale default
# quietly reset BACK to the old branch and rebuilt the old code.)
#
# The Companion cross-build (COMPANION_BUILD_FROM_SOURCE=1) needs outbound
# internet (~1 GB MSVC SDK the first time) and can take 10-30 min; the build
# cache makes later runs far faster. This script SERVES the new Companion from
# ClipAI — the final install is one "Update" click in the Companion app on the
# Windows GPU box (that step can't run from Unraid).
#
# The companion-builder stage is fail-soft, so a failed cross-build bakes an
# EMPTY installer dir into the image — and BuildKit then serves that cached
# empty result on every rebuild of the same commit. When this script finds no
# .exe (or a stale version) in the built image, it now automatically re-runs
# JUST the companion cross-build with a cache-bust token and publishes the
# result straight into ./data/companion-cache (which ClipAI serves) — no app
# rebuild or restart, and the real build error finally shows in this log.
# It also no longer wipes the currently-served installer unless a fresh one
# actually exists to replace it.
#
# Usage: bash update-all.sh [branch]
set -uo pipefail
cd "$(dirname "$0")"

DEFAULT_BRANCH="claude/clipai-bookmarks-bulk-upload-0nsv1p"
BRANCH="${1:-${CLIPAI_UPDATE_BRANCH:-$DEFAULT_BRANCH}}"
START=$(date +%s)
log(){ local t=$(( $(date +%s) - START )); printf '\n[all] %02d:%02d %s\n' $((t/60)) $((t%60)) "$*"; }

if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1
# Clean-rebuild escape hatch: CLIPAI_NOCACHE=1 forces --no-cache so a suspected
# stale layer can never serve old code (slower). BuildKit already invalidates
# layers on any changed file, so this is only for paranoia / a reported mismatch.
NOCACHE=""; [ "${CLIPAI_NOCACHE:-0}" = "1" ] && NOCACHE="--no-cache"

# ── Docker disk preflight ────────────────────────────────────────────────────
# A real update died with "write /var/lib/docker/...: no space left on device"
# — on Unraid the Docker vDisk (docker.img) is a FIXED-size loopback, and the
# ClipAI image + superseded builds + BuildKit caches fill it over time. Before
# burning 20+ minutes on a doomed build, check free space in Docker's data
# root and reclaim safely, in escalating steps:
#   1. dangling images  — untagged layers from previous clipai-app builds;
#      always safe to delete (they're the *old* versions of this same image).
#   2. BuildKit cache over a keep-budget — LRU trim, so the hot companion
#      cross-build caches (cargo target, MSVC SDK) survive as long as they fit.
# NEVER pruned automatically: containers, volumes, or other apps' tagged
# images. If that still isn't enough, abort with exact instructions instead of
# failing 20 minutes in.
MIN_FREE_GB="${CLIPAI_MIN_FREE_GB:-8}"
KEEP_CACHE_GB="${CLIPAI_BUILDCACHE_KEEP_GB:-6}"
DOCKER_ROOT="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null)"
[ -z "$DOCKER_ROOT" ] && DOCKER_ROOT="/var/lib/docker"
_free_gb(){ df -Pk "$DOCKER_ROOT" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}'; }

ensure_docker_space(){
  local free; free="$(_free_gb)"; free="${free:-0}"
  log "docker storage: ${free}GB free at $DOCKER_ROOT (want ≥ ${MIN_FREE_GB}GB)"
  [ "$free" -ge "$MIN_FREE_GB" ] && return 0
  log "low on docker disk — pruning dangling images (superseded clipai builds)…"
  docker image prune -f 2>&1 | tail -1
  free="$(_free_gb)"; free="${free:-0}"
  [ "$free" -ge "$MIN_FREE_GB" ] && { log "…now ${free}GB free ✓"; return 0; }
  log "still ${free}GB — trimming BuildKit cache to ${KEEP_CACHE_GB}GB (LRU keeps the hot companion caches)…"
  docker builder prune -f --keep-storage "${KEEP_CACHE_GB}GB" 2>&1 | tail -1
  free="$(_free_gb)"; free="${free:-0}"
  [ "$free" -ge "$MIN_FREE_GB" ] && { log "…now ${free}GB free ✓"; return 0; }
  log "STILL only ${free}GB free after safe pruning. Not deleting anything riskier automatically."
  log "  Unraid fix: Settings -> Docker -> stop the service -> increase the vDisk size (the ClipAI image alone needs many GB)."
  log "  Or reclaim harder by hand (understand what each deletes first):"
  log "    docker builder prune -af          # ALL build cache — next build recompiles everything"
  log "    docker image prune -af            # ALL images not used by a container, other apps' too"
  return 1
}

# Free-space knob for one aggressive retry when a build dies of ENOSPC anyway
# (the preflight passed but the build itself outgrew the disk).
reclaim_hard(){
  log "reclaiming docker disk the hard way (all build cache + dangling images)…"
  docker builder prune -af 2>&1 | tail -1
  docker image prune -f 2>&1 | tail -1
  log "…now $(_free_gb)GB free at $DOCKER_ROOT"
}

log "pulling latest ($BRANCH)…"
# Retry the fetch — a transient network blip must not leave you on old code.
_fetched=""
for _i in 1 2 3 4; do
  if git fetch origin "$BRANCH"; then _fetched=1; break; fi
  log "git fetch failed (attempt $_i) — retrying in $((2 ** _i))s"; sleep $((2 ** _i))
done
[ -z "$_fetched" ] && { log "git fetch FAILED after 4 tries — check the network"; exit 1; }
git reset --hard "origin/$BRANCH"     || { log "git reset FAILED"; exit 1; }
export BUILD_SHA="$(git rev-parse --short HEAD)"
export BUILD_SUBJECT="$(git log -1 --pretty=%s)"
# Monotonic Companion build number: the repo's commit count. The builder
# stamps it into the version's PATCH slot (0.3.0 → 0.3.<count>), so every
# deploy's Companion carries a strictly higher version than the last —
# update prompts and logs stop showing five identical "v0.2.9" installs.
BUILD_NUM="$(git rev-list --count HEAD 2>/dev/null || echo 0)"
BASE_V="$(grep -m1 '^version' companion/src-tauri/Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
EXPECTED="${BASE_V%.*}.${BUILD_NUM}"
[ "$BUILD_NUM" = "0" ] && EXPECTED="$BASE_V"
log "commit $BUILD_SHA — building container + Companion v$EXPECTED (from source; needs internet)"

ensure_docker_space || { log "ABORTING before the build — free docker disk first (see above), then re-run: bash update-all.sh"; exit 1; }

# Build with the output tee'd so an ENOSPC failure is detectable: when the
# disk fills DURING the build (preflight passed but the new layers outgrew
# it), reclaim hard and retry ONCE instead of leaving the box on old code.
BUILD_LOG="$(mktemp /tmp/clipai-build.XXXXXX)"
run_app_build(){
  $DC build $NOCACHE --build-arg COMPANION_BUILD_FROM_SOURCE=1 \
      --build-arg COMPANION_BUILD_NUMBER="$BUILD_NUM" app 2>&1 | tee "$BUILD_LOG"
  return "${PIPESTATUS[0]}"
}
( while :; do sleep 60; log "…still building ($(( ($(date +%s)-START)/60 ))m elapsed)"; done ) & HB=$!
if ! run_app_build; then
  if grep -qi "no space left on device" "$BUILD_LOG"; then
    log "BUILD FAILED: docker ran OUT OF DISK mid-build."
    reclaim_hard
    log "retrying the build once on the freed disk (uncached parts rebuild from scratch)…"
    if ! run_app_build; then
      kill "$HB" 2>/dev/null || true; rm -f "$BUILD_LOG"
      log "BUILD FAILED AGAIN — the Docker vDisk is genuinely too small for the ClipAI image."
      log "  Unraid: Settings -> Docker -> stop the service -> increase the vDisk size, then re-run: bash update-all.sh"
      exit 1
    fi
  else
    kill "$HB" 2>/dev/null || true; rm -f "$BUILD_LOG"
    log "BUILD FAILED — error is above (the Companion cross-build needs outbound internet)"
    exit 1
  fi
fi
kill "$HB" 2>/dev/null || true
rm -f "$BUILD_LOG"
log "build complete."

log "restarting ClipAI (Ollama + models stay up)…"
$DC up -d --no-deps --force-recreate app || true
up=""
for _ in $(seq 1 45); do
  if [ "$(docker inspect -f '{{.State.Running}}' clipai-app 2>/dev/null)" = "true" ]; then up=1; break; fi
  $DC up -d --no-deps app >/dev/null 2>&1 || true
  sleep 2
done
[ -z "$up" ] && { log "app never came up — check: docker logs clipai-app"; exit 1; }

log "verifying the running container is the commit we just pulled…"
# The banner ("ClipAI build: <sha>") can lag a few seconds after start, and the
# baked /app/BUILD_INFO is authoritative regardless — check both, with a short
# wait. This is the proof-of-freshness that stops a silent stale run.
RUN_SHA=""
for _ in $(seq 1 15); do
  RUN_SHA="$(docker exec clipai-app sh -c 'head -1 /app/BUILD_INFO 2>/dev/null' 2>/dev/null | tr -d '[:space:]')"
  [ -z "$RUN_SHA" ] && RUN_SHA="$(docker logs --tail 400 clipai-app 2>&1 \
      | grep -m1 'ClipAI build:' | sed -E 's/.*ClipAI build:[[:space:]]*([0-9a-f]+).*/\1/')"
  [ -n "$RUN_SHA" ] && break
  sleep 2
done
if [ -n "$RUN_SHA" ] && [ "${RUN_SHA:0:7}" = "${BUILD_SHA:0:7}" ]; then
  log "UP TO DATE ✓ — container is running $BUILD_SHA (latest on $BRANCH)"
elif [ -n "$RUN_SHA" ]; then
  log "MISMATCH ✗ — container reports '$RUN_SHA' but latest is '$BUILD_SHA'."
  log "    A cached layer may have served stale code. Force a clean rebuild:"
  log "    CLIPAI_NOCACHE=1 bash update-all.sh"
else
  log "(could not read the container build id yet — check: docker exec clipai-app cat /app/BUILD_INFO)"
fi

log "publishing the fresh Companion installer…"
mkdir -p ./data/companion-cache
EXE="$(docker exec clipai-app sh -c 'ls -t /app/static/companion/*.exe 2>/dev/null | head -1' 2>/dev/null)"
if [ -n "$EXE" ]; then
  # Only clear the cache once we KNOW a fresh installer replaces it — a build
  # that produced no exe must never delete the one currently being served.
  rm -f ./data/companion-cache/*.exe ./data/companion-cache/manifest.json 2>/dev/null || true
  docker cp clipai-app:/app/static/companion/. ./data/companion-cache/ 2>/dev/null || true
fi

# The image can lack the installer (or carry a stale one) when BuildKit
# reused a previously FAILED fail-soft companion-builder layer: the whole
# build is CACHED in seconds and /out is empty — nothing ever retries it.
# Detect that here and re-run JUST the companion cross-build with a fresh
# cache-bust token, exporting the exe + manifest straight into the served
# ./data/companion-cache dir (no image rebuild, no app restart — the
# downloads router serves the newest installer across both locations).
_has_expected() { ls ./data/companion-cache/*"$EXPECTED"*.exe >/dev/null 2>&1; }
if [ -z "$EXE" ] || ! _has_expected; then
  ST="$(docker exec clipai-app sh -c 'cat /app/static/companion/BUILD_STATUS 2>/dev/null' 2>/dev/null)"
  if [ -z "$EXE" ]; then
    log "no Companion .exe in the image (builder status: ${ST:-unknown — likely a cached failed cross-build})."
  else
    log "image's Companion installer is not v$EXPECTED (builder status: ${ST:-unknown})."
  fi
  ensure_docker_space || log "…continuing the Companion retry anyway — it may fail on disk space (see above)."
  log "retrying the Companion cross-build directly (cache-busted; streams the real build log; 10-30 min cold, minutes when the caches are warm)…"
  ( while :; do sleep 60; log "…still cross-building the Companion ($(( ($(date +%s)-START)/60 ))m elapsed)"; done ) & HB2=$!
  if DOCKER_BUILDKIT=1 docker build --target companion-artifacts \
      --build-arg COMPANION_BUILD_FROM_SOURCE=1 \
      --build-arg COMPANION_BUILD_NUMBER="$BUILD_NUM" \
      --build-arg BUILD_SHA="$BUILD_SHA" \
      --build-arg COMPANION_REBUILD="$(date +%s)" \
      -f Dockerfile.gpu -o ./data/companion-cache .; then
    kill "$HB2" 2>/dev/null || true
    log "companion retry build finished — status: $(cat ./data/companion-cache/BUILD_STATUS 2>/dev/null || echo unknown)"
  else
    kill "$HB2" 2>/dev/null || true
    log "companion retry build FAILED — the real error is in the log above (it needs outbound internet for the MSVC SDK on a cold cache)."
  fi
fi

log "===== DONE — container @ $BUILD_SHA ====="
SERVED="$(ls -t ./data/companion-cache/*.exe 2>/dev/null | head -1)"
if [ -n "$SERVED" ]; then
  ls -lh "$SERVED" 2>/dev/null || true
  case "$SERVED" in
    *"$EXPECTED"*)
      log "OK: Companion v$EXPECTED is now SERVED by ClipAI." ;
      log "    On the 4070 PC: open the Companion app -> Update (or ClipAI -> Settings -> GPU Companion -> Download) to install it. That last step only runs on Windows." ;;
    *)
      log "WARN: served installer is NOT v$EXPECTED (got: $(basename "$SERVED")) — the retry above should say why; run CLIPAI_NOCACHE=1 bash update-all.sh to rule out every stale layer." ;;
  esac
else
  log "ERROR: no Companion .exe available at all — the from-source build failed even after the cache-busted retry (see its log above)."
fi
log "finished."
