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
# Checkout sanity: a reset interrupted by an earlier disk/FS problem can leave
# a tree that LOOKS reset but is missing directories — and the docker build
# then dies with a baffling '"/backend": not found'. Catch that here instead.
for _must in backend frontend companion Dockerfile.gpu docker-compose.yml; do
  if [ ! -e "$_must" ]; then
    log "checkout is INCOMPLETE — '$_must' is missing after git reset."
    log "  Repair with: git status && git reset --hard origin/$BRANCH  — then re-run: bash update-all.sh"
    exit 1
  fi
done
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

# Build with the output tee'd so RECOVERABLE docker failures are detectable:
#   - ENOSPC mid-build (preflight passed but the new layers outgrew the disk);
#   - corrupted BuildKit state, the classic AFTERMATH of an earlier disk-full
#     crash (the daemon died mid-write to its metadata db). Symptom: the build
#     dies with "failed to compute cache key: failed to calculate checksum of
#     ref …: '/backend': not found" for a path that plainly exists — a broken
#     cached context snapshot, not a missing directory.
# Both are cured the same way: purge the build cache (reclaim_hard) and retry
# ONCE, instead of leaving the box on old code.
_recoverable_build_failure(){  # $1 = build log file
  grep -qiE "no space left on device|failed to compute cache key|failed to calculate checksum|containerdmeta\.db|snapshot [^ ]+ does not exist" "$1"
}

# ── Watched build runner (stall detection) ──────────────────────────────────
# BuildKit is silent for minutes at a time during big downloads, so "no output"
# alone doesn't mean stuck. But a build whose log hasn't grown in STALL_MIN
# minutes IS hung, and the old heartbeat could not tell the difference: a real
# deploy sat for TWO HOURS on a stalled torch-wheel read while printing
# "…still building" every minute. (Root cause: PIP_DEFAULT_TIMEOUT=300 ×
# PIP_RETRIES=10 = up to 50 min of silence per stalled file; now lowered.)
#
# run_watched streams the command's output into the update log (so you still
# see everything live), reports a heartbeat that names the LAST line and how
# long it's been quiet, and kills a build that goes silent past the threshold.
# Returns: 0 ok · 1 failed · 2 stalled-and-killed.
STALL_MIN="${CLIPAI_STALL_MIN:-15}"
BUILD_LOG="$(mktemp /tmp/clipai-build.XXXXXX)"
BUILD_RC="$(mktemp /tmp/clipai-build-rc.XXXXXX)"

# Kill a stalled build AND everything it spawned. Killing only the direct
# child leaves orphans (the docker client's own children), and an orphaned
# client can hold a BuildKit session open that then collides with the retry.
# Walks the descendant tree depth-first via `pgrep -P` and signals children
# before parents, so nothing is reparented out of reach mid-kill. Deliberately
# NOT a process-group kill: resolving the group is unreliable here, and
# getting it wrong kills this very script.
_kill_descendants(){                 # $1 = pid, $2 = signal
  local pid="$1" sig="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    _kill_descendants "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}
_kill_tree(){                        # $1 = pid — polite, then final
  _kill_descendants "$1" TERM
  sleep 3
  _kill_descendants "$1" KILL
}

run_watched(){                       # "$@" = command to run
  : > "$BUILD_LOG"; : > "$BUILD_RC"
  ( "$@" > "$BUILD_LOG" 2>&1; echo $? > "$BUILD_RC" ) &
  local pid=$! tailpid last_size=0 last_change now size quiet hb
  tail -n +1 -f "$BUILD_LOG" 2>/dev/null & tailpid=$!
  last_change=$(date +%s); hb=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    size="$(stat -c %s "$BUILD_LOG" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    if [ "$size" != "$last_size" ]; then last_size="$size"; last_change="$now"; fi
    quiet=$(( (now - last_change) / 60 ))
    if [ $(( now - hb )) -ge 60 ]; then
      hb="$now"
      # Name what it is actually doing — a bare "still building" taught us nothing.
      local tailline
      tailline="$(grep -v '^[[:space:]]*$' "$BUILD_LOG" 2>/dev/null | tail -1 | cut -c1-110)"
      if [ "$quiet" -ge 3 ]; then
        log "…building ($(( (now-START)/60 ))m elapsed) — NO output for ${quiet}m (kills at ${STALL_MIN}m) — last: ${tailline:-<none>}"
      else
        log "…building ($(( (now-START)/60 ))m elapsed) — last: ${tailline:-<none>}"
      fi
    fi
    if [ "$quiet" -ge "$STALL_MIN" ]; then
      log "STALLED — no build output for ${quiet}m. Killing it (a hung download, not slow progress)."
      _kill_tree "$pid"
      kill "$tailpid" 2>/dev/null || true
      return 2
    fi
  done
  wait "$pid" 2>/dev/null || true
  sleep 1                              # let tail flush the final lines
  kill "$tailpid" 2>/dev/null || true
  return "$(cat "$BUILD_RC" 2>/dev/null || echo 1)"
}

# Build, retrying ONCE on anything recoverable: a stall (kill + retry with the
# caches warm, which is what actually clears a wedged CDN connection), an
# ENOSPC, or corrupted BuildKit state (purge the cache first).
build_attempt=1
while :; do
  rc=0
  run_watched $DC build $NOCACHE --build-arg COMPANION_BUILD_FROM_SOURCE=1 \
      --build-arg COMPANION_BUILD_NUMBER="$BUILD_NUM" app || rc=$?
  [ "$rc" = 0 ] && break

  if [ "$build_attempt" -ge 2 ]; then
    rm -f "$BUILD_LOG" "$BUILD_RC"
    log "BUILD FAILED AGAIN (attempt $build_attempt)."
    log "  Stalled twice? The box can't hold a stable connection to the CDNs (pypi / download.pytorch.org / crates.io). Check the network, then re-run: bash update-all.sh"
    log "  Still 'no space left on device'? The Docker vDisk is genuinely too small — Unraid: Settings -> Docker -> stop the service -> increase the vDisk size."
    log "  Still 'failed to compute cache key' / checksum errors? Docker's state db is damaged — restart the Docker service (Unraid: Settings -> Docker -> Enable Docker: No, Apply, then Yes) and re-run."
    exit 1
  fi

  if [ "$rc" = 2 ]; then
    log "retrying the build once — completed layers are cached, so it resumes rather than starting over…"
  elif _recoverable_build_failure "$BUILD_LOG"; then
    if grep -qi "no space left on device" "$BUILD_LOG"; then
      log "BUILD FAILED: docker ran OUT OF DISK mid-build."
    else
      log "BUILD FAILED: docker's BuildKit state looks CORRUPTED (an earlier disk-full crash damaged its cache — 'failed to compute cache key' on a path that exists)."
    fi
    reclaim_hard
    log "retrying the build once on the cleaned state (uncached parts rebuild from scratch)…"
  else
    rm -f "$BUILD_LOG" "$BUILD_RC"
    log "BUILD FAILED — error is above (the Companion cross-build needs outbound internet)"
    exit 1
  fi
  build_attempt=$((build_attempt + 1))
done
rm -f "$BUILD_RC"
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
  run_companion_build(){             # same watchdog: stalls are killed, not waited on
    run_watched docker build --target companion-artifacts \
        --build-arg COMPANION_BUILD_FROM_SOURCE=1 \
        --build-arg COMPANION_BUILD_NUMBER="$BUILD_NUM" \
        --build-arg BUILD_SHA="$BUILD_SHA" \
        --build-arg COMPANION_REBUILD="$(date +%s)" \
        -f Dockerfile.gpu -o ./data/companion-cache .
  }
  comp_rc=0; run_companion_build || comp_rc=$?
  if [ "$comp_rc" != 0 ]; then
    if [ "$comp_rc" = 2 ]; then
      log "companion retry STALLED and was killed — trying once more (caches stay warm)…"
    elif _recoverable_build_failure "$BUILD_LOG"; then
      log "companion retry hit a recoverable docker failure (disk / corrupted build cache) — reclaiming and trying once more…"
      reclaim_hard
    else
      log "companion retry build FAILED — the real error is in the log above (it needs outbound internet for the MSVC SDK on a cold cache)."
      comp_rc=99                     # unrecoverable: don't burn another 30 min
    fi
    [ "$comp_rc" != 99 ] && { comp_rc=0; run_companion_build || comp_rc=$?; }
  fi
  if [ "$comp_rc" = 0 ]; then
    log "companion retry build finished — status: $(cat ./data/companion-cache/BUILD_STATUS 2>/dev/null || echo unknown)"
  else
    log "companion retry build did not produce an installer — see the log above."
  fi
fi
rm -f "$BUILD_LOG"

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
