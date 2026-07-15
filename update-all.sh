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
#   git fetch origin claude/transcription-coverage-loss-87p7ho && \
#     git reset --hard origin/claude/transcription-coverage-loss-87p7ho
#   nohup bash update-all.sh > /mnt/user/appdata/clipai-update.log 2>&1 &
#   tail -f /mnt/user/appdata/clipai-update.log
#
# The Companion cross-build (COMPANION_BUILD_FROM_SOURCE=1) needs outbound
# internet (~1 GB MSVC SDK the first time) and can take 10-30 min; the build
# cache makes later runs far faster. This script SERVES the new Companion from
# ClipAI — the final install is one "Update" click in the Companion app on the
# Windows GPU box (that step can't run from Unraid).
#
# Usage: bash update-all.sh [branch]
set -uo pipefail
cd "$(dirname "$0")"

BRANCH="${1:-claude/transcription-coverage-loss-87p7ho}"
START=$(date +%s)
log(){ local t=$(( $(date +%s) - START )); printf '\n[all] %02d:%02d %s\n' $((t/60)) $((t%60)) "$*"; }

if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi
export DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1

log "pulling latest ($BRANCH)…"
git fetch origin "$BRANCH"            || { log "git fetch FAILED"; exit 1; }
git reset --hard "origin/$BRANCH"     || { log "git reset FAILED"; exit 1; }
export BUILD_SHA="$(git rev-parse --short HEAD)"
export BUILD_SUBJECT="$(git log -1 --pretty=%s)"
EXPECTED="$(grep -m1 '^version' companion/src-tauri/Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
log "commit $BUILD_SHA — building container + Companion v$EXPECTED (from source; needs internet)"

( while :; do sleep 60; log "…still building ($(( ($(date +%s)-START)/60 ))m elapsed)"; done ) & HB=$!
if ! $DC build --progress=plain --build-arg COMPANION_BUILD_FROM_SOURCE=1 app; then
  kill "$HB" 2>/dev/null || true
  log "BUILD FAILED — error is above (the Companion cross-build needs outbound internet)"
  exit 1
fi
kill "$HB" 2>/dev/null || true
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

log "container is up — confirming the new build shipped:"
docker logs --tail 300 clipai-app 2>&1 | grep -m1 'ClipAI build:' || log "(build banner not in logs yet)"

log "publishing the fresh Companion installer…"
mkdir -p ./data/companion-cache
rm -f ./data/companion-cache/*.exe ./data/companion-cache/manifest.json 2>/dev/null || true
docker cp clipai-app:/app/static/companion/. ./data/companion-cache/ 2>/dev/null || true
EXE="$(docker exec clipai-app sh -c 'ls -t /app/static/companion/*.exe 2>/dev/null | head -1' 2>/dev/null)"

log "===== DONE — container @ $BUILD_SHA ====="
if [ -n "$EXE" ]; then
  docker exec clipai-app sh -c "ls -lh '$EXE'" 2>/dev/null || true
  case "$EXE" in
    *"$EXPECTED"*)
      log "OK: Companion v$EXPECTED is now SERVED by ClipAI." ;
      log "    On the 4070 PC: open the Companion app -> Update (or ClipAI -> Settings -> GPU Companion -> Download) to install it. That last step only runs on Windows." ;;
    *)
      log "WARN: served installer is NOT v$EXPECTED (got: $EXE)." ;;
  esac
else
  log "ERROR: no Companion .exe in the image — the from-source build produced none (see the build log above)."
fi
log "finished."
