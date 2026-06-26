#!/usr/bin/env bash
#
# Redeploy ClipAI with the build identity baked in, so the startup log shows the
# real commit ("ClipAI build: <sha> — <subject>") instead of "unknown".
#
# The image can't read git at runtime (.git is excluded from the build context),
# so the SHA/subject must be passed as build args. docker-compose reads them from
# the BUILD_SHA / BUILD_SUBJECT environment variables — this script captures them
# from git and exports them before building, so a plain `docker compose build`
# never has to be remembered with the right flags.
#
# Usage:
#   ./deploy.sh [branch]
#     branch  — branch to fetch + deploy (default: the currently checked-out
#               branch, or "main" if detached).
#
set -euo pipefail
cd "$(dirname "$0")"

BRANCH="${1:-}"
if [ -z "$BRANCH" ]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
fi
if [ -z "$BRANCH" ] || [ "$BRANCH" = "HEAD" ]; then
  BRANCH="main"
fi

echo "==> Stopping containers"
docker compose down || true

echo "==> Fetching origin/$BRANCH"
git fetch origin "$BRANCH"
git reset --hard FETCH_HEAD

# Capture the build identity from the freshly-checked-out commit.
export BUILD_SHA="$(git rev-parse --short HEAD)"
export BUILD_SUBJECT="$(git log -1 --pretty=%s)"
echo "==> Building ClipAI @ ${BUILD_SHA} — ${BUILD_SUBJECT}"
docker compose build --no-cache

echo "==> Starting"
docker compose up -d
echo "==> Deployed. Startup log will show:  ClipAI build: ${BUILD_SHA} — ${BUILD_SUBJECT}"
