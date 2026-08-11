# Deploy — a rolled-back container can no longer hide behind "up to date"

"The companion app never sees the latest update." It wasn't the Companion: the
CONTAINER was being rebuilt from older code on every run, and the update card
reported that as good news.

The Companion's version IS the repo's commit count (`set_build_version.py`
stamps `0.11.<git rev-list --count>`). The deploy command fetched one branch
but hard-reset to a different, older one, landing on commit count 753 — while
the installed Companion, from an earlier correct build, was 760. The card then
said "Up to date", because there genuinely was nothing NEWER to install. It
also printed the LOCAL version in a sentence about what the server serves, so
the two numbers never appeared side by side.

Worse, it was self-perpetuating: that reset also overwrites `update-all.sh`
with the stale branch's copy, whose `DEFAULT_BRANCH` points back at the stale
branch — so every run re-pinned the old branch and the box could never move
forward. (Only the caller's command can break that loop, which is why the fix
below refuses loudly instead of pretending to repair the branch itself.)

- **`update-all.sh` refuses a rollback.** The commit count going DOWN means
  older code. State (`build number / branch / sha`) is recorded under `./data/`
  — gitignored, so `git reset --hard` can't erase the memory of what is
  deployed — and written only once the container is CONFIRMED running the new
  commit. A downgrade aborts before the build with the numbers, both branches,
  the likely cause, and the corrected command; `CLIPAI_ALLOW_ROLLBACK=1`
  overrides for a deliberate downgrade. Equal counts (re-deploying the same
  commit) and first-ever runs pass through untouched.
- **The Companion says so out loud** (0.11.11 → 0.11.12). `check_app_update`
  now returns `server_behind` when the served version is strictly older than
  the running one, and the GUI shows a red panel — on the quiet launch check
  too, so it announces itself without anyone clicking: "⚠ Your ClipAI server
  is serving an OLDER Companion (v0.11.753)… the CONTAINER is running older
  code." The "Up to date" line now quotes the SERVER's version and build id
  instead of the local one.
- Verified: the guard exercised end-to-end against this incident's real
  numbers — refuses 760 → 753 with the full explanation, passes 760 → 764,
  passes an equal-count re-deploy, passes a first run with no state, and
  proceeds under `CLIPAI_ALLOW_ROLLBACK=1`; new Rust test pinning
  `server_behind` as distinct from up-to-date (40 Rust tests), 9 version-sync
  tests, `tsc` clean.

---

# ClipAI — Concurrent Analyses in Settings + the Companion shows every job separately

Two follow-ups to the sequential-import fix:

- **Concurrent Analyses is now a Settings option** (Settings > Advanced >
  Concurrent Analyses, buttons 1–4, default "1 · sequential"). Backed by new
  `GET/POST /api/processing/settings` and a new resizable `AnalysisGate`
  replacing the fixed `asyncio.Semaphore`: the change applies **live, no
  container restart** — raising the limit admits queued jobs immediately;
  lowering it never interrupts running analyses (they finish, then the queue
  continues under the new cap). Persisted to `user_settings.json` (new
  `_PERSISTABLE_KEYS` entry) AND upserted into `.env`, so it survives both
  container restarts and image rebuilds. Values clamp to 1–8 server-side.
- **The GPU Companion now tracks and shows each ClipAI job separately**
  (Companion 0.11.10 → 0.11.11). The Windows app kept ONE `reported_job`
  slot, so when ClipAI ran two pipelines their heartbeats overwrote each
  other and the GUI flickered between jobs. It now keeps a per-job map
  (`reported_jobs`, stale entries evicted after the 300 s heartbeat window):
  - The dashboard renders **one card per running pipeline** — own title,
    stage, elapsed, progress bar, and its own **Force end** button ("Analyzing
    (1 of 2) — …"). Force-ending one job never wipes the other, and the GPU
    is only freed when NO job remains (previously ending one of two evicted
    the models out from under the survivor — both `end_active_job` and
    `force_end_everything`/`/v1/jobs/force-end` now account for every job).
  - "Pipeline activity" job logs keep per-job stage/progress even for
    heartbeat-only jobs (a pipeline whose stages so far all ran on the
    server now appears as its own entry instead of not at all), and one
    job's `X-ClipAI-Job-Ended` clears only that job.
  - Status payload gains `active_jobs` (oldest started first); `current_job`/
    `job_progress` stay for the single-job consumers and idle reapers.
- Note: the Companion GPU intentionally still runs ONE transcription at a
  time (`whisper_slot`); with 2 concurrent analyses the second pipeline's
  whisper call queues briefly — the cards make that visible instead of
  confusing.
- Verified: 6 new backend tests (`test_processing_settings.py`: endpoint
  read/write/clamp, live gate resize, raise-admits-immediately,
  lower-never-interrupts, persistence key) — 45 backend tests green across
  the touched suites; new e2e section 8 in `proxy_e2e.rs` (two concurrent
  heartbeats tracked separately, one job's end signal spares the other,
  suppression after end, per-id force end) — 40 Rust tests green; `tsc`
  clean; 159 frontend tests + production build green.

---

# ClipAI — Multi-video imports run strictly one pipeline at a time

A dashboard screenshot showed two analyses running side by side (6 % and
12 %) with two more queued behind them after a multi-select import. Two
causes, both fixed:

- **`CONCURRENT_ANALYSES` default 2 → 1.** The pipeline semaphore itself
  allowed two concurrent analyses, which split the same GPU/CPU and finish
  slower in total than back-to-back. Strictly sequential by default now;
  raise via env only on hardware with real headroom for two full pipelines.
- **The dialog's multi-select "Import N" now uses the sequential queue.**
  It previously imported file-by-file through the SINGLE-file endpoint,
  which fire-and-forgets `run_analysis` the moment each download lands — a
  few fast LAN downloads piled several pipelines up at once. Multi-selected
  VIDEOS now post the selection to `/import-folder` (new `files` mode:
  explicit `{name, path, size}` list, selection order preserved, non-videos
  dropped), so they get the same download → full analysis → next sequencing,
  the same persistence/revive resilience, and the same Dashboard progress
  panel as "Import all". Media/font multi-select keeps the per-file loop (no
  pipelines involved).
- Verified: 2 new endpoint tests (selection mode state/order/labeling, empty
  selection 400), a new component test proving the footer "Import 2" makes
  ONE `/import-folder` POST with the picked files and never touches the
  per-file `/import` endpoint; full suites green (23 bulk/bookmark backend,
  159 frontend). The 7 `test_flag_defaults_stable` failures are pre-existing
  missing-module issues, byte-identical on the base commit.

---

# ClipAI — Stalled runs get detected, revived, and finished (the bulk-import "randomly stopped")

Diagnosed from a real bulk import's logs: video 1 of 5 downloaded and
analyzed normally until 18:49:52, then the pipeline went **completely silent
for 13+ minutes** mid-face-loop — no progress, no error — while the job
looked alive (`analyzing_scenes`, fresh `updated_at`). Root causes and fixes,
from specific to systemic:

- **The hang itself: a wedged decoder on a corrupt file.** This video's audio
  stream was full of "Invalid data found when processing input" (the audio
  extraction retried around it). The face loop's frame-reader thread calls
  cv2 `read()`/`grab()`, which can block forever inside FFmpeg on corrupt
  packets — and the consumer sat in a blocking `q.get()` behind it. The
  consumer now times out (`REFRAMER_ACQUIRE_STALL_S`, default 120 s — real
  per-sample acquire cost is fractions of a second), abandons the wedged
  reader, and **finishes the analysis with the samples already collected**:
  degraded coverage beats a job that hangs forever.
- **The blindness: the heartbeat masked the wedge from the revive system.**
  The staleness-based revive keys off `heartbeat_at`/`updated_at` — and the
  pipeline heartbeat keeps stamping those while a run is wedged, so
  "alive but making zero progress" was undetectable *by design*. New
  **progress-based stall watchdog** in the heartbeat: no REAL progress
  emit for `PIPELINE_STALL_MINUTES` (default 60 — generous because a
  40-minute CPU Whisper pass between writes is legitimate; 0 disables) →
  the run is presumed wedged, the job is requeued with the stall named in
  its status, and the run TASK is cancelled — which releases the pipeline
  semaphore even though the wedged native thread can never be killed —
  then **resumed from its checkpoint** (same attempt cap + kill switch as
  startup auto-resume: `CLIPAI_MAX_RESUME_ATTEMPTS` / `CLIPAI_AUTO_RESUME`).
  Attempts exhausted → FAILED with an honest "stage stalled" message, never
  an eternal spinner. Subtlety that mattered: the pipeline's `CancelledError`
  is a *custom* Exception for user cancels — the watchdog's task-cancel
  raises `asyncio.CancelledError` (a BaseException), handled in its own
  branch so a stall-revive can't masquerade as "Cancelled by user" and a
  real shutdown cancellation still propagates.
- **The bulk queue now follows revives and survives restarts.** The
  sequential runner no longer trusts one `run_analysis` await: it follows
  each video's job to a TERMINAL status (`_bulk_wait_terminal`), so a
  stall-revived job's requeue isn't misread as failure. And the whole queue
  state is persisted to disk on every transition
  (`companion_bulk_imports.json`); startup recovery reloads it after the
  per-job auto-resume: terminal items keep their outcomes, a mid-download
  item re-downloads fresh, a mid-analysis item is adopted and followed to
  its revived outcome, and the remaining videos import one at a time as
  before. The Dashboard panel re-attaches automatically via `/active`.
- Verified: 7 new watchdog tests (fires once with the stage after real
  silence, touch resets it, 0 disables, requeue-with-attempts / fail-at-cap /
  auto-resume-off / no-task decisions), 3 new bulk-resilience tests
  (follows a revived job, persist + restart resume with mid-download reset,
  resumed runner adopts analyzing + skips terminal + imports the tail);
  full companion/bulk/archive/reconcile/checkpoint regression batch green
  (91 passed).

---

# ClipAI — Companion importer feels local; bulk import on the Dashboard; multi-select transcripts

Three user-visible fixes/features from a real session's screenshots.

**1. The import dialog now behaves like a local file browser.** The reported
breakage — breadcrumb reading `Shared > ? > C:` and "path is not inside a
shared folder (403)" when going up a folder — was Windows "verbatim" path
prefixes (`\\?\C:\…`) leaking out of the Companion's canonicalized listings,
plus two breadcrumb-builder bugs (roots ending in a separator never matched
their children; a bare `C:` drive crumb resolves drive-relative on Windows
and lands outside every share). Fixed at every layer:

- Companion (v0.11.10): `strip_verbatim` cleans every path `/v1/files/list`
  returns (`\\?\C:\…` → `C:\…`, `\\?\UNC\srv\…` → `\\srv\…`).
- Frontend: normalizes verbatim prefixes at every edge (listings, roots,
  bookmarks, pasted input) so it's fixed even with an older Companion
  installed; breadcrumbs rebuilt (root-anchored, separator-suffixed roots,
  case-insensitive on Windows, drive crumbs always `C:\`).
- Backend: bookmark paths normalized on read/write/delete — legacy
  `\\?\`-prefixed bookmarks display clean and can finally be un-starred.
- New affordances: an **Up** button (goes to the parent; from a shared root,
  back to the Shared view — never outside the share), pasted paths accept
  quotes (Windows "Copy as path") and **file** paths (opens the parent
  folder with the file selected + scrolled into view, ready to Import),
  listings are cached so revisits render instantly while refreshing in the
  background, the previous listing stays (dimmed) during navigation instead
  of flashing a skeleton, and the 403 message explains itself.

**2. Bulk import status lives on the Dashboard now, not just in the popup.**
The progress panel was extracted into a shared, self-discovering
`BulkImportPanel` (finds a running import via `/import-folder/active`, polls,
cancel/dismiss) mounted on the Dashboard AND inside the import dialog. Close
the popup, navigate away, or open ClipAI from another device — the running
import is right there, each item linking to its `/analysis/<job_id>` page.

**3. Multi-select transcript download.** The Dashboard's existing
multi-select gains "Transcripts (SRT)" and "Transcripts (TXT)": one ZIP with
a `.srt`/`.txt` per selected video (translated track preferred, same
sanitize + fps treatment as the single-file download; duplicates deduped
`clip (2).srt`; transcript-less videos listed in `_skipped.txt` instead of
failing the batch) via new `POST /api/jobs/transcripts/archive`.

- Verified: 6 new archive tests, 2 new bookmark-normalization tests (18
  total in that suite), 9 CompanionBrowser tests including verbatim
  breadcrumbs / Up walking / quoted-file-path selection; full frontend suite
  158 passed + build clean; companion `strip_verbatim` unit test + proxy e2e
  green; version-sync suite green on 0.11.10.

---

# ClipAI — Build: `sharing=locked` cache mounts wedge forever; use `private`

With the stall watchdog in place the failure finally became legible: two
consecutive attempts each sat exactly 15 minutes and printed **not one line**
from the step. The pip step's first command is `pip3 install --upgrade pip`,
which prints within seconds — silence means the step never began executing.
Both stalling steps (`stage-4` torch install, `companion-builder`) mount
caches with `sharing=locked`, which makes a build WAIT for a mount another
session holds. Killing a docker *client* does not release the server-side
BuildKit lease, so the orphaned sessions left by the earlier concurrent-run
pile-up held those mounts and wedged every subsequent build — silently, with
no output to diagnose from.

- All `sharing=locked` cache mounts in both Dockerfiles are now
  `sharing=private`: a build that finds the cache in use gets its own
  instance instead of blocking. No wedge, and no apt-cache corruption either
  (which is what `locked` was there to prevent). Cost is only a cold cache
  for a concurrent build — and the update lock makes those rare anyway.
- This also explains the earlier misdiagnoses. It was never a slow CDN; the
  pip timeout values were a real latent hazard but not the cause, and the
  concurrency lock removed the *source* of the orphaned sessions without
  freeing the leases they already held.
- Verified: no non-comment `sharing=locked` remains, every `--mount`
  continuation is intact, and both Dockerfiles keep all other cache mounts
  (cargo registry, target dir, xwin SDK, npm) untouched so warm-cache rebuild
  times are unchanged.

---

# ClipAI — Update flow: refuse concurrent runs (the real cause of the 2h hang)

`ps` on the deployment box showed **four** live `update-all.sh` runs (09:06,
08:51, 06:28, 06:19). That is the actual root cause of the "two-hour stall"
previously blamed on a slow CDN: the companion-builder stage mounts its apt
caches with `sharing=locked`, so a second concurrent build **blocks forever,
silently**, waiting for a mount the first one holds. All four also wrote to
the same log file, which is why the output read as one incoherent stream and
why a stale run's old-format heartbeat appeared inside a fresh run's log.

- **Single-instance lock.** `update-all.sh` now takes an exclusive `flock`
  (`/tmp/clipai-update.lock`, override with `CLIPAI_LOCK_FILE`) and refuses to
  start when another update holds it, naming the holder's PID so you know
  exactly what to kill. The lock is released automatically on exit — including
  a crash — because it lives on an open fd, not a file that has to be cleaned
  up. Where `flock` is unavailable it falls back to a PID file with a liveness
  check, so a crashed run leaves a stale lock that the next update clears
  instead of being blocked by it forever.
- Subtle bug caught while testing: opening the lock with `9>` truncates the
  file, blanking the holder's PID before it can be read — the refusal could
  only ever say "pid unknown", which is the single fact the operator needs.
  Now opened `9<>` (no truncate) and rewritten only after the lock is held.
- The pip timeout change (60 × 5) and the stall watchdog from the previous
  entry both stand — 300 × 10 was a genuine latent hazard, and the watchdog
  would have caught this deadlock too (15 min of silence → kill → retry)
  instead of letting it run for two hours. They were just not the root cause.
- Verified: a second run while one holds the lock is refused with exit 1 and
  the holder's real PID; the lock frees once the holder exits; the no-flock
  fallback clears a dead holder's stale lock but still refuses a live one; and
  a full end-to-end dry run is unaffected by the lock.

---

# ClipAI — Update flow: detect a stalled build instead of waiting forever

A deploy sat for **two hours** on a hung torch-wheel download while the
heartbeat printed a cheerful "…still building (121m elapsed)" every minute.
Two defects, both fixed:

- **The hang.** `PIP_DEFAULT_TIMEOUT=300` × `PIP_RETRIES=10` meant a single
  STALLED read could sit silent for up to 50 minutes *per file*. A stalled
  socket isn't cured by waiting on it — it's cured by giving up and
  reconnecting — so both Dockerfiles now use `60` × `5`. Slow-but-moving
  downloads are unaffected: the timeout is a per-read no-data window, not a
  cap on transfer time.
- **The blindness.** `update-all.sh` had no way to tell "silent because
  BuildKit batches output" from "silent because it's wedged". New
  `run_watched` runner:
  * streams build output live (unchanged) while tracking log growth;
  * the heartbeat now names the **last line** and, once output has been quiet
    3+ minutes, says how long and when it will kill — so a stall is visible
    within minutes instead of never;
  * kills a build with no output for `CLIPAI_STALL_MIN` (default 15) minutes
    and retries once with the caches warm (which is what actually clears a
    wedged CDN connection); a second stall exits with the network diagnosis
    rather than hanging;
  * the kill walks the descendant tree depth-first (`pgrep -P`, children
    before parents) so no orphaned docker client survives to collide with the
    retry. Deliberately not a process-group kill — resolving the group proved
    unreliable, and getting it wrong kills the update script itself.
  The Companion cross-build retry uses the same watchdog.
- Verified with a scripted harness driving the real runner: a 90-second build
  printing every 3s survives a 1-minute stall threshold (proving it measures
  SILENCE, not runtime); a build that prints once then hangs is killed at the
  threshold with its grandchild confirmed dead by exact PID; clean and failing
  builds still return the right codes; and full-script runs cover
  stall→kill→retry→successful deploy and stall-twice→exit-with-guidance.

---

# ClipAI — Update flow: recover from BuildKit cache corruption

The deploy after the disk-full episode failed differently: `failed to compute
cache key: failed to calculate checksum of ref …: "/backend": not found` —
for a directory that plainly exists. That's the classic aftermath of the
earlier ENOSPC crash: the Docker daemon died mid-write to BuildKit's metadata
db, leaving a corrupted cached context snapshot that every later build trips
over. `update-all.sh` now:

- **Classifies recoverable build failures** beyond ENOSPC: "failed to compute
  cache key", "failed to calculate checksum", containerdmeta.db errors, and
  missing-snapshot errors all trigger the same self-heal — purge the build
  cache (`reclaim_hard`) and retry the build ONCE on clean state. The failure
  headline says which case it was, and the second-failure message gives the
  matching manual fix (grow the vDisk vs restart the Docker service).
- **The Companion cross-build retry** gets the same tee + classify + reclaim
  + one-more-try treatment.
- **Checkout sanity check** after `git reset`: if backend/, frontend/,
  companion/, Dockerfile.gpu or docker-compose.yml is missing (a reset
  interrupted by an earlier disk/FS problem can leave a tree that LOOKS reset
  but isn't), abort immediately with the repair command instead of letting
  docker report a baffling `"/backend": not found`.
- Verified with the stubbed dry-run harness: corrupted-cache signature →
  headline names corruption → `builder prune -af` → retry → successful
  deploy; and an incomplete checkout aborting before any build attempt.

---

# ClipAI — Companion 0.11.9: remote performance & caption-quality control

The GPU Companion's "Performance" knob — the one control on its desktop GUI
that sets the Ollama speed profile (pipeline parallelism) AND the Whisper
transcription quality (beam search + model) together — can now be driven from
ClipAI's Settings, so the user never has to walk to the GPU PC to change it.

- **Companion (v0.11.9):** new authed `GET/POST /v1/config/quality` proxy
  route, mirroring `/v1/config/vram`. GET returns the two config fields plus
  what they RESOLVE to on that GPU right now (effective whisper model, beam
  size, parallelism, budget). POST validates against the same allow-lists as
  the local GUI (`auto|eco|balanced|turbo`, `auto|fast|balanced|max`) —
  invalid remote values 400 loudly instead of the local path's silent ignore,
  and validation happens before anything applies so a bad field can't
  half-apply. Side effects match the local GUI exactly: a speed change
  restarts the managed Ollama (new NUM_PARALLEL / MAX_LOADED), a quality
  change drops the whisper sidecar so the next transcription starts with the
  new decode.
- **ClipAI backend:** `GET/POST /api/providers/companion/quality` proxies it,
  with value validation before anything leaves the container, a read-only
  `/v1/health` fallback for pre-0.11.9 Companions ("update the app to change
  quality remotely"), and the usual paired/unreachable error shapes.
- **Settings UI:** a "Companion performance & caption quality" card next to
  the VRAM card in the GPU Companion section — the same four levels as the
  desktop GUI (Auto / Eco / Balanced / Turbo, with the same caption-accuracy
  hover hints), a live "2× parallel · large-v3 beam 5" effective line, a
  caption-quality-only override select, and a custom-pairing notice when the
  two fields don't match a level. Hidden when no Companion is connected;
  read-only with an update notice for old Companions.
- Version discipline: 0.11.9 across Cargo.toml/lock, tauri.conf.json,
  package.json(+lock), RELEASE (fires the installer release workflow), and
  `EXPECTED_COMPANION_VERSION` (so ClipAI's version handshake nudges stale
  installs to update).
- Verified: `cargo check` clean and the proxy e2e suite green with a new
  section covering the quality route (401 unauthed, defaults read, normalized
  write landing in persisted config, whisper-only change not restarting
  Ollama, invalid profile 400 with no half-apply, works while sharing is
  paused); 9 new backend proxy tests (passthrough, health fallback,
  normalization, validation-before-send, 404→update message, no-companion);
  version-sync suite green on 0.11.9; frontend build + full suite 154 passed.

---

# ClipAI — Update flow: survive a full Docker vDisk

The very next deploy attempt died before the build even started:
`write /var/lib/docker/btrfs/subvolumes/…: no space left on device` — on
Unraid the Docker vDisk (docker.img) is a fixed-size loopback, and the ClipAI
image + superseded builds + BuildKit caches fill it over time. `update-all.sh`
now:

- **Preflights Docker's data root** (`docker info -f '{{.DockerRootDir}}'`)
  before building: below `CLIPAI_MIN_FREE_GB` (default 8) it reclaims in safe
  escalating steps — dangling images first (untagged layers from previous
  clipai-app builds), then an LRU BuildKit-cache trim to
  `CLIPAI_BUILDCACHE_KEEP_GB` (default 6, so the hot companion cross-build
  caches survive). It never auto-prunes containers, volumes, or other apps'
  tagged images; if that still isn't enough it ABORTS immediately with the
  exact Unraid fix (grow the vDisk) instead of failing 20 minutes in.
- **Detects ENOSPC mid-build** (build output tee'd + grepped): reclaims hard
  (`docker builder prune -af` + dangling images) and retries the build ONCE,
  so a marginal disk recovers unattended instead of leaving the box on old
  code. A second failure aborts with the grow-the-vDisk instruction.
- The Companion cross-build retry path gets the same preflight (best-effort).
- Verified with scripted dry-runs over stubbed docker/git/df: low-disk
  preflight (5 → 8 GB via the two safe prunes), mid-build ENOSPC → hard
  reclaim → successful retry → normal publish, and the hopeless case (1 GB,
  prunes reclaim ~nothing) aborting BEFORE any build attempt.

---

# ClipAI — Update flow: stop caching a failed Companion cross-build forever

A real deploy hit "ERROR: no Companion .exe in the image" on a 5-second fully
CACHED build. Root cause chain: the fail-soft `companion-builder` stage bakes
an EMPTY `/out` when the cross-build fails, BuildKit then reuses that cached
empty layer on every rebuild of the same commit (nothing in the cache key
changes), and the publish step *deleted the currently-served installer* from
`data/companion-cache` before discovering there was nothing to replace it
with. Fixes:

- **`update-all.sh` auto-retry.** When the built image has no installer (or a
  stale version), the script now prints the baked `BUILD_STATUS` reason and
  automatically re-runs JUST the companion cross-build (`companion-artifacts`
  target) with a fresh `COMPANION_REBUILD` cache-bust token, exporting the
  exe + manifest straight into the served `./data/companion-cache` dir — no
  image rebuild, no second app restart, and the real cross-build error
  finally streams into the update log instead of hiding behind "CACHED".
- **Never wipe the served installer on a failed build.** The cache dir is now
  cleared only after confirming the image actually contains a fresh exe.
- **`COMPANION_REBUILD` build arg** added to the `companion-builder` stage in
  both Dockerfiles (cache-bust token; default 0 keeps normal builds fully
  cached), and the stage now writes `/out/BUILD_STATUS`
  (`ok:`/`failed:`/`skipped:` + reason) so a missing exe is diagnosable.
- **`DEFAULT_BRANCH` updated** to `claude/clipai-bookmarks-bulk-upload-0nsv1p`
  — the exact trap the script's own header warns about bit again: the caller
  reset to the new work branch, then the stale default quietly reset BACK to
  the old branch and rebuilt old code (the log even said "UP TO DATE ✓" on
  the wrong branch).
- Verified: both Dockerfiles' edited RUN blocks executed standalone in skip
  mode (BUILD_STATUS written, token echoed, continuation structure intact),
  plus full dry-runs of `update-all.sh` with stubbed docker/git covering the
  missing-exe → retry → publish path (old installer preserved, correct build
  args) and the happy path (no retry, normal wipe + publish).

---

# ClipAI — Companion importer: path bookmarks + sequential bulk folder import

Two quality-of-life features for the "Import a file" dialog that browses a
paired GPU Companion's shared folders (Upload page and Media Library).

- **Bookmark / star paths.** Folders can now be starred from the file list
  (list rows, grid cards, and the breadcrumb bar's star for the folder you're
  in). Starred paths appear in a "Bookmarks" section at the top of the Shared
  home view — one click jumps straight to a deep path like
  `D:\media\shows\S2`. Bookmarks persist **server-side** per Companion
  (`companion_bookmarks.json` in the data dir, keyed by `host_id`, capped at
  100, newest first), so they survive container restarts and follow the user
  across browsers/devices, unlike localStorage. New endpoints:
  `GET/POST/DELETE /api/providers/companion-files/bookmarks`.
- **Bulk folder import — one video at a time.** Folder rows (and the toolbar,
  for the folder being viewed) gain an "Import all" action for video imports.
  After a confirm step that names the exact count, ClipAI runs the folder
  SEQUENTIALLY: download video N over the LAN (same parallel-Range fast path
  as single imports, with progress heartbeats to the Companion GUI), create
  the job with the dialog's language picks, and **await the full analysis
  pipeline** (transcription → translation → clips) before video N+1 even
  starts downloading. Before each video a free-space gate checks the /data
  volume (file size × 1.5 + a 2 GB floor); when the ClipAI device runs out
  of space the run stops honestly — that video is marked "no space", the
  rest "skipped", and the panel says so. One failed video is recorded and
  skipped over; the batch keeps going. Cancel aborts an in-flight download
  immediately and signals the currently-analyzing job.
  New endpoints: `POST /api/providers/companion-files/import-folder`, plus
  `/progress`, `/cancel`, and `/active`. The run lives on the server: closing
  the dialog doesn't stop it, and reopening re-attaches to the live progress
  panel via `/active`. Only one bulk run at a time (409 otherwise) — that's
  the sequential contract.
- No Companion app changes needed: both features ride the existing
  `/v1/files/roots|list|read` share endpoints.
- Verified: 16 new backend tests (bookmark CRUD/persistence/cap, strict
  one-at-a-time ordering, out-of-space stop, failed-video continuation,
  cancel mid-download and mid-analysis, progress/active endpoints) and 5 new
  CompanionBrowser component tests; full frontend suite 154 passed, settings
  router suites green (the 2 pre-existing `test_companion_register` failures
  reproduce identically on the base commit).

---

# ClipAI — Profiled-run fixes: kill the u2netp tax, grammar-lock translation, canonical names

Driven by a real 24:27 anime run (640x360, GTX 1650 + 12 GB Companion) that
took 44.5 min on the PREVIOUS build: reframer_analysis 926 s (its own
instrumentation: acquire 36 s, detect 223 s, **other 612 s**), translation
26.5 min (gemma3:12b, batch 20, single slot, **12 of 35 batches parse-missed**
into split-and-retry cascades), everything else already fast (remote Whisper
transcribed the full episode in 34 s; extraction 42 s; summary 6.5 s;
clips 56 s).

- **Face-loop "other" tax dissected and removed.** The 612 s was ~75-90 %
  u2netp CPU saliency forwards on faceless anime samples. New: static-scene
  skip (unchanged frames carry every per-sample output forward —
  `REFRAMER_STATIC_SKIP`), u2netp mask stride-2 with carry-forward on
  faceless runs (never crosses a scene cut, absorbed by the hotspot EMA +
  planner smoothing), an LK static-gap shortcut, and a tiled-YuNet
  empty-streak throttle (no-op below 960 px det width). The Sample-loop
  timing line now breaks "other" into lk/saliency/motion/bookkeeping
  sub-buckets plus skip/reuse counters, so the next run proves the split.
  Expected on the profiled run: **~250-400 s off the 870 s loop**.
- **Translation grammar-locked and budget-parallelized.**
  `TRANSLATION_STRUCTURED_OUTPUTS` constrains the Ollama decode to a JSON
  array of exactly the batch's line count (parse-miss → ~0; old servers
  auto-downgrade to `format=json`). Large models get a second in-flight
  batch when the Companion's advertised `vram_budget_gb` minus estimated
  weights leaves ≥ `TRANSLATION_LARGE_PARALLEL_HEADROOM_GB` (Ollama
  continuous batching: ~1.5-1.8× throughput, identical outputs; the
  measured rig: 9.5 − 7.3 = 2.2 GB → 2 slots). Expected:
  **26.5 min → ~8-12 min** at unchanged quality.
- **Latent bug fixed:** `AIOrchestrator.text_completion` didn't accept
  `json_mode`, so the batched untranslated-cue cleanup prefill (added to fix
  the ~13-min recovery tail) had been raising TypeError into its
  except-block and **silently never ran**. The parameter now exists and
  forwards (with `json_schema`) to Ollama.
- **Canonical names** (`services/canonical_names.py`): the auto glossary
  mines Whisper's romaji and used to lock WRONG spellings in ("Ririna",
  "Zex", "Hero Yuu", "Trott", "Katō"). One title-anchored LLM call now maps
  mined terms to official romanizations (Relena Darlian, Zechs, Heero Yuy,
  Trowa, Quatre) and the glossary renders "mined → canonical" so the whole
  episode uses official-subtitle names. Defensive parse (no invented names,
  identity/sentence/profane/collision values dropped), fail-soft, per-job
  cached; user custom vocabulary always wins. 24 new tests.
- **Glossary-echo guard:** the run shipped a comma-joined dump of the
  glossary as a subtitle cue (the model "translating" hallucinated
  music-section source). Cues that are ≥60 % glossary terms now revert to
  the source line and flow into the per-cue cleanup.
- Verified: translation/glossary/canonical suites green (42 + 46);
  reframer sweep 102 passed with the failure set byte-identical to main's
  (17 pre-existing scipy/env failures, none introduced).

---

# ClipAI — Speed audit Tier 1+2: overlap the LLM chain with clips, window the timing pass, un-block COMPLETE

Implements the full speed audit (docs/perf-audit-46min-to-15-20.md): a 24-min
video that took ~46 min should now land ~15-20 min on a Companion rig and
~20-26 min local-only, with deliverables unchanged. Every overlap runs the
same calls with the same inputs, just earlier; small shared local cards keep
the proven serial order automatically. All gates are config-flippable.

- **Candidate-clip stream copy ON by default** (`CLIP_EXPORT_STREAM_COPY=True`).
  Candidate clips are review artifacts (final exports re-read the source); the
  default-config re-encode tail (~45-135 s/clip × dozens) drops to seconds and
  the previews become bit-identical to the source. Preview starts snap to the
  previous keyframe; set False for frame-accurate starts.
- **Translate+polish → summary chain runs CONCURRENT with clip detection** on a
  remote ≥7 GB Ollama host (`PIPELINE_OVERLAP_TRANSLATION_CLIPS`). Clip
  detection reads only the RAW perception transcript (verified false
  dependency); the chain joins before the clip-dependent followups. The
  summary∥clips gate now also recognizes a high-VRAM remote host when
  editorial is "local", and the pre-clip editorial unload is skipped there
  (it only existed for the shared 4 GB card and forced a 60-150 s cold reload
  before Auto-SEO).
- **Hybrid word-timing reference is windowed** (`HYBRID_WHISPER_REF_WINDOWED`).
  The second Whisper-EN pass (text discarded, only `.words` kept) now decodes
  only `clip_timestamps` windows around cues the readability enforcer could
  actually split (~10-20 % of the audio) instead of the whole video — the
  audit's single biggest translation-path cost (~8-10 min on a 24-min
  source). Cues outside the windows keep tier-B timing, which only ever
  renders when a cue splits. Long videos that used to skip the pass entirely
  (full pass would blow the timeout) now afford the windowed one — more
  tier-A timing than before, not less.
- **Local Whisper overlaps the face loop on roomy local cards**
  (`WHISPER_LOCAL_CONCURRENT_WITH_FACES`, ≥6 GB free): the same transcribe()
  call the remote-Companion overlap already runs early, extended to local
  CUDA engines. CPU-selected engines defer to the sequential pass (they'd
  fight the face loop for cores). 4 GB cards keep the handoff order.
- **Auto-SEO runs AFTER the COMPLETE save** (`SEO_AFTER_COMPLETE`): the
  per-clip/per-platform LLM copy no longer holds the job out of COMPLETE for
  minutes; clips are final and target-language-captioned at COMPLETE and the
  SEO copy fills in via the existing background_task/clips_refreshed events,
  re-persisted with a deleted-job guard.
- **Pre-translation source polish default OFF**
  (`TRANSLATION_POLISH_SOURCE_FIRST=False`): one full-transcript LLM pass
  (3-8 min) whose only beneficiary is the cosmetics of the shipped SOURCE
  track — the translator handles raw colloquial input (its documented
  strength) and the MT post-edit still repairs the translated track against
  the source lines. Set True to buy the old behavior back.
- **Translation redundancy removed**: translate_via_llm's internal 3-pass
  completeness loop drops to 1 pass on the pipeline path
  (`TRANSLATION_INTERNAL_RECOVERY_PASSES`; the bilingual-SRT download path
  keeps 3 — it has no downstream net), since `_llm_cleanup_untranslated` is
  guaranteed to run and is strictly more robust. The cleanup's batched wave
  and per-cue recovery widths now follow the Companion's advertised
  `num_parallel` (same signal as the main translation fan-out); the duplicate
  `TRANSLATION_LLM_CLEANUP_CONCURRENCY` config field is gone.
- **No more mid-run model evictions**: Ollama TEXT calls now carry
  `keep_alive` (`OLLAMA_TEXT_KEEP_ALIVE=30m`; explicit keep_alive=0 handoffs
  untouched) — the 5-min server default was evicting the translation model
  across longer stage gaps. Companion: managed keep_alive 10m→30m, idle
  whole-GPU reaper + job-freshness window 45 s→300 s, so an AI-quiet clip
  encode (up to 180 s) can no longer trigger a full eviction mid-job.
- **Remote-host warmup no longer blocks the pipeline start**: the 60 s warmup
  + 15 s unload + 2 s sleep dance runs in the background when the primary
  Ollama host is remote (it only ever protected the local card's VRAM).
- **Summary map-reduce**: chunk concurrency now follows the remote-VRAM
  ladder the vision path already uses (2-3 on ≥7/≥10 GB hosts, 6 on cloud,
  serial on the local card) — chunks are disjoint spans, outputs identical.
  Bug fix: the reduce input was head-truncated at 2500 chars, silently
  dropping the back half of long videos; it now stride-samples chunk
  summaries evenly across the whole timeline under the same cap.
- **Batch export parallelized** (user-facing): up to
  min(CLIP_EXPORT_CONCURRENCY, 3) concurrent exports, per-clip error
  isolation and result ordering preserved.
- `.env.example` documents the previously-hidden speed knobs
  (CLIP_EXPORT_STREAM_COPY/CONCURRENCY, REFRAMER_MAX_SAMPLES,
  REFRAMER_MIN_SAMPLE_FPS, SUMMARY_MAX_CHUNKS, the new overlap gates).
- Verified: full suite 2011 passed; the failure set is byte-identical to
  main's (122 pre-existing environment failures, none introduced).
  Companion: cargo check + 32 unit tests pass.

---

# ClipAI — Remote desktop GPU sharing: multi-Ollama failover, remote Whisper, GPU Companion

Turns the 4 GB GTX 1650 bottleneck inside out: the AI stages (Whisper, VLM,
editorial/translation LLMs) can now run on a bigger GPU elsewhere on the LAN
(e.g. a Windows desktop's RTX 4070), while FFmpeg NVENC/NVDEC stays on the
Unraid card where it belongs. Remote hosts only ever receive prompts, frames,
and extracted audio — never source video. With nothing configured, behavior
is identical to before (the full pre-existing test suite passes unchanged).

- **Multi-host Ollama registry** (`services/ollama_registry.py`, new setting
  `OLLAMA_HOSTS`): a JSON array of hosts where ARRAY ORDER IS PRIORITY —
  index 0 primary, the rest ordered fallbacks. Per-host bearer tokens, URL
  base-path support (`…:11500/ollama`), ~10 s probe cache, ~30 s unhealthy
  cooldown, 503 Retry-After handling, and a model-substitution ladder when a
  host lacks the configured model. Provider/orchestrator/translator call
  sites fail over mid-job; after a full pass the existing AI_FALLBACK_CHAIN
  takes over. `OLLAMA_HOST` stays synced to the primary for back-compat.
  Settings UI: drag-and-drop "Ollama Hosts" card (@dnd-kit) with add/test/
  edit/toggle/delete; all probing server-side.
- **Remote Whisper** (`WHISPER_REMOTE_URL/_API_KEY/_MODEL`): transcription
  POSTs the already-extracted WAV to any OpenAI-compatible
  `/v1/audio/transcriptions` server (verbose_json + word timestamps) and
  maps the response through the same post chain as cloud STT — downstream
  schema byte-identical. Selection: healthy remote → local CUDA ladder →
  CPU; mid-stage failure falls back locally. The 4 GB VRAM serialization
  dance (`_release_whisper_vram`) is skipped when the local GPU never
  loaded. New `JobResult.stage_locations` + Compute-card badges show where
  each stage ran; Ollama host names ride `provider_used`.
- **GPU Companion** (`companion/`, Tauri 2, Windows + macOS): tray/dashboard
  app that manages Ollama (winget/brew install, localhost-only, keep-alive +
  `OLLAMA_GPU_OVERHEAD` from a soft VRAM-budget slider), lazy-starts a
  Whisper sidecar (faster-whisper on Windows/NVIDIA, whisper.cpp Metal on
  macOS, idle auto-shutdown), and exposes one authenticated LAN port
  (11500) with a live "what is ClipAI running on my GPU" activity feed.
  Pairing endpoint `POST /api/settings/companion-register` (API-key
  protected) registers the Companion as primary + remote Whisper in one
  step. CI workflow builds NSIS `.exe`/`.msi` + universal `.dmg` on
  `companion-v*` tags with a sha256 manifest.
- **In-container distribution** (`routers/downloads.py`, Dockerfile
  `companion-fetch` stage, Settings "GPU Companion" card): installers are
  baked into the image (build-arg `COMPANION_VERSION`, never fails on fresh
  forks), refreshable at runtime into `/config/companion-cache`
  (sha256-verified), served with a hosted-vs-GitHub indicator, and fall
  back to a GitHub Releases redirect.
- Docs: `docs/remote-gpu.md` (one-click + manual paths, VRAM-tier model
  table, troubleshooting); `.env.example` updated; 55 new tests including
  real-socket failover/whisper contract fixtures.

---

# ClipAI — GPU-first LLM stages: fit translation to VRAM, trim summary chunks, batch the recovery loop

Follow-up to the quality-neutral speed pass, targeting the stages that actually
dominate a long-video run on a small GPU — all Ollama/LLM-bound and untouched
by the earlier I/O work. Diagnosed from a 128-min run on a 4 GB GTX 1650 where
the perception stage was fine (~25 min, slightly faster than before) but
translation (~29 min), summary (~19 min) and the untranslated-cue recovery
(~13 min of that) dominated. Root cause of the translation cost: the configured
translation model `qwen3:4b-…-q4_K_M` is ~2.5 GB and, after Whisper releases,
only ~2.5 GB is free — it OOM'd by ~160 MB and dropped half its layers onto the
CPU (the slow path).

Principle applied: **GPU-first — when a GPU is present, no model should run on
the CPU unless there is genuinely no GPU-fitting option.** All changes are
fail-soft and config-gated.

- **GPU-fitting quant selection** (`local_models.py`, `providers/ollama_provider.py`,
  `ai_orchestrator.py`). When the configured translation/polish model won't fit
  the card (would engage the CPU-spilling partial-offload ladder), auto-substitute
  a **smaller-quant build of the SAME model** that runs fully on the GPU —
  *only if that build is already installed* on the Ollama host (e.g.
  `qwen3:4b-instruct-2507-q3_K_M`, ~1.7 GB). Picks the largest quant that fits
  (best fidelity still on GPU). Memoized per model so translation's hundreds of
  per-batch calls never re-probe VRAM; applied centrally where the orchestrator
  swaps in a model override, so translation *and* polish are covered. If no
  fitting quant is installed, it logs an actionable `ollama pull …-q3_K_M` hint
  and leaves the model as-is (legacy behavior). New settings:
  `OLLAMA_TRANSLATION_FIT_GPU_QUANT` (True), `OLLAMA_GPU_BASELINE_RESERVE_GB`
  (1.2), `OLLAMA_GPU_KV_HEADROOM_GB` (0.55). Fit test:
  `weights_gb + kv_headroom ≤ total_vram − baseline_reserve` — the 1.2 GB
  reserve matches the ~2.5 GB free after Whisper on a 4 GB card.
- **Summary chunk-count cap** (`ai_orchestrator._map_reduce_summary`). Each map
  chunk is one sequential Ollama call; a 128-min video at the tier's 5-min
  chunks was ~21 calls ≈ 19 min. `SUMMARY_MAX_CHUNKS` (default 12) enlarges the
  chunk span so the count stays bounded — roughly halving summary time on long
  videos. Chunks only ever get BIGGER (never more granular than the tier), and
  the per-chunk transcript is now sampled by even stride instead of head-
  truncation so an enlarged chunk still spans its whole range. 0 = uncapped
  (legacy).
- **Batched untranslated-cue recovery** (`pipeline._llm_cleanup_untranslated`).
  The post-translation cleanup re-translated leftover source-language cues one
  LLM round trip each (the ~13-min tail). A batched pre-pass now translates the
  unique leftovers a few per call (JSON map, parsed defensively) and pre-fills
  the per-cue cache; anything a batch can't handle still falls through to the
  existing per-cue + cloud-escalation path, so robustness is unchanged. New
  settings: `TRANSLATION_LLM_CLEANUP_BATCH` (True), `TRANSLATION_LLM_CLEANUP_BATCH_CUES`
  (30).

Note: these are behavior changes (translation may run a slightly lower quant of
the same model on a small card; the summary is coarser on long videos), made
deliberately to honor the GPU-first / speed intent — not quality-neutral like
the earlier pass. The perception/Whisper/clip stages are unchanged. Real GPU
behavior must be validated on the target box (the CI environment here has no
CUDA/Ollama); the pure selection/parsing/arithmetic logic is unit-tested in
`tests/test_gpu_llm_speedups.py`.

# ClipAI — Speed & efficiency pass (quality-neutral): I/O amplification, redundant decode/hash passes, event-loop stalls

Every change is quality-neutral by construction: same frames at the same
timestamps, same Whisper parameters, same thresholds, same pipeline order.
The wins are write amplification, duplicate full-file reads, second decode
passes, and serialization on the event loop. All new behavior is fail-soft
(any error falls back to the previous code path) and every knob lives in
`backend/config.py`.

- **Task 1 — job.json write amplification** (`database.py`). Every
  `update_job_status` used to round-trip the FULL multi-MB job.json through
  disk (read → parse → Pydantic-validate → dump → pretty-print → atomic
  rewrite) for every ~1-3 s progress tick — hundreds of multi-MB cycles per
  job competing with ffmpeg/Whisper. Now: (1a) an mtime-checked, LRU-bounded
  (8 jobs) in-memory cache of parsed `JobResult`s serves reads and is updated
  with a re-validated object on every save; (1b) progress-only updates
  (progress/message, unchanged non-terminal status) mutate the cache
  immediately — readers and the WebSocket stay real-time — but persist at most
  once per `JOB_PROGRESS_FLUSH_INTERVAL` (default 2.0 s) with a guaranteed
  trailing flush; status changes, field writes, terminal statuses and
  whole-object saves write through unchanged, so a crash costs ≤~2 s of
  progress-bar position; (1c) job.json is written compact (`JOB_JSON_PRETTY`
  default False restores indent=2 for debugging). The atomic-write, per-job
  lock, `protect_terminal` and save-guard semantics are preserved exactly
  (guard-adjusted saves invalidate the cache instead of caching a divergent
  object; the direct `_persist_complete_job` writer invalidates it
  explicitly). Expected: removes hundreds of multi-MB JSON round trips and
  most steady-state disk churn per job.
- **Task 2 — source hashing off the critical path** (`pipeline.py`,
  `pipeline_helpers.py`). The post-extraction `await _hash_file_sha256`
  (full sequential read, 10-60 s on multi-GB array sources) blocked the
  engine start, and re-analyze runs hashed AGAIN inside the cache probe. One
  background hash task now starts right after metadata extraction (the page
  cache absorbs the read while ffmpeg streams the same bytes);
  `_maybe_use_cached_extraction` accepts the precomputed digest (hashing
  internally only for standalone callers); the DB `source_sha256`, the local
  copy and the engine-checkpoint signature all consume the single result.
  Exactly one full-file hash per run, fully overlapped with extraction; the
  task is cancelled and swallowed on failure/cancellation. Expected: 10-60 s
  recovered per fresh run, up to double on re-analyze.
- **Task 3 — PTS probe folded into the primary extraction pass**
  (`frame_extractor.py`). `_bulk_probe_pts` ran a SECOND ffmpeg decode over
  every extracted JPEG and was skipped above 240 frames. The extraction
  filter graphs now end in `,showinfo` (after `format=`, reporting exactly
  the frames written) and the extraction run's own stderr yields the ordered
  `pts_time` list — no extra decode, no ≤240 cap, count-mismatch falls back
  to the rough estimates for the unmatched tail. Bonus correctness fix found
  while verifying parity: the old probe fed the JPEG SEQUENCE through
  ffmpeg's image2 demuxer, which returns sequence timestamps (0.00, 0.04,
  0.08 …) rather than source pts — so ≤240-frame jobs got corrupted
  timestamps and every frame was tagged a scene cut (`gap < rate*0.7`). All
  jobs now carry exact source timestamps — never worse, usually strictly
  better. Expected: one full decode pass over all extracted JPEGs removed
  (seconds to tens of seconds per job) + correct scene-cut identification.
- **Task 4 — sidecar JSON dumps off the event loop** (`pipeline.py`).
  `render_plan.json` and `detection_overlay.json` (tens of MB on long
  sources — full face/person timelines) were built and `json.dump`-ed
  synchronously on the event loop, freezing heartbeats and WS broadcasts for
  seconds around the 60-62 % band. Both are now built + serialized inside
  `asyncio.to_thread` with the same try/except + log lines.
- **Task 5 — fail-soft hardware decode for the Perceiver**
  (`reframer_perceiver.py`, `REFRAMER_CV2_HWACCEL` default True). The
  Perceiver's `cv2.VideoCapture` open now requests FFmpeg
  `VIDEO_ACCELERATION_ANY` (guarded by `hasattr`); if the hw capture fails
  to open, its first read fails, or it can't rewind to frame 0, it is
  released and the plain software constructor is used. Only WHERE decoding
  happens changes — seeks, grab()/read() and returned BGR frames are
  identical (parity-tested). The winning path is logged at the PERCEIVE
  stage (`decode=hw(any)` / `decode=sw`). Expected: meaningful decode
  offload on long videos where OpenCV's FFmpeg build supports hwaccel.
- **Task 6 — cached saliency meshgrids** (`reframer_perceiver.py`). The
  speaker-bump block and the `_fuse_saliency` motion prior rebuilt
  `np.mgrid[0:sal_size, 0:sal_size]` on every no-face sample (thousands per
  run on gameplay/anime). One cached `(yy, xx)` pair per `sal_size` (same
  pattern as `_center_prior_cache`); identical numerics, pure allocation
  removal.
- **Task 7 — OPT-IN fix: `_prev_frame_for_face_motion`**
  (`REFRAMER_FIX_PREV_FRAME_MOTION` default **False** — the one change that
  is NOT output-identical when enabled). Legacy code updates the previous-
  frame reference INSIDE the per-face loop, so the 2nd+ face in a frame
  always scores motion/mouth_motion 0 (multi-face speaker detection
  degraded) and zero-face frames never update the reference (stale,
  seconds-old comparisons after face-less stretches). With the flag on, the
  reference is captured once at entry, used for all faces, and updated
  exactly once per sample including the zero-face path. Flag off is
  byte-identical to legacy (locked in by tests).
- **Task 8 — orjson with strict stdlib fallback**
  (`backend/services/fastjson.py`, wired into job.json save/load, engine
  checkpoint save/load, and the Task-4 sidecars; `orjson>=3.9` added to
  `requirements.txt`, installed by both Dockerfiles via `-r
  requirements.txt`). Compact separators, `OPT_NON_STR_KEYS` (matches
  stdlib's int-key stringification for the checkpoint timelines),
  `OPT_SERIALIZE_NUMPY` plus the existing `_numpy_safe_default` net; on
  ImportError (or any orjson-specific reject) it falls back to stdlib with
  identical semantics. Human-facing pretty output stays on stdlib json.
  Expected: several-× faster encode/parse on every persist/load of the
  multi-MB payloads, lowering steady-state CPU contention with the
  CPU-bound stages.

New settings: `JOB_PROGRESS_FLUSH_INTERVAL` (2.0), `JOB_JSON_PRETTY`
(False), `REFRAMER_CV2_HWACCEL` (True), `REFRAMER_FIX_PREV_FRAME_MOTION`
(False).

Tests: `tests/test_perf_speedups.py` (cache/debounce/compact, hash-once,
fastjson round-trips + fallback, checkpoint round-trip),
`tests/test_reframer_perf.py` (hw-decode fallback + hw/sw frame parity,
meshgrid numerics, prev-frame flag on/off), and
`backend/tests/test_bulk_pts_probe.py` rewritten for the in-pass showinfo
parsing (parser unit test, error-extraction with showinfo noise, ffmpeg
end-to-end pts check). Existing persistence-race / save-guard / recovery /
checkpoint tests all pass unchanged.

# ClipAI — 100%-target-language transcripts, user-accurate reframe grades, denser subject tracking

Implements the full run-13 assessment: language purity + coherence, per-clip
"what the viewer actually sees" grading, and the detection-density /
vertical-framing reframing builds.

## Transcript: entirely in the selected language, contextually coherent
- **Verified purity loop.** The per-cue recovery now escalates ECHO failures
  (not just exceptions) to the cloud polish model — run 13 shipped 6/17
  flagged romaji cues because qwen2.5:3b echoed them back and the cloud net
  only caught thrown errors. Every candidate is validated (preamble-stripped,
  re-run through the romaji/CJK detector) before acceptance, and any cue
  still impure after recovery lands in the job warnings with its timestamp —
  a visible defect instead of a silent one.
- **The English track finally gets polished.** On the LLM-translate path the
  post-edit was deliberately skipped ("the LLM translation is final"), so
  claude-haiku polished the Japanese source while the English viewers read
  shipped raw — typos ("bigdest", "Pantss") and word-salad ("Hand kimchi")
  included. ``correct_transcript(mode="translation")`` now runs on the
  translated track too (``TRANSLATION_POLISH_LLM_OUTPUT``), source-aligned
  when cue counts match, fenced by the existing never-reintroduce-source
  purity guard.
- **ASR-confidence marks.** Whisper's per-segment ``avg_logprob`` rides into
  the polish prompt: lines under the redecode threshold are tagged
  ``"asr_confidence": "low"`` with an instruction that nonsense there may be
  rewritten toward context while unmarked (reliable) lines are preserved.
- **Wrong-script hallucination gate** (``WHISPER_SCRIPT_FILTER``): with the
  source pinned/confidently-detected CJK, a 4+-word all-Latin cue ("See you
  next time in the video." over a musical outro) is quarantined like the
  other TACT filters. Conservative: mixed romaji/Japanese lines untouched.

## Reframe grading that matches what a user would say
- **Per-clip grades + clip-weighted headline.** ``evaluate_per_clip`` slices
  the evaluator to each exported clip's window (evaluator ``run()`` now takes
  ``start_sec``/``end_sec``); every clip carries its own
  ``reframe_report`` (grade, score, p10, HIGH density) and the job report
  gains ``clip_weighted`` — the duration-weighted aggregate over what
  viewers actually watch. A face_missing in a never-exported region no
  longer moves the headline.
- **Worst-moments scoring.** ``p10_window_score`` (10th-percentile 30s
  window) reports the stretches users remember; the letter grade is CAPPED
  at C above 1.5 HIGH problems/min and at D above 4/min — a clip that keeps
  losing its subject can't grade B on its averages (``grade_uncapped``
  preserves the raw letter).
- **VLM spot-check** (``REFRAME_VLM_SPOTCHECK``): ~12 cropped frames sampled
  across the exported clips are judged by the local vision model ("is the
  main subject well-framed?") → ``vlm_framing_pct``, a human-proxy
  calibration signal reported alongside the geometric metrics, never blended
  into the grade (VLM availability must not change grading).
- **Operator corrections as ground truth.** Clip exports now carry
  ``manual_crop_overrides`` (how many crop segments the user manually
  pinned); the backend logs a stable ``REFRAME-CORRECTION`` tracer line and
  persists ``operator_corrections`` onto the clip's reframe report —
  corrections-per-clip is the metric that converges on the user's judgment.

## Reframing: denser tracking, self-repair, per-scene vertical framing
- **Inter-sample LK tracking** (``REFRAMER_INTER_SAMPLE_TRACKING``): between
  sparse detection samples the perceiver already grab()-decodes every frame;
  now it retrieves ~3 of them per gap and tracks the last sample's face
  boxes forward with pyramidal Lucas-Kanade (median flow over a 3×3 point
  grid, confidence-decayed, ``tracked: True``). A 0.14 Hz face timeline
  becomes effectively several-Hz for near-zero decode cost — directly
  attacking the largest HIGH-problem class (interpolation drift).
- **Problem-driven repair** (``REFRAMER_PROBLEM_REPAIR``): before the
  bridge, a quiet evaluation localizes HIGH face_missing windows; YuNet
  re-detects densely inside just those (no VRAM), fresh samples join the
  face timeline, and corrective keyframes pull the crop back onto the
  confirmed face (min-gap 1.2s, capped windows). The evaluator stops being
  a report and becomes a repair loop; the post-run ReframeReport shows the
  post-repair numbers.
- **Per-scene vertical eye-line** (``REFRAMER_PER_SCENE_EYELINE``): the
  planner tags each scene's keyframes with a scene-local ``y`` (same 1/3
  eye-line + headroom rule, measured per scene); the bridge emits it on
  each op's motion-path rects, and the clip exporter's cached-plan tier
  reads the plan's vertical framing at the clip midpoint instead of the
  coarse scene average. Sub-full-height crops (1:1, 4:5) now follow the
  subject vertically scene-by-scene instead of one video-wide compromise.
- **Live-action classifier fix.** Run 13's live content graded
  "animated/other" because faces show in only ~60% of samples (turned away,
  intimate angles) — under the flat 70% bar. 40-70% coverage now counts as
  live action when corroborated by a PERSISTENT track (one YuNet track
  visible across ≥15% of samples), which anime/gaming essentially never
  produces.

# ClipAI — Bridge off the event loop; windowed thumbnails; extract-band + refinement label fixes

The 2026-07-03 21:32 run (first on build 720e347) confirmed the duplication
fix — 403 persisted cues, zero phantom copies — and the new phase labels, and
exposed the next layer:

- **Bridge conversion no longer freezes the pipeline.** ``to_fez_*`` ran
  synchronously ON the event loop; when it took 480s the browser got nothing
  from 34m14s→42m15s and the 60% CONVERT update only arrived when the bridge
  finished, with an 8-minute-stale heartbeat clock. The whole bridge now runs
  in a worker thread (``asyncio.to_thread``) so heartbeats + WS updates keep
  flowing.
- **Batch scene thumbnails are decode-windowed.** The single-pass select
  filter used a bare ``-i`` — every chunk decoded from t=0, so the tail chunk
  of a 128-min source decoded ~110 minutes of video it discarded and hit the
  flat 120s timeout, degrading to ~224 per-scene seeks (most of the 480s).
  Each chunk now seeks (``-ss`` before ``-i``) to its own time window with
  select windows rebased to the seek point, and the timeout scales with the
  windowed span.
- **Post-bridge phases are labeled.** Speaker fusion + the multi-minute
  source polish ran under the stale "render plan conversion" heartbeat; a 61%
  update now names them ("source transcript polish").
- **EXTRACT band is forward-only.** Frame (8-9%) and audio-precondition
  (9-14%) callbacks share one percent floor, killing the visible 13%→8%
  regressions of the interleaved messages.
- **No more "Transcribing (100%)" during refinement.** The gap-fill pass
  re-emits transcription fractions; after the ``transcript_refine`` /
  ``diarization`` phase hints those are latched to the current phase's
  message instead of resurrecting the Whisper progress line.
- **LLM preamble strip.** One shipped cue read "Sure, here is the
  translation:\n\nNothing really matters." — ``strip_llm_preamble`` now
  removes chatty preambles + Translation:/English: labels in both the batch
  translate parse and the per-cue recovery path (never stripping a line to
  empty).

Run-213250 audit: transcript clean (403 cues, only a genuine repeated
"Amazing."); polish again 51/155 via the cloud fallback (21 rescues); romaji
recovery 11/17 flagged cues; translated readability grade A (95.9);
ReframeReport steady at grade B 89.8 (identical metrics to the previous run —
same source, deterministic pipeline). Remaining known gaps: ~8 short
mixed-romaji lines still shipped ("Goi nurete…", "Shii desho") — flagged but
the recovery model echoed them; and the reframer face-sampling rate remains
the top reframing-quality lever.

# ClipAI — Kill the transcript-duplication corruption; truthful progress labels

Driven by the 2026-07-03 05:36 run (build c704bef) audit: the pipeline
persisted a CLEAN 406-cue translated transcript, but the downloaded track had
625 cues with 117 lines repeated ~3x each at timestamps minutes apart (one
phantom family at a constant ~+19min offset, another splayed further). The
writer was the NLE reverse-sync: a timeline holding stacked stale generations
of the same subtitle cues replaced the canonical transcript wholesale. The
existing dedup passes can't catch that shape — ``drop_scattered_duplicates``
fires at 4+ occurrences, ``collapse_repeated_runs`` needs CONSECUTIVE runs;
the phantoms were 3x and interleaved with real dialogue.

- **Union-write guard (backend).** ``PUT /jobs/{id}/transcript`` now rejects
  (409) a replace that is much bigger than the stored track (>25% and >15
  cues) AND much more text-duplicated (+10 points of duplicate-text share) —
  the stale-union signature. Genuine edits (splits, merges, denser imports)
  pass: they change counts modestly or add NEW text, not hundreds of verbatim
  copies. ``clean_and_sort_segments`` additionally drops verbatim duplicates
  at the same position (a double-backfilled track). ``detect_union_write`` in
  ``backend/services/transcript_sync.py``; tests in
  ``backend/tests/test_transcript_union_guard.py``.
- **The union can no longer form (frontend).**
  ``addSubtitlesFromTranscript`` is idempotent — it replaces the subtitle
  track instead of stacking a second generation when a double effect-fire or
  an IndexedDB restore races the empty-track check. The VideoEditor
  stale-track sync now also rebuilds on the union signatures themselves:
  duplicate ``transcriptIndex`` values among cues, or a cue count far above
  the transcript rows overlapping the clip window (the old "half the texts
  mismatch" test stays below threshold on a fresh+stale union, which is
  exactly how 625 survived). The reverse-sync PUT mirrors the backend guard
  (count blowup + duplicate-share) so a corrupt track is never even sent —
  this also keeps NLE subtitle elements 1:1 with the transcript's length and
  text unless the user actually edited them.
- **Truthful per-step progress.** The engine progress relay accepts phase
  hints and the perceiver/audio stages emit them at the two long dark
  stretches from the run report: ``transcript_refine`` when the Whisper
  segment loop ends (the bar used to freeze at "Transcribing 79%" through
  minutes of redecode + gap-fill + alignment) and ``diarization`` before the
  speaker pass (8 minutes of silence labeled "scene analysis"). Emitted
  percent is now forward-only (interleaved precondition/extract callbacks
  made the bar jump backwards), and the 15%/57%/60% pipeline updates carry
  explicit heartbeat labels ("face + motion detection", "speaker analysis
  wrap-up", "render plan conversion") so keepalives name the real work with
  a freshly-reset elapsed clock instead of "scene analysis — 8m 5s elapsed".

Run-073154 audit vs the previous run (983c5d0), same video: polish went 0/155
→ 51/155 segments improved (every local batch still failed on the degraded
Ollama chain, but the cloud fallback rescued 19/19 batches via
anthropic/claude-haiku-4.5); romaji cleanup recovered 8/13 flagged cues and
zero romaji lines shipped; the early preview proxy was ready 2 minutes into
the run (before analysis started); redecode re-decoded 8 difficult segments;
translated readability graded A (96.3). ReframeReport held steady at grade B
(overall 89.8, face coverage 94.5%, hold 77.4%, safe-area 87.1%, 2.14
cuts/min; HIGH problems 125 → 117).

# ClipAI — Polish that can't silently die, snappy preview, real YOLO-World classes

Driven by the 2026-07-03 run logs: every polish batch of both jobs failed
("All providers failed"), romaji shipped in the English track, YOLO-World
ran on generic COCO classes, the editor scrubbed against a raw long-GOP
file, and bridge_conversion spent 167s on per-scene thumbnail seeks.

- **Polish reliability (local + cloud).** The first batch gets a 3x
  cold-load timeout (capped 300s) — the flat 90s expired during Ollama's
  partial-offload model load, struck the circuit breaker 3x and killed
  polish for the whole job. When the local chain still fails, the batch
  retries via OpenRouter (``SUBTITLE_POLISH_CLOUD_FALLBACK``, only when a
  key is configured; ``SUBTITLE_POLISH_CLOUD_MODEL`` or the shortlist's
  top efficient pick). A pinned OpenRouter polish model now routes
  straight to the cloud instead of failing through an Ollama-only chain.
  The same cloud net backs the per-cue untranslated-recovery pass.
  Ollama keep_alive raised 30s → 5m (explicit gpu_preflight eviction
  already protects the GPU stages; the short value only added the
  cold-load churn that blew the timeouts).
- **Netflix-consistency auto-glossary**
  (``SUBTITLE_POLISH_AUTO_GLOSSARY``): recurring capitalized names are
  clustered by fuzzy similarity (Zeks/Zecks/Zeck → Zechs, most frequent
  variant wins) and pinned in the polish prompt alongside the custom
  vocabulary, so names render identically in every cue.
- **Romaji gate.** The per-cue cleanup now SELECTS romaji cues (it
  filtered on CJK ratio only, so "Nametotte ageru kara." was never even
  considered), rejects romaji echoed back by the model, and the detector
  handles mixed romaji/English lines (max over head/tail token windows)
  plus distinctive particle evidence (desu/masu/-chan/kudasai...).
  Detection now also applies to auto-detected sources at a stricter 0.75
  bar — declared non-Japanese sources are still exempt. Verified against
  all six shipped romaji lines with zero English false positives.
- **CLIP for YOLO-World** — both images install the ultralytics CLIP fork
  (+ ftfy/regex, --no-deps so torch pins hold) and pre-bake the ViT-B/32
  text encoder. Fixes "set_classes failed: No module named 'clip'": the
  open-vocabulary classes (person/head/face/character/mecha...) now
  actually apply instead of silently degrading to COCO — the biggest
  subject-detection accuracy lever in the log.
- **Preview proxy for long-GOP sources.** ``_needs_preview`` now
  triggers on sparse keyframes (median interval > 3s over the first
  minute) and missing faststart — the two things that actually make
  scrubbing snappy — not just codec/size/bitrate. Stream-copy is
  forbidden when the GOP is sparse (it would copy the problem into the
  "preview"). The proxy builds with NVENC first (seconds on a GTX 1650)
  falling back to libx264, and is now kicked off EARLY in the pipeline
  (right after metadata) so the editor plays the dense-GOP proxy even
  while analysis is running. Preview filename bumped v3 → v4 so stale
  "no preview needed" sentinels re-evaluate.
- **Single-decode scene thumbnails.** ``to_fez_scenes`` extracts ALL
  scene thumbnails in one ffmpeg select-filter pass (chunked at 200)
  instead of one seek per scene; the per-scene seek remains only as a
  fallback for missed frames. On the 128-min run this stage measured
  167s for 224 scenes.

# ClipAI — Fix: analysis died at 15% with "cv2 has no attribute CascadeClassifier"

First run on the rebuilt image failed at the FACES stage:
``module 'cv2' has no attribute 'CascadeClassifier'`` — while YuNet
(``cv2.FaceDetectorYN``) had loaded fine seconds earlier. That split
symptom is a MIXED OpenCV install: ultralytics declares the GUI
``opencv-python`` as a dependency, and with the unpinned
``opencv-python-headless>=4.8.0`` + ``ultralytics>=8.0.0`` a fresh image
build can end up with two distributions sharing one ``cv2/`` directory,
where some symbols resolve and others don't.

Fixed in two layers:

- **Image hygiene** — both Dockerfiles now purge every opencv
  distribution and force-reinstall the single pinned
  ``opencv-python-headless==4.10.0.84`` as the LAST pip layer, then FAIL
  THE BUILD if ``CascadeClassifier``/``FaceDetectorYN`` are missing.
  ``YOLO_AUTOINSTALL=false`` stops ultralytics from pip-installing
  anything at runtime (which would overwrite cv2 mid-process on the
  first YOLO import). requirements.txt pins opencv exactly and bounds
  ultralytics ``<9``. **Rebuild the image to pick this up**
  (``./deploy.sh``).
- **Code resilience** — Haar is the last-resort detector tier; its init
  is now wrapped so a broken cascade logs one clear line (naming the
  mixed-OpenCV cause) instead of killing an analysis where YuNet + YOLO
  are healthy. All Haar call sites (``_detect_haar``,
  ``_detect_haar_relaxed``, ``_detect_yolo_assisted_haar``, eye check)
  tolerate the missing cascade.

# ClipAI — Phase 5: Container speed & efficiency

- **Perception decode no longer seeks every sample.** The sampler's seek
  threshold was ``gap > 5`` frames — at 5 fps sampling of 30 fps video
  the gap is exactly 6, so EVERY sample did a ``CAP_PROP_POS_FRAMES``
  seek, and on long-GOP H.264 each seek re-decodes from the previous
  keyframe. Sequential ``grab()`` (decode-skip) now covers gaps up to
  ``REFRAMER_SEEK_GAP_FRAMES`` (60 ≈ 2x a typical GOP); only genuinely
  large jumps seek. ``scripts/bench_perception_decode.py`` measures
  seek-vs-grab-vs-piped-ffmpeg (± NVDEC) on a real fixture so the piped
  decoder can be promoted if it wins on your content — run it on the
  30-min 1080p fixture and record the numbers.
- **NVENC tuned for the TU117 (GTX 1650)**: ``-spatial_aq 1
  -temporal_aq 1`` on all NVENC encodes (pure quality win at equal
  bitrate, supported since Kepler), ``-bf 0`` on the HEVC 4K path (the
  1650's Volta-gen NVENC has no HEVC B-frames), and a speed/quality
  toggle ``GPU_NVENC_PRESET`` (p1..p7, default p5, persisted in
  Settings). The GPU→CPU retry already strips these flags. ``-rc vbr
  -cq N -b:v 0`` was already in place. Full-GPU filter pipelines
  (``-hwaccel_output_format cuda`` + scale_cuda) stay out for now — the
  current filter graphs run CPU-side filters (subtitles/overlays) that
  would force a download mid-graph.
- **Models pre-baked into the image**: u2netp.onnx (learned saliency) is
  now downloaded at build time in BOTH Dockerfiles and auto-discovered
  from ``backend/models/`` / ``/data/models/`` — no more manual
  ``REFRAMER_U2NET_MODEL_PATH``. The CPU image also gains the SFace
  embedding model (the GPU image already had it; YuNet/YOLOv8n/Demucs/
  animeface were already baked). torch==2.5.1/torchaudio==2.5.1 pins
  verified in both images (guards the pyannote upgrade path); pip runs
  --no-cache-dir throughout.
- **Vocal separation runs concurrent with perception**
  (``VOCAL_SEPARATION_CONCURRENT``, default on): Demucs starts before the
  engine and the perceiver blocks on its future only when it reaches
  transcription, so separation gets the visual pass's wall time for
  free. Auto-degrades to the proven sequential order on GPUs under 6 GB
  total — the 4 GB budget is never shared between Demucs and YOLO.
- **Not done (documented follow-ups)**: extracting VLM/thumbnail frames
  from the perception decode stream (single-decode architecture) — the
  VLM path wants higher-resolution frames at scene-change timestamps, so
  sharing the 640x360 detection stream trades quality for speed and
  needs measurement first; and streaming polish batches during
  transcription (requires cross-thread segment streaming out of the
  perceiver). Both are scoped in the audit and remain open.

# ClipAI — Phase 4: Cloud transcription + a real subtitle-polish model picker

There was no cloud STT path — transcription was local-only, and polish
shared the general translation/editorial model. Both fixed.

- **Cloud transcription providers** (``TRANSCRIPTION_PROVIDER=local|groq|
  openai``, new ``backend/services/cloud_transcription.py``): Groq
  ``whisper-large-v3-turbo`` (word timestamps) and OpenAI ``whisper-1`` /
  ``gpt-4o-transcribe`` (no word timestamps — the Phase 3 forced aligner
  re-times them). Cloud output is mapped to the exact local segment
  schema and flows through the SAME post chain — hallucination filter,
  repetition-loop drop, end clamp, forced alignment, then the usual
  polish + formatter — so quality rules are uniform across providers.
  Custom vocabulary is injected as the provider prompt. Any API failure
  falls back to local Whisper automatically; a configured cloud provider
  also works when faster-whisper isn't installed locally. Selector +
  OpenAI key field in Settings > Subtitle Quality.
- **"Recommended for subtitle polish"** — new
  ``GET /api/providers/models/recommended/subtitle-polish``: a curated,
  ordered shortlist (``SUBTITLE_POLISH_SHORTLIST`` in
  ``openrouter_provider.py`` — data, not code) of models strong at
  constrained editing, intersected at request time with the live
  OpenRouter /models list (availability + pricing + context length),
  returning the top pick per tier (free/efficient/premium) with a
  one-line rationale and $/1M-token cost. Surfaced in
  ``SubtitleQualitySettings.jsx`` with a refresh button.
- **Built-in polish benchmark** (``backend/services/polish_benchmark.py``
  + ``POST /api/providers/models/polish-benchmark``): 20 canned
  Whisper-medium-style error segments with gold corrections run through
  ``transcript_polisher``'s own prompt via ``model_override``; scores
  exact-fix rate + format compliance (segment count, ±15% word budget).
  Scores persist in ``polish_benchmark_scores.json`` and measured winners
  outrank the static shortlist order in the recommendation. "Test" button
  per model in the UI.
- **``SUBTITLE_POLISH_MODEL``** pins polishing to the selected model —
  ``_resolve_polish_model_override`` prefers it over the
  translation-model default, so polish stops sharing the general model
  once a pick is made. Persisted via user_settings.json.

# ClipAI — Phase 3: Netflix-grade local transcription on a GTX 1650

Timing precision and edge-case accuracy work on the faster-whisper
pipeline; the model/VRAM logic is untouched except for new registry
entries.

- **CTC forced-alignment refinement** (``SUBTITLE_FORCED_ALIGN``, default
  ON): after transcription, ``forced_aligner.refine_word_timestamps``
  re-aligns each segment's words against the audio (torchaudio wav2vec2
  CTC for English; the ctc-forced-aligner package slot is wired for
  multilingual when installed) and snaps word + cue boundaries to speech
  onset/offset. Whisper's 50-200 ms word drift is the visible difference
  from Netflix-grade cueing. Shifts past 600 ms are rejected as aligner
  mis-anchors; every failure path leaves original timestamps untouched.
  GPU used only when >1.5 GB VRAM is free, else CPU.
- **VAD tuning + hallucination hardening**: ``vad_parameters`` are now
  settings (``WHISPER_VAD_MIN_SILENCE_MS=300``,
  ``WHISPER_VAD_SPEECH_PAD_MS=150`` — down from a hardcoded 200), and
  ``_decoding_kwargs`` feature-detects faster-whisper's
  ``hallucination_silence_threshold`` (2 s). The TACT filter stays as the
  second line of defense.
- **Two-pass difficult-segment redecode** (``WHISPER_REDECODE_*``):
  segments flagged by the hallucination filter, below the avg_logprob
  floor (-0.8), or with degenerate word timestamps (the batched-inference
  word-timing failure mode — detected by ``_words_degenerate``) are
  re-decoded individually via ``clip_timestamps`` with beam_size=8 and
  patience=1.5, bounded to 10% of segments worst-first, before the polish
  LLM sees them. Replacements only land when measurably more confident.
- **distil-large-v3 / v3.5 in the registry + VRAM tables** (fp16 1.6 GB /
  int8_float16 0.9 GB — fits the 4 GB class one notch under turbo).
  ``WHISPER_PREFER_DISTIL_ENGLISH=1`` retargets the GPU auto-upgrade to
  distil for English-only libraries; multilingual keeps large-v3-turbo.
  Pinned distil picks are never silently swapped.
- **Per-stage VRAM ledger** (``backend/services/vram_ledger.py``): one
  greppable ``VRAM ledger | job=… stage=… free=…MB`` line at
  analysis_start, pre_whisper, post_whisper_release, pre_voiceprint and
  pre_offline_nmt, plus ``get_ledger(job_id)`` for reports. Confirms the
  load-bearing ordering (Whisper released before NLLB/pyannote) on every
  run. Existing stage ordering audited and unchanged — it was already
  correct.

# ClipAI — Phase 2: Reframing that frames like a human operator

Builds on the L1-optimal camera path (already default ON) with the
behaviors that separate a good auto-reframe from a human one: real
holds, cuts instead of whip-pans, composition instead of centroid
chasing, and metrics that make all of it measurable.

- **Deadband/hysteresis on the L1 targets** (``REFRAMER_L1_DEADBAND_FRAC``,
  2.5% of crop width): subject motion under the deadband snaps to the hold
  anchor before the solve, so holds come out truly static instead of
  micro-drifting. Decisive moves re-anchor and pass through unchanged.
- **Saccade cut-vs-pan** (``REFRAMER_SACCADE_CUT_FRAC``, 38% of crop width):
  large displacements become ``transition: 'cut'`` keyframes in both the
  Planner's pan/cut decision (previously live-action only at 20%) and the
  L1 keyframe rebuild (jump > threshold within 700 ms → cut). The Smoother
  already exempts cuts from velocity clamping and enforces a post-cut hold.
- **Vertical eye-line composition** (``REFRAMER_VERTICAL_EYELINE``): when the
  target is wider than the source (16:9/4:3 outputs), ``crop_y`` places the
  median subject eye-line (face top + 35% of face height) at 1/3 from the
  crop top, clamped by headroom, instead of blind vertical centering.
  Static per clip — per-keyframe y animation remains future work (RenderPlan
  carries a single ``crop_y``).
- **Headroom clamps** (``REFRAMER_HEADROOM_MIN_FRAC``, 8% of crop height):
  ``_validate_face_in_crop`` margins now have an 8%-of-crop-height floor so
  faces never touch the crop edge.
- **Adaptive tiled detection** (``REFRAMER_TILED_ADAPTIVE``): the 2x2 tiled
  YuNet pass — the most expensive detector — now runs only when the largest
  face found by the cheaper passes is under 4% of frame height (or none was
  found), which is exactly when it adds recall. YOLO-World person-box fusion
  for lost faces (torso framing before saliency fallback) already existed in
  the Planner and is unchanged.
- **Dense export keypoints** (``REFRAMER_EXPORT_KEYPOINT_HZ``, 10 Hz): the
  bridge samples eased keyframe transitions at ≥10 Hz before building
  MotionKeypoints, then prunes collinear samples, so the FFmpeg
  piecewise-LINEAR x expression reproduces easing without velocity steps
  while holds stay 2 points.
- **Stability metrics in the evaluator**: ``ReframeReport`` gains
  ``jerk_integral`` (30 Hz path, cut-discontinuities excluded, normalized by
  crop width), ``hold_ratio_pct`` (% of samples with |v| < 2% crop_w/s),
  ``safe_area_pct`` (best face inside the 10%-inset safe area) and
  ``cuts_per_minute`` — all printed in the ReframeReport log line so tuning
  is measurable. Regression tests in
  ``backend/tests/test_reframer_human_operator.py``.

# ClipAI — Phase 1: NLE preview ↔ export parity (audio, speed, fades, fallback)

The WYSIWYG contract is "same renderFrame() for preview and export".
Video honored it; audio and time-mapping did not. All fixed in the
client export engine, with the divergences that remain (server-side)
documented in a generated checklist.

- **Client export audio was fundamentally broken (silent).**
  ``ExportEngine._extractAudio`` called ``createMediaElementSource()`` on an
  ``OfflineAudioContext`` — unsupported everywhere; browsers throw and the
  catch block silently skipped every clip. Rewritten to fetch + ``decodeAudioData``
  each clip's media (cached per asset) and schedule ``AudioBufferSourceNode``s
  with ``start(when, offset, duration)`` derived from ``clip.start``,
  ``clip.trimStart`` and the export range.
- **Per-clip speed now exports.** Frame stepping and audio scheduling both map
  timeline→source time as ``trimStart + (t − start) × speed`` (shared
  ``timelineToSourceTime`` helper); audio speed uses
  ``AudioBufferSourceNode.playbackRate``. Parity test asserts 2× source-frame
  correspondence.
- **Audio fades/ramps now export.** ``buildGainAutomation`` schedules gain
  automation matching ``fadeIn``/``fadeOut`` exactly like RenderEngine's
  opacity fade math (same elapsed/fadeIn formula), including mid-fade export
  ranges and overlapping-fade sampling.
- **Muted means muted.** ``clip.muted`` and ``track.audioMuted`` (with legacy
  ``track.muted`` fallback) are excluded from export audio; previously the
  clone was force-unmuted.
- **MediaRecorder fallback is now gated.** The real-time fallback (dropped
  frames, word-highlight drift) requires explicit user confirmation via
  ``onFallbackRequired``; which path ran is recorded in ``lastExportMeta`` and
  passed to ``onComplete``.
- **Shared active-word timing module.** The "MUST match exactly" constants +
  word-index algorithm duplicated across ``RenderEngine`` and
  ``SubtitleOverlay`` (and the constants in ``ClipPreview``) now live in ONE
  module, ``frontend/src/utils/activeWordTiming.js``.
- **Gaming blurfill preview matched to export.** Canvas used ``blur(20px)``
  with no darkening vs FFmpeg ``gblur=sigma=50,eq=brightness=-0.1`` — now
  ``blur(50px) brightness(0.9)``, the same equivalence renderPlanRenderer uses.
- **Safari canvas filters detected.** ``RenderEngine.supportsCanvasFilter()``
  probes ``ctx.filter`` support and warns once instead of silently rendering
  effects unfiltered.
- **Effects coverage checklist + parity harness.**
  ``frontend/src/utils/featureParityMatrix.js`` enumerates every panel-exposed
  property with its status per render path (enforced by
  ``featureParityMatrix.test.js``); ``scripts/generate-parity-checklist.mjs``
  generates ``docs/parity-checklist.md``; ``scripts/parity_harness.py``
  compares client vs server renders frame-by-frame (SSIM + subtitle
  bounding-box deltas) on hosts with FFmpeg.

# ClipAI — Honest build label + no stale "Exporting clips" on reconnect

Two cosmetic fixes:

- **Build label was "unknown — (no subject)".** The image can't read git at
  runtime (``.git`` is excluded from the build context), so the SHA/subject must
  be passed as build args — but a plain ``docker compose build`` (without
  ``export BUILD_SHA=…``) leaves them at their "unknown" default. Added
  ``deploy.sh``, which captures the SHA + commit subject from git and exports
  them before building, so the startup log reads e.g. ``ClipAI build: d531150 —
  Collapse duplicate transcript lines``. Use ``./deploy.sh`` to redeploy.

- **A reconnect hours later showed a bogus "Exporting clips (59/63)".** On WS
  connect the server replays the job's current state, but for a finished job it
  was sending the stored ``progress_message`` — often a stale intermediate
  ("Exporting clips… (59/63)", because clip export emits progress out of order
  under concurrency). It now sends a clean terminal line ("Analysis complete" /
  "Cancelled" / the failure reason), so a late reconnect reflects reality.

# ClipAI — Collapse duplicate transcript lines (Whisper repetition)

Even on a clean run the translated transcript repeated whole lines at different
times — e.g. "panties look cute." at 0:00 and 0:09, "specific places before?" at
2:39 and 6:47. Cause: Whisper repetition/hallucination on this kind of non-speech
audio (music, moans, silence) emits the same source line at several timestamps,
and translation is strictly 1:1, so it carries every copy through.

Fix: ``transcript_sanitize`` now collapses a SUBSTANTIAL line (≥16 chars — a real
sentence/phrase) to a single occurrence instead of keeping up to two. SHORT lines
("Yes?", "No no") can legitimately recur and still keep a couple; markers
("[♪ music ♪]") are never deduped (music plays at multiple points). The pass
stays idempotent and order-preserving, and runs both before persist and on the
read self-heal, so old transcripts clean up on next view too.

Note: this removes verbatim REPEATS. The remaining short mid-sentence fragments
(e.g. "I'm actually") are inherent to this source's pause-heavy pacing — Whisper
splits on the pauses — and are left as-is to avoid over-merging unrelated lines.

# ClipAI — Clip judge works offline again (no more HTTP 400 on every clip)

In Offline Mode the clip-scoring judge is the best local *editorial* model —
a TEXT model (e.g. qwen2.5:3b-instruct). But ``OllamaJudge`` sends clip keyframes
as ``images``, and a text-only model doesn't ignore them — Ollama rejects the
request with HTTP 400. So every judged clip failed ("Ollama judge error: HTTP
Error 400") and ranking silently fell back to signal-only heuristics.

Fix: ``OllamaJudge`` now retries text-only when a request with images is rejected,
and latches that so the rest of the batch skips images outright (one wasted
attempt, not one per clip). The judge scores clips on transcript + signals
instead of failing — and a genuinely vision-capable local model still gets the
keyframes. (The docstring's wrong claim that "non-vision models simply ignore"
images is corrected.)

# ClipAI — Dashboard / new-tab loads fast again (don't full-parse every transcript)

Opening a new tab sat on "Connecting to container…" for a long time. Cause:
``GET /api/jobs`` (the dashboard list) calls ``database.list_jobs()``, which read
AND fully Pydantic-validated EVERY job on disk — including a job bloated by the
crash corruption (thousands of cues, each with per-word timestamps) — just to
return ~10 summary fields. That validation ran on the event loop, so it both made
the list slow and starved the ``/api/health`` ping that the "Connecting to
container…" banner waits on.

Fix: ``list_jobs(light=True)`` (used by the dashboard list + startup recovery)
drops the per-cue ``transcript`` / ``translated_transcript`` arrays before
validation — callers there only need summary fields, status, clips and summary —
and the parse now runs in a worker thread. A bloated job can no longer slow the
list or block the health ping, so a fresh tab connects and renders quickly.

# ClipAI — Pipeline progress is responsive on mobile

The pipeline progress UI didn't fit a phone screen:
- ``ProgressBar``'s header put the status message and the % in a space-between
  row with no truncation, so a long message (e.g. "Re-using cached extraction —
  563 frames (SHA matched)") wrapped and pushed the % around on narrow widths.
- ``PipelineTracker`` rendered 10 proportional stage labels in one row; on a
  phone the small-weight stages collapsed to a few pixels and the 9px labels
  ellipsized to nothing.

Fixes:
- ``ProgressBar``: the message now truncates to a single line (ellipsis) and the
  % is ``flexShrink: 0`` so it's always pinned and readable, at any width.
- ``PipelineTracker``: on mobile (<768px) the cramped 10-label row is replaced by
  one readable line for the CURRENT stage — e.g. "Faces · step 3/10" (or "All
  stages complete" / "Stopped — re-analyse to resume"). The proportional colored
  stage bar (which scales fine) stays on every size, and the full per-stage label
  row still shows on tablet/desktop.

# ClipAI — Clip export now resumes instead of stopping short after a crash

Report: if a job dies while exporting clips, it never finishes exporting all the
clips — you end up with fewer than were found. Cause: the clip list is
snapshotted BEFORE export (so a crash doesn't lose the candidates), so a job that
died mid-export has the full found-count in ``clips`` but only some MP4s on disk.
Startup recovery saw "has clips" and marked it COMPLETE — stranding the unfinished
exports.

Fix (two parts):
- **Recovery now gates "complete" on the export actually finishing** — the
  ``clips_manifest.json`` (written right after the export loop) or every clip's
  ``*_clip*.mp4`` being present. A job with the list but missing files is treated
  as "needs resume", so it re-queues and finishes the rest instead of being
  declared done. (A genuinely-complete job, or one with only a summary, is
  unaffected; any inspection error fails safe to "complete" so nothing loops.)
- **Clip export is now idempotent** — ``_export_clip`` skips a clip whose MP4 is
  already on disk, so a resumed run finishes only the clips it never reached
  rather than re-encoding the whole batch.

Net: a crash at clip 30/63 now resumes and produces all 63, instead of completing
with 30.

# ClipAI — Checkpoint indicator in the processing log

You can now see, right in the processing log, when the pipeline reaches its resume
checkpoint — so if a job fails you know exactly where it will pick up. After
detection + transcription are saved, a cyan ⚑ "Checkpoint reached — detection +
transcription saved. If the job restarts, it resumes from here." line appears
(and on a resumed run, a matching "Resumed from checkpoint…" line). The marker is
styled as a milestone (cyan, bold, left-bar) so it stands out from ordinary
status lines, and it's persisted to the job event log so it survives reloads.

# ClipAI — Fix the per-segment PUT storm that kept the GUI from loading

The container logs showed hundreds of `PUT /api/jobs/<id>/transcript/<N>` requests
for the SAME cue indices, repeating across 6 connections nonstop — each one
re-saving the whole (bloated) job. That saturated the backend so the app's own API
calls were starved and the GUI wouldn't load.

Cause: the VideoEditor's reverse-sync (timeline subtitles → transcript). The TEXT
change was guarded by "did I already sync this value", but **start / end / speaker
were not**. So any timeline⇄transcript timing mismatch re-PUT every run — and the
sync's `onTranscriptUpdated()` reloads the transcript, which re-triggers the sync.
On a corrupted job (every cue mistimed) that's an infinite per-cue PUT loop over
all ~367 cues.

Fix: remember the last value we synced for EACH field (text/speaker/start/end) per
timeline item and skip a PUT when it matches — so each cue is written at most once
per distinct value. The reload→re-sync feedback can no longer loop. Genuine edits
still go through (a new value differs from the last-synced one).

Combined with moving job load/save off the event loop, the backend stays
responsive. The corrupted job is still the trigger — re-analyzing (or deleting) it
is the real cure for that record.

# ClipAI — Transcript no longer freezes the container ("Connection lost")

Report: opening a job's transcript "takes forever to load, slows the whole
container, and shows Connection lost." Cause: a job bloated by the earlier
crash-resume corruption (tens of thousands of transcript cues, each with per-word
timestamps) was being parsed, sanitized, and serialized **synchronously on the
asyncio event loop** — so while one request churned through it, every other
request (including the frontend's health ping) stalled and the UI flipped to
"Connection lost."

Fix — move the heavy per-job CPU work OFF the event loop:
- `database._load_job_unlocked`: `json.loads` + Pydantic validation now run in a
  worker thread (this is the hot path — `load_job` runs on every poll).
- `database._save_job_unlocked`: the coerce + `model_dump` + `json.dumps` now run
  in a thread (so progress writes during a run don't stall the loop either).
- `GET /jobs/{id}/transcripts`: the on-read self-heal sanitize and the per-cue
  dump of both tracks now run in a thread.

The loop stays responsive regardless of one job's size, so the UI no longer drops
its connection. (The bloated job itself is still large to load; re-analyzing it on
the current build gives a normal-size, clean transcript — that's the real cure for
that specific record.)

# ClipAI — Diagnostics bundle stays readable (cap the reframe problems list)

A "full logs" export came back as 106k lines — but only 65 were actual runtime
logs; the rest was the diagnostics bundle, dominated by one job's
``reframe_report.problems`` list (~6,200 per-keyframe findings) dumped verbatim.
The on-disk artifacts were already size-capped; the DB report's ``problems`` list
was not.

Fix: the diagnostics export now caps ``problems`` to the first 40 entries with a
``problems_omitted`` count (SSH in for the full report). The aggregate scores at
the top of the report — the part that actually matters for triage — are
untouched. Future bundles are a fraction of the size and actually readable.

# ClipAI — Fix the GUI not loading: stop the Pydantic serializer warning flood

The GUI stopped loading and the backend log filled with thousands of
`UserWarning: Pydantic serializer warnings: PydanticSerializationUnexpectedValue
(Expected TranscriptSegment … input_type=dict)` — one per cue, on every save.

Cause: `translated_transcript` / `transcript` are typed `list[TranscriptSegment]`,
but several callers assign plain **dicts** — the transcript editor dumps segments
to dicts before saving (`s.model_dump()`), and the on-read self-heal sanitizes to
dicts. Pydantic does NOT re-validate on plain attribute assignment, so those dicts
sit in the field unconverted; the next `job.model_dump()` then warns once per cue.
On this job's bloated transcript (thousands of cues from the crash-resume
corruption) that's thousands of synchronous `stderr` writes on every save —
enough to stall the asyncio event loop so concurrent GUI requests can't complete.
The job edits still returned `200`, but the UI couldn't load.

Fix, two layers:
- **Coerce at the save chokepoint.** `database._save_job_unlocked` (which every
  save path funnels through) now converts any dict items in `transcript` /
  `translated_transcript` back to `TranscriptSegment` before `model_dump()`. Load
  already validates dicts→models (`JobResult(**data)`), so reads were clean; this
  makes writes clean too. Defensive: fills missing required fields, never raises.
- **Silence the benign warning process-wide** as a safety net (`models.py`), so no
  un-coerced path (a broadcast, a future caller) can ever reintroduce the
  loop-stalling flood. The data serializes correctly either way.

Note: the underlying transcript bloat is the same crash-resume corruption tracked
elsewhere; a clean run yields a normal-size transcript. This fix makes the GUI
load regardless.

# ClipAI — Transcript panel ordering is now a hard, stable guarantee

Report: "the translate UI doesn't properly order the subtitle lines." The panel
already sorted by timestamp, but the sort was `((a.start ?? 0) - (b.start ?? 0))`
which left two weak spots:

- **Equal-start cues kept their array position.** JS `sort` is stable, so cues
  sharing a start time fell back to *incoming array order*. The translated track
  is handed to the panel in different orders across the 15 s poll (re-analyze
  writes, readability splits, and crash-resume unions interleave it), so those
  ties could visibly reshuffle between polls.
- **A non-numeric `start`** (a numeric string, `null`, or a missing field — which
  a crash-resumed union can produce) made the subtraction `NaN`, and `NaN`
  comparisons make `Array.sort`'s result undefined.

Fix: ordering now uses `compareCues`, a TOTAL order over the cue's own fields —
numeric-safe start, then end, then text — so the on-screen position is a pure
function of cue *content*, never of array index. The same cues always render in
the same order; a poll that hands them over reordered can't reshuffle the panel.
The `.srt` / `.txt` exports use the same comparator, so downloads and the on-
screen list match exactly.

Note: this guarantees ORDER. It does not fix cue *timestamps* that are themselves
wrong — e.g. a long line stamped `0:15` whose content belongs at ~33 min, or a
short phrase repeated at two times. That is crash-resume corruption (this job
re-translated across 3 container restarts and the stored track is a damaged
union), addressed separately by the engine-checkpoint reliability fix (fewer
re-translations) — a clean run is the real cure.

# ClipAI — Stop wasting a doomed GPU attempt on every clip export

A run's backend log showed `clip export attempt 1/2 failed (rc=218): … Nothing was
written into output file … frame= 0` for **all 63 clips** — every clip tried the
GPU encoder (`h264_nvenc`), got nothing, then fell back to CPU. On a 4 GB GTX 1650
the NVENC session can't be created while Ollama is resident (it holds the VRAM
during clip detection/SEO), so the GPU path is doomed for the whole batch —
`detect_gpu_capabilities()` even reports NVENC "available" after its own test fails,
*expecting* this per-clip CPU fallback. The result: 63 wasted NVENC attempts and a
noisy log.

Fix: `_export_clip` now tags each candidate command as GPU or CPU, and the first
time the GPU path fails while the CPU path succeeds it latches a batch-scoped flag
(`_gpu_encode_unavailable`, reset at the start of each export run) so the remaining
clips skip straight to CPU. One real GPU probe per run instead of one per clip; the
clips still export identically (CPU), just without the wasted attempt and the
misleading failure spam.

# ClipAI — Transcript self-heal stops churning the DB (and the misleading log)

The same log showed `Self-healed translated_transcript: 512 → 512 cue(s) (dropped
source-language / duplicate artifacts)` logged **dozens of times** in one run —
with the count unchanged (512 → 512), so *nothing was actually dropped*. Cause: the
on-read heal sorts cues and treats a reorder as "changed", then persists. But mid
run the pipeline still holds the in-memory job and overwrites
`translated_transcript` at translate time, so the DB heal never sticks and every
poll re-sorts → re-persists → re-logs. (The sanitizer itself is idempotent — a pure
reorder converges in one pass; verified — so this was churn, not a heal loop, and it
self-resolved when the run finished.)

Fix: the read path now persists the heal only when it removed real corruption (cues
dropped) **or** the run is already terminal (a reorder-only fix, persisted once).
Mid-run reorders are still served sorted for display but not written, so the DB and
log stop churning. The WARN now fires only when cues were genuinely dropped — no
more "512 → 512 (dropped …)" that dropped nothing.

# ClipAI — A job that dies after detection actually resumes (no more "died at translate → restarted faces")

A job ran through face detection + Whisper transcription + planning, reached the
translate stage, then the container died (the recurring NVIDIA driver RPC
timeout). On auto-resume it re-ran **faces from scratch** instead of restoring
the engine checkpoint — wasting the most expensive ~25 minutes of work on every
revive. The previous build *claimed* a same-build crash-revive resumes from the
checkpoint; in practice it often didn't.

Root cause: the checkpoint **save was failing silently**. `save_engine_checkpoint`
is best-effort — it wraps everything in `except: log + swallow` so checkpointing
can never break a run. But `json.dumps(..., default=hook)` only routes *values*
through the hook; a single non-serializable dict **key** (a numpy int from a
detector, a tuple, etc.) makes the whole `dumps` raise *before* the hook is ever
consulted. The save threw, got swallowed, and nothing on disk recorded the
failure — so the next revive found no checkpoint and re-ran detection +
transcription. Worse, **nothing verified the checkpoint was reloadable**, so the
failure was invisible until the next crash exposed it.

Fix — make the checkpoint durable, self-checking, and self-diagnosing:

- **`_json_safe()` normalizes the whole payload before `dumps`** — including dict
  keys (integral keys → `int` so they round-trip, everything else → `str`) and
  numpy/`set`/`Path`/unknown leaves. The save can no longer raise on a stray type.
- **Verify-after-save**: right after writing, the checkpoint is reloaded and its
  signature re-checked *while the engine output is still in hand*. A write that
  can't be reloaded (corrupt JSON, an empty source SHA the loader always rejects,
  a partial flush) now fails **loudly at save time** instead of silently stranding
  the next revive.
- **`fsync` on file + directory** in the atomic write, so the checkpoint survives
  an abrupt container kill — the exact crash this module exists for.
- **Loud, specific logs**: a failed save is now `ERROR` ("a future resume will
  have to RE-RUN detection + transcription"); a load-miss reports *why* — which
  files are missing, or the exact signature field that differs (`source_language:
  saved='ja' != expected='auto'`, etc.). The next occurrence is diagnosable from
  one log line instead of guesswork.

So a same-build crash-revive of a job that died any time after the engine finished
(bridge, summary, translate, clip export) now restores detection + transcription +
the reframe plan and jumps straight to ~56% — as originally intended. (The
underlying cause of the deaths themselves is the host NVIDIA driver instability,
which is infra, not app code — this makes the recovery actually work.)

# ClipAI — Progress bar tells the truth when a job is revived

When a crashed job auto-resumed, it genuinely restarts the early stages (only
SHA-cached frames are reused; faces/Whisper/translation re-run unless a full
engine checkpoint from the *same build* exists), but the progress bar stayed
pinned at its last value (e.g. 95%) — so it looked like the job would finish in
seconds when it had actually started over. Cause: `ProgressBar` is monotonic
(peak-only) and reset only at exactly 0, but a revive emits 1%+, never 0, so the
peak never dropped.

Fix: `ProgressBar` now follows a **large backward drop** (≥15 percentage points)
as a real restart, while still absorbing small out-of-order jitter. A revived job
now visibly drops to where it actually resumes and climbs from there, matching the
"Resuming after restart…" message.

(Reusing more than frames across a revive is limited by design — the engine
checkpoint that skips detection/transcription is invalidated when you redeploy a
new build between runs, since a new build may analyze differently. A same-build
crash-revive still resumes from the checkpoint.)

# ClipAI — Transcript lines stop shifting while you read them

The Transcript tab re-shuffled cues every few seconds while open. Cause: the
on-read self-heal (added to repair interrupted-run corruption) re-ran
`enforce_readability` — a cue MERGE/split pass — on **every poll** of
`GET /jobs/{id}/transcripts`. That pass is not a fixed point of the sanitizer, so
each poll re-merged into a slightly different cue count and re-persisted it; the
Analysis page (which replaces the list when the length changes) then re-rendered
with reordered lines. That's the "lines move around while viewing" symptom.

Fix — the read path now does **only the idempotent sanitize** (drop
source-language relapse + gross duplication + sort). It heals genuine corruption
**once**, and because sanitize is a proven fixed point, every later poll returns
identical data → the panel makes it a no-op → the transcript stays put. The
readability re-flow (merge) now happens only in the pipeline at translate time
(once per run), never on every read. Verified to converge: pass 1 heals, passes
2+ report no change.

# ClipAI — Clips survive a crash mid-export (no more "No clips detected" after a death)

A job that died mid clip-export (e.g. at 48/63 when the container restarts) came
back with **zero clips** — "No clips detected" — even though detection had found
63 and rendered 48. Cause: clip detection + export is one monolithic call and the
clip list was only persisted *after* the whole thing returned, so a crash in the
long export tail lost the entire list.

Fix — **snapshot the ranked clips before the export loop**. `ClipExtractor.run`
gained an `on_candidates` hook that fires the moment clips are ranked (before any
FFmpeg encode); the pipeline persists them immediately. Now a death mid-export
keeps the clip list (the user sees the clips and can re-export individually via
the per-clip export button), and any clips already rendered are on disk. The
final enriched persist (captions/SEO) still overwrites on a clean finish.

(Same underlying trigger as the transcript/preview corruption: the container
restarting mid-job. This makes the clip stage resilient to it.)

# ClipAI — Preview video no longer stalls on a corrupt/partial cached preview

The in-editor preview could sit on "Loading video…" forever even after analysis
finished. Cause: the browser-preview transcode wrote **directly** to
`browser_preview.v2.mp4`, so a container restart / OOM-kill mid-encode (frequent
on this deployment) left a **truncated** file that still passed the
`size>0 && mtime` freshness check — the `<video>` element was then handed an
unplayable partial file and stalled, with no event to recover from.

Fixes:
- **Atomic preview write** (`browser_preview.ensure_browser_preview`): encode to a
  `.building.tmp.mp4` and `os.replace()` onto the final name only on success, so
  the cached name never points at a partial file. Bumped the cache name to
  `browser_preview.v3.mp4` so any already-corrupt `v2` file is abandoned and
  rebuilt cleanly (the raw source is served meanwhile).
- **Frontend escape hatch** (`VideoEditor`): a 30s watchdog detects a `<video>`
  that never reaches `canplay` and surfaces **"Open video directly ↗"** (loads the
  source in a new tab / native player) + **Reload**, instead of an endless
  spinner. The hard-error state gained the same direct-open link.

(The underlying trigger is the environment restarting mid-job; the atomic write
makes the preview resilient to it, and the escape hatch guarantees the user is
never stuck.)

# ClipAI — Self-heal transcripts corrupted by an interrupted run / resume

A run that suffered a mid-pipeline **container restart** (GPU dropped to CPU) plus
~15 websocket reconnects shipped a `translated_transcript` that was a union of the
source + translated tracks with every cue duplicated ~10× — 584 cues, 37% still
Japanese. The backend itself had persisted a clean 482-cue English track
(`0% still source-script`); the damage happened *afterwards*, in the resume/relay
path, which no amount of translation-logic tuning prevents.

Fix — a defensive **`sanitize_translated_transcript`** (drop cues still in the
source script for a non-CJK target + collapse gross duplication, leaving a clean
track untouched):
- **`GET /jobs/{id}/transcripts`** self-heals on read: if the stored track is
  corrupted it's cleaned, persisted back once (so edit indices stay consistent),
  and served clean — the panel and the download stop showing the garbled union.
  On the real corrupted file this took 584 → 226 cues, 37% Japanese → 0%.
- The pipeline also sanitizes once more right before persisting, so an
  interrupted resume starts from a clean base.
- No-op on a healthy track; never touches a CJK target; idempotent.

(Root cause is environment instability — the GPU keeps vanishing and the
container restarts mid-job. A stable run with no restart produced a clean,
English-only transcript. This change makes the output resilient when a restart
does happen.)

# ClipAI — Stop the CPS splitter from re-shattering merged captions

A deployed run (`build 6b9d08d`) proved the merge worked but was being undone:
the log showed `merged 193 → 174` then the reading-speed (CPS) splitter
**re-exploded it to 503** — so the shipped track was still 43% ≤3-word cues. The
20 cps cap splits every merged sentence straight back into 2-3 word flashes.

Fix — **`SUBTITLE_SPLIT_CPS_TOLERANCE`** (default 1.5). A cue is now kept WHOLE
up to `max_cps × tolerance` (30 cps) and only split above that; the same relaxed
cap governs merging, so complete sentences survive. Extend-into-idle-time still
targets the strict 20 cps wherever there's silence, so lines read at the proper
speed where there's room and only genuinely cramped ones run fast. Re-merging the
last run's output under the new rule cut ≤3-word cues 43% → ~20% and mid-sentence
cues 38% → 30%; the live gain is larger because it prevents the split at the
source. Set the tolerance to 1.0 to restore strict Netflix CPS.

# ClipAI — Save-guard keeps the CLEANER translated track (no source-language relapse)

On long jobs with a mid-run reconnect, ~3% of the shipped "translated" cues came
back in the source language (Japanese), interleaved with their English — even
though the backend persisted a 100%-English track (`0% still source-script`,
grade A). Root cause: a stale/resumed snapshot with the same cue COUNT but source
text re-merged in overwrote the clean track; the existing save-guard only blocked
*empty* incoming saves, not *dirtier* ones.

Fix — the guard now compares the source-script fraction of the incoming vs.
on-disk `translated_transcript` and keeps whichever is more fully translated (a
real re-translation is cleaner, so it still wins). Skipped for CJK targets. This
stops the reconnect-correlated Japanese relapse without affecting normal saves.

# ClipAI — Complete-sentence captions: finish a mid-sentence cue into one line

Follow-up to extend-before-split. Measuring a real run, **42% of translated cues
ended mid-sentence** ("and it's your" → "boobs here.", "Is it the way" / "that
makes" / "someone feel" / "shameless the best?") — they read as incomplete and
choppy. The merge that fixes this only bridged a 3s gap, but the median gap to
the continuation was ~4s, so most fragments slipped through.

Fix — **sentence-completion merge** (`_merge_for_readability` +
`SUBTITLE_SENTENCE_MERGE_GAP_MS`, default 6s). When the previous cue has no
terminal punctuation (it's mid-thought), the merge bridge widens from 3s to 6s
to pull in its continuation, so the line finishes as one cue. A cue that already
ends a sentence still starts fresh (one-sentence-per-cue), and the 2-line /
max-duration / CPS caps still bound the result — so fragments that are genuinely
far apart (sparse, breathy speech) stay split rather than parking one line on
screen for 12s. Simulated on the last run's output it cut ≤3-word cues from 236
→ 82 and mid-sentence cues from 215 → 105.

Both this and extend-before-split now log a line when they fire
(`enforce_readability: merged N → M …` / `extended N/M …`) so a deployed build
is verifiable from the run log.

---

# ClipAI — Less choppy translated captions: borrow idle time before splitting

A translated CJK→EN cue is usually longer than the short source window that
timed it, so the CPS (reading-speed) enforcer used to **shatter** it into 2-3
word flashes — the last run shipped 520 cues from 193 translated lines, 44% of
them ≤3 words. That's the opposite of the YouTube/Netflix look.

Fix — **Pass 0.7: extend-before-split** (`subtitle_formatter.enforce_readability`,
gated by `SUBTITLE_EXTEND_BEFORE_SPLIT`, default on). Before splitting an
over-fast cue, first stretch its on-screen time into the **idle gap after it**
(this clip was only ~24% speech — there's plenty), bounded by the next cue's
start and the max display duration so it can never overrun a neighbour or
inflate the timeline. Only cues that are *still* over the cap after stretching
get split. On the failing example a 23-CPS line drops to exactly the 20 cap by
gaining 0.3 s and stays a single readable phrase. `SUBTITLE_MIN_SPLIT_CHARS`
also raised 10 → 14 so any split that *does* happen can't strand a ~2-word cue.

(Diagnostic note for the 4 GB GTX 1650 run: the dominant quality limiter was
VRAM — `llava:7b` (~4 GB) and `qwen2.5:3b` both OOM'd, so source polishing
failed 0/107 and the LLM circuit-breaker went DEGRADED mid-run. Smaller models
— e.g. `moondream:1.8b` for vision — avoid the cascade. The translated track the
backend *persisted* was 100% English at grade A; any source-language lines in an
exported file came from a snapshot taken across the run's 33-min reconnect.)

---

# ClipAI — Every exported clip now ships an SEO ``.txt`` next to the MP4

Exporting/downloading a clip now also gives you a human-readable ``.txt`` with
everything you'd paste into a social upload form: **viral score** (+ the
hook/flow/value/trend breakdown and reasoning), **title** (and SEO title),
**suggested caption**, **hashtags**, **recommended platform**, **per-platform
SEO** (title/description/tags/tips for each target), description, "why this
works", the **export details**, and the clip's **captions/transcript** (range-
filtered, clip-relative timestamps).

How it works:
- **`backend/services/clip_seo_sidecar.py`** (new): pure-stdlib formatter +
  best-effort writer. Reads every field defensively (pydantic model *or*
  persisted dict), filters captions to the clip window, and renders a tidy
  sectioned report. A sidecar problem never fails the actual export.
- **`clip_exporter.export_clip`** drops `[QUALITY] Title.txt` next to the MP4
  after each render (gated by `CLIP_EXPORT_SEO_SIDECAR`, default on), so **all**
  export paths — UI, API, agent, batch, share — get it for free.
- **Auto-download**: the export's WebSocket `export_complete` now carries a
  `seo_url`; `useEncodingManager` fetches and saves it right after the clip,
  so a single Export click lands both `[1080P] Title.mp4` and
  `[1080P] Title.txt`. The share-page URL rewriter is applied to it too.
- **Re-download**: a new `GET /api/jobs/{job}/clips/{clip}/seo.txt` builds the
  file *fresh* from the live job (so it reflects SEO edits made after export and
  works for clips exported before this shipped). Companion "SEO .txt" links sit
  next to every "Download clip" affordance — the ClipSEO toolbar, the Analysis
  **Exported Clips** list, and the Logs **Exported** list.

---

# ClipAI — Transcript-tab edits now reach the subtitle elements (and back), even with a translation

Editing a line in the Transcript tab (text, speaker, split/merge, delete, insert)
didn't change the subtitle elements on the video — and timeline edits didn't always
land in the transcript either. Root cause: when a translation exists, the tab shows
the **translated** transcript by default, but every edit endpoint blindly wrote to
the **source** transcript. The two tracks have *different segmentation* (one source
cue → several translated cues), so a translated-list index lands on a different
source cue or out of range — the edit silently hit the wrong line or 404'd, so it
"didn't take."

Fix — every single-track edit now declares which track it targets, and the backend
honours it:
- **`backend/routers/jobs.py`**: new `_resolve_transcript_rows(job, target)` /
  `_persist_transcript_rows(...)` helpers pick and save the right list. `target=
  "translated"` edits `translated_transcript` *only when one exists*, else falls
  back to the source. Wired through PUT (update), DELETE, POST (insert) and
  bulk-update-speaker; each response echoes the resolved `target`.
- **`TranscriptViewer.jsx`**: derives `editTarget = (hasTranslation &&
  !showingOriginal) ? 'translated' : 'original'` and sends it on all five edit
  calls, so it always edits exactly the track it's displaying.
- **`VideoEditor.jsx`**: new `transcriptTarget` prop is sent on the forward-sync
  PUT (timeline → transcript) so element edits hit the displayed track too;
  reverse-sync already keys `transcriptIndex` off the same track.
- **`Analysis.jsx`** passes a `transcriptTarget` that flips together with
  `activeTranscript`; **`ClipSEO.jsx`** passes one mirroring its `stableTranscript`.

Net effect: text/speaker/timing/add/delete stay in lockstep both ways, on the
original *and* the translated track, on desktop and mobile.

---

# ClipAI — No false "Reconnecting to container…" during analysis: health probe decoupled from Ollama

The connection banner showed "Reconnecting to container…" mid-analysis even though
the container was reachable and the pipeline was progressing. `useConnectionStatus`
pings every 8 s with a 5 s `AbortSignal` and flips to "reconnecting" after 2
consecutive failures — but it pinged `GET /api/providers/status`, which `await`s a
5 s Ollama `/api/tags` call. During offline processing Ollama is busy serving the
pipeline, so that call is slow; the frontend's 5 s timeout cancels the request
*before its 30 s cache is populated*, so the next poll misses the cache and times
out too → two consecutive failures → false banner.

Fix: a dependency-free liveness probe.
- **`GET /api/health`** (new, `main.py`): returns `{"status":"ok"}` instantly — no
  provider/Ollama/disk work. It only reflects whether the web server's event loop
  is responsive, which is what "is the container reachable" actually means.
- **`useConnectionStatus`** now pings `/api/health` instead of
  `/api/providers/status`. Provider/Ollama health is a separate concern (still shown
  in the models/diagnostics panels).

(The heavy pipeline work is offloaded to threads, so the event loop stays responsive
and the trivial probe returns well under the 5 s timeout even mid-export.)

---

# ClipAI — Preview player loads during analysis (no more blocking browser-preview transcode)

The in-page preview player wouldn't load while an offline job was processing. Root
cause: `GET /api/files/{id}/video.*` (`main.py:serve_file`) `await`ed
`ensure_browser_preview()` for the source video on EVERY request — which re-ran
`ffprobe` per Range request and, for non-browser-friendly sources, kicked off a
full (up to 30-min) ffmpeg transcode. During offline analysis that work contended
with the saturated pipeline, so the `<video>` request hung and the player never
loaded.

Fix — the request path never blocks on ffprobe/ffmpeg now:
- **`browser_preview.cached_browser_preview()`** (new): non-blocking resolution —
  returns the cached `browser_preview.v2.mp4` if fresh, the raw source if we've
  already determined none is needed (a new `.none` sentinel, written once so
  scrubbing an already-compatible MP4 no longer spawns an ffprobe per seek), or
  `None` when unresolved.
- **`serve_file`** serves the cached preview when ready, otherwise streams the RAW
  source immediately (range requests still work), and only schedules preview
  generation **in the background** — and only when `pipeline.is_job_analyzing(id)`
  is False, so the transcode can't steal CPU/GPU from the live pipeline.
- Background generation (`_schedule_browser_preview`) is deduped per source and
  holds strong task refs.

Net: the player loads instantly during analysis (raw source); the polished
browser preview is still produced afterward for incompatible/oversize sources.
New unit tests in `tests/test_browser_preview_nonblocking.py` (added to CI).

---

# ClipAI — Durable PROCESSING LOG: survives tab reload, new device, container restart

The Analysis page's PROCESSING LOG was built purely from live WebSocket messages
into local React state (`activityLog`), with no fetch of history on mount — so a
tab reload, opening the job on another device, or a container restart reset it to
empty and only showed FUTURE events. The full pipeline journey was lost.

Now every user-facing event is persisted server-side and re-fetched on load:
- **`services/job_events.py`** (new, stdlib-only): `broadcast_ws` appends each event
  append-only to `/data/uploads/{job_id}/events.jsonl` — cheap one-line writes that
  never rewrite the multi-MB `job.json`, captured even when no client is connected.
  Keepalive `heartbeat`/`pong` pings are skipped and rapid same-stage `status` ticks
  are throttled (8 s) so the durable log carries meaningful events, not noise; a
  one-time compaction caps file growth on very long runs.
- **`GET /api/jobs/{id}/events`** (new, same owner/admin auth as the other job
  routes) returns the most-recent events in chronological order.
- **Analysis.jsx** hydrates `activityLog` from that endpoint on mount and merges it
  with any live entries (time-ordered, de-duped via `eventToLogEntry` +
  `mergeLogEntries`), so the log renders in full on a fresh tab/device and keeps
  streaming live afterward.
- **PIPELINE PROGRESS tracker is durable too.** Each persisted status event carries
  `stage_id` + `ts`, so the same hydration fetch rebuilds the multi-stage bar's
  state (`reconstructStageState`): the active stage, per-stage durations, pipeline
  start, and elapsed — instead of resetting to a blank bar on refresh. (Stage
  CHANGES are never throttled server-side, so every stage's first event survives and
  the transitions are exact.) The elapsed timer is gated on a non-terminal, loaded
  job so a refreshed COMPLETE job shows a frozen total, not a runaway count.

New unit tests in `tests/test_job_events_persistence.py` (added to CI).

---

# ClipAI — Audio preconditioning runs at 16 kHz (GPU not applicable): ~3× faster offline

Answering "would audio preconditioning speed up on the GPU?": **no — ffmpeg's audio
filters (`highpass`/`afftdn`/`loudnorm`) are CPU-only; there is no CUDA build of
them (ffmpeg GPU accel is video decode/encode/scale).** A torch/torchaudio GPU
rewrite is possible but not worth the VRAM contention on a 4 GB GTX 1650 that
Whisper needs. The real, GPU-free win: the preconditioning filtergraph ran at the
SOURCE rate (e.g. 48 kHz) and only resampled to 16 kHz at output, so `afftdn`/
`loudnorm` chewed ~3× more samples than necessary. `build_precondition_filters` now
prepends `aresample=16000` so the whole chain runs at Whisper's 16 kHz target —
lossless for ASR (Whisper consumes 16 kHz mono regardless) and ~3× fewer samples,
stacking with the earlier "drop afftdn on long videos" change. Tests updated.

---

# ClipAI — "Stuck at faces, no VRAM" was the audio-precondition pass: drop afftdn on long videos + live audio progress

A 128-min offline run looked frozen at the face step with the VRAM bar reading
"No models loaded". The log told the real story: the `frame+audio extraction`
stage runs frame extraction and audio extraction CONCURRENTLY (`asyncio.gather`)
and only advances when BOTH finish. GPU frame extraction completed in ~7 min
(339 frames at 05:03:02), but there was no "Audio extraction complete" line after
it — the audio-preconditioning ffmpeg pass (`highpass,afftdn=nf=-25,loudnorm`) was
still grinding. `afftdn` is an FFT spectral denoiser that runs CPU-bound at only a
few × realtime, so on a 2 h track it alone adds ~15-20 min. The faces/perceiver
step can't start until the stage finishes, so the run *looks* stuck at faces, and
"No models loaded" is actually CORRECT — Ollama was unloaded to free the 4 GB GTX
1650 for Whisper, Whisper hasn't loaded yet, and ffmpeg audio filtering is CPU, not
a GPU model. Nothing was broken; the audio precondition was just the silent long pole.

Fixes:
- **Drop only `afftdn` on long videos** (`pipeline_helpers.build_precondition_filters`,
  used by both `frame_extractor.extract_audio` and the `reframer_audio` fallback).
  highpass + single-pass loudnorm are far cheaper and carry most of the coverage
  benefit, so tracks over `WHISPER_PRECONDITION_DENOISE_MAX_MIN` minutes (default
  45; 0 disables the cap) keep `highpass,loudnorm` and skip the expensive denoise.
  Cuts the 2 h audio pass from ~15-20 min toward ~3-5 min.
- **Live audio progress** (`extract_audio` now takes `video_duration` +
  `progress_callback`; `_run_subprocess_with_file_progress` polls the growing WAV
  size against the expected PCM byte count). The pipeline's `_audio_progress` pushes
  "Preconditioning audio for transcription… N%", so once GPU frames finish the UI no
  longer freezes on the last "Extracted N frames" message — it shows the audio pass
  advancing, which also keeps the stuck-timer reset. Accurate messaging for the
  phase that previously looked dead.

New unit tests in `tests/test_audio_precondition_chain.py` (added to CI).

---

# ClipAI — Transcript tab: clearer blue "now-playing" highlight + smoother mobile autoscroll

The transcript tab already tracks playback (TranscriptViewer computes the active
line from the player's currentTime, highlights it, and auto-centers it via a
manual-override-aware lerp; both the Analysis tab — side-by-side inline player on
desktop, stacked on mobile — and the ClipSEO pages feed it a live currentTime). Two
polish fixes so it reads clearly and scrolls right on both desktop and mobile:
- The active line's highlight was a faint 18% blue tint; bumped to a clear 32% blue
  with a 4px ``#0a84ff`` left bar so the currently-spoken line is obvious.
- Added ``WebkitOverflowScrolling: touch`` to the transcript scroll container for
  momentum scrolling on iOS. (The container already had ``overscroll-behavior:
  contain`` + ``touch-action: pan-y``, so manual scrolling and the auto-center
  coexist on touch devices.)
Both changes live in the shared ``TranscriptViewer``, so they apply on the Analysis
transcript tab and the SEO pages at once.

---

# ClipAI — Much faster clip export: GPU encode + parallel + optional stream-copy

Clip export was the single longest tail of the run — ~48 min for 63 clips in the
latest log. Cause: ``reframer_clipper._export_clip`` re-encoded every candidate
clip on the CPU (``libx264 -preset fast -crf 18``) even though the box has NVENC,
and exported them one at a time. The candidate clips are plain time-cuts of the
source (no reframing/subtitles burned in — that happens later on export), so the
heavy CPU re-encode was pure waste.

Speedups (fastest first; ``_export_clip`` falls through each):
- **GPU encode (default).** Reuses the exporter's ``_gpu_encode_args`` (NVENC/
  VAAPI/QSV, respecting the GPU toggle) at ``CLIP_EXPORT_CRF`` (21, visually
  transparent for review clips; 18 was overkill) — frame-accurate and ~5-10×
  faster than CPU libx264. Falls back to ``libx264 -preset veryfast`` if the GPU
  encode fails. ``-ss``/``-to`` stay before ``-i`` (fast keyframe seek).
- **Parallel export.** ``CLIP_EXPORT_CONCURRENCY`` (default 2) runs several
  independent clip encodes at once via a thread pool; progress is reported on
  completion and stays serialized, so the "Exporting clips… (k/N)" status and the
  stuck-timer still behave. Clips are re-sorted to chronological order after.
- **Stream-copy (opt-in, ``CLIP_EXPORT_STREAM_COPY``).** No transcode at all —
  remux the bytes; the whole export phase drops from many minutes to seconds. The
  cut snaps to the nearest keyframe (a clip may start a second or two early), so
  it's off by default and ideal when you just want fast previews to review.

Net for that 63-clip run: ~48 min → roughly 5-10 min on GPU (×~2 with the default
concurrency), or seconds with stream-copy. New config: ``CLIP_EXPORT_CRF``,
``CLIP_EXPORT_CONCURRENCY``, ``CLIP_EXPORT_STREAM_COPY``. (Separately, fewer clips
also helps linearly — set ``CLIP_COUNT`` if 63 candidates is more than you need.)

---

# ClipAI — Readable captions: merge choppy fragments + recurring-name glossary in cleanup

Follow-up to the fresh run (now fully translated + phantom-free): the transcript
still read choppily — strings of 2-3 word cues ("So am I planning on" / "eating
with" / "you today too") because VAD over-segments slow/paused speech and the
translator emits one cue per fragment. Short cues also fail the readability grade's
duration sub-score (sub-833ms flashes), capping it at B.

A — Greedy phrase merge (the YouTube/Netflix look). New
``subtitle_formatter._merge_for_readability`` runs BEFORE the splitter in
``enforce_readability``: it combines consecutive SAME-SPEAKER cues into the longest
caption that still satisfies every limit — ≤ 2 lines × 42 chars, ≤ 4.5 s, and
≤ the CPS cap (Netflix 17/21 Latin, 13 CJK) — bridging only small gaps (continuous
speech, ≤ ``SUBTITLE_MERGE_MAX_GAP_MS`` = 1200 ms), never a real pause, a speaker
change, or a ``[♪ music ♪]`` marker. So fragments become complete phrases and only
genuinely-long cues are then split at clause boundaries. This lifts the duration +
gap sub-scores (fewer, fuller cues) toward an A and reads far more naturally, the
same on offline and cloud (shared formatter). New config:
``SUBTITLE_MERGE_MAX_GAP_MS``.

B — Recurring-name glossary in the per-cue cleanup. The auto recurring-terms
glossary (``glossary.extract_recurring_terms``) was already fed to the main
``translate_via_llm`` pass; it now also primes the per-cue LLM cleanup
(``_llm_cleanup_untranslated``) so any re-translated stragglers render names the
SAME way (no "Mika"/"Mikka"/"Mikako" drift, no coined name flattened to an ordinary
word). (One-off ASR mis-hearings of a name still need a bigger model or a
user-supplied glossary.json — the recurrence-based glossary only pins names that
actually repeat.)

Regression tests for the merge + word-boundary split added to
``backend/tests/test_repetition_and_timeline.py``.

---

# ClipAI — Subtitle splitter no longer breaks mid-word ("You said th" / "is is …")

A fresh run came back fully translated and phantom-free (the gap-fill + QA fixes
landed), but the readability splitter produced mid-word cue breaks like
``[4:40] You said th`` / ``[4:43] is is delicious cake``. Cause: in
``subtitle_formatter._find_split_point``, the Latin word-boundary fallback used
``text.index(words[mid])`` — which returns the first SUBSTRING match. For
"You said this is delicious cake" the middle word "is" matched INSIDE "th[is]",
so the split landed at column 11 (mid-word). It now splits on the actual
whitespace position nearest the centre, so every emitted piece is whole-worded.
Regression tests added (in CI). This mainly bites LLM-translated cues, which have
no word-level timing and therefore fall through to this text-only split path.

---

# ClipAI — Translation QA on every path: no subtitle is left in the source language

Subtitles were still shipping with source-language cues. The cause was a hole in
where the cleanup ran, not a missing capability:

- The per-cue re-translate QA (`_llm_cleanup_untranslated`) ran **only on the
  offline-NMT path**. The latest run took the **editorial-LLM path**
  (`fraction_untranslated < 0.20` → returned early), so it shipped the residual
  source-language cues *without* cleanup. The Whisper-native path had the same hole.
- The final purity gate only **rejected** a wholesale (>20%) half-source draft
  (keeping the labelled source transcript) — it never **fixed** the ≤20% stragglers.

Why the earlier steps don't cover this:
- **Transcript step** dedups/cleans the *source* transcript (repetition-loop +
  phantom filters). It doesn't translate, so it can't guarantee target-language.
- **Subtitle polishing** (MT post-edit) only *refines already-translated* text and
  is explicitly forbidden from reintroducing the source language — it assumes its
  input is already translated, so it doesn't re-translate source-language cues.

Fix — the QA now runs on the FINAL transcript and on every engine path:
- `_llm_cleanup_untranslated` is now invoked on **all three** `translate_subtitles`
  paths (LLM / Whisper-native / NMT), not just NMT — so no engine can ship
  source-language stragglers.
- A **final QA pass** runs on the polished, deduped, resegmented transcript right
  before the purity gate: it detects any cue still in the source script and
  re-translates it one-by-one with the LLM (idempotent + fail-soft; a clean track
  is a no-op). The existing purity gate remains the backstop.

Combined with the gap-fill flood fix (below — which removes the hallucinated
run-ons at the source, before they ever reach translation), the persisted track is
fully target-language with no repeated/hallucinated lines. (Offline runs without
any LLM are unchanged: the cleanup no-ops without an orchestrator and the purity
gate still guards.)

---

# ClipAI — Phantom flood traced to gap-fill: cap it on mostly-silent video + make the confidence gate actually fire

The latest run's transcript still showed the phantom flood (repeated "Don't let" /
"So nice" / "Hmm." and verbatim Japanese run-ons, many left untranslated). The log
pinned the real cause — and why the earlier phantom filter didn't catch it:

- **Gap-fill was the flood.** On this ~70%-silent (128-min) video the gap-fill pass
  re-transcribed **5326 s (89 min) of "gaps" with VAD off**, adding **653 segments**
  to a 133-segment main pass — almost all hallucinated cues over music/silence. The
  untranslated Japanese run-ons in the output were exactly these (FuguMT can't
  translate looped run-ons, so they shipped in source form).
- **The confidence gate could never fire on them.** Gap-fill only keeps segments
  BELOW its `no_speech_threshold` (0.25), so every gap-fill cue has a LOW
  `no_speech_prob` — but the gate required `no_speech_prob ≥ 0.50`. Result: only
  ~4 cues quarantined on a video drowning in them. And the gate was only wired into
  the main loop, never applied to gap-fill output.

Fixes:
- **`WHISPER_GAP_FILL_MAX_FRACTION` (new, default 0.6):** when uncovered gaps exceed
  this fraction of the video, the content is mostly non-speech and `_gap_fill_pass`
  now SKIPS — the VAD main pass is far more reliable there. Speech-heavy videos
  (small gap fraction, e.g. the Gundam case gap-fill exists for) are unaffected.
  This alone removes the 653-cue flood here (786 → ~133 segments).
- **`WHISPER_PHANTOM_MIN_NO_SPEECH` default 0.50 → 0.0:** the low-confidence
  conjunction (avg < 0.40 AND ≥80% of words < 0.4) is the real discriminator and
  must not be gated behind a HIGH `no_speech_prob` it will never see on gap-fill /
  VAD-kept cues. Real speech essentially never has 80%+ of its words below 0.4
  confidence, so it's preserved.
- **Confidence gate now applied to gap-fill output** (`_gap_fill_pass`), where the
  hallucinations actually live, using the same `is_low_confidence_phantom` helper
  with the no_speech requirement off.

Net: the gap-fill hallucination flood is removed at the source on mostly-silent
content, and on normal content low-confidence gap-fill phantoms are dropped — so the
transcript tracks the real (sparse) dialogue ~1:1 and far fewer cues reach
translation. New regression test in `tests/test_phantom_hallucination_filter.py`.

> Note: this run's audio was extracted before the deploy (old `afftdn` chain, reused
> on resume), so the 16 kHz-resample/afftdn-skip speedups will only show on a fresh
> upload.

---

# ClipAI — Phantom-hallucination removal: TACT confidence gate + unconditional loop dedup (1:1 transcript)

A ~79 %-silent JA source produced a transcript flooded with invented cues that
translated through unchanged, so the output read nothing like 1:1: short English
phantoms over silence — "Don't let" (13×), "So nice" (12×), "Hmm." (11×) — plus
long Japanese run-ons re-emitted VERBATIM 10–11× each (decoder loops over the
quiet stretches). Counting cue bodies in the shipped transcript: **551 of 1806
cues (30 %) were scattered loop-repeats**, and the short English ones weren't
repeats of real dialogue at all — they were silence phantoms.

**Is TACT being used?** Yes — the Temporal Audio Coverage ledger (20 ms bins:
covered_speech / low_confidence / covered_silence / quarantined …) is built every
run and logged ("TACT coverage: … covered_silence 79.2 %, low_confidence 6.6 %,
covered_speech 14.2 %"), and it drives the reframer. **But until now it was a
reporting/perception layer only** — its silence/low-confidence classification was
never fed back to DROP transcript cues. The cues handed to translation were
filtered solely by the exact-match boilerplate blocklist, a `no_speech_prob>0.7`
clamp, and an in-segment repetition test — none of which catch a short,
low-confidence, repeated fragment sitting in silence. That gap is the phantom flood.

Fix — make TACT actually filter the transcript:
- **Confidence-gated phantom filter (new `transcript_dedup.is_low_confidence_phantom`,
  wired as hallucination check #4 in `reframer_audio.transcribe`).** Reuses the
  ledger's OWN low-confidence signal (word conf < 0.4): a cue is dropped when its
  words are overwhelmingly low-confidence (avg < `WHISPER_PHANTOM_MAX_AVG_CONF`
  0.40 AND ≥ `WHISPER_PHANTOM_MIN_LOWCONF_FRAC` 0.80 of words under 0.4) **AND**
  Whisper itself doubted there was speech (`no_speech_prob` ≥
  `WHISPER_PHANTOM_MIN_NO_SPEECH` 0.50). All three must hold, so genuine quiet
  speech — confident words even at a moderate no_speech_prob — is preserved.
  Gated by `WHISPER_PHANTOM_FILTER_ENABLED` (default on).
- **Repetition-loop dedup now runs UNCONDITIONALLY** (was only invoked when the
  gap-fill pass produced segments). The MAIN pass loops too, and on a mostly-silent
  video gap-fill may add nothing yet the primary transcript still carries the
  verbatim repeats. `drop_repetition_loops` keeps the earliest occurrence (one for
  long lines, ≤3 for short interjections; bracketed `[♪ music ♪]` markers exempt).
  Verified on the real transcript: 1806 → 1255 cues (551 loop-repeats removed),
  each long run-on collapsed 11→1.

Net: the long Japanese lines appear once each, the short English silence-phantoms
are removed before they're ever translated (also saving translation budget), and
the transcript tracks the audio 1:1. New unit tests in
`tests/test_phantom_hallucination_filter.py` (+ `backend/tests/test_repetition_and_timeline.py`
added to CI) pin both behaviours.

---

# ClipAI — Offline translation completeness: per-cue plain-text LLM cleanup of NMT leftovers

A 2 h JA→EN run still shipped ~25 % of cues in Japanese. The logs pinned it
exactly: the local LLM ran (`text_completion via ollama qwen2.5:3b … completed
in 128.9s`) but its batched JSON-array output was unparseable
(`editorial model returned no usable output`), so `translate_via_llm` bailed →
fell to FuguMT, which left `185/742 cue(s) untranslated (recovered 0/185)` on the
long colloquial run-ons. The previous leftover-cleanup also used the JSON-array
path, so it bailed the same way.

Fix: the NMT-leftover cleanup (`pipeline._llm_cleanup_untranslated`) now
re-translates each still-source cue **one at a time with a PLAIN-TEXT request**
("translate this line; reply with only the translation"). That's the most robust
local-model call — no JSON to misparse and no cross-cue alignment risk — so it
recovers the cues FuguMT couldn't. It dedups identical run-ons (they repeat
across many timestamps) so it's fast + consistent, is bounded by a cue cap +
wall-clock budget, surfaces an accurate "Recovering N untranslated subtitle(s)…"
status, and is fully fail-soft (any failure keeps the existing cue). New knobs:
`TRANSLATION_LLM_CLEANUP_MAX_CUES` (default 500; 0 disables),
`TRANSLATION_LLM_CLEANUP_BUDGET_S` (default 1200).

Separate ASR-quality issue (not translation): on this ~79 %-silent source Whisper
also emits short phantom repeats ("Don't let", "So nice") that translate through
unchanged — the hallucination blocklist catches fixed phrases but not these.
**Resolved by the TACT confidence gate + unconditional loop dedup — see the entry above.**

---

# ClipAI — Accurate live messaging during Whisper + translation (no false "stuck" banner)

The long Whisper and translation phases showed the red "No progress update —
pipeline may be stuck" banner even though the run was alive (just slow), and the
messaging was inaccurate (the heartbeat said "scene analysis" while Whisper ran;
translation sat at a static 63 % "Translating…").

Root cause: the frontend `heartbeat` handler's comment claimed to "reset the
stuck timer" but never did — so the 15 s heartbeats (and the per-batch
`background_task` translation pings) never reset it, and the timer climbed to the
300 s alarm. And the heartbeat label was derived from the coarse JobStatus, not
the actual sub-phase.

Fixes:
- **Frontend (`Analysis.jsx`):** a flowing heartbeat means the event loop is
  alive and the run is NOT stuck — so `heartbeat` AND running `background_task`
  messages now reset the stuck-timer. The "may be stuck" banner now only fires
  when heartbeats actually STOP (event loop blocked / process dead).
- **Backend (`pipeline.py`):** `_update_progress` takes a `heartbeat_label`
  override so the heartbeat names the real sub-phase — "transcription" while
  Whisper runs inside the ANALYZING_SCENES stage (was the misleading "scene
  analysis"), "face + motion detection" during face analysis.
- **Backend translation progress:** the per-batch translation status now pushes
  a REAL progress update (not just a `background_task` ping) — `translation_progress_pct`
  (extracted + unit-tested) maps the "(a/b)" cue count onto the 63→69 % band, so
  the bar advances and the message reads "Translating subtitles… (310/621)"
  instead of a static label.

---

# ClipAI — Clip export reports per-clip progress (no more false "pipeline may be stuck" banner)

Exporting many reframed clips (e.g. 63 clips × ~45-135 s each ≈ 47 min) is the
longest tail of a run, but the UI sat at ~96 % with a single static "Exporting
top clips…" the whole time and tripped the red "No progress update for 36m — the
pipeline may be stuck" banner. The clipper already called its progress callback
per clip, but the pipeline relay (a) compressed export into ~94→97 % (≈3 integer
ticks across 60+ clips) and (b) only emitted on an integer-% increase with a
static label — and the frontend's stuck-timer keys off `progress_message`
changes, which never changed.

Fix: the export loop now reports a per-clip MESSAGE ("Exporting clip k/N …"),
and the progress relay (`resolve_clip_progress`, extracted + unit-tested) always
forwards a message even when the % is flat (while staying monotonic). The
changing text resets the stuck-timer every clip (~45-135 s), so the alarming
300 s "may be stuck" banner no longer fires during a healthy long export, and
each tick also stamps `updated_at` so backend stale-job recovery sees liveness.
No frontend change needed.

---

# ClipAI — Offline translation no longer ships half-Japanese (LLM timeout + NMT leftover cleanup)

The offline JA→EN translation left ~18% of lines in Japanese. From the logs:
the editorial LLM (qwen2.5:3b) hit its per-batch timeout — `Text completion
timed out after 90s` — on the FIRST batch, so `translate_via_llm` bailed
entirely and fell back to FuguMT, which then left 119/661 colloquial run-on cues
untranslated (`completeness pass recovered 0/119`). The LLM path is the better,
COMPLETE one (it has its own 3-pass source-script retry); it just needed to not
get killed mid-answer.

Two fixes (both, per request):
- **Let the local LLM finish (`translator.py`).** The per-batch translation
  timeout is now a generous, configurable CEILING
  (`TRANSLATION_LLM_TIMEOUT_FLOOR`=180 s, `TRANSLATION_LLM_SECONDS_PER_SEGMENT`=12,
  was `max(60, 5×batch)`) — raising a ceiling never slows the fast path, it only
  stops a slow local model being killed and dropped to NMT. Batches are also
  smaller on Ollama (`TRANSLATION_LLM_BATCH`, auto=8) so each produces a short
  JSON array quickly + reliably.
- **Harden the NMT fallback (`pipeline.py`).** The offline router is LLM-free by
  design, but `translate_subtitles` (LLM-first → NMT) now runs a fail-soft
  leftover cleanup: after NMT, any cue still in source script is re-translated
  with the editorial LLM (leftovers only, bounded, never raises). So even when
  the run falls to FuguMT, the shipped subtitles aren't half-source.

Combined with the previous hallucination-blocklist additions, the translated
track should now be complete English without the "see you next time" / おわり
phantom repeats.

---

# ClipAI — Long videos: face detection no longer 5x over-samples + drop "see you next time"/"おわり" hallucinations

Two issues from a 128-min offline run (GTX 1650):

**Slowness — face/motion detection sampled 5x more than its own cap.** The log
showed `Reframer sample rate: 1.20 fps (128.0 min video, ~9212 samples,
cap=1800)` and `reframer_analysis finished in 3352.0s` (~56 min). The sample
cap (1800) was being overridden by the min-fps *floor* (1.2): for any video
longer than ~25 min, `max(floor, …)` sampled at the floor instead of honoring
the cap, so a 2 h video analysed ~9 200 frames instead of 1 800 — ~5x the
intended work. Fixed: `resolve_sample_fps` (extracted + unit-tested) now treats
the floor as a comfort minimum that only applies while it stays within the cap;
long videos honor the cap (128 min → 1 800 samples → ~12 min instead of ~56).
A configurable `REFRAMER_ABS_MIN_SAMPLE_FPS` (default 0.2) keeps multi-hour
videos from under-sampling.

**Transcript/translation quality — "see you next time" / "おわり" hallucinations.**
The video is ~79% silence, and Whisper hallucinated phantom end-of-segment
phrases over the quiet stretches — "See you next time", "I'll see you next
time", and the bare Japanese 終わり/おわり ("the end") — which then dominated the
translated subtitles. Those weren't in the multilingual hallucination blocklist;
added them (and the JA また次回/また来週 family). Exact-cue match only, so real
dialogue (e.g. おわりにしましょう) is untouched.

NOTE: this run ALSO had the offline NMT (FuguMT) leave ~18% of cues
untranslated (long un-punctuated run-on lines) after the local LLM (qwen2.5:3b)
hit its 90 s per-chunk timeout — that half-Japanese output is a separate,
model-capability issue still being addressed.

---

# ClipAI — Transcription no longer times out on long videos (the real "translation didn't run" cause)

A 2 h video came back with `0 transcript segments`, an empty summary ("no
transcript was available"), and no subtitle translation. Root cause from the
logs: the reframer's `transcribe()` RE-EXTRACTED audio from the video with the
preconditioning chain (`highpass + afftdn denoise + loudnorm`) under a hardcoded
**240 s** timeout — but that exact chain takes ~10 min on a 2 h file (the main
pipeline's own `audio.wav` extraction logged 624 s). So it timed out, Whisper
got no audio, and with no transcript the pipeline silently skipped translation
and produced an empty summary. Translation was never the problem — there was
simply nothing to translate (mode-independent: same in offline and cloud).

Fixes (`backend/services/reframer_audio.py`):
- `transcribe()` now **reuses the pipeline's already-extracted `audio.wav`**
  (same job dir, identical preconditioning chain) instead of re-extracting —
  eliminating ~10 min of duplicate ffmpeg work AND the timeout entirely.
- The fallback extraction (when no `audio.wav` exists) now uses a
  **duration-scaled timeout** (floor 10 min, ~realtime + slack) and catches
  `TimeoutExpired` so a slow/odd codec falls back to a plain copy instead of
  killing the whole transcript.

Supporting changes:
- `pipeline.py` now emits a visible job warning + WS notice when analysis
  produces **0 transcript segments**, so a transcription failure can't masquerade
  as a clean COMPLETE with an empty summary again.
- `pipeline_checkpoint.py` `CHECKPOINT_VERSION` bumped to 2 so the empty-transcript
  checkpoint the failed run cached is invalidated — the next analyze/resume
  re-runs the engine and transcribes correctly.

Verify on the host: re-analyse the 2 h video; the log should show
`[AUDIO] Reusing pre-extracted audio.wav … skipping redundant re-extraction`,
a non-zero `transcript segments` count, a populated summary, and — for a
non-English source — the subtitle translation running.

---

# ClipAI — Auto-resume now defers + serializes its runs (transcript/translation completes)

Follow-up to the resume feature: a revived job could finish WITHOUT its
transcript polish or translation. Root cause was timing, not the resume logic
(which reuses the exact same post-engine code path). Auto-resume fired its
`run_analysis` as a fire-and-forget task *during* container startup, so the
resumed pipeline raced the Whisper/GPU preload and the model + settings/auth
warmup. The AI stages — transcript polish, subtitle translation, summary — then
hit GPU/model contention (and, for cloud, keys/overlay not yet applied) that a
normal user-triggered run never sees, so they failed-soft and the job completed
with a raw/untranslated transcript.

Fix (`backend/main.py`): resumes are now **queued and drained by a single
background task** that (a) waits a grace period for startup to settle
(`CLIPAI_RESUME_DELAY_S`, default 30s) and (b) runs each resumed job to
completion **sequentially** — never two heavy runs at once on the shared GPU.
A revived job therefore executes in the same fully-warmed environment a
continuous run does and completes every step (polish → translation → summary →
clips → COMPLETE) identically. New test:
`tests/test_auto_resume.py::test_resume_drainer_runs_jobs_sequentially_after_delay`.

Verify on the host: analyse a non-English video, restart the container
mid-run, and confirm the log shows `Auto-resume: starting deferred run …` after
the grace delay, the `subtitle_translation … complete` background-task event,
and a populated translated transcript on the finished job.

---

# ClipAI — Resume failed/interrupted jobs (engine checkpoint + auto-resume on restart)

When the container went down mid-analysis (restart, crash, OOM), the worker
thread died and the job could only be RE-ANALYSED FROM SCRATCH — re-running the
single most expensive stage (face/motion detection + Whisper transcription +
the reframe planner) every time. The frame+audio extraction was already cached,
but nothing after it was. This adds a true resume.

**1. Engine checkpoint (`backend/services/pipeline_checkpoint.py`).** Right after
the reframer engine finishes, its in-memory `PerceptionResult` + `RenderPlan`
are serialized to a per-job `checkpoint/` dir. On the next run, if a checkpoint
exists for THIS exact source + analysis config, the pipeline restores it and
skips straight to bridge → summary → clip detection — no re-detection, no
re-transcription. The round-trip is faithful for everything the post-engine
stages read: the int-keyed timeline dicts (`face_timeline`, `motion_timeline`,
…) are restored to **integer** keys (downstream does `int(t_ms / 1000)` on them
and would crash on JSON's string keys). The millisecond-resolution
`coverage_ledger` is deliberately NOT checkpointed — it's read only by the
planner (already run by checkpoint time), and serializing/rebuilding its
tens-of-thousands of 20ms bins in pure Python on resume held the GIL long
enough to starve the event loop and freeze the live `/diagnostics/gpu-status`
VRAM poll. Dropping it keeps the checkpoint tiny and resume non-blocking.

Reuse is gated on a *signature* — source SHA-256, sample-fps, aspect ratio,
source language, vocal-separation setting, and a fingerprint of the reframer
env flags — so a stale checkpoint is never used after the source or analysis
settings change. Bypass with `CLIPAI_FORCE_REANALYZE=1`.

**2. Auto-resume on container restart (`backend/main.py`).** Startup recovery
now: (a) completes jobs that already have results, then (b) **re-queues**
result-less interrupted jobs to RESUME from the checkpoint (continue where they
left off) instead of marking them FAILED. A persisted `resume_attempts` counter
caps this at `CLIPAI_MAX_RESUME_ATTEMPTS` (default 3) so a job that crashes the
container can't relaunch itself forever — past the cap it's marked FAILED.
Disable the whole behavior with `CLIPAI_AUTO_RESUME=0` (reverts to the old
fail-and-wait-for-manual-re-analyse flow). The counter resets on COMPLETE and
on a user-triggered re-analysis.

New env knobs: `CLIPAI_AUTO_RESUME` (default on), `CLIPAI_MAX_RESUME_ATTEMPTS`
(default 3), `CLIPAI_FORCE_REANALYZE` (force a clean engine re-run).

Verify on the Unraid host (no GPU/weights/media here, so this is unit-tested +
code-traced only): analyse a video, `docker compose restart app` mid-run, and
confirm the log shows `RESUME: restored engine checkpoint … skipping detection
+ transcription` and the job finishes without re-running Whisper. New tests:
`backend/tests/test_engine_checkpoint.py` (serialization round-trip, int-key
restoration, signature gating) and `tests/test_auto_resume.py` (startup
complete/resume/fail routing + attempt cap).

---

# ClipAI — Transcription recall on music-heavy content (vocal separation + preconditioning)

Diagnosed from a real run's `clipai_logs_*.txt` on the Gundam Wing episode:
whole dialogue sections (e.g. 2:55–4:27, 10:08–11:08) came back as **silence**.
The log showed `covered_silence: 64.7%`, `0 hallucinations quarantined`, and
music suppression dropping only **3** cues — so the loss was NOT over-zealous
music suppression. The audio is dialogue under a loud music/SFX bed that
Whisper's VAD hears as no-speech and drops, even on the vad-off gap-fill pass.

Three stacked fixes (priority order):

1. **Vocal separation before ASR (`vocal_separator.py`).** Run Demucs
   `--two-stems vocals` as a *subprocess* (so all its GPU memory frees on exit)
   BEFORE Whisper loads, then transcribe only the isolated vocal stem. CUDA
   with a small `--segment` to fit the 4 GB GTX 1650, automatic CPU fallback,
   self-healing to the original audio on any failure. Threaded through
   `ReframeEngine → Perceiver → AudioIntelligence.transcribe(audio_path_override=)`.
   Gated by `VOCAL_SEPARATION_ENABLED` (default on, **no-op until
   `pip install demucs`**; htdemucs downloads ~80 MB on first run → needs net).

2. **Audio preconditioning now reaches the ASR.** `WHISPER_AUDIO_PRECONDITION`
   (highpass + afftdn denoise + loudnorm) was only applied to the
   diarization/music copy; the reframer transcribe path re-extracted RAW audio.
   Mirrored the exact chain into `transcribe()`'s own extraction (with a raw
   fallback) so faint speech is recovered as documented.

3. **Music suppression no longer deletes real dialogue.** Suppression inside a
   "music" span now drops only non-lexical vocalisations (`ああああ`/`lalala`),
   keeping lexically-diverse dialogue when the spectral classifier mislabels a
   loud-BGM scene as music (`SUBTITLE_MUSIC_SUPPRESS_VOCALIZATIONS_ONLY`).

Proper-noun errors (Darlian→"Dorian", "Hatsune Miku") are the **empty Custom
Vocabulary glossary**, not a bug — populate Settings → Custom Vocabulary.

Verify on the Unraid host (no GPU/weights/media here, so this is unit-tested +
code-traced only). After rebuild + `pip install demucs`, the
`clipai_logs_*.txt` should show: a `vocal_separation` stage, `[AUDIO] Using
pre-separated vocal track`, a higher `covered_speech` / lower
`covered_silence`, and the 2:55–4:27 / 10:08–11:08 dialogue present. New tests:
`backend/tests/test_vocal_separator.py`, extended `test_music_marking.py`.

---

# ClipAI — Offline Mode: one switch runs the whole pipeline on the local GPU

Settings → AI Provider now has an **Offline Mode (Local GPU)** toggle. One
switch routes all four stages a user thinks about — clip detection,
transcription, translation, and polishing — onto the local GPU (the GTX 1650)
with no cloud calls, and a four-stage status grid shows where each stage
actually runs (LOCAL / CLOUD), updated live from the backend.

It drives the existing self-hosted machinery rather than a parallel knob:
turning it on sets `SELF_HOSTED_MODE` and resets the per-engine overrides to
Auto, so clip detection runs on the local Ollama vision model (Replicate
disabled), the editorial chain collapses to `["ollama"]` (polishing +
LLM-first translation, then offline Whisper/NMT), and Whisper transcription
stays local as always. A collapsible "Advanced overrides" block keeps the
per-engine Local/Cloud/Auto dropdowns for power users. The control moved out
of the Prompts tab into AI Provider where the user asked for it.

Sequencing for a 4 GB card: the pipeline already evicts Ollama before
Whisper and releases Whisper VRAM before the editorial LLM. Added the missing
hand-off — `_free_editorial_vram_before_local_clips` unloads the editorial
Ollama model (keep_alive=0) and flushes the torch allocator right before
local clip detection loads its vision model, so the editorial LLM and the
vision model never have to share the GPU. No-op in the cloud.

New: `Settings.resolve_stage_source(stage)` maps the four user-facing stages
onto the two engine knobs; `/api/self-hosted/settings` now returns a `stages`
block. Covered by `tests/test_offline_mode_settings.py`.

---

# ClipAI — Fix the real reason translations were half-Japanese: source="auto"

The LLM translation logged "0% still source-script" yet the track was ~40%
Japanese. Root cause: the job's source language is "auto", and the language-
purity check only counted CJK for KNOWN CJK source codes — so for "auto" it
returned 0, the gate was effectively OFF, and the LLM (told to translate from
"the source language") left long narration / lyrics untranslated, which then
sailed through.

Fixes:
- The purity check is now CONTENT-based and TARGET-aware (`fraction_untranslated`):
  any CJK in the output of a non-CJK-target translation counts as untranslated,
  regardless of how the source was declared. So it works for "auto".
- `translate_via_llm` runs a completeness cleanup: after the main pass it
  re-translates any cue still in CJK script (up to 3 passes), so nothing is left
  in the source language.
- Resolve source "auto" → the language Whisper detected, so the LLM prompt is
  precise ("Japanese → English") and the gate has a concrete target.

Net: the translated track is fully target-language — the gate can no longer be
silently disabled by an "auto" source, and any stray untranslated cue is caught
and re-translated.

---

# ClipAI — Stop the post-edit from reverting LLM translations back to Japanese

The LLM translation (previous change) works: logs show `LLM translation: 116/116
segments -> en (0% still source-script)` — a perfect, complete English draft. But
the persisted track came back ~40% Japanese. Cause: the MT post-edit ran on it
"source-aligned" — comparing each English line against the Japanese SOURCE — and
reverted chunks (long narration, song lyrics) back to the source language.

Fix:
- Skip the post-edit entirely when the engine is the LLM (_used_llm): the LLM
  translated the source directly, so its output is already final, clean target
  text — there is nothing to "post-edit toward the source".
- For the NMT / Whisper paths that still post-edit, add a safety net: if the
  post-edit RAISES the source-script fraction (i.e. it reintroduced the source
  language), discard it and keep the pre-edit translation.

Net: the perfect LLM translation is persisted intact — fully English, every cue.

---

# ClipAI — Fundamental rethink: translate with the LLM, not Whisper's translate task

The translated track kept coming back HALF JAPANESE — dialogue in English, but
song lyrics, narration and hard segments left in the source language. Root cause
is architectural: translation was hardcoded "offline-only, never an LLM", forcing
Whisper's task='translate' as the primary path. On mixed/music content Whisper's
translate task silently TRANSCRIBES (source language) instead of translating, so
the output is a JP/EN mix. The coverage guard only checked timeline coverage, not
whether the result was actually English, so it passed the mix through.

Rethink — translate the SOURCE transcript text-to-text with the editorial LLM
(the same orchestrator already used for summary / SEO / polish):
- New translate_via_llm(): batches the source cues, asks the model to translate
  EVERY numbered line (lyrics + narration included) into a JSON array, maps the
  result 1:1 back onto the cues (timing + speaker preserved), applies the
  glossary, and splits-and-retries on any batch the model mangles. It bails
  cleanly (returns None) if the model is unusable, so offline users fall back.
- translate_offline now tries the LLM FIRST (TRANSLATION_PREFER_LLM, default on),
  then Whisper-native, then offline NMT.
- Language-purity gate (fraction_source_script): any "translation" still >20%
  source-script (CJK) is REJECTED — applied to both the LLM result and the
  Whisper-native result — so a half-translated track is never persisted; it
  falls through to a complete engine instead.

Net: every cue is translated, 1:1, with the high-quality model you already have
configured — no more Japanese left in the English track.

---

# ClipAI — Stop stale saves from wiping the translation (the real "not translated" bug)

Direct from a corrupted job on disk: `translated_transcript` was 0 (the pipeline
had persisted 279 English segments), the source was a doubled 217-segment copy,
and status had reverted to `detecting_clips` — all AFTER the run finished. A
whole-object `save_job` (a transcript-segment edit / reverse-sync that loaded the
job microseconds before the translation landed, then wrote its stale snapshot
back) clobbered the good data. Every display-side fix was futile because the
English was being destroyed on disk.

Fix — an anti-clobber guard in the central write (`_save_job_unlocked`):
- **Never wipe a non-empty `translated_transcript` with an empty one.** A real
  re-translation writes a NEW non-empty value, which still wins; only the
  empty-wipe (always a stale snapshot) is blocked. Applies to every write.
- **Never revert a terminal status** (complete/failed/cancelled) to a
  non-terminal one — but only on the `save_job` whole-object path, so a
  deliberate re-analysis reset via `update_job_status` is still allowed.

Both re-read the current on-disk copy under the per-job lock and refuse the
downgrade. Tests cover wipe-protection, terminal-status protection, legit
re-translation overwrite, and re-analysis still resetting status.

---

# ClipAI — Reliable translated-transcript loading (lightweight endpoint)

Across incognito + multiple devices the Analysis page kept showing the SOURCE
(Japanese) transcript even though the backend had persisted the English
(`translated_transcript`). Root cause: the UI reads the FULL job over the network
to get the translation, and that payload is large enough that over a tunnel the
fetch didn't reliably complete — so `translated_transcript` never reached any
client and `activeTranscript` fell back to the source.

Fix: a dedicated `GET /jobs/{id}/transcripts` endpoint that returns ONLY the
source + translated transcripts (+ status) — a tiny response that always lands.
The Analysis page pulls it on load, on every pipeline status change, and on a
short timer while running, and merges the result in. The English now appears as
soon as it's persisted, regardless of how big the rest of the job is.

---

# ClipAI — Whisper subtitle cues land on the audio (word-timed resegmentation)

Audit follow-up on the Whisper-native path. Whisper's translate task emits a few
LONG, multi-sentence cues but with per-word timestamps. The pipeline split those
into sentence cues only AFTER the MT post-edit — which clears word timestamps
when it rewrites text — so the split always fell back to a char-length
PROPORTIONAL guess. On a 40 s cue that puts internal sentence boundaries several
seconds off the audio (a short sentence with long dictation gets a tiny slice; a
long sentence spoken fast gets too much).

Fix — carry the word timestamps through and split BEFORE polishing:
- `_whisper_native_translate_segments` now passes Whisper's `words` through.
- The Whisper-native path resegments by sentence **before** the post-edit, using
  those word times for accurate boundaries (the glossary is re-applied to the
  split cues, since the word-timed split rebuilds text from the pre-glossary
  words). The post-edit then polishes the already-correctly-timed short cues
  (it preserves start/end), and the post-edit-stage resegmentation is skipped for
  this path. The NMT path (source-aligned 1:1) is unchanged.

Tests: word-timed vs proportional split (boundary at 30 s vs <15 s), and a
pipeline test that a long Whisper cue splits word-timed before polish.

---

# ClipAI — Stop dropping legitimately-repeated subtitle cues + steadier diarizer threshold

Two tuning fixes from comparing against repo-60 (which never mangles translated
transcripts because it does **no** post-translation processing — it saves the raw
cues).

**Don't loop-drop NMT output.** The translated transcript runs a
`drop_repetition_loops` dedup that removes Whisper's hallucinated repetition loops
(it loops on music/silence). But it drops **any** recurring long line across the
whole timeline — which on a song/AMV deletes legitimately repeated chorus and
narration lines (kept only the first of 6, in testing). Offline NMT translates the
source 1:1 and never hallucinates loops, so its repeats are real: the loop-drop is
now **gated to the Whisper-native path only**. NMT-translated choruses survive.

**ECAPA diarizer threshold 0.55 → 0.70.** 0.55 over-split noisy / music-heavy
audio straight into the 8-speaker cap. 0.70 is a steadier default for ECAPA
agglomerative clustering (configurable via `LOCAL_DIARIZER_THRESHOLD`).

---

# ClipAI — Fix: jobs stuck at "translating" forever (interrupted-run recovery)

Root-caused from a real run: a job re-analysed after completing died at the
translate stage and then **span at "translating" forever** in the UI — no clips,
source-language subtitles. The first run had finished cleanly (clips + 348
translated segments verified on disk); a second run re-entered, set status
`translating` (the only place that status is written), and never finished.

The bug: both recovery paths used a **hardcoded** in-progress set
(`analyzing_scenes, extracting_frames, generating_summary, detecting_clips`) that
**omitted `translating`** (and `transcribing`/`queued`). So a job stuck at those
statuses was neither reconciled-to-complete (it has results) nor failed — it just
spun.

Fix:
- Derive the non-terminal set from the `JobStatus` enum, so it can **never drift
  out of sync** again (covers every running phase).
- Staleness-guarded recovery (`_recover_stale_jobs`): a non-terminal job that no
  live worker is advancing (old `updated_at`) is flipped to **COMPLETE** when it
  has results, or **FAILED** when it doesn't. Startup runs it unconditionally
  (workers are dead); the periodic self-heal (every 10 min) uses 30 min / 2 h
  thresholds so it **never touches an in-flight run**. Stuck jobs now recover
  without a restart.

---

# ClipAI — Coverage guard: don't let sparse Whisper-translate leave Japanese gaps

Follow-up to the Whisper-translate reuse work. On a music/lyric-heavy video
(e.g. an anime AMV), Whisper's audio→English *translate* task skips/merges the
singing, so it emitted only ~35 cues where the source transcription had captured
122 — leaving long stretches with no English subtitle (they read as "still
Japanese"). Because the reuse change made Whisper-native run on the 4 GB card
instead of NLLB, those gaps surfaced in real output.

Fix: a **coverage guard**. After a Whisper-native pass, compare how much of the
audio timeline it covers against the source transcript; when it covers less than
`WHISPER_TRANSLATE_MIN_COVERAGE` (default 0.6) of the source, discard the sparse
output and translate the **full source via offline NMT** instead — so every
source cue gets an English line. Dialogue-heavy videos (where Whisper-native
covers ~all the speech) are unaffected and still use the higher-accuracy
single-pass translate. Threshold is configurable.

---

# ClipAI — Local audio diarization without a HF token + Whisper-translate reuse

Two local-first quality wins that need no external API and no token.

**Speaker diarization without a HF_TOKEN.** Without a token, pyannote's gated
model can't load and diarization fell back to a *visual* left/right mouth-motion
heuristic — useless for off-screen or audio-only voices, and it collapses
same-side speakers into one. There's now a real LOCAL audio diarizer
(`local_diarizer.py`): SpeechBrain ECAPA-TDNN speaker embeddings on Whisper's
speech segments, clustered (scipy, cosine) into who-spoke-when, emitting the
same `{time_ms: "SPEAKER_xx"}` timeline pyannote does — so speaker fusion and
the "Speaker N" labelling work unchanged. The ECAPA checkpoint is a PUBLIC
model (no token) fetched once (~80 MB) into `/data/models`, then fully offline.
Order is now: pyannote (if a token is set) → **local ECAPA** → visual heuristic.
It runs on CPU by default to avoid GPU contention during perception. Quality:
pyannote 3.1 (token) > local ECAPA > visual-only.

**Whisper native →English translate reuses the loaded model (works on 4 GB).**
Whisper's audio→English translate is the best LOCAL English path but was skipped
whenever <4 GB VRAM was free, because it would *reload* a second model — so on a
4 GB card it never ran. The analyze stage now KEEPS the transcription Whisper
engine loaded when an English translate is pending (instead of releasing it
early), the translate gate REUSES that resident model (no second load, so the
4 GB reload gate no longer applies), and the per-pass pre-flight uses a lower
free-VRAM floor when reusing (1.5 GB vs 3.0) so it can stay on the GPU — with a
clean OOM→CPU fallback and a defensive, idempotent VRAM release before the VLM
summary. Net: high-quality local Whisper translation for English targets even on
small GPUs; non-English targets still use NLLB + the AI post-edit.

---

# ClipAI — Translate to ANY language, completely (no cap, no source-language leftovers)

Subtitle translation must work — and finish — for whatever language the user
picks, with the transcript in the *target* language, not the source.

**Any target language, no "unsupported" stop.** The offline NLLB engine covers
~200 languages, but the ISO→Flores map (and the friendly-name
`SUPPORTED_LANGUAGES` the `translate-subtitles` endpoint validated against) only
listed 23 — so picking, say, Czech/Greek/Hebrew/Persian was rejected outright,
and an auto-detected *source* outside the 23 silently failed. Both maps now cover
~100 common languages (kept in sync), the endpoint guard accepts any
Flores-capable target (capability, not a short name list), and the Upload
language dropdown was widened to match.

**No source-language leftovers for any pair.** The completeness retry was
CJK-only — it caught Japanese left in an English track but not, e.g., English
left in a German track. It now flags an untranslated cue for ANY pair (output
still in a CJK source script when the target isn't CJK, OR a substantial cue the
engine echoed back unchanged) and retries it per-cue (which chunks long inputs).
The offline path has no early-stop cap; the rate-limit/consecutive-failure bail
only ever lived in the unused LLM translator, never in the NMT path. Net: the
persisted `translated_transcript` (what the UI shows when a target is set) is
fully in the user's chosen language.

---

# ClipAI — Complete offline translation + AI post-edit, and no phantom speaker color

Three fixes from a `ja→en` run whose subtitles came back half-Japanese with a
duplicate speaker swatch.

**Duplicate speaker colors.** Non-speech cues (`[♪ music ♪]`) are emitted with
an EMPTY speaker. The transcript-derived speaker list (`Analysis.jsx`,
`TranscriptViewer.jsx`) included that empty string as a distinct "speaker", so it
rendered as a phantom, unnamed swatch whose palette-by-position color collided
with a real speaker's — the two identical oranges. Both speaker lists now skip
blank/whitespace speakers (and de-dupe after trim).

**Translation left long cues untranslated.** The log said `70/78 changed → en`,
but the 8 misses were the long run-on cues (Whisper loops) — and those dominate
by volume, so the output read as mostly Japanese once the readability splitter
fanned them out. NLLB was handed each long cue whole, exceeding the fixed
256-token decode cap, and a failed cue silently kept its source text. Now the
offline engine is made *complete*:

  * `NMTTranslator`/`OpusMTTranslator` CHUNK over-long source cues at sentence →
    clause → hard boundaries before translating, and scale the decode budget
    with input length (was a fixed 256) — so a long run-on can't truncate into
    untranslated source.
  * `translate_with_context` skips the fragile context-join (which loses its `¶`
    markers when long) and goes per-segment for long batches.
  * A **completeness pass** in `_translate_via_nmt` detects any cue still in the
    source script (non-CJK target) and retries it per-cue (which chunks), so the
    offline engine never returns source-language text. Still offline-only — the
    AI is never called to translate.

**AI polishing now closes the quality gap (MTPE).** Per request, the offline NMT
still does the base translation, but the editorial LLM step is now machine-
translation *post-editing* instead of "readability-only": it aggressively
rewrites the rough NMT draft into natural, professional subtitles while
preserving meaning, line count, and timing. It runs as ONE dedicated, source-
aligned pass (the readability reflow stays in the steps after it), so each draft
line is shown next to its ORIGINAL source line as ground truth — letting the
model repair mistranslations without translating from scratch. The translation-
mode length guards were loosened (no word-count clamp) so fluent rewrites aren't
rejected. Gated on `AI_TRANSCRIPT_CORRECTION` as before.

---

# ClipAI — Offline NMT uses the GPU when there's room (4 GB card), CPU fallback

A `ja→en` run finally translated offline via NLLB end-to-end (tokenizer fix
landed: `saved tokenizer file … → using NMTTranslator … for ja→en`), but it
picked CPU and crawled: the device heuristic looked at *total* VRAM (`≤4 GB →
cpu`) instead of what's *free*. Translation runs after Whisper's VRAM is
released, so a 4 GB GTX 1650 has ~2.6 GB free and NLLB-600M int8 needs only
~1 GB. NMT_DEVICE=auto now decides on **free VRAM** (`_NLLB_CUDA_MIN_FREE_GB`,
1.8 GB) and uses the GPU when the model + workspace genuinely fit — and a CUDA
load error (OOM) now **retries on CPU** instead of dropping to the LLM, so cuda
is never fatal. Net: much faster offline translation on small cards, same safety.

---

# ClipAI — Run translation right after transcription/speakers (before summary)

Per request, subtitle translation now runs **immediately after transcription +
speaker assignment**, before the VLM summary and clip stages — instead of after
the summary. In `_run_analysis_inner` the translate+polish block was moved above
the summary block (new order: analyze → **translate+polish** → summary → clips →
post-clip finishers). It operates on the freshly transcribed, speaker-labelled,
deduped, music-marked transcript; the summary then runs on the polished source
transcript, and the clip-dependent finishers (caption refresh + Auto-SEO) still
run after clips. Progress stays monotonic (analysis 62% → translate 63% →
summary 70% → clips 80%); the summary is persisted right after it is generated so
post-clip Auto-SEO still sees it. Translation already ran before clips (so a
clip-stage failure can't skip it); this moves it earlier still.

---

# ClipAI — Offline NMT: save the SentencePiece tokenizer with the converted model

The torch.load fix let the NLLB convert succeed (log: `converted to int8 … kept
599.4 MB`), but translation *still* fell back to a stalling Ollama path. Cause:
CTranslate2's converter writes `model.bin` + the CT2 vocab but **not** the
SentencePiece tokenizer the runtime needs, so `NMTTranslator._model_files_present`
was False → `pick_local_engine("ja","en")` returned None → `NMT: no local model …
unsupported — caller will fall back to the LLM` (the `:free` model was correctly
skipped, then `qwen2.5:3b` limped through 59/78 segments with JSON parse errors).

Fix:
- The convert now fetches the tokenizer alongside the model — `sentencepiece.bpe.model`
  for NLLB, `source.spm`/`target.spm`/`vocab.json` for Opus-MT — reusing the HF
  snapshot the convert already pulled (a copy, not a re-download).
- `ensure_nllb_downloaded` **fails loud** if the tokenizer (or `ctranslate2` /
  `sentencepiece`) is still missing, instead of silently falling back to the LLM.
- **Repair shortcut:** an existing `model.bin` with no tokenizer (the broken
  state already on the user's `/data`) is fixed by fetching just the ~5 MB
  tokenizer — no 2.4 GB re-convert. So the next run self-heals and translates
  offline via NLLB (Ollama is never reached).

---

# ClipAI — Translation AI dropdown filter + translation-readability polish

Two follow-up requests after the offline-NMT fix:

**Translation AI dropdown — only models that can translate.** The dropdown was
fed the full text-model list, which (on OpenRouter's catalog) includes reasoning
/ "thinking" models whose chain-of-thought breaks the strict JSON-array the
batch translator parses (the `…-1.2b-thinking:free` the user hit) and
image/audio generators. `available_models()` now also returns a dedicated
`translation` list — text models minus those two classes (`_is_translation_capable`)
— and the Translation AI dropdown uses it. Against the live 343-model catalog
this excludes 39 (28 reasoning, 11 image/audio gens) and keeps every standard
instruct translator; ids that merely contain an `o` (e.g. `grok-2`, `claude-opus`)
are not false-matched by the o1/o3/o4 rule.

**Offline-first, OpenRouter polishes readability only.** Offline NMT already
translates first; the OpenRouter editorial model only polishes the result. That
polish now runs in a new **`mode="translation"`** profile: a readability-only
persona (`_SYSTEM_PROMPT_TRANSLATION`) that fixes punctuation / capitalisation /
spacing / obvious MT grammar glitches but **must not re-translate, change
meaning, reorder, merge/split, or change timing**. It replaces the old
ASR-correction framing (which would "phonetically correct" already-correct
translated words), and the tight length band + word-count guard are forced on so
the LLM can't drift into paraphrase. Timing and segment count were already
preserved (those fields aren't in the LLM's output); this protects *meaning* too.

---

# ClipAI — Offline translation reliability + transcription quality redesign (TACT)

A 24.5-min Japanese anime episode (Gundam Wing) with subtitle target = English
came back with: subtitles still in Japanese, the opening narration repeated at
six separated timestamps, hallucinated vocalisations over the music bed,
mistimed cues, and mangled song lyrics. Three stacked failures, fixed in
priority order. Full design rationale: `docs/redesign-transcription-translation.md`.

> Branch note: this work was developed on `claude/sharp-lamport-JBXmK` (the
> session's designated branch), which is identical to the task's stated base
> `fix/translate-step-reliability`.

## Task 1 (P0) — Fix the NMT convert bug so offline translation can succeed

**Root cause.** `nmt_translator._convert_with_cleanup` did
`os.makedirs(target_dir, exist_ok=True)` and then
`converter.convert(target_dir, quantization="int8", force=False)`. CTranslate2's
`TransformersConverter.convert` *refuses* a pre-existing output dir unless
`force=True` — and the `makedirs` had just created it. So **every** offline NMT
download raised, the cleanup ran only after the failure, and the caller fell
back to the LLM. Offline NMT had never succeeded on any branch.

**Fix.** Convert into a **fresh sibling temp dir** on the same volume (a path the
converter creates itself, so `force` is irrelevant), then promote it onto
`target_dir` with an atomic `os.replace` (clearing any stale/partial target
first). On any convert/download failure, remove the temp dir **and** any partial
`target_dir`, so a retry is never blocked by a stale directory and a half-written
model is never visible as "present". The HF-intermediate-cache redirect+cleanup,
the disk-space guard, and the Opus-MT pair cap are all retained.

**Network / disk.** The converter pulls NLLB-200 from `huggingface.co` (must be
reachable) — a ~2.5 GB transient full-precision download into a redirected HF
cache that is deleted afterwards, leaving the ~600 MB int8 CT2 model under
`/data/models/nllb/…`. Documented in the function docstring.

**Follow-up — second blocker found in the 2026-06-02 log.** With the dir bug
fixed, the convert reached model loading and hit a *different* deterministic
failure: `transformers ≥ 4.50` refuses `torch.load` of a `pytorch_model.bin` on
`torch < 2.6` (CVE-2025-32434) via `check_torch_load_is_safe()` — and NLLB-200 /
Opus-MT ship **only** `.bin` (no safetensors), while this repo pins
`torch==2.5.1+cu121` (torch 2.6 has **no** cu121 wheel, so bumping it would force
a whole cu121→cu124 CUDA-base migration). Result: the convert raised *“…require
users to upgrade torch to at least v2.6… does not apply when loading files with
safetensors”* and fell back to the LLM every time. Fix: `_allow_trusted_torch_load()`
temporarily neutralises that over-cautious version guard **only around our
convert** of trusted, HTTPS-fetched official HF checkpoints (loaded with
`weights_only=True` — exactly the case the guard itself skips when the caller
opts out), then restores it. No torch bump, no CUDA-base change; a no-op on
torch ≥ 2.6 and harmless if a future transformers renames the symbol.

## Task 2 (P0) — Offline NMT the guaranteed default; never grind a free model

- Offline-NMT-as-default was already correct in
  `translator._resolve_translation_engine` (auto → NLLB when no cloud keys +
  `NMT_AUTODOWNLOAD`); the convert bug was what stopped it from ever running.
- **`:free` model policy.** Before touching the orchestrator LLM, the router now
  detects whether the model the LLM path would actually call is a `:free`
  OpenRouter model (the translation override, else the orchestrator's active
  OpenRouter model, else settings). If so it **skips the OpenRouter LLM entirely**
  — no probe, no 10-minute 429 storm — and falls through to a local Ollama model
  if one is configured, otherwise raises `TranslationFailedError` **fast** with an
  actionable reason (use offline NMT / a paid or local model).
- The retained untranslated transcript is no longer logged as "clean"; it is the
  **source-language** transcript (deduped, explicitly *not* a translation). The
  fail-loud `translation_failed` status + reason are unchanged.

## Task 3 — Kill repetition-loop generation at the source (transcription)

`condition_on_previous_text=True` on the main Whisper pass drove the
repetition-loop pathology through the musical opening. New `_decoding_kwargs()`
helper (signature-filtered like `_vocab_bias_kwargs`, so an older faster-whisper
build never sees an unknown kwarg) now supplies, on the main + native-translate
passes:

- `condition_on_previous_text` defaulting **OFF** (`WHISPER_CONDITION_ON_PREVIOUS_TEXT`);
- `no_repeat_ngram_size=3`, `compression_ratio_threshold=2.4`,
  `log_prob_threshold=-1.0`, `repetition_penalty=1.1`, and a `temperature`
  fallback ladder — so degenerate/looped output is **rejected by the decoder**
  (re-decoded hotter) instead of emitted.

The gap-fill pass forces `condition_on_previous_text=False` regardless of the
global default (it lives over music/quiet regions where priming is harmful).

## Task 4 — Music-aware suppression + fuzzy hallucination quarantine

- **Music suppression.** `audio_analyzer.mark_and_suppress_music` classifies the
  audio **once**, drops Whisper "speech" cues that sit ≥ 60 %
  (`SUBTITLE_MUSIC_SUPPRESS_OVERLAP`) inside a sustained music-only span
  (≥ `SUBTITLE_MUSIC_MIN_SEC`) — the hallucinated lyrics/vocalisations — then
  positively labels the span `[♪ music ♪]`. BGM-under-dialogue is classified
  `speech` (not `music`) by the spectral classifier, so real dialogue over music
  is untouched. The pipeline now suppresses, then marks.
- **Fuzzy repetition quarantine.** `transcript_dedup.drop_repetition_loops` now
  catches near-identical long blocks (> 24 normalised chars), not just exact
  ones: (a) an **exact normalised-key** match with unbounded timeline reach —
  `_normalize_text` strips Unicode punctuation + whitespace, so the narration
  re-emitted at six separated timestamps with only punctuation/spacing
  differences collapses to one (the case the old exact-only filter missed); plus
  (b) a **length-gated fuzzy match (≥ 0.9)** over a bounded recent window for
  genuine ASR character drift. The length gate (`min/max ≥ 0.9`) means a
  distinct *longer* line that merely *contains* a shorter kept line is never
  dropped, and the window keeps the pass O(n) instead of O(n²) on long episodes.
  Short interjections keep the exact cap of 3; `[♪ music ♪]` markers are exempt.

## Task 5 — Otter-parity segmentation + glossary

- The dialogue-only source (markers held out) is passed through
  `sentence_segmenter.resegment_by_sentence` **before** translation, so the NMT
  sees clean one-utterance-per-cue units aligned to word timestamps rather than
  run-on blocks; the target side still resegments post-translation.
- Per-noun glossary enforcement already flows into the NMT path
  (`apply_glossary`) and is left in place.
- TACT heterogeneous-engine consensus (a Parakeet/Canary second pass) is left as
  documented future work — explicitly optional, VRAM-risky on the 1650, and not
  verifiable in this environment.

## New / changed settings

`WHISPER_CONDITION_ON_PREVIOUS_TEXT` (False), `WHISPER_NO_REPEAT_NGRAM_SIZE` (3),
`WHISPER_COMPRESSION_RATIO_THRESHOLD` (2.4), `WHISPER_LOG_PROB_THRESHOLD` (-1.0),
`WHISPER_REPETITION_PENALTY` (1.1), `WHISPER_TEMPERATURE_FALLBACK` (ladder),
`SUBTITLE_SUPPRESS_SPEECH_IN_MUSIC` (True), `SUBTITLE_MUSIC_SUPPRESS_OVERLAP` (0.6).

## Verification

**Done here** — 16 dependency-light unit tests
(`tests/test_transcription_translation_redesign.py`, all green):

- Convert fix with a converter mock faithful to CTranslate2's `force=False`
  refusal — the fresh-target case **fails against the old code, passes against
  the fix**; stale-target replacement; failure cleanup + retry-unblocked.
- `:free` detection + the fast-skip path: `text_completion` is **never called**
  (no 429 grind); fail-fast when no Ollama; fall-through to Ollama when set.
- `_decoding_kwargs` defaults (cond_prev OFF), signature-filtering for old
  builds, `**kwargs`-only safety, and the gap-fill override.
- Fuzzy quarantine: six near-identical narration copies → one, dialogue + 3×
  short interjection kept; music suppression: hallucinated lyrics dropped,
  marker + dialogue kept; source resegmentation: run-on → one-utterance cues.

**Must be run on the Unraid host** (this sandbox has no GPU, no
faster-whisper/CTranslate2 weights, and no media, so a full pipeline run +
fresh `clipai_logs_*.txt` cannot be produced here):

1. Build/deploy on the branch, preserving `/data` (so the NMT model persists)
   and `.env`. First `ja→en` job: expect log lines
   `NMT: neutralised transformers' torch<2.6 torch.load guard …`,
   `NMT: downloading + converting NLLB …`, `NMT: … converted to int8 at
   /data/models/nllb/…`, `NMT: using NMTTranslator (…) for ja→en`, and **no**
   `convert failed (… upgrade torch to at least v2.6 …)` and **no**
   `text_completion attempting via openrouter` for translation. Re-run: expect
   `NLLB … already downloaded` (no re-download, no convert error).
2. Re-run the Gundam Wing video (target=English) and confirm against the
   reference transcript: opening narration appears **once** at its real time; the
   song renders as `[♪ music ♪]` (no looped lyric fragments); no cues over
   silent/music-only spans; one clean utterance per cue. Capture the log + a
   side-by-side of the new output vs the reference.

---

# ClipAI — Make translate-then-polish actually run (decouple from clip extraction)

A user picked **English** as the translate-to language for a 24.5-min Japanese
video, but the subtitles came back in the original Japanese — untranslated and
unpolished. The pipeline log for job `5ee727f1` proved the target was captured
correctly (`detected_language=ja`, `subtitle_language=en`, plan logged as
`translate-then-polish in target language`) and that the source-language polish
was then **skipped** "in anticipation" of translation — but the translation it
deferred to **never ran**. Clip extraction hit repeated Replicate `429`
(`rate limit … less than $5.0 in credit`) on the VideoLLaMA3 calls, and because
the critical-path translate+polish call sat **after** the clip stage, the clip
failure bypassed it. The job still finalized `complete`, with the raw Japanese
transcript — the worst-case outcome (neither translated nor polished).

All changes on this branch (`fix/translate-step-reliability`).

## Root cause

In `backend/services/pipeline.py`, `_run_analysis_inner` logged the plan, **skipped**
the source-language polish/resegment/reflow when a translation was planned, ran
the clip-extraction stage, and only **then** invoked the critical-path
`await _background_post_processing(...)`. So any failure/early-return in the clip
region (the `429`s) reached the outer finalize before translation ever ran.
Meanwhile the source polish had already been skipped — so the transcript was
neither translated nor polished.

## Task 1 — Run translate-then-polish BEFORE clip extraction (decoupled)

- `_run_analysis_inner` now invokes `_background_post_processing(...)`
  **immediately after summary generation and before the clip-detection stage**.
  Subtitle translation no longer depends on clip detection succeeding.
- The clip-dependent work (caption refresh + Auto-SEO) was split out of
  `_background_post_processing` into a new module-level
  **`_run_post_clip_followups(...)`** that runs *after* the clip stage — so the
  parts that genuinely need clips still run once clips exist, while translation
  runs ahead of them.
- The clip stage is wrapped so a Replicate `429` (or any clipper exception)
  **degrades to zero clips and continues** — it can never abort or skip the
  remainder of `_run_analysis_inner`. The post-clip follow-ups are likewise
  best-effort and never abort the finalize.
- `_background_post_processing` is still **awaited on the critical path** (the
  translated+polished transcript is persisted before the COMPLETE save). The
  `_background_post_processing entry`, `Translation engine resolved`,
  `Translate START` log lines now fire on every translating job because the
  call is reached unconditionally.

## Task 2 — Never skip source polish unless translation actually runs

- `_background_post_processing` now **returns an outcome contract**
  (`will_translate`, `translated`, `source_transcript`, `target_transcript`,
  `seo_transcript`, `target_name`, `failed_reason`). `_run_analysis_inner`
  **adopts `source_transcript`** for the COMPLETE save, so the persisted
  `transcript` field is the *polished* source on the no-translation /
  translation-failed paths — never the raw text. (Previously the COMPLETE save
  re-persisted `_run_analysis_inner`'s own raw local `transcript`, silently
  clobbering the polished source the fallback had just written.)
- On translation failure the existing `_polish_source_if_needed()` +
  `_dedup_source_transcript()` fallback runs (polish + readability-enforce +
  dedup in the **source** language). A new **last-resort polish** in
  `_run_analysis_inner` covers the (near-impossible) case where
  `_background_post_processing` itself crashes — so there is **no code path
  where the source polish is skipped but translation does not run**. End state:
  the final transcript is always either (a) translated+polished in the target
  language, or (b) polished in the source language. Never raw + unpolished.

## Task 3 — Fail loud when a planned translation does not happen

- New `JobResult.translation_status` (`None` / `"translated"` /
  `"translation_failed"`) + `translation_error` (short reason) in
  `backend/models.py`, persisted on the job and surfaced over the existing
  websocket channel via the new `_set_translation_status(...)` helper
  (`{"type":"translation_status","state":...}`).
- `_background_post_processing` sets `"translated"` on success and
  `"translation_failed"` + reason on failure. `_run_analysis_inner` adds a
  **fail-loud backstop**: if a translation was *planned* (`subtitle_language` ≠
  source) but produced no target output, it sets `translation_failed` with a
  reason — a planned-but-missing translation can never silently present as a
  clean COMPLETE with source subtitles.
- Explicit call-site logging: `"[job] invoking translate+polish (target=…,
  source=…, will_translate=…)"` is logged immediately before the call, so the
  "it just didn't run" failure is visible in the log next time.

## Task 4 — Status clarity

- New `JobStatus.TRANSLATING = "translating"`. The translating progress update
  now uses this **distinct status** (at 77 %) instead of reusing
  `DETECTING_CLIPS` at 97 %, so a stuck/failed translation is no longer mistaken
  for clip detection in diagnostics. Wired through `PIPELINE_STAGES` (a new
  `translation` stage, 76–80 %), the heartbeat stage labels, the frontend
  `Dashboard` status badge + cancellable set, and the `PipelineTracker`
  (`STAGE_ORDER` / `STAGE_LABELS` / `STAGE_WEIGHTS` — the teal `translation`
  colour was already defined). Non-terminal, so `Analysis.jsx`'s `isProcessing`
  treats it correctly and the `_finalizing_jobs` gate still blocks only
  terminal reverts.

## Files changed

- `backend/models.py` — `JobStatus.TRANSLATING`; `translation_status` +
  `translation_error` fields.
- `backend/services/pipeline.py` — reorder translate before clips; split out
  `_run_post_clip_followups`; `_background_post_processing` returns an outcome
  contract, drops the premature `status=COMPLETE` pin, sets `translation_status`;
  `_set_translation_status` helper; fail-loud backstop + last-resort source
  polish; `_refresh_clips_with_translation` no longer pins COMPLETE (it now runs
  before finalization); `TRANSLATING` PIPELINE_STAGE + heartbeat label.
- `frontend/src/pages/Dashboard.jsx`, `frontend/src/components/PipelineTracker.jsx`
  — render the new `translating` status / `translation` stage.
- `tests/test_translate_step_reliability.py` — new regression tests.

## Verification

Run from the repo root in this environment (the heavy ML deps are lazy-imported,
so `backend.services.pipeline` imports with only `pydantic`/`httpx`/SDK shims):

- `python -m pytest tests/test_translate_step_reliability.py` → **5 passed**.
  Covers: translation success sets `translation_status="translated"` and
  **never** pins `status=COMPLETE`; a `TranslationRateLimitedError` (the `429`
  case) falls back to the polished **source** transcript, sets
  `translation_failed` + reason, and never relabels source as translated; the
  no-translation path returns the source for the COMPLETE save;
  `_run_post_clip_followups` refreshes captions **only** when a translation
  happened and always seeds Auto-SEO from the right transcript;
  `_set_translation_status` persists the field + broadcasts.
- `python -m py_compile backend/services/pipeline.py backend/models.py` → OK.
- `npx esbuild src/pages/Dashboard.jsx src/components/PipelineTracker.jsx
  --loader:.jsx=jsx` (from `frontend/`) → parses clean.
- `tests/test_job_persistence_race.py` → **6 passed** in isolation (no
  regression to the COMPLETE-finalize logic). The only suite failures are
  pre-existing and environment-only: tests importing `backend.services.object_detector`
  (a module that does not exist in this checkout) and `fastapi`/`cv2`/`torch`-backed
  modules that aren't installed here; plus the suite's pre-existing
  `HOME`-set-at-import cross-file ordering fragility (each affected file passes
  alone).

### Acceptance criteria → how to confirm on the live Unraid run

> The live Unraid build/deploy + the actual Japanese-video run (with a real
> Replicate `429` induced by low credit) was **not run from this environment** —
> there is no Unraid host, GPU, Replicate credential or test video here. Build
> and deploy with the standard nohup command on `fix/translate-step-reliability`
> (preserve `/data` and `.env`), then capture a fresh `clipai_logs_*.txt` and
> confirm:

1. **Translation runs to completion** — the log shows, in order:
   `Post-processing plan: source=ja target=en → translate-then-polish` →
   `invoking translate+polish (target=en, source=ja, will_translate=True)` →
   `_background_post_processing entry` → `Translation engine resolved: …` →
   `Translate START: Japanese → English (N segments)` → `Translate DONE` →
   `Polish START on translated text (lang=en)` →
   `Translated transcript readability: grade …` →
   `Persisting translated_transcript (N segments)` → then clip detection → COMPLETE.
2. **Clip failure does not block translation** — induce the Replicate `429`
   (low credit). The log shows `Clip extraction failed: …429…` **after** the
   `Translate DONE` / `Persisting translated_transcript` lines, the job
   completes with `clips=0` (degraded) and the **English** translated transcript
   intact. (Proven structurally + at the unit level by the tests above.)
3. **Translation genuinely can't run** → the transcript is still polished in the
   **source** language (`Source-language transcript polish (fallback path …)` /
   `Source transcript dedup …`), and the job carries a visible
   `translation_status="translation_failed"` with a reason (also broadcast as
   `{"type":"translation_status","state":"translation_failed"}`).
4. **No skip-without-translate window** — the source polish is skipped only when
   `_will_translate` is true, and on that branch `_background_post_processing`
   is always reached (translation runs, or its source fallback / the last-resort
   polish does).

### Before / after transcript (illustrative of the target transformation)

> Illustrative shape, not a capture from a live model run (no NMT/LLM weights or
> test video in this environment):

```
BEFORE (raw source, what the buggy path shipped):
  [00:03] 今日はいい天気ですね。
  [00:07] 散歩に行きましょう。

AFTER (translated + polished, target=en — what this fix ships):
  [00:03] It's such nice weather today.
  [00:07] Let's go for a walk.
```


# ClipAI — Whisper model: show the real one + let the user change it

Two Settings fixes for the Whisper transcription model: (1) the GUI now shows
the model that **actually loaded** (not just the configured value — the 4 GB
GTX 1650 can auto-downgrade), and (2) the read-only "Whisper Model" row is now a
**dropdown** whose pick is persisted, marked as a deliberate user choice, and
applied to the next transcription without a container restart.

All changes on this branch (`claude/inspiring-maxwell-GG90I`).

### Task 1 — Expose the effective (actually-loaded) Whisper model

- `backend/services/reframer_audio.py`: new **sticky** class attrs
  `AudioIntelligence._last_loaded_model_name` / `_last_loaded_device`, stamped by
  `_record_loaded(...)` at every real load (GPU tier success, base fallback,
  cache reuse, CPU reload). Unlike `_cached_model_name` — which
  `_release_whisper_vram()` nulls after a job — these survive the VRAM release so
  the GUI can always report what ran.
- `backend/routers/settings.py`: new helper `_effective_whisper_info()` +
  endpoint **`GET /api/providers/whisper/effective`** (and the same fields added
  to `GET /api/providers/models/available` → `current`). Returns:
  `whisper_model_selected`, `whisper_model_user_set`, `whisper_model_effective`
  (cached → sticky → `null`), `whisper_model_loaded`, and `whisper_downgraded`
  (`effective != selected`). Reads the `AudioIntelligence` class attrs out of
  `sys.modules` **without** importing the heavy reframer_audio module, so it's
  cheap to poll.

### Task 2 — Real dropdown instead of a read-only label

- `frontend/src/pages/Settings.jsx`: the Analysis-Settings "Whisper Model" row is
  now a `<select>` over the valid faster-whisper models (`tiny`, `base`, `small`,
  `medium`, `large-v3`, `large-v3-turbo`, `distil-small.en`, `distil-medium.en`,
  `distil-large-v3`, with 4 GB-oriented VRAM hints; `.en` variants labelled
  English-only). Below it, an effective-model line shows
  *"Selected: large-v3-turbo · Running: medium (downgraded — not enough VRAM…)"*
  when they differ, just *"Running: <model>"* when they match, "Not loaded yet"
  before the first job, and a ↻ refresh button. `backend/routers/settings.py`
  `_WHISPER_MODELS` gained the two distil `.en` entries (`english_only: true`).

### Task 3 — Persist the override and actually apply it

- `_perUserModelPatch('transcript')` now returns
  `{ WHISPER_MODEL: id, WHISPER_MODEL_USER_SET: true }`; the picker saves
  immediately (global `/providers/models/save` **and** the per-user patch).
- `backend/routers/auth.py`: added `WHISPER_MODEL_USER_SET` to the per-user
  `PUT /me/settings` allow-list so the flag actually persists.
- `backend/routers/settings.py` `save_models`: on a model change it now
  **invalidates** `AudioIntelligence` (new `invalidate_cache()` classmethod,
  sys.modules-guarded) so the next job loads the new selection without a restart
  (the old `reload_model()` was a no-op). The cache is also model-name-keyed, so
  a new pick reloads regardless.

### Task 4 — Honest downgrade behavior (and stop overwriting a user pick)

- `backend/main.py`: the startup Whisper auto-upgrade now skips when
  `WHISPER_MODEL_USER_SET` is true — a deliberate `small`/`medium`/`base` pick is
  no longer silently bumped to `large-v3-turbo` on a GPU box. Auto-upgrade still
  applies for the default (non-user-set) case.
- `reframer_audio.py` already records the real loaded model in `_cached_model_name`
  for both the downgrade and base-fallback paths; the new sticky fields mirror it
  for display so the GUI reflects reality.

### Tests

- `backend/tests/test_whisper_effective_endpoint.py` — effective/selected/
  downgraded/sticky reporting + the distil `.en` model list. (5 tests.)

```
$ python3 -m pytest backend/tests/test_whisper_effective_endpoint.py -q
5 passed
```

> The Settings dropdown + the GTX-1650 selected-vs-downgraded screenshot must be
> captured on the GPU box (no GPU/CUDA/faster-whisper here). The frontend JSX was
> verified to parse/transform with esbuild; the backend logic is unit-tested.

---

# ClipAI — Automatic + reliable offline translation, model lifecycle, Whisper fix

Make offline NMT translation the automatic default (no manual Settings download),
fail loudly instead of silently shipping the untranslated source, clean up model
downloads so disk/VRAM don't fill, honor the user-selected Whisper model, and
guarantee a clean, de-duplicated transcript even when translation fails.

All changes on this branch (`claude/inspiring-maxwell-GG90I`).

## New / changed settings (`backend/config.py`)

| Setting | Default | Purpose |
| --- | --- | --- |
| `NMT_AUTODOWNLOAD` | `True` | Auto-download + convert the offline NMT model the first time a translation needs it — no manual Settings step. The LLM is only used if the download itself fails. |
| `NMT_MAX_OPUS_PAIRS` | `5` | LRU cap on per-pair Opus-MT model dirs; least-recently-used pairs beyond the cap are pruned after a new download. NLLB-200 is a single model and is never counted against this cap. |
| `NMT_DEVICE` | `auto` (unchanged) | **Behavior change:** when `auto` and the GPU has ≤ 4 GB total VRAM (e.g. GTX 1650) the NMT engine resolves to **CPU** (OOM-safe on a card Whisper + Ollama already share). Larger GPUs keep CUDA. Set `cuda`/`cpu` to force. |

### Task 1 — Offline NMT auto-downloads on demand (no manual Settings step)

- `backend/services/translator.py`
  - `_resolve_translation_engine()`: with `TRANSLATION_ENGINE=auto` and **no cloud
    keys**, when no local model is on disk it now resolves to **`nllb`** (or
    `opus-mt` for pairs outside NLLB's Flores map) instead of falling straight to
    `llm`. The NMT path downloads it on first use; the LLM is the last resort only
    when `NMT_AUTODOWNLOAD` is off or the download fails.
  - `_translate_via_nmt()`: when no local engine exists and auto-download is on, it
    downloads + converts in a worker thread (`asyncio.to_thread(auto_download_for_pair,…)`),
    broadcasts **"Downloading offline translation model (one-time)…"** over the
    websocket `background_task` channel, re-resolves the local engine, and translates
    offline. Only a *download* failure falls back to the LLM, logged distinctly.
- `backend/services/nmt_translator.py`: new `auto_download_for_pair(src, tgt, prefer="nllb")`
  — NLLB-200 preferred (one model, 200 languages); Opus-MT only when explicitly
  requested or when the pair isn't in NLLB's Flores map.
- **faster-whisper** keeps its own on-first-use model download (unchanged).
- **pyannote diarization** is the only model needing a manual token (`HF_AUTH_TOKEN`);
  absent → `try_load()` returns `False` and the pipeline degrades to the existing
  spatial pseudo-diarization / mouth-motion ↔ audio-energy speaker mapping. Never
  blocks. (Documented; no code change.)

### Task 2 — No more silent fall-through to a rate-limited free model

- `backend/services/translator.py`: new `TranslationFailedError` /
  `TranslationRateLimitedError`. `translate_segments()` detects sustained 429s
  (`ProviderRateLimitError`) and **aborts after 2 rate-limited attempts** instead of
  crawling every batch; the counter **resets on success** so a recovered transient
  429 never trips it. A `:free` translation model logs a one-line warning (not blocked).
- `backend/services/ai_orchestrator.py`: `text_completion()` raises
  `ProviderRateLimitError` (not generic `AllProvidersFailedError`) when every provider
  failed with 429, so callers can fail fast + loud.
- `backend/services/pipeline.py`: translation failure now ends in a visible
  **`translation_failed`** state — a `pipeline_warnings` entry plus a `background_task`
  `failed` broadcast (`state: "translation_failed"`) with an actionable message. The
  source is never persisted under `translated_transcript` when translation didn't
  complete.

### Task 3 — Model lifecycle: download, cache, clean up (`nmt_translator.py`)

- **HF intermediate-cache cleanup:** `_hf_cache_redirect()` points HF cache env at a
  throwaway dir on the **same volume** as the models for the convert, then deletes it
  — only the ~600 MB int8 CT2 model survives. **Bytes reclaimed are logged.**
- **Disk-space guard** before download (~6 GB NLLB, ~3 GB per Opus pair) → clear failure
  instead of a half-written dir.
- **Partial-convert cleanup:** target dir removed if `convert()` raises (clean retry).
- **Opus-MT LRU cap** (`NMT_MAX_OPUS_PAIRS`, `.last_used` markers).
- **VRAM:** `NMT_DEVICE=auto` → CPU on ≤ 4 GB GPUs; confirmed (already correct) that
  `NMTTranslator.unload()` + `empty_cache()` run in a `finally` after translation,
  `_release_whisper_vram()` runs before the NMT load, and NMT is unloaded before polish.

### Task 4 — Honor the user-selected Whisper model (`large-v3-turbo` → `medium` mismatch)

- `reframer_audio.py`: `AudioIntelligence` records `requested_model_name`, logs the
  requested model up front, honors it (falls to **CPU** rather than swapping size when
  GPU VRAM is short — logged), and logs the genuine last-resort change as an explicit
  **`DOWNGRADE: requested <model> … → loading base on CPU`** with free-VRAM numbers.
- `reframer_engine.py` + `pipeline.py`: the **effective** loaded model is surfaced in
  the compute summary (`summary["whisper"]["model"]`/`requested_model`) so the active
  config can't disagree with what ran.
- `settings_overlay.py`: a per-user `WHISPER_MODEL` that differs from the global pin is
  logged loudly (`global=… → per-user=…`). `WHISPER_AUTO_UPGRADE` (non-user-set)
  unchanged.

### Task 5 — Clean timing/dedup even when translation fails (`pipeline.py`)

- New `_dedup_source_transcript()` runs `collapse_adjacent_duplicates` →
  `collapse_overlapping_duplicates` → `drop_repetition_loops` on the **source**
  transcript whenever translation is skipped or fails, mirroring the post-translation
  dedup, and persists the cleaned transcript (removed-count logged). Verified
  `collapse_overlapping_duplicates` collapses identical / near-identical-timestamp
  duplicates (the `[6:23]`/`[6:23]`, `[2:30]`/`[2:30]` cases).

### Tests (dependency-light — no ctranslate2 / torch / GPU)

- `backend/tests/test_nmt_autodownload_and_cleanup.py` — disk guard, HF-cache
  redirect + cleanup, partial-convert cleanup, Opus-MT LRU cap, `auto_download_for_pair`
  routing.
- `backend/tests/test_translation_fail_loud.py` — rate-limit detection, exception
  hierarchy, fast abort on sustained 429s, no-abort on a recovered transient 429.

```
$ python3 -m pytest backend/tests/test_nmt_autodownload_and_cleanup.py \
                    backend/tests/test_translation_fail_loud.py -q
12 passed
```

> The Unraid build/deploy + the two Japanese-video runs (empty `/data/models`, then
> cached) must be run on the GPU box — this dev container has no GPU / CUDA /
> ctranslate2 / faster-whisper / `/data`, so the heavy ML paths can't run here. The
> logic above is unit-tested; the six acceptance criteria are confirmable from a fresh
> `clipai_logs_*.txt` on the box.

---

# ClipAI — Transcription / Translation / Polish fixes

Branch base: `claude/otter-transcription-parity-mTjAh` (identical commit to the
session branch this was developed on). All changes are surgical and leave the
reframer camera / clip pipeline untouched.

## New config key

| Key | Default | Meaning |
| --- | --- | --- |
| `OPENROUTER_TRANSLATION_MODEL` | `""` (blank) | Dedicated OpenRouter model for **subtitle translation**. Blank → falls back to `OPENROUTER_EDITORIAL_MODEL`. Mirrors the existing `OLLAMA_TRANSLATION_MODEL`. |
| `WHISPER_GAP_FILL_MAX_SEC` | `8.0` | Hard ceiling on a single gap-fill segment's length. Longer cues are split at word boundaries so run-on music/narration can't collapse into one timestamp. `0` disables the split. |

---

## Task 1 — Translate, then polish, in the target language (core fix)

**`backend/services/pipeline.py`**

- **`_run_analysis_inner`** now decides up-front whether a translation will
  follow (`_will_translate`, computed from `job.subtitle_language` /
  `job.language` / Whisper's detected language, including the
  "auto-translate non-English → English" default). When it will translate, the
  **source-language** critical-path work is skipped — the heavy LLM polish, the
  sentence resegmentation and the readability reflow no longer run on the
  Japanese text. Music marking + dedup + the readability score still run, so the
  source artifact is still clean. When no translation is needed, behaviour is
  unchanged (full source polish, as before).
- **`_background_post_processing`** was reordered to **translate → polish →
  resegment → reflow → dedup**:
  - (a) translate the source segments to the target language,
  - (b) LLM polish on the **translated** text with `correction_lang = target_lang`
    (the old `_whisper_translated` assumption — that Whisper had already
    translated — was removed; the reframer always runs `task="transcribe"`),
  - (c) sentence resegmentation + readability reflow in the target language,
  - (d) final dedup (adjacent + overlap + repetition-loop),
  - then persist `translated_transcript` + `transcript_readability`, refresh clip
    captions, broadcast the `background_task` websocket events, and seed SEO.
  - If translation produces **0 changed segments** (or throws), it bails to the
    source path and keeps the source transcript — it never persists Japanese
    under `translated_transcript`.
- **Reliability:** the step is no longer a fire-and-forget `asyncio.create_task`
  scheduled after COMPLETE (which was being lost on this deployment — the job
  stalled at `detecting_clips` with the Japanese transcript and zero translation
  in the log). It is now **awaited on the critical path, before the COMPLETE
  save**, so the translated + polished transcript is part of the finalize. The
  clips are reloaded afterwards so the single COMPLETE save carries the
  translated captions; `translated_transcript` / `transcript_readability` are
  preserved by `_persist_complete_job`'s load→merge.
- Explicit `logger.info` lines (all tagged with `job_id`) now mark: entry,
  before/after translate (with the model used), before/after polish (`lang=…`),
  resegment/readability/dedup counts, and the final persist.

## Task 2 — Gap-fill timing + duplicate fixes

**`backend/services/reframer_audio.py`**

- `_gap_fill_pass` now **splits over-long cues** (> `WHISPER_GAP_FILL_MAX_SEC`,
  default 8 s) at word boundaries — and at any inter-word silence ≥ 1 s — so a
  whole OP song / minutes of narration can't collapse into one run-on segment.
  Each emitted sub-cue carries correct **absolute** start/end times.
- The `_inside_gap` overlap guard is applied on **both** the `clip_timestamps`
  path and the full-file fallback path (segments outside a real gap are dropped
  and counted).
- A summary line logs **segments kept / dropped-outside-gap / dropped-as-
  duplicate / max segment length**.
- `_cross_validate_segments` now runs a second, **non-adjacent** collapse pass.

**`backend/services/transcript_dedup.py`**

- New `collapse_overlapping_duplicates()` removes near-duplicates by **timestamp
  overlap + text similarity** (whitespace-insensitive exact, containment, word-
  Jaccard, or CJK char-bigram), not just adjacency — collapsing the repeated
  `作戦名オペレーション・メテオ`-style re-transcriptions. Used in the gap-fill pass,
  `_cross_validate_segments`, the pipeline's final transcript dedup, and the
  post-translation dedup. The existing `collapse_adjacent_duplicates` /
  `drop_repetition_loops` are unchanged (their tests still pass).

## Task 3 — Source-language dropdown → Whisper

**`backend/services/pipeline.py`**

- `ReframeEngine(...)` is now constructed with
  `source_language=(job.language or "auto")`, so the user's upload-page language
  pick reaches `AudioIntelligence.transcribe(language=…)` and the log shows
  `language=ja` (or whatever was picked) instead of always `auto`. A log line
  confirms both the source language and the `subtitle_language` target.

## Task 4 — Dedicated OpenRouter translation model + dropdown

- **`backend/config.py`** — added `OPENROUTER_TRANSLATION_MODEL` (blank → editorial).
- **`backend/services/ai_orchestrator.py`** — `text_completion(...)` takes an
  optional `model_override`; when set and the active provider is OpenRouter, it
  temporarily swaps `provider._editorial_model` (restored in `finally`), mirroring
  the Ollama override pattern.
- **`backend/services/translator.py`** — `translate_segments` accepts
  `model_override`; `translate_segments_with_fallback` passes
  `settings.OPENROUTER_TRANSLATION_MODEL or None` into the probe + full LLM
  translation calls and logs the model used. Polishing keeps the editorial model.
- **`backend/routers/settings.py`** — `OPENROUTER_TRANSLATION_MODEL` added to the
  persisted-keys list, the `SaveModelsRequest` (`translation_model`), the
  `/providers/models/save` writer (+ `.env` upsert) and the
  `/providers/models/available` + save `current` blocks.
- **`backend/app/auth/settings_overlay.py`** + **`backend/routers/auth.py`** —
  `OPENROUTER_TRANSLATION_MODEL` added to the per-user overlay / PUT allow-lists.
- **`frontend/src/pages/Settings.jsx`** — new **"Translation AI (OpenRouter)"**
  dropdown under the Editorial AI controls, populated from the OpenRouter
  catalog, wired through `pendingModels.translation_model` /
  `currentModels.translation_model` with the same buffer-then-save pattern and
  POSTed to `OPENROUTER_TRANSLATION_MODEL`. The **"Editorial AI"** control was
  relabelled/clarified as the **transcript-polishing** model so the two are
  clearly distinct.

## Task 5 — Sanity

- `TRANSLATION_ENGINE='auto'` with only OpenRouter configured still resolves to
  the `llm` engine (`_resolve_translation_engine`), and that path now honours the
  translation-model override.
- No `HF_TOKEN` requirement was added; reframer camera/clip logic is untouched.

---

## Verification

- All changed Python modules compile (`python -m py_compile`).
- `frontend/src/pages/Settings.jsx` transpiles cleanly via `esbuild`.
- The existing `transcript_dedup` regression cases pass, and the new
  `collapse_overlapping_duplicates` + gap-fill split logic were unit-checked
  (run-on cues split to ≤ 8 s with accurate times; overlapping CJK/Latin
  near-duplicates collapsed; non-overlapping distinct lines untouched).

> Note: the live Unraid build/deploy + fresh `clipai_logs_*.txt` capture was
> not run from this environment (no access to the Unraid host). The acceptance
> criteria are satisfied by the code paths above; run the standard nohup build
> on the branch and upload the Japanese test video to capture the before/after
> transcript + the `language=ja → translate ja→en → polish lang=en → persisted
> COMPLETE` log sequence.

---

# Offline NMT translation — finish & verify NLLB-200 + Opus-MT (CTranslate2)

Branch: `claude/clipai-offline-nmt-GsHqe`. Finishes wiring the **offline**
neural-MT path that already existed so subtitle translation can run fully
local (no OpenRouter/LLM call) on the Unraid GTX 1650 box. No new engine was
added and the reframer camera/clip pipeline is untouched. This builds on the
translate→polish reordering above — the offline engine plugs in at the existing
`translate_segments_with_fallback` call, so the flow stays **translate (now
possibly offline NMT) → LLM polish in target language → resegment/readability →
dedup**. Transcript *polishing* still uses the editorial LLM; only the
*translation* step changes engine.

## New config key

| Key | Default | Meaning |
| --- | --- | --- |
| `NMT_DEVICE` | `auto` | Device policy for local NMT (`auto` \| `cpu` \| `cuda`). `auto` → CUDA when available (int8_float16), else CPU (int8). **`cpu` is the safe choice on 4 GB GPUs** (GTX 1650) where Whisper + Ollama already compete for VRAM — a 24-min video still translates in ~a couple of minutes on CPU and can't OOM. Opus-MT always runs on CPU regardless. |

## New dependency

- **`transformers>=4.40,<5`** added to `backend/requirements.txt`. It powers
  `ctranslate2.converters.TransformersConverter`, which the Settings
  "Download model" button uses to fetch + convert NLLB / Opus-MT to the int8
  CTranslate2 format. Without it the download endpoint raised
  `RuntimeError("ctranslate2 with the transformers converter is required …")`.
- **`ctranslate2>=4.0,<5`** now pinned explicitly (was pulled in transitively by
  faster-whisper) so the int8 inference + converter are guaranteed present.
- The conversion is a **one-time, on-demand download — NEVER at startup**. The
  converted int8 model is cached under the models dir (Docker `/data/models`,
  i.e. `/data/models/nllb/...` and `/data/models/opus-mt/{src}-{tgt}/...`) so it
  survives a container rebuild as long as `/data` is preserved.

## Files touched

- **`backend/requirements.txt`** — added `transformers>=4.40,<5`, explicit
  `ctranslate2>=4.0,<5` pin (kept `sentencepiece`). A fresh
  `pip install -r backend/requirements.txt` now imports
  `from ctranslate2.converters import TransformersConverter` cleanly.
- **`backend/config.py`** — new `NMT_DEVICE` setting (`auto|cpu|cuda`, default
  `auto`) with a doc comment about the 4 GB-GPU CPU recommendation.
- **`backend/services/nmt_translator.py`** —
  - `NMTTranslator.__init__` now reads `NMT_DEVICE` from settings (explicit
    `device=` arg still wins) instead of hard-coding `"auto"`.
  - `NMTTranslator.load()` logs a **"Whisper VRAM freed before NLLB load — N MB
    free"** line (+ a defensive `empty_cache()`) when resolving to CUDA, so the
    OOM-avoidance ordering is visible in the log. Keeps `int8_float16` on CUDA /
    `int8` on CPU. Opus-MT still forces CPU (unchanged). `unload()` +
    `torch.cuda.empty_cache()` in the `finally` block is unchanged.
- **`backend/services/translator.py`** —
  - `_resolve_translation_engine` now logs **"NMT: no local model for ja→en,
    falling back to LLM"** when `auto` finds no downloaded model (so it's obvious
    why the offline path didn't run), and logs probe failures.
  - `_translate_via_nmt` now names the **exact engine + model** per job
    (`NMT: using NMTTranslator (facebook/nllb-200-distilled-600M) for ja→en …`
    or `OpusMTTranslator (Helsinki-NLP/opus-mt-ja-en) …`) and logs the no-model
    fall-through.
- **`backend/services/pipeline.py`** — before the `translate_segments_with_fallback`
  call in `_background_post_processing`, when the resolved engine is a local NMT
  (`nllb`/`opus-mt`) it re-runs `_release_whisper_vram(job_id)` defensively and
  logs **"Local NMT engine '…' selected — freeing Whisper VRAM before NMT
  load"**. (Whisper VRAM is already freed during analysis at the post-reframer
  stage; this makes the ordering explicit and robust to future reordering.)
- **`backend/routers/settings.py`** — `NMT_DEVICE` added to the persisted-keys
  list, to `_subtitle_quality_state()` (`nmt_device`), to
  `SaveSubtitleQualityRequest`, and validated against `{auto,cpu,cuda}` in the
  save handler. (`TRANSLATION_ENGINE` was already persisted + saved; the
  `/api/translation/download-model` route already matches the frontend path —
  `router` prefix `/api` + `/translation/download-model`.)
- **`frontend/src/components/SubtitleQualitySettings.jsx`** —
  - The single hard-coded `downloadNLLB()` (always `{engine:'nllb'}`) is replaced
    by an engine-aware `downloadModel(engine)`: **Download Opus-MT (src→tgt)**
    passes `{engine:'opus-mt', source, target}`, **Download NLLB-200** passes
    `{engine:'nllb'}`. Both surface the endpoint's `path`/`message` (success +
    error) instead of failing silently.
  - Added two ISO-code language `<select>`s (the same codes as the Upload page)
    to pick the Opus-MT pair to pre-download. **Default target = `en`**, default
    source = `ja`. The Opus-MT button is disabled when source == target.
  - Added a **Local NMT device** `<select>` (`auto|cpu|cuda`) wired to
    `nmt_device` → `NMT_DEVICE`, persisted with the rest of the form.

## Offline-translation usage (download → select engine → run)

1. **Download a model** (one-time): Settings → Subtitle Quality → Translation
   Engine. For NLLB click **Download NLLB-200 (~600 MB)**. For Opus-MT pick the
   source→target pair (e.g. Japanese → English) and click **Download Opus-MT**.
   The button shows `… ready ✓ (/data/models/…)` on success.
2. **Select the engine**: set **Engine** to `nllb`, `opus-mt`, or `auto`
   (auto prefers Opus-MT, then NLLB, before the LLM). Optionally set **Local NMT
   device** to `cpu` on a 4 GB GPU. Click **Save** — `TRANSLATION_ENGINE` +
   `NMT_DEVICE` persist to `user_settings.json` and survive a container restart.
3. **Run** the Japanese test video with source=Japanese, target=English. With a
   model downloaded the log shows `NMT: using NMTTranslator (…nllb…) for ja→en`
   (or `OpusMTTranslator (…opus-mt-ja-en…)`) — **no OpenRouter/LLM call for
   translation** — followed by the LLM polish pass on the English text. With no
   model downloaded, `auto` logs `NMT: no local model for ja→en, falling back to
   LLM` and uses the LLM path as before.

## Verification

- All changed Python modules compile (`python -m py_compile`).
- The download endpoint returns `{status:"ok", path:…}` for both `nllb` and an
  `opus-mt` ja→en pair once `transformers` + `ctranslate2` are installed; on a
  stock container (before this change) it returned the converter `RuntimeError`.

> Note: the live Unraid build/deploy + `pip install` + fresh `clipai_logs_*.txt`
> capture (pre-download NLLB from Settings, run the Japanese test video offline)
> was not run from this environment (no access to the Unraid host). Run the
> standard nohup build on `claude/clipai-offline-nmt-GsHqe`, preserving `/data`
> (so downloaded NMT models persist) and `.env`, then capture the
> `NMT: using …` / `Whisper VRAM freed before NMT load` / `Polish START on
> translated text (lang=en)` log sequence and the before/after transcript.
