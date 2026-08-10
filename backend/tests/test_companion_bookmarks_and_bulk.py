"""Companion file-browser bookmarks (starred paths, server-persisted) and the
sequential bulk folder import: every video in a shared folder is downloaded and
FULLY analyzed one at a time, until the folder is done, the run is cancelled,
or the ClipAI device runs out of disk space."""

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import backend.routers.settings as S


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fresh bookmark file + empty bulk registry + a temp /data root per test."""
    monkeypatch.setattr(S, "_BOOKMARKS_PATH", str(tmp_path / "bookmarks.json"))
    monkeypatch.setattr(S, "_BULK_STATE_PATH", str(tmp_path / "bulk_state.json"))
    monkeypatch.setattr(S, "_BULK_TERMINAL_POLL_S", 0.01)
    monkeypatch.setattr(S, "_bulk_imports", {})
    monkeypatch.setattr(S, "_bulk_data_root", lambda: str(tmp_path))
    monkeypatch.setattr(S, "_bulk_disk_free", lambda: 10 ** 12)

    async def _no_hb(*a, **k):
        return None
    monkeypatch.setattr(S, "_post_import_hb", _no_hb)
    yield


def _fake_host(host_id="h1"):
    return SimpleNamespace(id=host_id, url="http://192.168.1.9:11500/ollama",
                           token="tok", name="Desktop 4070")


@pytest.fixture
def _host(monkeypatch):
    h = _fake_host()
    monkeypatch.setattr(S, "_companion_by_id", lambda hid: h if hid == h.id else None)
    return h


def _vids(*names, size=1024):
    return [{"name": n, "path": f"D:\\media\\{n}", "is_dir": False,
             "size": size, "ext": n.rsplit(".", 1)[-1]} for n in names]


# ── Bookmarks ───────────────────────────────────────────────────────────────

def test_bookmark_add_list_remove_roundtrip():
    out = asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="D:\\media\\anime", name="anime")))
    assert out["ok"] is True
    out = asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="D:\\media\\films", name="films")))
    # Newest star sorts first.
    assert [b["name"] for b in out["bookmarks"]] == ["films", "anime"]

    listed = asyncio.run(S.companion_bookmarks_list("h1"))
    assert [b["path"] for b in listed["bookmarks"]] == ["D:\\media\\films", "D:\\media\\anime"]

    out = asyncio.run(S.companion_bookmark_remove("h1", "D:\\media\\films"))
    assert [b["name"] for b in out["bookmarks"]] == ["anime"]
    # Removing an unknown path is a no-op, not an error.
    out = asyncio.run(S.companion_bookmark_remove("h1", "D:\\nope"))
    assert [b["name"] for b in out["bookmarks"]] == ["anime"]


def test_bookmark_persists_to_disk_and_is_idempotent():
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="/mnt/media", name="media")))
    # Re-adding the same path replaces (no duplicate) and moves it to the front.
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="/mnt/other", name="other")))
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="/mnt/media", name="renamed")))
    with open(S._BOOKMARKS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert [b["name"] for b in data["h1"]] == ["renamed", "other"]
    # Survives a "restart" (fresh read path).
    listed = asyncio.run(S.companion_bookmarks_list("h1"))
    assert len(listed["bookmarks"]) == 2


def test_bookmark_hosts_are_separate_and_name_defaults_to_basename():
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="D:\\shows\\Gundam Wing")))
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h2", path="/srv/movies")))
    a = asyncio.run(S.companion_bookmarks_list("h1"))["bookmarks"]
    b = asyncio.run(S.companion_bookmarks_list("h2"))["bookmarks"]
    assert [m["name"] for m in a] == ["Gundam Wing"]
    assert [m["name"] for m in b] == ["movies"]


def test_bookmark_normalizes_windows_verbatim_paths():
    # Adds store the CLEAN path even when the client sends the \\?\ form…
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="\\\\?\\C:\\Users\\jalon\\Videos")))
    marks = asyncio.run(S.companion_bookmarks_list("h1"))["bookmarks"]
    assert marks[0]["path"] == "C:\\Users\\jalon\\Videos"
    assert marks[0]["name"] == "Videos"
    # …and re-adding the clean form dedupes instead of duplicating.
    asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
        host_id="h1", path="C:\\Users\\jalon\\Videos")))
    assert len(asyncio.run(S.companion_bookmarks_list("h1"))["bookmarks"]) == 1


def test_legacy_verbatim_bookmark_reads_clean_and_can_be_unstarred():
    # A bookmark file written by an older build with the raw \\?\ path.
    S._save_bookmarks({"h1": [
        {"path": "\\\\?\\C:\\Antigravity IDE", "name": "Antigravity IDE",
         "is_dir": True, "added_ms": 1},
        {"path": "\\\\?\\UNC\\nas\\media", "name": "media", "is_dir": True, "added_ms": 2},
    ]})
    marks = asyncio.run(S.companion_bookmarks_list("h1"))["bookmarks"]
    assert [m["path"] for m in marks] == ["C:\\Antigravity IDE", "\\\\nas\\media"]
    # Un-starring with the CLEAN path removes the legacy entry.
    out = asyncio.run(S.companion_bookmark_remove("h1", "C:\\Antigravity IDE"))
    assert [m["path"] for m in out["bookmarks"]] == ["\\\\nas\\media"]


def test_bookmark_rejects_empty_path():
    with pytest.raises(HTTPException) as e:
        asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
            host_id="h1", path="   ")))
    assert e.value.status_code == 400


def test_bookmark_capped_per_host(monkeypatch):
    monkeypatch.setattr(S, "_BOOKMARKS_MAX_PER_HOST", 3)
    for i in range(5):
        asyncio.run(S.companion_bookmark_add(S.CompanionBookmarkRequest(
            host_id="h1", path=f"/p/{i}")))
    marks = asyncio.run(S.companion_bookmarks_list("h1"))["bookmarks"]
    assert [m["path"] for m in marks] == ["/p/4", "/p/3", "/p/2"]


# ── Bulk folder import: start endpoint ──────────────────────────────────────

async def _start_bulk(req):
    """Call the start endpoint, then yield once so a no-op runner task finishes."""
    out = await S.companion_folder_import(req)
    await asyncio.sleep(0)
    return out


def test_import_folder_unknown_host_404(monkeypatch):
    monkeypatch.setattr(S, "_companion_by_id", lambda hid: None)
    with pytest.raises(HTTPException) as e:
        asyncio.run(_start_bulk(S.CompanionFolderImportRequest(host_id="nope", path="D:\\x")))
    assert e.value.status_code == 404


def test_import_folder_no_videos_400(_host, monkeypatch):
    async def _list(h, path):
        return []
    monkeypatch.setattr(S, "_companion_list_videos", _list)
    with pytest.raises(HTTPException) as e:
        asyncio.run(_start_bulk(S.CompanionFolderImportRequest(host_id="h1", path="D:\\empty")))
    assert e.value.status_code == 400


def test_import_folder_creates_state_and_refuses_second_concurrent(_host, monkeypatch):
    async def _list(h, path):
        return _vids("b.mp4", "a.mkv")
    monkeypatch.setattr(S, "_companion_list_videos", _list)

    async def _noop_runner(bulk_id):
        return None
    monkeypatch.setattr(S, "_run_bulk_import", _noop_runner)

    out = asyncio.run(_start_bulk(S.CompanionFolderImportRequest(
        host_id="h1", path="D:\\media", source_language="ja", target_language="en")))
    assert out["ok"] is True and out["total"] == 2
    st = S._bulk_imports[out["bulk_id"]]
    assert st["status"] == "running"
    assert st["source_language"] == "ja" and st["target_language"] == "en"
    assert all(i["status"] == "queued" for i in st["items"])

    # Only one bulk run at a time — that's the whole "sequential" contract.
    with pytest.raises(HTTPException) as e:
        asyncio.run(_start_bulk(S.CompanionFolderImportRequest(host_id="h1", path="D:\\media")))
    assert e.value.status_code == 409

    # /active re-attaches a reopened dialog to the running bulk.
    active = asyncio.run(S.companion_folder_import_active())
    assert active["bulk_id"] == out["bulk_id"]


def test_import_folder_accepts_explicit_file_selection(_host, monkeypatch):
    """Multi-select mode: the dialog sends the picked files and they run
    through the SAME sequential queue — the old per-file path fired one
    analysis per completed download and piled pipelines onto the GPU."""
    async def _noop(bulk_id):
        return None
    monkeypatch.setattr(S, "_run_bulk_import", _noop)
    out = asyncio.run(_start_bulk(S.CompanionFolderImportRequest(
        host_id="h1",
        files=[
            {"name": "b.mkv", "path": "D:\\m\\b.mkv", "size": 7},
            {"name": "a.mp4", "path": "D:\\m\\a.mp4", "size": 5},
            {"name": "notes.txt", "path": "D:\\m\\notes.txt", "size": 1},
        ],
        source_language="ja", target_language="en")))
    st = S._bulk_imports[out["bulk_id"]]
    assert out["total"] == 2
    # Selection order preserved; the non-video is dropped.
    assert [i["name"] for i in st["items"]] == ["b.mkv", "a.mp4"]
    assert st["folder_name"] == "2 selected videos"
    assert st["folder"] == "selection"
    assert st["source_language"] == "ja"


def test_import_folder_empty_selection_400(_host):
    with pytest.raises(HTTPException) as e:
        asyncio.run(_start_bulk(S.CompanionFolderImportRequest(
            host_id="h1", files=[{"name": "x.txt", "path": "p", "size": 1}])))
    assert e.value.status_code == 400


def test_active_returns_none_when_nothing_running():
    assert asyncio.run(S.companion_folder_import_active())["bulk_id"] is None


# ── Bulk folder import: the sequential runner ───────────────────────────────

def _wire_runner(monkeypatch, tmp_path, *, job_states=None, download_fail=None):
    """Patch the runner's seams; returns the (ordered) event log."""
    events = []
    saved_jobs = []

    async def fake_download(read_url, params, headers, dest, total, on_done):
        name = os.path.basename(params["path"].replace("\\", "/"))
        if download_fail and name in download_fail:
            raise RuntimeError("LAN dropped")
        with open(dest, "wb") as f:
            f.write(b"x" * 64)
        on_done(64)
        events.append(f"dl:{name}")
        return 64
    monkeypatch.setattr(S, "_companion_download", fake_download)

    async def fake_save(job):
        saved_jobs.append(job)
    monkeypatch.setattr(S, "_bulk_save_job", fake_save)

    async def fake_analyze(job_id):
        job = next(j for j in saved_jobs if j.job_id == job_id)
        events.append(f"an:{job.filename}")
    monkeypatch.setattr(S, "_bulk_run_analysis", fake_analyze)

    async def fake_state(job_id):
        job = next(j for j in saved_jobs if j.job_id == job_id)
        return (job_states or {}).get(job.filename, ("complete", ""))
    monkeypatch.setattr(S, "_bulk_job_state", fake_state)
    return events, saved_jobs


def _seed_bulk(monkeypatch, names, **req_kwargs):
    """Create bulk state through the real endpoint with the runner stubbed out."""
    async def _list(h, path):
        return _vids(*names)
    monkeypatch.setattr(S, "_companion_list_videos", _list)
    real_runner = S._run_bulk_import

    async def _noop(bulk_id):
        return None
    monkeypatch.setattr(S, "_run_bulk_import", _noop)
    out = asyncio.run(_start_bulk(S.CompanionFolderImportRequest(
        host_id="h1", path="D:\\media", **req_kwargs)))
    monkeypatch.setattr(S, "_run_bulk_import", real_runner)
    return out["bulk_id"]


def test_bulk_runs_strictly_one_video_at_a_time(_host, monkeypatch, tmp_path):
    events, saved = _wire_runner(monkeypatch, tmp_path)
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv", "c.mov"],
                         source_language="ja", target_language="en")
    asyncio.run(S._run_bulk_import(bulk_id))

    # Download N+1 must come AFTER analysis N completed — never interleaved.
    assert events == ["dl:a.mp4", "an:a.mp4", "dl:b.mkv", "an:b.mkv", "dl:c.mov", "an:c.mov"]
    st = S._bulk_imports[bulk_id]
    assert st["status"] == "complete"
    assert (st["ok"], st["failed"], st["done"]) == (3, 0, 3)
    assert all(i["status"] == "complete" for i in st["items"])
    # Jobs got the dialog's language picks and the canonical video.<ext> path.
    assert all(j.language == "ja" and j.subtitle_language == "en" for j in saved)
    assert saved[0].file_path.endswith("video.mp4") and os.path.isfile(saved[0].file_path)
    assert saved[1].file_path.endswith("video.mkv")


def test_bulk_stops_when_device_runs_out_of_space(_host, monkeypatch, tmp_path):
    events, _ = _wire_runner(monkeypatch, tmp_path)
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv", "c.mov"])
    # Plenty of room for the first video, nothing after it.
    free = {"n": 0}

    def fake_free():
        free["n"] += 1
        return 10 ** 12 if free["n"] == 1 else 0
    monkeypatch.setattr(S, "_bulk_disk_free", fake_free)
    asyncio.run(S._run_bulk_import(bulk_id))

    st = S._bulk_imports[bulk_id]
    assert st["status"] == "out_of_space"
    assert events == ["dl:a.mp4", "an:a.mp4"]          # b was never downloaded
    assert [i["status"] for i in st["items"]] == ["complete", "no_space", "skipped"]
    assert "disk space" in st["items"][1]["error"]


def test_bulk_records_a_failed_video_and_keeps_going(_host, monkeypatch, tmp_path):
    events, _ = _wire_runner(monkeypatch, tmp_path,
                             job_states={"b.mkv": ("failed", "whisper exploded")})
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv", "c.mov"])
    asyncio.run(S._run_bulk_import(bulk_id))

    st = S._bulk_imports[bulk_id]
    assert st["status"] == "complete"
    assert (st["ok"], st["failed"]) == (2, 1)
    assert [i["status"] for i in st["items"]] == ["complete", "failed", "complete"]
    assert st["items"][1]["error"] == "whisper exploded"
    assert events[-1] == "an:c.mov"                    # c still ran


def test_bulk_download_failure_skips_to_next_video(_host, monkeypatch, tmp_path):
    events, saved = _wire_runner(monkeypatch, tmp_path, download_fail={"a.mp4"})
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv"])
    asyncio.run(S._run_bulk_import(bulk_id))

    st = S._bulk_imports[bulk_id]
    assert [i["status"] for i in st["items"]] == ["failed", "complete"]
    assert "LAN dropped" in st["items"][0]["error"]
    # No job row was created for the video that never arrived.
    assert [j.filename for j in saved] == ["b.mkv"]
    # The half-made job dir was cleaned up.
    assert len(os.listdir(os.path.join(str(tmp_path), "uploads"))) == 1


def test_bulk_cancel_stops_after_current_video(_host, monkeypatch, tmp_path):
    events, saved = _wire_runner(monkeypatch, tmp_path)
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv", "c.mov"])

    cancel_calls = []
    monkeypatch.setattr(S, "_bulk_request_job_cancel", lambda jid: cancel_calls.append(jid))

    async def cancelling_analyze(job_id):
        job = next(j for j in saved if j.job_id == job_id)
        events.append(f"an:{job.filename}")
        # User hits Cancel while video 1 is mid-analysis.
        await S.companion_folder_import_cancel(bulk_id)
    monkeypatch.setattr(S, "_bulk_run_analysis", cancelling_analyze)

    async def cancelled_state(job_id):
        return ("cancelled", "")
    monkeypatch.setattr(S, "_bulk_job_state", cancelled_state)

    asyncio.run(S._run_bulk_import(bulk_id))
    st = S._bulk_imports[bulk_id]
    assert st["status"] == "cancelled"
    assert cancel_calls == [saved[0].job_id]           # the running job was signalled
    assert [i["status"] for i in st["items"]] == ["cancelled", "skipped", "skipped"]
    assert events == ["dl:a.mp4", "an:a.mp4"]          # b and c never started


def test_bulk_cancel_aborts_inflight_download(_host, monkeypatch, tmp_path):
    _, saved = _wire_runner(monkeypatch, tmp_path)
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4", "b.mkv"])

    async def cancelling_download(read_url, params, headers, dest, total, on_done):
        await S.companion_folder_import_cancel(bulk_id)
        on_done(1)                                     # raises _BulkCancelled
        raise AssertionError("on_done should have aborted the download")
    monkeypatch.setattr(S, "_companion_download", cancelling_download)

    asyncio.run(S._run_bulk_import(bulk_id))
    st = S._bulk_imports[bulk_id]
    assert st["status"] == "cancelled"
    assert [i["status"] for i in st["items"]] == ["cancelled", "skipped"]
    assert saved == []                                 # no job was ever created
    # The partial download's job dir was removed.
    assert os.listdir(os.path.join(str(tmp_path), "uploads")) == []


def test_bulk_follows_a_stall_revived_job_to_completion(_host, monkeypatch, tmp_path):
    """A run the stall watchdog cancelled + requeued reports QUEUED again
    mid-way. The bulk must FOLLOW the revive to its real outcome instead of
    misreading the requeue as a failure and racing to the next video."""
    events, _ = _wire_runner(monkeypatch, tmp_path)
    states = iter(["queued", "transcribing", "complete"])

    async def revived_state(job_id):
        return (next(states, "complete"), "")
    monkeypatch.setattr(S, "_bulk_job_state", revived_state)

    bulk_id = _seed_bulk(monkeypatch, ["a.mp4"])
    asyncio.run(S._run_bulk_import(bulk_id))
    st = S._bulk_imports[bulk_id]
    assert [i["status"] for i in st["items"]] == ["complete"]
    assert st["status"] == "complete" and (st["ok"], st["failed"]) == (1, 0)


def test_bulk_state_persists_and_resumes_after_restart(_host, monkeypatch, tmp_path):
    """The queue itself survives a container restart: the persisted state is
    reloaded, a mid-download item is reset for a fresh download, and the
    runner is respawned. Terminal runs are reloaded for display only."""
    async def _list(h, path):
        return _vids("a.mp4", "b.mkv")
    monkeypatch.setattr(S, "_companion_list_videos", _list)

    async def _noop(bulk_id):
        return None
    monkeypatch.setattr(S, "_run_bulk_import", _noop)
    out = asyncio.run(_start_bulk(S.CompanionFolderImportRequest(
        host_id="h1", path="D:\\media")))
    bulk_id = out["bulk_id"]

    # Simulate dying mid-download of video 1, plus an older finished run.
    st = S._bulk_imports[bulk_id]
    st["items"][0]["status"] = "downloading"
    st["items"][0]["done_bytes"] = 12345
    S._bulk_imports["old-done"] = {"status": "complete", "items": [], "total": 0}
    S._bulk_persist()
    S._bulk_imports.clear()                    # the process died

    spawned = []

    async def rec_runner(b):
        spawned.append(b)
    monkeypatch.setattr(S, "_run_bulk_import", rec_runner)

    async def _resume():
        n = await S.resume_interrupted_bulk_imports()
        await asyncio.sleep(0)
        return n
    assert asyncio.run(_resume()) == 1
    assert spawned == [bulk_id]
    revived = S._bulk_imports[bulk_id]
    assert revived["items"][0]["status"] == "queued"      # partial file redone
    assert revived["items"][0]["done_bytes"] == 0
    assert revived["cancel"] is False
    # The finished run is visible again but was NOT respawned.
    assert S._bulk_imports["old-done"]["status"] == "complete"


def test_resumed_runner_adopts_analyzing_and_skips_terminal(_host, monkeypatch, tmp_path):
    """After a restart: terminal items keep their outcome, a mid-analysis item
    is FOLLOWED (its job was revived by startup auto-resume — never
    re-downloaded), and the still-queued tail imports normally."""
    events, saved = _wire_runner(monkeypatch, tmp_path)
    waited = []

    async def fake_wait(job_id):
        waited.append(job_id)
        return ("complete", "")
    monkeypatch.setattr(S, "_bulk_wait_terminal", fake_wait)

    st = {
        "bulk_id": "r1", "host_id": "h1", "folder": "D:\\m", "folder_name": "m",
        "status": "running", "error": "", "total": 3, "done": 1, "ok": 1,
        "failed": 0, "current": -1, "cancel": False, "started_ms": 1,
        "source_language": "", "target_language": "",
        "items": [
            {"name": "a.mp4", "path": "D:\\m\\a.mp4", "size": 5,
             "status": "complete", "job_id": "ja", "error": "", "done_bytes": 5},
            {"name": "b.mkv", "path": "D:\\m\\b.mkv", "size": 5,
             "status": "analyzing", "job_id": "jb", "error": "", "done_bytes": 5},
            {"name": "c.mov", "path": "D:\\m\\c.mov", "size": 5,
             "status": "queued", "job_id": "", "error": "", "done_bytes": 0},
        ],
    }
    S._bulk_imports["r1"] = st
    asyncio.run(S._run_bulk_import("r1"))

    assert waited[0] == "jb"                    # adopted, not re-imported
    assert events == ["dl:c.mov", "an:c.mov"]   # only c was downloaded
    assert [i["status"] for i in st["items"]] == ["complete", "complete", "complete"]
    assert (st["ok"], st["done"], st["status"]) == (3, 3, "complete")
    assert [j.filename for j in saved] == ["c.mov"]


def test_bulk_progress_endpoint_reports_state_without_cancel_flag(_host, monkeypatch, tmp_path):
    _wire_runner(monkeypatch, tmp_path)
    bulk_id = _seed_bulk(monkeypatch, ["a.mp4"])
    out = asyncio.run(S.companion_folder_import_progress(bulk_id))
    assert out["total"] == 1 and out["status"] == "running"
    assert "cancel" not in out
    assert out["items"][0]["name"] == "a.mp4"

    with pytest.raises(HTTPException) as e:
        asyncio.run(S.companion_folder_import_progress("nope"))
    assert e.value.status_code == 404

    asyncio.run(S._run_bulk_import(bulk_id))
    out = asyncio.run(S.companion_folder_import_progress(bulk_id))
    assert out["status"] == "complete" and out["ok"] == 1
