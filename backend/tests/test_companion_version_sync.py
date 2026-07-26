"""Companion version discipline + the remote force-end endpoint.

The version-sync tests are the enforcement the companion_version docstring
always promised: every version source must agree, and the build-number
stamping (which makes each deployed build's version strictly higher than the
last) must actually rewrite every file that feeds the built app. Before the
stamping existed, one day's log showed FIVE self-update installs all
announcing "v0.2.9" — indistinguishable in logs and update prompts.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPANION = os.path.join(REPO, "companion")


def _repo_versions() -> dict:
    with open(os.path.join(COMPANION, "src-tauri", "tauri.conf.json")) as f:
        tauri = json.load(f)["version"]
    with open(os.path.join(COMPANION, "src-tauri", "Cargo.toml")) as f:
        cargo = re.search(r'^version = "([^"]+)"', f.read(), re.M).group(1)
    with open(os.path.join(COMPANION, "package.json")) as f:
        pkg = json.load(f)["version"]
    with open(os.path.join(COMPANION, "RELEASE")) as f:
        release = f.readline().strip()
    from backend.services.companion_version import EXPECTED_COMPANION_VERSION
    return {
        "tauri.conf.json": tauri,
        "Cargo.toml": cargo,
        "package.json": pkg,
        "RELEASE": release.replace("companion-v", ""),
        "EXPECTED_COMPANION_VERSION": EXPECTED_COMPANION_VERSION,
    }


def test_all_version_sources_agree():
    vs = _repo_versions()
    assert len(set(vs.values())) == 1, f"version drift: {vs}"


def test_cargo_lock_matches():
    with open(os.path.join(COMPANION, "src-tauri", "Cargo.lock")) as f:
        lock = f.read()
    base = _repo_versions()["Cargo.toml"]
    assert f'name = "clipai-companion"\nversion = "{base}"' in lock


def test_set_build_version_stamps_every_source(tmp_path):
    # Copy just the version-bearing files into the layout the script expects.
    (tmp_path / "src-tauri").mkdir()
    for rel in ("src-tauri/tauri.conf.json", "src-tauri/Cargo.toml",
                "src-tauri/Cargo.lock", "package.json"):
        shutil.copy(os.path.join(COMPANION, rel), tmp_path / rel)
    script = os.path.join(COMPANION, "scripts", "set_build_version.py")

    r = subprocess.run([sys.executable, script, "412"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    base = _repo_versions()["tauri.conf.json"]
    majmin = ".".join(base.split(".")[:2])
    newv = f"{majmin}.412"
    assert json.load(open(tmp_path / "src-tauri/tauri.conf.json"))["version"] == newv
    assert json.load(open(tmp_path / "package.json"))["version"] == newv
    cargo = (tmp_path / "src-tauri/Cargo.toml").read_text()
    assert f'version = "{newv}"' in cargo
    lock = (tmp_path / "src-tauri/Cargo.lock").read_text()
    assert f'name = "clipai-companion"\nversion = "{newv}"' in lock

    # The stamped version must be STRICTLY higher than the base, and a later
    # build (higher commit count) higher still — the monotonic guarantee.
    from backend.services.companion_version import parse_version
    assert parse_version(newv) > parse_version(base)
    assert parse_version(f"{majmin}.413") > parse_version(newv)


def test_set_build_version_zero_is_noop(tmp_path):
    (tmp_path / "src-tauri").mkdir()
    for rel in ("src-tauri/tauri.conf.json", "src-tauri/Cargo.toml",
                "src-tauri/Cargo.lock", "package.json"):
        shutil.copy(os.path.join(COMPANION, rel), tmp_path / rel)
    script = os.path.join(COMPANION, "scripts", "set_build_version.py")
    before = (tmp_path / "src-tauri/tauri.conf.json").read_text()
    r = subprocess.run([sys.executable, script, "0"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert (tmp_path / "src-tauri/tauri.conf.json").read_text() == before


def _dockerfiles_that_build_the_companion() -> dict:
    """Every Dockerfile in the repo root with a companion-builder stage.

    Checking only ``Dockerfile`` is how the stamping silently stopped working:
    compose builds ``Dockerfile.gpu``, which had neither the ARG nor the script
    call, so ``--build-arg COMPANION_BUILD_NUMBER=...`` was discarded and every
    from-source build reported the repo's base version. Enumerate instead of
    naming one file, so a new build path can't skip the stamp either."""
    out = {}
    for name in sorted(os.listdir(REPO)):
        if not (name == "Dockerfile" or name.startswith("Dockerfile.")):
            continue
        path = os.path.join(REPO, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            text = f.read()
        if "AS companion-builder" in text:
            out[name] = text
    return out


def test_every_companion_build_path_stamps_the_version():
    files = _dockerfiles_that_build_the_companion()
    # Sanity: the enumeration must actually find the build paths.
    assert "Dockerfile" in files and "Dockerfile.gpu" in files, sorted(files)
    for name, text in files.items():
        assert "ARG COMPANION_BUILD_NUMBER" in text, (
            f"{name} builds the Companion but never declares "
            "COMPANION_BUILD_NUMBER — docker discards the build-arg and the "
            "version never moves")
        assert "set_build_version.py" in text, (
            f"{name} builds the Companion but never runs set_build_version.py")


def test_build_number_is_wired_through_the_build():
    # update-all.sh must compute and pass the commit count.
    with open(os.path.join(REPO, "update-all.sh")) as f:
        upd = f.read()
    assert "COMPANION_BUILD_NUMBER" in upd
    assert "rev-list --count" in upd


def test_expected_version_not_outdated_by_stamped_builds():
    # A stamped build (0.X.<big commit count>) must never read as OLDER than
    # the container's expected base version.
    from backend.services.companion_version import is_outdated
    base = _repo_versions()["EXPECTED_COMPANION_VERSION"]
    majmin = ".".join(base.split(".")[:2])
    assert not is_outdated(f"{majmin}.9999", base)
    assert is_outdated("0.2.9", base)  # the pre-bump release IS outdated


# ── Remote force-end endpoint ────────────────────────────────────────


def test_force_end_jobs_cancels_active_and_signals_companions(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.routers.settings as S
    from backend import database
    from backend.models import JobResult, JobStatus

    # Three jobs on disk: running, queued, complete — only the first two end.
    monkeypatch.setattr(database, "_job_dir",
                        lambda job_id: str(tmp_path / job_id))
    now = "2026-07-20T00:00:00+00:00"
    for jid, status in (("j-run", JobStatus.DETECTING_CLIPS),
                        ("j-queued", JobStatus.QUEUED),
                        ("j-done", JobStatus.COMPLETE)):
        d = tmp_path / jid
        d.mkdir()
        job = JobResult(job_id=jid, filename="v.mp4", file_path="/x/v.mp4",
                        status=status, created_at=now, updated_at=now)
        (d / "job.json").write_text(job.model_dump_json())

    cancel_calls = []
    monkeypatch.setattr("backend.services.pipeline.request_cancel",
                        lambda job_id: cancel_calls.append(job_id))

    # One fake online companion; capture the outbound force-end POST.
    class _Host:
        id = "comp-1"
        name = "Gaming PC"
        url = "http://192.168.8.12:11500"

    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "get_hosts", lambda: [_Host()])
    monkeypatch.setattr(oreg, "is_local_gpu_host", lambda url: False)
    monkeypatch.setattr(oreg, "companion_base", lambda h: h.url)
    monkeypatch.setattr(oreg, "auth_headers", lambda h: {"X-Companion-Token": "t"})

    posted = []

    class _Resp:
        status_code = 200
        def json(self):
            return {"ended_job": "j-run", "whisper_stopped": True,
                    "ollama_unloaded": 2}

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None):
            posted.append((url, headers))
            return _Resp()

    monkeypatch.setattr(S.httpx, "AsyncClient", _Client)

    app = FastAPI()
    app.include_router(S.router)
    client = TestClient(app)
    r = client.post("/api/providers/companion/force-end-jobs")
    assert r.status_code == 200
    data = r.json()

    assert sorted(data["jobs_cancelled"]) == ["j-queued", "j-run"]
    assert sorted(cancel_calls) == ["j-queued", "j-run"]
    # Both cancelled jobs persisted as CANCELLED; the complete one untouched.
    for jid, want in (("j-run", "cancelled"), ("j-queued", "cancelled"),
                      ("j-done", "complete")):
        saved = json.loads((tmp_path / jid / "job.json").read_text())
        assert saved["status"] == want, jid
    # The companion was signalled on the new route with auth attached.
    assert len(posted) == 1
    assert posted[0][0].endswith("/v1/jobs/force-end")
    assert posted[0][1]["X-Companion-Token"] == "t"
    assert data["companions"][0]["ok"] is True
    assert data["companions"][0]["ollama_unloaded"] == 2


def test_force_end_jobs_reports_stale_companion(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import backend.routers.settings as S

    async def _no_jobs(**kw):
        return []
    from backend import database
    monkeypatch.setattr(database, "list_jobs", _no_jobs)

    class _Host:
        id = "comp-1"
        name = "Gaming PC"
        url = "http://192.168.8.12:11500"

    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "get_hosts", lambda: [_Host()])
    monkeypatch.setattr(oreg, "is_local_gpu_host", lambda url: False)
    monkeypatch.setattr(oreg, "companion_base", lambda h: h.url)
    monkeypatch.setattr(oreg, "auth_headers", lambda h: {})

    class _Resp:
        status_code = 404
        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None):
            return _Resp()

    monkeypatch.setattr(S.httpx, "AsyncClient", _Client)

    app = FastAPI()
    app.include_router(S.router)
    client = TestClient(app)
    data = client.post("/api/providers/companion/force-end-jobs").json()
    assert data["jobs_cancelled"] == []
    assert data["companions"][0]["ok"] is False
    # An old Companion without the route gets an actionable message.
    assert "update" in data["companions"][0]["error"].lower()
