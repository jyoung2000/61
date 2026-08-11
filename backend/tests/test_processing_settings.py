"""Concurrent Analyses setting (Settings > Advanced): the /processing/settings
GET/POST endpoints and the live-resizable AnalysisGate behind them — raising
the limit admits queued jobs immediately, lowering it never interrupts running
analyses, and the value persists to user_settings.json + .env."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.settings as S
from backend.config import settings as cfg
from backend.services import pipeline as P


def _client():
    app = FastAPI()
    app.include_router(S.router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Keep the test from writing the real user_settings.json / .env, and
    restore the concurrency limit afterwards."""
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", str(tmp_path / "user_settings.json"))
    monkeypatch.setattr(S, "_find_env_file", lambda: None)
    old_limit = cfg.CONCURRENT_ANALYSES
    old_gate = P._analysis_semaphore
    P._analysis_semaphore = None
    yield
    cfg.CONCURRENT_ANALYSES = old_limit
    P._analysis_semaphore = old_gate


def test_get_processing_settings_reports_limit_and_active():
    c = _client()
    r = c.get("/api/processing/settings")
    assert r.status_code == 200
    data = r.json()
    assert data["concurrent_analyses"] == cfg.CONCURRENT_ANALYSES
    assert data["active_analyses"] == 0


def test_post_updates_limit_and_clamps():
    c = _client()
    r = c.post("/api/processing/settings", json={"concurrent_analyses": 2})
    assert r.status_code == 200
    assert r.json()["concurrent_analyses"] == 2
    assert cfg.CONCURRENT_ANALYSES == 2

    # Out-of-range values are clamped, never rejected or applied raw.
    assert (
        c.post("/api/processing/settings", json={"concurrent_analyses": 99}).json()[
            "concurrent_analyses"
        ]
        == 8
    )
    assert (
        c.post("/api/processing/settings", json={"concurrent_analyses": 0}).json()[
            "concurrent_analyses"
        ]
        == 1
    )

    # Empty body changes nothing.
    r = c.post("/api/processing/settings", json={})
    assert r.status_code == 200
    assert r.json()["concurrent_analyses"] == 1


def test_post_resizes_live_gate():
    async def main():
        gate = P.get_semaphore()
        assert gate.limit == cfg.CONCURRENT_ANALYSES
        P.apply_concurrency(3)
        assert gate.limit == 3
        assert cfg.CONCURRENT_ANALYSES == 3

    asyncio.run(main())


def test_gate_raise_admits_waiter_immediately():
    """Limit 1: second job blocks. Raising to 2 admits it WITHOUT any
    release — the live-resize the Settings page promises."""

    async def main():
        gate = P.AnalysisGate(1)
        order = []
        hold_a, hold_b = asyncio.Event(), asyncio.Event()

        async def worker(name, hold):
            async with gate:
                order.append(f"{name}-in")
                await hold.wait()
            order.append(f"{name}-out")

        ta = asyncio.create_task(worker("a", hold_a))
        await asyncio.sleep(0.01)
        tb = asyncio.create_task(worker("b", hold_b))
        await asyncio.sleep(0.01)
        assert order == ["a-in"], "limit 1 keeps the second job queued"
        assert gate.active == 1

        gate.set_limit(2)
        await asyncio.sleep(0.01)
        assert "b-in" in order, "raising the limit admits the queued job live"
        assert gate.active == 2

        hold_a.set()
        hold_b.set()
        await asyncio.gather(ta, tb)
        assert gate.active == 0

    asyncio.run(main())


def test_gate_lower_never_interrupts_running_jobs():
    """Two jobs running at limit 2; lowering to 1 lets both finish and only
    admits ONE of the queued jobs at a time afterwards."""

    async def main():
        gate = P.AnalysisGate(2)
        events = {n: asyncio.Event() for n in ("a", "b", "c")}
        running = set()

        async def worker(name):
            async with gate:
                running.add(name)
                await events[name].wait()
                running.discard(name)

        ta = asyncio.create_task(worker("a"))
        tb = asyncio.create_task(worker("b"))
        await asyncio.sleep(0.01)
        assert running == {"a", "b"}

        gate.set_limit(1)
        await asyncio.sleep(0.01)
        assert running == {"a", "b"}, "lowering must not interrupt running jobs"

        tc = asyncio.create_task(worker("c"))
        await asyncio.sleep(0.01)
        assert "c" not in running, "over the new cap — c waits"

        events["a"].set()
        await asyncio.sleep(0.01)
        assert "c" not in running, "still 1 active (b) at limit 1 — c keeps waiting"

        events["b"].set()
        await asyncio.sleep(0.01)
        assert "c" in running, "a free slot under the new cap admits c"

        events["c"].set()
        await asyncio.gather(ta, tb, tc)
        assert gate.active == 0

    asyncio.run(main())


def test_concurrent_analyses_is_persisted():
    assert "CONCURRENT_ANALYSES" in S._PERSISTABLE_KEYS, (
        "the setting must ride user_settings.json so it survives container "
        "rebuilds like every other Settings-page knob"
    )


def test_bulk_min_free_gb_roundtrip_and_clamp():
    c = _client()
    r = c.post("/api/processing/settings", json={"bulk_min_free_gb": 10})
    assert r.status_code == 200
    assert r.json()["bulk_min_free_gb"] == 10.0
    assert c.get("/api/processing/settings").json()["bulk_min_free_gb"] == 10.0
    # Clamped to a sane band, never applied raw.
    assert c.post("/api/processing/settings",
                  json={"bulk_min_free_gb": 0}).json()["bulk_min_free_gb"] == 0.5
    assert c.post("/api/processing/settings",
                  json={"bulk_min_free_gb": 9999}).json()["bulk_min_free_gb"] == 500.0


def test_bulk_disk_floor_feeds_the_import_gate(monkeypatch):
    monkeypatch.setattr(cfg, "BULK_IMPORT_MIN_FREE_GB", 4.0)
    assert S._bulk_disk_floor() == int(4.0 * 1024 ** 3)
    assert "BULK_IMPORT_MIN_FREE_GB" in S._PERSISTABLE_KEYS
