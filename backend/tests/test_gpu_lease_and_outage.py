"""Run-49 brown-out fixes: GPU class lease, degraded-residency sizing,
concurrent-lane progress, and the Companion-import language fields.

Root cause chain on the real run: clip vision (llava:7b) loaded next to the
translation 12B on a 9.5 GB budget → the 12B CPU-spilled to 1.8/8.3 GB →
every text call blew the warm 90s ceiling → the "recovery" unloaded the
TIMED-OUT model (leaving the squatter) and its /api/show downgrade probes
401'd → 21 minutes of churn and an untranslated shipped transcript.
"""

import asyncio

import pytest

from backend.services.ai_orchestrator import _GpuClassLease, _est_model_gb


# ── GPU class lease ──────────────────────────────────────────────────


def test_lease_same_class_runs_concurrently():
    async def run():
        lease = _GpuClassLease(grace_s=5.0)
        await lease.acquire("text")
        # A second same-class holder must NOT wait.
        await asyncio.wait_for(lease.acquire("text"), timeout=0.5)
        await lease.release()
        await lease.release()
    asyncio.run(run())


def test_lease_opposing_class_waits_for_drain_and_grace():
    async def run():
        lease = _GpuClassLease(grace_s=0.3)
        await lease.acquire("text")

        acquired = asyncio.Event()

        async def want_vision():
            await lease.acquire("vision")
            acquired.set()

        task = asyncio.create_task(want_vision())
        await asyncio.sleep(0.1)
        assert not acquired.is_set()          # blocked while text holds
        await lease.release()
        await asyncio.sleep(0.1)
        assert not acquired.is_set()          # still inside the grace window
        await asyncio.wait_for(acquired.wait(), timeout=2.0)  # grace expires
        await lease.release()
        task.result()
    asyncio.run(run())


def test_lease_free_lease_admits_any_class():
    async def run():
        lease = _GpuClassLease(grace_s=10.0)
        # Never held: no class stamp, no grace to wait out.
        await asyncio.wait_for(lease.acquire("vision"), timeout=0.5)
        await lease.release()
    asyncio.run(run())


# ── Model size estimation (co-fit decision) ──────────────────────────


def test_est_model_gb_matches_observed_sizes():
    # Observed on the rig: gemma3:12b ≈ 8.3 GB resident, llava:7b ≈ 5-6 GB.
    g = _est_model_gb("gemma3:12b-it-q4_K_M")
    l = _est_model_gb("llava:7b")
    assert 7.5 <= g <= 9.5
    assert 4.5 <= l <= 6.5
    # The pair must NOT co-fit a 9.5 GB Companion budget…
    assert g + l > 9.5 * 0.95
    # …but comfortably co-fits a 24 GB card (no serialization there).
    assert g + l < 24.0 * 0.95


def test_est_model_gb_unknown_is_zero():
    assert _est_model_gb("mystery-model") == 0.0
    assert _est_model_gb("") == 0.0


# ── Remote VRAM sizing + the sticky num_gpu rung ─────────────────────


def _bare_provider(host="http://192.168.8.12:11500/ollama"):
    from backend.services.providers.ollama_provider import OllamaProvider
    p = OllamaProvider.__new__(OllamaProvider)
    p._host = host
    p._force_cpu = False
    p._force_cpu_model = None
    p._available_vram_mb = 0
    p._gpu_layers_good = {}
    return p


def test_large_model_on_remote_host_forces_gpu(monkeypatch):
    # Run 51: milliseconds into warmup (advertisement not landed,
    # _available_vram_mb still 0) the provider decided llava:7b "exceeds 4GB
    # VRAM" — of the LOCAL card — and pinned the Companion's vision to CPU
    # for the whole job. Remote hosts must be judged roomy.
    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "is_local_gpu_host", lambda url: False)
    p = _bare_provider()
    assert p._get_num_gpu("llava:7b") == 99


def test_large_model_on_small_local_host_stays_cpu(monkeypatch):
    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "is_local_gpu_host", lambda url: True)
    p = _bare_provider(host="http://127.0.0.1:11434")
    p._available_vram_mb = 4096
    assert p._get_num_gpu("llava:7b") == 0


def test_reset_gpu_layers_memo_unpins_the_rung():
    # The memo is a one-way ratchet: a first load under (since-cleared)
    # whisper pressure pinned gemma at ~22% residency for an entire run,
    # because every defrag reload reused the low remembered rung. The
    # defrag path resets it so the clean pool is actually used.
    p = _bare_provider()
    p._gpu_layers_good["gemma3:12b-it-q4_K_M"] = 11
    p.reset_gpu_layers_memo("gemma3:12b-it-q4_K_M")
    assert p._gpu_layers_good == {}
    # Other models' memos survive; matching is tag-exact (same semantics as
    # every other name comparison in the provider).
    p._gpu_layers_good = {"gemma3:12b-it-q4_K_M": 11, "qwen3:4b": 20}
    p.reset_gpu_layers_memo("gemma3:12b-it-q4_K_M")
    assert p._gpu_layers_good == {"qwen3:4b": 20}


# ── Concurrent-lane progress semantics ───────────────────────────────


def test_lane_update_behind_scalar_is_broadcast_only(monkeypatch):
    from backend.services import pipeline as pl
    from backend.models import JobStatus

    writes, events = [], []

    async def _fake_update_job_status(job_id, **kw):
        writes.append(kw)

    async def _fake_broadcast(job_id, msg):
        events.append(msg)

    monkeypatch.setattr(pl.database, "update_job_status", _fake_update_job_status)
    monkeypatch.setattr(pl, "broadcast_ws", _fake_broadcast)
    job = "lane-test-job"
    pl._last_scalar_progress[job] = 90   # clips lane owns the bar

    asyncio.run(pl._update_progress(
        job, JobStatus.TRANSLATING, 65, "Translating subtitles… (10/50)",
        lane="translation"))

    # Behind the bar: no scalar write, but the event still went out with the
    # lane tag and WITHOUT a progress field (so the UI bar can't regress).
    assert writes == []
    assert len(events) == 1
    assert events[0]["lane"] == "translation"
    assert "progress" not in events[0]
    assert events[0]["stage_id"] == "translation"
    pl._last_scalar_progress.pop(job, None)


def test_lane_update_ahead_of_scalar_still_writes(monkeypatch):
    from backend.services import pipeline as pl
    from backend.models import JobStatus

    writes, events = [], []

    async def _fake_update_job_status(job_id, **kw):
        writes.append(kw)

    async def _fake_broadcast(job_id, msg):
        events.append(msg)

    monkeypatch.setattr(pl.database, "update_job_status", _fake_update_job_status)
    monkeypatch.setattr(pl, "broadcast_ws", _fake_broadcast)
    job = "lane-test-job-2"
    pl._last_scalar_progress[job] = 40   # translation IS the critical path

    asyncio.run(pl._update_progress(
        job, JobStatus.TRANSLATING, 65, "Translating subtitles… (10/50)",
        lane="translation"))

    assert len(writes) == 1 and writes[0]["progress"] == 65
    assert len(events) == 1 and events[0]["progress"] == 65
    assert events[0]["lane"] == "translation"
    assert pl._last_scalar_progress[job] == 65
    pl._last_scalar_progress.pop(job, None)


def test_translation_stage_band_matches_live_emitters():
    # The stage table used to declare translation at 76-80% while the live
    # code emitted 63-69% — every live translation event was mislabeled
    # "Video Summary" in the processing log.
    from backend.services.pipeline import _resolve_pipeline_stage
    stage = _resolve_pipeline_stage("translating", 65)
    assert stage.get("id") == "translation"
    stage = _resolve_pipeline_stage("generating_summary", 72)
    assert stage.get("id") == "summary"


# ── Companion import language fields ─────────────────────────────────


def test_companion_import_request_accepts_languages():
    from backend.routers.settings import CompanionImportRequest
    req = CompanionImportRequest(host_id="h", path="/x/v.mp4")
    assert req.source_language == "" and req.target_language == ""
    req = CompanionImportRequest(
        host_id="h", path="/x/v.mp4", source_language="ja", target_language="en")
    assert req.source_language == "ja" and req.target_language == "en"
