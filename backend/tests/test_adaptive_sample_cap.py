"""GPU-adaptive reframer sample cap.

The perceiver's cost scales with the sample count. The vision offload moves only
YOLO; YuNet + u2net saliency + Farneback motion stay LOCAL on every sampled
frame, so the LOCAL card is the per-frame floor and the governor. A capable card
is scaled up from the weak baseline; a weak card is held at baseline (inflating
it only adds local work the offload can't remove).
"""
import backend.services.pipeline as P
from backend.config import settings


def _cap(monkeypatch, *, total_mb, name, offload, base=1200, ceiling=4200):
    monkeypatch.setattr(P, "_query_vram", lambda: (total_mb // 2, total_mb), raising=False)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(_i):
            return name

        @staticmethod
        def mem_get_info():
            return (total_mb * 1024 * 1024 // 2, total_mb * 1024 * 1024)

    import types
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda())
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setattr(P.settings, "REMOTE_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(P.settings, "REFRAMER_SAMPLE_CAP_CEILING", ceiling, raising=False)
    # Offload "possible" is gated on a paired Companion (remote whisper base).
    import backend.services.reframer_audio as RA
    monkeypatch.setattr(RA, "_remote_whisper_base",
                        lambda: ("http://gpu:11500" if offload else ""), raising=False)
    return P._adaptive_reframer_sample_cap(base_cap=base, job_id="")


def test_weak_local_stays_at_baseline_no_offload(monkeypatch):
    # GTX 1650, 4 GB, no offload → exactly the baseline (never inflated).
    assert _cap(monkeypatch, total_mb=4096, name="NVIDIA GeForce GTX 1650", offload=False) == 1200


def test_weak_local_not_boosted_even_with_offload(monkeypatch):
    # The offload removes only YOLO; the weak card's local floor is unchanged, so
    # its sample count must NOT rise (more samples there = slower, not better).
    assert _cap(monkeypatch, total_mb=4096, name="NVIDIA GeForce GTX 1650", offload=True) == 1200


def test_weak_name_override_beats_misreported_vram(monkeypatch):
    # A mobile chip that over-reports VRAM is still pinned to the weak tier by name.
    assert _cap(monkeypatch, total_mb=8192, name="NVIDIA GeForce GTX 1650", offload=False) == 1200


def test_strong_local_scales_up(monkeypatch):
    # 8 GB (3070-class), no offload → 1200 * 2.2 = 2640.
    assert _cap(monkeypatch, total_mb=8192, name="NVIDIA GeForce RTX 3070", offload=False) == 2640


def test_strong_local_gets_modest_offload_boost(monkeypatch):
    # 8 GB with offload → 2640 * 1.25 = 3300.
    assert _cap(monkeypatch, total_mb=8192, name="NVIDIA GeForce RTX 3070", offload=True) == 3300


def test_very_strong_local_clamped_to_ceiling(monkeypatch):
    # 12 GB (4070) + offload → 1200*3.0*1.25 = 4500, clamped to the 4200 ceiling.
    assert _cap(monkeypatch, total_mb=12288, name="NVIDIA GeForce RTX 4070", offload=True) == 4200


def test_medium_local_tier(monkeypatch):
    # ~6 GB (2060) → 1200 * 1.5 = 1800.
    assert _cap(monkeypatch, total_mb=6144, name="NVIDIA GeForce RTX 2060", offload=False) == 1800


def test_degrades_to_baseline_without_torch(monkeypatch):
    # No torch / no VRAM reading → baseline, never crashes.
    monkeypatch.setattr(P, "_query_vram", lambda: None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert P._adaptive_reframer_sample_cap(base_cap=1200, job_id="") == 1200
