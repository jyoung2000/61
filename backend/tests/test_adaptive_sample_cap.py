"""GPU-adaptive reframer sample cap.

The perceiver's cost scales with the sample count, and every per-frame pass
(YuNet faces, u2net saliency, motion, the strided YOLO subject detector) runs on
the LOCAL card. So the local card is the per-frame floor and the governor: a
capable card is scaled up from the weak baseline; a weak card is held at
baseline (inflating it only piles on local work).
"""
import backend.services.pipeline as P


def _cap(monkeypatch, *, total_mb, name, base=1200, ceiling=4200):
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
    monkeypatch.setattr(P.settings, "REFRAMER_SAMPLE_CAP_CEILING", ceiling, raising=False)
    return P._adaptive_reframer_sample_cap(base_cap=base, job_id="")


def test_weak_local_stays_at_baseline(monkeypatch):
    # GTX 1650, 4 GB → exactly the baseline (never inflated).
    assert _cap(monkeypatch, total_mb=4096, name="NVIDIA GeForce GTX 1650") == 1200


def test_weak_name_override_beats_misreported_vram(monkeypatch):
    # A mobile chip that over-reports VRAM is still pinned to the weak tier by name.
    assert _cap(monkeypatch, total_mb=8192, name="NVIDIA GeForce GTX 1650") == 1200


def test_medium_local_tier(monkeypatch):
    # ~6 GB (2060) → 1200 * 1.5 = 1800.
    assert _cap(monkeypatch, total_mb=6144, name="NVIDIA GeForce RTX 2060") == 1800


def test_strong_local_scales_up(monkeypatch):
    # 8 GB (3070-class) → 1200 * 2.2 = 2640.
    assert _cap(monkeypatch, total_mb=8192, name="NVIDIA GeForce RTX 3070") == 2640


def test_very_strong_local_tier(monkeypatch):
    # 12 GB (4070) → 1200 * 3.0 = 3600 (under the 4200 ceiling).
    assert _cap(monkeypatch, total_mb=12288, name="NVIDIA GeForce RTX 4070") == 3600


def test_ceiling_clamps_the_biggest_cards(monkeypatch):
    # A high base + strong card is clamped to the ceiling.
    assert _cap(monkeypatch, total_mb=24576, name="NVIDIA GeForce RTX 4090",
                base=1600, ceiling=4200) == 4200


def test_degrades_to_baseline_without_torch(monkeypatch):
    # No torch / no VRAM reading → baseline, never crashes.
    monkeypatch.setattr(P, "_query_vram", lambda: None, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)
    assert P._adaptive_reframer_sample_cap(base_cap=1200, job_id="") == 1200
