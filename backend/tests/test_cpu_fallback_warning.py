"""The pipeline must surface a warning when a heavy stage runs on CPU despite
GPU being enabled (the "GPU unavailable → silently 10-30x slower / looks stuck"
case). cpu_fallback_stages() is the detector behind that warning."""

from backend.services.pipeline_helpers import cpu_fallback_stages


def test_all_gpu_no_cpu_stages():
    summary = {
        "frame_extract": {"device": "cuda:0"},
        "yolo_world": {"device": "cuda:0"},
        "whisper": {"device": "cuda:0", "detail": "CTranslate2 float16"},
    }
    assert cpu_fallback_stages(summary) == []


def test_whisper_on_cpu_flagged():
    summary = {
        "yolo_world": {"device": "cuda:0"},
        "whisper": {"device": "cpu", "detail": "CTranslate2 int8"},
    }
    assert cpu_fallback_stages(summary) == ["Transcription (Whisper)"]


def test_multiple_cpu_stages_flagged_with_labels():
    # Mirrors the log: torch CPU-only → YOLO on CPU, CTranslate2 CUDA failed
    # → Whisper on CPU.
    summary = {
        "frame_extract": {"device": "cuda:0"},
        "yolo_world": {"device": "cpu"},
        "whisper": {"device": "cpu"},
    }
    labels = cpu_fallback_stages(summary)
    assert "Subject detection (YOLO-World)" in labels
    assert "Transcription (Whisper)" in labels
    assert "Frame extraction" not in labels


def test_empty_or_malformed_summary_safe():
    assert cpu_fallback_stages({}) == []
    assert cpu_fallback_stages(None) == []
    assert cpu_fallback_stages({"whisper": "not-a-dict"}) == []


def test_unknown_stage_key_kept_verbatim():
    assert cpu_fallback_stages({"mystery": {"device": "cpu"}}) == ["mystery"]
