"""Fixes from the 2026-07-12 run analysis (41m45s, build 679027f).

Four defects verified against that run's logs + exported transcript:

1. AI post-edit crawled at ~80 s/batch with 90 s timeouts because num_ctx
   flapped 2048→3072→4096 per request — every distinct value forces an
   Ollama runner reload. Fix: coarse ctx buckets + a sticky per-model
   high-water mark so the runner loads once.
2. One cue shipped ~1400 chars in a 5 s slot (the [6:32] blob): the LLM
   free-ran past the source line. Fix: length-ratio clamp vs the source.
3. Translated cues shipped invented "Mechanoid:" / "MECA:" speaker labels
   the source never had. Fix: deterministic label strip when the source
   line carries no colon.
4. Hybrid timing projected tier B = 0/582 because the whisper.cpp sidecar
   nests words PER SEGMENT while the container only read OpenAI's flat
   top-level list — every remote segment arrived word-less and the whole
   translated track became split-proof. Fix: parse the nested shape and
   request token_timestamps explicitly.
"""

import sys
import types


def _stub_provider_sdks():
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    if "groq" not in sys.modules:
        gr = types.ModuleType("groq")
        gr.AsyncGroq = object
        sys.modules["groq"] = gr


sys.modules.setdefault("cv2", types.ModuleType("cv2"))
_stub_provider_sdks()

from backend.services.providers.ollama_provider import OllamaProvider  # noqa: E402
from backend.services.translator import (  # noqa: E402
    clamp_runaway_translation,
    strip_invented_speaker_labels,
)
from backend.services.cloud_transcription import _map_verbose_json  # noqa: E402
from backend.config import settings  # noqa: E402

XLATE = "qwen3:4b-instruct-2507-q4_K_M"


def _provider():
    p = OllamaProvider()
    p._gpu_available = True
    p._force_cpu = False
    return p


# ── 1. Sticky num_ctx ───────────────────────────────────────────────────────

def test_nearby_prompt_sizes_share_one_ctx_bucket():
    """5.5k and 8.5k char prompts must resolve to the SAME num_ctx (4096) —
    the observed 3072↔4096 flap reloaded the runner on every batch."""
    p = _provider()
    a = p._get_effective_ctx(XLATE, prompt_chars=5560)   # required ~2877
    b = p._get_effective_ctx(XLATE, prompt_chars=8445)   # required ~3839
    assert a == b == 4096


def test_small_followup_prompts_keep_the_raised_ctx():
    """After one raise, a tiny prompt (or a no-prompt probe) must NOT bounce
    num_ctx back to 2048 — that bounce is another runner reload."""
    p = _provider()
    assert p._get_effective_ctx(XLATE, prompt_chars=7000) == 4096
    assert p._get_effective_ctx(XLATE, prompt_chars=800) == 4096
    assert p._get_effective_ctx(XLATE) == 4096


def test_sticky_ctx_is_per_model():
    p = _provider()
    assert p._get_effective_ctx(XLATE, prompt_chars=7000) == 4096
    # A different model starts at its own (flat) base.
    assert p._get_effective_ctx("qwen2.5:3b-instruct", prompt_chars=100) == 2048


def test_ctx_can_still_escalate_above_the_first_bucket():
    p = _provider()
    p._available_vram_mb = 12_000    # Companion-class card → 8192 ceiling
    assert p._get_effective_ctx(XLATE, prompt_chars=7000) == 4096
    assert p._get_effective_ctx(XLATE, prompt_chars=20_000) == 8192
    # ...and stays at the new high-water mark.
    assert p._get_effective_ctx(XLATE, prompt_chars=500) == 8192


def test_ctx_ceiling_still_capped_without_big_vram():
    p = _provider()
    p._available_vram_mb = 0         # unknown / small card
    assert p._get_effective_ctx(XLATE, prompt_chars=40_000) == 4096


# ── 2. Runaway-translation clamp ────────────────────────────────────────────

def test_normal_cjk_expansion_is_untouched():
    src = "ミカ、誕生日おめでとう！ケーキを買ってきたよ。"
    out = "Mika, happy birthday! I bought you a cake."
    assert clamp_runaway_translation(out, src) == out


def test_runaway_paragraph_is_cut_to_whole_sentences():
    src = "ミカが言ったんだよ。"                     # 10 chars → cap = 200 floor
    out = " ".join(f"Sentence number {i} keeps going." for i in range(60))
    clamped = clamp_runaway_translation(out, src)
    assert len(clamped) <= 200
    assert clamped.startswith("Sentence number 0 keeps going.")
    assert clamped.endswith(".")                     # whole-sentence boundary


def test_unbroken_runaway_is_hard_cut_at_a_word_boundary():
    src = "うん。"
    out = ("This single opening sentence is itself far longer than the cap "
           "would ever allow because the model rambled on without any "
           "sentence boundary at all for a very very very long time and then "
           "kept going and going and going and going and going and going")
    clamped = clamp_runaway_translation(out, src)
    assert clamped                                   # never emptied
    assert len(clamped) <= max(200, len(src) * 4)
    assert not clamped.endswith(" ")
    # cut lands on a word boundary, not mid-word
    assert out.startswith(clamped)
    assert out[len(clamped)] == " "


def test_ratio_zero_disables_the_clamp():
    old = getattr(settings, "TRANSLATION_MAX_EXPANSION_RATIO", 4.0)
    try:
        settings.TRANSLATION_MAX_EXPANSION_RATIO = 0
        out = "x" * 5000
        assert clamp_runaway_translation(out, "ソース") == out
    finally:
        settings.TRANSLATION_MAX_EXPANSION_RATIO = old


# ── 3. Invented speaker-label strip ─────────────────────────────────────────

def test_leading_invented_label_is_stripped():
    assert strip_invented_speaker_labels(
        "Mechanoid: Mika, did you eat any?", "ミカ、食べた？"
    ) == "Mika, did you eat any?"


def test_midtext_invented_label_is_stripped():
    got = strip_invented_speaker_labels(
        "I ate all of it. It was delicious. Mechanoid: Thank you, Mika.",
        "全部食べたよ。美味しかった。ありがとう、ミカ。")
    assert got == "I ate all of it. It was delicious. Thank you, Mika."


def test_allcaps_label_is_stripped():
    assert strip_invented_speaker_labels(
        "MECA: How many calories do you think this has?", "これ何カロリーだと思う？"
    ) == "How many calories do you think this has?"


def test_source_with_latin_colon_is_left_alone():
    # A source colon only protects when the source carries Latin script —
    # a purely CJK source can't legitimately yield a Latin "NAME:" prefix
    # (tightened after "MECA:" survived four runs behind a ja "：").
    txt = "Warning: do not eat this."
    assert strip_invented_speaker_labels(txt, "Warning: これを食べないで") == txt


def test_strip_never_empties_a_cue():
    assert strip_invented_speaker_labels("MECA: ", "メカ") == "MECA: "


def test_plain_dialogue_is_untouched():
    txt = "Thank you, Mika. That was delicious."
    assert strip_invented_speaker_labels(txt, "ありがとうミカ。美味しかった。") == txt


# ── 4. whisper.cpp per-segment words ───────────────────────────────────────

def test_map_verbose_json_reads_whispercpp_nested_words():
    payload = {
        "language": "ja",
        "segments": [{
            "start": 0.0, "end": 2.0, "text": "こんにちは世界",
            "avg_logprob": -0.2, "no_speech_prob": 0.01,
            "words": [
                {"word": "こんにちは", "start": 0.0, "end": 1.0, "probability": 0.95},
                {"word": "世界", "start": 1.1, "end": 1.9, "probability": 0.9},
            ],
        }],
    }
    segs = _map_verbose_json(payload)
    assert len(segs) == 1
    words = segs[0]["words"]
    assert [w["word"] for w in words] == ["こんにちは", "世界"]
    assert words[0]["start"] == 0.0 and words[1]["end"] == 1.9
    assert 0.0 < words[0]["confidence"] <= 1.0


def test_map_verbose_json_skips_nested_words_without_timing():
    """Old whisper.cpp builds (token_timestamps off) emit words with no
    start/end — those must not produce fake-timed words."""
    payload = {
        "segments": [{
            "start": 0.0, "end": 2.0, "text": "hello world",
            "words": [{"word": "hello", "probability": 0.9},
                      {"word": "world", "probability": 0.9}],
        }],
    }
    segs = _map_verbose_json(payload)
    assert segs[0]["words"] == []


def test_map_verbose_json_flat_word_list_still_works():
    payload = {
        "words": [{"word": "hello", "start": 0.0, "end": 0.4},
                  {"word": "world", "start": 0.5, "end": 0.9}],
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
    }
    segs = _map_verbose_json(payload)
    assert [w["word"] for w in segs[0]["words"]] == ["hello", "world"]


def test_remote_request_asks_for_token_timestamps():
    """Source pin: the remote upload form must carry token_timestamps=true so
    older whisper.cpp servers emit per-word start/end."""
    import inspect
    from backend.services import reframer_audio
    src = inspect.getsource(reframer_audio)
    assert '"token_timestamps": "true"' in src


def test_pipeline_reapplies_sanitizers_after_post_edit():
    """Source pin: the post-edit branch re-runs the deterministic sanitizers
    while source alignment holds."""
    import inspect
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    i = src.find("AI post-edit DONE on LLM-translated text")
    assert i > 0
    window = src[i:i + 2000]
    assert "strip_invented_speaker_labels" in window
    assert "clamp_runaway_translation" in window
