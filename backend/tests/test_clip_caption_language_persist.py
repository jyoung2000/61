"""Tests: the post-clip finishers must NOT clobber the target-language caption
refresh.

Root cause (clipai_logs_20260619_025328): during post-processing the DB reads
back 0 clips, so both the caption refresh and Auto-SEO fall back to the
in-process list. The refresh rebuilt English captions, but Auto-SEO then ran on
the STALE source-language fallback and re-persisted Japanese over them. The fix
threads the refreshed clips through SEO and back to the COMPLETE save in-process.
"""

import sys
import types
import asyncio


def _stub_sdks():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
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


_stub_sdks()

import backend.services.pipeline as P  # noqa: E402


def test_seo_runs_on_refreshed_clips_not_stale_source(monkeypatch):
    japanese = [{"id": 1, "suggested_caption": "[0:11] あの リリーナ様",
                 "hook_text": "あの リリーナ様", "title": "Clip 1"}]
    english = [{"id": 1, "suggested_caption": "[0:11] Oh, Miss Relena",
                "hook_text": "Oh, Miss Relena", "title": "The awkward invite"}]
    seen = {}

    async def fake_refresh(job_id, translated, fallback_clips=None):
        # Simulate the post-translation rebuild → English clips.
        return (1, english)

    async def fake_seo(job_id, transcript, orchestrator,
                       fallback_clips=None, output_language=""):
        seen["seo_input"] = fallback_clips
        seen["lang"] = output_language
        out = [dict(c, seo_title="EN title", seo_description="EN caption")
               for c in (fallback_clips or [])]
        return (1, 0, out)

    async def fake_ws(*a, **k):
        return None

    monkeypatch.setattr(P, "_refresh_clips_with_translation", fake_refresh)
    monkeypatch.setattr(P, "_auto_generate_clip_seo", fake_seo)
    monkeypatch.setattr(P, "broadcast_ws", fake_ws)

    pp = {
        "translated": True,
        "target_transcript": [{"text": "Oh, Miss Relena"}],
        "seo_transcript": [{"text": "Oh, Miss Relena"}],
        "output_lang": "en",
        "target_name": "English",
    }
    result = asyncio.run(P._run_post_clip_followups("job1", None, pp, japanese))

    # SEO must run on the ENGLISH refreshed clips — not the Japanese fallback.
    assert seen["seo_input"] == english
    assert seen["lang"] == "en"
    # And the returned (authoritative) list carries the English caption/hook.
    assert result[0]["suggested_caption"] == "[0:11] Oh, Miss Relena"
    assert result[0]["hook_text"] == "Oh, Miss Relena"
    assert result[0]["seo_title"] == "EN title"


def test_no_translation_keeps_source_clips(monkeypatch):
    src = [{"id": 1, "suggested_caption": "src cap", "hook_text": "src hook"}]
    seen = {}

    async def fake_refresh(*a, **k):
        raise AssertionError("refresh must not run when translated is falsy")

    async def fake_seo(job_id, transcript, orchestrator,
                       fallback_clips=None, output_language=""):
        seen["seo_input"] = fallback_clips
        return (1, 0, [dict(c, seo_title="t") for c in (fallback_clips or [])])

    async def fake_ws(*a, **k):
        return None

    monkeypatch.setattr(P, "_refresh_clips_with_translation", fake_refresh)
    monkeypatch.setattr(P, "_auto_generate_clip_seo", fake_seo)
    monkeypatch.setattr(P, "broadcast_ws", fake_ws)

    pp = {"translated": False, "seo_transcript": [], "output_lang": "ja"}
    result = asyncio.run(P._run_post_clip_followups("job1", None, pp, src))

    # No translation → SEO runs on the original (source) clips, unchanged.
    assert seen["seo_input"] == src
    assert result[0]["seo_title"] == "t"


def test_followups_return_clips_for_final_save_when_seo_skipped(monkeypatch):
    # SEO returns no clips (e.g. all already had SEO / skipped) → the finishers
    # still hand back the refreshed list so the COMPLETE save ships English.
    english = [{"id": 1, "suggested_caption": "EN", "hook_text": "EN"}]

    async def fake_refresh(job_id, translated, fallback_clips=None):
        return (1, english)

    async def fake_seo(*a, **k):
        return (0, 0, [])   # nothing generated, no clips returned

    async def fake_ws(*a, **k):
        return None

    monkeypatch.setattr(P, "_refresh_clips_with_translation", fake_refresh)
    monkeypatch.setattr(P, "_auto_generate_clip_seo", fake_seo)
    monkeypatch.setattr(P, "broadcast_ws", fake_ws)

    pp = {"translated": True, "target_transcript": [{"text": "EN"}],
          "seo_transcript": [{"text": "EN"}], "output_lang": "en"}
    japanese = [{"id": 1, "suggested_caption": "JA", "hook_text": "JA"}]
    result = asyncio.run(P._run_post_clip_followups("job1", None, pp, japanese))

    # Falls back to the refreshed English list, never the Japanese input.
    assert result == english
