"""Fixes from the 43m03s run + editor/companion issues (2026-07-13).

1. Film strip: the single-clip editor strip seeked a hidden <video> 15×
   into the 2-hour file (multi-second Range storms; 15s load timeout;
   silent `catch {}`) — never using the sprite system built for it. Now
   sprite-backed (source pins below; behavior is browser-side).
2. "MECA:" labels survived 4 runs because the ja source contained a colon
   and the strip gate treated ANY source colon as protection — a purely
   CJK source can never legitimately yield a Latin all-caps label.
3. Stretched vocalizations ("Uuuuuuuuuu.") shipped as glyph walls.
4. The untranslated-cue recovery ran strictly serially (a visible
   multi-minute XLATE tail); the AI post-edit ran with ZERO Processing-Log
   output for ~11 minutes.
5. Companion self-update dead-ended on manually-added pairings: the app
   never knew ClipAI's URL. It now learns it from inbound traffic
   (peer IP + the X-ClipAI-Port header every container request carries).
"""

import inspect
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))


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


_stub_provider_sdks()

from backend.services.translator import strip_invented_speaker_labels  # noqa: E402
from backend.services.transcript_dedup import collapse_char_runs  # noqa: E402
from backend.services.request_context import clipai_headers  # noqa: E402


# ── 2. Label strip: non-Latin source colon must not protect ────────────────

def test_cjk_source_colon_does_not_protect_invented_label():
    """The observed leak: whisper emitted a ja line containing '：', the
    translator prefixed 'MECA:', and the old any-colon gate kept it."""
    got = strip_invented_speaker_labels(
        "MECA: I ate all of it. It was delicious.", "メカ：全部食べた。おいしかった。")
    assert got == "I ate all of it. It was delicious."


def test_latin_source_colon_still_protects():
    txt = "Warning: do not eat this."
    assert strip_invented_speaker_labels(txt, "Warning: これを食べないで") == txt


def test_plain_cjk_source_strips_as_before():
    assert strip_invented_speaker_labels(
        "MIKA: Thank you for the meal.", "ごちそうさま。"
    ) == "Thank you for the meal."


# ── 3. Vocalization collapse ────────────────────────────────────────────────

def test_char_runs_collapse_to_three():
    segs = [
        {"text": "Uuuuuuuuuuuuuu."},
        {"text": "AAAAAAAAAA AIBON"},
        {"text": "Eeeeeeeee!"},
        {"text": "Keep going!"},          # no 5-run — untouched
        {"text": "[♪ music ♪]"},          # marker — untouched
    ]
    out, changed = collapse_char_runs(segs)
    assert changed == 3
    assert [s["text"] for s in out] == [
        "Uuu.", "AAA AIBON", "Eee!", "Keep going!", "[♪ music ♪]",
    ]


def test_real_words_are_safe():
    segs = [{"text": "The bookkeeper called; success followed immediately."}]
    _, changed = collapse_char_runs(segs)
    assert changed == 0


# ── 4. XLATE tail: recovery concurrency + post-edit progress ───────────────

def test_recovery_loop_is_concurrent():
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    i = src.find("async def _recover_one")
    assert i > 0
    window = src[i:i + 3000]
    assert "TRANSLATION_LLM_CLEANUP_CONCURRENCY" in window
    assert "asyncio.gather" in window


def test_post_edit_reports_progress_to_the_ui():
    from backend.services import pipeline
    src = inspect.getsource(pipeline)
    i = src.find("async def _polish_progress")
    assert i > 0
    window = src[i:i + 2500]
    assert "Polishing translated subtitles" in window
    assert "progress_callback=_polish_progress" in window


# ── 5. Companion learns ClipAI's URL from inbound traffic ──────────────────

def test_every_outbound_request_carries_the_port_header():
    headers = clipai_headers()
    assert headers.get("X-ClipAI-Port") == "1353"


def test_companion_learns_url_and_falls_back(tmp_path):
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proxy = open(os.path.join(repo, "companion", "src-tauri", "src", "proxy.rs"),
                 encoding="utf-8").read()
    lib = open(os.path.join(repo, "companion", "src-tauri", "src", "lib.rs"),
               encoding="utf-8").read()
    assert "x-clipai-port" in proxy
    assert "into_make_service_with_connect_info" in proxy
    assert "note_clipai_origin" in proxy
    assert "seen_clipai_url" in lib        # paired_base fallback


# ── 1. Film strip uses the sprite path ──────────────────────────────────────

def test_editor_filmstrip_is_sprite_backed():
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ve = open(os.path.join(repo, "frontend", "src", "components", "VideoEditor.jsx"),
              encoding="utf-8").read()
    # The old hidden-<video> seek loop is gone from the strip generator...
    assert "tv.onseeked" not in ve
    # ...and the sprite-backed util drives it, with the upgrade listener.
    assert "ensureThumbnail(src, time, tW, tH" in ve
    assert "FILMSTRIP_UPDATED_EVENT" in ve
