"""Polish coverage + companion-URL fixes from the 49m27s run (2026-07-13).

The run confirmed the prior round landed (no MECA: prefixes, vocalizations
collapsed, post-edit progress visible) but exposed the dominant quality gap:
the AI post-edit polished only 211/930 cues (23%) before the 600 s budget
died at 35/62 batches — the other 77% ship as raw 4B output ("subtitles
don't make sense"). Two structural fixes:

1. Worst-first batch ordering — rank remaining batches by Whisper
   low-confidence cue count so a budget-limited pass fixes the GARBLED cues,
   not just the first third of the track.
2. Slim aligned context — a source-aligned batch already carries each line's
   ground-truth source ref, so the ±3 neighbor blocks (half the ~6000-char
   prompt) shrink to ±1, ~halving the prompt and roughly doubling throughput.

Plus the companion self-update URL fix: learn ClipAI's origin from an
explicit X-ClipAI-Origin header (odd networks) or peer-IP + X-ClipAI-Port,
now on the /v1/health route too.
"""

import os
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. Worst-first ordering + slim aligned context (source pins) ────────────

def test_polish_worst_first_ranks_by_confidence():
    import inspect
    from backend.services import transcript_polisher as P
    src = inspect.getsource(P.correct_transcript)
    assert "SUBTITLE_POLISH_WORST_FIRST" in src
    # Ranks by low-confidence count, and reorders EXECUTION (not context).
    assert "WHISPER_REDECODE_LOGPROB" in src
    assert "def _neediness" in src
    assert "key=lambda p: (-p[1], p[0])" in src  # descending by score


def test_polish_context_slims_when_aligned():
    import inspect
    from backend.services import transcript_polisher as P
    src = inspect.getsource(P.correct_transcript)
    assert "SUBTITLE_POLISH_CONTEXT_ALIGNED" in src
    assert "SUBTITLE_POLISH_CONTEXT" in src
    # The window is chosen by whether source_texts is present.
    i = src.find("_ctx_n = (")
    assert i > 0
    assert "source_texts is not None" in src[i:i + 300]


def test_polish_config_defaults():
    from backend.config import settings
    assert getattr(settings, "SUBTITLE_POLISH_CONTEXT_ALIGNED") == 1
    assert getattr(settings, "SUBTITLE_POLISH_CONTEXT") == 3
    assert getattr(settings, "SUBTITLE_POLISH_WORST_FIRST") is True
    # Budget nudged up modestly (slim prompt does the real work).
    assert getattr(settings, "SUBTITLE_POLISH_MAX_S") >= 700


def test_worst_first_reorders_to_neediest():
    """Functional check of the ranking key the loop uses: batches with more
    low-confidence cues sort first; the reorder never touches context."""
    thr = -0.8
    batches = {
        1: [{"text": "A clean line.", "avg_logprob": -0.1},
            {"text": "Another clean one.", "avg_logprob": -0.2}],       # clean
        2: [{"text": "garbled kimchi", "avg_logprob": -1.5},
            {"text": "word salad here", "avg_logprob": -0.9}],          # 2 low
        3: [{"text": "This reads fine.", "avg_logprob": -0.1},
            {"text": "co-ko-ri thing", "avg_logprob": -1.2}],           # 1 low
    }

    def neediness(idx):
        score = 0
        for v in batches[idx]:
            lp = v.get("avg_logprob")
            if lp is not None and float(lp) < thr:
                score += 1
            elif len((v.get("text") or "").strip()) <= 3:
                score += 1
        return score

    order = [idx for idx, _ in
             sorted([(i, neediness(i)) for i in batches],
                    key=lambda p: (-p[1], p[0]))]
    assert order == [2, 3, 1]


# ── 2. Companion self-update URL learning ───────────────────────────────────

def test_container_sends_origin_and_port_headers(monkeypatch):
    from backend.services.request_context import clipai_headers
    monkeypatch.setenv("CLIPAI_PUBLIC_URL", "http://192.168.8.5:1353")
    h = clipai_headers()
    assert h["X-ClipAI-Port"] == "1353"
    assert h["X-ClipAI-Origin"] == "http://192.168.8.5:1353"


def test_origin_header_omitted_when_unset(monkeypatch):
    from backend.services.request_context import clipai_headers
    monkeypatch.delenv("CLIPAI_PUBLIC_URL", raising=False)
    h = clipai_headers()
    assert "X-ClipAI-Origin" not in h
    assert h["X-ClipAI-Port"] == "1353"


def test_companion_learns_origin_on_health_and_ollama():
    proxy = open(os.path.join(REPO, "companion", "src-tauri", "src", "proxy.rs"),
                 encoding="utf-8").read()
    # X-ClipAI-Origin takes priority over peer-IP inference.
    assert "x-clipai-origin" in proxy
    # Learned on BOTH the ollama proxy and the health route.
    assert proxy.count("note_clipai_origin(&ctx.state") >= 2
    # Health route reads the peer address SOFTLY (Option), mirroring
    # ollama_proxy — a hard extractor would 500 the route before auth if the
    # router were ever served without connect-info wiring.
    i = proxy.find("async fn health(")
    window = proxy[i:i + 900]
    assert "Option<axum::extract::ConnectInfo<SocketAddr>>" in window
    assert "peer.map(|ci| ci.0)" in window


def test_update_error_names_the_url_and_fix():
    lib = open(os.path.join(REPO, "companion", "src-tauri", "src", "lib.rs"),
               encoding="utf-8").read()
    i = lib.find("async fn check_app_update")
    window = lib[i:i + 1500]
    assert "Couldn't reach ClipAI at {base}" in window
    assert "Pair now" in window


def test_companion_version_bumped_and_synced():
    import re
    cargo = re.search(r'^version = "([^"]+)"',
                      open(os.path.join(REPO, "companion", "src-tauri", "Cargo.toml"),
                           encoding="utf-8").read(), re.M).group(1)
    from backend.services.companion_version import EXPECTED_COMPANION_VERSION
    assert cargo == EXPECTED_COMPANION_VERSION == "0.2.4"


# ── 3. Self-update hardening (adversarial-review CONFIRMED HIGH findings) ────

def test_installer_integrity_verified_before_execute():
    """The self-update must SHA-256 the download against the server's published
    hash before running it — a size gate alone let a corrupted/tampered binary
    execute (review Finding 2)."""
    lib = open(os.path.join(REPO, "companion", "src-tauri", "src", "lib.rs"),
               encoding="utf-8").read()
    # Expected hash is fetched from the manifest…
    assert "async fn fetch_installer_sha256" in lib
    # …and the installer body is hashed + compared, refusing to run on mismatch.
    i = lib.find("async fn install_app_update")
    body = lib[i:i + 5000]
    assert "expected_sha256" in body
    assert "Sha256::new()" in body
    assert "eq_ignore_ascii_case" in body
    assert "integrity check FAILED" in body
    # The verify must sit BEFORE the process launch, not after.
    assert body.index("integrity check FAILED") < body.index("Command::new")


def test_sha2_and_hex_are_direct_deps():
    cargo = open(os.path.join(REPO, "companion", "src-tauri", "Cargo.toml"),
                 encoding="utf-8").read()
    assert "\nsha2 = " in cargo
    assert "\nhex = " in cargo


def test_origin_header_validated_receiver_side():
    """X-ClipAI-Origin becomes the download base for a download-and-run, so the
    receiver must do more than a scheme check — length cap + reject userinfo /
    paths / whitespace (review Finding 3)."""
    proxy = open(os.path.join(REPO, "companion", "src-tauri", "src", "proxy.rs"),
                 encoding="utf-8").read()
    assert "fn plausible_clipai_origin" in proxy
    i = proxy.find("fn plausible_clipai_origin")
    body = proxy[i:i + 900]
    assert "s.len() > 255" in body           # length cap
    assert "contains('@')" in body           # reject userinfo
    # note_clipai_origin routes the header through the validator.
    assert "filter(|s| plausible_clipai_origin(s))" in proxy
