"""Translation context/coherence upgrades:
- episode context brief (builder guards + cache)
- full-track coherence audit + targeted re-translate (item 4)
- [UNRELIABLE ASR] marks in the batch prompt (item 5)
- scene-break/translated-tail context block (item 6)
"""
import asyncio
import json
import re
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
    for name, attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                       ("anthropic", "AsyncAnthropic")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, attr, object)
            sys.modules[name] = mod


_stub_provider_sdks()

from backend.services import translator as T  # noqa: E402


class _FakeOrch:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def text_completion(self, prompt, **kw):
        self.calls += 1
        return self.reply


def _segs(n=20):
    return [{"text": f"これは第{i}話のセリフです。登場人物が会話します。"}
            for i in range(n)]


def test_brief_built_cached_and_readable_by_job():
    T._EPISODE_BRIEFS.clear()
    orch = _FakeOrch(
        "Relena returns to Earth with her father, the Vice Foreign Minister. "
        "A mysterious boy named Heero crashes nearby and enrolls at her "
        "school. Officers Zechs and Treize discuss the Gundam threat in a "
        "formal military register. The tone is serious wartime drama.")
    brief = asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", orch, "job-1", None))
    assert brief and "Relena" in brief
    assert T.episode_brief_for_job("job-1") == brief
    assert T.episode_brief_for_job("other-job") == ""


def test_brief_rejects_fragments_and_leaked_structure():
    T._EPISODE_BRIEFS.clear()
    # A JSON-shaped or fragment reply must be rejected, never injected into
    # every batch prompt as noise.
    assert asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", _FakeOrch('{"summary": "..."}'), "j2", None)) == ""
    assert asyncio.run(T._build_episode_brief(
        _segs(), "Japanese", _FakeOrch("Okay."), "j3", None)) == ""
    assert T.episode_brief_for_job("j2") == ""


def test_brief_skips_short_transcripts_without_an_llm_call():
    T._EPISODE_BRIEFS.clear()
    orch = _FakeOrch("irrelevant")
    out = asyncio.run(T._build_episode_brief(
        _segs(5), "Japanese", orch, "j4", None))
    assert out == "" and orch.calls == 0


# ── item 4: coherence audit ─────────────────────────────────────────────────

def test_parse_int_array_tolerates_json_loose_and_junk():
    assert T._parse_int_array("[3, 17]", 100) == [3, 17]
    assert T._parse_int_array("```json\n[5]\n```", 100) == [5]
    assert T._parse_int_array("Flagged lines: 3 and 17.", 100) == [3, 17]
    assert T._parse_int_array("[]", 100) == []
    assert T._parse_int_array("no issues found", 100) == []
    # out-of-range numbers are dropped, duplicates collapse
    assert T._parse_int_array("[0, 3, 3, 999]", 10) == [3]


class _AuditOrch:
    """Audit call → flagged numbers; fix call → the canned repair."""

    def __init__(self, flags, fix_reply):
        self.flags = flags
        self.fix_reply = fix_reply
        self.fix_prompts = []

    async def text_completion(self, prompt, **kw):
        if "JSON array of integers" in prompt:
            return json.dumps(self.flags)
        self.fix_prompts.append(prompt)
        return self.fix_reply


def _track(texts):
    from backend.models import TranscriptSegment
    return [TranscriptSegment(text=t, start=float(i * 2),
                              end=float(i * 2 + 1.5), speaker="Speaker 1")
            for i, t in enumerate(texts)]


_EN_TRACK = [
    "[♪ Opening theme ♪]",          # marker — excluded from numbering
    "The colony declared war.",
    "We must protect the shuttle.",
    "He never fired the missile.",  # ← the contradictory line (audit #3)
    "The missile hit the base.",
    "Casualties are still unknown.",
    "Send the report to command.",
    "Understood, sir.",
    "We move at dawn.",
]
_SRC_TRACK = [{"text": f"日本語のセリフ{i}です。"} for i in range(len(_EN_TRACK))]


def test_coherence_audit_repairs_flagged_cue_through_marker_offset():
    out = _track(_EN_TRACK)
    # Numbering skips the marker, so audit number 3 = list index 3.
    orch = _AuditOrch([3], "He fired the missile after all.")
    n = asyncio.run(T.coherence_audit_and_fix(
        _SRC_TRACK, out, orch, "Japanese", "English", job_id="j-coh"))
    assert n == 1
    assert out[3].text == "He fired the missile after all."
    # timing + speaker survive the swap; the marker is untouched
    assert out[3].start == 6.0 and out[3].end == 7.5
    assert out[0].text == "[♪ Opening theme ♪]"
    # the fix prompt carried the scene and the source line
    assert ">> He never fired the missile." in orch.fix_prompts[0]
    assert "日本語のセリフ3です。" in orch.fix_prompts[0]


def test_coherence_audit_no_flags_changes_nothing():
    out = _track(_EN_TRACK)
    orch = _AuditOrch([], "unused")
    n = asyncio.run(T.coherence_audit_and_fix(
        _SRC_TRACK, out, orch, "Japanese", "English"))
    assert n == 0
    assert [s.text for s in out] == _EN_TRACK
    assert orch.fix_prompts == []


def test_coherence_audit_rejects_bad_repairs():
    # Unchanged, ballooned, and still-source-language replies must all be
    # rejected — the audit can never make a line worse than its draft.
    for bad in ("He never fired the missile.",           # identical
                "x" * 400,                                # runaway length
                "彼はミサイルを発射しなかった。"):          # untranslated
        out = _track(_EN_TRACK)
        n = asyncio.run(T.coherence_audit_and_fix(
            _SRC_TRACK, out, _AuditOrch([3], bad),
            "Japanese", "English"))
        assert n == 0, bad
        assert out[3].text == "He never fired the missile."


def test_coherence_audit_skips_tiny_tracks_without_a_call():
    out = _track(_EN_TRACK[:5])
    orch = _AuditOrch([1], "unused")

    async def _boom(prompt, **kw):
        raise AssertionError("must not be called")
    orch.text_completion = _boom
    n = asyncio.run(T.coherence_audit_and_fix(
        _SRC_TRACK[:5], out, orch, "Japanese", "English"))
    assert n == 0


# ── items 5+6: batch prompt shape (marks, scene breaks, translated tail) ────

class _BatchOrch:
    """Answers translation batches with EN<i> echoes; records every prompt.

    The numbered source lines are 'セリフ<i>です' so the reply can be built
    per line — including a deliberate [UNRELIABLE ASR] echo on the marked
    line to prove the output-side strip."""

    def __init__(self):
        self.prompts = []

    async def text_completion(self, prompt, **kw):
        self.prompts.append(prompt)
        nums = re.findall(r"^\d+\. (?:\[UNRELIABLE ASR\] )?セリフ(\d+)です$",
                          prompt.split("Lines:\n", 1)[-1], flags=re.M)
        out = []
        for n in nums:
            if f"[UNRELIABLE ASR] セリフ{n}です" in prompt:
                out.append(f"[UNRELIABLE ASR] EN{n}")
            else:
                out.append(f"EN{n}")
        return json.dumps(out)


def _mk_source_segments():
    """10 dialogue cues; cue 2 is low-confidence; a 12.5s hard cut sits
    between cues 3 and 4."""
    starts = [0.0, 2.0, 4.0, 6.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0]
    segs = []
    for i, st in enumerate(starts):
        s = {"text": f"セリフ{i}です", "start": st, "end": st + 1.5,
             "speaker": "Speaker 1"}
        if i == 2:
            s["avg_logprob"] = -1.5     # below WHISPER_REDECODE_LOGPROB
        else:
            s["avg_logprob"] = -0.2
        segs.append(s)
    return segs


def _run_translate(monkeypatch):
    monkeypatch.setattr(T.settings, "TRANSLATION_LLM_BATCH", 6, raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_PARALLEL_BATCHES", False,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_AUTO_GLOSSARY", False,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_CONTEXT_BRIEF", False,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_NAME_SECOND_VOTE", False,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_COHERENCE_AUDIT", False,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_LLM_CONTEXT_BEFORE", 4,
                        raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_LLM_CONTEXT_AFTER", 2,
                        raising=False)
    orch = _BatchOrch()
    out = asyncio.run(T.translate_via_llm(
        _mk_source_segments(), "ja", "en", orch, recovery_passes=0))
    return orch, out


def test_batch_prompts_carry_marks_context_and_tail(monkeypatch):
    orch, out = _run_translate(monkeypatch)
    assert out is not None and len(out) == 10
    batch_prompts = [p for p in orch.prompts if "Lines:\n" in p]
    assert len(batch_prompts) >= 2
    b0, b1 = batch_prompts[0], batch_prompts[1]

    # item 5: the low-confidence line is marked, the rule ships only then,
    # and the model's echoed marker is stripped from the final output
    assert "3. [UNRELIABLE ASR] セリフ2です" in b0
    assert "unreliable speech recognition" in b0
    assert not any("[UNRELIABLE ASR]" in (s.text or "") for s in out)
    assert out[2].text == "EN2"
    # batch 1 has no low-confidence line → no mark, no rule
    assert "[UNRELIABLE ASR]" not in b1

    # item 6: 4-before / 2-after windows around each batch
    assert "(after) セリフ6です" in b0 and "(after) セリフ7です" in b0
    for i in (2, 3, 4, 5):
        assert f"(before) セリフ{i}です" in b1

    # item 6: the 12.5s hard cut between cues 3 and 4 is announced inside
    # batch 1's before-context, exactly once
    pre = b1.split("Lines:\n", 1)[0]
    assert pre.count("(scene break)") == 1
    assert pre.index("(before) セリフ3です") < pre.index("(scene break)")
    assert pre.index("(scene break)") < pre.index("(before) セリフ4です")

    # item 6: batch 1 sees the tail of batch 0's TRANSLATED output
    assert "(already translated) EN4" in b1
    assert "(already translated) EN5" in b1
    # ...and batch 0 (no predecessor) has no tail
    assert "(already translated)" not in b0

    # 1:1 output integrity: every cue translated, timing preserved
    assert [s.text for s in out] == [f"EN{i}" for i in range(10)]
    assert out[4].start == 20.0 and out[4].end == 21.5
