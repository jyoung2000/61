"""The readability/export stack self-heal.

A user_settings.json snapshot that pinned SUBTITLE_CPS_ENFORCEMENT (or its
sibling toggles) to False re-ships raw, unwrapped subtitle downloads forever —
a measured file carried 97-char single-line cues and baked speaker labels.
The heal runs ONCE (marker key), drops the pinned False in favour of the
shipped default, and afterwards an explicit False sticks.
"""
import json

from backend.config import settings


def test_readability_stack_self_heal_runs_once(tmp_path, monkeypatch):
    from backend.routers import settings as S

    p = tmp_path / "user_settings.json"
    p.write_text(json.dumps({
        "SUBTITLE_CPS_ENFORCEMENT": False,
        "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES": False,
        "SUBTITLE_SMART_LINE_BREAKS": True,
    }))
    monkeypatch.setattr(S, "USER_SETTINGS_PATH", str(p))

    prev = {
        "SUBTITLE_CPS_ENFORCEMENT": settings.SUBTITLE_CPS_ENFORCEMENT,
        "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES":
            settings.SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES,
        "SUBTITLE_SMART_LINE_BREAKS": settings.SUBTITLE_SMART_LINE_BREAKS,
    }
    try:
        S._restore_user_settings()
        # The pinned Falses were dropped — runtime keeps the shipped defaults.
        assert settings.SUBTITLE_CPS_ENFORCEMENT is True
        assert settings.SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES is True
        # An unrelated True restores normally.
        assert settings.SUBTITLE_SMART_LINE_BREAKS is True
        data = json.loads(p.read_text())
        assert data.get("_READABILITY_STACK_HEALED_V1") is True
        assert "SUBTITLE_CPS_ENFORCEMENT" not in data
        assert "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES" not in data

        # AFTER the heal, an explicit False is a deliberate choice and sticks.
        data["SUBTITLE_CPS_ENFORCEMENT"] = False
        p.write_text(json.dumps(data))
        S._restore_user_settings()
        assert settings.SUBTITLE_CPS_ENFORCEMENT is False
    finally:
        for k, v in prev.items():
            setattr(settings, k, v)
