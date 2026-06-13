"""Shared pytest fixtures for the top-level test suite.

The suite runs as a single process and several modules mutate process-global
state, so co-running modules can leak into one another (a test passes alone but
fails in the full run). These autouse fixtures restore that state around every
test:

  * ``_restore_settings`` snapshots the global ``backend.config.settings`` object
    before each test and rolls every field back afterwards. A test that mutates a
    setting without ``monkeypatch`` (or via a helper that never restores it) then
    can't pollute a later test that reads it back.
  * ``_ensure_event_loop`` guarantees every test starts with an open, current
    event loop. Some modules drive coroutines with ``asyncio.run()``, which
    closes its loop and clears the current one (Python >= 3.10); a later module
    using the legacy ``asyncio.get_event_loop()`` pattern would otherwise raise
    "There is no current event loop" purely because of collection order. This is
    what made ``tests/test_model_persistence.py`` pass alone but fail when run
    after the translation tests.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.config import settings


@pytest.fixture(autouse=True)
def _restore_settings():
    snapshot = settings.model_dump()
    try:
        yield
    finally:
        for k, v in snapshot.items():
            try:
                setattr(settings, k, v)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def _ensure_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield
    finally:
        # The test may have installed and/or closed its own loop (e.g. via
        # asyncio.run()); close ours if it's still open, then re-arm a fresh
        # current loop so the next test's get_event_loop() always succeeds.
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())
