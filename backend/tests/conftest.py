"""Shared pytest fixtures / test-isolation guards for the backend suite."""
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_backend_config():
    """Restore ``sys.modules['backend.config']`` after every test.

    Several NMT unit tests replace ``sys.modules['backend.config']`` with a
    partial ``SimpleNamespace`` stub (so ``nmt_translator``'s lazy
    ``from backend.config import settings`` works without pydantic-settings)
    and never put the real module back. That stub — which lacks almost every
    real setting — then leaks into any later test that reads ``settings``,
    making order-dependent failures (e.g. a monkeypatched flag silently
    reverting to a stub default). Snapshotting + restoring the module around
    each test makes those tests hermetic regardless of run order."""
    saved = sys.modules.get("backend.config")
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["backend.config"] = saved
        else:
            sys.modules.pop("backend.config", None)
