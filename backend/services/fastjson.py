"""Fast JSON encode/decode with a strict stdlib fallback.

``orjson`` serializes/parses the multi-MB ``job.json`` and engine-checkpoint
payloads several times faster than stdlib ``json`` — and those run on every
progress persist and every job load, competing with ffmpeg/Whisper for CPU.
This module wraps it so callers get the speedup when ``orjson`` is installed
and *identical semantics* from stdlib ``json`` when it isn't (older images,
pure-Python sandboxes, test environments).

Semantics matched to stdlib:
  * compact output — ``separators=(",", ":")`` (orjson is always compact);
  * non-str dict keys are stringified (``OPT_NON_STR_KEYS``) — the engine
    checkpoint timelines are int-keyed (``time_ms``) and stdlib ``json``
    stringifies those on write;
  * a ``default=`` hook rescues unhandled leaf values (numpy scalars etc.);
    orjson additionally serializes numpy arrays natively
    (``OPT_SERIALIZE_NUMPY``) before consulting ``default``.

Human-facing pretty output (``JOB_JSON_PRETTY``) stays on stdlib ``json`` —
orjson's ``OPT_INDENT_2`` differs cosmetically and pretty mode is a debug
path, not a hot path.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via the monkeypatched fallback test
    import orjson as _orjson
except ImportError:  # fail-soft: stdlib json keeps everything working
    _orjson = None
    logger.info("orjson not installed — falling back to stdlib json "
                "(slower job.json / checkpoint serialization)")


def dumps_bytes(obj, default=None) -> bytes:
    """Serialize ``obj`` to compact JSON bytes (UTF-8).

    ``default`` receives unhandled objects, exactly like stdlib
    ``json.dumps(default=...)``.
    """
    if _orjson is not None:
        try:
            return _orjson.dumps(
                obj,
                default=default,
                option=(_orjson.OPT_SERIALIZE_NUMPY | _orjson.OPT_NON_STR_KEYS),
            )
        except Exception as exc:
            # A payload orjson can't take (e.g. an exotic default interplay)
            # must never fail the save — stdlib accepts a superset here.
            logger.info("orjson dumps failed (%s); using stdlib json", exc)
    return json.dumps(obj, separators=(",", ":"), default=default).encode("utf-8")


def loads(data):
    """Parse JSON from ``bytes`` or ``str``."""
    if _orjson is not None:
        try:
            return _orjson.loads(data)
        except Exception as exc:
            # e.g. stdlib-accepted extensions (NaN/Infinity literals) that
            # orjson rejects — fall back rather than fail the load.
            logger.info("orjson loads failed (%s); using stdlib json", exc)
    return json.loads(data)
