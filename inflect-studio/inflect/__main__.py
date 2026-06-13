"""Entry point: ``python -m inflect``."""

from __future__ import annotations

import sys

from .app import run

if __name__ == "__main__":
    sys.exit(run())
