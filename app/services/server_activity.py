from __future__ import annotations

import os
import time
from pathlib import Path


ACTIVITY_FILE_ENV = "SMART_RECIPE_ACTIVITY_FILE"
_last_touch = 0.0


def mark_server_activity() -> None:
    """Notify the managed server launcher that an HTTP request was received."""
    global _last_touch
    activity_file = os.getenv(ACTIVITY_FILE_ENV, "").strip()
    if not activity_file:
        return

    # One filesystem update per second is enough for a five-minute watchdog.
    now = time.monotonic()
    if now - _last_touch < 1.0:
        return
    try:
        Path(activity_file).touch(exist_ok=True)
        _last_touch = now
    except OSError:
        # Activity tracking must never turn a successful API request into 500.
        pass
