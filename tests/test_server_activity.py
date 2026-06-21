from __future__ import annotations

from pathlib import Path

from src.services import server_activity


def test_mark_server_activity_touches_managed_heartbeat(monkeypatch) -> None:
    heartbeat = Path(__file__).resolve().parent.parent / "logs" / "test_server_activity.heartbeat"
    heartbeat.unlink(missing_ok=True)
    monkeypatch.setenv(server_activity.ACTIVITY_FILE_ENV, str(heartbeat))
    monkeypatch.setattr(server_activity, "_last_touch", 0.0)

    try:
        server_activity.mark_server_activity()

        assert heartbeat.exists()
    finally:
        heartbeat.unlink(missing_ok=True)


def test_mark_server_activity_is_optional(monkeypatch) -> None:
    monkeypatch.delenv(server_activity.ACTIVITY_FILE_ENV, raising=False)
    server_activity.mark_server_activity()
