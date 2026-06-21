from __future__ import annotations

from pathlib import Path

from src.services import logger as logger_service


def test_logging_configuration_creates_log_file() -> None:
    logger_service.configure_logging()
    log = logger_service.get_logger("tests")
    log.debug("logging test message")

    assert logger_service.LOG_FILE.exists()
    assert logger_service.LOG_FILE.name.startswith("smart_recipe-")
    assert logger_service.LOG_FILE.suffix == ".log"


def test_logs_directory_is_ignored() -> None:
    root = Path(__file__).resolve().parent.parent
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    assert "logs/" in gitignore.splitlines()
