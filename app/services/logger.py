from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "smart_recipe.log"


def configure_logging() -> None:
    configure_console_encoding()
    logger = logging.getLogger("smart_recipe")
    if logger.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"smart_recipe.{name}")


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
