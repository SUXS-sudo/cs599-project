from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT_DIR / "logs"
# Do not rename an open log file on Windows.  Uvicorn's reload process and a
# previous development server may briefly have the same file open, which makes
# TimedRotatingFileHandler fail with WinError 32 during its midnight rollover.
# A date in the filename gives each server start a date-stamped log without an
# in-place rename. A server that stays up across midnight keeps its start date.
LOG_FILE = LOG_DIR / f"smart_recipe-{date.today():%Y-%m-%d}.log"


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

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
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
