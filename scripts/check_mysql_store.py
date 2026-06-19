from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    from app.services.mysql_store import MySQLConfig, MySQLStore

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    try:
        stats = store.stats()
    except Exception as exc:
        print(f"mysql_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_target={config.host}:{config.port}/{config.database}")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
