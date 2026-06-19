from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize SmartRecipe MySQL schema.")
    parser.add_argument(
        "--print-sql",
        action="store_true",
        help="Print schema SQL without connecting to MySQL.",
    )
    args = parser.parse_args()

    from app.services.mysql_store import MySQLConfig, MySQLStore, schema_sql

    config = MySQLConfig.from_env()
    if args.print_sql:
        print(schema_sql(config.database))
        return 0

    store = MySQLStore(config)
    try:
        store.ensure_schema()
    except Exception as exc:
        print(f"mysql_schema_init_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_schema_ready={config.host}:{config.port}/{config.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
