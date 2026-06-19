from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import data/recipes.json into Neo4j graph.")
    parser.add_argument(
        "--data-path",
        default=str(ROOT_DIR / "data" / "recipes.json"),
        help="Path to recipes JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print graph mapping counts without connecting to Neo4j.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all Neo4j nodes and relationships before importing.",
    )
    args = parser.parse_args()

    from app.services.neo4j_store import Neo4jConfig, Neo4jStore, dry_run_graph_counts, load_recipe_json

    data_path = Path(args.data_path)
    recipes = load_recipe_json(data_path)
    counts = dry_run_graph_counts(recipes)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "data_path": str(data_path),
                    **counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = Neo4jConfig.from_env()
    store = Neo4jStore(config)
    try:
        if args.reset:
            store.clear_graph()
        imported = store.import_recipes(recipes)
        stats = store.stats()
    except Exception as exc:
        print(f"neo4j_recipe_import_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"neo4j_import_target={config.uri}")
    if config.database:
        print(f"neo4j_database={config.database}")
    print("imported:")
    for key, value in imported.items():
        print(f"- {key}={value}")
    print("graph_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
