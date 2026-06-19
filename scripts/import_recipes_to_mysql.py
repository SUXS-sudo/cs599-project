from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import data/recipes.json into MySQL.")
    parser.add_argument(
        "--data-path",
        default=str(ROOT_DIR / "data" / "recipes.json"),
        help="Path to recipes JSON.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate recipe-related tables before importing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSON and print counts without connecting to MySQL.",
    )
    args = parser.parse_args()

    from app.services.mysql_store import MySQLConfig, MySQLStore, load_recipe_json

    data_path = Path(args.data_path)
    recipes = load_recipe_json(data_path)
    unique_ingredients = {
        ingredient
        for recipe in recipes
        for ingredient in recipe.get("ingredients", [])
        if ingredient.strip()
    }
    unique_tags = {
        tag
        for recipe in recipes
        for tag in recipe.get("tags", [])
        if tag.strip()
    }
    unique_targets = {
        target
        for recipe in recipes
        for target in recipe.get("suitable_for", [])
        if target.strip()
    }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "data_path": str(data_path),
                    "recipes": len(recipes),
                    "unique_ingredients": len(unique_ingredients),
                    "unique_tags": len(unique_tags),
                    "unique_suitable_for": len(unique_targets),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    try:
        store.ensure_schema()
        if args.reset:
            store.reset_recipe_tables()
        counts = store.import_recipes(recipes)
        stats = store.stats()
    except Exception as exc:
        print(f"mysql_recipe_import_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_import_target={config.host}:{config.port}/{config.database}")
    print("imported:")
    for key, value in counts.items():
        print(f"- {key}={value}")
    print("database_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
