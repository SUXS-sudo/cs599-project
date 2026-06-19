from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SmartRecipe Neo4j graph.")
    parser.add_argument(
        "--ingredient",
        default="鸡胸肉",
        help="Ingredient name used for a sample graph query.",
    )
    parser.add_argument(
        "--goal",
        default="减脂",
        help="Goal name used for a sample graph query.",
    )
    args = parser.parse_args()

    from app.services.neo4j_store import Neo4jConfig, Neo4jStore

    config = Neo4jConfig.from_env()
    store = Neo4jStore(config)
    try:
        stats = store.stats()
        ingredient_rows = store.execute_read(
            """
            MATCH (recipe:Recipe)-[:USES]->(:Ingredient {name: $ingredient})
            RETURN recipe.name AS name, recipe.calories AS calories
            ORDER BY recipe.calories ASC
            LIMIT 5
            """,
            {"ingredient": args.ingredient},
        )
        goal_rows = store.execute_read(
            """
            MATCH (recipe:Recipe)-[:SUITABLE_FOR]->(:Goal {name: $goal})
            RETURN recipe.name AS name, recipe.calories AS calories
            ORDER BY recipe.calories ASC
            LIMIT 5
            """,
            {"goal": args.goal},
        )
    except Exception as exc:
        print(f"neo4j_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"neo4j_target={config.uri}")
    if config.database:
        print(f"neo4j_database={config.database}")
    print("graph_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    print(f"sample_ingredient={args.ingredient}")
    for row in ingredient_rows:
        print(f"- {row['name']} | calories={row['calories']}")
    print(f"sample_goal={args.goal}")
    for row in goal_rows:
        print(f"- {row['name']} | calories={row['calories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
