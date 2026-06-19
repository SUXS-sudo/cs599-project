from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.llm_client import load_dotenv
from app.services.logger import get_logger


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MEAL_TIMES = {"早餐", "午餐", "晚餐", "下午茶"}
CONSTRAINT_TAGS = {"低脂", "低糖", "低热量", "低盐", "少油", "清淡", "低碳", "高蛋白", "高纤维", "素食"}


CONSTRAINT_STATEMENTS = [
    "CREATE CONSTRAINT recipe_name IF NOT EXISTS FOR (n:Recipe) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (n:Ingredient) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (n:Tag) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT goal_name IF NOT EXISTS FOR (n:Goal) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT meal_time_name IF NOT EXISTS FOR (n:MealTime) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT constraint_name IF NOT EXISTS FOR (n:Constraint) REQUIRE n.name IS UNIQUE",
]
logger = get_logger("services.neo4j")


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str | None = None

    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        load_dotenv(override=True)
        database = os.getenv("NEO4J_DATABASE", "").strip() or None
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j_password"),
            database=database,
        )


class Neo4jStore:
    def __init__(self, config: Neo4jConfig | None = None) -> None:
        self.config = config or Neo4jConfig.from_env()

    def driver(self):
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise RuntimeError("neo4j is required. Install it with: pip install neo4j") from exc
        return GraphDatabase.driver(self.config.uri, auth=(self.config.user, self.config.password))

    def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        with self.driver() as driver:
            with driver.session(database=self.config.database) as session:
                session.execute_write(lambda tx: tx.run(query, parameters or {}).consume())

    def execute_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self.driver() as driver:
            with driver.session(database=self.config.database) as session:
                result = session.execute_read(lambda tx: list(tx.run(query, parameters or {})))
        rows = [dict(record) for record in result]
        logger.debug("neo4j read query rows=%s", len(rows))
        return rows

    def ensure_constraints(self) -> None:
        with self.driver() as driver:
            with driver.session(database=self.config.database) as session:
                for statement in CONSTRAINT_STATEMENTS:
                    session.run(statement).consume()

    def clear_graph(self) -> None:
        self.execute_write("MATCH (n) DETACH DELETE n")

    def import_recipes(self, recipes: list[dict[str, Any]]) -> dict[str, int]:
        self.ensure_constraints()
        graph_rows = [recipe_to_graph_row(recipe) for recipe in recipes]
        query = """
        UNWIND $recipes AS item
        MERGE (recipe:Recipe {name: item.name})
        SET recipe.category = item.category,
            recipe.cooking_time = item.cooking_time,
            recipe.difficulty = item.difficulty,
            recipe.calories = item.calories,
            recipe.steps = item.steps

        WITH recipe, item
        UNWIND item.ingredients AS ingredient_name
        MERGE (ingredient:Ingredient {name: ingredient_name})
        MERGE (recipe)-[:USES]->(ingredient)

        WITH recipe, item
        UNWIND item.tags AS tag_name
        MERGE (tag:Tag {name: tag_name})
        MERGE (recipe)-[:HAS_TAG]->(tag)

        WITH recipe, item
        UNWIND item.constraints AS constraint_name
        MERGE (constraint:Constraint {name: constraint_name})
        MERGE (recipe)-[:MATCHES]->(constraint)

        WITH recipe, item
        UNWIND item.goals AS goal_name
        MERGE (goal:Goal {name: goal_name})
        MERGE (recipe)-[:SUITABLE_FOR]->(goal)

        WITH recipe, item
        UNWIND item.meal_times AS meal_name
        MERGE (meal:MealTime {name: meal_name})
        MERGE (recipe)-[:SUITABLE_FOR]->(meal)
        """
        self.execute_write(query, {"recipes": graph_rows})
        return graph_counts_from_rows(graph_rows)

    def stats(self) -> dict[str, int]:
        query = """
        MATCH (n)
        WITH labels(n)[0] AS label, count(n) AS count
        RETURN label, count
        ORDER BY label
        """
        rows = self.execute_read(query)
        stats = {str(row["label"]): int(row["count"]) for row in rows}
        rel_query = """
        MATCH ()-[r]->()
        WITH type(r) AS type, count(r) AS count
        RETURN type, count
        ORDER BY type
        """
        rel_rows = self.execute_read(rel_query)
        for row in rel_rows:
            stats[f"REL:{row['type']}"] = int(row["count"])
        logger.info("neo4j stats collected keys=%s", len(stats))
        return stats


def recipe_to_graph_row(recipe: dict[str, Any]) -> dict[str, Any]:
    tags = sorted({tag.strip() for tag in recipe.get("tags", []) if tag.strip()})
    suitable_for = sorted({target.strip() for target in recipe.get("suitable_for", []) if target.strip()})
    constraints = sorted({tag for tag in tags if tag in CONSTRAINT_TAGS})
    goals = sorted({target for target in suitable_for if target not in MEAL_TIMES})
    meal_times = sorted({target for target in suitable_for if target in MEAL_TIMES})
    return {
        "name": recipe["name"],
        "category": recipe.get("category", ""),
        "cooking_time": recipe.get("cooking_time", ""),
        "difficulty": recipe.get("difficulty", ""),
        "calories": int(recipe.get("calories", 0)),
        "steps": recipe.get("steps", ""),
        "ingredients": sorted({item.strip() for item in recipe.get("ingredients", []) if item.strip()}),
        "tags": tags,
        "constraints": constraints,
        "goals": goals,
        "meal_times": meal_times,
    }


def graph_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "recipes": len(rows),
        "ingredients": len({item for row in rows for item in row["ingredients"]}),
        "tags": len({item for row in rows for item in row["tags"]}),
        "constraints": len({item for row in rows for item in row["constraints"]}),
        "goals": len({item for row in rows for item in row["goals"]}),
        "meal_times": len({item for row in rows for item in row["meal_times"]}),
        "uses_relationships": sum(len(row["ingredients"]) for row in rows),
        "tag_relationships": sum(len(row["tags"]) for row in rows),
        "constraint_relationships": sum(len(row["constraints"]) for row in rows),
        "suitable_for_relationships": sum(len(row["goals"]) + len(row["meal_times"]) for row in rows),
    }


def load_recipe_json(path: Path = ROOT_DIR / "data" / "recipes.json") -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def dry_run_graph_counts(recipes: list[dict[str, Any]]) -> dict[str, int]:
    return graph_counts_from_rows([recipe_to_graph_row(recipe) for recipe in recipes])
