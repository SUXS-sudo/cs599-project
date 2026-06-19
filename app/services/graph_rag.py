from __future__ import annotations

from typing import Any

from app.retriever import Recipe
from app.services.logger import get_logger
from app.services.neo4j_store import Neo4jStore


logger = get_logger("services.graph_rag")


class GraphRAG:
    def __init__(self, store: Neo4jStore | None = None) -> None:
        self.store = store or Neo4jStore()

    def enrich(self, recipes: list[tuple[Recipe, float]], limit: int = 3) -> dict[str, list[dict[str, Any]]]:
        names = [recipe.name for recipe, _ in recipes[:limit]]
        if not names:
            return {}
        query = """
        MATCH (recipe:Recipe)
        WHERE recipe.name IN $names
        OPTIONAL MATCH (recipe)-[:USES]->(ingredient:Ingredient)<-[:USES]-(peer:Recipe)
        WHERE peer.name <> recipe.name
        WITH recipe, collect(DISTINCT ingredient.name)[0..5] AS shared_ingredients,
             collect(DISTINCT peer.name)[0..5] AS ingredient_peers
        OPTIONAL MATCH (recipe)-[:SUITABLE_FOR]->(goal)<-[:SUITABLE_FOR]-(goal_peer:Recipe)
        WHERE goal_peer.name <> recipe.name
        WITH recipe, shared_ingredients, ingredient_peers,
             collect(DISTINCT goal.name)[0..5] AS shared_goals,
             collect(DISTINCT goal_peer.name)[0..5] AS goal_peers
        OPTIONAL MATCH (category_peer:Recipe {category: recipe.category})
        WHERE category_peer.name <> recipe.name
        RETURN recipe.name AS name,
               recipe.category AS category,
               shared_ingredients,
               ingredient_peers,
               shared_goals,
               goal_peers,
               collect(DISTINCT category_peer.name)[0..5] AS category_peers
        """
        rows = self.store.execute_read(query, {"names": names})
        logger.info("graph_rag enrich recipe_count=%s row_count=%s", len(names), len(rows))
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            name = str(row.get("name", ""))
            if name:
                grouped.setdefault(name, []).append(normalize_graph_row(row))
        return grouped


def normalize_graph_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": row.get("category") or "",
        "shared_ingredients": clean_list(row.get("shared_ingredients")),
        "ingredient_peers": clean_list(row.get("ingredient_peers")),
        "shared_goals": clean_list(row.get("shared_goals")),
        "goal_peers": clean_list(row.get("goal_peers")),
        "category_peers": clean_list(row.get("category_peers")),
    }


def clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def format_graph_context(context: dict[str, list[dict[str, Any]]]) -> str:
    if not context:
        return ""
    lines = ["图谱增强信息："]
    for recipe_name, rows in context.items():
        row = rows[0] if rows else {}
        parts = []
        if row.get("shared_ingredients"):
            parts.append("共现食材=" + "、".join(row["shared_ingredients"]))
        if row.get("goal_peers"):
            parts.append("同人群菜谱=" + "、".join(row["goal_peers"][:3]))
        if row.get("category_peers"):
            parts.append("同菜系菜谱=" + "、".join(row["category_peers"][:3]))
        if parts:
            lines.append(f"- {recipe_name}：" + "；".join(parts))
    return "\n".join(lines) if len(lines) > 1 else ""
