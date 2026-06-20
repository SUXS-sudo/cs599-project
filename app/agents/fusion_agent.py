from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.cypher_agent import build_cypher_plan
from app.agents.sql_agent import build_sql_plan
from app.retriever import Recipe, RecipeRetriever
from app.services.neo4j_store import Neo4jStore
from app.services.logger import get_logger
from app.services.mysql_store import MySQLStore
from app.services.query_guard import ensure_limit, validate_readonly_cypher, validate_readonly_sql
from app.state import AgentState


SOURCE_WEIGHTS = {
    "rag": 0.55,
    "sql": 0.45,
    "cypher": 0.50,
}
logger = get_logger("agents.fusion")


@dataclass
class FusionItem:
    name: str
    score: float
    sources: set[str] = field(default_factory=set)
    payload: dict[str, Any] = field(default_factory=dict)


class FusionAgent:
    def __init__(
        self,
        retriever: RecipeRetriever | None = None,
        mysql_store: MySQLStore | None = None,
        neo4j_store: Neo4jStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.mysql_store = mysql_store
        self.neo4j_store = neo4j_store

    def run(self, state: AgentState) -> AgentState:
        source_results = state.meta.get("fusion_sources")
        if not isinstance(source_results, dict):
            source_results = self._collect_sources(state)

        fused = fuse_source_results(source_results, state.top_k)
        state.fusion_results = fused
        state.meta["fusion_results"] = fused
        state.meta["fusion_sources"] = {source: len(rows) for source, rows in source_results.items()}
        state.meta["fusion_status"] = "ok" if fused else "empty"
        state.agent_output = format_fusion_answer(fused)
        if not fused:
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "fusion_empty"
        logger.info("融合完成 状态=%s 结果数=%s", state.meta["fusion_status"], len(fused))
        return state

    def _collect_sources(self, state: AgentState) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {"rag": [], "sql": [], "cypher": []}
        if self.retriever:
            try:
                for recipe, score in self.retriever.search(state.user_input, max(state.top_k * 2, state.top_k)):
                    results["rag"].append(recipe_to_fusion_row(recipe, score))
            except Exception as exc:
                state.meta["fusion_rag_error"] = f"{type(exc).__name__}: {exc}"

        if self.mysql_store:
            plan = build_sql_plan(state.user_input, state.top_k)
            if plan:
                try:
                    sql = ensure_limit(plan["sql"], max(state.top_k * 2, state.top_k))
                    validate_readonly_sql(sql)
                    results["sql"] = self.mysql_store.read_query(sql, tuple(plan["params"]))
                except Exception as exc:
                    state.meta["fusion_sql_error"] = f"{type(exc).__name__}: {exc}"

        if self.neo4j_store:
            plan = build_cypher_plan(state.user_input, state.top_k)
            if plan:
                try:
                    cypher = ensure_limit(plan["cypher"], max(state.top_k * 2, state.top_k))
                    validate_readonly_cypher(cypher)
                    results["cypher"] = self.neo4j_store.execute_read(cypher, plan["params"])
                except Exception as exc:
                    state.meta["fusion_cypher_error"] = f"{type(exc).__name__}: {exc}"
        return results


def fuse_source_results(source_results: dict[str, list[dict[str, Any]]], top_k: int = 5) -> list[dict[str, Any]]:
    fused: dict[str, FusionItem] = {}
    for source, rows in source_results.items():
        weight = SOURCE_WEIGHTS.get(source, 0.4)
        for index, row in enumerate(rows):
            name = extract_name(row)
            if not name:
                continue
            base_score = extract_score(row, index, len(rows))
            item = fused.setdefault(name, FusionItem(name=name, score=0.0))
            item.sources.add(source)
            item.score += weight * base_score
            item.payload.update({key: value for key, value in row.items() if value is not None})

    ranked = []
    for item in fused.values():
        source_bonus = 0.2 * max(len(item.sources) - 1, 0)
        final_score = round(item.score + source_bonus, 4)
        ranked.append(
            {
                "name": item.name,
                "score": final_score,
                "sources": sorted(item.sources),
                "source_count": len(item.sources),
                "payload": item.payload,
            }
        )
    ranked.sort(key=lambda row: (row["score"], row["source_count"], row["name"]), reverse=True)
    return ranked[:top_k]


def extract_name(row: dict[str, Any]) -> str:
    for key in ("name", "recipe_name", "title"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def extract_score(row: dict[str, Any], index: int, total: int) -> float:
    if isinstance(row.get("score"), (int, float)):
        return float(row["score"])
    if isinstance(row.get("calories"), (int, float)) and row["calories"] > 0:
        return max(0.1, 1.0 - min(float(row["calories"]), 800.0) / 1000.0)
    if total <= 1:
        return 1.0
    return max(0.1, 1.0 - index / total)


def recipe_to_fusion_row(recipe: Recipe, score: float) -> dict[str, Any]:
    return {
        "name": recipe.name,
        "score": score,
        "category": recipe.category,
        "calories": recipe.calories,
        "ingredients": recipe.ingredients,
        "tags": recipe.tags,
    }


def format_fusion_answer(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Fusion Agent 暂时没有从多源结果中融合出匹配菜谱。"
    lines = ["Fusion Agent 已融合 RAG/SQL/Cypher 多源结果："]
    for index, item in enumerate(items, start=1):
        sources = "、".join(item["sources"])
        calories = item.get("payload", {}).get("calories")
        calorie_text = f"，约{calories}千卡" if calories else ""
        lines.append(f"{index}. {item['name']}：综合分 {item['score']:.2f}，来源 {sources}{calorie_text}。")
    return "\n".join(lines)
