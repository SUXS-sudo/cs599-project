from __future__ import annotations

import json
import os
import re
from typing import Any

from app.services.llm_client import LLMClient, load_dotenv
from app.services.logger import get_logger
from app.services.neo4j_store import Neo4jStore
from app.services.query_guard import ensure_limit, validate_readonly_cypher
from app.state import AgentState


KNOWN_INGREDIENTS = ("鸡胸肉", "西兰花", "豆腐", "鸡蛋", "番茄", "虾仁", "黄瓜", "牛肉", "南瓜", "青菜")
KNOWN_GOALS = ("减脂", "健身", "高蛋白", "控糖", "早餐", "午餐", "晚餐", "老人", "儿童")
KNOWN_CONSTRAINTS = ("低脂", "低糖", "低盐", "低热量", "少油", "清淡", "高蛋白", "高纤维", "素食")
KNOWN_RECIPES = ("西兰花炒鸡胸肉", "番茄炒蛋", "鸡胸肉沙拉", "虾仁西兰花", "豆腐青菜汤", "低脂酸辣汤")
logger = get_logger("agents.cypher")


class CypherAgent:
    def __init__(
        self,
        store: Neo4jStore | None = None,
        llm_client: LLMClient | None = None,
        enable_llm_query: bool | None = None,
    ) -> None:
        self.store = store or Neo4jStore()
        self.llm_client = llm_client
        if enable_llm_query is None:
            load_dotenv()
            enable_llm_query = parse_bool(os.getenv("ENABLE_LLM_QUERY_GENERATION", "false"))
        self.enable_llm_query = enable_llm_query

    def run(self, state: AgentState) -> AgentState:
        plan = self._build_llm_plan(state) or build_cypher_plan(state.user_input, state.top_k)
        if plan is None:
            state.agent_output = "Cypher Agent 暂时只支持食材搭配、目标推荐、标签/约束关系查询。"
            state.meta["cypher_status"] = "unsupported"
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "cypher_rule_unsupported"
            return state

        cypher = ensure_limit(plan["cypher"], max(state.top_k, 5))
        validate_readonly_cypher(cypher)
        try:
            rows = self.store.execute_read(cypher, plan["params"])
        except Exception as exc:
            logger.exception("Cypher 查询失败")
            state.agent_output = f"Cypher Agent 查询失败：{type(exc).__name__}: {exc}"
            state.meta["cypher_status"] = "failed"
            state.meta["cypher_query"] = cypher
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "cypher_failed"
            return state

        state.agent_output = format_cypher_answer(rows, plan["title"])
        state.meta["cypher_status"] = "ok" if rows else "empty"
        state.meta["cypher_query"] = cypher
        state.meta["cypher_rows"] = rows
        if not rows:
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "cypher_empty"
        logger.info("Cypher 查询成功 行数=%s 模式=%s", len(rows), state.meta.get("cypher_query_mode", "rule"))
        return state

    def _build_llm_plan(self, state: AgentState) -> dict[str, Any] | None:
        if not self.enable_llm_query:
            state.meta["cypher_query_mode"] = "rule_disabled"
            return None
        if not self.llm_client or not self.llm_client.available:
            state.meta["cypher_query_mode"] = "rule_unavailable"
            return None

        raw = self.llm_client.generate(build_cypher_prompt(state.user_input, state.top_k), max_tokens=300, timeout=8)
        if not raw:
            logger.debug("Cypher 生成 LLM 返回为空")
            state.meta["cypher_query_mode"] = "rule_llm_empty"
            return None
        plan = parse_llm_cypher_plan(raw)
        if not plan:
            state.meta["cypher_query_mode"] = "rule_llm_parse_failed"
            return None
        try:
            validate_readonly_cypher(ensure_limit(plan["cypher"], max(state.top_k, 5)))
        except ValueError as exc:
            logger.warning("Cypher 生成结果被安全校验拒绝：%s", exc)
            state.meta["cypher_query_mode"] = "rule_llm_guard_failed"
            state.meta["cypher_guard_error"] = str(exc)
            return None
        state.meta["cypher_query_mode"] = "llm"
        return plan


def build_cypher_prompt(message: str, top_k: int) -> str:
    return (
        "你是 SmartRecipe 的 Text2Cypher 生成器。只允许为 Neo4j 生成只读 MATCH/OPTIONAL MATCH 查询。\n"
        "图谱节点：Recipe(name,category,cooking_time_minutes,difficulty,calories_per_100g,"
        "protein_g_per_100g,fat_g_per_100g,nutrition_estimated)、Ingredient(name)、Tag(name)、"
        "Constraint(name)、Goal(name)、MealTime(name)。关系：USES、HAS_TAG、MATCHES、SUITABLE_FOR。\n"
        "不要生成 CREATE/MERGE/SET/DELETE/DETACH/REMOVE/DROP/CALL dbms。参数使用 $name。\n"
        "返回紧凑 JSON，不要 markdown："
        '{"cypher":"MATCH ... RETURN ...","params":{"name":"参数"},"title":"查询标题"}\n'
        f"默认 LIMIT 不超过 {max(top_k, 5)}。\n"
        f"用户问题：{message}"
    )


def parse_llm_cypher_plan(raw: str) -> dict[str, Any] | None:
    data = parse_json_object(raw)
    if not data:
        return None
    cypher = str(data.get("cypher", "")).strip()
    if not cypher:
        return None
    params = data.get("params", {})
    if not isinstance(params, dict):
        params = {}
    return {
        "cypher": cypher,
        "params": params,
        "title": str(data.get("title") or "图谱关系查询"),
    }


def build_cypher_plan(message: str, top_k: int) -> dict[str, Any] | None:
    text = message.strip()
    ingredient = first_match(text, KNOWN_INGREDIENTS)
    ingredients = all_matches(text, KNOWN_INGREDIENTS)
    goal = first_match(text, KNOWN_GOALS)
    constraint = first_match(text, KNOWN_CONSTRAINTS)
    recipe_name = first_match(text, KNOWN_RECIPES)

    if recipe_name and any(word in text for word in ("关系", "图谱", "关联", "包含", "标签")):
        return {
            "title": f"{recipe_name}的图谱关系",
            "cypher": """
            MATCH (recipe:Recipe {name: $recipe_name})-[rel]->(node)
            RETURN type(rel) AS relation, labels(node)[0] AS label, node.name AS name
            ORDER BY relation, label, name
            """,
            "params": {"recipe_name": recipe_name},
        }

    if len(ingredients) >= 2:
        left, right = ingredients[:2]
        return {
            "title": f"同时包含{left}和{right}的菜谱",
            "cypher": """
            MATCH (recipe:Recipe)-[:USES]->(:Ingredient {name: $left})
            MATCH (recipe)-[:USES]->(:Ingredient {name: $right})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"left": left, "right": right},
        }

    if ingredient and any(word in text for word in ("搭配", "一起", "关联", "相关")):
        return {
            "title": f"和{ingredient}常一起出现的食材",
            "cypher": """
            MATCH (:Ingredient {name: $ingredient})<-[:USES]-(recipe:Recipe)-[:USES]->(other:Ingredient)
            WHERE other.name <> $ingredient
            RETURN other.name AS name, count(recipe) AS recipe_count, collect(recipe.name)[0..3] AS examples
            ORDER BY recipe_count DESC, name ASC
            """,
            "params": {"ingredient": ingredient},
        }

    if ingredient and (goal or constraint):
        relation = "Goal" if goal else "Constraint"
        target = goal or constraint
        edge = "SUITABLE_FOR" if goal else "MATCHES"
        return {
            "title": f"同时包含{ingredient}并匹配{target}的菜谱",
            "cypher": f"""
            MATCH (recipe:Recipe)-[:USES]->(:Ingredient {{name: $ingredient}})
            MATCH (recipe)-[:{edge}]->(:{relation} {{name: $target}})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"ingredient": ingredient, "target": target},
        }

    if goal and constraint:
        return {
            "title": f"同时适合{goal}并匹配{constraint}的菜谱",
            "cypher": """
            MATCH (recipe:Recipe)-[:SUITABLE_FOR]->(:Goal {name: $goal})
            MATCH (recipe)-[:MATCHES]->(:Constraint {name: $constraint})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"goal": goal, "constraint": constraint},
        }

    if goal:
        return {
            "title": f"适合{goal}的菜谱",
            "cypher": """
            MATCH (recipe:Recipe)-[:SUITABLE_FOR]->(:Goal {name: $goal})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"goal": goal},
        }

    if constraint:
        return {
            "title": f"匹配{constraint}约束的菜谱",
            "cypher": """
            MATCH (recipe:Recipe)-[:MATCHES]->(:Constraint {name: $constraint})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"constraint": constraint},
        }

    if ingredient:
        return {
            "title": f"包含{ingredient}的菜谱",
            "cypher": """
            MATCH (recipe:Recipe)-[:USES]->(:Ingredient {name: $ingredient})
            RETURN recipe.name AS name, recipe.category AS category,
                   recipe.calories_per_100g AS calories,
                   recipe.protein_g_per_100g AS protein_g_per_100g,
                   recipe.fat_g_per_100g AS fat_g_per_100g
            ORDER BY recipe.calories_per_100g ASC
            """,
            "params": {"ingredient": ingredient},
        }

    return None


def first_match(text: str, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in text:
            return candidate
    return None


def all_matches(text: str, candidates: tuple[str, ...]) -> list[str]:
    return [candidate for candidate in candidates if candidate in text]


def format_cypher_answer(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"{title}：暂时没有查到匹配结果。"
    lines = [f"{title}："]
    for index, row in enumerate(rows, start=1):
        if "examples" in row:
            examples = "、".join(row.get("examples") or [])
            lines.append(f"{index}. {row['name']}：共同出现 {row['recipe_count']} 次，示例菜谱：{examples}。")
        elif "relation" in row:
            lines.append(f"{index}. {row['relation']} -> {row['label']}：{row['name']}。")
        else:
            lines.append(
                f"{index}. {row['name']}：{row.get('category') or '未分类'}，约{row.get('calories')} kcal/100g。"
            )
    return "\n".join(lines)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
