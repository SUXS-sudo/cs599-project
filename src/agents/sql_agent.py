from __future__ import annotations

import json
import os
import re
from typing import Any

from src.services.llm_client import LLMClient, load_dotenv
from src.services.logger import get_logger
from src.services.cache_store import cache_data_version, cache_ttl_seconds, get_cache_store, stable_cache_key
from src.services.mysql_store import MySQLStore
from src.services.query_guard import ensure_limit, validate_readonly_sql
from src.state import AgentState


MEAL_WORDS = ("早餐", "午餐", "晚餐", "下午茶")
TAG_WORDS = ("低脂", "低糖", "低盐", "低热量", "少油", "清淡", "高蛋白", "高纤维", "减脂", "健身", "控糖")
CATEGORY_WORDS = ("早餐", "汤羹", "家常菜", "轻食", "减脂餐", "素菜", "面食", "粥", "沙拉", "便当")
INGREDIENT_WORDS = ("鸡胸肉", "西兰花", "豆腐", "鸡蛋", "番茄", "虾仁", "黄瓜", "牛肉", "南瓜", "青菜", "红薯", "燕麦")
logger = get_logger("agents.sql")


class SQLAgent:
    def __init__(
        self,
        store: MySQLStore | None = None,
        llm_client: LLMClient | None = None,
        enable_llm_query: bool | None = None,
    ) -> None:
        self.store = store or MySQLStore()
        self.cache = get_cache_store()
        self.llm_client = llm_client
        if enable_llm_query is None:
            load_dotenv()
            enable_llm_query = parse_bool(os.getenv("ENABLE_LLM_QUERY_GENERATION", "false"))
        self.enable_llm_query = enable_llm_query

    def run(self, state: AgentState) -> AgentState:
        plan = self._build_llm_plan(state) or build_sql_plan(state.user_input, state.top_k)
        if plan is None:
            state.agent_output = "SQL Agent 暂时只支持按热量、时间、分类和食材筛选菜谱。"
            state.meta["sql_status"] = "unsupported"
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "sql_rule_unsupported"
            return state

        sql = ensure_limit(plan["sql"], max(state.top_k, 5))
        validate_readonly_sql(sql)
        parameters = tuple(plan["params"])
        cached_rows = self._get_cached_rows(sql, parameters)
        if cached_rows is not None:
            rows = cached_rows
            state.meta["sql_cache_hit"] = True
        else:
            try:
                rows = self.store.read_query(sql, parameters)
            except Exception as exc:
                logger.exception("SQL 查询失败")
                state.agent_output = f"SQL Agent 查询失败：{type(exc).__name__}: {exc}"
                state.meta["sql_status"] = "failed"
                state.meta["sql_query"] = sql
                state.meta["recipe_source"] = "llm_fallback_query"
                state.meta["fallback_reason"] = "sql_failed"
                return state
            self._set_cached_rows(sql, parameters, rows)
            state.meta["sql_cache_hit"] = False

        if not rows:
            title = str(plan.get("title") or "符合条件的菜谱")
            state.agent_output = f"{title}：本地数据库暂时没有查到匹配结果。"
            state.meta["sql_status"] = "empty"
            state.meta["sql_query"] = sql
            state.meta["sql_rows"] = []
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "sql_empty"
            state.meta.pop("answer_mode", None)
            logger.info("SQL 查询为空 行数=0 模式=%s", state.meta.get("sql_query_mode", "rule"))
            return state

        state.agent_output = format_sql_answer(rows, plan["title"])
        state.meta["sql_status"] = "ok"
        state.meta["sql_query"] = sql
        state.meta["sql_rows"] = rows
        state.meta["answer_mode"] = "direct"
        state.meta["recipe_source"] = "sql_fast"
        logger.info("SQL 查询成功 行数=%s 模式=%s", len(rows), state.meta.get("sql_query_mode", "rule"))
        return state

    def _sql_cache_key(self, sql: str, parameters: tuple[Any, ...]) -> str:
        normalized_sql = re.sub(r"\s+", " ", sql).strip()
        return stable_cache_key(
            "sql",
            {
                "sql": normalized_sql,
                "parameters": list(parameters),
                "data_version": cache_data_version(),
            },
        )

    def _get_cached_rows(self, sql: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]] | None:
        data = self.cache.get_json(self._sql_cache_key(sql, parameters))
        if not isinstance(data, list):
            return None
        rows = [row for row in data if isinstance(row, dict)]
        return rows if rows else None

    def _set_cached_rows(self, sql: str, parameters: tuple[Any, ...], rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        ttl = cache_ttl_seconds("CACHE_SQL_TTL_SECONDS", 24 * 60 * 60)
        self.cache.set_json(self._sql_cache_key(sql, parameters), rows, ttl_seconds=ttl)

    def _build_llm_plan(self, state: AgentState) -> dict[str, Any] | None:
        if not self.enable_llm_query:
            state.meta["sql_query_mode"] = "rule_disabled"
            return None
        if not self.llm_client or not self.llm_client.available:
            state.meta["sql_query_mode"] = "rule_unavailable"
            return None

        prompt = build_sql_prompt(state.user_input, state.top_k)
        raw = self.llm_client.generate(prompt, max_tokens=300, timeout=8)
        if not raw:
            logger.debug("SQL 生成 LLM 返回为空")
            state.meta["sql_query_mode"] = "rule_llm_empty"
            return None
        plan = parse_llm_sql_plan(raw)
        if not plan:
            state.meta["sql_query_mode"] = "rule_llm_parse_failed"
            return None
        try:
            validate_readonly_sql(ensure_limit(plan["sql"], max(state.top_k, 5)))
        except ValueError as exc:
            logger.warning("SQL 生成结果被安全校验拒绝：%s", exc)
            state.meta["sql_query_mode"] = "rule_llm_guard_failed"
            state.meta["sql_guard_error"] = str(exc)
            return None
        state.meta["sql_query_mode"] = "llm"
        return plan


def build_sql_prompt(message: str, top_k: int) -> str:
    return (
        "你是 SmartRecipe 的 Text2SQL 生成器。只允许为 MySQL 生成只读 SELECT 查询。\n"
        "可用表：recipes r(id,name,category,cooking_time_minutes,difficulty,calories_per_100g,"
        "protein_g_per_100g,fat_g_per_100g,nutrition_estimated), "
        "ingredients i(id,name), recipe_ingredients ri(recipe_id,ingredient_id,amount_text)。\n"
        "不要生成 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE。参数用 %s 占位。\n"
        "返回紧凑 JSON，不要 markdown："
        '{"sql":"SELECT ...","params":["参数"],"title":"查询标题","answer_type":"recipes"}\n'
        f"默认 LIMIT 不超过 {max(top_k, 5)}。\n"
        f"用户问题：{message}"
    )


def parse_llm_sql_plan(raw: str) -> dict[str, Any] | None:
    data = parse_json_object(raw)
    if not data:
        return None
    sql = str(data.get("sql", "")).strip()
    if not sql:
        return None
    params = data.get("params", [])
    if not isinstance(params, list):
        params = []
    return {
        "sql": sql,
        "params": params,
        "title": str(data.get("title") or "符合条件的菜谱"),
        "answer_type": str(data.get("answer_type") or "recipes"),
    }


def build_sql_plan(message: str, top_k: int) -> dict[str, Any] | None:
    text = message.strip()
    order = "ASC" if any(word in text for word in ("最低", "低", "少", "最少")) else "DESC"
    title = "符合条件的菜谱"
    where = []
    params: list[Any] = []
    joins = []

    aggregate = None
    if "平均" in text and any(word in text for word in ("热量", "卡路里")):
        aggregate = "average_calories"
        title = "平均热量统计"
    elif any(word in text for word in ("多少道", "几个", "数量")) and "推荐" not in text:
        aggregate = "count"
        title = "菜谱数量统计"

    if any(word in text for word in ("热量", "卡路里", "低卡")):
        title = "按热量排序的菜谱"
    if "分钟" in text or "以内" in text:
        minutes = extract_first_number(text)
        if minutes:
            where.append("r.cooking_time_minutes <= %s")
            params.append(minutes)
            title = f"{minutes} 分钟以内的菜谱"

    for word in CATEGORY_WORDS:
        if word in text:
            where.append("r.category = %s")
            params.append(word)
            title = f"适合{word}的菜谱"
            break

    for word in INGREDIENT_WORDS:
        if word in text:
            joins.append("LEFT JOIN recipe_ingredients ri ON ri.recipe_id = r.id")
            joins.append("LEFT JOIN ingredients i ON i.id = ri.ingredient_id")
            where.append("i.name = %s")
            params.append(word)
            title = f"包含{word}的菜谱"
            break

    if "高蛋白" in text:
        where.append("r.protein_g_per_100g >= %s")
        params.append(12)
        title = "高蛋白菜谱"
    if "低脂" in text or ("脂肪" in text and "低" in text):
        where.append("r.fat_g_per_100g <= %s")
        params.append(8)
        title = "低脂菜谱" if "高蛋白" not in text else "低脂高蛋白菜谱"
    if "低热量" in text or "低卡" in text:
        where.append("r.calories_per_100g <= %s")
        params.append(150)
        title = "低热量菜谱"

    unsupported_tags = tuple(word for word in TAG_WORDS if word not in {"低脂", "低热量", "高蛋白"})
    if any(word in text for word in unsupported_tags) and not where:
        return None

    if not where and not any(word in text for word in ("热量", "卡路里", "最高", "最低", "多少", "平均", "数量")):
        return None

    where_sql = " AND ".join(where) if where else "1 = 1"
    join_sql = "\n    ".join(dict.fromkeys(joins))
    if aggregate == "average_calories":
        sql = f"""
        SELECT ROUND(AVG(r.calories_per_100g), 1) AS avg_calories, COUNT(DISTINCT r.id) AS recipe_count
        FROM recipes r
        {join_sql}
        WHERE {where_sql}
        """
        return {"sql": sql, "params": params, "title": title, "answer_type": aggregate}
    if aggregate == "count":
        sql = f"""
        SELECT COUNT(DISTINCT r.id) AS recipe_count
        FROM recipes r
        {join_sql}
        WHERE {where_sql}
        """
        return {"sql": sql, "params": params, "title": title, "answer_type": aggregate}

    sql = f"""
    SELECT DISTINCT
      r.name,
      r.category,
      r.cooking_time_minutes,
      r.difficulty,
      r.calories_per_100g AS calories,
      r.protein_g_per_100g,
      r.fat_g_per_100g,
      r.nutrition_estimated
    FROM recipes r
    {join_sql}
    WHERE {where_sql}
    ORDER BY r.calories_per_100g {order}, r.cooking_time_minutes ASC
    """
    return {"sql": sql, "params": params, "title": title, "answer_type": "recipes"}


def extract_first_number(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if not match:
        return None
    return int(match.group(0))


def format_sql_answer(rows: list[dict[str, Any]], title: str) -> str:
    if not rows:
        return f"{title}：暂时没有查到匹配结果。"
    lines = [f"{title}："]
    if "avg_calories" in rows[0]:
        row = rows[0]
        return f"{title}：共 {row.get('recipe_count', 0)} 道菜，平均热量约 {row.get('avg_calories')} kcal/100g。"
    if "recipe_count" in rows[0] and len(rows[0]) == 1:
        return f"{title}：共 {rows[0].get('recipe_count', 0)} 道菜。"
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. {row['name']}：{row.get('category') or '未分类'}，"
            f"{format_minutes(row.get('cooking_time_minutes'))}，"
            f"难度{row.get('difficulty') or '未知'}，约{row.get('calories')} kcal/100g，"
            f"蛋白质{row.get('protein_g_per_100g') or 0}g/100g，脂肪{row.get('fat_g_per_100g') or 0}g/100g。"
        )
    return "\n".join(lines)


def format_minutes(value: Any) -> str:
    try:
        minutes = int(value or 0)
    except (TypeError, ValueError):
        minutes = 0
    return f"{minutes}分钟" if minutes > 0 else "时间未知"


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
