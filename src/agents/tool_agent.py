from __future__ import annotations

import json
import os
import re
from typing import Any

from src.services.llm_client import LLMClient
from src.services.logger import get_logger
from src.state import AgentState
from src.tools.registry import ToolRegistry


logger = get_logger("agents.tool")


class ToolAgent:
    def __init__(self, registry: ToolRegistry, llm_client: LLMClient | None = None) -> None:
        self.registry = registry
        self.llm_client = llm_client
        self.enable_llm_planner = parse_bool(os.getenv("ENABLE_LLM_TOOL_PLANNER", "true"))

    def run(self, state: AgentState) -> AgentState:
        calls = self._build_llm_plan(state) or self._build_rule_plan(state)
        calls = sanitize_calls(calls, allowed_tools=set(self.registry.names()))
        state.meta["tool_calls"] = calls

        results = []
        retrieved_docs = []
        for call in calls:
            tool = self.registry.get(call["tool"])
            if tool is None:
                results.append(
                    {
                        "tool": call["tool"],
                        "ok": False,
                        "content": "",
                        "error": "unknown tool",
                    }
                )
                continue
            try:
                result = tool.run(call.get("args", {}), state)
            except Exception as exc:
                logger.exception("工具调用失败 名称=%s", call["tool"])
                results.append(
                    {
                        "tool": call["tool"],
                        "ok": False,
                        "content": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            if "retrieved_docs" in result.data:
                retrieved_docs = result.data["retrieved_docs"]
            results.append(
                {
                    "tool": result.name,
                    "ok": result.ok,
                    "content": result.content,
                    "data": serialize_tool_data(result.data),
                    "error": result.error,
                }
            )

        if retrieved_docs:
            state.retrieved_docs = retrieved_docs
        state.agent_output = format_tool_observations(results)
        state.meta["tool_results"] = results
        state.meta["tool_status"] = "ok" if any(result["ok"] for result in results) else "failed"
        return state

    def _build_llm_plan(self, state: AgentState) -> list[dict[str, Any]] | None:
        if not self.enable_llm_planner:
            state.meta["tool_planner_mode"] = "rule_disabled"
            return None
        if not self.llm_client or not self.llm_client.available:
            return None

        prompt = (
            "You are SmartRecipe Tool Planner. Choose at most 8 tools.\n"
            "Return only compact JSON, no markdown, in this shape:\n"
            '{"calls":[{"tool":"search_recipes","args":{"query":"...","top_k":3}}]}\n\n'
            "Available tools:\n"
            + json.dumps(self.registry.descriptions(), ensure_ascii=False)
            + "\n\nUser question: "
            + state.user_input
        )
        raw = self.llm_client.generate(
            prompt,
            max_tokens=env_int("TOOL_PLANNER_MAX_TOKENS", 1200),
            timeout=env_int("TOOL_PLANNER_TIMEOUT", 30),
        )
        if not raw:
            state.meta["tool_planner_mode"] = "rule_llm_empty"
            state.meta["tool_planner_failure"] = {
                "kind": getattr(self.llm_client, "last_failure_kind", ""),
                "detail": getattr(self.llm_client, "last_failure_detail", ""),
                "model": getattr(self.llm_client, "model", ""),
            }
            return None
        plan = parse_tool_plan(raw)
        if not plan:
            state.meta["tool_planner_mode"] = "rule_llm_parse_failed"
            return None
        state.meta["tool_planner_mode"] = "llm"
        return plan

    @staticmethod
    def _build_rule_plan(state: AgentState) -> list[dict[str, Any]]:
        state.meta["tool_planner_mode"] = "rule"
        calls: list[dict[str, Any]] = []
        if should_read_preferences(state.user_input):
            calls.append({"tool": "get_user_preferences", "args": {}})
        if should_search_documents(state.user_input):
            calls.append(
                {
                    "tool": "search_document_chunks",
                    "args": {
                        "query": state.user_input,
                        "top_k": state.top_k,
                    },
                }
            )
        if should_query_mysql(state.user_input):
            calls.append(
                {
                    "tool": "query_mysql_recipes",
                    "args": {
                        "query": build_search_query(state),
                        "top_k": state.top_k,
                    },
                }
            )
        if should_query_neo4j(state.user_input):
            calls.append(
                {
                    "tool": "query_neo4j_relationships",
                    "args": {
                        "query": build_search_query(state),
                        "top_k": state.top_k,
                    },
                }
            )
        recipe_search_call = {
            "tool": "search_recipes",
            "args": {
                "query": build_search_query(state),
                "top_k": state.top_k,
            },
        }
        calls.append(recipe_search_call)
        if should_filter_recipes(state.user_input):
            calls.append({"tool": "filter_recipes_by_constraints", "args": {"query": build_search_query(state), "top_k": state.top_k}})
        if should_build_shopping_list(state.user_input):
            calls.append({"tool": "build_shopping_list", "args": {"query": build_search_query(state), "top_k": state.top_k}})
        if should_plan_weekly_menu(state.user_input):
            calls.append({"tool": "plan_weekly_menu", "args": {"query": build_search_query(state), "days": 7, "meals_per_day": infer_meals_per_day(state.user_input)}})
        return calls


def should_read_preferences(message: str) -> bool:
    markers = ("偏好", "忌口", "过敏", "不吃", "之前", "上次", "历史", "根据我")
    return any(marker in message for marker in markers)


def should_query_mysql(message: str) -> bool:
    markers = ("最低", "最高", "以内", "平均", "排序", "查询", "几道", "多少", "热量", "低脂", "低糖", "低盐")
    return any(marker in message for marker in markers)


def should_search_documents(message: str) -> bool:
    markers = ("文档", "PDF", "pdf", "菜谱书", "书里", "资料", "原文", "章节")
    return any(marker in message for marker in markers)


def should_query_neo4j(message: str) -> bool:
    markers = ("搭配", "一起", "关联", "关系", "图谱", "同时包含", "适合")
    return any(marker in message for marker in markers)


def should_filter_recipes(message: str) -> bool:
    markers = ("过滤", "筛选", "忌口", "过敏", "不吃", "低脂", "低糖", "低盐", "以内")
    return any(marker in message for marker in markers)


def should_build_shopping_list(message: str) -> bool:
    markers = ("购物清单", "采购", "买什么", "食材清单", "备菜清单")
    return any(marker in message for marker in markers)


def should_plan_weekly_menu(message: str) -> bool:
    markers = ("一周", "7天", "七天", "周菜单", "菜单规划", "每天", "三餐")
    return any(marker in message for marker in markers)


def infer_meals_per_day(message: str) -> int:
    if "三餐" in message or "早午晚" in message:
        return 3
    if "两餐" in message or "午晚" in message:
        return 2
    return 1


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def build_search_query(state: AgentState) -> str:
    preferences = state.meta.get("user_preferences")
    if not isinstance(preferences, dict):
        return state.user_input
    parts = [state.user_input]
    for key in ("preferences", "allergies", "dislikes"):
        values = preferences.get(key, [])
        if isinstance(values, list) and values:
            parts.append(f"{key}: " + " ".join(str(value) for value in values))
    return "\n".join(parts)


def parse_tool_plan(raw: str) -> list[dict[str, Any]] | None:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    calls = data.get("calls") if isinstance(data, dict) else None
    return calls if isinstance(calls, list) else None


def sanitize_calls(calls: list[dict[str, Any]], allowed_tools: set[str]) -> list[dict[str, Any]]:
    sanitized = []
    for call in calls[:8]:
        if not isinstance(call, dict):
            continue
        name = str(call.get("tool") or call.get("name") or "").strip()
        if name not in allowed_tools:
            continue
        args = call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        if name in {
            "search_recipes",
            "query_mysql_recipes",
            "query_neo4j_relationships",
            "search_document_chunks",
            "filter_recipes_by_constraints",
            "build_shopping_list",
        }:
            top_k = args.get("top_k", 3)
            try:
                top_k = int(top_k)
            except (TypeError, ValueError):
                top_k = 3
            args["top_k"] = min(max(top_k, 1), 5)
            args["query"] = str(args.get("query") or "").strip()
        if name == "plan_weekly_menu":
            args["query"] = str(args.get("query") or "").strip()
            for key, default, minimum, maximum in (("days", 7, 1, 7), ("meals_per_day", 1, 1, 3)):
                try:
                    value = int(args.get(key, default))
                except (TypeError, ValueError):
                    value = default
                args[key] = min(max(value, minimum), maximum)
        sanitized.append({"tool": name, "args": args})
    return sanitized


def serialize_tool_data(data: dict[str, Any]) -> dict[str, Any]:
    serialized = {}
    for key, value in data.items():
        if key == "retrieved_docs":
            continue
        serialized[key] = value
    return serialized


def format_tool_observations(results: list[dict[str, Any]]) -> str:
    if not results:
        return "ToolAgent 未执行任何允许的工具。"
    lines = ["ToolAgent 工具观察："]
    for result in results:
        if result["ok"]:
            lines.append(f"- {result['tool']}: {result['content']}")
        else:
            lines.append(f"- {result['tool']}: 失败（{result.get('error') or '未知错误'}）")
    return "\n".join(lines)
