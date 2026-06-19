from __future__ import annotations

import json
import os
import re

from app.services.llm_client import LLMClient
from app.services.logger import get_logger
from app.state import AgentState


INTENT_TO_AGENT = {
    "recipe_search": "recipe_agent",
    "recipe_detail": "recipe_agent",
    "ingredient_replace": "recipe_agent",
    "nutrition_query": "nutrition_agent",
    "general_chat": "general_agent",
    "out_of_scope": "general_agent",
    "structured_recipe_query": "sql_agent",
    "relationship_query": "cypher_agent",
    "multi_source_query": "fusion_agent",
    "tool_query": "tool_agent",
}

NUTRITION_KEYWORDS = (
    "热量",
    "营养",
    "减脂",
    "增肌",
    "蛋白",
    "低脂",
    "低糖",
    "低盐",
    "控糖",
    "健康",
    "高蛋白",
    "卡路里",
    "脂肪",
    "碳水",
    "糖尿病",
    "高血压",
    "肾病",
    "孕妇",
    "老人",
    "儿童",
)
REPLACE_KEYWORDS = ("替换", "替代", "没有", "换成", "可以用什么", "代替", "能不能用", "可以不放", "不想放")
DETAIL_KEYWORDS = ("怎么做", "做法", "步骤", "教程", "咋做", "如何做", "流程", "蒸多久", "火候", "提前泡")
RECIPE_KEYWORDS = (
    "菜谱",
    "食谱",
    "食材",
    "推荐",
    "吃什么",
    "晚餐",
    "早餐",
    "午餐",
    "鸡蛋",
    "番茄",
    "西兰花",
    "鸡胸肉",
    "黄瓜",
    "豆腐",
    "清淡",
    "快手",
    "便当",
    "沙拉",
    "汤",
    "粥",
    "空气炸锅",
    "家常",
    "素",
    "饱腹",
    "开火",
    "菜",
    "菜谱",
    "虾仁",
    "南瓜",
    "蒸蛋",
)
GENERAL_CHAT_KEYWORDS = (
    "你好",
    "您好",
    "谢谢",
    "感谢",
    "辛苦了",
    "你是谁",
    "介绍一下",
    "能做什么",
    "机器人",
    "随便聊",
    "记住",
    "哪些事情",
)
OUT_OF_SCOPE_KEYWORDS = ("股票", "电影", "图片", "求职信", "机票", "天气", "计算")
CODE_KEYWORDS = ("Java", "java", "代码", "快速排序", "函数", "算法")
POLITE_CHAT_KEYWORDS = ("谢谢", "感谢", "辛苦了")
RECIPE_REQUEST_KEYWORDS = (
    "推荐",
    "菜谱",
    "食谱",
    "早餐",
    "午餐",
    "晚餐",
    "做什么",
    "吃什么",
    "来点",
    "安排",
    "想吃",
)
HEALTH_CONDITION_KEYWORDS = ("控糖", "糖尿病", "高血压", "肾病", "孕妇", "老人", "儿童")
NUTRITION_QUESTION_KEYWORDS = ("营养建议", "怎么选", "注意", "够吗", "要不要", "高不高")
STRUCTURED_QUERY_KEYWORDS = ("最低", "最高", "以内", "平均", "排序", "前", "5道", "五道", "有哪些", "查询", "几个")
RELATIONSHIP_QUERY_KEYWORDS = ("搭配", "一起", "同时", "关联", "关系", "包含", "相关", "约束")
MULTI_SOURCE_QUERY_KEYWORDS = ("综合", "多源", "融合", "数据库和图谱", "一起查", "全面")
TOOL_QUERY_KEYWORDS = ("偏好", "忌口", "过敏", "不吃", "之前", "上次", "历史", "根据我")
logger = get_logger("agents.router")


class RouterAgent:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        enable_database_agents: bool | None = None,
        enable_v2: bool | None = None,
        enable_fusion: bool | None = None,
    ) -> None:
        self.llm_client = llm_client
        if enable_database_agents is None:
            enable_database_agents = enable_v2
        self.enable_database_agents = (
            env_bool("ENABLE_DATABASE_AGENTS", "ENABLE_V2", default=False)
            if enable_database_agents is None
            else enable_database_agents
        )
        self.enable_fusion = parse_bool(os.getenv("ENABLE_FUSION", "true")) if enable_fusion is None else enable_fusion

    def run(self, state: AgentState) -> AgentState:
        intent = self._route_with_rules(state.user_input)
        if intent != "out_of_scope" or not self.llm_client or not self.llm_client.available:
            state.intent = intent
            state.target_agent = INTENT_TO_AGENT[intent]
            state.meta["router_mode"] = "rule_fast"
            logger.info("路由决策 模式=规则快速 意图=%s Agent=%s", state.intent, state.target_agent)
            return state

        routed = self._route_with_llm(state)
        if routed:
            state.intent = routed["intent"]
            state.target_agent = routed["target_agent"]
            state.meta["router_mode"] = "llm"
            state.meta["router_reason"] = routed.get("reason", "")
            logger.info("路由决策 模式=LLM 意图=%s Agent=%s", state.intent, state.target_agent)
            return state

        state.intent = intent
        state.target_agent = INTENT_TO_AGENT[intent]
        state.meta["router_mode"] = "rule_fallback"
        logger.info("路由决策 模式=规则兜底 意图=%s Agent=%s", state.intent, state.target_agent)
        return state

    def _route_with_llm(self, state: AgentState) -> dict[str, str] | None:
        if not self.llm_client or not self.llm_client.available:
            return None

        intent_lines = [
            "- recipe_search: recipe recommendation based on ingredients, goals, meals, taste.",
            "- recipe_detail: asking how to cook a specific dish.",
            "- nutrition_query: calories, nutrition, fat loss, muscle gain, protein, low sugar, health.",
            "- ingredient_replace: ingredient substitution or missing ingredient.",
            "- general_chat: harmless general chat related enough to answer briefly.",
            "- out_of_scope: unrelated to recipe, ingredients, nutrition, or diet advice.",
            "- tool_query: requires controlled tool use, such as combining stored preferences with recipe search.",
        ]
        agent_lines = [
            "- recipe_agent for recipe_search, recipe_detail, ingredient_replace.",
            "- nutrition_agent for nutrition_query.",
            "- general_agent for general_chat or out_of_scope.",
            "- tool_agent for tool_query.",
        ]
        if self.enable_database_agents:
            intent_lines.extend(
                [
                    "- structured_recipe_query: SQL-style filtering, ranking, counting, average, calories/time/category queries.",
                    "- relationship_query: graph relationship query about ingredients, goals, tags, constraints, and combinations.",
                ]
            )
            agent_lines.extend(
                [
                    "- sql_agent for structured_recipe_query.",
                    "- cypher_agent for relationship_query.",
                ]
            )
            if self.enable_fusion:
                intent_lines.append("- multi_source_query: combine RAG, SQL and graph evidence for broad recipe recommendation or analysis.")
                agent_lines.append("- fusion_agent for multi_source_query.")

        prompt = (
            "You are the Router Agent for a Chinese SmartRecipe multi-agent system.\n"
            "Classify the user question into exactly one intent and choose the target agent.\n\n"
            "Allowed intents:\n"
            + "\n".join(intent_lines)
            + "\n\nAllowed target agents:\n"
            + "\n".join(agent_lines)
            + "\n\nReturn only compact JSON, no markdown:\n"
            '{"intent":"recipe_search","target_agent":"recipe_agent","reason":"..."}\n\n'
            f"Conversation history:\n{state.chat_history}\n\n"
            f"User question: {state.user_input}"
        )
        raw = self.llm_client.generate(prompt, max_tokens=120, timeout=8)
        if not raw:
            logger.debug("路由 LLM 返回为空")
            return None
        return self._parse_router_json(raw, self.enable_database_agents)

    @staticmethod
    def _parse_router_json(raw: str, enable_database_agents: bool = True) -> dict[str, str] | None:
        text = raw.strip()
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        intent = str(data.get("intent", "")).strip()
        target_agent = str(data.get("target_agent", "")).strip()
        if intent not in INTENT_TO_AGENT:
            return None
        if not enable_database_agents and intent in {"structured_recipe_query", "relationship_query"}:
            return None
        expected_agent = INTENT_TO_AGENT[intent]
        if target_agent not in {"recipe_agent", "nutrition_agent", "general_agent", "sql_agent", "cypher_agent", "fusion_agent", "tool_agent"}:
            target_agent = expected_agent
        if target_agent != expected_agent:
            target_agent = expected_agent
        return {
            "intent": intent,
            "target_agent": target_agent,
            "reason": str(data.get("reason", "")).strip(),
        }

    def _route_with_rules(self, message: str) -> str:
        text = message.strip()
        if not text:
            return "out_of_scope"
        if any(keyword in text for keyword in POLITE_CHAT_KEYWORDS):
            return "general_chat"
        if any(keyword in text for keyword in REPLACE_KEYWORDS) and "有没有" not in text:
            return "ingredient_replace"
        if any(keyword in text for keyword in DETAIL_KEYWORDS):
            return "recipe_detail"
        if any(keyword in text for keyword in OUT_OF_SCOPE_KEYWORDS) or any(keyword in text for keyword in CODE_KEYWORDS):
            return "out_of_scope"
        has_nutrition = any(keyword in text for keyword in NUTRITION_KEYWORDS)
        has_recipe = any(keyword in text for keyword in RECIPE_KEYWORDS)
        has_recipe_request = any(keyword in text for keyword in RECIPE_REQUEST_KEYWORDS)
        has_structured = any(keyword in text for keyword in STRUCTURED_QUERY_KEYWORDS)
        has_relationship = any(keyword in text for keyword in RELATIONSHIP_QUERY_KEYWORDS)
        has_multi_source = any(keyword in text for keyword in MULTI_SOURCE_QUERY_KEYWORDS)
        has_tool_query = any(keyword in text for keyword in TOOL_QUERY_KEYWORDS)
        has_health_condition = any(keyword in text for keyword in HEALTH_CONDITION_KEYWORDS)
        has_nutrition_question = any(keyword in text for keyword in NUTRITION_QUESTION_KEYWORDS)
        if has_tool_query and (has_recipe or has_nutrition or has_recipe_request):
            return "tool_query"
        if self.enable_database_agents and self.enable_fusion and has_multi_source and (has_recipe or has_nutrition or has_structured or has_relationship):
            return "multi_source_query"
        if self.enable_database_agents and has_relationship and (has_recipe or has_nutrition):
            return "relationship_query"
        if self.enable_database_agents and has_structured and (has_recipe or has_nutrition):
            return "structured_recipe_query"
        if self.enable_database_agents and has_nutrition and (has_recipe_request or has_recipe):
            return "structured_recipe_query"
        if has_nutrition and (has_health_condition or has_nutrition_question):
            return "nutrition_query"
        if has_nutrition and has_recipe_request and "饮食" not in text and "营养" not in text:
            return "recipe_search"
        if has_nutrition:
            return "nutrition_query"
        if has_recipe:
            return "recipe_search"
        if any(keyword in text for keyword in GENERAL_CHAT_KEYWORDS):
            return "general_chat"
        return "out_of_scope"


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_bool(primary: str, fallback: str | None = None, default: bool = False) -> bool:
    value = os.getenv(primary)
    if value is None and fallback:
        value = os.getenv(fallback)
    if value is None:
        return default
    return parse_bool(value)
