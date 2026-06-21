from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.agents.cypher_agent import CypherAgent
from src.agents.sql_agent import SQLAgent
from src.state import AgentState
from src.tools.base import ToolResult


class QueryMySQLRecipesTool:
    name = "query_mysql_recipes"
    description = "Run a controlled read-only recipe query through SQLAgent for ranking, counting, calories, time, category, tag, and audience filters."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    }

    def __init__(self, sql_agent: SQLAgent) -> None:
        self.sql_agent = sql_agent

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query") or state.user_input).strip()
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        if not query:
            return ToolResult(self.name, False, "", error="query is required")

        result_state = self.sql_agent.run(clone_state_for_tool(state, query, top_k))
        status = str(result_state.meta.get("sql_status") or "")
        return ToolResult(
            self.name,
            status not in {"failed", "unsupported"},
            f"query_mysql_recipes: {result_state.agent_output}",
            data={
                "sql_status": status,
                "sql_query": result_state.meta.get("sql_query"),
                "sql_rows": result_state.meta.get("sql_rows", []),
                "sql_query_mode": result_state.meta.get("sql_query_mode"),
            },
            error=result_state.agent_output if status in {"failed", "unsupported"} else "",
        )


class QueryNeo4jRelationshipsTool:
    name = "query_neo4j_relationships"
    description = "Run a controlled read-only graph query through CypherAgent for ingredient, goal, tag, and constraint relationships."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    }

    def __init__(self, cypher_agent: CypherAgent) -> None:
        self.cypher_agent = cypher_agent

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query") or state.user_input).strip()
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        if not query:
            return ToolResult(self.name, False, "", error="query is required")

        result_state = self.cypher_agent.run(clone_state_for_tool(state, query, top_k))
        status = str(result_state.meta.get("cypher_status") or "")
        return ToolResult(
            self.name,
            status not in {"failed", "unsupported"},
            f"query_neo4j_relationships: {result_state.agent_output}",
            data={
                "cypher_status": status,
                "cypher_query": result_state.meta.get("cypher_query"),
                "cypher_rows": result_state.meta.get("cypher_rows", []),
                "cypher_query_mode": result_state.meta.get("cypher_query_mode"),
            },
            error=result_state.agent_output if status in {"failed", "unsupported"} else "",
        )


def clone_state_for_tool(state: AgentState, query: str, top_k: int) -> AgentState:
    return replace(
        state,
        user_input=query,
        top_k=top_k,
        meta=dict(state.meta),
        retrieved_docs=list(state.retrieved_docs),
        fusion_results=list(state.fusion_results),
        vision_result=dict(state.vision_result),
    )


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)
