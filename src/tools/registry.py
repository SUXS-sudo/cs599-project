from __future__ import annotations

from src.agents.cypher_agent import CypherAgent
from src.agents.sql_agent import SQLAgent
from src.retriever import RecipeRetriever
from src.services.memory import MemoryStore
from src.tools.base import Tool
from src.tools.database_tools import QueryMySQLRecipesTool, QueryNeo4jRelationshipsTool
from src.tools.document_tools import SearchDocumentChunksTool
from src.tools.planning_tools import BuildShoppingListTool, FilterRecipesByConstraintsTool, PlanWeeklyMenuTool
from src.tools.recipe_tools import GetUserPreferencesTool, SearchRecipesTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def descriptions(self) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]


def build_default_tool_registry(
    retriever: RecipeRetriever,
    memory_store: MemoryStore,
    sql_agent: SQLAgent | None = None,
    cypher_agent: CypherAgent | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchRecipesTool(retriever))
    registry.register(GetUserPreferencesTool(memory_store))
    registry.register(SearchDocumentChunksTool())
    registry.register(FilterRecipesByConstraintsTool(retriever))
    registry.register(BuildShoppingListTool(retriever))
    registry.register(PlanWeeklyMenuTool(retriever))
    if sql_agent is not None:
        registry.register(QueryMySQLRecipesTool(sql_agent))
    if cypher_agent is not None:
        registry.register(QueryNeo4jRelationshipsTool(cypher_agent))
    return registry
