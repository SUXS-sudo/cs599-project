from __future__ import annotations

from src.agents.router_agent import RouterAgent
from src.agents.tool_agent import ToolAgent
from src.retriever import Recipe
from src.services.llm_client import LLMClient
from src.services.memory import MemoryStore
from src.state import AgentState
from src.tools.base import ToolResult
from src.tools.database_tools import QueryMySQLRecipesTool, QueryNeo4jRelationshipsTool
from src.tools.document_tools import SearchDocumentChunksTool
from src.tools.planning_tools import BuildShoppingListTool, FilterRecipesByConstraintsTool, PlanWeeklyMenuTool
from src.tools.recipe_tools import GetUserPreferencesTool, SearchRecipesTool
from src.tools.registry import ToolRegistry


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []
        self.recipe = Recipe(
            name="番茄炒蛋",
            ingredients=["番茄", "鸡蛋"],
            category="家常菜",
            cooking_time="10分钟",
            difficulty="简单",
            tags=["快手"],
            calories=180,
            suitable_for=["晚餐"],
            steps="炒鸡蛋，再炒番茄，合炒调味。",
        )
        self.second = Recipe(
            name="清炒西兰花",
            ingredients=["西兰花", "蒜"],
            category="素菜",
            cooking_time="8分钟",
            difficulty="简单",
            tags=["低脂", "清淡"],
            calories=120,
            suitable_for=["晚餐"],
            steps="西兰花焯水，蒜末清炒。",
        )
        self.third = Recipe(
            name="花生拌菠菜",
            ingredients=["花生", "菠菜"],
            category="凉菜",
            cooking_time="12分钟",
            difficulty="简单",
            tags=["清淡"],
            calories=260,
            suitable_for=["晚餐"],
            steps="菠菜焯水，加入花生调味。",
        )

    def search(self, query: str, top_k: int = 3):
        self.calls.append({"query": query, "top_k": top_k})
        return [(self.recipe, 1.2), (self.second, 1.0), (self.third, 0.8)][:top_k]


class JsonPlanLLM:
    available = True
    model = "planner-test-model"

    def generate(self, *args, **kwargs):
        return '{"calls":[{"tool":"get_user_preferences","args":{}},{"tool":"search_recipes","args":{"query":"低脂晚餐","top_k":9}}]}'


class BadPlanLLM:
    available = True

    def generate(self, *args, **kwargs):
        return "not json"


class BrokenTool:
    name = "broken_tool"
    description = "Always fails."
    parameters = {"type": "object", "properties": {}}

    def run(self, args, state):
        raise RuntimeError("boom")


class FakeSQLAgent:
    def __init__(self) -> None:
        self.states = []

    def run(self, state):
        self.states.append(state)
        state.agent_output = "SQL rows: 番茄炒蛋"
        state.meta["sql_status"] = "ok"
        state.meta["sql_query"] = "SELECT name FROM recipes LIMIT 3"
        state.meta["sql_rows"] = [{"name": "番茄炒蛋"}]
        state.meta["sql_query_mode"] = "rule_disabled"
        return state


class FakeCypherAgent:
    def __init__(self) -> None:
        self.states = []

    def run(self, state):
        self.states.append(state)
        state.agent_output = "Cypher rows: 鸡蛋 related 番茄"
        state.meta["cypher_status"] = "ok"
        state.meta["cypher_query"] = "MATCH (n) RETURN n LIMIT 3"
        state.meta["cypher_rows"] = [{"name": "番茄"}]
        state.meta["cypher_query_mode"] = "rule_disabled"
        return state


def test_tool_registry_registers_and_lists_tools() -> None:
    registry = ToolRegistry()
    tool = BrokenTool()

    registry.register(tool)

    assert registry.get("broken_tool") is tool
    assert registry.get("missing") is None
    assert registry.names() == ["broken_tool"]
    assert registry.descriptions()[0]["name"] == "broken_tool"


def test_search_recipes_tool_writes_retrieval_data() -> None:
    retriever = FakeRetriever()
    tool = SearchRecipesTool(retriever)

    result = tool.run({"query": "番茄", "top_k": 20}, AgentState(user_input="番茄", session_id="s", top_k=3))

    assert result.ok is True
    assert retriever.calls == [{"query": "番茄", "top_k": 5}]
    assert "番茄炒蛋" in result.content
    assert result.data["retrieved_docs"][0][0].name == "番茄炒蛋"


def test_get_user_preferences_tool_reads_current_session() -> None:
    store = MemoryStore()
    store.update_preferences("s", preferences=["清淡"], allergies=["花生"], dislikes=["香菜"])
    tool = GetUserPreferencesTool(store)

    result = tool.run({}, AgentState(user_input="推荐", session_id="s", top_k=3))

    assert result.ok is True
    assert result.data["preferences"]["preferences"] == ["清淡"]
    assert "花生" in result.content


def test_mysql_tool_delegates_to_sql_agent_without_mutating_parent_state() -> None:
    sql_agent = FakeSQLAgent()
    state = AgentState(user_input="原始问题", session_id="s", top_k=3, meta={"keep": "yes"})

    result = QueryMySQLRecipesTool(sql_agent).run({"query": "低脂晚餐", "top_k": 9}, state)

    assert result.ok is True
    assert sql_agent.states[0].user_input == "低脂晚餐"
    assert sql_agent.states[0].top_k == 5
    assert state.user_input == "原始问题"
    assert result.data["sql_rows"] == [{"name": "番茄炒蛋"}]
    assert "SQL rows" in result.content


def test_neo4j_tool_delegates_to_cypher_agent_without_mutating_parent_state() -> None:
    cypher_agent = FakeCypherAgent()
    state = AgentState(user_input="原始问题", session_id="s", top_k=3)

    result = QueryNeo4jRelationshipsTool(cypher_agent).run({"query": "鸡蛋和番茄搭配", "top_k": 2}, state)

    assert result.ok is True
    assert cypher_agent.states[0].user_input == "鸡蛋和番茄搭配"
    assert cypher_agent.states[0].top_k == 2
    assert result.data["cypher_rows"] == [{"name": "番茄"}]
    assert "Cypher rows" in result.content


def test_search_document_chunks_tool_uses_injected_searcher() -> None:
    def fake_searcher(query, top_k, preview_chars):
        return [
            {
                "rank": 1,
                "chunk_id": "cookbook-0001",
                "source": "cookbook.pdf",
                "source_type": "pdf",
                "score": 0.9,
                "metadata": {"dish_name": "番茄炒蛋"},
                "preview": "番茄炒蛋 原料 鸡蛋 番茄",
            }
        ][:top_k]

    result = SearchDocumentChunksTool(searcher=fake_searcher).run(
        {"query": "番茄炒蛋怎么做", "top_k": 1},
        AgentState(user_input="查菜谱书", session_id="s", top_k=3),
    )

    assert result.ok is True
    assert result.data["document_chunks"][0]["chunk_id"] == "cookbook-0001"
    assert "番茄炒蛋" in result.content


def test_filter_recipes_by_constraints_uses_preferences_and_avoids_allergies() -> None:
    retriever = FakeRetriever()
    state = AgentState(
        user_input="筛选低脂晚餐",
        session_id="s",
        top_k=3,
        retrieved_docs=retriever.search("晚餐", 3),
        meta={"user_preferences": {"preferences": ["低脂"], "allergies": ["花生"], "dislikes": []}},
    )

    result = FilterRecipesByConstraintsTool(retriever).run({"max_calories": 200}, state)

    names = [row["name"] for row in result.data["filtered_recipes"]]
    assert "清炒西兰花" in names
    assert "花生拌菠菜" not in names
    assert "filter_recipes_by_constraints" in result.content


def test_build_shopping_list_merges_recipe_ingredients() -> None:
    retriever = FakeRetriever()
    state = AgentState(user_input="生成购物清单", session_id="s", top_k=2, retrieved_docs=retriever.search("晚餐", 2))

    result = BuildShoppingListTool(retriever).run({}, state)

    ingredients = {item["ingredient"] for item in result.data["shopping_list"]}
    assert {"番茄", "鸡蛋", "西兰花"}.issubset(ingredients)
    assert "recipes=番茄炒蛋, 清炒西兰花" in result.content


def test_plan_weekly_menu_builds_bounded_menu() -> None:
    retriever = FakeRetriever()
    state = AgentState(user_input="规划一周晚餐", session_id="s", top_k=3)

    result = PlanWeeklyMenuTool(retriever).run({"days": 9, "meals_per_day": 1}, state)

    assert result.ok is True
    assert len(result.data["weekly_menu"]) == 7
    assert result.data["weekly_menu"][0]["meals"][0]["meal"] == "dinner"


def test_tool_agent_executes_llm_plan_and_sanitizes_args(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LLM_TOOL_PLANNER", "true")
    retriever = FakeRetriever()
    store = MemoryStore()
    store.update_preferences("s", dislikes=["香菜"])
    registry = ToolRegistry()
    registry.register(GetUserPreferencesTool(store))
    registry.register(SearchRecipesTool(retriever))

    state = ToolAgent(registry, JsonPlanLLM()).run(AgentState(user_input="根据我的忌口推荐晚餐", session_id="s", top_k=3))

    assert state.meta["tool_planner_mode"] == "llm"
    assert state.meta["tool_calls"][1]["args"]["top_k"] == 5
    assert state.meta["tool_status"] == "ok"
    assert state.retrieved_docs[0][0].name == "番茄炒蛋"
    assert "get_user_preferences" in state.agent_output
    assert "search_recipes" in state.agent_output


def test_tool_agent_falls_back_to_rule_plan_when_llm_plan_is_invalid() -> None:
    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register(SearchRecipesTool(retriever))
    registry.register(GetUserPreferencesTool(MemoryStore()))

    state = ToolAgent(registry, BadPlanLLM()).run(AgentState(user_input="根据我的偏好推荐晚餐", session_id="s", top_k=2))

    assert state.meta["tool_planner_mode"] == "rule"
    assert [call["tool"] for call in state.meta["tool_calls"]] == ["get_user_preferences", "search_recipes"]
    assert retriever.calls[0]["top_k"] == 2


def test_tool_agent_rule_plan_can_include_database_tools() -> None:
    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register(SearchRecipesTool(retriever))
    registry.register(GetUserPreferencesTool(MemoryStore()))
    registry.register(QueryMySQLRecipesTool(FakeSQLAgent()))
    registry.register(QueryNeo4jRelationshipsTool(FakeCypherAgent()))

    state = ToolAgent(registry).run(
        AgentState(user_input="根据我的忌口查询热量最低并适合搭配的晚餐", session_id="s", top_k=3)
    )

    assert [call["tool"] for call in state.meta["tool_calls"]] == [
        "get_user_preferences",
        "query_mysql_recipes",
        "query_neo4j_relationships",
        "search_recipes",
    ]
    assert "query_mysql_recipes" in state.agent_output
    assert "query_neo4j_relationships" in state.agent_output


def test_tool_agent_rule_plan_can_include_document_and_planning_tools() -> None:
    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register(SearchRecipesTool(retriever))
    registry.register(SearchDocumentChunksTool(searcher=lambda query, top_k, preview_chars: []))
    registry.register(FilterRecipesByConstraintsTool(retriever))
    registry.register(BuildShoppingListTool(retriever))
    registry.register(PlanWeeklyMenuTool(retriever))

    state = ToolAgent(registry).run(
        AgentState(user_input="查PDF文档并规划一周低脂晚餐购物清单", session_id="s", top_k=2)
    )

    tools = [call["tool"] for call in state.meta["tool_calls"]]
    assert "search_document_chunks" in tools
    assert "filter_recipes_by_constraints" in tools
    assert "build_shopping_list" in tools
    assert "plan_weekly_menu" in tools


def test_tool_agent_records_tool_exceptions_without_crashing() -> None:
    registry = ToolRegistry()
    registry.register(BrokenTool())

    state = ToolAgent(registry).run(AgentState(user_input="test", session_id="s", top_k=1))

    assert state.meta["tool_status"] == "failed"
    assert "search_recipes" not in state.meta["tool_calls"]


def test_router_routes_preference_recipe_requests_to_tool_agent() -> None:
    state = RouterAgent(None).run(
        AgentState(user_input="根据我的忌口推荐一道晚餐", session_id="s", top_k=3)
    )

    assert state.intent == "tool_query"
    assert state.target_agent == "tool_agent"
    assert state.meta["router_mode"] == "rule_fast"


def test_prefixed_llm_client_reads_vision_model(monkeypatch) -> None:
    monkeypatch.setenv("SMART_RECIPE_PROVIDER", "openai")
    monkeypatch.setenv("BASE_URL", "https://main.example/v1")
    monkeypatch.setenv("API_KEY", "main-key")
    monkeypatch.setenv("MODEL", "deepseek-chat")
    monkeypatch.setenv("VISION_PROVIDER", "anthropic")
    monkeypatch.setenv("VISION_BASE_URL", "https://vision.example/anthropic")
    monkeypatch.setenv("VISION_API_KEY", "vision-key")
    monkeypatch.setenv("VISION_MODEL", "mimo-v2.5")

    client = LLMClient(env_prefix="VISION")

    assert client.provider == "anthropic"
    assert client.base_url == "https://vision.example/anthropic"
    assert client.api_key == "vision-key"
    assert client.model == "mimo-v2.5"
    assert client.vision_model == "mimo-v2.5"


def test_tool_agent_records_planner_empty_failure(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LLM_TOOL_PLANNER", "true")

    class EmptyPlannerLLM:
        available = True
        model = "deepseek-chat"
        last_failure_kind = "empty_response"
        last_failure_detail = "finish_reason=stop, message_fields=none"

        def generate(self, *args, **kwargs):
            return None

    retriever = FakeRetriever()
    registry = ToolRegistry()
    registry.register(SearchRecipesTool(retriever))

    state = ToolAgent(registry, EmptyPlannerLLM()).run(AgentState(user_input="推荐晚餐", session_id="s", top_k=2))

    assert state.meta["tool_planner_mode"] == "rule"
    assert state.meta["tool_planner_failure"]["kind"] == "empty_response"
    assert state.meta["tool_planner_failure"]["model"] == "deepseek-chat"
