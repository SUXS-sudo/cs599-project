from __future__ import annotations

from app.agents.cypher_agent import CypherAgent
from app.agents.sql_agent import SQLAgent
from app.state import AgentState


class FakeLLM:
    available = True

    def __init__(self, response: str | None) -> None:
        self.response = response
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 800, timeout: int = 45) -> str | None:
        self.calls += 1
        return self.response


class FakeSQLStore:
    def __init__(self) -> None:
        self.queries = []

    def read_query(self, sql, parameters=()):
        self.queries.append((sql, parameters))
        return [{"name": "番茄炒蛋", "category": "家常菜", "cooking_time_minutes": 15, "difficulty": "简单", "calories": 120, "protein_g_per_100g": 8, "fat_g_per_100g": 6}]


class FakeEmptySQLStore:
    def __init__(self) -> None:
        self.queries = []

    def read_query(self, sql, parameters=()):
        self.queries.append((sql, parameters))
        return []


class FakeNeo4jStore:
    def __init__(self) -> None:
        self.queries = []

    def execute_read(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        return [{"name": "鸡胸肉沙拉", "category": "轻食", "calories": 260}]


def make_state(message: str = "推荐热量最低的菜谱") -> AgentState:
    return AgentState(user_input=message, session_id="test", top_k=3)


def test_sql_agent_uses_valid_llm_query() -> None:
    llm = FakeLLM(
        '{"sql":"SELECT r.name, r.category, r.cooking_time_minutes, r.difficulty, r.calories_per_100g AS calories FROM recipes r ORDER BY r.calories_per_100g ASC","params":[],"title":"低热量菜谱"}'
    )
    store = FakeSQLStore()
    state = SQLAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state())

    assert llm.calls == 1
    assert state.meta["sql_query_mode"] == "llm"
    assert "ORDER BY r.calories_per_100g ASC" in store.queries[0][0]
    assert "番茄炒蛋" in state.agent_output


def test_sql_agent_falls_back_when_llm_query_is_unsafe() -> None:
    llm = FakeLLM('{"sql":"DELETE FROM recipes","params":[],"title":"bad"}')
    store = FakeSQLStore()
    state = SQLAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state())

    assert llm.calls == 1
    assert state.meta["sql_query_mode"] == "rule_llm_guard_failed"
    assert store.queries
    assert store.queries[0][0].strip().lower().startswith("select")


def test_sql_agent_falls_back_when_llm_fails(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "sql-llm-empty-test")
    llm = FakeLLM(None)
    store = FakeSQLStore()
    state = SQLAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state())

    assert llm.calls == 1
    assert state.meta["sql_query_mode"] == "rule_llm_empty"
    assert store.queries


def test_sql_agent_does_not_call_llm_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "sql-disabled-test")
    llm = FakeLLM('{"sql":"SELECT name FROM recipes","params":[]}')
    store = FakeSQLStore()
    SQLAgent(store=store, llm_client=llm, enable_llm_query=False).run(make_state())

    assert llm.calls == 0
    assert store.queries


def test_sql_agent_filters_low_fat_high_protein_with_macros() -> None:
    store = FakeSQLStore()
    state = SQLAgent(store=store, llm_client=None, enable_llm_query=False).run(
        make_state("我想吃低脂高蛋白套餐")
    )

    sql, params = store.queries[0]
    assert "r.protein_g_per_100g >= %s" in sql
    assert "r.fat_g_per_100g <= %s" in sql
    assert params == (12, 8)
    assert state.meta["sql_status"] == "ok"


def test_sql_agent_caches_rule_query_rows(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "sql-cache-test")
    llm = FakeLLM(None)
    store = FakeSQLStore()
    agent = SQLAgent(store=store, llm_client=llm, enable_llm_query=False)

    first = agent.run(make_state("推荐热量最低的菜谱"))
    second = agent.run(make_state("推荐热量最低的菜谱"))

    assert len(store.queries) == 1
    assert first.meta["sql_cache_hit"] is False
    assert second.meta["sql_cache_hit"] is True
    assert "番茄炒蛋" in second.agent_output


def test_sql_agent_does_not_cache_empty_rows_and_triggers_llm_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "sql-empty-no-cache-test")
    llm = FakeLLM(None)
    store = FakeEmptySQLStore()
    agent = SQLAgent(store=store, llm_client=llm, enable_llm_query=False)

    first = agent.run(make_state("推荐鸡胸肉菜谱"))
    second = agent.run(make_state("推荐鸡胸肉菜谱"))

    assert len(store.queries) == 2
    assert first.meta["sql_cache_hit"] is False
    assert second.meta["sql_cache_hit"] is False
    assert second.meta["sql_status"] == "empty"
    assert second.meta["recipe_source"] == "llm_fallback_query"
    assert "answer_mode" not in second.meta


def test_cypher_agent_uses_valid_llm_query() -> None:
    llm = FakeLLM(
        '{"cypher":"MATCH (recipe:Recipe)-[:USES]->(:Ingredient {name: $ingredient}) RETURN recipe.name AS name, recipe.category AS category, recipe.calories AS calories","params":{"ingredient":"鸡胸肉"},"title":"鸡胸肉菜谱"}'
    )
    store = FakeNeo4jStore()
    state = CypherAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state("鸡胸肉相关菜谱"))

    assert llm.calls == 1
    assert state.meta["cypher_query_mode"] == "llm"
    assert "MATCH" in store.queries[0][0]
    assert "鸡胸肉沙拉" in state.agent_output


def test_cypher_agent_falls_back_when_llm_query_is_unsafe() -> None:
    llm = FakeLLM('{"cypher":"MATCH (n) DETACH DELETE n","params":{},"title":"bad"}')
    store = FakeNeo4jStore()
    state = CypherAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state("鸡胸肉相关菜谱"))

    assert llm.calls == 1
    assert state.meta["cypher_query_mode"] == "rule_llm_guard_failed"
    assert store.queries


def test_cypher_agent_falls_back_when_llm_fails() -> None:
    llm = FakeLLM(None)
    store = FakeNeo4jStore()
    state = CypherAgent(store=store, llm_client=llm, enable_llm_query=True).run(make_state("鸡胸肉相关菜谱"))

    assert llm.calls == 1
    assert state.meta["cypher_query_mode"] == "rule_llm_empty"
    assert store.queries


def test_cypher_agent_does_not_call_llm_when_disabled() -> None:
    llm = FakeLLM('{"cypher":"MATCH (n) RETURN n","params":{}}')
    store = FakeNeo4jStore()
    CypherAgent(store=store, llm_client=llm, enable_llm_query=False).run(make_state("鸡胸肉相关菜谱"))

    assert llm.calls == 0
    assert store.queries
