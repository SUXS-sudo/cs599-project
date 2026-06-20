from __future__ import annotations

from dataclasses import replace

from app.agents.recipe_agent import RecipeAgent, extract_recipe_detail_target
from app.retriever import Recipe
from app.services.graph_rag import GraphRAG, format_graph_context
from app.services.neo4j_store import recipe_to_graph_row
from app.state import AgentState


class FakeNeo4jStore:
    def __init__(self) -> None:
        self.calls = []

    def execute_read(self, query, parameters=None):
        self.calls.append((query, parameters or {}))
        return [
            {
                "name": "番茄炒蛋",
                "category": "家常菜",
                "shared_ingredients": ["番茄", "鸡蛋"],
                "ingredient_peers": ["番茄鸡蛋面"],
                "shared_goals": ["午餐"],
                "goal_peers": ["青椒炒蛋"],
                "category_peers": ["土豆丝"],
            }
        ]


class FakeRetriever:
    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self.recipes = [recipe]
        self.queries = []

    def search(self, query: str, top_k: int):
        self.queries.append(query)
        return [(self.recipe, 0.9)]


class FakeRecipeStore:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def read_query(self, sql, parameters=()):
        self.queries.append((sql, parameters))
        return self.rows


def make_recipe() -> Recipe:
    return Recipe(
        name="番茄炒蛋",
        ingredients=["番茄", "鸡蛋"],
        category="家常菜",
        cooking_time="15分钟",
        difficulty="简单",
        tags=["家常菜", "快手菜"],
        calories=260,
        suitable_for=["午餐", "晚餐"],
        steps="准备食材；炒鸡蛋；炒番茄并合炒。",
    )


def test_graph_rag_enriches_recipes_with_related_context() -> None:
    store = FakeNeo4jStore()
    recipe = make_recipe()

    context = GraphRAG(store).enrich([(recipe, 0.9)])

    assert store.calls
    assert store.calls[0][1]["names"] == ["番茄炒蛋"]
    assert context["番茄炒蛋"][0]["shared_ingredients"] == ["番茄", "鸡蛋"]
    assert "同菜系菜谱" in format_graph_context(context)


def test_recipe_agent_appends_graph_context() -> None:
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe), GraphRAG(FakeNeo4jStore()))
    state = AgentState(user_input="推荐番茄鸡蛋", session_id="s", top_k=1, intent="recipe_search")

    result = agent.run(state)

    assert result.meta["graph_rag_status"] == "ok"
    assert "图谱增强信息" in result.agent_output
    assert "共现食材=番茄、鸡蛋" in result.agent_output


def test_neo4j_recipe_row_uses_same_canonical_fields_as_mysql() -> None:
    row = recipe_to_graph_row(
        {
            "name": "鸡胸肉沙拉",
            "ingredients": ["鸡胸肉", "生菜"],
            "category": "轻食",
            "cooking_time_minutes": 18,
            "difficulty": "简单",
            "calories_per_100g": 135,
            "protein_g_per_100g": 18.5,
            "fat_g_per_100g": 4.2,
            "nutrition_estimated": True,
            "tags": ["低脂", "高蛋白"],
            "suitable_for": ["减脂", "健身"],
            "steps": "拌匀。",
        }
    )

    assert row["cooking_time_minutes"] == 18
    assert row["calories_per_100g"] == 135
    assert row["protein_g_per_100g"] == 18.5
    assert row["fat_g_per_100g"] == 4.2
    assert "cooking_time" not in row
    assert "calories" not in row


def test_recipe_detail_missing_target_uses_llm_fallback_marker() -> None:
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe))
    state = AgentState(user_input="红烧肘子怎么做", session_id="s", top_k=1, intent="recipe_detail")

    result = agent.run(state)

    assert result.retrieved_docs == []
    assert result.meta["recipe_source"] == "llm_fallback"
    assert result.meta["recipe_detail_target"] == "红烧肘子"
    assert result.meta["recipe_mismatch_candidates"] == ["番茄炒蛋"]
    assert "暂未收录" in result.agent_output


def test_recipe_detail_exact_database_hit_uses_direct_fast_path() -> None:
    store = FakeRecipeStore(
        [
            {
                "name": "番茄炒蛋",
                "category": "家常菜",
                "cooking_time_text": "15分钟",
                "cooking_time_minutes": 15,
                "difficulty": "简单",
                "calories": 260,
                "steps": "准备食材；炒鸡蛋；炒番茄并合炒。",
                "ingredients": "番茄、鸡蛋",
                "tags": "家常菜、快手菜",
                "suitable_for": "午餐、晚餐",
            }
        ]
    )
    agent = RecipeAgent(FakeRetriever(make_recipe()), mysql_store=store)
    state = AgentState(user_input="番茄炒蛋怎么做", session_id="s", top_k=1, intent="recipe_detail")

    result = agent.run(state)

    assert store.queries
    assert result.retrieved_docs == []
    assert result.meta["answer_mode"] == "direct"
    assert result.meta["recipe_fast_path"] is True
    assert result.meta["recipe_source"] == "database_fast"
    assert "菜名：番茄炒蛋" in result.agent_output
    assert "未调用大模型改写" in result.agent_output


def test_recipe_detail_alias_exact_match_uses_fast_path() -> None:
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe))
    state = AgentState(user_input="西红柿炒鸡蛋怎么做", session_id="s", top_k=1, intent="recipe_detail")

    result = agent.run(state)

    assert result.meta["answer_mode"] == "direct"
    assert result.meta["recipe_match_level"] == "alias_exact"
    assert result.meta["recipe_matched_name"] == "番茄炒蛋"
    assert "菜名：番茄炒蛋" in result.agent_output


def test_recipe_detail_target_prefers_full_alias_over_cooking_verb_fragment() -> None:
    assert extract_recipe_detail_target("我想吃西红柿炒鸡蛋") == "西红柿炒鸡蛋"


def test_recipe_detail_target_extracts_unknown_concrete_dish() -> None:
    assert extract_recipe_detail_target("红烧大肘子") == "红烧大肘子"
    assert extract_recipe_detail_target("我想吃红烧大肘子") == "红烧大肘子"


def test_unknown_desired_dish_rejects_unrelated_retrieval_candidate() -> None:
    unrelated = replace(make_recipe(), name="高蛋白炖藜麦减脂餐")
    agent = RecipeAgent(FakeRetriever(unrelated))

    result = agent.run(
        AgentState(user_input="我想吃红烧大肘子", session_id="s", top_k=1, intent="recipe_detail")
    )

    assert result.retrieved_docs == []
    assert result.meta["recipe_source"] == "llm_fallback"
    assert result.meta["recipe_detail_target"] == "红烧大肘子"
    assert result.meta["recipe_mismatch_candidates"] == ["高蛋白炖藜麦减脂餐"]


def test_recipe_agent_retrieval_uses_current_turn_not_full_history() -> None:
    recipe = make_recipe()
    retriever = FakeRetriever(recipe)
    agent = RecipeAgent(retriever)
    state = AgentState(
        user_input="推荐番茄鸡蛋",
        session_id="s",
        top_k=1,
        intent="recipe_search",
        chat_history="用户：推荐低脂高蛋白晚餐，家里有鸡胸肉和西兰花",
    )

    result = agent.run(state)

    assert retriever.queries
    assert "推荐番茄鸡蛋" in retriever.queries[0]
    assert "鸡胸肉" not in retriever.queries[0]
    assert result.meta["retrieval_query_scope"] == "current_turn"


def test_recipe_detail_match_cache_hits_on_second_request(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "recipe-match-cache-test")
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe))

    first = agent.run(AgentState(user_input="西红柿炒鸡蛋怎么做", session_id="s", top_k=1, intent="recipe_detail"))
    second = agent.run(AgentState(user_input="西红柿炒鸡蛋怎么做", session_id="s", top_k=1, intent="recipe_detail"))

    assert first.meta["recipe_match_cache_hit"] is False
    assert second.meta["recipe_match_cache_hit"] is True
    assert second.meta["recipe_match_level"] == "alias_exact"


def test_recipe_detail_fuzzy_candidate_match_uses_fast_path() -> None:
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe))
    state = AgentState(user_input="番茄抄蛋怎么做", session_id="s", top_k=1, intent="recipe_detail")

    result = agent.run(state)

    assert result.meta["answer_mode"] == "direct"
    assert result.meta["recipe_match_level"] == "fuzzy_candidate"
    assert result.meta["recipe_matched_name"] == "番茄炒蛋"


def test_recipe_detail_vector_candidate_rejects_low_name_similarity() -> None:
    recipe = make_recipe()
    agent = RecipeAgent(FakeRetriever(recipe))
    state = AgentState(user_input="红烧肘子怎么做", session_id="s", top_k=1, intent="recipe_detail")

    result = agent.run(state)

    assert result.meta["recipe_source"] == "llm_fallback"
    assert result.retrieved_docs == []
    assert "暂未收录" in result.agent_output
