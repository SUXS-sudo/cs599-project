from __future__ import annotations

from app.agents.fusion_agent import FusionAgent, fuse_source_results
from app.state import AgentState


def test_fusion_single_source_passthrough_top_k() -> None:
    rows = fuse_source_results(
        {
            "rag": [
                {"name": "番茄炒蛋", "score": 0.9},
                {"name": "鸡胸肉沙拉", "score": 0.8},
            ]
        },
        top_k=1,
    )

    assert len(rows) == 1
    assert rows[0]["name"] == "番茄炒蛋"
    assert rows[0]["sources"] == ["rag"]


def test_fusion_merges_multiple_sources_and_adds_bonus() -> None:
    rows = fuse_source_results(
        {
            "rag": [{"name": "鸡胸肉沙拉", "score": 0.8}],
            "sql": [{"name": "鸡胸肉沙拉", "calories": 260}],
            "cypher": [{"name": "豆腐青菜汤", "calories": 120}],
        },
        top_k=3,
    )

    chicken = next(row for row in rows if row["name"] == "鸡胸肉沙拉")
    assert chicken["source_count"] == 2
    assert chicken["sources"] == ["rag", "sql"]
    assert chicken["score"] > next(row for row in rows if row["name"] == "豆腐青菜汤")["score"]


def test_fusion_deduplicates_by_recipe_name() -> None:
    rows = fuse_source_results(
        {
            "rag": [{"name": "番茄炒蛋", "score": 0.7}],
            "sql": [{"name": "番茄炒蛋", "score": 0.6, "category": "家常菜"}],
            "cypher": [{"name": "番茄炒蛋", "score": 0.5}],
        },
        top_k=5,
    )

    assert len(rows) == 1
    assert rows[0]["source_count"] == 3
    assert rows[0]["payload"]["category"] == "家常菜"


def test_fusion_agent_uses_meta_sources_without_external_services() -> None:
    state = AgentState(user_input="综合推荐", session_id="s", top_k=2)
    state.meta["fusion_sources"] = {
        "rag": [{"name": "番茄炒蛋", "score": 0.8}],
        "sql": [{"name": "豆腐青菜汤", "score": 0.9}],
    }

    result = FusionAgent().run(state)

    assert result.meta["fusion_status"] == "ok"
    assert len(result.fusion_results) == 2
    assert "Fusion Agent" in result.agent_output
