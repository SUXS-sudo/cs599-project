from __future__ import annotations

from fastapi.testclient import TestClient

import src.main as main
from src.agents.router_agent import RouterAgent
from src.services.ablation import load_ablation_config, metric_definitions
from src.state import AgentState


def test_ablation_config_reads_feature_flags(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_GRAPHRAG", "false")
    monkeypatch.setenv("ENABLE_FUSION", "false")
    monkeypatch.setenv("ENABLE_ANSWER_GUARD", "false")

    config = load_ablation_config()

    assert config.enable_graph_rag is False
    assert config.enable_fusion is False
    assert config.enable_answer_guard is False
    assert "retrieval_hit@1" in metric_definitions()


def test_debug_ablation_endpoint_exposes_options_and_metrics() -> None:
    client = TestClient(main.app)

    response = client.get("/debug/ablation")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert "RERANK_ENABLED" in data["options"]
    assert "router_intent_accuracy" in data["metrics"]


def test_router_fusion_toggle_routes_multisource_to_recipe_when_disabled() -> None:
    router = RouterAgent(None, enable_database_agents=True, enable_fusion=False)

    state = router.run(
        AgentState(
            user_input="综合推荐几个减脂鸡胸肉晚餐",
            session_id="s",
            top_k=3,
        )
    )

    assert state.intent != "multi_source_query"
    assert state.target_agent != "fusion_agent"
