from __future__ import annotations

from app.agents.rerank_agent import DEFAULT_CROSS_ENCODER_MODEL, RerankAgent
from app.retriever import Recipe


def make_recipe(name: str) -> Recipe:
    return Recipe(
        name=name,
        ingredients=[name],
        category="test",
        cooking_time="10分钟",
        difficulty="简单",
        tags=[],
        calories=100,
        suitable_for=[],
        steps=f"{name} 做法",
    )


def test_rerank_defaults_do_not_reuse_embedding_model(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "models/bge-small-zh-v1.5")
    monkeypatch.delenv("RERANK_CROSS_ENCODER_MODEL", raising=False)
    monkeypatch.setattr(RerankAgent, "_load_models", lambda self: None)

    reranker = RerankAgent()

    assert reranker.cross_encoder_model_name == DEFAULT_CROSS_ENCODER_MODEL


def test_cross_encoder_score_can_change_rerank_order(monkeypatch) -> None:
    monkeypatch.setattr(RerankAgent, "_load_models", lambda self: None)
    reranker = RerankAgent()
    reranker._cross_encoder = object()
    reranker._load_error = ""
    monkeypatch.setattr(reranker, "_cross_encoder_scores", lambda query, docs: [0.1, 0.95])
    docs = [(make_recipe("A"), 0.5), (make_recipe("B"), 0.5)]

    ranked = reranker._rerank_with_cross_encoder("query", docs)

    assert ranked is not None
    assert ranked[0][0].name == "B"
