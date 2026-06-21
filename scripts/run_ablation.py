from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


PRESETS: dict[str, dict[str, str]] = {
    "full": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "true",
        "RERANK_ENABLED": "true",
        "ENABLE_GRAPHRAG": "true",
        "ENABLE_FUSION": "true",
        "ENABLE_ANSWER_GUARD": "true",
        "ENABLE_MEMORY_SUMMARY": "true",
        "ENABLE_LLM_QUERY_GENERATION": "false",
        "ENABLE_VISION_LLM": "false",
    },
    "no_rerank": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "true",
        "RERANK_ENABLED": "false",
        "ENABLE_GRAPHRAG": "true",
        "ENABLE_FUSION": "true",
        "ENABLE_ANSWER_GUARD": "true",
        "ENABLE_MEMORY_SUMMARY": "true",
    },
    "no_graphrag": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "true",
        "RERANK_ENABLED": "true",
        "ENABLE_GRAPHRAG": "false",
        "ENABLE_FUSION": "true",
        "ENABLE_ANSWER_GUARD": "true",
        "ENABLE_MEMORY_SUMMARY": "true",
    },
    "no_fusion": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "true",
        "RERANK_ENABLED": "true",
        "ENABLE_GRAPHRAG": "true",
        "ENABLE_FUSION": "false",
        "ENABLE_ANSWER_GUARD": "true",
        "ENABLE_MEMORY_SUMMARY": "true",
    },
    "no_answer_guard": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "true",
        "RERANK_ENABLED": "true",
        "ENABLE_GRAPHRAG": "true",
        "ENABLE_FUSION": "true",
        "ENABLE_ANSWER_GUARD": "false",
        "ENABLE_MEMORY_SUMMARY": "true",
    },
    "rag_only": {
        "RAG_BACKEND": "bm25",
        "ENABLE_DATABASE_AGENTS": "false",
        "RERANK_ENABLED": "true",
        "ENABLE_GRAPHRAG": "false",
        "ENABLE_FUSION": "false",
        "ENABLE_ANSWER_GUARD": "true",
        "ENABLE_MEMORY_SUMMARY": "false",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SmartRecipe ablation experiments.")
    parser.add_argument(
        "--presets",
        default="full,no_rerank,no_graphrag,no_fusion,no_answer_guard,rag_only",
        help="Comma-separated preset names.",
    )
    parser.add_argument("--output", default=str(ROOT_DIR / "data" / "evals" / "ablation_results.json"))
    parser.add_argument("--markdown", default=str(ROOT_DIR / "docs" / "ABLATION_RESULTS.md"))
    parser.add_argument("--max-k", type=int, default=5)
    args = parser.parse_args()

    suppress_smart_recipe_info_logs()

    selected = [name.strip() for name in args.presets.split(",") if name.strip()]
    unknown = [name for name in selected if name not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets: {', '.join(unknown)}")

    results = []
    for name in selected:
        with temporary_env(PRESETS[name]):
            started = time.perf_counter()
            retrieval = evaluate_retrieval(max_k=args.max_k)
            router = evaluate_router(enable_database_agents=env_bool("ENABLE_DATABASE_AGENTS"), enable_fusion=env_bool("ENABLE_FUSION"))
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "preset": name,
                    "config": dict(PRESETS[name]),
                    "metrics": {
                        **retrieval,
                        **router,
                        "eval_elapsed_ms": round(elapsed_ms, 2),
                    },
                }
            )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "metric_notes": {
            "retrieval_hit@k": "期望菜谱是否出现在 Top-K 检索结果中。",
            "retrieval_mrr@5": "正确菜谱越靠前，分数越高。",
            "router_intent_accuracy": "意图分类正确率。",
            "router_agent_accuracy": "目标 Agent 选择正确率。",
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(format_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def evaluate_retrieval(max_k: int) -> dict[str, Any]:
    from src.agents.rerank_agent import RerankAgent
    from src.retriever import RecipeRetriever
    from src.state import AgentState

    cases = read_jsonl(ROOT_DIR / "data" / "evals" / "rag_retrieval.jsonl")
    retriever = RecipeRetriever(ROOT_DIR / "data" / "recipes.json")
    reranker = RerankAgent() if env_bool("RERANK_ENABLED") else None
    cutoffs = [k for k in (1, 3, 5) if k <= max_k]
    if max_k not in cutoffs:
        cutoffs.append(max_k)
    totals = {k: 0 for k in cutoffs}
    reciprocal_rank_sum = 0.0
    for case in cases:
        hits = retriever.search(case["query"], top_k=max_k)
        if reranker is not None:
            state = AgentState(
                user_input=case["query"],
                session_id="ablation-retrieval",
                top_k=max_k,
                retrieved_docs=hits,
            )
            hits = reranker.run(state).retrieved_docs
        ranked_names = [recipe.name for recipe, _ in hits]
        expected = set(case["expected"])
        rank = first_hit_rank(ranked_names, expected)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank
        for k in cutoffs:
            totals[k] += int(any(name in expected for name in ranked_names[:k]))
    total = len(cases)
    metrics = {
        "retrieval_cases": total,
        "retrieval_active_backend": retriever.backend,
        "retrieval_embedding_backend": retriever.embedding_backend,
        "retrieval_rerank_enabled": reranker is not None,
        "retrieval_mrr@5": round(reciprocal_rank_sum / total, 4) if total else 0.0,
    }
    for k in cutoffs:
        metrics[f"retrieval_hit@{k}"] = round(totals[k] / total, 4) if total else 0.0
    return metrics


def suppress_smart_recipe_info_logs() -> None:
    from src.services.logger import configure_logging

    configure_logging()
    logger = logging.getLogger("smart_recipe")
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)


def evaluate_router(enable_database_agents: bool, enable_fusion: bool) -> dict[str, Any]:
    from src.agents.router_agent import RouterAgent
    from src.state import AgentState

    cases = read_jsonl(ROOT_DIR / "data" / "evals" / "router_intents.jsonl")
    router = RouterAgent(None, enable_database_agents=enable_database_agents, enable_fusion=enable_fusion)
    correct_intent = 0
    correct_agent = 0
    for case in cases:
        state = AgentState(user_input=case["message"], session_id="ablation-router", top_k=3)
        result = router.run(state)
        correct_intent += int(result.intent == case["intent"])
        expected_agent = case.get("target_agent")
        correct_agent += int(expected_agent is None or result.target_agent == expected_agent)
    total = len(cases)
    return {
        "router_cases": total,
        "router_intent_accuracy": round(correct_intent / total, 4) if total else 0.0,
        "router_agent_accuracy": round(correct_agent / total, 4) if total else 0.0,
    }


def first_hit_rank(ranked_names: list[str], expected_names: set[str]) -> int | None:
    for index, name in enumerate(ranked_names, start=1):
        if name in expected_names:
            return index
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@contextmanager
def temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def env_bool(key: str) -> bool:
    return os.getenv(key, "false").strip().lower() in {"1", "true", "yes", "on"}


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SmartRecipe Ablation Results",
        "",
        f"Generated at: {payload['generated_at']}",
        "",
        "| Preset | hit@1 | hit@3 | hit@5 | MRR@5 | Router Intent | Router Agent | Backend |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        metrics = row["metrics"]
        lines.append(
            "| {preset} | {h1:.2%} | {h3:.2%} | {h5:.2%} | {mrr:.4f} | {ri:.2%} | {ra:.2%} | {backend} |".format(
                preset=row["preset"],
                h1=metrics.get("retrieval_hit@1", 0.0),
                h3=metrics.get("retrieval_hit@3", 0.0),
                h5=metrics.get("retrieval_hit@5", 0.0),
                mrr=metrics.get("retrieval_mrr@5", 0.0),
                ri=metrics.get("router_intent_accuracy", 0.0),
                ra=metrics.get("router_agent_accuracy", 0.0),
                backend=metrics.get("retrieval_active_backend", ""),
            )
        )
    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- `full`: 当前推荐方案。",
            "- `no_rerank`: 关闭重排，用来衡量 Rerank 对排序的收益。",
            "- `no_graphrag`: 关闭 GraphRAG，用来衡量图谱上下文的收益。",
            "- `no_fusion`: 综合问题退回普通 RAG，用来衡量多源融合的收益。",
            "- `no_answer_guard`: 关闭最终证据检查，用来观察幻觉防护影响。",
            "- `rag_only`: 关闭数据库增强 Agent，只保留 RAG 主链路，用来和完整版对比。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
