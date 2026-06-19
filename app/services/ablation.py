from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AblationConfig:
    rag_backend: str
    enable_database_agents: bool
    enable_rerank: bool
    enable_graph_rag: bool
    enable_fusion: bool
    enable_answer_guard: bool
    enable_memory_summary: bool
    enable_vision_llm: bool
    enable_llm_query_generation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ablation_config() -> AblationConfig:
    return AblationConfig(
        rag_backend=normalize_rag_backend(os.getenv("RAG_BACKEND", "bm25")),
        enable_database_agents=env_bool("ENABLE_DATABASE_AGENTS", "ENABLE_V2", default=False),
        enable_rerank=parse_bool(os.getenv("RERANK_ENABLED", "true")),
        enable_graph_rag=parse_bool(os.getenv("ENABLE_GRAPHRAG", "true")),
        enable_fusion=parse_bool(os.getenv("ENABLE_FUSION", "true")),
        enable_answer_guard=parse_bool(os.getenv("ENABLE_ANSWER_GUARD", "true")),
        enable_memory_summary=parse_bool(os.getenv("ENABLE_MEMORY_SUMMARY", "true")),
        enable_vision_llm=parse_bool(os.getenv("ENABLE_VISION_LLM", "false")),
        enable_llm_query_generation=parse_bool(os.getenv("ENABLE_LLM_QUERY_GENERATION", "false")),
    )


def metric_definitions() -> dict[str, str]:
    return {
        "retrieval_hit@1": "RAG Top-1 命中率，越高说明首条召回越准。",
        "retrieval_hit@3": "RAG Top-3 命中率，适合衡量推荐候选覆盖。",
        "retrieval_hit@5": "RAG Top-5 命中率，适合衡量召回上限。",
        "retrieval_mrr@5": "第一个正确结果的倒数排名均值，越高表示正确结果越靠前。",
        "router_intent_accuracy": "Router 意图分类准确率。",
        "router_agent_accuracy": "Router 目标 Agent 选择准确率。",
        "chat_success_rate": "端到端聊天用例通过率，需要运行服务后评估。",
        "latency_ms_avg": "平均响应时间，越低越好。",
        "evidence_guard_rate": "Answer Guard 触发或通过比例，用于观察幻觉防护覆盖。",
    }


def ablation_options() -> dict[str, dict[str, str]]:
    return {
        "RAG_BACKEND": {
            "bm25": "轻量本地 BM25 关键词检索，启动快、可离线，不加载向量库。",
            "chroma": "持久化向量库，适合展示标准 RAG 架构。",
            "faiss": "FAISS HNSW 向量索引，适合大规模相似度检索。",
        },
        "RERANK_ENABLED": {
            "true": "召回后重排，提高 Top-K 排序质量。",
            "false": "只看原始召回，用于衡量重排收益。",
        },
        "ENABLE_GRAPHRAG": {
            "true": "RAG 命中后补充 Neo4j 图谱上下文。",
            "false": "只使用普通 RAG，用于衡量图谱增强收益。",
        },
        "ENABLE_FUSION": {
            "true": "综合问题融合 RAG、SQL、Cypher。",
            "false": "综合问题退回普通 RAG，用于衡量多源融合收益。",
        },
        "ENABLE_ANSWER_GUARD": {
            "true": "最终回答前检查证据，降低幻觉。",
            "false": "关闭最终证据检查，用于观察防护影响。",
        },
        "ENABLE_MEMORY_SUMMARY": {
            "true": "短期窗口外生成长期摘要。",
            "false": "只保留短期窗口，用于衡量多轮记忆收益。",
        },
        "ENABLE_VISION_LLM": {
            "true": "图片识别优先使用视觉模型。",
            "false": "图片识别只用规则 fallback，用于衡量视觉模型收益。",
        },
    }


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_bool(primary: str, fallback: str | None = None, default: bool = False) -> bool:
    value = os.getenv(primary)
    if value is None and fallback:
        value = os.getenv(fallback)
    if value is None:
        return default
    return parse_bool(value)


def normalize_rag_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend in {"bm25", "keyword", "keywords", "sklearn"}:
        return "bm25"
    return backend
