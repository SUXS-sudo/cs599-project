from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.retriever import Recipe


@dataclass
class AgentState:
    user_input: str
    session_id: str
    top_k: int
    chat_history: str = ""
    intent: str = "out_of_scope"
    target_agent: str = "general_agent"
    retrieved_docs: list[tuple[Recipe, float]] = field(default_factory=list)
    fusion_results: list[dict[str, Any]] = field(default_factory=list)
    vision_result: dict[str, Any] = field(default_factory=dict)
    agent_output: str = ""
    final_answer: str = ""
    generator: str = "template"
    meta: dict[str, Any] = field(default_factory=dict)
