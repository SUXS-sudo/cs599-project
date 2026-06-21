from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.data_pipeline import normalize_recipe, run_recipe_pipeline
from app.services.query_boundary_guard import QueryBoundaryGuard
from app.state import AgentState


class SafetyAgent:
    """Structured boundary guard before routing."""

    def __init__(self, boundary_guard: QueryBoundaryGuard | None = None) -> None:
        self.boundary_guard = boundary_guard or QueryBoundaryGuard()

    def run(self, state: AgentState) -> AgentState:
        result = self.boundary_guard.evaluate(state.user_input)
        state.meta["query_boundary"] = result.to_meta()
        state.meta["normalized_query"] = result.normalized_text

        if result.reason_code == "EMPTY_QUERY":
            state.meta["safety_status"] = "empty"
            return state

        if result.decision == "block":
            state.intent = "out_of_scope"
            state.target_agent = "general_agent"
            state.agent_output = (
                "这个问题涉及不安全或违法风险，我不能提供相关做法。"
                "如果你需要的是正常饮食、食材处理或营养建议，可以换个安全的描述方式继续问我。"
            )
            state.meta["safety_status"] = "blocked"
            state.meta["safety_reason"] = result.reason_code
            return state

        if result.decision == "caution":
            state.meta["safety_status"] = "caution"
            state.meta["safety_note"] = result.reason_code
            return state

        state.meta["safety_status"] = "passed"
        return state


class GeneralAgent:
    def run(self, state: AgentState) -> AgentState:
        state.agent_output = "当前系统主要支持菜谱、食材、营养和饮食建议相关问题。"
        state.retrieved_docs = []
        return state


class DataAgent:
    """Offline recipe parsing and pipeline helper.

    This agent is intentionally not on the user chat path. It supports the
    final-version data-construction workflow: normalize raw recipe records,
    produce clean JSON artifacts, and expose a concise report.
    """

    def parse_recipe(self, item: dict[str, Any]) -> dict[str, Any] | None:
        return normalize_recipe(item)

    def run_pipeline(self, source_path: Path, output_path: Path) -> dict[str, Any]:
        return run_recipe_pipeline(source_path, output_path).to_dict()

    def run(self, state: AgentState) -> AgentState:
        source = Path(str(state.meta.get("pipeline_source", "data/recipes.json")))
        output = Path(str(state.meta.get("pipeline_output", "data/processed/recipes_clean.json")))
        report = self.run_pipeline(source, output)
        state.meta["data_pipeline_report"] = report
        state.agent_output = (
            "Data Agent 已完成菜谱清洗："
            f"输入 {report['source_count']} 条，输出 {report['cleaned_count']} 条，"
            f"拒绝 {report['rejected_count']} 条。"
        )
        return state
