from __future__ import annotations

import os

from src.services.llm_client import LLMClient
from src.services.query_boundary_guard import QueryBoundaryGuard, normalize_query
from src.services.query_understanding import (
    build_extraction_prompt,
    build_selection_prompt,
    generate_correction_candidates,
    parse_json_object,
    preserves_critical_constraints,
)
from src.state import AgentState


class QueryUnderstandingAgent:
    """Two-stage LLM intent/entity extraction and candidate-constrained correction."""

    def __init__(self, llm_client: LLMClient | None, vocabulary: set[str] | None = None) -> None:
        self.llm_client = llm_client
        self.vocabulary = vocabulary or set()
        self.min_confidence = env_float("QUERY_UNDERSTANDING_MIN_CONFIDENCE", 0.8)
        self.boundary_guard = QueryBoundaryGuard()

    def run(self, state: AgentState) -> AgentState:
        original = state.user_input
        cleaned = normalize_query(original)
        state.meta["original_query"] = original
        state.meta["cleaned_query"] = cleaned

        if os.getenv("QUERY_UNDERSTANDING_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
            state.meta["query_understanding"] = {"status": "disabled", "resolved_query": cleaned}
            state.user_input = cleaned
            return state

        extraction: dict = {}
        llm_available = bool(self.llm_client and self.llm_client.available)
        if llm_available:
            extraction = parse_json_object(
                self.llm_client.generate(build_extraction_prompt(cleaned, state.chat_history), max_tokens=350, timeout=12)
            )

        candidates = generate_correction_candidates(cleaned, extraction, self.vocabulary)
        selected_index = 0
        llm_confidence = 0.0
        selection: dict = {}
        mode = "rule_fallback"
        if llm_available:
            selection = parse_json_object(
                self.llm_client.generate(build_selection_prompt(cleaned, extraction, candidates), max_tokens=180, timeout=10)
            )
            try:
                selected_index = int(selection.get("candidate_index", 0))
                llm_confidence = float(selection.get("confidence", 0.0))
            except (TypeError, ValueError):
                selected_index = 0
                llm_confidence = 0.0
            mode = "llm_candidate_selection"
        elif len(candidates) > 1 and candidates[1].score >= 0.95:
            selected_index = 1
            llm_confidence = candidates[1].score

        cleanup_priority = {
            "dish_name_cleanup": 3,
            "inline_noise_and_typo_cleanup": 2,
            "inline_noise_cleanup": 1,
        }
        cleanup_indexes = [index for index, candidate in enumerate(candidates) if candidate.source in cleanup_priority]
        deterministic_cleanup_index = (
            max(cleanup_indexes, key=lambda index: cleanup_priority[candidates[index].source])
            if cleanup_indexes
            else None
        )
        if deterministic_cleanup_index is not None:
            selected_index = deterministic_cleanup_index
            llm_confidence = max(llm_confidence, candidates[deterministic_cleanup_index].score)
            mode = f"deterministic_{candidates[deterministic_cleanup_index].source}"

        if selected_index < 0 or selected_index >= len(candidates):
            selected_index = 0
        selected = candidates[selected_index]
        confidence = llm_confidence if selected_index == 0 else (0.7 * llm_confidence + 0.3 * selected.score)
        is_dish_cleanup = selected.source == "dish_name_cleanup"
        is_safe_cleanup = selected.source in {
            "dish_name_cleanup",
            "inline_noise_and_typo_cleanup",
            "inline_noise_cleanup",
        }
        accepted = (
            selected_index != 0
            and confidence >= self.min_confidence
            and (is_safe_cleanup or preserves_critical_constraints(cleaned, selected.query))
        )
        resolved = selected.query if accepted else cleaned
        state.user_input = resolved
        state.meta["resolved_query"] = resolved
        if accepted and is_dish_cleanup:
            state.meta["resolved_dish_name"] = resolved
            state.meta["dish_name_only"] = True
        state.meta["query_understanding"] = {
            "status": "corrected" if accepted else "unchanged",
            "mode": mode,
            "extracted_intent": str(extraction.get("intent") or ""),
            "entities": extraction.get("entities", []),
            "candidates": [candidate.to_dict() for candidate in candidates],
            "selected_index": selected_index,
            "confidence": round(confidence, 4),
            "threshold": self.min_confidence,
            "resolved_query": resolved,
            "reason": str(selection.get("reason") or extraction.get("reason") or ""),
        }
        self._recheck_resolved_query_boundary(state, resolved)
        return state

    def _recheck_resolved_query_boundary(self, state: AgentState, resolved: str) -> None:
        result = self.boundary_guard.evaluate(resolved)
        if result.decision == "block":
            state.intent = "out_of_scope"
            state.target_agent = "general_agent"
            state.agent_output = (
                "纠错后的查询涉及不安全或越权风险，我不能继续执行。"
                "如果你需要正常的菜谱、食材处理或营养建议，可以换个安全的描述方式。"
            )
            state.meta["safety_status"] = "blocked"
            state.meta["safety_reason"] = f"RESOLVED_{result.reason_code}"
            state.meta["query_boundary"] = result.to_meta()
        elif result.decision == "caution" and state.meta.get("safety_status") != "blocked":
            state.meta["safety_status"] = "caution"
            state.meta["safety_note"] = f"RESOLVED_{result.reason_code}"
            state.meta["query_boundary"] = result.to_meta()


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
