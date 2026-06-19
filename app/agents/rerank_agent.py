from __future__ import annotations

import os

from app.retriever import Recipe
from app.state import AgentState


DEFAULT_CROSS_ENCODER_MODEL = "models/BAAI/bge-reranker-base"


class RerankAgent:
    """Rerank retrieved recipes with a cross-encoder when available."""

    def __init__(self) -> None:
        self.enabled = os.getenv("RERANK_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
        self.cross_encoder_model_name = os.getenv("RERANK_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
        self._cross_encoder = None
        self._load_error = ""
        self._load_models()

    def run(self, state: AgentState) -> AgentState:
        if not state.retrieved_docs:
            state.meta["rerank_status"] = "skipped_no_docs"
            return state
        if not self.enabled:
            state.meta["rerank_status"] = "disabled"
            return state

        if self._cross_encoder is not None:
            reranked = self._rerank_with_cross_encoder(state.user_input, state.retrieved_docs)
            if reranked:
                state.retrieved_docs = reranked[: state.top_k]
                state.meta["rerank_status"] = "cross_encoder"
                state.meta["rerank_cross_encoder_model"] = self.cross_encoder_model_name
                if self._load_error:
                    state.meta["rerank_warning"] = self._load_error
                return state

        state.retrieved_docs = self._rerank_with_overlap(state.user_input, state.retrieved_docs)[: state.top_k]
        state.meta["rerank_status"] = "overlap_fallback"
        if self._load_error:
            state.meta["rerank_error"] = self._load_error
        return state

    def _load_models(self) -> None:
        if not self.enabled:
            return
        errors = []
        if self.cross_encoder_model_name:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(self.cross_encoder_model_name)
            except Exception as exc:
                self._cross_encoder = None
                errors.append(f"cross_encoder load failed: {type(exc).__name__}: {exc}")
        self._load_error = "; ".join(errors)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        docs: list[tuple[Recipe, float]],
    ) -> list[tuple[Recipe, float]] | None:
        try:
            reranked = []
            cross_scores = self._cross_encoder_scores(query, docs)
            for (recipe, base_score), cross_score in zip(docs, cross_scores):
                final_score = 0.75 * cross_score + 0.25 * float(base_score)
                reranked.append((recipe, final_score))
            reranked.sort(key=lambda item: item[1], reverse=True)
            return reranked
        except Exception:
            return None

    def _cross_encoder_scores(self, query: str, docs: list[tuple[Recipe, float]]) -> list[float]:
        pairs = [(query, recipe.searchable_text()) for recipe, _ in docs]
        raw_scores = self._cross_encoder.predict(pairs)
        return [normalize_cross_encoder_score(float(score)) for score in raw_scores]

    @staticmethod
    def _rerank_with_overlap(query: str, docs: list[tuple[Recipe, float]]) -> list[tuple[Recipe, float]]:
        reranked = []
        for recipe, base_score in docs:
            overlap = sum(0.08 for ingredient in recipe.ingredients if ingredient in query)
            overlap += sum(0.05 for tag in recipe.tags + recipe.suitable_for if tag in query)
            reranked.append((recipe, float(base_score) + overlap))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked


def normalize_cross_encoder_score(score: float) -> float:
    try:
        import numpy as np

        return float(1.0 / (1.0 + np.exp(-score)))
    except Exception:
        return score
