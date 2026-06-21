from __future__ import annotations

from src.retriever import RecipeRetriever
from src.services.image_analyzer import ImageAnalyzer
from src.state import AgentState


class VisionAgent:
    def __init__(self, analyzer: ImageAnalyzer, retriever: RecipeRetriever) -> None:
        self.analyzer = analyzer
        self.retriever = retriever

    def run(self, state: AgentState) -> AgentState:
        image_bytes = state.meta.get("image_bytes", b"")
        if isinstance(image_bytes, str):
            image_bytes = image_bytes.encode("utf-8")
        filename = str(state.meta.get("image_filename") or "")
        analysis = self.analyzer.analyze(bytes(image_bytes), filename=filename, user_hint=state.user_input)
        state.vision_result = analysis.to_dict()
        state.meta["vision_result"] = state.vision_result
        state.meta.pop("image_bytes", None)

        query_parts = [
            analysis.dish_name,
            " ".join(analysis.ingredients),
            analysis.cooking_method,
            state.user_input,
        ]
        query = " ".join(part for part in query_parts if part and part != "未知菜品")
        if query.strip():
            state.retrieved_docs = self.retriever.search(query, state.top_k)

        recipe_names = "、".join(recipe.name for recipe, _ in state.retrieved_docs) or "暂无匹配菜谱"
        state.agent_output = (
            f"Vision Agent 图片识别结果：可能菜品={analysis.dish_name}，"
            f"置信度={analysis.confidence:.2f}，主要食材={ '、'.join(analysis.ingredients) or '待确认' }，"
            f"烹饪方式={analysis.cooking_method}。"
            f"相似菜谱检索结果：{recipe_names}。"
        )
        return state
