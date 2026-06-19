from __future__ import annotations

from app.agents.preference_agent import preferences_to_query_suffix, violates_ingredients
from app.retriever import RecipeRetriever
from app.services.memory import preferences_from_dict
from app.state import AgentState


class NutritionAgent:
    def __init__(self, retriever: RecipeRetriever) -> None:
        self.retriever = retriever

    def run(self, state: AgentState) -> AgentState:
        preferences = preferences_from_dict(state.meta.get("user_preferences"))
        query_suffix = preferences_to_query_suffix(preferences)
        query = f"{state.chat_history}\n{state.user_input}\n{query_suffix}".strip()
        candidates = self.retriever.search(query, max(state.top_k * 3, state.top_k))
        allergy_safe = [item for item in candidates if not violates_preferences(item[0].ingredients, [], preferences.allergies)]
        preference_safe = [
            item for item in allergy_safe if not violates_preferences(item[0].ingredients, preferences.dislikes, [])
        ]
        state.retrieved_docs = (preference_safe or allergy_safe)[: state.top_k]
        if not state.retrieved_docs:
            state.agent_output = (
                "该问题属于营养分析，但暂时没有匹配菜谱作为依据。"
                "回答时应说明只能提供一般饮食建议，不能替代医生或营养师建议。"
            )
            return state

        recipes = []
        for recipe, _ in state.retrieved_docs:
            recipes.append(
                f"{recipe.name}: 热量约{recipe.calories}千卡；"
                f"标签={','.join(recipe.tags)}；适合={','.join(recipe.suitable_for)}；"
                f"主要食材={','.join(recipe.ingredients)}"
            )
        state.agent_output = (
            "Nutrition Agent 专项要求："
            "请围绕热量、蛋白质/脂肪/碳水倾向、饮食目标匹配度进行解释；"
            "避免医疗诊断和绝对化健康承诺；"
            "如涉及疾病、孕妇、儿童、老人或过敏，应加入谨慎提醒。"
            "候选依据："
            + "；".join(recipes)
        )
        return state


def violates_preferences(ingredients: list[str], dislikes: list[str], allergies: list[str]) -> bool:
    return violates_ingredients(ingredients, [item for item in dislikes + allergies if item])
