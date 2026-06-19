from __future__ import annotations

from typing import Any

from app.retriever import Recipe, RecipeRetriever
from app.services.memory import MemoryStore
from app.state import AgentState
from app.tools.base import ToolResult


class SearchRecipesTool:
    name = "search_recipes"
    description = "Search the local SmartRecipe recipe index by natural-language query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: RecipeRetriever) -> None:
        self.retriever = retriever

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query") or state.user_input).strip()
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        if not query:
            return ToolResult(self.name, False, "", error="query is required")

        results = self.retriever.search(query, top_k=top_k)
        if not results:
            return ToolResult(
                self.name,
                True,
                "search_recipes: no matching local recipes found.",
                data={"retrieved_docs": []},
            )

        lines = ["search_recipes results:"]
        for index, (recipe, score) in enumerate(results, start=1):
            lines.append(format_recipe_line(index, recipe, score))
        return ToolResult(
            self.name,
            True,
            "\n".join(lines),
            data={
                "retrieved_docs": results,
                "recipes": [recipe_to_dict(recipe, score) for recipe, score in results],
            },
        )


class GetUserPreferencesTool:
    name = "get_user_preferences"
    description = "Read stored preferences, allergies, and dislikes for the current session."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        preferences = self.memory_store.get_preferences(state.session_id).to_dict()
        parts = []
        if preferences["preferences"]:
            parts.append("preferences=" + ", ".join(preferences["preferences"]))
        if preferences["allergies"]:
            parts.append("allergies=" + ", ".join(preferences["allergies"]))
        if preferences["dislikes"]:
            parts.append("dislikes=" + ", ".join(preferences["dislikes"]))
        content = "get_user_preferences: " + ("; ".join(parts) if parts else "no stored preferences.")
        return ToolResult(self.name, True, content, data={"preferences": preferences})


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def format_recipe_line(index: int, recipe: Recipe, score: float) -> str:
    ingredients = ", ".join(recipe.ingredients)
    tags = ", ".join(recipe.tags + recipe.suitable_for)
    return (
        f"{index}. {recipe.name} | score={score:.3f} | ingredients={ingredients} | "
        f"time={recipe.cooking_time} | difficulty={recipe.difficulty} | "
        f"calories={recipe.calories} | tags={tags} | steps={recipe.steps}"
    )


def recipe_to_dict(recipe: Recipe, score: float) -> dict[str, Any]:
    return {
        "name": recipe.name,
        "score": round(score, 4),
        "ingredients": list(recipe.ingredients),
        "category": recipe.category,
        "cooking_time": recipe.cooking_time,
        "difficulty": recipe.difficulty,
        "tags": list(recipe.tags),
        "calories": recipe.calories,
        "suitable_for": list(recipe.suitable_for),
    }
