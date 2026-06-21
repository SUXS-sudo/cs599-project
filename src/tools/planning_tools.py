from __future__ import annotations

from collections import Counter
from typing import Any

from src.retriever import Recipe, RecipeRetriever
from src.state import AgentState
from src.tools.base import ToolResult


class FilterRecipesByConstraintsTool:
    name = "filter_recipes_by_constraints"
    description = "Filter candidate recipes by preferences, allergies, dislikes, calories, time, tags, and available ingredients."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            "max_calories": {"type": "integer"},
            "max_minutes": {"type": "integer"},
            "required_tags": {"type": "array", "items": {"type": "string"}},
            "avoid_ingredients": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
    }

    def __init__(self, retriever: RecipeRetriever) -> None:
        self.retriever = retriever

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        candidates = recipes_from_state(state)
        if not candidates:
            query = str(args.get("query") or state.user_input).strip()
            candidates = self.retriever.search(query, top_k=max(top_k * 3, top_k))

        constraints = build_constraints(args, state)
        filtered = []
        for recipe, score in candidates:
            ok, reasons = recipe_matches_constraints(recipe, constraints)
            if ok:
                filtered.append((recipe, score, reasons))
        filtered = filtered[:top_k]
        if not filtered:
            return ToolResult(
                self.name,
                True,
                "filter_recipes_by_constraints: no candidate recipes satisfied all constraints.",
                data={"filtered_recipes": [], "constraints": constraints},
            )

        lines = ["filter_recipes_by_constraints results:"]
        for index, (recipe, score, reasons) in enumerate(filtered, start=1):
            lines.append(
                f"{index}. {recipe.name} | score={score:.3f} | calories={recipe.calories} | "
                f"time={recipe.cooking_time} | matched={', '.join(reasons) or 'basic match'}"
            )
        return ToolResult(
            self.name,
            True,
            "\n".join(lines),
            data={
                "filtered_recipes": [recipe_to_dict(recipe, score) for recipe, score, _ in filtered],
                "constraints": constraints,
            },
        )


class BuildShoppingListTool:
    name = "build_shopping_list"
    description = "Build a merged shopping list from selected or retrieved recipes."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": [],
    }

    def __init__(self, retriever: RecipeRetriever) -> None:
        self.retriever = retriever

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        recipes = [recipe for recipe, _ in recipes_from_state(state)[:top_k]]
        if not recipes:
            query = str(args.get("query") or state.user_input).strip()
            recipes = [recipe for recipe, _ in self.retriever.search(query, top_k=top_k)]
        if not recipes:
            return ToolResult(self.name, True, "build_shopping_list: no recipes available.", data={"shopping_list": []})

        counts = Counter()
        for recipe in recipes:
            for ingredient in recipe.ingredients:
                if ingredient.strip():
                    counts[ingredient.strip()] += 1
        items = [{"ingredient": name, "recipe_count": count} for name, count in counts.most_common()]
        lines = ["build_shopping_list results:", "recipes=" + ", ".join(recipe.name for recipe in recipes)]
        for item in items:
            lines.append(f"- {item['ingredient']} | used_by={item['recipe_count']} recipe(s)")
        return ToolResult(
            self.name,
            True,
            "\n".join(lines),
            data={"shopping_list": items, "recipes": [recipe.name for recipe in recipes]},
        )


class PlanWeeklyMenuTool:
    name = "plan_weekly_menu"
    description = "Plan a simple multi-day menu from local recipes using the user's goal and constraints."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
            "meals_per_day": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": [],
    }

    def __init__(self, retriever: RecipeRetriever) -> None:
        self.retriever = retriever

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        days = clamp_int(args.get("days", infer_days(state.user_input)), default=7, minimum=1, maximum=7)
        meals_per_day = clamp_int(args.get("meals_per_day", infer_meals_per_day(state.user_input)), default=1, minimum=1, maximum=3)
        total = days * meals_per_day
        query = str(args.get("query") or state.user_input).strip()
        recipes = [recipe for recipe, _ in recipes_from_state(state)]
        if len(recipes) < total:
            seen = {recipe.name for recipe in recipes}
            for recipe, _ in self.retriever.search(query, top_k=min(max(total * 2, total), 20)):
                if recipe.name not in seen:
                    recipes.append(recipe)
                    seen.add(recipe.name)
        if not recipes:
            return ToolResult(self.name, True, "plan_weekly_menu: no recipes available.", data={"weekly_menu": []})

        menu = []
        meal_names = ["breakfast", "lunch", "dinner"]
        for day in range(1, days + 1):
            meals = []
            for meal_index in range(meals_per_day):
                recipe = recipes[((day - 1) * meals_per_day + meal_index) % len(recipes)]
                meals.append(
                    {
                        "meal": meal_names[meal_index] if meals_per_day > 1 else "dinner",
                        "recipe": recipe.name,
                        "calories": recipe.calories,
                        "ingredients": list(recipe.ingredients),
                    }
                )
            menu.append({"day": day, "meals": meals})

        lines = ["plan_weekly_menu results:"]
        for day in menu:
            meal_text = "; ".join(f"{meal['meal']}: {meal['recipe']} ({meal['calories']} kcal)" for meal in day["meals"])
            lines.append(f"Day {day['day']}: {meal_text}")
        return ToolResult(self.name, True, "\n".join(lines), data={"weekly_menu": menu})


def recipes_from_state(state: AgentState) -> list[tuple[Recipe, float]]:
    return [(recipe, score) for recipe, score in state.retrieved_docs]


def build_constraints(args: dict[str, Any], state: AgentState) -> dict[str, Any]:
    preferences = state.meta.get("user_preferences") if isinstance(state.meta.get("user_preferences"), dict) else {}
    avoid = list_from_args(args.get("avoid_ingredients"))
    avoid.extend(str(item) for item in preferences.get("allergies", []) if str(item).strip())
    avoid.extend(str(item) for item in preferences.get("dislikes", []) if str(item).strip())
    required_tags = list_from_args(args.get("required_tags"))
    required_tags.extend(str(item) for item in preferences.get("preferences", []) if str(item).strip())
    max_calories = optional_int(args.get("max_calories"))
    max_minutes = optional_int(args.get("max_minutes"))
    return {
        "avoid_ingredients": unique(avoid),
        "required_tags": unique(required_tags),
        "max_calories": max_calories,
        "max_minutes": max_minutes,
    }


def recipe_matches_constraints(recipe: Recipe, constraints: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    avoid = constraints.get("avoid_ingredients") or []
    if any(term and any(term in ingredient or ingredient in term for ingredient in recipe.ingredients) for term in avoid):
        return False, []
    max_calories = constraints.get("max_calories")
    if max_calories is not None and recipe.calories > max_calories:
        return False, []
    max_minutes = constraints.get("max_minutes")
    minutes = first_int(recipe.cooking_time)
    if max_minutes is not None and minutes is not None and minutes > max_minutes:
        return False, []
    required_tags = constraints.get("required_tags") or []
    recipe_tags = recipe.tags + recipe.suitable_for + [recipe.category]
    for tag in required_tags:
        if any(tag in candidate or candidate in tag for candidate in recipe_tags):
            reasons.append(tag)
    return True, reasons


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


def list_from_args(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_int(text: str) -> int | None:
    digits = "".join(char if char.isdigit() else " " for char in text).split()
    return int(digits[0]) if digits else None


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def infer_days(message: str) -> int:
    if "一周" in message or "7天" in message or "七天" in message:
        return 7
    return 3


def infer_meals_per_day(message: str) -> int:
    if "三餐" in message or "早午晚" in message:
        return 3
    if "午晚" in message or "两餐" in message:
        return 2
    return 1


def unique(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result
