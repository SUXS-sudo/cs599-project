from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "name",
    "ingredients",
    "category",
    "cooking_time",
    "difficulty",
    "tags",
    "calories",
    "suitable_for",
    "steps",
)
TIME_RE = re.compile(r"(\d+)")


@dataclass
class PipelineReport:
    source_count: int
    cleaned_count: int
    rejected_count: int
    categories: dict[str, int]
    tag_count: int
    ingredient_count: int
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "cleaned_count": self.cleaned_count,
            "rejected_count": self.rejected_count,
            "categories": self.categories,
            "tag_count": self.tag_count,
            "ingredient_count": self.ingredient_count,
            "output_path": self.output_path,
        }


def normalize_recipe(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name") or item.get("title") or "").strip()
    ingredients = normalize_list(item.get("ingredients"))
    steps = normalize_steps(item.get("steps") or item.get("description") or "")
    if not name or not ingredients or len(split_steps(steps)) < 2:
        return None

    cooking_time = str(item.get("cooking_time") or item.get("time") or "").strip()
    if cooking_time and "分钟" not in cooking_time and cooking_time.isdigit():
        cooking_time = f"{cooking_time}分钟"
    calories = item.get("calories", 0)
    try:
        calories = int(float(calories))
    except (TypeError, ValueError):
        calories = 0

    tags = normalize_list(item.get("tags"))
    suitable_for = normalize_list(item.get("suitable_for"))
    category = str(item.get("category") or infer_category(tags, suitable_for) or "家常菜").strip()
    return {
        "name": name,
        "ingredients": ingredients,
        "category": category,
        "cooking_time": cooking_time or "时间未知",
        "difficulty": str(item.get("difficulty") or "未知").strip(),
        "tags": tags or ["家常菜", "可检索"],
        "calories": calories if calories > 0 else 300,
        "suitable_for": suitable_for or ["午餐", "晚餐"],
        "steps": steps,
    }


def run_recipe_pipeline(source_path: Path, output_path: Path) -> PipelineReport:
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    cleaned = []
    seen_names = set()
    for item in raw:
        recipe = normalize_recipe(item)
        if recipe is None or recipe["name"] in seen_names:
            continue
        seen_names.add(recipe["name"])
        cleaned.append(recipe)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    categories: dict[str, int] = {}
    tags = set()
    ingredients = set()
    for recipe in cleaned:
        categories[recipe["category"]] = categories.get(recipe["category"], 0) + 1
        tags.update(recipe["tags"])
        ingredients.update(recipe["ingredients"])
    return PipelineReport(
        source_count=len(raw),
        cleaned_count=len(cleaned),
        rejected_count=len(raw) - len(cleaned),
        categories=dict(sorted(categories.items())),
        tag_count=len(tags),
        ingredient_count=len(ingredients),
        output_path=str(output_path),
    )


def build_eval_seed(recipes: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    rows = []
    for recipe in recipes[:limit]:
        ingredient = recipe["ingredients"][0]
        rows.append(
            {
                "message": f"我有{ingredient}，推荐一道适合{recipe['suitable_for'][0]}的菜",
                "expected_intent": "recipe_search",
                "contains": [recipe["name"]],
            }
        )
    return rows


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,，、;；\s]+", value)
    else:
        parts = [str(item) for item in value]
    result = []
    for part in parts:
        item = part.strip()
        if item and item not in result:
            result.append(item)
    return result


def normalize_steps(value: Any) -> str:
    if isinstance(value, list):
        text = "；".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value).strip()
    return text.replace("\n", "；")


def split_steps(steps: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。；;]", steps) if part.strip()]


def infer_category(tags: list[str], suitable_for: list[str]) -> str:
    for candidate in ("减脂餐", "增肌餐", "儿童餐", "素食", "早餐"):
        if candidate in tags or candidate in suitable_for:
            return candidate
    return "家常菜"
