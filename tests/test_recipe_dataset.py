from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RECIPES_PATH = ROOT / "data" / "recipes.json"
REQUIRED_FIELDS = {
    "name",
    "ingredients",
    "category",
    "cooking_time",
    "difficulty",
    "tags",
    "calories",
    "suitable_for",
    "steps",
}
REQUIRED_CATEGORIES = {"家常菜", "川菜", "粤菜", "湘菜", "汤羹", "粥", "面食", "减脂餐", "增肌餐", "儿童餐", "素食"}


def load_recipes() -> list[dict]:
    return json.loads(RECIPES_PATH.read_text(encoding="utf-8"))


def test_recipe_dataset_has_at_least_300_items() -> None:
    recipes = load_recipes()
    assert len(recipes) >= 300
    assert len({item["name"] for item in recipes}) == len(recipes)


def test_recipe_dataset_has_required_fields_and_no_nulls() -> None:
    for item in load_recipes():
        assert REQUIRED_FIELDS <= set(item)
        for field in REQUIRED_FIELDS:
            assert item[field] is not None, f"{item.get('name')} has null {field}"
        assert isinstance(item["ingredients"], list)
        assert item["ingredients"]
        assert isinstance(item["tags"], list)
        assert len(item["tags"]) >= 2
        assert isinstance(item["suitable_for"], list)
        assert item["suitable_for"]
        assert isinstance(item["calories"], (int, float))
        assert item["calories"] > 0
        steps = [part.strip() for part in str(item["steps"]).replace("。", "；").split("；") if part.strip()]
        assert len(steps) >= 3


def test_recipe_dataset_covers_required_categories() -> None:
    categories = {item["category"] for item in load_recipes()}
    assert REQUIRED_CATEGORIES <= categories
