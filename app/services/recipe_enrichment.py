"""Shared recipe validation, enrichment and inference utilities.

Provides strict validation, nutrition/category/difficulty inference,
and ingredient quality checks used by both the heterogeneous pipeline
and the PDF1 pipeline.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_TERMS = (
    "洗净", "切丝", "切片", "切段", "切块", "切开", "切成", "去皮", "去蒂", "去籽",
    "倒入", "放入", "加入", "下入", "捞出", "沥干", "盛入", "装盘", "备用", "焯", "煮",
    "翻炒", "炸", "蒸", "炖", "搅拌", "腌", "烧热", "锅中", "锅内", "炒锅", "关火",
)
INGREDIENT_FRAGMENTS = {"青", "红", "白", "黑", "鲜", "熟", "干", "丝", "片", "段", "块"}
FIELD_MARKERS = ("菜名", "原料", "调料", "制作方法", "小提示", "来源页码")
QUANTITY_PATTERN = re.compile(
    r"(?:约|各|共|少许|适量|若干)?\s*\d+(?:\.\d+)?(?:\s*[-~～至]\s*\d+(?:\.\d+)?)?\s*"
    r"(?:千克|公斤|克|kg|g|毫升|ml|升|l|个|只|片|根|匙|勺|杯|棵|张|块|粒|瓣|条|朵)?",
    re.IGNORECASE,
)

NUTRITION_REFERENCE: dict[str, tuple[int, float, float]] = {
    "鸡胸肉": (165, 31.0, 3.6), "鸡肉": (190, 27.0, 7.0), "牛肉": (250, 26.0, 15.0),
    "猪肉": (290, 24.0, 21.0), "五花肉": (518, 9.0, 53.0), "鱼": (120, 22.0, 3.0),
    "虾": (99, 24.0, 0.3), "鸡蛋": (143, 13.0, 10.0), "豆腐": (84, 8.0, 5.0),
    "花生": (567, 26.0, 49.0), "土豆": (77, 2.0, 0.1), "山药": (57, 1.9, 0.2),
    "红薯": (86, 1.6, 0.1), "米": (130, 2.7, 0.3), "面": (138, 4.5, 2.1),
    "燕麦": (389, 17.0, 7.0), "西兰花": (34, 2.8, 0.4), "黄瓜": (15, 0.7, 0.1),
    "番茄": (18, 0.9, 0.2), "木耳": (25, 1.5, 0.2), "银耳": (36, 1.4, 0.2),
}


# ---------------------------------------------------------------------------
# Ingredient helpers
# ---------------------------------------------------------------------------

def contains_action_phrase(value: str) -> bool:
    return any(term in value for term in ACTION_TERMS)


def is_suspect_ingredient(value: str) -> bool:
    if not value or len(value) > 16:
        return True
    if value in INGREDIENT_FRAGMENTS:
        return True
    if any(marker in value for marker in FIELD_MARKERS):
        return True
    if contains_action_phrase(value):
        return True
    if re.search(r"[。；;：:]", value):
        return True
    return not bool(re.search(r"[一-鿿A-Za-z]", value))


def normalize_ingredient_name(value: str) -> str:
    cleaned = value.strip(" ：:（）()[]【】。；;，,、")
    cleaned = re.sub(r"^(?:原料|主料|辅料)[：:]?", "", cleaned)
    cleaned = QUANTITY_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"(?:各|共)?(?:适量|少许|若干)$", "", cleaned)
    cleaned = re.sub(r"各$", "", cleaned)
    return cleaned.strip(" ：:（）()[]【】。；;，,、")


def trim_ingredient_field_at_actions(text: str) -> str:
    normalized = re.sub(r"\s+", "", text.strip())
    sentences = re.split(r"(?<=[。；;])", normalized)
    kept: list[str] = []
    for sentence in sentences:
        value = sentence.strip()
        if not value:
            continue
        if kept and contains_action_phrase(value):
            break
        kept.append(value)
    return "".join(kept)


def split_ingredients(text: str) -> list[str]:
    ingredient_prefix = trim_ingredient_field_at_actions(text)
    parts = re.split(r"[、，,；;。\n]+", ingredient_prefix)
    result: list[str] = []
    for part in parts:
        cleaned = normalize_ingredient_name(part)
        if not cleaned or is_suspect_ingredient(cleaned):
            continue
        if cleaned not in result:
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def infer_category(text: str) -> str:
    rules = (
        (("汤", "羹"), "汤羹"), (("粥",), "粥"), (("沙拉", "凉拌", "拌"), "凉菜"),
        (("炸", "香酥", "香脆"), "炸物"), (("蒸",), "蒸菜"), (("炖", "煲", "焖"), "炖菜"),
        (("烤",), "烤箱菜"), (("面", "饺", "馄饨", "饼"), "面食"),
    )
    for keywords, category in rules:
        if any(word in text for word in keywords):
            return category
    return "家常菜"


def infer_cooking_minutes(text: str, category: str) -> int:
    defaults = {"凉菜": 15, "炸物": 25, "蒸菜": 25, "炖菜": 60, "烤箱菜": 40, "汤羹": 35, "粥": 45, "面食": 35}
    if "腌" in text:
        return max(defaults.get(category, 25), 30)
    return defaults.get(category, 20)


def infer_difficulty(steps: str, minutes: int) -> str:
    action_count = sum(steps.count(word) for word in ("切", "焯", "腌", "炸", "蒸", "炖", "翻炒", "调汁", "勾芡"))
    if minutes >= 60 or action_count >= 7 or len(steps) >= 180:
        return "困难"
    if minutes >= 35 or action_count >= 4 or len(steps) >= 90:
        return "中等"
    return "简单"


def estimate_nutrition_per_100g(
    ingredients: list[str], text: str, category: str
) -> tuple[float, float, float]:
    matches: list[tuple[int, float, float]] = []
    for ingredient in ingredients:
        for keyword, values in NUTRITION_REFERENCE.items():
            if keyword in ingredient:
                matches.append(values)
                break
    if matches:
        calories = sum(item[0] for item in matches) / len(matches)
        protein = sum(item[1] for item in matches) / len(matches)
        fat = sum(item[2] for item in matches) / len(matches)
    else:
        calories, protein, fat = (140.0, 6.0, 6.0)
    if category == "炸物":
        calories += 90
        fat += 10
    elif any(word in text for word in ("炒", "煎", "油")):
        calories += 35
        fat += 4
    elif category in {"汤羹", "粥"}:
        calories *= 0.7
        protein *= 0.8
        fat *= 0.7
    return min(max(calories, 30), 600), min(max(protein, 0), 40), min(max(fat, 0), 55)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_duration_minutes(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def positive_int(value: Any) -> int:
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def positive_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def unique_strings(value: Any, fallback: list[str] | None = None) -> list[str]:
    items = value if isinstance(value, list) else (fallback or [])
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


# ---------------------------------------------------------------------------
# Strict validation + enrichment
# ---------------------------------------------------------------------------

def validate_and_normalize_recipe(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Strictly validate a recipe candidate and enrich with inferred fields.

    Returns ``(normalized_dict, [])`` on success, ``(None, errors)`` on rejection.
    """
    errors: list[str] = []
    name = str(candidate.get("name") or "").strip()

    # -- normalise ingredients --
    raw_ingredients = candidate.get("ingredients")
    if isinstance(raw_ingredients, str):
        ingredients = split_ingredients(raw_ingredients)
    elif isinstance(raw_ingredients, list):
        ingredients = []
        for item in raw_ingredients:
            raw_name = item.get("name") if isinstance(item, dict) else item
            cleaned = normalize_ingredient_name(str(raw_name or ""))
            if cleaned and not is_suspect_ingredient(cleaned) and cleaned not in ingredients:
                ingredients.append(cleaned)
    else:
        ingredients = []

    # -- normalise steps --
    steps_value = candidate.get("steps")
    if isinstance(steps_value, list):
        steps = "；".join(
            str(item.get("text") if isinstance(item, dict) else item).strip()
            for item in steps_value
        )
    else:
        steps = str(steps_value or "").strip()

    # -- validation rules --
    if len(name) < 2 or len(name) > 24 or any(marker in name for marker in FIELD_MARKERS):
        errors.append("invalid_dish_name")
    if name.startswith(("将", "把", "放入", "加入", "倒入", "洗净", "切成")):
        errors.append("dish_name_looks_like_step")
    if not ingredients:
        errors.append("no_valid_ingredients")
    if any(is_suspect_ingredient(item) for item in ingredients):
        errors.append("ingredient_looks_like_step")
    if len(steps) < 8:
        errors.append("missing_or_short_steps")

    if errors:
        return None, errors

    enriched = enrich_recipe_fields({**candidate, "name": name, "ingredients": ingredients, "steps": steps})
    return {
        "name": name,
        "ingredients": ingredients,
        "category": enriched["category"],
        "cooking_time_minutes": enriched["cooking_time_minutes"],
        "difficulty": enriched["difficulty"],
        "tags": enriched["tags"],
        "calories_per_100g": enriched["calories_per_100g"],
        "protein_g_per_100g": enriched["protein_g_per_100g"],
        "fat_g_per_100g": enriched["fat_g_per_100g"],
        "nutrition_estimated": enriched["nutrition_estimated"],
        "suitable_for": enriched["suitable_for"],
        "steps": steps,
    }, []


def enrich_recipe_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    """Produce canonical fields (category, nutrition, tags, etc.) from a recipe candidate."""
    name = str(candidate.get("name") or "").strip()
    steps = str(candidate.get("steps") or "").strip()
    ingredients = [str(item).strip() for item in candidate.get("ingredients", []) if str(item).strip()]
    evidence = f"{name} {steps}"

    category = str(candidate.get("category") or "").strip() or infer_category(evidence)
    minutes = positive_int(candidate.get("cooking_time_minutes"))
    if not minutes:
        minutes = parse_duration_minutes(candidate.get("cooking_time")) or infer_cooking_minutes(evidence, category)
    difficulty = str(candidate.get("difficulty") or "").strip() or infer_difficulty(steps, minutes)

    supplied_calories = positive_float(candidate.get("calories_per_100g"))
    if not supplied_calories:
        supplied_calories = positive_float(candidate.get("calories"))
    supplied_protein = positive_float(candidate.get("protein_g_per_100g"))
    supplied_fat = positive_float(candidate.get("fat_g_per_100g"))
    estimated_calories, estimated_protein, estimated_fat = estimate_nutrition_per_100g(ingredients, evidence, category)
    calories = int(round(supplied_calories or estimated_calories))
    protein = round(supplied_protein or estimated_protein, 1)
    fat = round(supplied_fat or estimated_fat, 1)
    nutrition_estimated = bool(candidate.get("nutrition_estimated", not bool(supplied_calories)))

    tags = unique_strings(candidate.get("tags"))
    for value in (category,):
        if value and value not in tags:
            tags.append(value)
    is_fried = category == "炸物" or any(word in name for word in ("炸", "油炸", "香酥", "香脆"))
    if protein >= 12 and "高蛋白" not in tags:
        tags.append("高蛋白")
    if fat <= 8 and not is_fried and "低脂" not in tags:
        tags.append("低脂")
    if calories <= 150 and "低热量" not in tags:
        tags.append("低热量")
    if is_fried and "油炸" not in tags:
        tags.append("油炸")
    suitable_for = unique_strings(candidate.get("suitable_for"))
    if "高蛋白" in tags and "健身" not in suitable_for:
        suitable_for.append("健身")
    if "低脂" in tags and "低热量" in tags and "减脂" not in suitable_for:
        suitable_for.append("减脂")

    return {
        "category": category,
        "cooking_time_minutes": max(1, minutes),
        "difficulty": difficulty,
        "calories_per_100g": max(1, calories),
        "protein_g_per_100g": max(0.0, protein),
        "fat_g_per_100g": max(0.0, fat),
        "nutrition_estimated": nutrition_estimated,
        "tags": tags,
        "suitable_for": suitable_for,
    }
