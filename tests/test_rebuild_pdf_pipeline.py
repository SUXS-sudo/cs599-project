from __future__ import annotations

from scripts.rebuild_pdf1_pipeline import (
    document_metadata_to_recipes,
    split_ingredients,
    validate_and_normalize_recipe,
)


def test_split_ingredients_stops_before_ocr_step_text() -> None:
    text = "土豆350克，红椒若干。土豆去皮，切丝，用清水漂洗干净；红椒洗净切段。锅置火上，倒入土豆丝。"

    ingredients = split_ingredients(text)

    assert ingredients == ["土豆", "红椒"]
    assert "切丝" not in ingredients
    assert "倒入土豆丝" not in ingredients


def test_split_ingredients_keeps_prepared_food_names() -> None:
    assert split_ingredients("水发黑木耳100克，水发银耳150克。") == ["水发黑木耳", "水发银耳"]


def test_recipe_validation_allows_cooking_method_in_real_dish_name() -> None:
    recipe, errors = validate_and_normalize_recipe(
        {
            "name": "白煮肉",
            "ingredients": ["带皮五花肉"],
            "steps": "五花肉洗净后煮熟，切片装盘。",
        }
    )

    assert errors == []
    assert recipe is not None
    assert recipe["name"] == "白煮肉"


def test_document_metadata_quarantines_bad_rows_and_keeps_clean_recipe() -> None:
    metadata = {
        "chunks": [
            {
                "chunk_id": "good",
                "text": (
                    "菜名：香脆土豆丝\n原料：土豆350克，红椒若干。土豆去皮，切丝。\n"
                    "调料：盐适量。\n制作方法：土豆切丝漂洗，和红椒一起炸至金黄。"
                ),
                "metadata": {
                    "dish_name": "香脆土豆丝",
                    "ingredients": "土豆350克，红椒若干。土豆去皮，切丝。",
                },
            },
            {
                "chunk_id": "bad",
                "text": "菜名：子\n原料：切丝。\n调料：盐。\n制作方法：切丝后装盘。",
                "metadata": {"dish_name": "子", "ingredients": "切丝"},
            },
        ]
    }

    recipes, report = document_metadata_to_recipes(metadata)

    assert len(recipes) == 1
    assert recipes[0]["ingredients"] == ["土豆", "红椒"]
    assert report["accepted_recipe_count"] == 1
    assert report["rejected_recipe_count"] == 1
    assert report["rejected"][0]["chunk_id"] == "bad"


def test_recipe_submission_enriches_canonical_query_fields() -> None:
    recipe, errors = validate_and_normalize_recipe(
        {
            "name": "香炸鸡胸肉",
            "ingredients": ["鸡胸肉", "面包糠"],
            "steps": "鸡胸肉切片腌制，裹上面包糠后炸至金黄熟透。",
        }
    )

    assert errors == []
    assert recipe is not None
    assert recipe["category"] == "炸物"
    assert recipe["cooking_time_minutes"] > 0
    assert recipe["difficulty"] in {"简单", "中等", "困难"}
    assert recipe["calories_per_100g"] > 0
    assert recipe["protein_g_per_100g"] >= 12
    assert recipe["fat_g_per_100g"] > 0
    assert recipe["nutrition_estimated"] is True
    assert "高蛋白" in recipe["tags"]
    assert "油炸" in recipe["tags"]
    assert "cooking_time" not in recipe
    assert "calories" not in recipe
