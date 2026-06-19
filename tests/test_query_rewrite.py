from __future__ import annotations

from app.services.query_rewrite import rewrite_recipe_query


def test_rewrite_recipe_query_expands_method_query() -> None:
    result = rewrite_recipe_query("老醋花生米怎么做")

    assert result.intent == "method"
    assert "制作方法" in result.expanded_query
    assert "原料" in result.expanded_query
    assert "老醋花生米怎么做" in result.expanded_query


def test_rewrite_recipe_query_expands_ingredient_query() -> None:
    result = rewrite_recipe_query("花生米可以做什么")

    assert result.intent in {"ingredient", "recommend"}
    assert "花生米可以做什么" in result.expanded_query
    assert any(term in result.expanded_query for term in ("原料", "制作方法", "菜名"))
