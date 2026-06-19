from __future__ import annotations

from app.services.recipe_chunk_refiner import refine_recipe_chunks


def test_refine_recipe_chunks_splits_two_recipes_and_assigns_tips() -> None:
    raw_chunks = [
        {
            "chunk_id": "sample-0000",
            "source": "pdf\\sample.pdf",
            "source_type": "pdf",
            "start_char": 0,
            "end_char": 500,
            "text": """[page 14]
CMenu
原料
老醋花生米
花生米300克，青椒、洋葱50克。
调料
香醋80克，白糖40克，盐、油适量。
制作方法
锅置火上，倒油烧至四成热，放入花生米炸至呈金黄时捞出。
小提示
老醋花生米
此菜具有防癌抗癌、通乳、增强记忆等功效。
挂霜花生
此菜具有防癌抗癌、抗衰老、滋血通乳、增强记忆等功效。
原料
挂霜花生
花生米200克。
调料
白糖35克，淀粉、油适量。
制作方法
锅置火上，放油烧热，放入花生米炸熟，捞出晾凉。""",
        }
    ]

    chunks = refine_recipe_chunks(raw_chunks)

    assert [chunk.title for chunk in chunks] == ["老醋花生米", "挂霜花生"]
    assert "香醋80克" in chunks[0].text
    assert "抗衰老" in chunks[1].text
    assert chunks[0].metadata["dish_name"] == "老醋花生米"
    assert chunks[0].metadata["pages"] == [14]
