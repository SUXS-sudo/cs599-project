from __future__ import annotations

import json
import zipfile
from pathlib import Path

import app.services.document_faiss as document_faiss
import app.services.heterogeneous_recipe_pipeline as heterogeneous_pipeline
from app.services.document_chunking import DocumentChunk
from app.services.recipe_chunk_refiner import RefinedRecipeChunk
from app.services.heterogeneous_recipe_pipeline import (
    load_source_records,
    parse_components,
    parse_recipe_record,
    read_jsonl,
    run_heterogeneous_recipe_pipeline,
)
from app.services.recipe_enrichment import (
    estimate_nutrition_per_100g,
    infer_category,
    infer_difficulty,
    is_suspect_ingredient,
    validate_and_normalize_recipe,
)


ARTIFACT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed" / "test_heterogeneous_pipeline"


def test_valid_record_is_optimized_by_llm_by_default() -> None:
    class FakeLLM:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _prompt: str, **_kwargs):
            self.calls += 1
            return json.dumps(
                {
                    "name": "番茄炒蛋",
                    "ingredients": ["番茄300克", "鸡蛋2个"],
                    "seasonings": ["盐适量"],
                    "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。",
                    "category": "家常菜",
                    "tags": ["LLM优化"],
                },
                ensure_ascii=False,
            )

    llm = FakeLLM()
    parsed, errors = parse_recipe_record(
        {
            "name": "番茄炒蛋",
            "ingredients": ["番茄300克", "鸡蛋2个"],
            "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。",
        },
        Path("recipes.json"),
        0,
        llm_client=llm,
    )

    assert errors == []
    assert parsed is not None
    assert parsed["_llm_enriched"] is True
    assert "LLM优化" in parsed["tags"]
    assert llm.calls == 1


def test_pdf_text_is_extracted_once_when_faiss_is_enabled(monkeypatch) -> None:
    test_dir = ARTIFACT_ROOT / "pdf_single_extraction"
    test_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = test_dir / "recipe.pdf"
    pdf_path.touch()
    calls = 0

    def fake_chunk_documents(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [
            DocumentChunk(
                chunk_id="recipe-0000",
                source=str(pdf_path),
                source_type="pdf",
                text=(
                    "菜名：番茄炒蛋\n"
                    "原料：番茄300克、鸡蛋2个\n"
                    "制作方法：鸡蛋炒熟；番茄炒出汁；合炒调味。"
                ),
                start_char=0,
                end_char=49,
            )
        ]

    class FakeIndex:
        def save(self, index_path: Path, metadata_path: Path) -> None:
            index_path.touch()
            metadata_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(heterogeneous_pipeline, "chunk_documents", fake_chunk_documents)
    monkeypatch.setattr(heterogeneous_pipeline, "refine_recipe_chunks", lambda _chunks: [])
    monkeypatch.setattr(document_faiss, "build_document_faiss_index", lambda _chunks: FakeIndex())

    report = run_heterogeneous_recipe_pipeline(
        [pdf_path],
        test_dir / "output",
        build_faiss=True,
        import_mysql=False,
        import_neo4j=False,
    )

    assert calls == 1
    assert report.cleaned_recipe_count == 1
    assert "faiss_index" in report.outputs


def test_faiss_accepts_refined_recipe_chunks_without_source_offsets(monkeypatch) -> None:
    test_dir = ARTIFACT_ROOT / "faiss_refined_chunks"
    test_dir.mkdir(parents=True, exist_ok=True)
    captured_chunks: list[DocumentChunk] = []

    refined = RefinedRecipeChunk(
        chunk_id="recipe-0001",
        source="cookbook.pdf",
        source_type="pdf",
        text="菜名：番茄炒蛋",
        title="番茄炒蛋",
        metadata={"pages": [1]},
    )

    class FakeIndex:
        def save(self, index_path: Path, metadata_path: Path) -> None:
            index_path.touch()
            metadata_path.write_text("{}", encoding="utf-8")

    def fake_build(chunks: list[DocumentChunk]):
        captured_chunks.extend(chunks)
        return FakeIndex()

    monkeypatch.setattr(heterogeneous_pipeline, "refine_recipe_chunks", lambda _chunks: [refined])
    monkeypatch.setattr(document_faiss, "build_document_faiss_index", fake_build)

    result = heterogeneous_pipeline._build_faiss_for_documents(
        [[DocumentChunk("raw-1", "cookbook.pdf", "pdf", "raw", 0, 3)]],
        test_dir,
    )

    assert result is not None
    assert captured_chunks[0].start_char == 0
    assert captured_chunks[0].end_char == len(refined.text)


def test_pipeline_reads_json_and_csv_and_builds_traceable_graph() -> None:
    test_dir = ARTIFACT_ROOT / "json_csv"
    test_dir.mkdir(parents=True, exist_ok=True)
    json_path = test_dir / "recipes.json"
    csv_path = test_dir / "recipes.csv"
    json_path.write_text(
        json.dumps(
            [
                {
                    "name": "番茄炒蛋",
                    "ingredients": ["番茄300克", "鸡蛋2个"],
                    "seasonings": ["盐适量"],
                    "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。",
                    "tags": ["家常菜"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    csv_path.write_text(
        "菜名,食材,调料,做法\n番茄炒蛋,番茄300克、鸡蛋2个,盐适量,鸡蛋炒熟；番茄炒出汁；合炒调味。\n",
        encoding="utf-8-sig",
    )

    report = run_heterogeneous_recipe_pipeline([json_path, csv_path], test_dir / "output", build_faiss=False, import_mysql=False, import_neo4j=False)

    assert report.raw_record_count == 2
    assert report.cleaned_recipe_count == 1
    assert report.duplicate_count == 1
    assert report.graph_node_labels["Recipe"] == 1
    assert report.graph_node_labels["SourceRecord"] == 2
    assert report.graph_node_labels["IngredientUse"] == 2
    assert report.graph_node_labels["RecipeStep"] == 3
    assert report.graph_relationship_count > report.graph_node_count
    assert len(read_jsonl(test_dir / "output" / "graph_nodes.jsonl")) == report.graph_node_count


def test_jsonl_pdf_recipe_chunk_is_adapted_to_recipe_schema() -> None:
    test_dir = ARTIFACT_ROOT / "pdf_jsonl"
    test_dir.mkdir(parents=True, exist_ok=True)
    path = test_dir / "pdf_chunks.jsonl"
    row = {
        "chunk_id": "pdf-recipe-1",
        "source": "cookbook.pdf",
        "source_type": "pdf",
        "text": "菜名：清炒西兰花\n原料：西兰花300克。\n调料：盐适量。\n制作方法：西兰花洗净；热锅快炒；加盐出锅。",
        "metadata": {"dish_name": "清炒西兰花", "ingredients": "西兰花300克。", "seasonings": "盐适量。"},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    report = run_heterogeneous_recipe_pipeline([path], test_dir / "output", build_faiss=False, import_mysql=False, import_neo4j=False)
    recipes = json.loads((test_dir / "output" / "recipes_clean.json").read_text(encoding="utf-8"))

    assert report.cleaned_recipe_count == 1
    assert recipes[0]["name"] == "清炒西兰花"
    assert recipes[0]["ingredients"] == ["西兰花"]
    assert recipes[0]["seasonings"] == ["盐"]
    assert recipes[0]["provenance"][0]["record_id"] == "pdf-recipe-1"


def test_component_parser_preserves_raw_quantity() -> None:
    details = parse_components("鸡胸肉200克、西兰花300克、盐适量")

    assert details == [
        {"name": "鸡胸肉", "raw": "鸡胸肉200克", "quantity": "200克"},
        {"name": "西兰花", "raw": "西兰花300克", "quantity": "300克"},
        {"name": "盐", "raw": "盐适量", "quantity": "适量"},
    ]


def test_xlsx_reader_uses_standard_library_without_optional_dependency() -> None:
    test_dir = ARTIFACT_ROOT / "xlsx"
    test_dir.mkdir(parents=True, exist_ok=True)
    path = test_dir / "recipes.xlsx"
    worksheet = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
      <row r="1"><c r="A1" t="inlineStr"><is><t>菜名</t></is></c><c r="B1" t="inlineStr"><is><t>食材</t></is></c><c r="C1" t="inlineStr"><is><t>做法</t></is></c></row>
      <row r="2"><c r="A2" t="inlineStr"><is><t>清炒西兰花</t></is></c><c r="B2" t="inlineStr"><is><t>西兰花300克</t></is></c><c r="C2" t="inlineStr"><is><t>洗净；快炒；调味。</t></is></c></row>
    </sheetData></worksheet>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)

    rows = load_source_records(path)

    assert rows == [{"菜名": "清炒西兰花", "食材": "西兰花300克", "做法": "洗净；快炒；调味。"}]


# --- Tests for strict validation and enrichment ---


def test_strict_validation_rejects_invalid_dish_name() -> None:
    candidate = {"name": "将鸡蛋打入碗中", "ingredients": ["鸡蛋"], "steps": "鸡蛋打入碗中搅拌均匀；下锅炒熟。"}
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is None
    assert "dish_name_looks_like_step" in errors


def test_strict_validation_rejects_no_ingredients() -> None:
    candidate = {"name": "番茄炒蛋", "ingredients": [], "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。"}
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is None
    assert "no_valid_ingredients" in errors


def test_strict_validation_rejects_short_steps() -> None:
    candidate = {"name": "番茄炒蛋", "ingredients": ["番茄", "鸡蛋"], "steps": "炒熟。"}
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is None
    assert "missing_or_short_steps" in errors


def test_strict_validation_rejects_suspect_ingredient() -> None:
    # "切丝" is filtered out by normalize_ingredient_name -> no valid ingredients left
    candidate = {"name": "番茄炒蛋", "ingredients": ["切丝", "洗净"], "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。"}
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is None
    assert "no_valid_ingredients" in errors


def test_strict_validation_accepts_valid_recipe() -> None:
    candidate = {
        "name": "番茄炒蛋",
        "ingredients": ["番茄300克", "鸡蛋2个"],
        "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。",
    }
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is not None
    assert errors == []
    assert result["name"] == "番茄炒蛋"
    assert "番茄" in result["ingredients"]
    assert "鸡蛋" in result["ingredients"]


def test_nutrition_inference_populates_fields() -> None:
    candidate = {
        "name": "牛肉炒西兰花",
        "ingredients": ["牛肉200克", "西兰花300克"],
        "steps": "牛肉切片腌制；西兰花焯水；热锅快炒；调味出锅。",
    }
    result, errors = validate_and_normalize_recipe(candidate)
    assert result is not None
    assert result["calories_per_100g"] > 0
    assert result["protein_g_per_100g"] > 0
    assert result["fat_g_per_100g"] >= 0
    assert result["nutrition_estimated"] is True


def test_category_inference_from_keywords() -> None:
    assert infer_category("番茄鸡蛋汤") == "汤羹"
    assert infer_category("皮蛋瘦肉粥") == "粥"
    assert infer_category("凉拌黄瓜") == "凉菜"
    assert infer_category("清蒸鲈鱼") == "蒸菜"
    assert infer_category("红烧肉炖土豆") == "炖菜"
    assert infer_category("烤鸡翅") == "烤箱菜"
    assert infer_category("炸酱面") == "炸物"  # "炸" is checked before "面"
    assert infer_category("干炸里脊") == "炸物"
    assert infer_category("阳春面") == "面食"
    assert infer_category("家常豆腐") == "家常菜"


def test_difficulty_inference() -> None:
    assert infer_difficulty("简单的炒菜步骤", 15) == "简单"
    # 4 action words + 40 min -> 中等
    assert infer_difficulty("切丝焯水腌制翻炒调味装盘", 40) == "中等"
    # 7+ action words + 60 min -> 困难
    assert infer_difficulty("切丝焯水腌制炸蒸炖翻炒调汁勾芡" * 3, 70) == "困难"


def test_suspect_ingredient_detection() -> None:
    assert is_suspect_ingredient("") is True
    assert is_suspect_ingredient("切丝") is True
    assert is_suspect_ingredient("洗净") is True
    assert is_suspect_ingredient("丝") is True
    assert is_suspect_ingredient("番茄") is False
    assert is_suspect_ingredient("鸡胸肉") is False


def test_pipeline_rejects_invalid_records_and_tracks_errors() -> None:
    test_dir = ARTIFACT_ROOT / "strict_validation"
    test_dir.mkdir(parents=True, exist_ok=True)
    path = test_dir / "bad_recipes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "好的菜谱",
                    "ingredients": ["番茄300克", "鸡蛋2个"],
                    "steps": "鸡蛋炒熟；番茄炒出汁；合炒调味。",
                },
                {
                    "name": "将鸡蛋打入碗中",
                    "ingredients": ["鸡蛋"],
                    "steps": "鸡蛋打入碗中搅拌均匀；下锅炒熟。",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_heterogeneous_recipe_pipeline([path], test_dir / "output", build_faiss=False, import_mysql=False, import_neo4j=False)

    assert report.raw_record_count == 2
    assert report.parsed_record_count == 1
    assert report.rejected_count == 1
    assert len(report.rejected_records) == 1
    assert "dish_name_looks_like_step" in report.rejected_records[0]["errors"]


def test_pipeline_enriches_recipes_with_nutrition() -> None:
    test_dir = ARTIFACT_ROOT / "enrichment"
    test_dir.mkdir(parents=True, exist_ok=True)
    path = test_dir / "recipes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "红烧牛肉",
                    "ingredients": ["牛肉500克", "土豆200克"],
                    "steps": "牛肉切块焯水；土豆去皮切块；锅中加水炖煮；调味收汁。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_heterogeneous_recipe_pipeline([path], test_dir / "output", build_faiss=False, import_mysql=False, import_neo4j=False)
    recipes = json.loads((test_dir / "output" / "recipes_clean.json").read_text(encoding="utf-8"))

    assert report.cleaned_recipe_count == 1
    recipe = recipes[0]
    assert recipe["calories_per_100g"] > 0
    assert recipe["protein_g_per_100g"] > 0
    assert recipe["category"] == "炖菜"  # inferred from "炖"
    assert recipe["difficulty"] != "未知"  # inferred from steps
    assert recipe["cooking_time_minutes"] > 0
