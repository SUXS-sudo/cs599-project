from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the PDF recipe pipeline: OCR/chunk pdf/1.pdf, build FAISS, "
            "reset/import MySQL document tables, and optionally import parsed PDF recipes "
            "into MySQL recipe tables and Neo4j."
        )
    )
    parser.add_argument("--pdf", default=str(ROOT_DIR / "pdf" / "1.pdf"), help="Source PDF path.")
    parser.add_argument("--output-prefix", default="1", help="Output prefix under data/processed.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--ocr-engine", default="rapidocr", choices=["auto", "rapidocr", "tesseract"])
    parser.add_argument("--ocr-dpi", type=int, default=160)
    parser.add_argument("--ocr-progress-every", type=int, default=5)
    parser.add_argument("--ocr-max-pages", type=int, default=None, help="Only process the first N pages (useful for smoke tests).")
    parser.add_argument("--mysql-batch-size", type=int, default=500)
    parser.add_argument(
        "--llm-structure",
        action="store_true",
        help="Ask the configured text LLM to return fixed recipe JSON before rule validation.",
    )
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail instead of using deterministic extraction when --llm-structure is unavailable or invalid.",
    )
    parser.add_argument(
        "--skip-structured-recipes",
        action="store_true",
        help="Only import document_indexes/document_chunks; do not fill recipes or Neo4j.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build files and print import counts, but do not reset or write MySQL/Neo4j.",
    )
    args = parser.parse_args()

    from app.services.llm_client import load_dotenv

    load_dotenv(override=True)

    pdf_path = resolve_path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print_step("1/4", f"building FAISS document index from {pdf_path}")
    build_summary = build_pdf_index(args, pdf_path)
    metadata_path = Path(build_summary["metadata_output"])
    index_path = Path(build_summary["faiss_output"])
    metadata = load_metadata(metadata_path)
    llm_cache_path = ROOT_DIR / "data" / "processed" / f"{args.output_prefix}_recipe_llm_cache.json"
    recipe_rows, quality_report = document_metadata_to_recipes(
        metadata,
        use_llm=args.llm_structure,
        require_llm=args.require_llm,
        llm_cache_path=llm_cache_path,
    )
    quality_report_path = ROOT_DIR / "data" / "processed" / f"{args.output_prefix}_recipe_quality.json"
    quality_report_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_json(
        "build_summary",
        {
            **build_summary,
            "structured_recipe_count": len(recipe_rows),
            "quality_report": str(quality_report_path),
            "rejected_recipe_count": quality_report["rejected_recipe_count"],
        },
    )

    if args.dry_run:
        print_step("dry-run", "skipping MySQL/Neo4j writes")
        print_json(
            "dry_run_import_plan",
            {
                "document_indexes": 1,
                "document_chunks": len(metadata.get("chunks") or []),
                "recipes": 0 if args.skip_structured_recipes else len(recipe_rows),
                "neo4j_recipes": 0 if args.skip_structured_recipes else len(recipe_rows),
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
            },
        )
        return 0

    print_step("2/4", "resetting and importing MySQL document tables")
    document_counts, mysql_stats = import_documents_to_mysql(
        metadata=metadata,
        metadata_path=metadata_path,
        index_path=index_path,
        batch_size=args.mysql_batch_size,
    )
    print_json("mysql_document_import", {"imported": document_counts, "stats": mysql_stats})

    if args.skip_structured_recipes:
        print_step("3/4", "skipping structured MySQL recipes and Neo4j import")
        return 0

    print_step("3/4", "resetting and importing MySQL structured recipe tables")
    recipe_counts, mysql_stats = import_recipes_to_mysql(recipe_rows)
    print_json("mysql_recipe_import", {"imported": recipe_counts, "stats": mysql_stats})

    print_step("4/4", "resetting and importing Neo4j recipe graph")
    neo4j_counts, neo4j_stats = import_recipes_to_neo4j(recipe_rows)
    print_json("neo4j_recipe_import", {"imported": neo4j_counts, "stats": neo4j_stats})
    return 0


def build_pdf_index(args: argparse.Namespace, pdf_path: Path) -> dict[str, Any]:
    from app.services.document_chunking import ChunkingConfig, DocumentChunk, OcrConfig, chunk_documents, write_chunks_jsonl
    from app.services.document_faiss import build_document_faiss_index
    from app.services.recipe_chunk_refiner import refine_recipe_chunks, write_recipe_chunks_jsonl

    processed_dir = ROOT_DIR / "data" / "processed"
    stem = sanitize_output_stem(args.output_prefix)
    if not stem.endswith("_recipe"):
        stem = f"{stem}_recipe"
    chunks_path = processed_dir / f"{stem}_chunks.jsonl"
    index_path = processed_dir / f"{stem}.index"
    metadata_path = processed_dir / f"{stem}_metadata.json"

    raw_chunks = chunk_documents(
        [str(pdf_path)],
        config=ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        ocr_config=OcrConfig(
            enabled=True,
            engine=args.ocr_engine,
            dpi=args.ocr_dpi,
            max_pages=args.ocr_max_pages,
            min_extracted_chars=1000,
            force=True,
            show_progress=True,
            progress_every=args.ocr_progress_every,
        ),
    )
    refined = refine_recipe_chunks([chunk.to_dict() for chunk in raw_chunks])
    if refined:
        write_recipe_chunks_jsonl(refined, chunks_path)
        final_chunks = []
        for chunk in refined:
            row = chunk.to_dict()
            text = str(row["text"])
            metadata = dict(row.get("metadata") or {})
            metadata["title"] = str(row.get("title") or metadata.get("dish_name") or "")
            final_chunks.append(
                DocumentChunk(
                    chunk_id=str(row["chunk_id"]),
                    source=str(row.get("source") or pdf_path),
                    source_type=str(row.get("source_type") or "pdf"),
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    metadata=metadata,
                )
            )
        recipe_refine_fallback = False
    else:
        print("[rebuild_pdf1_pipeline] recipe refinement produced 0 chunks; indexing raw OCR chunks", file=sys.stderr)
        write_chunks_jsonl(raw_chunks, chunks_path)
        final_chunks = raw_chunks
        recipe_refine_fallback = True
    if not final_chunks:
        raise ValueError("OCR and chunking produced no document chunks")

    index = build_document_faiss_index(final_chunks)
    index.save(index_path, metadata_path)
    return {
        "source_count": 1,
        "raw_chunk_count": len(raw_chunks),
        "final_chunk_count": len(final_chunks),
        "recipe_refine": True,
        "recipe_refine_fallback": recipe_refine_fallback,
        "embedding_backend": index.embedding_backend,
        "embedding_errors": list(getattr(index.embedding_provider, "errors", [])),
        "index_type": index.index_type,
        "chunks_output": str(chunks_path),
        "faiss_output": str(index_path),
        "metadata_output": str(metadata_path),
    }


def sanitize_output_stem(value: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip()).strip("._-")
    return stem or "document"


def import_documents_to_mysql(
    metadata: dict[str, Any],
    metadata_path: Path,
    index_path: Path,
    batch_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    from app.services.mysql_store import MySQLConfig, MySQLStore

    store = MySQLStore(MySQLConfig.from_env())
    store.ensure_schema()
    store.reset_document_tables()
    counts = store.import_document_index(
        index_name=index_path.stem,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        metadata=metadata,
        batch_size=batch_size,
    )
    return counts, store.stats()


def import_recipes_to_mysql(recipes: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    from app.services.mysql_store import MySQLConfig, MySQLStore

    store = MySQLStore(MySQLConfig.from_env())
    store.ensure_schema()
    store.reset_recipe_tables()
    counts = store.import_recipes(recipes)
    return counts, store.stats()


def import_recipes_to_neo4j(recipes: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    from app.services.neo4j_store import Neo4jConfig, Neo4jStore

    store = Neo4jStore(Neo4jConfig.from_env())
    store.clear_graph()
    counts = store.import_recipes(recipes)
    return counts, store.stats()


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

NUTRITION_REFERENCE = {
    "鸡胸肉": (165, 31.0, 3.6), "鸡肉": (190, 27.0, 7.0), "牛肉": (250, 26.0, 15.0),
    "猪肉": (290, 24.0, 21.0), "五花肉": (518, 9.0, 53.0), "鱼": (120, 22.0, 3.0),
    "虾": (99, 24.0, 0.3), "鸡蛋": (143, 13.0, 10.0), "豆腐": (84, 8.0, 5.0),
    "花生": (567, 26.0, 49.0), "土豆": (77, 2.0, 0.1), "山药": (57, 1.9, 0.2),
    "红薯": (86, 1.6, 0.1), "米": (130, 2.7, 0.3), "面": (138, 4.5, 2.1),
    "燕麦": (389, 17.0, 7.0), "西兰花": (34, 2.8, 0.4), "黄瓜": (15, 0.7, 0.1),
    "番茄": (18, 0.9, 0.2), "木耳": (25, 1.5, 0.2), "银耳": (36, 1.4, 0.2),
}


def document_metadata_to_recipes(
    metadata: dict[str, Any],
    use_llm: bool = False,
    require_llm: bool = False,
    llm_cache_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    llm_client = build_structuring_llm(use_llm, require_llm)
    llm_cache = load_json_object(llm_cache_path) if llm_cache_path else {}
    recipes: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    llm_calls = 0
    llm_cache_hits = 0

    for chunk in metadata.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_metadata = chunk.get("metadata")
        if not isinstance(chunk_metadata, dict):
            chunk_metadata = {}
        text = str(chunk.get("text") or "").strip()
        candidate = deterministic_recipe_from_chunk(text, chunk_metadata)
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if llm_client is not None:
            cached = llm_cache.get(cache_key)
            if isinstance(cached, dict):
                candidate = cached
                llm_cache_hits += 1
            else:
                llm_calls += 1
                structured = structure_recipe_with_llm(llm_client, text)
                if structured is not None:
                    candidate = structured
                    llm_cache[cache_key] = structured
                    if llm_cache_path:
                        write_json_object(llm_cache_path, llm_cache)
                elif require_llm:
                    raise RuntimeError(f"LLM structured extraction failed for chunk {chunk.get('chunk_id', '')}")

        normalized, errors = validate_and_normalize_recipe(candidate)
        if errors:
            rejected.append(
                {
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "dish_name": str(candidate.get("name") or ""),
                    "errors": errors,
                    "raw_ingredients": candidate.get("ingredients", []),
                }
            )
            continue
        assert normalized is not None
        if normalized["name"] in seen_names:
            rejected.append(
                {
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "dish_name": normalized["name"],
                    "errors": ["duplicate_dish_name"],
                }
            )
            continue
        recipes.append(normalized)
        seen_names.add(normalized["name"])

    quality_report = {
        "source_chunk_count": len(metadata.get("chunks") or []),
        "accepted_recipe_count": len(recipes),
        "rejected_recipe_count": len(rejected),
        "llm_enabled": llm_client is not None,
        "llm_calls": llm_calls,
        "llm_cache_hits": llm_cache_hits,
        "rejected": rejected,
    }
    return recipes, quality_report


def deterministic_recipe_from_chunk(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    name = str(metadata.get("dish_name") or metadata.get("title") or extract_field(text, "菜名", "原料")).strip()
    ingredients_text = str(metadata.get("ingredients") or extract_field(text, "原料", "调料"))
    category = str(metadata.get("category") or "").strip()
    steps = extract_field(text, "制作方法", "小提示") or extract_field(text, "制作方法", "来源页码")
    return {
        "name": name,
        "ingredients": split_ingredients(ingredients_text),
        "category": category,
        "cooking_time_minutes": 0,
        "difficulty": "",
        "tags": [category] if category else [],
        "calories_per_100g": 0,
        "protein_g_per_100g": 0,
        "fat_g_per_100g": 0,
        "nutrition_estimated": True,
        "suitable_for": [],
        "steps": steps.strip(),
    }


def extract_field(text: str, start_marker: str, end_marker: str) -> str:
    pattern = re.compile(
        rf"(?:^|\n){re.escape(start_marker)}[：:]?\s*(.*?)(?=\n{re.escape(end_marker)}[：:]?|$)",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


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


def normalize_ingredient_name(value: str) -> str:
    cleaned = value.strip(" ：:（）()[]【】。；;，,、")
    cleaned = re.sub(r"^(?:原料|主料|辅料)[：:]?", "", cleaned)
    cleaned = QUANTITY_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"(?:各|共)?(?:适量|少许|若干)$", "", cleaned)
    cleaned = re.sub(r"各$", "", cleaned)
    return cleaned.strip(" ：:（）()[]【】。；;，,、")


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
    return not bool(re.search(r"[\u4e00-\u9fffA-Za-z]", value))


def validate_and_normalize_recipe(candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    name = str(candidate.get("name") or "").strip()
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
    steps_value = candidate.get("steps")
    if isinstance(steps_value, list):
        steps = "；".join(str(item.get("text") if isinstance(item, dict) else item).strip() for item in steps_value)
    else:
        steps = str(steps_value or "").strip()

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
    """Produce the canonical fields shared by MySQL and Neo4j before import."""
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

    tags = unique_strings(candidate.get("tags"), fallback=[])
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
    suitable_for = unique_strings(candidate.get("suitable_for"), fallback=[])
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


def estimate_nutrition_per_100g(ingredients: list[str], text: str, category: str) -> tuple[float, float, float]:
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


def unique_strings(value: Any, fallback: list[str]) -> list[str]:
    items = value if isinstance(value, list) else fallback
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def build_structuring_llm(use_llm: bool, require_llm: bool):
    if not use_llm:
        return None
    from app.services.llm_client import LLMClient

    client = LLMClient()
    if client.available:
        return client
    if require_llm:
        raise RuntimeError("--require-llm was set, but BASE_URL/API_KEY/MODEL is incomplete")
    print("[structure] LLM unavailable; falling back to deterministic extraction", file=sys.stderr)
    return None


def structure_recipe_with_llm(llm_client, text: str) -> dict[str, Any] | None:
    prompt = (
        "你是菜谱OCR结构化提取器。菜名、食材和步骤只能使用输入证据；"
        "分类、时间、难度和每100克营养值可根据菜名、食材与做法做保守估算，并把nutrition_estimated设为true。"
        "把制作动作放入steps，绝不能把切丝、洗净、倒入等动作当作ingredient。"
        "只输出一个JSON对象，不要markdown。固定字段为："
        '{"name":"","ingredients":[{"name":"","amount":"","unit":""}],'
        '"category":"","cooking_time_minutes":0,"difficulty":"","tags":[],'
        '"calories_per_100g":0,"protein_g_per_100g":0,"fat_g_per_100g":0,"nutrition_estimated":true,'
        '"suitable_for":[],"steps":["..."]}。'
        "热量单位必须是kcal/100g，蛋白质和脂肪单位必须是g/100g。\n\nOCR菜谱块：\n" + text[:6000]
    )
    raw = llm_client.generate(prompt, max_tokens=1200, timeout=60)
    if not raw:
        return None
    return parse_json_object(raw)


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_json_object(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def print_step(label: str, message: str) -> None:
    print(f"\n[{label}] {message}", flush=True)


def print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}=")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
