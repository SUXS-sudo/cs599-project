from __future__ import annotations

import argparse
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
    parser.add_argument("--mysql-batch-size", type=int, default=500)
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
    recipe_rows = document_metadata_to_recipes(metadata)

    print_json(
        "build_summary",
        {
            **build_summary,
            "structured_recipe_count": len(recipe_rows),
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
    from scripts.build_document_index import build_one

    build_args = argparse.Namespace(
        sources=[str(pdf_path)],
        from_chunks=None,
        input_dir=None,
        glob="*.pdf",
        recursive=False,
        batch=False,
        continue_on_error=False,
        skip_existing=False,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        recipe_refine=True,
        ocr=True,
        ocr_force=True,
        ocr_engine=args.ocr_engine,
        ocr_dpi=args.ocr_dpi,
        ocr_max_pages=None,
        ocr_min_extracted_chars=1000,
        ocr_progress_every=args.ocr_progress_every,
        no_ocr_progress=False,
        output_prefix=args.output_prefix,
        chunks_output=None,
        faiss_output=None,
        metadata_output=None,
        preview=0,
    )
    return build_one(build_args)


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


def document_metadata_to_recipes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    recipes = []
    seen_names = set()
    for chunk in metadata.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_metadata = chunk.get("metadata")
        if not isinstance(chunk_metadata, dict):
            chunk_metadata = {}
        name = str(chunk_metadata.get("dish_name") or chunk_metadata.get("title") or "").strip()
        if not name or name in seen_names:
            continue
        text = str(chunk.get("text") or "")
        ingredients_text = str(chunk_metadata.get("ingredients") or "")
        category = str(chunk_metadata.get("category") or "")
        recipes.append(
            {
                "name": name,
                "ingredients": split_ingredients(ingredients_text),
                "category": category,
                "cooking_time": "",
                "difficulty": "",
                "tags": [category] if category else [],
                "calories": 0,
                "suitable_for": [],
                "steps": text,
            }
        )
        seen_names.add(name)
    return recipes


def split_ingredients(text: str) -> list[str]:
    parts = re.split(r"[、，,；;。\s銆锛]+", text.strip())
    result = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"\d+(\.\d+)?\s*(克|g|毫升|ml|个|只|片|根|匙|勺|杯)?", "", cleaned, flags=re.I)
        cleaned = cleaned.strip("：:（）()[] ")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


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
