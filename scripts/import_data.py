"""SmartRecipe 数据导入工具。

子命令:
  mysql     导入菜谱到 MySQL
  neo4j     导入菜谱到 Neo4j 图谱
  chunks    导入文档 RAG 分块到 MySQL
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ── mysql ──────────────────────────────────────────────────────────────────

def cmd_mysql(args: argparse.Namespace) -> int:
    from app.services.mysql_store import MySQLConfig, MySQLStore, load_recipe_json

    data_path = Path(args.data_path)
    recipes = load_recipe_json(data_path)
    unique_ingredients = {
        ingredient
        for recipe in recipes
        for ingredient in recipe.get("ingredients", [])
        if ingredient.strip()
    }
    unique_tags = {
        tag
        for recipe in recipes
        for tag in recipe.get("tags", [])
        if tag.strip()
    }
    unique_targets = {
        target
        for recipe in recipes
        for target in recipe.get("suitable_for", [])
        if target.strip()
    }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "data_path": str(data_path),
                    "recipes": len(recipes),
                    "unique_ingredients": len(unique_ingredients),
                    "unique_tags": len(unique_tags),
                    "unique_suitable_for": len(unique_targets),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    try:
        store.ensure_schema()
        if args.reset:
            store.reset_recipe_tables()
        counts = store.import_recipes(recipes)
        stats = store.stats()
    except Exception as exc:
        print(f"mysql_recipe_import_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_import_target={config.host}:{config.port}/{config.database}")
    print("imported:")
    for key, value in counts.items():
        print(f"- {key}={value}")
    print("database_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    return 0


# ── neo4j ──────────────────────────────────────────────────────────────────

def cmd_neo4j(args: argparse.Namespace) -> int:
    from app.services.neo4j_store import Neo4jConfig, Neo4jStore, dry_run_graph_counts, load_recipe_json

    data_path = Path(args.data_path)
    recipes = load_recipe_json(data_path)
    counts = dry_run_graph_counts(recipes)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "data_path": str(data_path),
                    **counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    config = Neo4jConfig.from_env()
    store = Neo4jStore(config)
    try:
        if args.reset:
            store.clear_graph()
        imported = store.import_recipes(recipes)
        stats = store.stats()
    except Exception as exc:
        print(f"neo4j_recipe_import_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"neo4j_import_target={config.uri}")
    if config.database:
        print(f"neo4j_database={config.database}")
    print("imported:")
    for key, value in imported.items():
        print(f"- {key}={value}")
    print("graph_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    return 0


# ── chunks ─────────────────────────────────────────────────────────────────

def cmd_chunks(args: argparse.Namespace) -> int:
    if args.metadata_dir:
        return _run_batch(args)

    summary = _build_import_summary(Path(args.metadata), Path(args.index), args.index_name)
    if args.dry_run:
        print(json.dumps({"dry_run": True, **summary}, ensure_ascii=False, indent=2))
        return 0

    counts, stats, target = _import_one(summary, reset=args.reset, batch_size=args.batch_size)
    print(f"mysql_import_target={target}")
    print(json.dumps({"imported": counts, "summary": summary, "database_stats": stats}, ensure_ascii=False, indent=2))
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    metadata_paths = sorted(path for path in Path(args.metadata_dir).glob(args.metadata_glob) if path.is_file())
    if not metadata_paths:
        raise ValueError(f"No metadata files found: {args.metadata_dir}/{args.metadata_glob}")

    if args.dry_run:
        results = []
        for metadata_path in metadata_paths:
            index_path = _infer_index_path(metadata_path)
            if args.skip_missing_index and not index_path.exists():
                results.append({"metadata_path": str(metadata_path), "status": "skipped", "reason": "missing index"})
                continue
            results.append({"status": "ok", **_build_import_summary(metadata_path, index_path, None)})
        print(json.dumps({"dry_run": True, "batch_total": len(results), "results": results}, ensure_ascii=False, indent=2))
        return 0

    from app.services.mysql_store import MySQLConfig, MySQLStore

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    store.ensure_schema()
    if args.reset:
        store.reset_document_tables()

    results = []
    for position, metadata_path in enumerate(metadata_paths, start=1):
        index_path = _infer_index_path(metadata_path)
        if args.skip_missing_index and not index_path.exists():
            result = {"metadata_path": str(metadata_path), "status": "skipped", "reason": "missing index"}
            results.append(result)
            print(json.dumps({"batch_progress": f"{position}/{len(metadata_paths)}", **result}, ensure_ascii=False))
            continue
        try:
            summary = _build_import_summary(metadata_path, index_path, None)
            counts = store.import_document_index(
                index_name=str(summary["index_name"]),
                index_path=str(summary["index_path"]),
                metadata_path=str(summary["metadata_path"]),
                metadata=_load_metadata(metadata_path),
                batch_size=args.batch_size,
            )
            result = {"status": "ok", "imported": counts, "summary": summary}
        except Exception as exc:
            result = {"metadata_path": str(metadata_path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if not args.continue_on_error:
                print(json.dumps({"batch_progress": f"{position}/{len(metadata_paths)}", **result}, ensure_ascii=False))
                raise
        results.append(result)
        print(json.dumps({"batch_progress": f"{position}/{len(metadata_paths)}", **result}, ensure_ascii=False))

    stats = store.stats()
    ok_count = sum(1 for row in results if row["status"] == "ok")
    skipped_count = sum(1 for row in results if row["status"] == "skipped")
    failed_count = sum(1 for row in results if row["status"] == "failed")
    print(f"mysql_import_target={config.host}:{config.port}/{config.database}")
    print(
        json.dumps(
            {
                "batch_total": len(results),
                "ok": ok_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "database_stats": stats,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed_count else 0


def _build_import_summary(metadata_path: Path, index_path: Path, index_name: str | None) -> dict[str, Any]:
    metadata = _load_metadata(metadata_path)
    chunks = metadata.get("chunks") or []
    if not isinstance(chunks, list) or not chunks:
        raise ValueError(f"No chunks found in metadata: {metadata_path}")
    resolved_index_name = index_name or index_path.stem
    return {
        "index_name": resolved_index_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "embedding_backend": metadata.get("embedding_backend"),
        "index_type": metadata.get("index_type"),
        "chunk_count": len(chunks),
    }


def _import_one(summary: dict[str, Any], reset: bool, batch_size: int):
    from app.services.mysql_store import MySQLConfig, MySQLStore

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    try:
        store.ensure_schema()
        if reset:
            store.reset_document_tables()
        counts = store.import_document_index(
            index_name=str(summary["index_name"]),
            index_path=str(summary["index_path"]),
            metadata_path=str(summary["metadata_path"]),
            metadata=_load_metadata(Path(str(summary["metadata_path"]))),
            batch_size=batch_size,
        )
        stats = store.stats()
    except Exception as exc:
        print(f"mysql_document_import_failed={type(exc).__name__}: {exc}")
        raise
    return counts, stats, f"{config.host}:{config.port}/{config.database}"


def _infer_index_path(metadata_path: Path) -> Path:
    name = metadata_path.name
    if name.endswith("_metadata.json"):
        return metadata_path.with_name(name[: -len("_metadata.json")] + ".index")
    return metadata_path.with_suffix(".index")


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return data


# ── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SmartRecipe 数据导入工具。")
    sub = parser.add_subparsers(dest="command", required=True)

    # mysql
    p_mysql = sub.add_parser("mysql", help="导入菜谱到 MySQL。")
    p_mysql.add_argument("--data-path", default=str(ROOT_DIR / "data" / "recipes.json"), help="菜谱 JSON 路径。")
    p_mysql.add_argument("--reset", action="store_true", help="导入前清空菜谱相关表。")
    p_mysql.add_argument("--dry-run", action="store_true", help="只验证 JSON，不连接 MySQL。")
    p_mysql.set_defaults(func=cmd_mysql)

    # neo4j
    p_neo4j = sub.add_parser("neo4j", help="导入菜谱到 Neo4j 图谱。")
    p_neo4j.add_argument("--data-path", default=str(ROOT_DIR / "data" / "recipes.json"), help="菜谱 JSON 路径。")
    p_neo4j.add_argument("--dry-run", action="store_true", help="只打印图谱映射计数，不连接 Neo4j。")
    p_neo4j.add_argument("--reset", action="store_true", help="导入前清空 Neo4j 所有节点和关系。")
    p_neo4j.set_defaults(func=cmd_neo4j)

    # chunks
    p_chunks = sub.add_parser("chunks", help="导入文档 RAG 分块到 MySQL。")
    p_chunks.add_argument("--metadata", default=str(ROOT_DIR / "data" / "processed" / "new_pdf_recipe_metadata.json"), help="FAISS metadata JSON 路径。")
    p_chunks.add_argument("--metadata-dir", default=None, help="批量模式：从此目录导入 metadata 文件。")
    p_chunks.add_argument("--metadata-glob", default="*_recipe_metadata.json", help="批量模式 glob。")
    p_chunks.add_argument("--index", default=str(ROOT_DIR / "data" / "processed" / "new_pdf_recipe.index"), help="FAISS 索引文件路径。")
    p_chunks.add_argument("--index-name", default=None, help="逻辑索引名。")
    p_chunks.add_argument("--reset", action="store_true", help="导入前清空 document_indexes 和 document_chunks。")
    p_chunks.add_argument("--batch-size", type=int, default=500, help="MySQL executemany 批次大小。")
    p_chunks.add_argument("--continue-on-error", action="store_true", help="批量模式：单文件失败时继续。")
    p_chunks.add_argument("--skip-missing-index", action="store_true", help="批量模式：跳过缺少索引的 metadata。")
    p_chunks.add_argument("--dry-run", action="store_true", help="只验证 metadata，不连接 MySQL。")
    p_chunks.set_defaults(func=cmd_chunks)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
