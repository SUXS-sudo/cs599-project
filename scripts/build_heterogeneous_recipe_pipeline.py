from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.data_engineering_agent import RecipeParsingAgent
from src.services.llm_client import LLMClient
from src.services.heterogeneous_recipe_pipeline import import_graph_manifest_to_neo4j


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse heterogeneous recipe files, clean records and build graph manifests.")
    parser.add_argument("sources", nargs="*", help="JSON/JSONL/CSV/TSV/XLSX/PDF/DOCX/HTML/TXT/MD sources.")
    parser.add_argument("--input-dir", help="Discover supported files under a directory.")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "processed" / "data_engineering"))
    parser.add_argument("--no-llm", action="store_false", dest="enable_llm", help="Skip default LLM optimization after document chunking.")
    parser.add_argument("--no-faiss", action="store_false", dest="build_faiss", help="Skip FAISS index building for document sources.")
    parser.add_argument("--no-mysql", action="store_false", dest="import_mysql", help="Skip MySQL import.")
    parser.add_argument("--reset-mysql", action="store_true", help="Truncate MySQL recipe tables before import.")
    parser.add_argument("--no-neo4j", action="store_false", dest="import_neo4j", help="Skip Neo4j import.")
    parser.add_argument("--reset-neo4j", action="store_true", help="Clear Neo4j before import.")
    parser.add_argument("--manual-minutes-per-record", type=float, default=2.0, help="Assumption used only for estimated saved time.")
    args = parser.parse_args()

    sources = [Path(value) for value in args.sources]
    if args.input_dir:
        root = Path(args.input_dir)
        iterator = root.rglob("*") if args.recursive else root.glob("*")
        supported = {".json", ".jsonl", ".ndjson", ".csv", ".tsv", ".xlsx", ".xlsm", ".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
        sources.extend(sorted(path for path in iterator if path.is_file() and path.suffix.lower() in supported))
    sources = list(dict.fromkeys(path.resolve() for path in sources))
    if not sources:
        parser.error("Provide sources or --input-dir.")
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        parser.error(f"Missing source files: {missing}")

    agent = RecipeParsingAgent(LLMClient() if args.enable_llm else None, enable_llm=args.enable_llm)
    report = agent.run_pipeline(
        sources, Path(args.output_dir),
        manual_minutes_per_record=args.manual_minutes_per_record,
        build_faiss=args.build_faiss,
        import_mysql=args.import_mysql,
        reset_mysql=args.reset_mysql,
        import_neo4j=args.import_neo4j,
        reset_neo4j=args.reset_neo4j,
    )
    payload = report.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
