from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.document_chunking import ChunkingConfig, DocumentChunk, OcrConfig, chunk_documents, write_chunks_jsonl
from app.services.document_faiss import build_document_faiss_index
from app.services.recipe_chunk_refiner import refine_recipe_chunks, write_recipe_chunks_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read PDF/Word/HTML documents, chunk them, optionally refine recipe chunks, and build a FAISS index."
    )
    parser.add_argument("sources", nargs="*", help="PDF, DOCX, HTML, TXT, MD paths or http(s) URLs.")
    parser.add_argument("--from-chunks", help="Reuse an existing chunks JSONL file instead of reading source documents.")
    parser.add_argument("--input-dir", help="Batch mode: read source files from this directory.")
    parser.add_argument("--glob", default="*.pdf", help="Batch mode file pattern under --input-dir. Default: *.pdf")
    parser.add_argument("--recursive", action="store_true", help="Batch mode: search --input-dir recursively.")
    parser.add_argument("--batch", action="store_true", help="Process each source as a separate index instead of merging sources.")
    parser.add_argument("--continue-on-error", action="store_true", help="Batch mode: keep processing later files when one file fails.")
    parser.add_argument("--skip-existing", action="store_true", help="Batch mode: skip files whose index and metadata already exist.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--recipe-refine", action="store_true", help="Refine OCR chunks into recipe-oriented chunks.")
    parser.add_argument("--ocr", action="store_true", help="Use OCR when a PDF has little or noisy text layer.")
    parser.add_argument("--ocr-force", action="store_true", help="Always OCR PDF pages even if a text layer exists.")
    parser.add_argument("--ocr-engine", default="auto", choices=["auto", "rapidocr", "tesseract"])
    parser.add_argument("--ocr-dpi", type=int, default=180)
    parser.add_argument("--ocr-max-pages", type=int, default=None)
    parser.add_argument("--ocr-min-extracted-chars", type=int, default=1000)
    parser.add_argument("--ocr-progress-every", type=int, default=1)
    parser.add_argument("--no-ocr-progress", action="store_true")
    parser.add_argument("--output-prefix", default=None, help="Output file prefix under data/processed. Defaults to the source PDF/chunk file name.")
    parser.add_argument("--chunks-output", default=None)
    parser.add_argument("--faiss-output", default=None)
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--preview", type=int, default=0, help="Print the first N final chunks.")
    args = parser.parse_args()

    if args.input_dir or args.batch:
        return run_batch(args)

    summary = build_one(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_batch(args: argparse.Namespace) -> int:
    if args.from_chunks:
        raise ValueError("Batch mode does not support --from-chunks.")
    if args.chunks_output or args.faiss_output or args.metadata_output:
        raise ValueError("Batch mode does not allow explicit --chunks-output/--faiss-output/--metadata-output.")

    sources = expand_batch_sources(args)
    if not sources:
        raise ValueError("No input files found for batch indexing.")

    results = []
    for index, source in enumerate(sources, start=1):
        per_args = copy.copy(args)
        per_args.sources = [str(source)]
        per_args.input_dir = None
        per_args.batch = False
        per_args.from_chunks = None
        output_paths = resolve_output_paths(per_args)
        if args.skip_existing and output_paths["faiss"].exists() and output_paths["metadata"].exists():
            result = {
                "source": str(source),
                "status": "skipped",
                "reason": "existing index and metadata",
                "faiss_output": str(output_paths["faiss"]),
                "metadata_output": str(output_paths["metadata"]),
            }
            results.append(result)
            print(json.dumps({"batch_progress": f"{index}/{len(sources)}", **result}, ensure_ascii=False))
            continue

        try:
            summary = build_one(per_args)
            result = {"source": str(source), "status": "ok", **summary}
        except Exception as exc:
            result = {"source": str(source), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            if not args.continue_on_error:
                print(json.dumps({"batch_progress": f"{index}/{len(sources)}", **result}, ensure_ascii=False))
                raise
        results.append(result)
        print(json.dumps({"batch_progress": f"{index}/{len(sources)}", **result}, ensure_ascii=False))

    ok_count = sum(1 for row in results if row["status"] == "ok")
    skipped_count = sum(1 for row in results if row["status"] == "skipped")
    failed_count = sum(1 for row in results if row["status"] == "failed")
    print(
        json.dumps(
            {
                "batch_total": len(results),
                "ok": ok_count,
                "skipped": skipped_count,
                "failed": failed_count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed_count else 0


def expand_batch_sources(args: argparse.Namespace) -> list[Path | str]:
    sources: list[Path | str] = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        pattern = f"**/{args.glob}" if args.recursive else args.glob
        sources.extend(sorted(path for path in input_dir.glob(pattern) if path.is_file()))
    sources.extend(args.sources)
    return sources


def build_one(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = resolve_output_paths(args)

    config = ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    ocr_config = OcrConfig(
        enabled=args.ocr,
        engine=args.ocr_engine,
        dpi=args.ocr_dpi,
        max_pages=args.ocr_max_pages,
        min_extracted_chars=args.ocr_min_extracted_chars,
        force=args.ocr_force,
        show_progress=args.ocr and not args.no_ocr_progress,
        progress_every=args.ocr_progress_every,
    )

    if args.from_chunks:
        raw_chunks = load_document_chunks_jsonl(Path(args.from_chunks))
    else:
        if not args.sources:
            raise ValueError("Provide at least one source document, or use --from-chunks.")
        raw_chunks = chunk_documents(args.sources, config=config, ocr_config=ocr_config)
    final_chunks: list[DocumentChunk]
    recipe_refine_fallback = False
    if args.recipe_refine:
        refined = refine_recipe_chunks([chunk.to_dict() for chunk in raw_chunks])
        if refined:
            write_recipe_chunks_jsonl(refined, output_paths["chunks"])
            final_chunks = [recipe_to_document_chunk(chunk.to_dict()) for chunk in refined]
        else:
            recipe_refine_fallback = True
            print(
                "[build_document_index] recipe-refine produced 0 chunks; "
                "falling back to raw document chunks for this source.",
                file=sys.stderr,
            )
            write_chunks_jsonl(raw_chunks, output_paths["chunks"])
            final_chunks = raw_chunks
    else:
        write_chunks_jsonl(raw_chunks, output_paths["chunks"])
        final_chunks = raw_chunks

    if not final_chunks:
        raise ValueError(
            "No chunks were produced from the input document. "
            "Check whether OCR extracted readable text, or try without --recipe-refine."
        )

    index = build_document_faiss_index(final_chunks)
    index.save(output_paths["faiss"], output_paths["metadata"])

    summary = {
        "source_count": len(args.sources),
        "from_chunks": args.from_chunks,
        "raw_chunk_count": len(raw_chunks),
        "final_chunk_count": len(final_chunks),
        "recipe_refine": args.recipe_refine,
        "recipe_refine_fallback": recipe_refine_fallback,
        "embedding_backend": index.embedding_backend,
        "embedding_errors": list(getattr(index.embedding_provider, "errors", [])),
        "index_type": index.index_type,
        "chunks_output": str(output_paths["chunks"]),
        "faiss_output": str(output_paths["faiss"]),
        "metadata_output": str(output_paths["metadata"]),
    }
    for chunk in final_chunks[: args.preview]:
        print(f"\n===== {chunk.chunk_id} =====")
        print(chunk.text[:1000])
    return summary


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    processed_dir = ROOT / "data" / "processed"
    prefix = args.output_prefix or infer_output_prefix(args)
    stem = sanitize_output_stem(prefix)
    if args.recipe_refine and not stem.endswith("_recipe"):
        stem = f"{stem}_recipe"
    return {
        "chunks": Path(args.chunks_output) if args.chunks_output else processed_dir / f"{stem}_chunks.jsonl",
        "faiss": Path(args.faiss_output) if args.faiss_output else processed_dir / f"{stem}.index",
        "metadata": Path(args.metadata_output) if args.metadata_output else processed_dir / f"{stem}_metadata.json",
    }


def infer_output_prefix(args: argparse.Namespace) -> str:
    if args.sources:
        return Path(str(args.sources[0])).stem
    if args.from_chunks:
        stem = Path(args.from_chunks).stem
        for suffix in ("_chunks", "_recipe_chunks"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem
    return "document"


def sanitize_output_stem(value: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value.strip())
    stem = re.sub(r"\s+", "_", stem).strip("._")
    return stem or "document"


def recipe_to_document_chunk(row: dict[str, Any]) -> DocumentChunk:
    metadata = dict(row.get("metadata", {}))
    title = row.get("title")
    if title:
        metadata["title"] = str(title)
    text = str(row["text"])
    return DocumentChunk(
        chunk_id=str(row["chunk_id"]),
        source=str(row.get("source", "")),
        source_type=str(row.get("source_type", "pdf")),
        text=text,
        start_char=int(row.get("start_char", 0)),
        end_char=int(row.get("end_char", len(text))),
        metadata=metadata,
    )


def load_document_chunks_jsonl(path: Path) -> list[DocumentChunk]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = []
    for row in rows:
        text = str(row["text"])
        chunks.append(
            DocumentChunk(
                chunk_id=str(row["chunk_id"]),
                source=str(row.get("source", "")),
                source_type=str(row.get("source_type", "pdf")),
                text=text,
                start_char=int(row.get("start_char", 0)),
                end_char=int(row.get("end_char", len(text))),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return chunks


if __name__ == "__main__":
    raise SystemExit(main())
