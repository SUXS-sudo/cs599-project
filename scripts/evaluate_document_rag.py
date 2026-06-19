from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.document_chunking import DocumentChunk
from app.services.embeddings import EmbeddingProvider
from app.services.hyde import HyDEGenerator
from app.services.llm_client import load_dotenv
from app.services.query_rewrite import rewrite_recipe_query
from scripts.search_document_faiss import collect_candidates, rank_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PDF document RAG with FAISS/BM25/metadata/Cross-Encoder.")
    parser.add_argument("--index", default=str(ROOT / "data" / "processed" / "new_pdf_recipe.index"))
    parser.add_argument("--metadata", default=str(ROOT / "data" / "processed" / "new_pdf_recipe_metadata.json"))
    parser.add_argument("--eval-file", default=None, help="Optional JSONL with query and expected_chunk_ids or expected_dish_names.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases after optional shuffle.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hyde", action="store_true", help="Enable LLM HyDE. This can be slow because it calls the LLM per case.")
    parser.add_argument("--no-query-rewrite", action="store_true")
    parser.add_argument("--no-cross-encoder-rerank", action="store_true")
    parser.add_argument("--cross-encoder-model", default=None)
    parser.add_argument("--output", default="", help="Optional JSON output path for detailed evaluation results.")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N cases; set 0 to disable.")
    args = parser.parse_args()

    started = time.perf_counter()
    load_dotenv()
    os.environ["HYDE_ENABLED"] = "true" if args.hyde else "false"

    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    log("configuration:")
    log(f"  index={index_path}")
    log(f"  metadata={metadata_path}")
    log(f"  top_k={args.top_k} candidate_k={args.candidate_k} limit={args.limit or 'all'}")
    log(f"  query_rewrite={not args.no_query_rewrite} hyde={args.hyde} cross_encoder_rerank={not args.no_cross_encoder_rerank}")

    log("loading FAISS index")
    import faiss

    index = faiss.read_index(str(index_path))
    log(f"loaded FAISS index type={type(index).__name__} dim={index.d} ntotal={index.ntotal}")
    log("loading metadata and chunks")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    set_embedding_model_from_metadata(metadata)
    chunks = [DocumentChunk(**item) for item in metadata.get("chunks", [])]
    if not chunks:
        raise ValueError(f"No chunks found in metadata: {metadata_path}")
    log(f"loaded chunks={len(chunks)} index_type={metadata.get('index_type')} embedding_backend={metadata.get('embedding_backend')}")

    log("initializing query embedding provider")
    provider = EmbeddingProvider([chunk.text for chunk in chunks])
    log(f"query_embedding_backend={provider.backend}")
    cases = load_cases(Path(args.eval_file), chunks) if args.eval_file else build_cases_from_chunks(chunks)
    if args.shuffle:
        random.Random(args.seed).shuffle(cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    log(f"evaluation_cases={len(cases)} source={'eval_file' if args.eval_file else 'metadata_dish_names'}")
    if not args.no_cross_encoder_rerank:
        log("Cross-Encoder rerank is enabled; first evaluated case may show 'Loading weights' while the reranker model loads")
    if args.hyde:
        log("HyDE is enabled; this will call the LLM once per case and may be slow")

    hyde_generator = HyDEGenerator()
    details = []
    totals = {
        "hit": 0,
        "recall": 0.0,
        "precision": 0.0,
        "average_precision": 0.0,
        "reciprocal_rank": 0.0,
    }

    for case_index, case in enumerate(cases, start=1):
        ranked = search_case(
            case,
            index=index,
            provider=provider,
            chunks=chunks,
            hyde_generator=hyde_generator,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            expected_dim=index.d,
            use_query_rewrite=not args.no_query_rewrite,
            use_hyde=args.hyde,
            use_cross_encoder=not args.no_cross_encoder_rerank,
            cross_encoder_model=args.cross_encoder_model,
        )
        metrics = score_case(ranked, set(case["expected_chunk_ids"]), args.top_k)
        for key in totals:
            totals[key] += metrics[key]
        detail = {
            "query": case["query"],
            "expected_chunk_ids": case["expected_chunk_ids"],
            "expected_dish_names": case.get("expected_dish_names", []),
            "ranked": [
                {
                    "rank": index + 1,
                    "chunk_id": item["chunk"].chunk_id,
                    "dish_name": dish_name(item["chunk"]),
                    "score": round(float(item["score"]), 6),
                    "sources": item["sources"],
                }
                for index, item in enumerate(ranked[: args.top_k])
            ],
            "metrics": metrics,
        }
        details.append(detail)
        if args.progress_every > 0 and (case_index == 1 or case_index % args.progress_every == 0 or case_index == len(cases)):
            elapsed = time.perf_counter() - started
            print_progress(case_index, len(cases), totals, elapsed)

    count = len(cases)
    summary = {
        "cases": count,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "index": str(index_path),
        "metadata": str(metadata_path),
        "index_type": metadata.get("index_type"),
        "embedding_backend": metadata.get("embedding_backend"),
        "query_embedding_backend": provider.backend,
        "query_rewrite": not args.no_query_rewrite,
        "hyde": args.hyde,
        "cross_encoder_rerank": not args.no_cross_encoder_rerank,
        "hit_at_k": safe_avg(totals["hit"], count),
        "recall_at_k": safe_avg(totals["recall"], count),
        "precision_at_k": safe_avg(totals["precision"], count),
        "map_at_k": safe_avg(totals["average_precision"], count),
        "mrr_at_k": safe_avg(totals["reciprocal_rank"], count),
    }
    log("final summary")
    print_summary(summary)
    if args.show_errors:
        for detail in details:
            if detail["metrics"]["hit"] == 0:
                print(json.dumps(detail, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"writing detailed output to {output_path}")
        output_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"output={output_path}")
    return 0


def set_embedding_model_from_metadata(metadata: dict[str, Any]) -> None:
    backend = str(metadata.get("embedding_backend") or "")
    prefix = "local_sentence_transformers:"
    if backend.startswith(prefix):
        os.environ["EMBEDDING_PROVIDER"] = "local"
        os.environ["EMBEDDING_MODEL"] = backend[len(prefix) :]


def build_cases_from_chunks(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        name = dish_name(chunk)
        if not name:
            continue
        grouped.setdefault(name, []).append(chunk.chunk_id)
    return [
        {
            "query": f"{name}怎么做",
            "expected_chunk_ids": chunk_ids,
            "expected_dish_names": [name],
        }
        for name, chunk_ids in grouped.items()
    ]


def load_cases(path: Path, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
    dish_to_chunks: dict[str, list[str]] = {}
    for chunk in chunks:
        name = dish_name(chunk)
        if name:
            dish_to_chunks.setdefault(name, []).append(chunk.chunk_id)
    cases = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        query = str(row.get("query") or "").strip()
        if not query:
            raise ValueError(f"{path}:{line_number} requires query")
        expected_chunk_ids = [str(item) for item in row.get("expected_chunk_ids", [])]
        expected_dish_names = [str(item) for item in row.get("expected_dish_names", [])]
        for name in expected_dish_names:
            expected_chunk_ids.extend(dish_to_chunks.get(name, []))
        expected_chunk_ids = list(dict.fromkeys(expected_chunk_ids))
        if not expected_chunk_ids:
            raise ValueError(f"{path}:{line_number} requires expected_chunk_ids or resolvable expected_dish_names")
        cases.append(
            {
                "query": query,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_dish_names": expected_dish_names,
            }
        )
    return cases


def search_case(
    case: dict[str, Any],
    index,
    provider: EmbeddingProvider,
    chunks: list[DocumentChunk],
    hyde_generator: HyDEGenerator,
    top_k: int,
    candidate_k: int,
    expected_dim: int,
    use_query_rewrite: bool,
    use_hyde: bool,
    use_cross_encoder: bool,
    cross_encoder_model: str | None,
) -> list[dict[str, Any]]:
    query = case["query"]
    rewrite = rewrite_recipe_query(query)
    expanded_query = rewrite.expanded_query if use_query_rewrite else query
    hyde_query = hyde_generator.generate(query).hypothetical_document if use_hyde else ""
    candidates = collect_candidates(
        index=index,
        provider=provider,
        chunks=chunks,
        original_query=query,
        expanded_query=expanded_query,
        hyde_query=hyde_query,
        rewrite_core_terms=rewrite.core_terms,
        vector_k=min(candidate_k, len(chunks)),
        keyword_k=min(candidate_k, len(chunks)),
        expected_dim=expected_dim,
        include_keyword=True,
    )
    return rank_results(
        query,
        expanded_query,
        chunks,
        candidates,
        mode="hybrid",
        use_cross_encoder=use_cross_encoder,
        cross_encoder_model=cross_encoder_model,
    )[:top_k]


def score_case(ranked: list[dict[str, Any]], expected_ids: set[str], top_k: int) -> dict[str, float]:
    retrieved_ids = [item["chunk"].chunk_id for item in ranked[:top_k]]
    hits = [1 if chunk_id in expected_ids else 0 for chunk_id in retrieved_ids]
    relevant_found = sum(hits)
    precision = relevant_found / top_k if top_k else 0.0
    recall = relevant_found / len(expected_ids) if expected_ids else 0.0
    reciprocal_rank = 0.0
    precision_sum = 0.0
    hit_count = 0
    for rank, hit in enumerate(hits, start=1):
        if not hit:
            continue
        if reciprocal_rank == 0.0:
            reciprocal_rank = 1.0 / rank
        hit_count += 1
        precision_sum += hit_count / rank
    average_precision = precision_sum / len(expected_ids) if expected_ids else 0.0
    return {
        "hit": 1.0 if relevant_found else 0.0,
        "precision": precision,
        "recall": min(recall, 1.0),
        "average_precision": min(average_precision, 1.0),
        "reciprocal_rank": reciprocal_rank,
    }


def dish_name(chunk: DocumentChunk) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("dish_name") or metadata.get("title") or "").strip()


def safe_avg(total: float, count: int) -> float:
    return total / count if count else 0.0


def log(message: str) -> None:
    print(f"[eval] {message}", flush=True)


def print_progress(done: int, total: int, totals: dict[str, float], elapsed: float) -> None:
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    hit = safe_avg(totals["hit"], done)
    recall = safe_avg(totals["recall"], done)
    precision = safe_avg(totals["precision"], done)
    mrr = safe_avg(totals["reciprocal_rank"], done)
    log(
        "progress {done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s "
        "hit@k={hit:.4f} recall@k={recall:.4f} precision@k={precision:.4f} mrr@k={mrr:.4f}".format(
            done=done,
            total=total,
            elapsed=elapsed,
            eta=remaining,
            hit=hit,
            recall=recall,
            precision=precision,
            mrr=mrr,
        )
    )


def print_summary(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}={value:.4f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    raise SystemExit(main())
