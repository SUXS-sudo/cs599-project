from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.document_chunking import DocumentChunk
from src.services.embeddings import EmbeddingProvider
from src.services.hyde import HyDEGenerator
from src.services.llm_client import load_dotenv
from src.services.query_rewrite import rewrite_recipe_query


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Search a document FAISS index and print matching chunks.")
    parser.add_argument("--index", default=str(ROOT / "data" / "processed" / "new_pdf.index"))
    parser.add_argument("--metadata", default=str(ROOT / "data" / "processed" / "new_pdf_metadata.json"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=700)
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    parser.add_argument("--no-query-rewrite", action="store_true", help="Disable recipe query expansion.")
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE vector recall for this run.")
    parser.add_argument("--no-hyde", action="store_true", help="Disable HyDE vector recall for this run.")
    parser.add_argument("--candidate-k", type=int, default=30, help="Candidates to collect from each recall route.")
    parser.add_argument("--no-cross-encoder-rerank", action="store_true", help="Disable Cross-Encoder reranking.")
    parser.add_argument("--cross-encoder-model", default=None, help="Model path/name for Cross-Encoder reranking.")
    parser.add_argument("--embedding-provider", default=None, help="Override EMBEDDING_PROVIDER for query embedding.")
    parser.add_argument("--embedding-model", default=None, help="Override EMBEDDING_MODEL for query embedding.")
    args = parser.parse_args()

    if args.embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    import faiss

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not args.embedding_model:
        backend = str(metadata.get("embedding_backend") or "")
        prefix = "local_sentence_transformers:"
        if backend.startswith(prefix):
            recorded_model = backend[len(prefix) :]
            if is_usable_sentence_transformer_path(recorded_model):
                os.environ["EMBEDDING_MODEL"] = recorded_model

    chunks = [DocumentChunk(**item) for item in metadata.get("chunks", [])]
    if not chunks:
        raise ValueError(f"No chunks found in metadata: {metadata_path}")

    provider = EmbeddingProvider([chunk.text for chunk in chunks])
    rewrite = rewrite_recipe_query(args.query)
    expanded_query = args.query if args.no_query_rewrite else rewrite.expanded_query
    if args.hyde:
        os.environ["HYDE_ENABLED"] = "true"
    if args.no_hyde:
        os.environ["HYDE_ENABLED"] = "false"
    hyde = HyDEGenerator().generate(args.query)
    vector_k = args.top_k if args.mode == "vector" else min(args.candidate_k, len(chunks))
    candidates = collect_candidates(
        index=index,
        provider=provider,
        chunks=chunks,
        original_query=args.query,
        expanded_query=expanded_query,
        hyde_query=hyde.hypothetical_document,
        rewrite_core_terms=rewrite.core_terms,
        vector_k=vector_k,
        keyword_k=min(args.candidate_k, len(chunks)),
        expected_dim=index.d,
        include_keyword=args.mode != "vector",
    )
    print(
        json.dumps(
            {
                "query": args.query,
                "retrieval_queries": {
                    "faiss_original": args.query,
                    "faiss_expanded": expanded_query,
                    "faiss_hyde": hyde.hypothetical_document,
                    "keyword_original": args.query,
                    "keyword_expanded": expanded_query,
                    "metadata_filter_query": args.query,
                    "cross_encoder_rerank_query": args.query,
                    "answer_generation_query": args.query,
                },
                "query_rewrite": {
                    "enabled": not args.no_query_rewrite,
                    "intent": rewrite.intent,
                    "added_terms": rewrite.added_terms,
                    "core_terms": rewrite.core_terms,
                },
                "hyde": {
                    "enabled": hyde.enabled,
                    "generator": hyde.generator,
                    "hypothetical_document": hyde.hypothetical_document,
                },
                "index": str(index_path),
                "metadata": str(metadata_path),
                "index_embedding_backend": metadata.get("embedding_backend"),
                "query_embedding_backend": provider.backend,
                "query_embedding_errors": provider.errors,
                "chunk_count": len(chunks),
                "candidate_count": len(candidates),
                "rerank": rerank_label(args.mode, not args.no_cross_encoder_rerank),
                "generation_constraint": "Use only retrieved chunks as evidence; answer the original query, not the expanded query.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    ranked = rank_results(
        args.query,
        expanded_query,
        chunks,
        candidates,
        mode=args.mode,
        use_cross_encoder=not args.no_cross_encoder_rerank,
        cross_encoder_model=args.cross_encoder_model,
    )[: args.top_k]

    for rank, item in enumerate(ranked, start=1):
        chunk = item["chunk"]
        print(
            "\n===== rank {rank} | score={score:.4f} | vector={vector:.4f} | keyword={keyword:.4f} | "
            "metadata={metadata:.4f} | cross_encoder={cross_encoder:.4f} | "
            "sources={sources} | vector_index={index} =====".format(
                rank=rank,
                score=item["score"],
                vector=item["vector_score"],
                keyword=item["keyword_score"],
                metadata=item["metadata_score"],
                cross_encoder=item["cross_encoder_score"],
                sources=",".join(item["sources"]),
                index=item["index"],
            )
        )
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"source: {chunk.source}")
        print(f"source_type: {chunk.source_type}")
        if chunk.metadata:
            print(f"metadata: {json.dumps(chunk.metadata, ensure_ascii=False)}")
        print(f"char_range: {chunk.start_char}-{chunk.end_char}")
        print("text:")
        print(preview_text(chunk.text, args.preview_chars))
    return 0


def preview_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n..."


def is_usable_sentence_transformer_path(model_name: str) -> bool:
    if not model_name:
        return False
    path = Path(model_name)
    if path.is_absolute():
        return (path / "modules.json").exists()
    return True


def rank_results(
    original_query: str,
    expanded_query: str,
    chunks: list[DocumentChunk],
    candidates: list[dict[str, Any]],
    mode: str,
    use_cross_encoder: bool,
    cross_encoder_model: str | None,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        raw_index = candidate["index"]
        chunk = chunks[raw_index]
        vector_score = candidate.get("vector_score", 0.0)
        lexical = candidate.get("keyword_score", 0.0)
        metadata = metadata_score(original_query, chunk)
        if mode == "vector":
            final_score = float(vector_score)
        else:
            final_score = 0.68 * float(vector_score) + 0.24 * lexical + 0.08 * metadata
        ranked.append(
            {
                "chunk": chunk,
                "index": raw_index,
                "score": final_score,
                "vector_score": float(vector_score),
                "keyword_score": lexical,
                "metadata_score": metadata,
                "cross_encoder_score": 0.0,
                "sources": sorted(candidate.get("sources", [])),
            }
        )
    if mode != "vector" and use_cross_encoder and ranked:
        apply_cross_encoder_rerank(original_query, ranked, cross_encoder_model)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def rerank_label(mode: str, use_cross_encoder: bool) -> str:
    if mode == "vector":
        return "disabled"
    return "cross-encoder" if use_cross_encoder else "disabled"


def collect_candidates(
    index,
    provider: EmbeddingProvider,
    chunks: list[DocumentChunk],
    original_query: str,
    expanded_query: str,
    hyde_query: str,
    rewrite_core_terms: list[str],
    vector_k: int,
    keyword_k: int,
    expected_dim: int,
    include_keyword: bool,
) -> list[dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    faiss_queries = [("faiss_original", original_query)]
    if expanded_query != original_query:
        faiss_queries.append(("faiss_expanded", expanded_query))
    if hyde_query and hyde_query not in {original_query, expanded_query}:
        faiss_queries.append(("faiss_hyde", hyde_query))
    for source, query in faiss_queries:
        query_vector = np.asarray(provider.embed([query]), dtype="float32")
        if query_vector.shape[1] != expected_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"query_dim={query_vector.shape[1]} index_dim={expected_dim}; "
                f"query_embedding_backend={provider.backend}; "
                f"query_embedding_errors={provider.errors}. "
                "Use the same EMBEDDING_MODEL/EMBEDDING_PROVIDER used when building the index, "
                "or rebuild the index with the current embedding backend."
            )
        scores, indices = index.search(query_vector, min(vector_k, len(chunks)))
        for score, raw_index in zip(scores[0], indices[0]):
            if raw_index < 0:
                continue
            candidate = candidates.setdefault(
                int(raw_index),
                {"index": int(raw_index), "vector_score": 0.0, "keyword_score": 0.0, "sources": set()},
            )
            candidate["vector_score"] = max(candidate["vector_score"], float(score))
            candidate["sources"].add(source)

    if include_keyword:
        keyword_queries = [original_query, expanded_query, " ".join(rewrite_core_terms)]
        for raw_index, score in keyword_candidates(keyword_queries, chunks, keyword_k):
            candidate = candidates.setdefault(
                raw_index,
                {"index": raw_index, "vector_score": 0.0, "keyword_score": 0.0, "sources": set()},
            )
            candidate["keyword_score"] = max(candidate["keyword_score"], score)
            candidate["sources"].add("keyword")
    return list(candidates.values())


def keyword_candidates(queries: list[str], chunks: list[DocumentChunk], top_k: int) -> list[tuple[int, float]]:
    tokenized_chunks = [extract_query_terms(searchable_text(chunk)) for chunk in chunks]
    avg_doc_len = (
        sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)
        if tokenized_chunks
        else 0.0
    )
    idf = build_bm25_idf(tokenized_chunks)
    raw_scores: dict[int, float] = {}
    for index, chunk in enumerate(chunks):
        score = max(
            (
                bm25_score(extract_query_terms(query), tokenized_chunks[index], idf, avg_doc_len)
                for query in queries
                if query.strip()
            ),
            default=0.0,
        )
        if query_exact_match(queries, searchable_text(chunk)):
            score += 0.35
        if score > 0:
            raw_scores[index] = score
    best_scores = normalize_scores(raw_scores)
    ranked = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def apply_cross_encoder_rerank(query: str, ranked: list[dict[str, Any]], rerank_model: str | None) -> None:
    scores = cross_encoder_scores(query, [item["chunk"] for item in ranked], rerank_model)
    for item, score in zip(ranked, scores):
        normalized = normalize_model_score(score)
        item["cross_encoder_score"] = normalized
        item["score"] = (
            0.38 * item["vector_score"]
            + 0.16 * item["keyword_score"]
            + 0.06 * item["metadata_score"]
            + 0.40 * normalized
        )


def cross_encoder_scores(query: str, chunks: list[DocumentChunk], model_name: str | None) -> list[float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(resolve_cross_encoder_model(model_name))
    pairs = [(query, rerank_text(chunk)) for chunk in chunks]
    return [float(score) for score in model.predict(pairs)]


def resolve_cross_encoder_model(model_name: str | None) -> str:
    if model_name:
        return model_name
    return os.getenv("RERANK_CROSS_ENCODER_MODEL") or "models/BAAI/bge-reranker-base"


def rerank_text(chunk: DocumentChunk) -> str:
    metadata = chunk.metadata or {}
    pieces = [
        str(metadata.get("dish_name") or ""),
        str(metadata.get("ingredients") or ""),
        str(metadata.get("seasonings") or ""),
        chunk.text,
    ]
    return "\n".join(piece for piece in pieces if piece)


def normalize_model_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + float(np.exp(-score)))


def metadata_score(query: str, chunk: DocumentChunk) -> float:
    metadata = chunk.metadata or {}
    score = 0.0
    dish_name = str(metadata.get("dish_name") or metadata.get("title") or "")
    ingredients = str(metadata.get("ingredients") or "")
    if dish_name and dish_name in query:
        score += 1.0
    elif dish_name and any(term in dish_name for term in extract_query_terms(query)):
        score += 0.55
    if ingredients and any(term in ingredients for term in extract_query_terms(query)):
        score += 0.35
    if metadata.get("has_steps"):
        score += 0.1
    return min(score, 1.0)


def searchable_text(chunk: DocumentChunk) -> str:
    metadata = chunk.metadata or {}
    pieces = [
        chunk.text,
        str(metadata.get("dish_name") or ""),
        str(metadata.get("ingredients") or ""),
        str(metadata.get("seasonings") or ""),
        str(metadata.get("category") or ""),
    ]
    return "\n".join(pieces)


def extract_query_terms(query: str) -> list[str]:
    compact = query.strip()
    words = re.findall(r"[\w\u4e00-\u9fff]+", compact)
    terms = []
    for word in words:
        if len(word) <= 1:
            continue
        terms.append(word)
    chinese_chars = [char for char in compact if "\u4e00" <= char <= "\u9fff"]
    terms.extend(chinese_chars)
    return list(dict.fromkeys(terms))


def build_bm25_idf(documents: list[list[str]]) -> dict[str, float]:
    doc_count = len(documents)
    token_doc_counts: dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            token_doc_counts[token] = token_doc_counts.get(token, 0) + 1
    return {
        token: float(np.log(1 + (doc_count - count + 0.5) / (count + 0.5)))
        for token, count in token_doc_counts.items()
    }


def bm25_score(query_tokens: list[str], document_tokens: list[str], idf: dict[str, float], avg_doc_len: float) -> float:
    if not query_tokens or not document_tokens or avg_doc_len <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    term_counts: dict[str, int] = {}
    for token in document_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1
    doc_len = len(document_tokens)
    denominator_base = k1 * (1 - b + b * doc_len / avg_doc_len)
    score = 0.0
    for token in dict.fromkeys(query_tokens):
        freq = term_counts.get(token, 0)
        if freq == 0:
            continue
        score += idf.get(token, 0.0) * (freq * (k1 + 1)) / (freq + denominator_base)
    return score


def normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {index: 1.0 for index in scores}
    return {
        index: (score - min_score) / (max_score - min_score)
        for index, score in scores.items()
    }


def query_exact_match(queries: list[str], haystack: str) -> bool:
    return any(query.strip() and query.strip() in haystack for query in queries)


if __name__ == "__main__":
    raise SystemExit(main())
