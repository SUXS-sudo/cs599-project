from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.services.document_chunking import DocumentChunk
from app.services.embeddings import EmbeddingProvider
from app.services.hyde import HyDEGenerator
from app.services.query_rewrite import rewrite_recipe_query
from app.state import AgentState
from app.tools.base import ToolResult


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


class SearchDocumentChunksTool:
    name = "search_document_chunks"
    description = "Search the local PDF/document FAISS index for cookbook or document evidence."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            "preview_chars": {"type": "integer", "minimum": 80, "maximum": 800},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        index_path: Path | None = None,
        metadata_path: Path | None = None,
        searcher: Callable[[str, int, int], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.searcher = searcher

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        query = str(args.get("query") or state.user_input).strip()
        top_k = clamp_int(args.get("top_k", state.top_k), default=state.top_k, minimum=1, maximum=5)
        preview_chars = clamp_int(args.get("preview_chars", 500), default=500, minimum=80, maximum=800)
        if not query:
            return ToolResult(self.name, False, "", error="query is required")

        try:
            rows = self.searcher(query, top_k, preview_chars) if self.searcher else self._search(query, top_k, preview_chars)
        except Exception as exc:
            return ToolResult(self.name, False, "", error=f"{type(exc).__name__}: {exc}")
        if not rows:
            return ToolResult(self.name, True, "search_document_chunks: no matching document chunks found.", data={"document_chunks": []})

        lines = ["search_document_chunks results:"]
        for row in rows:
            lines.append(
                f"{row['rank']}. {row['chunk_id']} | score={row['score']:.4f} | source={row['source']}\n{row['preview']}"
            )
        return ToolResult(self.name, True, "\n".join(lines), data={"document_chunks": rows})

    def _search(self, query: str, top_k: int, preview_chars: int) -> list[dict[str, Any]]:
        index_path, metadata_path = resolve_document_index(self.index_path, self.metadata_path)
        if index_path is None or metadata_path is None:
            return []

        import faiss
        from scripts.search_document_faiss import collect_candidates, rank_results

        index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in metadata.get("chunks", [])]
        if not chunks:
            return []

        provider = EmbeddingProvider([chunk.text for chunk in chunks])
        rewrite = rewrite_recipe_query(query)
        hyde = HyDEGenerator().generate(query)
        candidate_k = min(max(top_k * 6, top_k), len(chunks))
        candidates = collect_candidates(
            index=index,
            provider=provider,
            chunks=chunks,
            original_query=query,
            expanded_query=rewrite.expanded_query,
            hyde_query=hyde.hypothetical_document,
            rewrite_core_terms=rewrite.core_terms,
            vector_k=candidate_k,
            keyword_k=candidate_k,
            expected_dim=index.d,
            include_keyword=True,
        )
        ranked = rank_results(
            query,
            rewrite.expanded_query,
            chunks,
            candidates,
            mode="hybrid",
            use_cross_encoder=False,
            cross_encoder_model=None,
        )[:top_k]
        return [document_result_to_dict(rank, item, preview_chars) for rank, item in enumerate(ranked, start=1)]


def resolve_document_index(index_path: Path | None, metadata_path: Path | None) -> tuple[Path | None, Path | None]:
    if index_path and metadata_path and index_path.exists() and metadata_path.exists():
        return index_path, metadata_path
    metadata_candidates = sorted(PROCESSED_DIR.glob("*_metadata.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate_metadata in metadata_candidates:
        stem = candidate_metadata.name.removesuffix("_metadata.json")
        candidate_index = candidate_metadata.with_name(f"{stem}.index")
        if candidate_index.exists():
            return candidate_index, candidate_metadata
    return None, None


def document_result_to_dict(rank: int, item: dict[str, Any], preview_chars: int) -> dict[str, Any]:
    chunk = item["chunk"]
    preview = chunk.text[:preview_chars].rstrip()
    if len(chunk.text) > preview_chars:
        preview += "\n..."
    return {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "source_type": chunk.source_type,
        "score": float(item["score"]),
        "metadata": chunk.metadata or {},
        "preview": preview,
    }


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)
