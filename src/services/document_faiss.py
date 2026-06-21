from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.services.document_chunking import DocumentChunk
from src.services.embeddings import EmbeddingProvider


@dataclass
class DocumentSearchResult:
    chunk: DocumentChunk
    score: float


class DocumentFaissIndex:
    def __init__(self, chunks: list[DocumentChunk], embedding_provider: EmbeddingProvider | None = None) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider or EmbeddingProvider([chunk.text for chunk in chunks])
        self.embedding_backend = self.embedding_provider.backend
        self.index_type = "hnsw_ip"
        self.hnsw_config = {"metric": "inner_product", "m": 32, "ef_construction": 200, "ef_search": 64}
        self.index = None
        if chunks:
            self._build_index()

    def _build_index(self) -> None:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError("FAISS indexing requires faiss-cpu. Install requirements.txt first.") from exc

        embeddings = np.asarray(self.embedding_provider.embed([chunk.text for chunk in self.chunks]), dtype="float32")
        index = build_hnsw_ip_index(faiss, embeddings.shape[1])
        index.add(embeddings)
        self.index = index

    def search(self, query: str, top_k: int = 5) -> list[DocumentSearchResult]:
        if self.index is None or not query.strip():
            return []
        query_embedding = np.asarray(self.embedding_provider.embed([query]), dtype="float32")
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                results.append(DocumentSearchResult(self.chunks[int(index)], float(score)))
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            raise ValueError("Cannot save an empty FAISS index")
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError("FAISS persistence requires faiss-cpu. Install requirements.txt first.") from exc

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temp_index = tempfile.NamedTemporaryFile(prefix="smart_recipe_faiss_", suffix=".index", delete=False)
        temp_index_path = Path(temp_index.name)
        temp_index.close()
        try:
            faiss.write_index(self.index, str(temp_index_path))
            shutil.move(str(temp_index_path), str(index_path))
        finally:
            if temp_index_path.exists():
                temp_index_path.unlink()
        metadata = {
            "embedding_backend": self.embedding_backend,
            "embedding_errors": list(getattr(self.embedding_provider, "errors", [])),
            "index_type": self.index_type,
            "hnsw": self.hnsw_config,
            "chunk_count": len(self.chunks),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_hnsw_ip_index(faiss_module, dimension: int):
    m = 32
    try:
        index = faiss_module.IndexHNSWFlat(dimension, m, faiss_module.METRIC_INNER_PRODUCT)
    except TypeError:
        index = faiss_module.IndexHNSWFlat(dimension, m)
        index.metric_type = faiss_module.METRIC_INNER_PRODUCT
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 64
    return index


def build_document_faiss_index(chunks: list[DocumentChunk]) -> DocumentFaissIndex:
    return DocumentFaissIndex(chunks)
