from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.document_faiss import build_hnsw_ip_index
from app.services.embeddings import EmbeddingProvider
from app.services.hyde import HyDEGenerator
from app.services.cache_store import cache_data_version, cache_ttl_seconds, get_cache_store, stable_cache_key


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
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


def normalize_rag_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend in {"bm25", "keyword", "keywords", "sklearn"}:
        return "bm25"
    return backend


@dataclass(frozen=True)
class Recipe:
    name: str
    ingredients: list[str]
    category: str
    cooking_time: str
    difficulty: str
    tags: list[str]
    calories: int
    suitable_for: list[str]
    steps: str

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "Recipe":
        return cls(
            name=item["name"],
            ingredients=list(item.get("ingredients", [])),
            category=item.get("category", ""),
            cooking_time=item.get("cooking_time", ""),
            difficulty=item.get("difficulty", ""),
            tags=list(item.get("tags", [])),
            calories=int(item.get("calories", 0)),
            suitable_for=list(item.get("suitable_for", [])),
            steps=item.get("steps", ""),
        )

    def searchable_text(self) -> str:
        parts = [
            self.name,
            " ".join(self.ingredients),
            self.category,
            " ".join(self.tags),
            " ".join(self.suitable_for),
            self.steps,
        ]
        return " ".join(parts)


class RecipeRetriever:
    """Local RAG retriever.

    The default backend uses local BM25. When Chroma or FAISS is
    enabled, retrieval is hybrid: dense vector recall plus keyword recall.
    """

    def __init__(self, data_path: Path, llm_client=None) -> None:
        self.recipes = self._load_recipes(data_path)
        self.data_path = data_path
        self.requested_backend = normalize_rag_backend(os.getenv("RAG_BACKEND", "bm25"))
        self.backend = self.requested_backend
        self.backend_errors: list[str] = []
        self.recipe_texts = [recipe.searchable_text() for recipe in self.recipes]
        self.embedding_provider = EmbeddingProvider(self.recipe_texts)
        self.embedding_backend = self.embedding_provider.backend
        self.hybrid_vector_weight = _env_float("RAG_HYBRID_VECTOR_WEIGHT", 0.65)
        self.hybrid_keyword_weight = _env_float("RAG_HYBRID_KEYWORD_WEIGHT", 0.35)
        self.hyde = HyDEGenerator(llm_client)
        self.last_hyde_query = ""
        self.last_hyde_generator = ""
        self.cache = get_cache_store()
        self.last_cache_hit = False
        self._init_keyword_backend()
        self._init_vector_backend()
        self._recipes_by_name = {recipe.name: recipe for recipe in self.recipes}

    def search(self, query: str, top_k: int = 3) -> list[tuple[Recipe, float]]:
        if not query.strip():
            return []
        cached = self._get_cached_search(query, top_k)
        if cached is not None:
            self.last_cache_hit = True
            return cached
        self.last_cache_hit = False
        if self.backend == "chroma" and hasattr(self, "chroma_collection"):
            results = self._search_hybrid(query, top_k, self._search_chroma_scores)
        elif self.backend == "faiss" and hasattr(self, "faiss_index"):
            results = self._search_hybrid(query, top_k, self._search_faiss_scores)
        else:
            results = self._search_keyword(query, top_k)
        self._set_cached_search(query, top_k, results)
        return results

    def status(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "retrieval_strategy": "hybrid_vector_bm25" if self.backend in {"chroma", "faiss"} else "keyword_bm25",
            "keyword_backend": self.keyword_backend,
            "embedding_backend": self.embedding_backend,
            "hyde_enabled": self.hyde.enabled,
            "hyde_generator": self.hyde.generator,
            "cache_backend": self.cache.backend,
            "last_cache_hit": self.last_cache_hit,
            "recipe_count": len(self.recipes),
            "data_path": str(self.data_path),
            "errors": list(self.backend_errors),
        }

    def _search_cache_key(self, query: str, top_k: int) -> str:
        return stable_cache_key(
            "retrieval",
            {
                "query": query.strip(),
                "top_k": top_k,
                "backend": self.backend,
                "embedding_backend": self.embedding_backend,
                "recipe_count": len(self.recipes),
                "data_version": cache_data_version(),
            },
        )

    def _get_cached_search(self, query: str, top_k: int) -> list[tuple[Recipe, float]] | None:
        data = self.cache.get_json(self._search_cache_key(query, top_k))
        if not isinstance(data, list):
            return None
        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            recipe = self._recipes_by_name.get(str(item.get("name") or ""))
            if recipe is None:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            results.append((recipe, score))
        return results if results else None

    def _set_cached_search(self, query: str, top_k: int, results: list[tuple[Recipe, float]]) -> None:
        if not results:
            return
        payload = [{"name": recipe.name, "score": score} for recipe, score in results]
        ttl = cache_ttl_seconds("CACHE_RETRIEVAL_TTL_SECONDS", 24 * 60 * 60)
        self.cache.set_json(self._search_cache_key(query, top_k), payload, ttl_seconds=ttl)

    @staticmethod
    def _load_recipes(data_path: Path) -> list[Recipe]:
        with data_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return [Recipe.from_dict(item) for item in raw]

    def _init_keyword_backend(self) -> None:
        self.keyword_backend = "bm25"
        self.documents = [self._tokenize(recipe.searchable_text()) for recipe in self.recipes]
        self.bm25_idf = self._build_bm25_idf(self.documents)
        self.avg_doc_len = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )

    def _init_vector_backend(self) -> None:
        if self.requested_backend == "chroma" and self._init_chroma_backend():
            return
        if self.requested_backend == "faiss" and self._init_faiss_backend():
            return
        self.backend = self.keyword_backend

    def _init_chroma_backend(self) -> bool:
        try:
            import chromadb

            embeddings = self.embedding_provider.embed([recipe.searchable_text() for recipe in self.recipes])
            chroma_dir = self.data_path.parent / "chroma_store"
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_or_create_collection(name="smart_recipe_recipes")

            existing = collection.count()
            expected_metadata = f"{len(self.recipes)}:{self.embedding_backend}"
            should_rebuild = existing != len(self.recipes)
            if existing:
                try:
                    peek = collection.peek(limit=1)
                    current_metadata = peek.get("metadatas", [{}])[0].get("embedding_backend")
                    should_rebuild = should_rebuild or current_metadata != expected_metadata
                except Exception:
                    should_rebuild = True
            if should_rebuild:
                if existing:
                    client.delete_collection(name="smart_recipe_recipes")
                    collection = client.get_or_create_collection(name="smart_recipe_recipes")
                collection.add(
                    ids=[str(index) for index in range(len(self.recipes))],
                    documents=[recipe.searchable_text() for recipe in self.recipes],
                    metadatas=[
                        {"name": recipe.name, "embedding_backend": expected_metadata}
                        for recipe in self.recipes
                    ],
                    embeddings=embeddings,
                )

            self.chroma_collection = collection
            self.backend = "chroma"
            return True
        except Exception as exc:
            self.backend_errors.append(f"chroma init failed: {type(exc).__name__}: {exc}")
            return False

    def _init_faiss_backend(self) -> bool:
        try:
            import faiss
            import numpy as np

            embeddings = np.array(self.embedding_provider.embed([recipe.searchable_text() for recipe in self.recipes]), dtype="float32")
            index = build_hnsw_ip_index(faiss, embeddings.shape[1])
            index.add(embeddings)
            self.faiss_index = index
            self.backend = "faiss"
            return True
        except Exception as exc:
            self.backend_errors.append(f"faiss init failed: {type(exc).__name__}: {exc}")
            return False

    def _search_hybrid(
        self,
        query: str,
        top_k: int,
        vector_search,
    ) -> list[tuple[Recipe, float]]:
        recall_k = min(len(self.recipes), max(top_k * 4, top_k))
        vector_scores = self._vector_scores_with_hyde(query, recall_k, vector_search)
        keyword_scores = self._keyword_scores(query, recall_k)
        normalized_vector = _normalize_scores(vector_scores)
        normalized_keyword = _normalize_scores(keyword_scores)
        candidate_indices = set(normalized_vector) | set(normalized_keyword)

        ranked: list[tuple[Recipe, float]] = []
        for index in candidate_indices:
            recipe = self.recipes[index]
            score = (
                self.hybrid_vector_weight * normalized_vector.get(index, 0.0)
                + self.hybrid_keyword_weight * normalized_keyword.get(index, 0.0)
                + self._ingredient_overlap_bonus(query, recipe)
                + self._tag_overlap_bonus(query, recipe)
            )
            if score > 0:
                ranked.append((recipe, score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _vector_scores_with_hyde(self, query: str, top_k: int, vector_search) -> dict[int, float]:
        merged = dict(vector_search(query, top_k))
        self.last_hyde_query = ""
        self.last_hyde_generator = ""
        hyde = self.hyde.generate(query)
        if not hyde.enabled or not hyde.hypothetical_document:
            return merged

        self.last_hyde_query = hyde.hypothetical_document
        self.last_hyde_generator = hyde.generator
        for index, score in vector_search(hyde.hypothetical_document, top_k).items():
            merged[index] = max(merged.get(index, 0.0), score)
        return merged

    def _search_chroma_scores(self, query: str, top_k: int) -> dict[int, float]:
        query_embedding = self.embedding_provider.embed([query])[0]
        result = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self.recipes)),
        )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        scores = {}
        for raw_id, distance in zip(ids, distances):
            score = 1.0 - float(distance) if distance is not None else 0.0
            scores[int(raw_id)] = score
        return scores

    def _search_faiss_scores(self, query: str, top_k: int) -> dict[int, float]:
        import numpy as np

        query_embedding = np.array(self.embedding_provider.embed([query]), dtype="float32")
        scores, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.recipes)))
        ranked = {}
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            ranked[int(index)] = float(score)
        return ranked

    def _search_keyword(self, query: str, top_k: int) -> list[tuple[Recipe, float]]:
        keyword_scores = self._keyword_scores(query, top_k)
        ranked = []
        for index, score in keyword_scores.items():
            recipe = self.recipes[index]
            final_score = score + self._ingredient_overlap_bonus(query, recipe) + self._tag_overlap_bonus(query, recipe)
            if final_score > 0:
                ranked.append((recipe, final_score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def _keyword_scores(self, query: str, top_k: int) -> dict[int, float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        scored = []
        for index, doc_tokens in enumerate(self.documents):
            score = self._bm25_score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return dict(scored[:top_k])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        compact = text.lower()
        tokens = TOKEN_RE.findall(compact)
        chinese_chars = [char for char in compact if "\u4e00" <= char <= "\u9fff"]
        return tokens + chinese_chars

    @staticmethod
    def _build_bm25_idf(documents: list[list[str]]) -> dict[str, float]:
        doc_count = len(documents)
        token_doc_counts: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                token_doc_counts[token] = token_doc_counts.get(token, 0) + 1
        return {
            token: math.log(1 + (doc_count - count + 0.5) / (count + 0.5))
            for token, count in token_doc_counts.items()
        }

    def _bm25_score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        if not document_tokens or self.avg_doc_len <= 0:
            return 0.0
        k1 = 1.5
        b = 0.75
        term_counts: dict[str, int] = {}
        for token in document_tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        doc_len = len(document_tokens)
        denominator_base = k1 * (1 - b + b * doc_len / self.avg_doc_len)
        score = 0.0
        for token in dict.fromkeys(query_tokens):
            freq = term_counts.get(token, 0)
            if freq == 0:
                continue
            idf = self.bm25_idf.get(token, 0.0)
            score += idf * (freq * (k1 + 1)) / (freq + denominator_base)
        return score

    @staticmethod
    def _ingredient_overlap_bonus(query: str, recipe: Recipe) -> float:
        return sum(0.25 for ingredient in recipe.ingredients if ingredient in query)

    @staticmethod
    def _tag_overlap_bonus(query: str, recipe: Recipe) -> float:
        return sum(0.15 for tag in recipe.tags + recipe.suitable_for if tag in query)
