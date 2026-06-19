from __future__ import annotations

import os
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_MODEL = "bge-small-zh-v1.5"


class EmbeddingProvider:
    """Embedding provider for Chroma/FAISS backends.

    Priority:
    1. Online OpenAI-compatible embedding API, if configured.
    2. sentence-transformers multilingual model, if installed and loadable.
    3. Local TF-IDF + SVD dense embeddings, no model download required.
    4. Hashing embeddings as the final fallback.
    """

    def __init__(self, corpus: list[str] | None = None) -> None:
        self.corpus = corpus or []
        self.model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.provider = os.getenv("EMBEDDING_PROVIDER", "auto").strip().lower()
        self.base_url = os.getenv("EMBEDDING_BASE_URL", os.getenv("BASE_URL", "")).strip()
        self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("API_KEY", "")).strip()
        self.timeout = float(os.getenv("EMBEDDING_TIMEOUT", "30"))
        self.model_path = self._resolve_local_model_path(self.model_name)
        self.errors: list[str] = []
        self.backend = "hashing"
        self._model = None
        self._vectorizer = None
        self._svd = None
        self._online_client: Any | None = None
        self._load_online_embedding()
        if self._online_client is None and self.provider not in {"sklearn", "tfidf"}:
            self._load_sentence_transformer()
        if self._model is None and self.corpus:
            self._fit_local_svd(self.corpus)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self._online_client is not None:
            vectors = self._embed_online(texts)
            if vectors is not None:
                return vectors
            self._online_client = None
            self.backend = "hashing"
            if self.provider not in {"sklearn", "tfidf"}:
                self._load_sentence_transformer()
            if self._model is None and self.corpus and self._vectorizer is None:
                self._fit_local_svd(self.corpus)

        if self._model is not None:
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype("float32").tolist()

        if self._vectorizer is not None and self._svd is not None:
            from sklearn.preprocessing import normalize

            matrix = self._vectorizer.transform(texts)
            dense = self._svd.transform(matrix)
            dense = normalize(dense, norm="l2", axis=1)
            return dense.astype("float32").tolist()

        return self._hash_embeddings(texts)

    def _load_online_embedding(self) -> None:
        if self.provider not in {"openai", "openai_compatible"}:
            return
        if not self.api_key or not self.model_name:
            return
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._online_client = OpenAI(**kwargs)
            self.backend = f"online_openai:{self.model_name}"
        except Exception as exc:
            self.errors.append(f"online embedding init failed: {type(exc).__name__}: {exc}")
            self._online_client = None
            self.backend = "hashing"

    def _embed_online(self, texts: list[str]) -> list[list[float]] | None:
        if self._online_client is None:
            return None
        try:
            response = self._online_client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            vectors = [item.embedding for item in response.data]
            return self._normalize(vectors)
        except Exception as exc:
            self.errors.append(f"online embedding request failed: {type(exc).__name__}: {exc}")
            return None

    def _load_sentence_transformer(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            if self.model_path is not None:
                self._model = SentenceTransformer(str(self.model_path))
                self.backend = f"local_sentence_transformers:{self.model_path}"
                return
            if self.provider == "local":
                self._model = None
                self.backend = "hashing"
                return
            self._model = SentenceTransformer(self.model_name)
            self.backend = f"sentence_transformers:{self.model_name}"
        except Exception as exc:
            self.errors.append(f"sentence_transformer load failed: {type(exc).__name__}: {exc}")
            self._model = None
            self.backend = "hashing"

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> Path | None:
        candidates = []
        model_path = Path(model_name)
        if model_path.is_absolute():
            candidates.append(model_path)
        else:
            candidates.extend(
                [
                    ROOT_DIR / model_path,
                    ROOT_DIR / "models" / model_name,
                    Path.cwd() / model_path,
                    Path.cwd() / "models" / model_name,
                ]
            )
        for candidate in candidates:
            if (candidate / "modules.json").exists():
                return candidate.resolve()
        return None

    def _fit_local_svd(self, corpus: list[str]) -> None:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4))
            matrix = vectorizer.fit_transform(corpus)
            max_components = min(128, matrix.shape[0] - 1, matrix.shape[1] - 1)
            if max_components < 2:
                return
            svd = TruncatedSVD(n_components=max_components, random_state=42)
            svd.fit(matrix)
            self._vectorizer = vectorizer
            self._svd = svd
            self.backend = f"local_tfidf_svd:{max_components}"
        except Exception as exc:
            self.errors.append(f"local tfidf_svd embedding init failed: {type(exc).__name__}: {exc}")
            self._vectorizer = None
            self._svd = None
            self.backend = "hashing"

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> list[list[float]]:
        import numpy as np
        from sklearn.preprocessing import normalize

        matrix = np.asarray(vectors, dtype="float32")
        matrix = normalize(matrix, norm="l2", axis=1)
        return matrix.astype("float32").tolist()

    @staticmethod
    def _hash_embeddings(texts: list[str]) -> list[list[float]]:
        import numpy as np
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            n_features=1024,
            alternate_sign=False,
            norm=None,
        )
        matrix = vectorizer.transform(texts)
        matrix = normalize(matrix, norm="l2", axis=1)
        return np.asarray(matrix.toarray(), dtype="float32").tolist()
