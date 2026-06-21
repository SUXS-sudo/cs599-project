from __future__ import annotations

from pathlib import Path

from src.services.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingProvider


def test_embedding_provider_defaults_to_bge_model(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sklearn")

    provider = EmbeddingProvider([])

    assert provider.model_name == DEFAULT_EMBEDDING_MODEL


def test_embedding_provider_prefers_local_bge_model_path(monkeypatch) -> None:
    model_dir = Path.cwd() / "models" / "bge-small-zh-v1.5"
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if path == model_dir / "modules.json":
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setenv("EMBEDDING_MODEL", "models/bge-small-zh-v1.5")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sklearn")

    provider = EmbeddingProvider([])

    assert provider.model_path == model_dir.resolve()
