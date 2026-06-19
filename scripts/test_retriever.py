from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    backend = sys.argv[1] if len(sys.argv) > 1 else os.getenv("RAG_BACKEND", "bm25")
    query = " ".join(sys.argv[2:]).strip() or "鸡胸肉 西兰花 减脂 晚餐"
    os.environ["RAG_BACKEND"] = backend

    from app.retriever import RecipeRetriever

    retriever = RecipeRetriever(ROOT_DIR / "data" / "recipes.json")
    hits = retriever.search(query, top_k=3)
    print(f"requested_backend={backend}")
    print(f"active_backend={retriever.backend}")
    print(f"embedding_backend={retriever.embedding_backend}")
    if retriever.backend_errors:
        print("backend_errors:")
        for error in retriever.backend_errors:
            print(f"- {error}")
    print(f"query={query}")
    for recipe, score in hits:
        print(f"- {recipe.name} | score={score:.4f} | ingredients={','.join(recipe.ingredients)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
