from __future__ import annotations

from pathlib import Path

from app.retriever import Recipe, RecipeRetriever
from app.services.hyde import HyDEResult


class FakeHyDE:
    enabled = True
    generator = "llm"

    def __init__(self, text: str = "") -> None:
        self.text = text

    def generate(self, query: str) -> HyDEResult:
        return HyDEResult(query, self.text, "llm", bool(self.text))


def make_recipe(name: str, ingredient: str) -> Recipe:
    return Recipe(
        name=name,
        ingredients=[ingredient],
        category="test",
        cooking_time="10 min",
        difficulty="easy",
        tags=[],
        calories=100,
        suitable_for=[],
        steps=f"{name} steps",
    )


def test_hybrid_search_merges_vector_and_keyword_candidates() -> None:
    retriever = object.__new__(RecipeRetriever)
    retriever.recipes = [make_recipe("semantic", "chicken"), make_recipe("keyword", "tomato")]
    retriever.hybrid_vector_weight = 0.5
    retriever.hybrid_keyword_weight = 0.5
    retriever._vector_scores_with_hyde = lambda query, top_k, vector_search: vector_search(query, top_k)
    retriever._keyword_scores = lambda query, top_k: {1: 8.0}

    hits = retriever._search_hybrid("tomato", 2, lambda query, top_k: {0: 0.9})

    assert [recipe.name for recipe, _ in hits] == ["keyword", "semantic"]


def test_retriever_caches_search_results(monkeypatch) -> None:
    artifact_dir = Path(__file__).resolve().parent.parent / ".test_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    path = artifact_dir / "retriever_cache_recipes.json"
    path.write_text(
        '[{"name":"缓存番茄炒蛋","ingredients":["番茄","鸡蛋"],"category":"家常菜",'
        '"cooking_time":"15分钟","difficulty":"简单","tags":["快手菜"],'
        '"calories":220,"suitable_for":["午餐"],"steps":"炒鸡蛋；炒番茄。"}]',
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_BACKEND", "bm25")
    monkeypatch.setenv("CACHE_DATA_VERSION", "retriever-cache-test")
    retriever = RecipeRetriever(path)

    first = retriever.search("缓存番茄炒蛋", 1)
    second = retriever.search("缓存番茄炒蛋", 1)

    assert retriever.last_cache_hit is True
    assert first[0][0].name == second[0][0].name == "缓存番茄炒蛋"
    assert first[0][1] == second[0][1]


def test_hyde_adds_extra_vector_recall_query() -> None:
    retriever = object.__new__(RecipeRetriever)
    retriever.hyde = FakeHyDE("hypothetical recipe document")
    retriever.last_hyde_query = ""
    retriever.last_hyde_generator = ""
    seen_queries = []

    def fake_vector_search(query: str, top_k: int) -> dict[int, float]:
        seen_queries.append(query)
        return {0: 0.3} if query == "original" else {1: 0.9}

    scores = retriever._vector_scores_with_hyde("original", 3, fake_vector_search)

    assert seen_queries == ["original", "hypothetical recipe document"]
    assert scores == {0: 0.3, 1: 0.9}
    assert retriever.last_hyde_generator == "llm"
