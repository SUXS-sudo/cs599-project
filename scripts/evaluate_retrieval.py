from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if "query" not in item or "expected" not in item:
            raise ValueError(f"{path}:{line_number} requires query and expected")
        if not isinstance(item["expected"], list) or not item["expected"]:
            raise ValueError(f"{path}:{line_number} expected must be a non-empty list")
        cases.append(item)
    return cases


def hit_at(ranked_names: list[str], expected_names: set[str], k: int) -> bool:
    return any(name in expected_names for name in ranked_names[:k])


def first_hit_rank(ranked_names: list[str], expected_names: set[str]) -> int | None:
    for index, name in enumerate(ranked_names, start=1):
        if name in expected_names:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval hit rate.")
    parser.add_argument(
        "--eval-file",
        default=str(ROOT_DIR / "data" / "evals" / "rag_retrieval.jsonl"),
        help="JSONL file with query, expected recipe names, and optional category.",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("RAG_BACKEND", "bm25"),
        choices=["bm25", "chroma", "faiss"],
        help="Retriever backend to evaluate.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum number of retrieved recipes to inspect.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print cases that miss at max-k.",
    )
    args = parser.parse_args()

    os.environ["RAG_BACKEND"] = args.backend

    from app.retriever import RecipeRetriever

    cases = load_cases(Path(args.eval_file))
    retriever = RecipeRetriever(ROOT_DIR / "data" / "recipes.json")
    max_k = max(1, args.max_k)
    cutoffs = [k for k in (1, 3, 5) if k <= max_k]
    if max_k not in cutoffs:
        cutoffs.append(max_k)

    totals = {k: 0 for k in cutoffs}
    category_stats: dict[str, dict[int, int]] = {}
    category_totals: dict[str, int] = {}
    reciprocal_rank_sum = 0.0
    errors = []

    for case in cases:
        hits = retriever.search(case["query"], top_k=max_k)
        ranked_names = [recipe.name for recipe, _ in hits]
        expected_names = set(case["expected"])
        category = case.get("category", "uncategorized")
        category_totals[category] = category_totals.get(category, 0) + 1
        category_stats.setdefault(category, {k: 0 for k in cutoffs})

        rank = first_hit_rank(ranked_names, expected_names)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

        for k in cutoffs:
            is_hit = hit_at(ranked_names, expected_names, k)
            totals[k] += int(is_hit)
            category_stats[category][k] += int(is_hit)

        if not hit_at(ranked_names, expected_names, max_k):
            errors.append(
                {
                    "query": case["query"],
                    "expected": case["expected"],
                    "actual": ranked_names,
                    "category": category,
                }
            )

    total_cases = len(cases)
    print(f"cases={total_cases}")
    print(f"requested_backend={args.backend}")
    print(f"active_backend={retriever.backend}")
    print(f"embedding_backend={retriever.embedding_backend}")
    if retriever.backend_errors:
        print("backend_errors:")
        for error in retriever.backend_errors:
            print(f"- {error}")

    for k in cutoffs:
        rate = totals[k] / total_cases if total_cases else 0.0
        print(f"hit@{k}={rate:.2%} ({totals[k]}/{total_cases})")
    mrr = reciprocal_rank_sum / total_cases if total_cases else 0.0
    print(f"mrr@{max_k}={mrr:.4f}")

    print("by_category:")
    for category in sorted(category_totals):
        pieces = []
        for k in cutoffs:
            count = category_stats[category][k]
            total = category_totals[category]
            rate = count / total if total else 0.0
            pieces.append(f"hit@{k}={rate:.2%} ({count}/{total})")
        print(f"- {category}: " + ", ".join(pieces))

    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
