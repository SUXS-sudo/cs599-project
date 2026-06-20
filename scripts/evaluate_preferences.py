from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate preference memory and allergy/dislike filtering.")
    parser.add_argument(
        "--eval-file",
        default=str(ROOT_DIR / "data" / "evals" / "preference_memory_cases.jsonl"),
        help="JSONL file with setup messages, query, and blocked ingredients.",
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "redis"],
        default="memory",
        help="Memory backend used for this evaluation.",
    )
    parser.add_argument("--show-errors", action="store_true", help="Print failed cases.")
    args = parser.parse_args()

    os.environ["MEMORY_BACKEND"] = args.backend
    os.environ.setdefault("RAG_BACKEND", "bm25")
    os.environ.setdefault("RERANK_ENABLED", "false")

    from app.agents.preference_agent import PreferenceAgent
    from app.agents.recipe_agent import RecipeAgent
    from app.retriever import RecipeRetriever
    from app.services.memory import MemoryStore
    from app.services.redis_memory import RedisMemoryStore
    from app.state import AgentState

    cases = load_cases(Path(args.eval_file))
    memory_store = RedisMemoryStore(max_messages=10) if args.backend == "redis" else MemoryStore(max_messages=10)
    preference_agent = PreferenceAgent(memory_store)
    recipe_agent = RecipeAgent(RecipeRetriever(ROOT_DIR / "data" / "recipes.json"))

    ok = 0
    errors = []
    for index, case in enumerate(cases):
        session_id = f"{case['session_id']}-{index}"
        for message in case.get("setup_messages", []):
            state = AgentState(user_input=message, session_id=session_id, top_k=3)
            preference_agent.run(state)

        prefs = memory_store.get_preferences(session_id)
        query_state = AgentState(
            user_input=case["query"],
            session_id=session_id,
            top_k=3,
            meta={"user_preferences": prefs.to_dict()},
        )
        preference_agent.run(query_state)
        recipe_agent.run(query_state)

        expected_preferences = set(case.get("expected_preferences", []))
        expected_allergies = set(case.get("expected_allergies", []))
        expected_dislikes = set(case.get("expected_dislikes", []))
        blocked = case.get("blocked", [])
        returned = [recipe.name for recipe, _ in query_state.retrieved_docs]
        returned_ingredients = [
            ingredient
            for recipe, _ in query_state.retrieved_docs
            for ingredient in recipe.ingredients
        ]

        preferences_ok = expected_preferences.issubset(set(prefs.preferences))
        allergies_ok = expected_allergies.issubset(set(prefs.allergies))
        dislikes_ok = expected_dislikes.issubset(set(prefs.dislikes))
        filter_ok = not contains_blocked(returned_ingredients, blocked)
        case_ok = preferences_ok and allergies_ok and dislikes_ok and filter_ok and bool(returned)
        ok += int(case_ok)
        if not case_ok:
            errors.append(
                {
                    "session_id": session_id,
                    "query": case["query"],
                    "preferences": prefs.to_dict(),
                    "returned": returned,
                    "returned_ingredients": returned_ingredients,
                    "blocked": blocked,
                    "checks": {
                        "preferences_ok": preferences_ok,
                        "allergies_ok": allergies_ok,
                        "dislikes_ok": dislikes_ok,
                        "filter_ok": filter_ok,
                        "has_results": bool(returned),
                    },
                }
            )

    total = len(cases)
    print(f"cases={total}")
    print(f"memory_backend={getattr(memory_store, 'backend', 'memory')}")
    print(f"preference_success={ok / total:.2%} ({ok}/{total})")
    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if ok == total else 1


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contains_blocked(ingredients: list[str], blocked: list[str]) -> bool:
    return any(block in ingredient or ingredient in block for ingredient in ingredients for block in blocked if block)


if __name__ == "__main__":
    raise SystemExit(main())
