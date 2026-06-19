from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end v2 /chat flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="FastAPI base URL.")
    parser.add_argument(
        "--eval-file",
        default=str(ROOT_DIR / "data" / "evals" / "chat_v2_cases.jsonl"),
        help="JSONL v2 chat eval file.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    parser.add_argument("--show-errors", action="store_true", help="Print failed cases.")
    args = parser.parse_args()

    cases = load_cases(Path(args.eval_file))
    ok = 0
    errors = []
    for index, case in enumerate(cases):
        session_id = f"chat-v2-{case['name']}-{int(time.time())}-{index}"
        responses = []
        failed = False
        for turn, message in enumerate(case["messages"]):
            try:
                response = call_chat(args.base_url, message, session_id, args.timeout)
            except Exception as exc:
                errors.append({"case": case["name"], "error": f"{type(exc).__name__}: {exc}"})
                failed = True
                break
            responses.append(response)

            expected_intent = case.get("expect_intents", [None] * len(case["messages"]))[turn]
            expected_agent = case.get("expect_agents", [None] * len(case["messages"]))[turn]
            if expected_intent and response.get("intent") != expected_intent:
                errors.append({"case": case["name"], "turn": turn, "expected_intent": expected_intent, "actual": response})
                failed = True
            if expected_agent and response.get("agent") != expected_agent:
                errors.append({"case": case["name"], "turn": turn, "expected_agent": expected_agent, "actual": response})
                failed = True

        if failed:
            continue

        final_response = responses[-1]
        answer = final_response.get("answer", "")
        contains_ok = all(text in answer for text in case.get("contains", []))
        blocked_ok = not contains_blocked_recipes(final_response.get("recipes", []), case.get("blocked_recipe_ingredients", []))
        if contains_ok and blocked_ok:
            ok += 1
        else:
            errors.append(
                {
                    "case": case["name"],
                    "contains_ok": contains_ok,
                    "blocked_ok": blocked_ok,
                    "final_response": final_response,
                }
            )

    total = len(cases)
    print(f"cases={total}")
    print(f"chat_v2_success={ok / total:.2%} ({ok}/{total})")
    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    return 0 if ok == total else 1


def call_chat(base_url: str, message: str, session_id: str, timeout: int) -> dict:
    payload = json.dumps(
        {"message": message, "session_id": session_id, "top_k": 3},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def contains_blocked_recipes(recipes: list[dict], blocked: list[str]) -> bool:
    for recipe in recipes:
        for ingredient in recipe.get("ingredients", []):
            if any(block in ingredient or ingredient in block for block in blocked if block):
                return True
    return False


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
