from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_cases(path: Path) -> list[dict[str, str]]:
    cases = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if "message" not in item or "intent" not in item:
            raise ValueError(f"{path}:{line_number} requires message and intent")
        cases.append(item)
    return cases


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_bool(primary: str, fallback: str | None = None, default: bool = False) -> bool:
    value = os.getenv(primary)
    if value is None and fallback:
        value = os.getenv(fallback)
    if value is None:
        return default
    return parse_bool(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Router Agent intent accuracy.")
    parser.add_argument(
        "--eval-file",
        default=str(ROOT_DIR / "data" / "evals" / "router_intents.jsonl"),
        help="JSONL file with message, intent, and optional target_agent.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use configured LLM router before rule fallback.",
    )
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="Print every wrong prediction.",
    )
    parser.add_argument(
        "--enable-database-agents",
        action="store_true",
        help="Force-enable SQL/Cypher database-agent routing for this evaluation.",
    )
    parser.add_argument(
        "--disable-database-agents",
        action="store_true",
        help="Force-disable SQL/Cypher database-agent routing for this evaluation.",
    )
    parser.add_argument(
        "--enable-v2",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable-v2",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    from app.agents.router_agent import RouterAgent
    from app.services.llm_client import LLMClient
    from app.state import AgentState

    cases = load_cases(Path(args.eval_file))
    force_enable = args.enable_database_agents or args.enable_v2
    force_disable = args.disable_database_agents or args.disable_v2
    if force_enable and force_disable:
        raise ValueError("Use only one enable/disable database-agent option.")
    if force_enable:
        enable_database_agents = True
    elif force_disable:
        enable_database_agents = False
    else:
        enable_database_agents = env_bool("ENABLE_DATABASE_AGENTS", "ENABLE_V2", default=False)
    router = RouterAgent(LLMClient() if args.use_llm else None, enable_database_agents=enable_database_agents)

    total = len(cases)
    correct_intent = 0
    correct_agent = 0
    by_intent: dict[str, dict[str, int]] = {}
    errors = []

    for case in cases:
        state = AgentState(user_input=case["message"], session_id="router-eval", top_k=3)
        result = router.run(state)
        expected_intent = case["intent"]
        expected_agent = case.get("target_agent")
        intent_ok = result.intent == expected_intent
        agent_ok = expected_agent is None or result.target_agent == expected_agent
        correct_intent += int(intent_ok)
        correct_agent += int(agent_ok)

        bucket = by_intent.setdefault(expected_intent, {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(intent_ok)

        if not intent_ok or not agent_ok:
            errors.append(
                {
                    "message": case["message"],
                    "expected_intent": expected_intent,
                    "predicted_intent": result.intent,
                    "expected_agent": expected_agent,
                    "predicted_agent": result.target_agent,
                    "router_mode": result.meta.get("router_mode"),
                }
            )

    intent_accuracy = correct_intent / total if total else 0.0
    agent_accuracy = correct_agent / total if total else 0.0
    print(f"cases={total}")
    print(f"enable_database_agents={str(enable_database_agents).lower()}")
    print(f"intent_accuracy={intent_accuracy:.2%} ({correct_intent}/{total})")
    print(f"agent_accuracy={agent_accuracy:.2%} ({correct_agent}/{total})")
    print("by_intent:")
    for intent, stats in sorted(by_intent.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"- {intent}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")

    if args.show_errors and errors:
        print("errors:")
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))

    return 0 if correct_intent == total and correct_agent == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
