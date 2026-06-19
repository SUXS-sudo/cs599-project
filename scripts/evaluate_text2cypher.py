from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> int:
    from app.agents.cypher_agent import CypherAgent
    from app.agents.router_agent import RouterAgent
    from app.state import AgentState

    cases = load_cases(ROOT_DIR / "data" / "evals" / "text2cypher_cases.jsonl")
    router = RouterAgent()
    agent = CypherAgent()
    ok = 0
    for case in cases:
        state = AgentState(user_input=case["message"], session_id="text2cypher-eval", top_k=5)
        state = router.run(state)
        if state.intent != case["expected_intent"]:
            print(f"route_miss={case['message']} expected={case['expected_intent']} actual={state.intent}")
            continue
        state = agent.run(state)
        if state.meta.get("cypher_status") != "ok":
            print(f"cypher_miss={case['message']} status={state.meta.get('cypher_status')} output={state.agent_output}")
            continue
        if all(item in state.agent_output for item in case.get("contains", [])):
            ok += 1
        else:
            print(f"answer_miss={case['message']} output={state.agent_output}")
    total = len(cases)
    print(f"cases={total}")
    print(f"text2cypher_success={ok / total:.2%} ({ok}/{total})")
    return 0 if ok == total else 1


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
