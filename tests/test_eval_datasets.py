from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "evals"


EXPECTED_COUNTS = {
    "text2sql_cases.jsonl": 30,
    "text2cypher_cases.jsonl": 30,
    "preference_memory_cases.jsonl": 20,
    "chat_v2_cases.jsonl": 30,
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        assert line.strip(), f"{path.name}:{line_no} is blank"
        item = json.loads(line)
        assert isinstance(item, dict), f"{path.name}:{line_no} is not an object"
        rows.append(item)
    return rows


def test_eval_jsonl_files_are_valid_and_large_enough() -> None:
    for filename, expected_count in EXPECTED_COUNTS.items():
        rows = read_jsonl(EVAL_DIR / filename)
        assert len(rows) >= expected_count


def test_text2sql_case_schema() -> None:
    for row in read_jsonl(EVAL_DIR / "text2sql_cases.jsonl"):
        assert row["message"]
        assert row["expected_intent"] == "structured_recipe_query"
        assert isinstance(row["contains"], list)
        assert row["contains"]


def test_text2cypher_case_schema() -> None:
    for row in read_jsonl(EVAL_DIR / "text2cypher_cases.jsonl"):
        assert row["message"]
        assert row["expected_intent"] == "relationship_query"
        assert isinstance(row["contains"], list)
        assert row["contains"]


def test_preference_memory_case_schema() -> None:
    for row in read_jsonl(EVAL_DIR / "preference_memory_cases.jsonl"):
        assert row["session_id"]
        assert isinstance(row["setup_messages"], list)
        assert row["query"]
        assert isinstance(row["blocked"], list)
        assert any(key in row for key in ("expected_dislikes", "expected_preferences", "expected_allergies"))


def test_chat_v2_case_schema() -> None:
    names = set()
    for row in read_jsonl(EVAL_DIR / "chat_v2_cases.jsonl"):
        assert row["name"] not in names
        names.add(row["name"])
        assert isinstance(row["messages"], list)
        assert row["messages"]
        assert isinstance(row["expect_intents"], list)
        assert isinstance(row["expect_agents"], list)
        assert len(row["expect_intents"]) == len(row["expect_agents"])
        assert isinstance(row["contains"], list)
