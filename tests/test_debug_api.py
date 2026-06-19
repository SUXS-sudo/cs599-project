from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main


class FakeMemoryStore:
    backend = "memory"

    def debug_session(self, session_id: str) -> dict:
        return {
            "session_id": session_id,
            "preferences": {"preferences": ["清淡"], "allergies": [], "dislikes": ["牛肉"]},
            "history": [{"role": "user", "content": "我不吃牛肉"}],
            "turn_count": 1,
            "backend": "memory",
        }

    def active_session_count(self) -> int:
        return 2


class FakeStore:
    def __init__(self, data: dict) -> None:
        self.data = data

    def stats(self) -> dict:
        return self.data

    def get_user_preferences(self, session_id: str) -> dict:
        return {"preferences": ["低脂"], "allergies": [], "dislikes": []}


class FakeWorkflow:
    def __init__(self) -> None:
        self.sql_agent = type("SqlAgent", (), {"store": FakeStore({"recipes": 300})})()
        self.cypher_agent = type("CypherAgent", (), {"store": FakeStore({"Recipe": 300, "REL:USES": 900})})()


class FakeRetriever:
    def status(self) -> dict:
        return {
            "backend": "bm25",
            "embedding_backend": "local",
            "recipe_count": 300,
            "data_path": "data/recipes.json",
            "errors": [],
        }


def test_debug_session_returns_preferences_history_and_turn_count(monkeypatch) -> None:
    monkeypatch.setattr(main, "memory_store", FakeMemoryStore())
    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    client = TestClient(main.app)

    response = client.get("/debug/session/demo")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "demo"
    assert data["preferences"]["preferences"] == ["清淡"]
    assert data["mysql_preferences"]["ok"] is True
    assert data["mysql_preferences"]["data"]["preferences"] == ["低脂"]
    assert data["turn_count"] == 1
    assert data["history"][0]["role"] == "user"


def test_debug_stats_returns_service_sections(monkeypatch) -> None:
    monkeypatch.setattr(main, "memory_store", FakeMemoryStore())
    monkeypatch.setattr(main, "workflow", FakeWorkflow())
    monkeypatch.setattr(main, "retriever", FakeRetriever())
    client = TestClient(main.app)

    response = client.get("/debug/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["mysql"]["ok"] is True
    assert data["mysql"]["data"]["recipes"] == 300
    assert data["neo4j"]["data"]["Recipe"] == 300
    assert data["redis"]["active_sessions"] == 2
    assert data["retriever"]["recipe_count"] == 300
