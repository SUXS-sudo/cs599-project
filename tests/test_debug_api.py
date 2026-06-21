from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main


class FakeMemoryStore:
    backend = "memory"

    def __init__(self) -> None:
        self.deleted_session_id = ""

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

    def delete_session(self, session_id: str) -> int:
        self.deleted_session_id = session_id
        return 3


class FakeStore:
    def __init__(self, data: dict) -> None:
        self.data = data

    def stats(self) -> dict:
        return self.data

class FakeWorkflow:
    def __init__(self) -> None:
        self.sql_agent = type("SqlAgent", (), {"store": FakeStore({"recipes": 300})})()
        self.cypher_agent = type("CypherAgent", (), {"store": FakeStore({"Recipe": 300, "REL:USES": 900})})()
        self.deleted_checkpoint = ""

    def delete_checkpoint(self, session_id: str) -> bool:
        self.deleted_checkpoint = session_id
        return True


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
    assert data["turn_count"] == 1
    assert data["history"][0]["role"] == "user"


def test_root_redirects_to_chat_ui() -> None:
    client = TestClient(main.app, follow_redirects=False)

    response = client.get("/")

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/"


def test_delete_chat_session_removes_backend_memory(monkeypatch) -> None:
    store = FakeMemoryStore()
    workflow = FakeWorkflow()
    monkeypatch.setattr(main, "memory_store", store)
    monkeypatch.setattr(main, "workflow", workflow)
    client = TestClient(main.app)

    response = client.delete("/chat/session/chat-123")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "session_id": "chat-123",
        "deleted_keys": 3,
        "deleted_checkpoint": True,
    }
    assert store.deleted_session_id == "chat-123"
    assert workflow.deleted_checkpoint == "chat-123"


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
