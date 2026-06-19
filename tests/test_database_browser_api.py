from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main


class FakeDatabaseBrowser:
    def overview(self) -> dict:
        return {
            "mysql": {"ok": True, "data": {"target": "mysql", "tables": [{"name": "recipes", "rows": 1}]}},
            "redis": {"ok": True, "data": {"target": "redis", "dbsize": 1, "keys": [{"key": "session:demo:history"}]}},
            "neo4j": {"ok": True, "data": {"target": "neo4j", "labels": [{"label": "Recipe", "count": 1}]}},
        }

    def mysql_table(self, table: str, limit: int = 50) -> dict:
        return {"table": table, "limit": limit, "rows": [{"id": 1, "name": "番茄炒蛋"}]}

    def redis_key(self, key: str, limit: int = 50) -> dict:
        return {"key": key, "limit": limit, "type": "string", "value": "demo"}

    def neo4j_nodes(self, label: str, limit: int = 50) -> dict:
        return {"label": label, "limit": limit, "rows": [{"properties": {"name": "番茄炒蛋"}}]}


def test_database_overview_returns_three_sections(monkeypatch) -> None:
    monkeypatch.setattr(main, "database_browser", FakeDatabaseBrowser())
    client = TestClient(main.app)

    response = client.get("/debug/database/overview")

    assert response.status_code == 200
    data = response.json()
    assert data["mysql"]["data"]["tables"][0]["name"] == "recipes"
    assert data["redis"]["data"]["dbsize"] == 1
    assert data["neo4j"]["data"]["labels"][0]["label"] == "Recipe"


def test_database_preview_routes_wrap_results(monkeypatch) -> None:
    monkeypatch.setattr(main, "database_browser", FakeDatabaseBrowser())
    client = TestClient(main.app)

    mysql = client.get("/debug/database/mysql/recipes?limit=5").json()
    redis = client.get("/debug/database/redis/key?key=session%3Ademo%3Ahistory&limit=5").json()
    neo4j = client.get("/debug/database/neo4j/Recipe?limit=5").json()

    assert mysql["ok"] is True
    assert mysql["data"]["rows"][0]["name"] == "番茄炒蛋"
    assert redis["data"]["value"] == "demo"
    assert neo4j["data"]["rows"][0]["properties"]["name"] == "番茄炒蛋"
