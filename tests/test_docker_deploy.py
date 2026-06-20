from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_docker_compose_contains_database_services_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service in ("mysql:", "neo4j:", "redis:"):
        assert service in compose
    assert "app:" not in compose
    assert "mysql:8.0" in compose
    assert "neo4j:5" in compose
    assert "redis:7" in compose
    assert '"3307:3306"' in compose
    assert '"7474:7474"' in compose
    assert '"7687:7687"' in compose
    assert '"6379:6379"' in compose
    assert ".env.docker" not in compose


def test_mysql_init_sql_contains_schema_tables() -> None:
    init_sql = (ROOT / "docker" / "mysql" / "init.sql").read_text(encoding="utf-8").lower()

    assert "create table" in init_sql
    for table in (
        "recipes",
        "ingredients",
        "recipe_ingredients",
        "document_indexes",
        "document_chunks",
    ):
        assert f"create table if not exists {table}" in init_sql
    for removed_table in ("recipe_tags", "recipe_suitable_for", "user_profiles", "chat_turns", "eval_runs"):
        assert f"create table if not exists {removed_table}" not in init_sql


def test_gitignore_excludes_sensitive_and_generated_paths() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for entry in (".env", "logs/", ".test_artifacts/", "pytest-cache-files-*/"):
        assert entry in gitignore
