from __future__ import annotations

import json
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from src.services.llm_client import load_dotenv
from src.services.mysql_store import MySQLStore
from src.services.neo4j_store import Neo4jStore


MAX_LIMIT = 200


class DatabaseBrowser:
    def __init__(self, mysql_store: MySQLStore | None = None, neo4j_store: Neo4jStore | None = None) -> None:
        self.mysql_store = mysql_store or MySQLStore()
        self.neo4j_store = neo4j_store or Neo4jStore()

    def overview(self) -> dict[str, Any]:
        return {
            "mysql": safe_section(self.mysql_overview),
            "redis": safe_section(self.redis_overview),
            "neo4j": safe_section(self.neo4j_overview),
        }

    def mysql_overview(self) -> dict[str, Any]:
        tables = []
        with self.mysql_store.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                rows = cursor.fetchall()
                table_key = f"Tables_in_{self.mysql_store.config.database}"
                for row in rows:
                    table = str(row.get(table_key) or next(iter(row.values())))
                    cursor.execute(f"SELECT COUNT(*) AS count FROM `{table}`")
                    count_row = cursor.fetchone()
                    cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                    columns = [str(item["Field"]) for item in cursor.fetchall()]
                    tables.append({"name": table, "rows": int(count_row["count"]), "columns": columns})
        return {
            "target": f"{self.mysql_store.config.user}@{self.mysql_store.config.host}:{self.mysql_store.config.port}/{self.mysql_store.config.database}",
            "tables": tables,
        }

    def mysql_table(self, table: str, limit: int = 50) -> dict[str, Any]:
        table_names = {item["name"] for item in self.mysql_overview()["tables"]}
        if table not in table_names:
            raise ValueError(f"Unknown MySQL table: {table}")
        safe_limit = normalize_limit(limit)
        with self.mysql_store.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                columns = [str(item["Field"]) for item in cursor.fetchall()]
                cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (safe_limit,))
                rows = [serialize_value(row) for row in cursor.fetchall()]
        return {"table": table, "limit": safe_limit, "columns": columns, "rows": rows}

    def redis_overview(self, limit: int = 100) -> dict[str, Any]:
        client = self.redis_client()
        info = client.info()
        keys = []
        for key in client.scan_iter(count=limit):
            key_text = decode_redis_value(key)
            keys.append(
                {
                    "key": key_text,
                    "type": decode_redis_value(client.type(key)),
                    "ttl": int(client.ttl(key)),
                }
            )
            if len(keys) >= normalize_limit(limit):
                break
        return {
            "target": os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            "dbsize": int(client.dbsize()),
            "used_memory_human": info.get("used_memory_human", ""),
            "keys": keys,
        }

    def redis_key(self, key: str, limit: int = 50) -> dict[str, Any]:
        client = self.redis_client()
        safe_limit = normalize_limit(limit)
        key_type = decode_redis_value(client.type(key))
        ttl = int(client.ttl(key))
        if key_type == "none":
            return {"key": key, "type": key_type, "ttl": ttl, "value": None}
        if key_type == "string":
            value: Any = decode_redis_value(client.get(key))
        elif key_type == "list":
            value = [decode_redis_value(item) for item in client.lrange(key, 0, safe_limit - 1)]
        elif key_type == "hash":
            value = {
                decode_redis_value(k): decode_redis_value(v)
                for k, v in list(client.hgetall(key).items())[:safe_limit]
            }
        elif key_type == "set":
            value = [decode_redis_value(item) for item in list(client.sscan_iter(key, count=safe_limit))[:safe_limit]]
        elif key_type == "zset":
            value = [
                {"member": decode_redis_value(member), "score": score}
                for member, score in client.zrange(key, 0, safe_limit - 1, withscores=True)
            ]
        else:
            value = f"Preview is not supported for Redis type: {key_type}"
        return {"key": key, "type": key_type, "ttl": ttl, "value": parse_jsonish(value)}

    def neo4j_overview(self) -> dict[str, Any]:
        labels = self.neo4j_store.execute_read(
            """
            MATCH (n)
            WITH labels(n)[0] AS label, count(n) AS count
            RETURN label, count
            ORDER BY label
            """
        )
        relationships = self.neo4j_store.execute_read(
            """
            MATCH ()-[r]->()
            WITH type(r) AS type, count(r) AS count
            RETURN type, count
            ORDER BY type
            """
        )
        return {
            "target": self.neo4j_store.config.uri,
            "database": self.neo4j_store.config.database or "default",
            "labels": [{"label": str(row["label"]), "count": int(row["count"])} for row in labels],
            "relationships": [{"type": str(row["type"]), "count": int(row["count"])} for row in relationships],
        }

    def neo4j_nodes(self, label: str, limit: int = 50) -> dict[str, Any]:
        labels = {item["label"] for item in self.neo4j_overview()["labels"]}
        if label not in labels:
            raise ValueError(f"Unknown Neo4j label: {label}")
        safe_limit = normalize_limit(limit)
        rows = self.neo4j_store.execute_read(
            f"""
            MATCH (n:`{label}`)
            RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties
            LIMIT $limit
            """,
            {"limit": safe_limit},
        )
        return {"label": label, "limit": safe_limit, "rows": serialize_value(rows)}

    def redis_client(self):
        load_dotenv(override=True)
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("redis is required. Install it with: pip install redis") from exc
        return redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            username=os.getenv("REDIS_USERNAME", "").strip() or None,
            password=os.getenv("REDIS_PASSWORD", "").strip() or None,
        )


def safe_section(func) -> dict[str, Any]:
    try:
        return {"ok": True, "data": func()}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def normalize_limit(limit: int) -> int:
    return min(max(int(limit), 1), MAX_LIMIT)


def serialize_value(value: Any) -> Any:
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def decode_redis_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, list):
        return [parse_jsonish(item) for item in value]
    if isinstance(value, dict):
        return {key: parse_jsonish(item) for key, item in value.items()}
    return value
