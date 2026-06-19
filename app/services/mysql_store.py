from __future__ import annotations

import json
import os
import re
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.llm_client import load_dotenv
from app.services.logger import get_logger


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TIME_RE = re.compile(r"(\d+)")
logger = get_logger("services.mysql")


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS recipes (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(128) NOT NULL UNIQUE,
      category VARCHAR(64),
      cooking_time_text VARCHAR(64),
      cooking_time_minutes INT,
      difficulty VARCHAR(32),
      calories INT,
      steps TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS ingredients (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      name VARCHAR(128) NOT NULL UNIQUE,
      category VARCHAR(64)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS recipe_ingredients (
      recipe_id BIGINT NOT NULL,
      ingredient_id BIGINT NOT NULL,
      amount_text VARCHAR(128),
      PRIMARY KEY (recipe_id, ingredient_id),
      CONSTRAINT fk_recipe_ingredients_recipe
        FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE,
      CONSTRAINT fk_recipe_ingredients_ingredient
        FOREIGN KEY (ingredient_id) REFERENCES ingredients(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS recipe_tags (
      recipe_id BIGINT NOT NULL,
      tag VARCHAR(64) NOT NULL,
      PRIMARY KEY (recipe_id, tag),
      CONSTRAINT fk_recipe_tags_recipe
        FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS recipe_suitable_for (
      recipe_id BIGINT NOT NULL,
      target VARCHAR(64) NOT NULL,
      PRIMARY KEY (recipe_id, target),
      CONSTRAINT fk_recipe_suitable_for_recipe
        FOREIGN KEY (recipe_id) REFERENCES recipes(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
      session_id VARCHAR(128) PRIMARY KEY,
      preferences JSON,
      allergies JSON,
      dislikes JSON,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_turns (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      session_id VARCHAR(128) NOT NULL,
      user_message TEXT NOT NULL,
      assistant_message TEXT,
      intent VARCHAR(64),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_chat_turns_session_created (session_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      eval_type VARCHAR(64) NOT NULL,
      backend VARCHAR(64),
      metrics JSON NOT NULL,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      INDEX idx_eval_runs_type_created (eval_type, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS document_indexes (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      index_name VARCHAR(128) NOT NULL UNIQUE,
      index_path VARCHAR(512) NOT NULL,
      metadata_path VARCHAR(512) NOT NULL,
      embedding_backend VARCHAR(512),
      index_type VARCHAR(64),
      hnsw_config JSON,
      chunk_count INT NOT NULL DEFAULT 0,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS document_chunks (
      id BIGINT PRIMARY KEY AUTO_INCREMENT,
      index_name VARCHAR(128) NOT NULL,
      chunk_id VARCHAR(191) NOT NULL,
      source VARCHAR(512),
      source_type VARCHAR(64),
      text MEDIUMTEXT NOT NULL,
      start_char INT,
      end_char INT,
      metadata JSON,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      UNIQUE KEY uq_document_chunks_index_chunk (index_name, chunk_id),
      INDEX idx_document_chunks_index_name (index_name),
      INDEX idx_document_chunks_source (source(191)),
      CONSTRAINT fk_document_chunks_index
        FOREIGN KEY (index_name) REFERENCES document_indexes(index_name) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = "utf8mb4"
    ssl_disabled: bool = True

    @classmethod
    def from_env(cls) -> "MySQLConfig":
        load_dotenv(override=True)
        return cls(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "smart_recipe"),
            password=os.getenv("MYSQL_PASSWORD", "smart_recipe_password"),
            database=os.getenv("MYSQL_DATABASE", "smart_recipe"),
            ssl_disabled=parse_bool(os.getenv("MYSQL_SSL_DISABLED", "true")),
        )


class MySQLStore:
    def __init__(self, config: MySQLConfig | None = None) -> None:
        self.config = config or MySQLConfig.from_env()

    def connect(self, use_database: bool = True):
        try:
            import pymysql
        except ModuleNotFoundError as exc:
            raise RuntimeError("pymysql is required. Install it with: pip install pymysql") from exc

        kwargs: dict[str, Any] = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "charset": self.config.charset,
            "autocommit": False,
            "connect_timeout": 10,
            "cursorclass": pymysql.cursors.DictCursor,
        }
        if "ssl_disabled" in inspect.signature(pymysql.connect).parameters:
            kwargs["ssl_disabled"] = self.config.ssl_disabled
        if use_database:
            kwargs["database"] = self.config.database
        return pymysql.connect(**kwargs)

    def ensure_database(self) -> None:
        with self.connect(use_database=False) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.config.database}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            connection.commit()

    def ensure_schema(self) -> None:
        self.ensure_database()
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                for statement in SCHEMA_STATEMENTS:
                    cursor.execute(statement)
            connection.commit()

    def reset_recipe_tables(self) -> None:
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                for table in (
                    "recipe_ingredients",
                    "recipe_tags",
                    "recipe_suitable_for",
                    "ingredients",
                    "recipes",
                ):
                    cursor.execute(f"TRUNCATE TABLE {table}")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            connection.commit()

    def reset_document_tables(self) -> None:
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                cursor.execute("TRUNCATE TABLE document_chunks")
                cursor.execute("TRUNCATE TABLE document_indexes")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
            connection.commit()

    def import_recipes(self, recipes: list[dict[str, Any]]) -> dict[str, int]:
        self.ensure_schema()
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                imported_recipes = 0
                linked_ingredients = 0
                linked_tags = 0
                linked_targets = 0
                for item in recipes:
                    recipe_id = self._upsert_recipe(cursor, item)
                    self._clear_recipe_links(cursor, recipe_id)
                    linked_ingredients += self._insert_ingredients(cursor, recipe_id, item.get("ingredients", []))
                    linked_tags += self._insert_tags(cursor, recipe_id, item.get("tags", []))
                    linked_targets += self._insert_targets(cursor, recipe_id, item.get("suitable_for", []))
                    imported_recipes += 1
            connection.commit()
        return {
            "recipes": imported_recipes,
            "recipe_ingredients": linked_ingredients,
            "recipe_tags": linked_tags,
            "recipe_suitable_for": linked_targets,
        }

    def import_document_index(
        self,
        index_name: str,
        index_path: str,
        metadata_path: str,
        metadata: dict[str, Any],
        batch_size: int = 500,
    ) -> dict[str, int]:
        self.ensure_schema()
        chunks = list(metadata.get("chunks") or [])
        batch_size = max(1, batch_size)
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO document_indexes
                      (index_name, index_path, metadata_path, embedding_backend, index_type, hnsw_config, chunk_count)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      index_path = VALUES(index_path),
                      metadata_path = VALUES(metadata_path),
                      embedding_backend = VALUES(embedding_backend),
                      index_type = VALUES(index_type),
                      hnsw_config = VALUES(hnsw_config),
                      chunk_count = VALUES(chunk_count)
                    """,
                    (
                        index_name,
                        index_path,
                        metadata_path,
                        str(metadata.get("embedding_backend") or ""),
                        str(metadata.get("index_type") or ""),
                        json.dumps(metadata.get("hnsw") or {}, ensure_ascii=False),
                        len(chunks),
                    ),
                )
                cursor.execute("DELETE FROM document_chunks WHERE index_name = %s", (index_name,))
                imported_chunks = 0
                rows = [
                    (
                        index_name,
                        str(chunk["chunk_id"]),
                        str(chunk.get("source") or ""),
                        str(chunk.get("source_type") or ""),
                        str(chunk.get("text") or ""),
                        int(chunk.get("start_char") or 0),
                        int(chunk.get("end_char") or 0),
                        json.dumps(chunk.get("metadata") or {}, ensure_ascii=False),
                    )
                    for chunk in chunks
                ]
                for start in range(0, len(rows), batch_size):
                    batch = rows[start : start + batch_size]
                    cursor.executemany(
                        """
                        INSERT INTO document_chunks
                          (index_name, chunk_id, source, source_type, text, start_char, end_char, metadata)
                        VALUES
                          (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                          source = VALUES(source),
                          source_type = VALUES(source_type),
                          text = VALUES(text),
                          start_char = VALUES(start_char),
                          end_char = VALUES(end_char),
                          metadata = VALUES(metadata)
                        """,
                        batch,
                    )
                    imported_chunks += len(batch)
            connection.commit()
        return {"document_indexes": 1, "document_chunks": imported_chunks}

    def stats(self) -> dict[str, int]:
        tables = (
            "recipes",
            "ingredients",
            "recipe_ingredients",
            "recipe_tags",
            "recipe_suitable_for",
            "user_profiles",
            "chat_turns",
            "eval_runs",
            "document_indexes",
            "document_chunks",
        )
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                result = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) AS count FROM {table}")
                    row = cursor.fetchone()
                    result[table] = int(row["count"])
                logger.info("mysql stats collected table_count=%s", len(result))
                return result

    def read_query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, parameters)
                rows = list(cursor.fetchall())
                logger.debug("mysql read query rows=%s", len(rows))
                return rows

    def upsert_user_preferences(
        self,
        session_id: str,
        preferences: list[str],
        allergies: list[str],
        dislikes: list[str],
    ) -> None:
        self.ensure_schema()
        with self.connect(use_database=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO user_profiles (session_id, preferences, allergies, dislikes)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                      preferences = VALUES(preferences),
                      allergies = VALUES(allergies),
                      dislikes = VALUES(dislikes)
                    """,
                    (
                        session_id,
                        json.dumps(preferences, ensure_ascii=False),
                        json.dumps(allergies, ensure_ascii=False),
                        json.dumps(dislikes, ensure_ascii=False),
                    ),
                )
            connection.commit()

    def get_user_preferences(self, session_id: str) -> dict[str, list[str]] | None:
        rows = self.read_query(
            """
            SELECT preferences, allergies, dislikes
            FROM user_profiles
            WHERE session_id = %s
            LIMIT 1
            """,
            (session_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return {
            "preferences": json.loads(row.get("preferences") or "[]"),
            "allergies": json.loads(row.get("allergies") or "[]"),
            "dislikes": json.loads(row.get("dislikes") or "[]"),
        }

    @staticmethod
    def _upsert_recipe(cursor, item: dict[str, Any]) -> int:
        cursor.execute(
            """
            INSERT INTO recipes
              (name, category, cooking_time_text, cooking_time_minutes, difficulty, calories, steps)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              id = LAST_INSERT_ID(id),
              category = VALUES(category),
              cooking_time_text = VALUES(cooking_time_text),
              cooking_time_minutes = VALUES(cooking_time_minutes),
              difficulty = VALUES(difficulty),
              calories = VALUES(calories),
              steps = VALUES(steps)
            """,
            (
                item["name"],
                item.get("category", ""),
                item.get("cooking_time", ""),
                parse_minutes(item.get("cooking_time", "")),
                item.get("difficulty", ""),
                int(item.get("calories", 0)),
                item.get("steps", ""),
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _clear_recipe_links(cursor, recipe_id: int) -> None:
        cursor.execute("DELETE FROM recipe_ingredients WHERE recipe_id = %s", (recipe_id,))
        cursor.execute("DELETE FROM recipe_tags WHERE recipe_id = %s", (recipe_id,))
        cursor.execute("DELETE FROM recipe_suitable_for WHERE recipe_id = %s", (recipe_id,))

    @staticmethod
    def _insert_ingredients(cursor, recipe_id: int, ingredients: list[str]) -> int:
        count = 0
        for ingredient in ingredients:
            name = ingredient.strip()
            if not name:
                continue
            cursor.execute(
                """
                INSERT INTO ingredients (name)
                VALUES (%s)
                ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), name = VALUES(name)
                """,
                (name,),
            )
            ingredient_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO recipe_ingredients (recipe_id, ingredient_id, amount_text)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE amount_text = VALUES(amount_text)
                """,
                (recipe_id, ingredient_id, None),
            )
            count += 1
        return count

    @staticmethod
    def _insert_tags(cursor, recipe_id: int, tags: list[str]) -> int:
        count = 0
        for tag in tags:
            name = tag.strip()
            if not name:
                continue
            cursor.execute(
                """
                INSERT INTO recipe_tags (recipe_id, tag)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE tag = VALUES(tag)
                """,
                (recipe_id, name),
            )
            count += 1
        return count

    @staticmethod
    def _insert_targets(cursor, recipe_id: int, targets: list[str]) -> int:
        count = 0
        for target in targets:
            name = target.strip()
            if not name:
                continue
            cursor.execute(
                """
                INSERT INTO recipe_suitable_for (recipe_id, target)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE target = VALUES(target)
                """,
                (recipe_id, name),
            )
            count += 1
        return count


def parse_minutes(text: str) -> int | None:
    match = TIME_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_recipe_json(path: Path = ROOT_DIR / "data" / "recipes.json") -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_sql(database: str = "smart_recipe") -> str:
    statements = [
        f"CREATE DATABASE IF NOT EXISTS `{database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
        f"USE `{database}`",
        *[statement.strip() for statement in SCHEMA_STATEMENTS],
    ]
    return ";\n\n".join(statements) + ";\n"
