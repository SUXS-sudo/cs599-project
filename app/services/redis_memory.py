from __future__ import annotations

import json
import os
from urllib.parse import urlparse

from app.services.llm_client import load_dotenv
from app.services.memory import ChatMessage, MemoryStore, UserPreferences, merge_unique, preferences_from_dict, summarize_messages


class RedisMemoryStore(MemoryStore):
    backend = "redis"

    def __init__(self, max_messages: int = 10, ttl_seconds: int | None = None, redis_url: str | None = None) -> None:
        self.max_messages = max_messages
        load_dotenv(override=True)
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.redis_username = os.getenv("REDIS_USERNAME", "").strip() or None
        self.redis_password = os.getenv("REDIS_PASSWORD", "").strip() or None
        self.ttl_seconds = ttl_seconds or int(os.getenv("REDIS_TTL_SECONDS", str(7 * 24 * 3600)))
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("redis is required. Install it with: pip install redis") from exc
        self.client = redis.Redis.from_url(
            self.redis_url,
            username=self.redis_username if not url_contains_auth(self.redis_url) else None,
            password=self.redis_password if not url_contains_auth(self.redis_url) else None,
            decode_responses=True,
        )
        self.client.ping()

    def get_history(self, session_id: str) -> list[ChatMessage]:
        raw_items = self.client.lrange(self._history_key(session_id), 0, self.max_messages - 1)
        messages = []
        for raw_item in raw_items:
            try:
                item = json.loads(raw_item)
            except json.JSONDecodeError:
                continue
            messages.append(ChatMessage(role=str(item.get("role", "")), content=str(item.get("content", ""))))
        return messages

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self._history_key(session_id)
        existing = self.get_history(session_id)
        overflow = existing + [
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content=assistant_message),
        ]
        if len(overflow) > self.max_messages:
            self.client.set(
                self._summary_key(session_id),
                summarize_messages(overflow[: -self.max_messages]),
                ex=self.ttl_seconds,
            )
        pipe = self.client.pipeline()
        for message in (
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content=assistant_message),
        ):
            pipe.rpush(key, json.dumps(message.__dict__, ensure_ascii=False))
        pipe.ltrim(key, -self.max_messages, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def format_history(self, session_id: str) -> str:
        recent = super().format_history(session_id)
        summary = self.get_summary(session_id)
        if summary and not recent.startswith("Long-term summary:"):
            return f"Long-term summary: {summary}\nRecent conversation:\n{recent}"
        return recent

    def get_summary(self, session_id: str) -> str:
        return self.client.get(self._summary_key(session_id)) or ""

    def debug_session(self, session_id: str) -> dict:
        data = super().debug_session(session_id)
        data["summary"] = self.get_summary(session_id)
        return data

    def get_preferences(self, session_id: str) -> UserPreferences:
        raw = self.client.get(self._preferences_key(session_id))
        if not raw:
            return UserPreferences()
        try:
            return preferences_from_dict(json.loads(raw))
        except json.JSONDecodeError:
            return UserPreferences()

    def update_preferences(
        self,
        session_id: str,
        preferences: list[str] | None = None,
        allergies: list[str] | None = None,
        dislikes: list[str] | None = None,
    ) -> UserPreferences:
        current = self.get_preferences(session_id)
        current.preferences = merge_unique(current.preferences, preferences or [])
        current.allergies = merge_unique(current.allergies, allergies or [])
        current.dislikes = merge_unique(current.dislikes, dislikes or [])
        key = self._preferences_key(session_id)
        self.client.set(key, json.dumps(current.to_dict(), ensure_ascii=False), ex=self.ttl_seconds)
        return current

    def active_session_count(self) -> int:
        return int(len(list(self.client.scan_iter(match="session:*:history"))))

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"session:{session_id}:history"

    @staticmethod
    def _preferences_key(session_id: str) -> str:
        return f"session:{session_id}:prefs"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"session:{session_id}:summary"


def build_memory_store(max_messages: int = 10) -> MemoryStore:
    load_dotenv(override=True)
    backend = os.getenv("MEMORY_BACKEND", "memory").strip().lower()
    if backend != "redis":
        return MemoryStore(max_messages=max_messages)
    try:
        return RedisMemoryStore(max_messages=max_messages)
    except Exception:
        return MemoryStore(max_messages=max_messages)


def url_contains_auth(redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    return bool(parsed.username or parsed.password)
