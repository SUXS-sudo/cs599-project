from __future__ import annotations

import json
import os
from urllib.parse import urlparse

from src.services.context_budget import ContextBudgetManager, SummaryGenerator
from src.services.llm_client import load_dotenv
from src.services.memory import ChatMessage, MemoryStore, UserPreferences, merge_unique, preferences_from_dict


class RedisMemoryStore(MemoryStore):
    backend = "redis"

    def __init__(
        self,
        max_messages: int | None = None,
        ttl_seconds: int | None = None,
        redis_url: str | None = None,
        budget_manager: ContextBudgetManager | None = None,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        super().__init__(
            max_messages=max_messages,
            budget_manager=budget_manager,
            summary_generator=summary_generator,
        )
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
        try:
            self.client.ping()
        except redis.exceptions.AuthenticationError as exc:
            if "without any password configured" not in str(exc).lower():
                raise
            self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.client.ping()

    def get_history(self, session_id: str) -> list[ChatMessage]:
        raw_items = self.client.lrange(self._history_key(session_id), 0, -1)
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
        lock = self.client.lock(
            f"session:{session_id}:memory-lock",
            timeout=60,
            blocking_timeout=10,
        )
        with lock:
            messages = self.get_history(session_id) + [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=assistant_message),
            ]
            summary = self.get_summary(session_id)
            result = self.budget_manager.compact(
                summary,
                messages,
                max_messages=self.max_messages,
            )
            pipe = self.client.pipeline(transaction=True)
            pipe.delete(key)
            if result.recent_messages:
                pipe.rpush(
                    key,
                    *(json.dumps(message.__dict__, ensure_ascii=False) for message in result.recent_messages),
                )
                pipe.expire(key, self.ttl_seconds)
            if result.summary:
                pipe.set(self._summary_key(session_id), result.summary, ex=self.ttl_seconds)
            else:
                pipe.delete(self._summary_key(session_id))
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

    def delete_session(self, session_id: str) -> int:
        return int(
            self.client.delete(
                self._history_key(session_id),
                self._preferences_key(session_id),
                self._summary_key(session_id),
            )
        )

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"session:{session_id}:history"

    @staticmethod
    def _preferences_key(session_id: str) -> str:
        return f"session:{session_id}:prefs"

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"session:{session_id}:summary"


def build_memory_store(
    max_messages: int | None = None,
    budget_manager: ContextBudgetManager | None = None,
    summary_generator: SummaryGenerator | None = None,
) -> MemoryStore:
    load_dotenv(override=True)
    backend = os.getenv("MEMORY_BACKEND", "memory").strip().lower()
    if backend != "redis":
        return MemoryStore(
            max_messages=max_messages,
            budget_manager=budget_manager,
            summary_generator=summary_generator,
        )
    try:
        return RedisMemoryStore(
            max_messages=max_messages,
            budget_manager=budget_manager,
            summary_generator=summary_generator,
        )
    except Exception:
        return MemoryStore(
            max_messages=max_messages,
            budget_manager=budget_manager,
            summary_generator=summary_generator,
        )


def url_contains_auth(redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    return bool(parsed.username or parsed.password)
