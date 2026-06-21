from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from src.services.llm_client import load_dotenv


class CacheStore:
    backend = "memory"

    def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raise NotImplementedError


class MemoryCacheStore(CacheStore):
    backend = "memory"

    def __init__(self) -> None:
        self._items: dict[str, tuple[float | None, Any]] = {}

    def get_json(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at is not None and expires_at < time.time():
            self._items.pop(key, None)
            return None
        return value

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        self._items[key] = (expires_at, value)


class RedisCacheStore(CacheStore):
    backend = "redis"

    def __init__(self, redis_url: str | None = None) -> None:
        load_dotenv(override=True)
        try:
            import redis
        except ModuleNotFoundError as exc:
            raise RuntimeError("redis is required. Install it with: pip install redis") from exc

        self.client = redis.Redis.from_url(
            redis_url or os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            decode_responses=True,
        )
        self.client.ping()

    def get_json(self, key: str) -> Any | None:
        raw = self.client.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
        self.client.set(key, raw, ex=ttl_seconds)


_CACHE_STORE: CacheStore | None = None


def build_cache_store() -> CacheStore:
    load_dotenv(override=True)
    if os.getenv("CACHE_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
        return MemoryCacheStore()
    backend = os.getenv("CACHE_BACKEND", "memory").strip().lower()
    if backend == "redis":
        try:
            return RedisCacheStore()
        except Exception:
            return MemoryCacheStore()
    return MemoryCacheStore()


def get_cache_store() -> CacheStore:
    global _CACHE_STORE
    if _CACHE_STORE is None:
        _CACHE_STORE = build_cache_store()
    return _CACHE_STORE


def cache_data_version() -> str:
    return os.getenv("CACHE_DATA_VERSION", "recipes-v1").strip() or "recipes-v1"


def cache_ttl_seconds(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def stable_cache_key(prefix: str, payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"smart_recipe:{prefix}:{digest}"
