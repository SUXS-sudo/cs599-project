from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.services.llm_client import load_dotenv
from src.services.logger import get_logger


logger = get_logger("services.checkpoint")


@dataclass
class CheckpointStore:
    saver: Any | None = None
    backend: str = "memory"
    error: str = ""

    def config(self, session_id: str) -> dict[str, dict[str, str]]:
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }

    def delete_thread(self, session_id: str) -> bool:
        if self.saver is None:
            return False
        self.saver.delete_thread(session_id)
        return True

    def close(self) -> None:
        client = getattr(self.saver, "_redis", None)
        if client is not None:
            client.close()


def build_checkpoint_store() -> CheckpointStore:
    load_dotenv(override=True)
    backend = os.getenv("CHECKPOINT_BACKEND", os.getenv("MEMORY_BACKEND", "memory")).strip().lower()
    if backend != "redis":
        return CheckpointStore()

    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    username = os.getenv("REDIS_USERNAME", "").strip() or None
    password = os.getenv("REDIS_PASSWORD", "").strip() or None
    ttl_seconds = _env_int("CHECKPOINT_TTL_SECONDS", _env_int("REDIS_TTL_SECONDS", 7 * 24 * 3600))

    try:
        import redis
        from langgraph.checkpoint.redis import RedisSaver

        has_url_auth = bool(urlparse(redis_url).username or urlparse(redis_url).password)
        client = redis.Redis.from_url(
            redis_url,
            username=username if not has_url_auth else None,
            password=password if not has_url_auth else None,
            decode_responses=False,
        )
        saver = RedisSaver(
            redis_client=client,
            ttl={
                "default_ttl": max(1, ttl_seconds // 60),
                "refresh_on_read": True,
            },
            checkpoint_prefix="smartrecipe_checkpoint",
            checkpoint_write_prefix="smartrecipe_checkpoint_write",
        )
        saver.setup()
        logger.info("LangGraph Checkpoint 已启用 backend=redis-stack")
        return CheckpointStore(saver=saver, backend="redis-stack")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.warning("Redis Stack Checkpoint 初始化失败，退回无持久化 Checkpoint：%s", detail)
        return CheckpointStore(error=detail)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default
