from __future__ import annotations

from app.services.checkpoint_store import CheckpointStore
from app.services.context_budget import ContextBudgetConfig, ContextBudgetManager, TokenCounter
from app.services.memory import ChatMessage, MemoryStore


def small_budget() -> ContextBudgetConfig:
    return ContextBudgetConfig(
        total_tokens=2_048,
        system_reserve_tokens=0,
        retrieval_reserve_tokens=0,
        output_reserve_tokens=0,
        safety_reserve_tokens=0,
        summary_max_tokens=256,
        compaction_trigger_ratio=0.5,
        recent_window_ratio=0.3,
        min_recent_messages=2,
    )


def test_context_budget_compacts_old_messages_and_keeps_recent_window() -> None:
    prompts: list[str] = []

    def summarize(prompt: str, max_tokens: int) -> str:
        prompts.append(prompt)
        return "用户长期偏好：清淡；未完成事项：继续推荐晚餐。"

    manager = ContextBudgetManager(
        config=small_budget(),
        token_counter=TokenCounter(""),
        summary_generator=summarize,
    )
    messages = [
        ChatMessage(role="user" if index % 2 == 0 else "assistant", content="清淡晚餐" * 80)
        for index in range(8)
    ]

    result = manager.compact("", messages)

    assert result.compacted_messages > 0
    assert 2 <= len(result.recent_messages) < len(messages)
    assert "清淡" in result.summary
    assert result.after_tokens < result.before_tokens
    assert prompts


def test_memory_store_isolates_context_budget_by_session() -> None:
    manager = ContextBudgetManager(config=small_budget(), token_counter=TokenCounter(""))
    store = MemoryStore(budget_manager=manager)

    for index in range(6):
        store.add_turn("alpha", f"alpha-{index}-" + "甲" * 250, "收到")
    store.add_turn("beta", "beta-only", "收到")

    alpha = store.debug_session("alpha")
    beta = store.debug_session("beta")
    assert alpha["summary"]
    assert all("beta-only" not in item["content"] for item in alpha["history"])
    assert beta["summary"] == ""
    assert beta["history"][0]["content"] == "beta-only"
    assert alpha["context_budget"]["total_budget_tokens"] == 2_048


def test_context_budget_hard_caps_a_single_oversized_message() -> None:
    manager = ContextBudgetManager(config=small_budget(), token_counter=TokenCounter(""))
    messages = [
        ChatMessage(role="user", content="旧消息" * 400),
        ChatMessage(role="assistant", content="最新回答" * 4_000),
    ]

    result = manager.compact("", messages)

    assert result.after_tokens <= manager.config.history_budget_tokens
    assert result.recent_messages[-1].role == "assistant"


class FakeSaver:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


def test_checkpoint_store_uses_session_as_thread_id_and_deletes_it() -> None:
    saver = FakeSaver()
    store = CheckpointStore(saver=saver, backend="redis-stack")

    assert store.config("session-a") == {
        "configurable": {
            "thread_id": "session-a",
        }
    }
    assert store.delete_thread("session-a") is True
    assert saver.deleted == ["session-a"]
