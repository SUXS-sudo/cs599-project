from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence


class MessageLike(Protocol):
    role: str
    content: str


SummaryGenerator = Callable[[str, int], str | None]


@dataclass(frozen=True)
class ContextBudgetConfig:
    total_tokens: int = 131_072
    system_reserve_tokens: int = 12_288
    retrieval_reserve_tokens: int = 24_576
    output_reserve_tokens: int = 4_096
    safety_reserve_tokens: int = 4_096
    summary_max_tokens: int = 8_192
    compaction_trigger_ratio: float = 0.85
    recent_window_ratio: float = 0.55
    min_recent_messages: int = 6

    @property
    def history_budget_tokens(self) -> int:
        reserved = (
            self.system_reserve_tokens
            + self.retrieval_reserve_tokens
            + self.output_reserve_tokens
            + self.safety_reserve_tokens
        )
        return max(1_024, self.total_tokens - reserved)

    @classmethod
    def from_env(cls) -> "ContextBudgetConfig":
        return cls(
            total_tokens=_env_int("CONTEXT_TOKEN_BUDGET", 131_072),
            system_reserve_tokens=_env_int("CONTEXT_SYSTEM_RESERVE_TOKENS", 12_288),
            retrieval_reserve_tokens=_env_int("CONTEXT_RETRIEVAL_RESERVE_TOKENS", 24_576),
            output_reserve_tokens=_env_int("CONTEXT_OUTPUT_RESERVE_TOKENS", 4_096),
            safety_reserve_tokens=_env_int("CONTEXT_SAFETY_RESERVE_TOKENS", 4_096),
            summary_max_tokens=_env_int("CONTEXT_SUMMARY_MAX_TOKENS", 8_192),
            compaction_trigger_ratio=_env_float("CONTEXT_COMPACTION_TRIGGER_RATIO", 0.85),
            recent_window_ratio=_env_float("CONTEXT_RECENT_WINDOW_RATIO", 0.55),
            min_recent_messages=_env_int("CONTEXT_MIN_RECENT_MESSAGES", 6),
        )


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    recent_messages: list[MessageLike]
    compacted_messages: int
    before_tokens: int
    after_tokens: int


@dataclass(frozen=True)
class BudgetMessage:
    role: str
    content: str


class TokenCounter:
    """Model-aware token counter with a conservative multilingual fallback."""

    def __init__(self, model: str = "") -> None:
        self.model = model
        self._encoding = self._load_encoding(model)

    @staticmethod
    def _load_encoding(model: str):
        try:
            import tiktoken

            if model:
                try:
                    return tiktoken.encoding_for_model(model)
                except KeyError:
                    pass
            return tiktoken.get_encoding("cl100k_base")
        except (ImportError, ValueError):
            return None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))

        chinese = len(re.findall(r"[\u3400-\u9fff]", text))
        non_chinese = re.sub(r"[\u3400-\u9fff]", "", text)
        ascii_like = math.ceil(len(non_chinese.encode("utf-8")) / 3.5)
        return max(1, chinese + ascii_like)

    def count_messages(self, messages: Sequence[MessageLike]) -> int:
        return sum(4 + self.count_text(message.role) + self.count_text(message.content) for message in messages)

    def trim_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.count_text(text) <= max_tokens:
            return text
        if self._encoding is not None:
            tokens = self._encoding.encode(text, disallowed_special=())[-max_tokens:]
            return self._encoding.decode(tokens)

        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.count_text(text[-middle:]) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        return text[-low:] if low else ""


class ContextBudgetManager:
    def __init__(
        self,
        config: ContextBudgetConfig | None = None,
        token_counter: TokenCounter | None = None,
        summary_generator: SummaryGenerator | None = None,
    ) -> None:
        self.config = config or ContextBudgetConfig.from_env()
        self.counter = token_counter or TokenCounter(os.getenv("MODEL", ""))
        self.summary_generator = summary_generator

    def compact(
        self,
        existing_summary: str,
        messages: Sequence[MessageLike],
        max_messages: int | None = None,
    ) -> CompactionResult:
        recent = list(messages)
        before_tokens = self.counter.count_text(existing_summary) + self.counter.count_messages(recent)
        trigger_tokens = int(self.config.history_budget_tokens * self.config.compaction_trigger_ratio)
        over_message_limit = max_messages is not None and len(recent) > max_messages
        if before_tokens <= trigger_tokens and not over_message_limit:
            return CompactionResult(existing_summary, recent, 0, before_tokens, before_tokens)

        keep_from = self._recent_window_start(recent, max_messages)
        older = recent[:keep_from]
        kept = recent[keep_from:]
        if not older:
            summary, kept = self._fit_hard_budget(existing_summary, kept)
            after_tokens = self.counter.count_text(summary) + self.counter.count_messages(kept)
            return CompactionResult(summary, kept, 0, before_tokens, after_tokens)

        summary = self._summarize(existing_summary, older)
        summary = self.counter.trim_text(summary, self.config.summary_max_tokens)
        summary, kept = self._fit_hard_budget(summary, kept)
        after_tokens = self.counter.count_text(summary) + self.counter.count_messages(kept)
        return CompactionResult(summary, kept, len(older), before_tokens, after_tokens)

    def context_stats(self, summary: str, messages: Sequence[MessageLike]) -> dict[str, int | float]:
        used = self.counter.count_text(summary) + self.counter.count_messages(messages)
        budget = self.config.history_budget_tokens
        return {
            "total_budget_tokens": self.config.total_tokens,
            "history_budget_tokens": budget,
            "history_used_tokens": used,
            "history_usage_ratio": round(used / budget, 4),
            "summary_tokens": self.counter.count_text(summary),
            "recent_message_tokens": self.counter.count_messages(messages),
        }

    def _recent_window_start(self, messages: Sequence[MessageLike], max_messages: int | None) -> int:
        if max_messages is not None:
            return max(0, len(messages) - max_messages)

        target = int(self.config.history_budget_tokens * self.config.recent_window_ratio)
        used = 0
        keep_from = len(messages)
        minimum_start = max(0, len(messages) - self.config.min_recent_messages)
        for index in range(len(messages) - 1, -1, -1):
            message_tokens = self.counter.count_messages([messages[index]])
            if index < minimum_start and used + message_tokens > target:
                break
            used += message_tokens
            keep_from = index
        return max(1, keep_from) if len(messages) > 1 else keep_from

    def _summarize(self, existing_summary: str, messages: Sequence[MessageLike]) -> str:
        source = "\n".join(f"{message.role}: {message.content}" for message in messages)
        if self.summary_generator is not None:
            prompt = (
                "请把以下旧会话增量压缩为可供后续对话使用的中文记忆。"
                "必须保留用户偏好、过敏/忌口、关键事实、已做决定和未完成事项；"
                "删除寒暄、重复内容和冗长措辞，不得编造信息。\n\n"
                f"已有摘要：\n{existing_summary or '无'}\n\n待合并旧消息：\n{source}"
            )
            generated = self.summary_generator(prompt, self.config.summary_max_tokens)
            if generated and generated.strip():
                return generated.strip()

        combined = "\n".join(part for part in (existing_summary.strip(), source) if part)
        return self.counter.trim_text(combined, self.config.summary_max_tokens)

    def _fit_hard_budget(
        self,
        summary: str,
        messages: list[MessageLike],
    ) -> tuple[str, list[MessageLike]]:
        budget = self.config.history_budget_tokens
        kept = list(messages)
        while len(kept) > 1 and self.counter.count_text(summary) + self.counter.count_messages(kept) > budget:
            kept.pop(0)

        message_tokens = self.counter.count_messages(kept)
        summary = self.counter.trim_text(summary, max(0, budget - message_tokens))
        if kept and self.counter.count_messages(kept) > budget:
            latest = kept[-1]
            role_tokens = 4 + self.counter.count_text(latest.role)
            content = self.counter.trim_text(latest.content, max(1, budget - role_tokens))
            kept = [BudgetMessage(role=latest.role, content=content)]
            summary = ""
        return summary, kept


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
