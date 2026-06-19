from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
import os
from typing import Any


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class UserPreferences:
    preferences: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


class MemoryStore:
    backend = "memory"

    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self._sessions: dict[str, deque[ChatMessage]] = defaultdict(deque)
        self._preferences: dict[str, UserPreferences] = defaultdict(UserPreferences)
        self._summaries: dict[str, str] = defaultdict(str)

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self._sessions[session_id])

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        self._sessions[session_id].append(ChatMessage(role="user", content=user_message))
        self._sessions[session_id].append(ChatMessage(role="assistant", content=assistant_message))
        self._refresh_summary(session_id)

    def format_history(self, session_id: str) -> str:
        messages = self.get_history(session_id)
        if not messages:
            return "No previous conversation."
        recent = "\n".join(f"{message.role}: {message.content}" for message in messages[-self.max_messages :])
        summary = self.get_summary(session_id) if memory_summary_enabled() else ""
        if summary:
            return f"Long-term summary: {summary}\nRecent conversation:\n{recent}"
        return recent

    def get_summary(self, session_id: str) -> str:
        return self._summaries[session_id]

    def get_preferences(self, session_id: str) -> UserPreferences:
        return self._preferences[session_id]

    def update_preferences(
        self,
        session_id: str,
        preferences: list[str] | None = None,
        allergies: list[str] | None = None,
        dislikes: list[str] | None = None,
    ) -> UserPreferences:
        current = self._preferences[session_id]
        current.preferences = merge_unique(current.preferences, preferences or [])
        current.allergies = merge_unique(current.allergies, allergies or [])
        current.dislikes = merge_unique(current.dislikes, dislikes or [])
        return current

    def debug_session(self, session_id: str) -> dict[str, Any]:
        history = [asdict(message) for message in self.get_history(session_id)]
        preferences = self.get_preferences(session_id).to_dict()
        return {
            "session_id": session_id,
            "preferences": preferences,
            "summary": self.get_summary(session_id),
            "history": history,
            "turn_count": len([message for message in history if message["role"] == "user"]),
            "backend": self.backend,
        }

    def active_session_count(self) -> int:
        return len(self._sessions)

    def _refresh_summary(self, session_id: str) -> None:
        if not memory_summary_enabled():
            return
        messages = self.get_history(session_id)
        if len(messages) <= self.max_messages:
            return
        older = messages[: -self.max_messages]
        self._summaries[session_id] = summarize_messages(older)


def summarize_messages(messages: list[ChatMessage], max_chars: int = 360) -> str:
    facts = []
    for message in messages:
        content = message.content.strip().replace("\n", " ")
        if not content:
            continue
        if message.role == "user":
            facts.append(f"用户说：{content}")
        elif "推荐" in content or "识别" in content or "菜" in content:
            facts.append(f"系统答：{content}")
    summary = "；".join(facts)
    if len(summary) > max_chars:
        summary = summary[-max_chars:]
    return summary


def memory_summary_enabled() -> bool:
    return os.getenv("ENABLE_MEMORY_SUMMARY", "true").strip().lower() not in {"0", "false", "no", "off"}


def merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    result = list(existing)
    for item in incoming:
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return result


def preferences_from_dict(data: dict[str, Any] | None) -> UserPreferences:
    data = data or {}
    return UserPreferences(
        preferences=[str(item) for item in data.get("preferences", [])],
        allergies=[str(item) for item in data.get("allergies", [])],
        dislikes=[str(item) for item in data.get("dislikes", [])],
    )
