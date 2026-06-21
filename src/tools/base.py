from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.state import AgentState


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, args: dict[str, Any], state: AgentState) -> ToolResult:
        ...
