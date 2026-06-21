from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
PROMPT_INJECTION_PATTERNS = (
    re.compile(r"忽略.{0,12}(之前|以上|系统).{0,12}(指令|提示|规则)", re.I),
    re.compile(r"(显示|泄露|输出).{0,12}(系统提示|system prompt|开发者指令|api.?key)", re.I),
    re.compile(r"ignore.{0,20}(previous|system).{0,20}(instruction|prompt)", re.I),
    re.compile(r"(jailbreak|越狱模式|developer mode)", re.I),
)
UNSAFE_PATTERNS = (
    re.compile(r"(下药|迷药|投毒|毒杀|致命剂量|自杀方法|伤害他人)"),
    re.compile(r"(怎么|如何|教我|步骤|配方|制作).{0,18}(毒药|爆炸物|迷药|致命|自杀|伤害)"),
    re.compile(r"(毒药|爆炸物|迷药|致命|自杀|伤害).{0,18}(怎么|如何|步骤|配方|制作)"),
)
HEALTH_SENSITIVE_TERMS = (
    "糖尿病", "高血压", "肾病", "肝病", "孕妇", "婴儿", "过敏", "药物", "处方",
)


@dataclass(frozen=True)
class BoundaryDecision:
    decision: str
    scope: str
    risk_types: tuple[str, ...] = ()
    confidence: float = 1.0
    reason_code: str = "ALLOW"
    normalized_text: str = ""

    def to_meta(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "scope": self.scope,
            "risk_types": list(self.risk_types),
            "confidence": self.confidence,
            "reason_code": self.reason_code,
        }


class QueryBoundaryGuard:
    """Dependency-free first-line boundary detection with normalization."""

    def __init__(self, max_chars: int = 4_000) -> None:
        self.max_chars = max_chars

    def evaluate(self, text: str) -> BoundaryDecision:
        normalized = normalize_query(text)
        if not normalized:
            return BoundaryDecision("block", "empty", reason_code="EMPTY_QUERY", normalized_text=normalized)
        if len(normalized) > self.max_chars:
            return BoundaryDecision("block", "out_of_scope", ("oversized_input",), 1.0, "QUERY_TOO_LONG", normalized)
        if any(pattern.search(normalized) for pattern in PROMPT_INJECTION_PATTERNS):
            return BoundaryDecision("block", "out_of_scope", ("prompt_injection",), 0.99, "PROMPT_INJECTION", normalized)
        if any(pattern.search(normalized) for pattern in UNSAFE_PATTERNS):
            return BoundaryDecision("block", "out_of_scope", ("unsafe_instruction",), 0.99, "UNSAFE_INSTRUCTION", normalized)
        if any(term in normalized for term in HEALTH_SENSITIVE_TERMS):
            return BoundaryDecision("caution", "health_sensitive", ("health_sensitive",), 0.95, "HEALTH_SENSITIVE", normalized)
        return BoundaryDecision("allow", infer_scope(normalized), normalized_text=normalized)


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = ZERO_WIDTH_RE.sub("", normalized)
    return " ".join(normalized.split())


def infer_scope(text: str) -> str:
    if any(term in text for term in ("菜", "食材", "做法", "烹饪", "热量", "营养", "饮食", "过敏")):
        return "smart_recipe"
    return "unknown"
