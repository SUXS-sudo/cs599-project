from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.state import AgentState


EVIDENCE_INTENTS = {
    "recipe_search",
    "recipe_detail",
    "ingredient_replace",
    "nutrition_query",
    "structured_recipe_query",
    "relationship_query",
    "multi_source_query",
    "image_recipe_query",
}
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
QUOTED_ENTITY_RE = re.compile(r"[「『《]([^」』》]{2,30})[」』》]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")
LIST_NUMBER_RE = re.compile(r"^\s*\d+[.、]\s*")


@dataclass(frozen=True)
class AnswerVerification:
    status: str
    corrected_answer: str = ""
    unsupported_claims: tuple[str, ...] = ()
    claim_count: int = 0

    @property
    def passed(self) -> bool:
        return self.status in {"grounded", "not_required"}


def verify_answer(state: AgentState) -> AnswerVerification:
    """Verify evidence presence and high-confidence claim/evidence conflicts.

    The deterministic verifier focuses on claims that can be checked reliably:
    numbers and explicitly quoted entities. Semantic edge cases are left for
    the conservative repair/fallback path instead of pretending certainty.
    """

    if state.intent not in EVIDENCE_INTENTS:
        return AnswerVerification("not_required")

    evidence = build_evidence_text(state)
    if not evidence.strip():
        return AnswerVerification(
            "corrected_no_evidence",
            safe_no_evidence_answer(),
        )

    claims = split_claims(state.final_answer)
    unsupported: list[str] = []
    evidence_numbers = number_variants(evidence)
    for claim in claims:
        normalized_claim = LIST_NUMBER_RE.sub("", claim).strip()
        claim_numbers = set(NUMBER_RE.findall(normalized_claim))
        if claim_numbers - evidence_numbers:
            unsupported.append(claim)
            continue
        quoted_entities = QUOTED_ENTITY_RE.findall(normalized_claim)
        if any(entity not in evidence and entity not in state.user_input for entity in quoted_entities):
            unsupported.append(claim)

    if unsupported:
        return AnswerVerification(
            "retryable_unsupported_claims",
            unsupported_claims=tuple(unsupported),
            claim_count=len(claims),
        )
    return AnswerVerification("grounded", claim_count=len(claims))


def number_variants(text: str) -> set[str]:
    values = set(NUMBER_RE.findall(text))
    for raw in tuple(values):
        if raw.endswith("%"):
            continue
        try:
            number = float(raw)
        except ValueError:
            continue
        if 0 <= number <= 1:
            values.add(f"{number * 100:g}%")
    return values


def verify_answer_grounding(state: AgentState) -> tuple[str, str]:
    """Backward-compatible tuple API used by older callers and tests."""

    result = verify_answer(state)
    return result.status, result.corrected_answer


def build_evidence_text(state: AgentState) -> str:
    parts: list[str] = []
    for recipe, score in state.retrieved_docs:
        parts.append(
            " | ".join(
                (
                    recipe.name,
                    ",".join(recipe.ingredients),
                    recipe.category,
                    recipe.cooking_time,
                    recipe.difficulty,
                    str(recipe.calories),
                    ",".join(recipe.tags),
                    ",".join(recipe.suitable_for),
                    recipe.steps,
                    str(score),
                )
            )
        )
    if state.agent_output.strip():
        parts.append(state.agent_output)
    if state.fusion_results:
        parts.append(json.dumps(state.fusion_results, ensure_ascii=False, default=str))
    if state.vision_result:
        parts.append(json.dumps(state.vision_result, ensure_ascii=False, default=str))
    for key in ("sql_rows", "cypher_rows", "vision_result"):
        value = state.meta.get(key)
        if value:
            parts.append(json.dumps(value, ensure_ascii=False, default=str))
    return "\n".join(parts)


def split_claims(answer: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(answer or "") if part.strip()]


def safe_no_evidence_answer() -> str:
    return (
        "我暂时没有足够的菜谱、数据库或图谱证据来可靠回答这个问题。"
        "你可以补充菜名、食材、饮食目标或上传更清晰的图片，我再重新检索和推荐。"
    )


def safe_repair_fallback(state: AgentState, unsupported_claims: tuple[str, ...]) -> str:
    evidence = build_evidence_text(state).strip()
    if not evidence:
        return safe_no_evidence_answer()
    return (
        "部分候选内容无法由当前证据验证，已停止输出这些内容。"
        "下面仅保留数据源返回的原始结果供参考：\n"
        + evidence[:2_000]
    )
