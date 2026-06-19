from __future__ import annotations

from app.state import AgentState


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


def verify_answer_grounding(state: AgentState) -> tuple[str, str]:
    """Return guard status and an optional corrected answer.

    The guard is deliberately conservative: it does not try to fact-check every
    sentence, but it prevents evidence-requiring routes from presenting an
    unsupported answer as certain.
    """

    if state.intent not in EVIDENCE_INTENTS:
        return "not_required", ""
    has_recipe_evidence = bool(state.retrieved_docs)
    has_agent_evidence = bool(state.agent_output.strip())
    has_structured_evidence = bool(state.meta.get("sql_rows") or state.meta.get("cypher_rows") or state.fusion_results)
    has_vision_evidence = bool(state.vision_result or state.meta.get("vision_result"))
    if has_recipe_evidence or has_structured_evidence or has_vision_evidence or has_agent_evidence:
        return "grounded", ""
    return (
        "corrected_no_evidence",
        "我暂时没有足够的菜谱、数据库或图谱证据来可靠回答这个问题。"
        "你可以补充菜名、食材、饮食目标或上传更清晰的图片，我再重新检索和推荐。",
    )
