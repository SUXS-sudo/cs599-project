from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from src.services.query_boundary_guard import normalize_query


COMMON_CORRECTIONS = {
    "推见": "推荐",
    "推存": "推荐",
    "低旨": "低脂",
    "底脂": "低脂",
    "高旦白": "高蛋白",
    "蛋百质": "蛋白质",
    "万餐": "晚餐",
    "午反": "午饭",
    "早歺": "早餐",
    "番茄抄蛋": "番茄炒蛋",
    "西红市": "西红柿",
    "西蓝花": "西兰花",
    "鸡匈肉": "鸡胸肉",
    "卡路哩": "卡路里",
    "红晒": "红烧",
    "投讀": "投毒",
}
CORE_VOCABULARY = {
    "推荐", "菜谱", "食谱", "做法", "步骤", "食材", "替换", "热量", "营养",
    "低脂", "低糖", "低盐", "高蛋白", "蛋白质", "控糖", "减脂", "增肌",
    "早餐", "午餐", "午饭", "晚餐", "过敏", "不吃", "不要",
    "番茄", "西红柿", "鸡蛋", "西兰花", "鸡胸肉", "黄瓜", "豆腐", "虾仁",
    "糖尿病", "高血压", "肾病", "肝病", "孕妇", "婴儿", "药物", "处方",
    "投毒", "毒药", "迷药",
}
PROTECTED_TERMS = (
    "不", "别", "不要", "不能", "不吃", "过敏", "禁忌", "无糖", "低糖",
    "糖尿病", "高血压", "肾病", "肝病", "孕妇", "婴儿", "药物", "处方",
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
QUANTITY_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:道|份|个|分钟|小时|克|千克|千卡|卡路里)")
QUERY_MARKERS = ("怎么", "如何", "做法", "步骤", "推荐", "多少", "热量", "营养", "有哪些", "查询", "替换")
DISH_STYLE_PREFIXES = (
    "红烧", "清蒸", "糖醋", "宫保", "鱼香", "麻辣", "香辣", "蒜蓉", "凉拌", "爆炒", "油焖", "酱烧",
)
INLINE_LOWERCASE_NOISE_RE = re.compile(r"(?<=[\u4e00-\u9fff])[a-z]+(?=[\u4e00-\u9fff])")
INLINE_SYMBOL_NOISE_RE = re.compile(r"(?<=[\u4e00-\u9fff])[_@#$%^&*]+(?=[\u4e00-\u9fff])")
INLINE_DIGIT_NOISE_RE = re.compile(r"(?<=[\u4e00-\u9fff])\d+(?=[\u4e00-\u9fff])")
QUANTITY_UNIT_CHARS = set("道份个分时克千卡")


@dataclass(frozen=True)
class CorrectionCandidate:
    query: str
    score: float
    source: str
    replacements: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "score": round(self.score, 4),
            "source": self.source,
            "replacements": [{"source": source, "target": target} for source, target in self.replacements],
        }


def parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def generate_correction_candidates(
    query: str,
    extraction: dict[str, Any] | None = None,
    vocabulary: set[str] | None = None,
    limit: int = 8,
) -> list[CorrectionCandidate]:
    cleaned = normalize_query(query)
    vocab = {term.strip() for term in (vocabulary or set()) | CORE_VOCABULARY if len(term.strip()) >= 2}
    candidates: dict[str, CorrectionCandidate] = {
        cleaned: CorrectionCandidate(cleaned, 1.0, "original"),
    }

    inline_cleaned = clean_inline_noise(cleaned)
    if inline_cleaned != cleaned:
        candidates[inline_cleaned] = CorrectionCandidate(
            inline_cleaned,
            0.995,
            "inline_noise_cleanup",
            ((cleaned, inline_cleaned),),
        )
        combined_cleaned = inline_cleaned
        combined_replacements: list[tuple[str, str]] = [(cleaned, inline_cleaned)]
        for source, target in COMMON_CORRECTIONS.items():
            if source in combined_cleaned:
                combined_cleaned = combined_cleaned.replace(source, target)
                combined_replacements.append((source, target))
        if combined_cleaned != inline_cleaned:
            candidates[combined_cleaned] = CorrectionCandidate(
                combined_cleaned,
                0.997,
                "inline_noise_and_typo_cleanup",
                tuple(combined_replacements),
            )

    replacements = [(source, target) for source, target in COMMON_CORRECTIONS.items() if source in cleaned]
    noisy_dish = extract_noisy_bare_dish(cleaned, extraction, vocab)
    if noisy_dish:
        corrected_dish = noisy_dish
        dish_replacements: list[tuple[str, str]] = []
        for source, target in COMMON_CORRECTIONS.items():
            if source in corrected_dish:
                corrected_dish = corrected_dish.replace(source, target)
                dish_replacements.append((source, target))
        candidates[corrected_dish] = CorrectionCandidate(
            corrected_dish,
            0.999,
            "dish_name_cleanup",
            tuple(dish_replacements),
        )
    if replacements:
        combined = cleaned
        for source, target in replacements:
            combined = combined.replace(source, target)
        add_candidate(candidates, cleaned, combined, 0.98, "common_dictionary", replacements)

    for source, target in replacements:
        add_candidate(candidates, cleaned, cleaned.replace(source, target), 0.96, "common_dictionary", [(source, target)])

    for entity in (extraction or {}).get("entities", []):
        if not isinstance(entity, dict):
            continue
        source = str(entity.get("text") or "").strip()
        target = str(entity.get("normalized") or "").strip()
        if source and target and source in cleaned and (
            target in vocab or is_conservative_llm_entity_correction(source, target, str(entity.get("type") or ""))
        ):
            add_candidate(candidates, cleaned, cleaned.replace(source, target), 0.9, "llm_entity_validated", [(source, target)])

    for target in sorted(vocab, key=len, reverse=True):
        if target in cleaned or len(target) > 10:
            continue
        for width in {len(target) - 1, len(target), len(target) + 1}:
            if width < 2 or width > len(cleaned):
                continue
            for start in range(len(cleaned) - width + 1):
                source = cleaned[start : start + width]
                if source in vocab:
                    continue
                similarity = SequenceMatcher(None, source, target).ratio()
                if similarity < 0.72:
                    continue
                candidate = cleaned[:start] + target + cleaned[start + width :]
                add_candidate(candidates, cleaned, candidate, similarity, "domain_fuzzy", [(source, target)])

    ordered = sorted(candidates.values(), key=lambda item: (item.query != cleaned, item.score), reverse=True)
    original = candidates[cleaned]
    non_original = [item for item in ordered if item.query != cleaned][: max(limit - 1, 0)]
    return [original, *non_original]


def add_candidate(
    candidates: dict[str, CorrectionCandidate],
    original: str,
    candidate: str,
    score: float,
    source: str,
    replacements: list[tuple[str, str]],
) -> None:
    candidate = normalize_query(candidate)
    if not candidate or candidate == original or not preserves_critical_constraints(original, candidate):
        return
    current = candidates.get(candidate)
    item = CorrectionCandidate(candidate, score, source, tuple(replacements))
    if current is None or score > current.score:
        candidates[candidate] = item


def preserves_critical_constraints(original: str, candidate: str) -> bool:
    if NUMBER_RE.findall(original) != NUMBER_RE.findall(candidate):
        return False
    return all(term not in original or term in candidate for term in PROTECTED_TERMS)


def extract_noisy_bare_dish(
    query: str,
    extraction: dict[str, Any] | None,
    vocabulary: set[str],
) -> str:
    if not re.search(r"[^\u4e00-\u9fff]", query) or QUANTITY_RE.search(query):
        return ""
    chinese_only = "".join(re.findall(r"[\u4e00-\u9fff]", query))
    if not 2 <= len(chinese_only) <= 12 or any(marker in chinese_only for marker in QUERY_MARKERS):
        return ""
    normalized_entities = {
        str(entity.get("normalized") or "").strip()
        for entity in (extraction or {}).get("entities", [])
        if isinstance(entity, dict) and str(entity.get("type") or "").strip().lower() in {"dish", "recipe"}
    }
    corrected = chinese_only
    for source, target in COMMON_CORRECTIONS.items():
        corrected = corrected.replace(source, target)
    if corrected in vocabulary or corrected in normalized_entities or corrected.startswith(DISH_STYLE_PREFIXES):
        return chinese_only
    return ""


def clean_inline_noise(query: str) -> str:
    cleaned = INLINE_LOWERCASE_NOISE_RE.sub("", query)
    cleaned = INLINE_SYMBOL_NOISE_RE.sub("", cleaned)

    def replace_digits(match: re.Match[str]) -> str:
        next_char = cleaned[match.end() : match.end() + 1]
        return match.group(0) if next_char in QUANTITY_UNIT_CHARS else ""

    return INLINE_DIGIT_NOISE_RE.sub(replace_digits, cleaned)


def is_conservative_llm_entity_correction(source: str, target: str, entity_type: str) -> bool:
    allowed_types = {"dish", "recipe", "ingredient", "cooking_method", "diet_goal", "meal"}
    if entity_type.strip().lower() not in allowed_types:
        return False
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,12}", target):
        return False
    if abs(len(source) - len(target)) > 1:
        return False
    return SequenceMatcher(None, source, target).ratio() >= 0.72


def build_extraction_prompt(query: str, chat_history: str) -> str:
    return (
        "你是 SmartRecipe 查询理解器。只分析用户真实意图和实体，不直接回答问题。\n"
        "用户可能有错别字、同音字或口语表达。不得修改否定词、过敏信息、疾病、药品、数字和单位。\n"
        "菜名实体的 normalized 必须只包含菜名本身，去掉附着的字母、数字、符号和‘怎么做/推荐/识别结果’等说明。\n"
        "只返回 JSON："
        '{"intent":"recipe_search","entities":[{"text":"低旨","type":"diet_goal","normalized":"低脂"}],'
        '"needs_correction":true,"reason":"..."}\n'
        "intent 只能是 recipe_search、recipe_detail、ingredient_replace、nutrition_query、"
        "structured_recipe_query、relationship_query、general_chat、out_of_scope。\n"
        f"对话历史：{chat_history[-1500:]}\n"
        f"用户原文：{query}"
    )


def build_selection_prompt(query: str, extraction: dict[str, Any], candidates: list[CorrectionCandidate]) -> str:
    candidate_lines = [f"{index}: {item.query} | program_score={item.score:.3f}" for index, item in enumerate(candidates)]
    return (
        "你是 SmartRecipe 纠错裁决器。必须从候选列表中选择最符合原始意图的一项，不能创造新句子。\n"
        "必须保持否定、过敏、疾病、药品、数字和单位不变。没有必要纠错时选择 0。\n"
        "只返回 JSON："
        '{"candidate_index":0,"confidence":0.95,"reason":"..."}\n'
        f"用户原文：{query}\n"
        f"意图实体抽取：{json.dumps(extraction, ensure_ascii=False)}\n"
        "候选：\n" + "\n".join(candidate_lines)
    )
