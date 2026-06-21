from __future__ import annotations

import re
from dataclasses import dataclass, field


CHINESE_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
STOPWORDS = {
    "我",
    "想",
    "要",
    "请问",
    "一下",
    "怎么",
    "如何",
    "什么",
    "可以",
    "有没有",
    "推荐",
    "做",
    "做法",
    "吃",
    "的",
    "了",
    "呢",
    "吗",
}

INTENT_EXPANSIONS = {
    "method": ["做法", "制作方法", "步骤", "原料", "调料"],
    "recommend": ["菜名", "推荐", "原料", "制作方法", "小提示"],
    "ingredient": ["原料", "食材", "调料", "制作方法", "菜名"],
    "nutrition": ["小提示", "功效", "营养", "适合"],
}

SYNONYM_EXPANSIONS = {
    "怎么做": ["做法", "制作方法", "步骤"],
    "如何做": ["做法", "制作方法", "步骤"],
    "咋做": ["做法", "制作方法", "步骤"],
    "材料": ["原料", "食材"],
    "食材": ["原料", "材料"],
    "佐料": ["调料"],
    "调味": ["调料"],
    "凉拌": ["凉菜", "拌"],
    "营养": ["功效", "小提示"],
    "功效": ["营养", "小提示"],
}


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    expanded_query: str
    intent: str
    added_terms: list[str] = field(default_factory=list)
    core_terms: list[str] = field(default_factory=list)


def rewrite_recipe_query(query: str) -> QueryRewriteResult:
    original = query.strip()
    intent = infer_query_intent(original)
    terms = []
    terms.extend(extract_core_terms(original))
    terms.extend(INTENT_EXPANSIONS.get(intent, []))
    for key, expansions in SYNONYM_EXPANSIONS.items():
        if key in original:
            terms.extend(expansions)
    terms = unique_terms([term for term in terms if term and term not in STOPWORDS])
    expanded = " ".join(unique_terms([original, *terms]))
    return QueryRewriteResult(
        original_query=original,
        expanded_query=expanded,
        intent=intent,
        added_terms=[term for term in terms if term not in original],
        core_terms=extract_core_terms(original),
    )


def infer_query_intent(query: str) -> str:
    if any(word in query for word in ("怎么做", "如何做", "咋做", "做法", "步骤", "制作")):
        return "method"
    if any(word in query for word in ("推荐", "有什么", "有没有", "可以做什么", "吃什么")):
        return "recommend"
    if any(word in query for word in ("功效", "营养", "适合", "养生")):
        return "nutrition"
    if any(word in query for word in ("原料", "食材", "材料", "调料", "花生", "土豆", "茄子", "鸡蛋")):
        return "ingredient"
    return "method"


def extract_core_terms(query: str) -> list[str]:
    words = CHINESE_TOKEN_RE.findall(query)
    terms = []
    for word in words:
        cleaned = word.strip()
        if len(cleaned) <= 1 or cleaned in STOPWORDS:
            continue
        terms.append(cleaned)
    return unique_terms(terms)


def unique_terms(terms: list[str]) -> list[str]:
    result = []
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        result.append(term)
    return result
