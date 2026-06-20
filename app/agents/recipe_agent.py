from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.services.memory import preferences_from_dict
from app.services.graph_rag import GraphRAG, format_graph_context
from app.services.logger import get_logger
from app.services.cache_store import cache_data_version, cache_ttl_seconds, get_cache_store, stable_cache_key
from app.services.mysql_store import MySQLStore
from app.agents.preference_agent import preferences_to_query_suffix, violates_ingredients
from app.retriever import Recipe, RecipeRetriever
from app.state import AgentState


ALIAS_TO_STANDARD = {
    "西红柿炒鸡蛋": "番茄炒蛋",
    "番茄炒鸡蛋": "番茄炒蛋",
    "西红柿炒蛋": "番茄炒蛋",
    "番茄鸡蛋": "番茄炒蛋",
    "番茄蛋花汤": "紫菜蛋花汤",
    "水蒸蛋": "鸡蛋羹",
    "蒸鸡蛋": "鸡蛋羹",
    "蒸蛋": "鸡蛋羹",
    "凉拌青瓜": "凉拌黄瓜",
    "青瓜凉拌": "凉拌黄瓜",
}
FUZZY_NAME_THRESHOLD = 0.72
VECTOR_NAME_THRESHOLD = 0.76
EXPLICIT_DISH_STYLE_PREFIXES = (
    "红烧", "清蒸", "糖醋", "宫保", "鱼香", "麻辣", "香辣",
    "蒜蓉", "凉拌", "爆炒", "油焖", "酱烧",
)
GENERIC_RECIPE_TARGET_WORDS = (
    "什么", "啥", "推荐", "随便", "菜谱", "食谱", "早餐", "午餐",
    "晚餐", "夜宵", "低脂", "低糖", "低盐", "高蛋白", "减脂", "增肌",
)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


@dataclass(frozen=True)
class RecipeMatch:
    row: dict[str, object]
    level: str
    matched_name: str
    resolved_name: str
    score: float
    candidates: list[str]


class RecipeAgent:
    def __init__(
        self,
        retriever: RecipeRetriever,
        graph_rag: GraphRAG | None = None,
        mysql_store: MySQLStore | None = None,
    ) -> None:
        self.retriever = retriever
        self.graph_rag = graph_rag
        self.mysql_store = mysql_store
        self.cache = get_cache_store()
        self._last_recipe_match_cache_hit = False
        self.logger = get_logger("agents.recipe")

    def run(self, state: AgentState) -> AgentState:
        preferences = preferences_from_dict(state.meta.get("user_preferences"))
        if state.intent == "recipe_detail":
            target_name = extract_recipe_detail_target(state.user_input)
            if target_name:
                state.meta["recipe_detail_target"] = target_name
                if self._try_direct_recipe_answer(state, target_name):
                    return state
                if self._try_document_recipe_answer(state, target_name):
                    return state

        query_suffix = preferences_to_query_suffix(preferences)
        query = f"{state.user_input}\n{query_suffix}".strip()
        state.meta["retrieval_query_scope"] = "current_turn"
        candidates = self.retriever.search(query, max(state.top_k * 3, state.top_k))
        allergy_safe = [item for item in candidates if not violates_preferences(item[0].ingredients, [], preferences.allergies)]
        preference_safe = [
            item for item in allergy_safe if not violates_preferences(item[0].ingredients, preferences.dislikes, [])
        ]
        safe_candidates = preference_safe

        if state.intent == "recipe_detail":
            target_name = state.meta.get("recipe_detail_target") or extract_recipe_detail_target(state.user_input)
            if target_name:
                exact_matches = [
                    item for item in safe_candidates if recipe_name_matches(target_name, item[0].name)
                ]
                if not exact_matches:
                    state.retrieved_docs = []
                    state.meta["recipe_source"] = "llm_fallback"
                    state.meta["recipe_detail_target"] = target_name
                    state.meta["recipe_mismatch_candidates"] = [recipe.name for recipe, _ in safe_candidates[: state.top_k]]
                    state.agent_output = (
                        f"当前菜谱库中暂未收录「{target_name}」的标准菜谱。"
                        "不要把相似检索结果当作该菜的库内做法；如需继续帮助，"
                        "只能生成明确标注为通用烹饪知识的参考做法。"
                    )
                    self.logger.info("菜谱详情目标未命中 目标=%s 候选数=%s", target_name, len(safe_candidates))
                    return state
                safe_candidates = exact_matches
                state.meta["recipe_detail_target"] = target_name
                state.meta["recipe_source"] = "database"

        state.retrieved_docs = safe_candidates[: state.top_k]
        self.logger.info("菜谱检索完成 候选数=%s 选中数=%s", len(candidates), len(state.retrieved_docs))
        if not state.retrieved_docs:
            state.agent_output = "本地菜谱候选经过当前会话的过敏与不吃食材过滤后为空。"
            state.meta["recipe_source"] = "llm_fallback_query"
            state.meta["fallback_reason"] = "rag_empty_after_preferences"
            return state

        names = "、".join(recipe.name for recipe, _ in state.retrieved_docs)
        if state.intent == "recipe_detail":
            state.agent_output = (
                f"Recipe Agent 已检索到相关做法：{names}。"
                "回答时优先说明目标菜品的主要食材、步骤和火候/调味注意事项。"
            )
        elif state.intent == "ingredient_replace":
            state.agent_output = (
                f"Recipe Agent 已检索到可用于食材替换判断的菜谱：{names}。"
                "回答时必须说明：可替换食材、替换比例或用量建议、口感变化、"
                "不建议替换的情况，以及过敏/禁忌提醒。"
            )
        else:
            state.agent_output = (
                f"Recipe Agent 已检索到可推荐菜谱：{names}。"
                "回答时说明推荐理由、主要食材、简要步骤和适用场景。"
            )
        self._append_graph_context(state)
        return state

    def _append_graph_context(self, state: AgentState) -> None:
        if not self.graph_rag or not state.retrieved_docs:
            return
        try:
            context = self.graph_rag.enrich(state.retrieved_docs, limit=state.top_k)
        except Exception as exc:
            self.logger.exception("GraphRAG 增强失败")
            state.meta["graph_rag_status"] = "failed"
            state.meta["graph_rag_error"] = f"{type(exc).__name__}: {exc}"
            return
        graph_text = format_graph_context(context)
        state.meta["graph_rag_status"] = "ok" if graph_text else "empty"
        state.meta["graph_rag_context"] = context
        if graph_text:
            state.agent_output = f"{state.agent_output}\n{graph_text}"

    def _try_direct_recipe_answer(self, state: AgentState, target_name: str) -> bool:
        match = self._match_recipe_name(target_name)
        if match is None:
            return False

        state.retrieved_docs = []
        state.agent_output = format_direct_recipe_answer(match.row)
        state.meta["recipe_source"] = "database_fast"
        state.meta["answer_mode"] = "direct"
        state.meta["recipe_fast_path"] = True
        state.meta["recipe_match_level"] = match.level
        state.meta["recipe_matched_name"] = match.matched_name
        state.meta["recipe_resolved_name"] = match.resolved_name
        state.meta["recipe_match_score"] = round(match.score, 4)
        state.meta["recipe_match_candidates"] = match.candidates
        state.meta["recipe_match_cache_hit"] = self._last_recipe_match_cache_hit
        state.meta["recipe_fast_path_source"] = match.level
        state.meta["recipe_fast_path_row"] = match.row
        self.logger.info(
            "菜谱详情快速命中 级别=%s 目标=%s 匹配=%s 分数=%.3f",
            match.level,
            target_name,
            match.matched_name,
            match.score,
        )
        return True

    def _try_document_recipe_answer(self, state: AgentState, target_name: str) -> bool:
        row = self._lookup_recipe_from_document_index_exact(target_name)
        if row is None:
            return False

        state.retrieved_docs = []
        state.agent_output = format_document_recipe_answer(row)
        state.meta["recipe_source"] = "document_fast"
        state.meta["answer_mode"] = "direct"
        state.meta["recipe_fast_path"] = True
        state.meta["recipe_match_level"] = "document_exact"
        state.meta["recipe_matched_name"] = str(row.get("name") or target_name)
        state.meta["recipe_resolved_name"] = str(row.get("name") or target_name)
        state.meta["recipe_match_score"] = 1.0
        state.meta["recipe_fast_path_source"] = "document_index"
        state.meta["recipe_fast_path_row"] = row
        self.logger.info("菜谱详情PDF文档命中 目标=%s 来源=%s", target_name, row.get("source"))
        return True

    def _match_recipe_name(self, target_name: str) -> RecipeMatch | None:
        cached = self._get_cached_recipe_match(target_name)
        if isinstance(cached, RecipeMatch):
            self._last_recipe_match_cache_hit = True
            return cached
        self._last_recipe_match_cache_hit = False

        exact = self._lookup_recipe_by_exact_name(target_name)
        if exact:
            match = RecipeMatch(exact, "standard_exact", str(exact["name"]), target_name, 1.0, [str(exact["name"])])
            self._set_cached_recipe_match(target_name, match)
            return match

        alias_target = resolve_recipe_alias(target_name)
        if alias_target and normalize_recipe_name(alias_target) != normalize_recipe_name(target_name):
            alias_match = self._lookup_recipe_by_exact_name(alias_target)
            if alias_match:
                match = RecipeMatch(
                    alias_match,
                    "alias_exact",
                    str(alias_match["name"]),
                    alias_target,
                    1.0,
                    [alias_target],
                )
                self._set_cached_recipe_match(target_name, match)
                return match

        fuzzy = self._lookup_recipe_by_fuzzy_name(target_name)
        if fuzzy and fuzzy.score >= FUZZY_NAME_THRESHOLD:
            self._set_cached_recipe_match(target_name, fuzzy)
            return fuzzy

        vector = self._lookup_recipe_by_vector_candidate(target_name)
        if vector and vector.score >= VECTOR_NAME_THRESHOLD:
            self._set_cached_recipe_match(target_name, vector)
            return vector
        return None

    def _recipe_match_cache_key(self, target_name: str) -> str:
        return stable_cache_key(
            "recipe_match",
            {
                "target_name": normalize_recipe_name(target_name),
                "data_version": cache_data_version(),
                "fuzzy_threshold": FUZZY_NAME_THRESHOLD,
                "vector_threshold": VECTOR_NAME_THRESHOLD,
            },
        )

    def _get_cached_recipe_match(self, target_name: str) -> RecipeMatch | None:
        data = self.cache.get_json(self._recipe_match_cache_key(target_name))
        if not isinstance(data, dict):
            return None
        if data.get("status") == "miss":
            return None
        row = data.get("row")
        if not isinstance(row, dict):
            return None
        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        candidates = data.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        return RecipeMatch(
            row=normalize_recipe_row(row),
            level=str(data.get("level") or "cached"),
            matched_name=str(data.get("matched_name") or row.get("name") or ""),
            resolved_name=str(data.get("resolved_name") or row.get("name") or ""),
            score=score,
            candidates=[str(item) for item in candidates],
        )

    def _set_cached_recipe_match(self, target_name: str, match: RecipeMatch | None) -> None:
        if match is None:
            return
        ttl = cache_ttl_seconds("CACHE_RECIPE_MATCH_TTL_SECONDS", 24 * 60 * 60)
        self.cache.set_json(
            self._recipe_match_cache_key(target_name),
            {
                "status": "hit",
                "row": match.row,
                "level": match.level,
                "matched_name": match.matched_name,
                "resolved_name": match.resolved_name,
                "score": match.score,
                "candidates": match.candidates,
            },
            ttl_seconds=ttl,
        )

    def _lookup_recipe_by_exact_name(self, target_name: str) -> dict[str, object] | None:
        row = self._lookup_recipe_from_mysql_exact(target_name)
        if row is not None:
            return row
        return self._lookup_recipe_from_local_exact(target_name)

    def _lookup_recipe_from_mysql_exact(self, target_name: str) -> dict[str, object] | None:
        if self.mysql_store is None:
            return None
        try:
            rows = self.mysql_store.read_query(
                """
                SELECT
                  r.name,
                  r.category,
                  r.cooking_time_minutes,
                  r.difficulty,
                  r.calories_per_100g AS calories,
                  r.protein_g_per_100g,
                  r.fat_g_per_100g,
                  r.nutrition_estimated,
                  r.steps,
                  GROUP_CONCAT(DISTINCT i.name ORDER BY i.name SEPARATOR '、') AS ingredients,
                  '' AS tags,
                  '' AS suitable_for
                FROM recipes r
                LEFT JOIN recipe_ingredients ri ON ri.recipe_id = r.id
                LEFT JOIN ingredients i ON i.id = ri.ingredient_id
                WHERE r.name = %s
                GROUP BY
                  r.id, r.name, r.category, r.cooking_time_minutes, r.difficulty,
                  r.calories_per_100g, r.protein_g_per_100g, r.fat_g_per_100g,
                  r.nutrition_estimated, r.steps
                LIMIT 1
                """,
                (target_name,),
            )
        except Exception as exc:
            state_message = f"{type(exc).__name__}: {exc}"
            self.logger.warning("recipe mysql fast lookup failed target=%s error=%s", target_name, state_message)
            return None
        return normalize_recipe_row(rows[0]) if rows else None

    def _lookup_recipe_from_local_exact(self, target_name: str) -> dict[str, object] | None:
        for recipe in getattr(self.retriever, "recipes", []):
            if normalize_recipe_name(target_name) == normalize_recipe_name(recipe.name):
                return recipe_to_row(recipe)
        return None

    def _lookup_recipe_from_document_index_exact(self, target_name: str) -> dict[str, object] | None:
        normalized_target = normalize_recipe_name(target_name)
        if not normalized_target or not PROCESSED_DIR.exists():
            return None

        metadata_paths = sorted(
            PROCESSED_DIR.glob("*_metadata.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for metadata_path in metadata_paths:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.logger.warning("document metadata lookup failed path=%s error=%s", metadata_path, exc)
                continue
            chunks = metadata.get("chunks") if isinstance(metadata, dict) else None
            if not isinstance(chunks, list):
                continue
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
                names = [
                    chunk_metadata.get("dish_name"),
                    chunk_metadata.get("title"),
                    chunk.get("title"),
                ]
                if not any(normalize_recipe_name(str(name or "")) == normalized_target for name in names):
                    continue
                return {
                    "name": str(chunk_metadata.get("dish_name") or chunk.get("title") or target_name),
                    "ingredients": str(chunk_metadata.get("ingredients") or ""),
                    "seasonings": str(chunk_metadata.get("seasonings") or ""),
                    "tips": str(chunk_metadata.get("tips") or ""),
                    "text": str(chunk.get("text") or ""),
                    "source": str(chunk.get("source") or metadata_path.name),
                    "source_type": str(chunk.get("source_type") or ""),
                    "chunk_id": str(chunk.get("chunk_id") or ""),
                    "pages": chunk_metadata.get("pages") or [],
                    "metadata_path": str(metadata_path),
                }
        return None

    def _lookup_recipe_by_fuzzy_name(self, target_name: str) -> RecipeMatch | None:
        candidates = []
        for recipe in getattr(self.retriever, "recipes", []):
            score = recipe_name_similarity(target_name, recipe.name)
            candidates.append((recipe, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if not candidates:
            return None
        recipe, score = candidates[0]
        names = [item[0].name for item in candidates[:5]]
        return RecipeMatch(recipe_to_row(recipe), "fuzzy_candidate", recipe.name, recipe.name, score, names)

    def _lookup_recipe_by_vector_candidate(self, target_name: str) -> RecipeMatch | None:
        try:
            candidates = self.retriever.search(target_name, 5)
        except Exception as exc:
            self.logger.warning("recipe vector candidate lookup failed target=%s error=%s", target_name, exc)
            return None
        if not candidates:
            return None
        scored = [(recipe, recipe_name_similarity(target_name, recipe.name)) for recipe, _ in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        recipe, score = scored[0]
        names = [item[0].name for item in scored[:5]]
        return RecipeMatch(recipe_to_row(recipe), "vector_candidate", recipe.name, recipe.name, score, names)


def violates_preferences(ingredients: list[str], dislikes: list[str], allergies: list[str]) -> bool:
    return violates_ingredients(ingredients, [item for item in dislikes + allergies if item])


def recipe_to_row(recipe: Recipe) -> dict[str, object]:
    return {
        "name": recipe.name,
        "category": recipe.category,
        "cooking_time_text": recipe.cooking_time,
        "cooking_time_minutes": None,
        "difficulty": recipe.difficulty,
        "calories": recipe.calories,
        "steps": recipe.steps,
        "ingredients": "、".join(recipe.ingredients),
        "tags": "、".join(recipe.tags),
        "suitable_for": "、".join(recipe.suitable_for),
    }


def normalize_recipe_row(row: dict[str, object]) -> dict[str, object]:
    cooking_time_minutes = row.get("cooking_time_minutes") or 0
    return {
        "name": row.get("name") or "",
        "category": row.get("category") or "",
        "cooking_time_text": f"{cooking_time_minutes}分钟" if cooking_time_minutes else "",
        "cooking_time_minutes": cooking_time_minutes,
        "difficulty": row.get("difficulty") or "",
        "calories": row.get("calories") or 0,
        "protein_g_per_100g": row.get("protein_g_per_100g") or 0,
        "fat_g_per_100g": row.get("fat_g_per_100g") or 0,
        "nutrition_estimated": bool(row.get("nutrition_estimated", True)),
        "steps": row.get("steps") or "",
        "ingredients": row.get("ingredients") or "",
        "tags": row.get("tags") or "",
        "suitable_for": row.get("suitable_for") or "",
    }


def format_direct_recipe_answer(row: dict[str, object]) -> str:
    ingredients = str(row.get("ingredients") or "暂无记录")
    tags = str(row.get("tags") or "暂无标签")
    suitable_for = str(row.get("suitable_for") or "暂无记录")
    steps = split_steps(str(row.get("steps") or "暂无步骤记录"))
    lines = [
        f"菜名：{row.get('name')}",
        "来源：本地菜谱库",
        "基础信息：",
        f"- 分类：{row.get('category') or '未分类'}",
        f"- 难度：{row.get('difficulty') or '未知'}",
        f"- 用时：{row.get('cooking_time_text') or '时间未知'}",
        f"- 热量：约 {row.get('calories') or 0} kcal/100g",
        f"- 蛋白质：约 {row.get('protein_g_per_100g') or 0} g/100g",
        f"- 脂肪：约 {row.get('fat_g_per_100g') or 0} g/100g",
        "主要食材：",
        f"- {ingredients}",
        "适用标签：",
        f"- 标签：{tags}",
        f"- 适合人群/场景：{suitable_for}",
        "做法步骤：",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))
    lines.extend(
        [
            "提示：",
            "- 以上内容来自本地菜谱库，未调用大模型改写。",
            "- 如有过敏、控糖、低盐或特殊健康需求，请按自身情况调整油盐糖用量。",
        ]
    )
    return "\n".join(lines)


def format_document_recipe_answer(row: dict[str, object]) -> str:
    text = str(row.get("text") or "").strip()
    pages = row.get("pages") or []
    page_text = "、".join(str(page) for page in pages) if isinstance(pages, list) else str(pages)
    source = str(row.get("source") or "本地PDF文档")
    chunk_id = str(row.get("chunk_id") or "")

    lines = [
        f"菜名：{row.get('name')}",
        "来源：本地PDF文档索引",
    ]
    source_bits = [
        bit
        for bit in (
            source,
            f"页码：{page_text}" if page_text else "",
            f"chunk：{chunk_id}" if chunk_id else "",
        )
        if bit
    ]
    if source_bits:
        lines.append("溯源：" + "；".join(source_bits))
    if text:
        lines.append("")
        lines.append(text)
    else:
        lines.extend(
            [
                "原料：" + str(row.get("ingredients") or "暂无记录"),
                "调料：" + str(row.get("seasonings") or "暂无记录"),
            ]
        )
        tips = str(row.get("tips") or "").strip()
        if tips:
            lines.append("小提示：" + tips)
    lines.extend(
        [
            "",
            "提示：",
            "- 以上内容来自本地PDF文档索引，不是通用知识兜底生成。",
            "- 如有过敏、控糖、低盐或特殊健康需求，请按自身情况调整油盐糖用量。",
        ]
    )
    return "\n".join(lines)


def split_steps(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[；;\n]+", text) if part.strip()]
    return parts or [text]


def extract_recipe_detail_target(message: str) -> str:
    text = message.strip()
    alias_hit = extract_known_recipe_alias(text)
    if alias_hit:
        return alias_hit
    patterns = [
        r"(?:想吃|要吃|来一份|来个)(.+?)(?:了|啦|吧)?$",
        r"(.+?)(?:怎么做|怎么烧|怎么煮|怎么炒|如何做|做法|教程|步骤)",
        r"^(?:做|烧|煮|炒)(.+?)(?:怎么做|做法|教程|步骤)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        target = cleanup_recipe_target(match.group(1))
        if target:
            return target
    standalone = cleanup_recipe_target(text)
    if standalone.startswith(EXPLICIT_DISH_STYLE_PREFIXES) and is_specific_dish_target(standalone):
        return standalone
    return ""


def is_specific_dish_target(target: str) -> bool:
    """Distinguish a concrete dish name from broad recommendation constraints."""
    value = cleanup_recipe_target(target)
    if not 2 <= len(value) <= 24:
        return False
    if any(word in value for word in GENERIC_RECIPE_TARGET_WORDS):
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def extract_known_recipe_alias(text: str) -> str:
    aliases = sorted(ALIAS_TO_STANDARD, key=len, reverse=True)
    for alias in aliases:
        if alias in text:
            return alias
    return ""


def cleanup_recipe_target(value: str) -> str:
    target = re.sub(r"^(请问|请|我想知道|帮我看看|帮我查查|给我讲讲|想问下)", "", value.strip())
    target = re.sub(r"(这道菜|这个菜|一道|一个|的)$", "", target.strip())
    return target.strip(" ：:，,。？?！!、")


def normalize_recipe_name(value: str) -> str:
    return re.sub(r"[\s：:，,。？?！!、（）()《》「」『』]", "", value)


def resolve_recipe_alias(target_name: str) -> str:
    normalized = normalize_recipe_name(target_name)
    normalized_aliases = {
        normalize_recipe_name(alias): standard for alias, standard in ALIAS_TO_STANDARD.items()
    }
    return normalized_aliases.get(normalized, "")


def recipe_name_similarity(left: str, right: str) -> float:
    normalized_left = normalize_recipe_name(left)
    normalized_right = normalize_recipe_name(right)
    if not normalized_left or not normalized_right:
        return 0.0
    if normalized_left == normalized_right:
        return 1.0
    sequence_score = SequenceMatcher(None, normalized_left, normalized_right).ratio()
    left_chars = set(normalized_left)
    right_chars = set(normalized_right)
    overlap = len(left_chars & right_chars) / max(len(left_chars | right_chars), 1)
    length_penalty = min(len(normalized_left), len(normalized_right)) / max(len(normalized_left), len(normalized_right))
    return max(sequence_score, overlap * length_penalty)


def recipe_name_matches(target_name: str, recipe_name: str) -> bool:
    target = normalize_recipe_name(target_name)
    recipe = normalize_recipe_name(recipe_name)
    if not target or not recipe:
        return False
    return target == recipe
