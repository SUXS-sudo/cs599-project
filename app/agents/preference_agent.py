from __future__ import annotations

import re

from app.services.memory import MemoryStore, UserPreferences
from app.services.mysql_store import MySQLStore
from app.state import AgentState


PREFERENCE_KEYWORDS = ("清淡", "少油", "低脂", "低糖", "高蛋白", "素食", "快手", "不辣", "低盐")
PREFERENCE_ALIASES = {
    "牛肉": ["牛肉", "牛里脊", "牛腩", "肥牛", "牛肉末"],
    "虾": ["虾", "虾仁", "虾皮"],
    "鸡蛋": ["鸡蛋", "蛋液"],
    "辣": ["辣", "干辣椒", "花椒", "豆瓣酱"],
}
ALLERGY_RE = re.compile(r"(?:我)?(?:对)?([\u4e00-\u9fffA-Za-z0-9]{1,12})(?:过敏|会过敏)")
DISLIKE_PATTERNS = (
    re.compile(r"(?:我)?不吃([\u4e00-\u9fffA-Za-z0-9]{1,12})"),
    re.compile(r"(?:以后|之后)?(?:别|不要|不想)(?:给我)?(?:推荐|吃)?([\u4e00-\u9fffA-Za-z0-9]{1,12})"),
)


class PreferenceAgent:
    def __init__(
        self,
        memory_store: MemoryStore,
        mysql_store: MySQLStore | None = None,
        sync_mysql: bool = True,
    ) -> None:
        self.memory_store = memory_store
        self.sync_mysql = sync_mysql
        self.mysql_store = mysql_store if mysql_store is not None else (MySQLStore() if sync_mysql else None)

    def run(self, state: AgentState) -> AgentState:
        extracted = extract_preferences(state.user_input)
        if any(extracted.values()):
            current = self.memory_store.update_preferences(
                state.session_id,
                preferences=extracted["preferences"],
                allergies=extracted["allergies"],
                dislikes=extracted["dislikes"],
            )
            state.meta["preference_update"] = extracted
        else:
            current = self.memory_store.get_preferences(state.session_id)

        if self.sync_mysql and self.mysql_store is not None and any(extracted.values()):
            try:
                self.mysql_store.upsert_user_preferences(
                    state.session_id,
                    current.preferences,
                    current.allergies,
                    current.dislikes,
                )
                state.meta["preference_mysql_sync"] = "ok"
            except Exception as exc:
                state.meta["preference_mysql_sync"] = f"failed:{type(exc).__name__}"

        state.meta["user_preferences"] = current.to_dict()
        state.meta["preference_backend"] = getattr(self.memory_store, "backend", "memory")
        return state


def extract_preferences(message: str) -> dict[str, list[str]]:
    text = message.strip()
    preferences = [keyword for keyword in PREFERENCE_KEYWORDS if keyword in text]
    allergies = clean_items(ALLERGY_RE.findall(text))
    dislikes: list[str] = []
    for pattern in DISLIKE_PATTERNS:
        dislikes.extend(pattern.findall(text))
    return {
        "preferences": clean_items(preferences),
        "allergies": clean_items(allergies),
        "dislikes": clean_items(dislikes),
    }


def clean_items(items: list[str]) -> list[str]:
    cleaned = []
    stop_words = {"推荐", "吃", "给我", "的", "菜", "东西", "了", "辣的"}
    for item in items:
        value = item.strip(" ，。,.!！?？")
        if value == "辣的":
            value = "辣"
        if value and value not in stop_words and value not in cleaned:
            cleaned.append(value)
    return cleaned


def preferences_to_query_suffix(preferences: UserPreferences) -> str:
    parts = []
    if preferences.preferences:
        parts.append("偏好：" + "、".join(preferences.preferences))
    if preferences.dislikes:
        parts.append("不喜欢或不吃：" + "、".join(preferences.dislikes))
    if preferences.allergies:
        parts.append("过敏：" + "、".join(preferences.allergies))
    return "；".join(parts)


def expand_preference_terms(items: list[str]) -> list[str]:
    expanded = []
    for item in items:
        terms = PREFERENCE_ALIASES.get(item, [item])
        for term in terms:
            if term and term not in expanded:
                expanded.append(term)
    return expanded


def violates_ingredients(ingredients: list[str], blocked_terms: list[str]) -> bool:
    expanded = expand_preference_terms(blocked_terms)
    return any(block in ingredient or ingredient in block for block in expanded for ingredient in ingredients)
