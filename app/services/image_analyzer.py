from __future__ import annotations

import hashlib
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.llm_client import LLMClient, load_dotenv
from app.services.logger import get_logger


logger = get_logger("services.image_analyzer")

IMAGE_HINTS = {
    "番茄炒蛋": ("番茄", "鸡蛋", "tomato", "egg", "hongshi", "xihongshi"),
    "鸡胸肉沙拉": ("鸡胸肉", "沙拉", "chicken", "salad"),
    "西兰花炒鸡胸肉": ("西兰花", "鸡胸肉", "broccoli", "chicken"),
    "虾仁西兰花": ("虾", "虾仁", "西兰花", "shrimp", "broccoli"),
    "豆腐青菜汤": ("豆腐", "青菜", "汤", "tofu", "soup"),
    "南瓜粥": ("南瓜", "粥", "pumpkin", "porridge"),
}


@dataclass
class ImageAnalysisResult:
    dish_name: str = "未知菜品"
    confidence: float = 0.0
    ingredients: list[str] = field(default_factory=list)
    cooking_method: str = "未知"
    description: str = ""
    source: str = "rule"
    image_hash: str = ""
    image_size: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dish_name": self.dish_name,
            "confidence": self.confidence,
            "ingredients": self.ingredients,
            "cooking_method": self.cooking_method,
            "description": self.description,
            "source": self.source,
            "image_hash": self.image_hash,
            "image_size": self.image_size,
            "warnings": self.warnings,
        }


class ImageAnalyzer:
    """Image understanding wrapper with deterministic fallback.

    The fallback is intentionally simple: it uses filename/query hints and
    image metadata so the final-version multimodal path remains runnable in a
    classroom/demo environment without a vision model.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        load_dotenv(override=True)
        self.llm_client = llm_client
        self.enable_llm_vision = parse_bool(os.getenv("ENABLE_VISION_LLM", "false"))

    def analyze(self, image_bytes: bytes, filename: str = "", user_hint: str = "") -> ImageAnalysisResult:
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        llm_result = self._analyze_with_llm(image_bytes, filename, user_hint)
        if llm_result:
            llm_result.image_hash = image_hash
            llm_result.image_size = len(image_bytes)
            return llm_result

        result = self._analyze_with_rules(filename, user_hint)
        result.image_hash = image_hash
        result.image_size = len(image_bytes)
        if result.source == "rule_fallback":
            result.description, result.warnings = self._fallback_reason()
        if not image_bytes:
            result.warnings.append("empty_image")
        return result

    def _fallback_reason(self) -> tuple[str, list[str]]:
        if not self.enable_llm_vision:
            return (
                "当前未启用视觉大模型，只能基于文件名和文字提示做保守识别。",
                ["vision_model_not_enabled"],
            )
        if not self.llm_client or not self.llm_client.available:
            return (
                "视觉大模型配置不完整或 LLM Client 不可用，只能基于文件名和文字提示做保守识别。",
                ["vision_llm_unavailable"],
            )
        return (
            "视觉大模型已启用，但本次图片识别调用失败，已回退到文件名和文字提示的保守识别。",
            ["vision_llm_failed"],
        )

    def _analyze_with_llm(
        self,
            image_bytes: bytes,
            filename: str,
            user_hint: str,
    ) -> ImageAnalysisResult | None:
        if not self.enable_llm_vision or not self.llm_client or not self.llm_client.available:
            return None
        try:
            mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
            prompt = (
                "你是 SmartRecipe 的 Vision Agent。请根据图片和用户提示识别菜品，"
                "只输出 JSON，不要 markdown，不要解释。"
                "返回紧凑 JSON："
                '{"dish_name":"...","confidence":0.0,"ingredients":["..."],'
                '"cooking_method":"...","description":"..."}\n'
                f"filename={filename}\nuser_hint={user_hint}\n"
            )
            raw = self.llm_client.generate_with_image(
                prompt,
                image_bytes=image_bytes,
                mime_type=mime_type,
                max_tokens=300,
                timeout=30,
            )
            if not raw:
                return None
            data = parse_json_object(raw)
            if not data:
                data = parse_vision_text(raw)
            return ImageAnalysisResult(
                dish_name=str(data.get("dish_name") or "未知菜品"),
                confidence=float(data.get("confidence") or 0.0),
                ingredients=[str(item) for item in data.get("ingredients", []) if str(item).strip()],
                cooking_method=str(data.get("cooking_method") or "未知"),
                description=str(data.get("description") or ""),
                source="llm_vision",
            )
        except Exception:
            logger.exception("vision llm analysis failed")
            return None

    @staticmethod
    def _analyze_with_rules(filename: str, user_hint: str) -> ImageAnalysisResult:
        text = f"{filename} {user_hint}".lower()
        for dish_name, hints in IMAGE_HINTS.items():
            if any(hint.lower() in text for hint in hints):
                ingredients = [hint for hint in hints if re.search(r"[\u4e00-\u9fff]", hint)]
                return ImageAnalysisResult(
                    dish_name=dish_name,
                    confidence=0.72,
                    ingredients=dedupe_ingredients(ingredients),
                    cooking_method=infer_method(text),
                    description=f"根据文件名或用户提示，图片可能是「{dish_name}」。",
                    source="rule_hint",
                )
        return ImageAnalysisResult(
            dish_name="未知菜品",
            confidence=0.25,
            ingredients=[],
            cooking_method=infer_method(text),
            description="当前未启用视觉大模型，只能基于文件名和文字提示做保守识别。",
            source="rule_fallback",
            warnings=["vision_model_not_enabled"],
        )


def infer_method(text: str) -> str:
    if any(word in text for word in ("汤", "soup")):
        return "煮"
    if any(word in text for word in ("粥", "porridge")):
        return "熬煮"
    if any(word in text for word in ("沙拉", "salad")):
        return "拌"
    if any(word in text for word in ("烤", "空气炸锅", "air")):
        return "烤/空气炸"
    return "炒/拌/煮待确认"


def dedupe_ingredients(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item not in result and item != "汤":
            result.append(item)
    return result


def parse_json_object(raw: str) -> dict[str, Any] | None:
    import json

    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_vision_text(raw: str) -> dict[str, Any]:
    text = raw.strip()
    dish_name = "未知菜品"
    for candidate in IMAGE_HINTS:
        if candidate in text:
            dish_name = candidate
            break
    ingredients = []
    for ingredient in ("番茄", "鸡蛋", "葱", "虾仁", "西兰花", "鸡胸肉", "豆腐", "南瓜"):
        if ingredient in text and ingredient not in ingredients:
            ingredients.append(ingredient)
    confidence = 0.75 if dish_name != "未知菜品" else 0.4
    percent_match = re.search(r"(\d{2,3})\s*%", text)
    if percent_match:
        confidence = min(int(percent_match.group(1)) / 100, 1.0)
    return {
        "dish_name": dish_name,
        "confidence": confidence,
        "ingredients": ingredients,
        "cooking_method": infer_method(text),
        "description": text[:240],
    }


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
