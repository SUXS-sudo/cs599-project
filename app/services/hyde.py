from __future__ import annotations

import os
from dataclasses import dataclass


def parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HyDEResult:
    original_query: str
    hypothetical_document: str
    generator: str
    enabled: bool
    error: str = ""


class HyDEGenerator:
    """Generate an LLM hypothetical document for vector recall only.

    The generated text is never evidence for answering. It is only an extra
    vector query used to improve recall in Chroma/FAISS.
    """

    def __init__(self, llm_client=None) -> None:
        try:
            from app.services.llm_client import load_dotenv

            load_dotenv()
        except Exception:
            pass
        self.enabled = parse_bool(os.getenv("HYDE_ENABLED", "false"))
        self.generator = "llm"
        self.max_chars = int(os.getenv("HYDE_MAX_CHARS", "420"))
        self.llm_client = llm_client

    def generate(self, query: str) -> HyDEResult:
        original = query.strip()
        if not self.enabled or not original:
            return HyDEResult(original, "", self.generator, False)

        llm_text = self._generate_with_llm(original)
        if not llm_text:
            return HyDEResult(original, "", self.generator, False, "llm_unavailable_or_empty")
        return HyDEResult(original, truncate_text(llm_text, self.max_chars), self.generator, True)

    def _generate_with_llm(self, query: str) -> str:
        try:
            client = self.llm_client
            if client is None:
                from app.services.llm_client import LLMClient

                client = LLMClient()
            if not client.available:
                return ""
            prompt = (
                "You are the HyDE generator for SmartRecipe retrieval. "
                "Generate one concise hypothetical recipe/document passage for vector search only. "
                "Do not answer the user directly. Do not invent exact quantities. "
                "Include likely dish names, ingredients, cooking method, seasonings, dietary goals, and related terms.\n\n"
                f"User query: {query}\n\n"
                "Hypothetical passage:"
            )
            return (client.generate(prompt, max_tokens=180, timeout=10) or "").strip()
        except Exception:
            return ""


def truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip()
