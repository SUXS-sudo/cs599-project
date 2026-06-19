from __future__ import annotations

from app.services.hyde import HyDEGenerator


class FakeLLM:
    available = True

    def generate(self, prompt: str, max_tokens: int = 180, timeout: int = 10) -> str:
        return "tomato egg recipe hypothetical passage with ingredients and cooking method"


class UnavailableLLM:
    available = False


def test_hyde_uses_llm_generated_hypothetical_document(monkeypatch) -> None:
    monkeypatch.setenv("HYDE_ENABLED", "true")
    monkeypatch.setenv("HYDE_MAX_CHARS", "120")

    result = HyDEGenerator(FakeLLM()).generate("tomato egg recipe")

    assert result.enabled is True
    assert result.generator == "llm"
    assert result.hypothetical_document.startswith("tomato egg recipe")
    assert len(result.hypothetical_document) <= 120


def test_hyde_skips_when_llm_is_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("HYDE_ENABLED", "true")

    result = HyDEGenerator(UnavailableLLM()).generate("tomato egg recipe")

    assert result.enabled is False
    assert result.hypothetical_document == ""
    assert result.error == "llm_unavailable_or_empty"
