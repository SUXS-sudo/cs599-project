from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Conversation session id")
    top_k: int = Field(3, ge=1, le=5, description="Number of retrieved recipes")

    def normalized_session_id(self) -> str:
        return self.session_id or str(uuid4())


class RecipeHit(BaseModel):
    name: str
    score: float
    ingredients: list[str]
    category: str
    cooking_time: str
    difficulty: str
    tags: list[str]
    calories: int


class ChatResponse(BaseModel):
    answer: str
    intent: str
    agent: str
    session_id: str
    recipes: list[RecipeHit]
    meta: dict[str, Any] = Field(default_factory=dict)
