from __future__ import annotations

import json
from pathlib import Path

from app.agents.vision_agent import VisionAgent
from app.retriever import Recipe
from app.services.data_pipeline import normalize_recipe, run_recipe_pipeline
from app.services.image_analyzer import ImageAnalyzer
from app.state import AgentState


class FakeRetriever:
    def search(self, query: str, top_k: int):
        return [
            (
                Recipe(
                    name="番茄炒蛋",
                    ingredients=["番茄", "鸡蛋"],
                    category="家常菜",
                    cooking_time="15分钟",
                    difficulty="简单",
                    tags=["快手菜", "家常菜"],
                    calories=260,
                    suitable_for=["午餐"],
                    steps="鸡蛋打散；番茄炒出汁；合炒调味。",
                ),
                0.9,
            )
        ][:top_k]


class FakeVisionClient:
    available = True

    def __init__(self) -> None:
        self.called = False
        self.provider = "anthropic"

    def generate_with_image(self, prompt, image_bytes, mime_type="image/jpeg", max_tokens=800, timeout=45, model=None):
        self.called = True
        assert image_bytes == b"real-image"
        assert mime_type == "image/png"
        return (
            '{"dish_name":"虾仁西兰花","confidence":0.86,'
            '"ingredients":["虾仁","西兰花"],"cooking_method":"炒",'
            '"description":"图中像一盘虾仁西兰花"}'
        )


def test_image_analyzer_uses_filename_or_hint_fallback() -> None:
    result = ImageAnalyzer().analyze(b"fake-image", filename="tomato-egg.jpg", user_hint="")

    assert result.dish_name == "番茄炒蛋"
    assert result.confidence > 0.5
    assert "番茄" in result.ingredients


def test_image_analyzer_calls_vision_model_when_enabled(monkeypatch) -> None:
    client = FakeVisionClient()
    monkeypatch.setenv("ENABLE_VISION_LLM", "true")

    result = ImageAnalyzer(client).analyze(b"real-image", filename="dish.png", user_hint="识别这道菜")

    assert client.called is True
    assert result.source == "llm_vision"
    assert result.dish_name == "虾仁西兰花"
    assert result.ingredients == ["虾仁", "西兰花"]


def test_vision_agent_sets_vision_result_and_retrieves_recipes() -> None:
    agent = VisionAgent(ImageAnalyzer(), FakeRetriever())
    state = AgentState(
        user_input="这是什么菜？",
        session_id="s",
        top_k=1,
        intent="image_recipe_query",
        target_agent="vision_agent",
        meta={"image_bytes": b"fake-image", "image_filename": "tomato-egg.jpg"},
    )

    result = agent.run(state)

    assert result.vision_result["dish_name"] == "番茄炒蛋"
    assert result.retrieved_docs[0][0].name == "番茄炒蛋"
    assert "Vision Agent" in result.agent_output


def test_data_pipeline_normalizes_recipe_and_writes_output() -> None:
    artifact_dir = Path(__file__).resolve().parent.parent / ".test_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    source = artifact_dir / "raw_pipeline_test.json"
    output = artifact_dir / "clean_pipeline_test.json"
    source.write_text(
        json.dumps(
            [
                {
                    "title": "测试鸡蛋饼",
                    "ingredients": "鸡蛋,面粉",
                    "time": "12",
                    "steps": ["打蛋", "加面粉", "煎熟"],
                    "tags": ["早餐"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = run_recipe_pipeline(source, output)
    rows = json.loads(output.read_text(encoding="utf-8"))

    assert report.cleaned_count == 1
    assert rows[0]["name"] == "测试鸡蛋饼"
    assert rows[0]["cooking_time"] == "12分钟"
    assert normalize_recipe(rows[0]) is not None
