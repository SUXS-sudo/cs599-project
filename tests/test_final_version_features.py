from __future__ import annotations

import json
from pathlib import Path

from app.agents.answer_agent import AnswerAgent
from app.agents.router_agent import RouterAgent
from app.retriever import Recipe
from app.services.memory import MemoryStore
from app.agents.support_agents import DataAgent
from app.state import AgentState


class NoLLM:
    available = False

    def generate(self, *args, **kwargs):
        return None


class RecipeFallbackLLM:
    available = True
    provider = "test"

    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt, *args, **kwargs):
        self.prompt = prompt
        return "\n".join(
            [
                "菜名：红烧肘子",
                "参考食材：",
                "- 主料：猪肘子",
                "- 辅料：葱、姜",
                "- 调味：生抽、老抽、冰糖",
                "做法步骤：",
                "1. 焯水。",
                "2. 炖煮。",
                "火候与时间：",
                "- 小火炖至软烂。",
                "口味调整：",
                "- 按口味补盐。",
                "注意事项：",
                "- 控制油盐糖。",
            ]
        )


class QueryFallbackLLM:
    available = True
    provider = "test"

    def __init__(self) -> None:
        self.prompt = ""
        self.calls = 0

    def generate(self, prompt, *args, **kwargs):
        self.prompt = prompt
        self.calls += 1
        return "\n".join(
            [
                "建议方向：",
                "- 选择高蛋白、低油、蔬菜占比高的晚餐。",
                "推荐方案：",
                "1. 名称：鸡胸肉蔬菜碗",
                "   食材：鸡胸肉、西兰花、番茄",
                "   做法：鸡胸肉煎熟，蔬菜焯水后组合。",
                "   适合原因：蛋白质较高，油脂可控。",
                "调整建议：",
                "- 少油少盐。",
                "注意事项：",
                "- 如有疾病、过敏、孕期、儿童或老人饮食限制，请按实际情况减少油盐糖，并咨询专业人士。",
            ]
        )


class CountingLLM:
    available = True
    provider = "test"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return None


class PromptCaptureLLM:
    available = True
    provider = "test"

    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt, *args, **kwargs):
        self.prompt = prompt
        return "图片识别结果：可能是地三鲜。\n菜品介绍：地三鲜是家常菜。\n相似菜谱参考：暂无。"


def make_image_recipe(name: str = "家常清炒青椒川菜") -> Recipe:
    return Recipe(
        name=name,
        ingredients=["青椒", "土豆", "茄子"],
        category="家常菜",
        cooking_time="15分钟",
        difficulty="简单",
        tags=["家常", "清炒"],
        calories=260,
        suitable_for=["午餐"],
        steps="切配食材；热锅翻炒；调味出锅。",
    )


def test_memory_store_builds_long_term_summary() -> None:
    store = MemoryStore(max_messages=4)
    for index in range(4):
        store.add_turn("s", f"第{index}轮我想吃清淡菜", f"推荐第{index}道菜")

    debug = store.debug_session("s")
    formatted = store.format_history("s")

    assert debug["summary"]
    assert "Long-term summary" in formatted
    assert "Recent conversation" in formatted


def test_data_agent_runs_pipeline() -> None:
    artifact_dir = Path(__file__).resolve().parent.parent / ".test_artifacts"
    artifact_dir.mkdir(exist_ok=True)
    source = artifact_dir / "data_agent_source.json"
    output = artifact_dir / "data_agent_clean.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "测试番茄蛋",
                    "ingredients": ["番茄", "鸡蛋"],
                    "steps": "切番茄；炒鸡蛋；合炒。",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = DataAgent().run_pipeline(source, output)

    assert report["cleaned_count"] == 1
    assert output.exists()


def test_answer_guard_corrects_evidence_free_recipe_answer() -> None:
    state = AgentState(
        user_input="推荐一道不存在的菜",
        session_id="s",
        top_k=1,
        intent="recipe_search",
        target_agent="recipe_agent",
    )

    result = AnswerAgent(NoLLM()).run(state)

    assert result.meta["answer_guard"] == "corrected_no_evidence"
    assert "没有足够" in result.final_answer


def test_answer_agent_declares_llm_recipe_fallback_format() -> None:
    llm = RecipeFallbackLLM()
    state = AgentState(
        user_input="红烧肘子怎么做",
        session_id="s",
        top_k=1,
        intent="recipe_detail",
        target_agent="recipe_agent",
        agent_output="当前菜谱库中暂未收录「红烧肘子」的标准菜谱。",
        meta={"recipe_source": "llm_fallback", "recipe_detail_target": "红烧肘子"},
    )

    result = AnswerAgent(llm).run(state)

    assert result.generator == "test"
    assert result.meta["answer_guard"] == "llm_fallback_declared"
    assert "不来自本地菜谱库" in result.final_answer
    assert "菜名：红烧肘子" in result.final_answer
    assert "参考食材：" in result.final_answer
    assert "做法步骤：" in result.final_answer
    assert "不能声称内容来自数据库" in llm.prompt


def test_answer_agent_caches_llm_recipe_fallback(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "llm-fallback-cache-test")
    llm = RecipeFallbackLLM()
    agent = AnswerAgent(llm)

    state = AgentState(
        user_input="红烧肘子怎么做",
        session_id="s",
        top_k=1,
        intent="recipe_detail",
        target_agent="recipe_agent",
        agent_output="当前菜谱库中暂未收录「红烧肘子」的标准菜谱。",
        meta={"recipe_source": "llm_fallback", "recipe_detail_target": "红烧肘子"},
    )
    first = agent.run(state)
    second = agent.run(
        AgentState(
            user_input="红烧肘子的做法",
            session_id="s",
            top_k=1,
            intent="recipe_detail",
            target_agent="recipe_agent",
            agent_output="当前菜谱库中暂未收录「红烧肘子」的标准菜谱。",
            meta={"recipe_source": "llm_fallback", "recipe_detail_target": "红烧肘子"},
        )
    )

    assert first.meta["llm_fallback_cache_hit"] is False
    assert second.meta["llm_fallback_cache_hit"] is True
    assert llm.prompt
    assert second.final_answer == first.final_answer


def test_answer_agent_uses_llm_when_sql_query_has_no_rows(monkeypatch) -> None:
    monkeypatch.setenv("CACHE_DATA_VERSION", "llm-query-fallback-test")
    llm = QueryFallbackLLM()
    state = AgentState(
        user_input="推荐低脂晚餐",
        session_id="s",
        top_k=1,
        intent="structured_recipe_query",
        target_agent="sql_agent",
        agent_output="低脂相关菜谱：本地数据库暂时没有查到匹配结果。",
        meta={"recipe_source": "llm_fallback_query", "sql_status": "empty", "sql_rows": []},
    )

    result = AnswerAgent(llm).run(state)

    assert llm.calls == 1
    assert result.generator == "test"
    assert result.meta["answer_guard"] == "llm_fallback_declared"
    assert "不来自本地菜谱库" in result.final_answer
    assert "建议方向：" in result.final_answer
    assert "不能声称内容来自数据库" in llm.prompt


def test_image_answer_prompt_requires_dish_intro_before_similar_recipes() -> None:
    llm = PromptCaptureLLM()
    state = AgentState(
        user_input="这是什么菜？推荐类似做法。",
        session_id="s",
        top_k=1,
        intent="image_recipe_query",
        target_agent="vision_agent",
        agent_output="Vision Agent 图片识别结果：可能菜品=地三鲜，置信度=0.86。相似菜谱检索结果：家常清炒青椒川菜。",
        retrieved_docs=[(make_image_recipe(), 0.8)],
        vision_result={
            "dish_name": "地三鲜",
            "confidence": 0.86,
            "ingredients": ["土豆", "茄子", "青椒"],
            "cooking_method": "炒",
            "description": "图中像一盘地三鲜",
        },
    )

    result = AnswerAgent(llm).run(state)

    assert result.generator == "test"
    assert "first state the image recognition result" in llm.prompt
    assert "second introduce the recognized dish itself" in llm.prompt
    assert "Do not start the answer with similar recipe recommendations" in llm.prompt


def test_image_template_introduces_recognized_dish_before_similar_recipes() -> None:
    llm = CountingLLM()
    state = AgentState(
        user_input="这是什么菜？推荐类似做法。",
        session_id="s",
        top_k=1,
        intent="image_recipe_query",
        target_agent="vision_agent",
        agent_output="Vision Agent 图片识别结果：可能菜品=地三鲜，置信度=0.86。相似菜谱检索结果：家常清炒青椒川菜。",
        retrieved_docs=[(make_image_recipe(), 0.8)],
        vision_result={
            "dish_name": "地三鲜",
            "confidence": 0.86,
            "ingredients": ["土豆", "茄子", "青椒"],
            "cooking_method": "炒",
            "description": "图中像一盘地三鲜",
        },
    )

    result = AnswerAgent(llm).run(state)

    assert llm.calls == 1
    assert result.final_answer.index("图片识别结果") < result.final_answer.index("菜品介绍")
    assert result.final_answer.index("菜品介绍") < result.final_answer.index("本地菜谱库匹配")
    assert result.final_answer.index("本地菜谱库匹配") < result.final_answer.index("相似菜谱参考")
    assert "地三鲜" in result.final_answer
    assert "家常清炒青椒川菜" in result.final_answer


def test_image_template_downgrades_similar_recipes_when_recognition_is_unknown() -> None:
    llm = CountingLLM()
    state = AgentState(
        user_input="这是什么菜？推荐类似做法。",
        session_id="s",
        top_k=1,
        intent="image_recipe_query",
        target_agent="vision_agent",
        agent_output="Vision Agent 图片识别结果：未知菜品，置信度=0.25。相似菜谱检索结果：家常清炒青椒川菜。",
        retrieved_docs=[(make_image_recipe(), 0.8)],
        vision_result={
            "dish_name": "未知菜品",
            "confidence": 0.25,
            "ingredients": [],
            "cooking_method": "炒/拌/煮待确认",
            "description": "视觉大模型调用失败，已回退到保守识别。",
        },
    )

    result = AnswerAgent(llm).run(state)

    assert "不能把某一道菜当成确定结果来介绍" in result.final_answer
    assert "弱相关参考" in result.final_answer
    assert result.final_answer.index("菜品介绍") < result.final_answer.index("相似菜谱参考")


def test_router_rule_fast_routes_low_fat_dinner_to_sql_without_llm() -> None:
    llm = CountingLLM()
    state = AgentState(user_input="有什么低脂晚餐", session_id="s", top_k=3)

    result = RouterAgent(llm, enable_database_agents=True).run(state)

    assert llm.calls == 0
    assert result.intent == "structured_recipe_query"
    assert result.target_agent == "sql_agent"
    assert result.meta["router_mode"] == "rule_fast"


def test_answer_agent_returns_direct_output_without_llm() -> None:
    llm = CountingLLM()
    state = AgentState(
        user_input="番茄炒蛋怎么做",
        session_id="s",
        top_k=1,
        intent="recipe_detail",
        target_agent="recipe_agent",
        agent_output="菜名：番茄炒蛋\n来源：本地菜谱库",
        meta={"answer_mode": "direct"},
    )

    result = AnswerAgent(llm).run(state)

    assert llm.calls == 0
    assert result.generator == "direct"
    assert result.final_answer == state.agent_output
    assert result.meta["answer_guard"] == "direct_structured_output"
