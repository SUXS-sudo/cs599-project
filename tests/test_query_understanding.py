from __future__ import annotations

from src.agents.answer_agent import AnswerAgent
from src.agents.query_understanding_agent import QueryUnderstandingAgent
from src.agents.router_agent import RouterAgent
from src.services.query_understanding import generate_correction_candidates, preserves_critical_constraints
from src.state import AgentState


class TwoStageLLM:
    available = True

    def __init__(self, selected_index: int = 1, confidence: float = 0.96) -> None:
        self.selected_index = selected_index
        self.confidence = confidence
        self.prompts: list[str] = []

    def generate(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        if "查询理解器" in prompt:
            return (
                '{"intent":"recipe_search","entities":['
                '{"text":"推见","type":"intent_word","normalized":"推荐"},'
                '{"text":"低旨","type":"diet_goal","normalized":"低脂"},'
                '{"text":"万餐","type":"meal","normalized":"晚餐"}],'
                '"needs_correction":true,"reason":"存在常见同音错字"}'
            )
        return (
            f'{{"candidate_index":{self.selected_index},"confidence":{self.confidence},'
            '"reason":"候选完整保留用户意图"}'
        )


class NoLLM:
    available = False


def make_state(message: str) -> AgentState:
    return AgentState(user_input=message, session_id="query-understanding", top_k=3)


def test_two_stage_llm_selects_program_generated_candidate() -> None:
    llm = TwoStageLLM()

    result = QueryUnderstandingAgent(llm).run(make_state("推见几个低旨万餐"))

    assert len(llm.prompts) == 2
    assert "抽取" not in llm.prompts[1] or "意图实体抽取" in llm.prompts[1]
    assert result.user_input == "推荐几个低脂晚餐"
    assert result.meta["original_query"] == "推见几个低旨万餐"
    assert result.meta["query_understanding"]["status"] == "corrected"
    assert result.meta["query_understanding"]["mode"] == "llm_candidate_selection"


def test_low_confidence_selection_keeps_original_query() -> None:
    result = QueryUnderstandingAgent(TwoStageLLM(confidence=0.4)).run(make_state("推见几个低旨万餐"))

    assert result.user_input == "推见几个低旨万餐"
    assert result.meta["query_understanding"]["status"] == "unchanged"


def test_rule_fallback_corrects_known_typo_when_llm_unavailable() -> None:
    result = QueryUnderstandingAgent(NoLLM()).run(make_state("番茄抄蛋怎么做"))

    assert result.user_input == "番茄炒蛋怎么做"
    assert result.meta["query_understanding"]["mode"] == "rule_fallback"


def test_red_braised_pork_elbow_typo_is_corrected_before_routing() -> None:
    result = QueryUnderstandingAgent(NoLLM()).run(make_state("红晒肘子怎么做"))

    assert result.user_input == "红烧肘子怎么做"
    assert result.meta["query_understanding"]["status"] == "corrected"


def test_noisy_correct_dish_name_is_reduced_to_pure_name() -> None:
    agent = QueryUnderstandingAgent(NoLLM(), vocabulary={"番茄炒蛋"})

    result = agent.run(make_state("###番茄炒蛋123abc!!!"))

    assert result.user_input == "番茄炒蛋"
    assert result.meta["resolved_dish_name"] == "番茄炒蛋"
    assert result.meta["dish_name_only"] is True


def test_dish_cleanup_is_deterministic_even_if_llm_selects_original() -> None:
    agent = QueryUnderstandingAgent(TwoStageLLM(selected_index=0), vocabulary={"番茄炒蛋"})

    result = agent.run(make_state("番茄炒蛋999xyz"))

    assert result.user_input == "番茄炒蛋"
    assert result.meta["query_understanding"]["mode"] == "deterministic_dish_name_cleanup"


def test_inline_character_inside_dish_name_is_removed_before_routing() -> None:
    result = QueryUnderstandingAgent(TwoStageLLM(selected_index=0)).run(make_state("西红柿c炒鸡蛋怎么做"))

    assert result.user_input == "西红柿炒鸡蛋怎么做"
    assert result.meta["query_understanding"]["mode"] == "deterministic_inline_noise_cleanup"
    routed = RouterAgent(None, enable_database_agents=True).run(result)
    assert routed.intent == "recipe_detail"


def test_inline_noise_and_typo_are_composed_for_bare_dish_name() -> None:
    result = QueryUnderstandingAgent(TwoStageLLM(selected_index=0)).run(make_state("红晒c肘子"))

    assert result.user_input == "红烧肘子"
    assert result.meta["resolved_dish_name"] == "红烧肘子"
    answer = AnswerAgent(NoLLM()).run(result)
    assert answer.final_answer == "红烧肘子"


def test_inline_noise_and_typo_are_composed_before_detail_routing() -> None:
    result = QueryUnderstandingAgent(TwoStageLLM(selected_index=0)).run(make_state("红晒c肘子怎么做"))

    assert result.user_input == "红烧肘子怎么做"
    assert result.meta["query_understanding"]["mode"] == "deterministic_inline_noise_and_typo_cleanup"
    assert "dish_name_only" not in result.meta
    routed = RouterAgent(None, enable_database_agents=True).run(result)
    assert routed.intent == "recipe_detail"


def test_inline_digits_are_removed_but_real_quantity_is_preserved() -> None:
    noisy = QueryUnderstandingAgent(NoLLM()).run(make_state("红烧123肘子怎么做"))
    quantity = QueryUnderstandingAgent(NoLLM()).run(make_state("推荐2道低脂晚餐"))

    assert noisy.user_input == "红烧肘子怎么做"
    assert quantity.user_input == "推荐2道低脂晚餐"


def test_noisy_typo_dish_name_is_corrected_and_cleaned() -> None:
    result = QueryUnderstandingAgent(NoLLM()).run(make_state("红晒肘子123abc"))

    assert result.user_input == "红烧肘子"
    assert result.meta["resolved_dish_name"] == "红烧肘子"


def test_real_quantity_is_not_removed_as_dish_noise() -> None:
    result = QueryUnderstandingAgent(NoLLM()).run(make_state("推荐2道低脂晚餐"))

    assert result.user_input == "推荐2道低脂晚餐"
    assert "dish_name_only" not in result.meta


def test_dish_name_only_answer_contains_no_extra_text() -> None:
    state = make_state("红烧肘子")
    state.meta.update({"dish_name_only": True, "resolved_dish_name": "红烧肘子"})

    result = AnswerAgent(NoLLM()).run(state)

    assert result.final_answer == "红烧肘子"
    assert result.meta["answer_mode"] == "dish_name_only"


def test_llm_normalized_unknown_dish_can_enter_program_candidates() -> None:
    candidates = generate_correction_candidates(
        "糖促鲤鱼怎么做",
        {"entities": [{"text": "糖促鲤鱼", "type": "dish", "normalized": "糖醋鲤鱼"}]},
    )

    assert any(candidate.query == "糖醋鲤鱼怎么做" for candidate in candidates)


def test_candidate_generation_preserves_negation_allergy_and_numbers() -> None:
    original = "我不吃虾，对花生过敏，推荐2道低旨晚餐"
    candidates = generate_correction_candidates(original)

    assert any(candidate.query == "我不吃虾,对花生过敏,推荐2道低脂晚餐" for candidate in candidates)
    assert all(preserves_critical_constraints(original, candidate.query) for candidate in candidates)


def test_fuzzy_candidates_do_not_expand_an_exact_known_term() -> None:
    candidates = generate_correction_candidates("西红柿c炒鸡蛋怎么做", vocabulary={"鸡蛋", "鸡蛋羹"})

    assert any(candidate.query == "西红柿炒鸡蛋怎么做" for candidate in candidates)
    assert all("鸡蛋羹" not in candidate.query for candidate in candidates)


def test_llm_cannot_select_candidate_outside_program_list() -> None:
    result = QueryUnderstandingAgent(TwoStageLLM(selected_index=99)).run(make_state("低旨万餐"))

    assert result.user_input == "低旨万餐"
    assert result.meta["query_understanding"]["selected_index"] == 0


def test_corrected_query_is_checked_against_safety_boundary() -> None:
    result = QueryUnderstandingAgent(NoLLM()).run(make_state("教我怎么投讀"))

    assert result.user_input == "教我怎么投毒"
    assert result.meta["safety_status"] == "blocked"
    assert result.meta["safety_reason"] == "RESOLVED_UNSAFE_INSTRUCTION"
