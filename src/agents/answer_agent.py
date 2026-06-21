from __future__ import annotations

import os
import re

from src.retriever import Recipe
from src.agents.preference_agent import expand_preference_terms
from src.services.answer_guard import build_evidence_text, safe_repair_fallback, verify_answer
from src.services.cache_store import cache_data_version, cache_ttl_seconds, get_cache_store, stable_cache_key
from src.services.llm_client import LLMClient
from src.state import AgentState


class AnswerAgent:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self.cache = get_cache_store()

    def run(self, state: AgentState) -> AgentState:
        if state.meta.get("dish_name_only"):
            dish_name = re.sub(r"[^\u4e00-\u9fff]", "", str(state.meta.get("resolved_dish_name") or ""))
            if dish_name:
                state.final_answer = dish_name
                state.generator = "rule"
                state.meta["answer_mode"] = "dish_name_only"
                state.meta["answer_guard"] = "not_required"
                return state
        if state.intent == "out_of_scope":
            state.final_answer = state.agent_output
            state.generator = "rule"
            state.meta["answer_guard"] = "not_required"
            return state
        if state.meta.get("answer_mode") == "direct":
            state.final_answer = state.agent_output
            state.generator = "direct"
            state.meta["answer_guard"] = "direct_structured_output"
            self._apply_boundary_response_policy(state)
            return state
        if state.meta.get("recipe_source") == "llm_fallback":
            state.final_answer = self._llm_recipe_fallback_answer(state)
            state.generator = self.llm_client.provider if self.llm_client.available else "rule"
            state.meta["answer_guard"] = "llm_fallback_declared"
            self._apply_boundary_response_policy(state)
            return state
        if state.meta.get("recipe_source") == "llm_fallback_query":
            state.final_answer = self._llm_query_fallback_answer(state)
            state.generator = self.llm_client.provider if self.llm_client.available else "rule"
            state.meta["answer_guard"] = "llm_fallback_declared"
            self._apply_boundary_response_policy(state)
            return state
        if state.intent == "image_recipe_query" and not self.llm_client.available:
            if not wants_image_recommendations(state.user_input):
                state.meta["image_reference_recipe_count"] = len(state.retrieved_docs)
                state.retrieved_docs = []
            state.final_answer = self._template_image_answer(state)
            state.generator = "template"
            self._apply_boundary_response_policy(state)
            self._apply_guard(state)
            return state
        if state.intent in {"structured_recipe_query", "relationship_query", "multi_source_query"} and not self.llm_client.available:
            state.final_answer = state.agent_output
            state.generator = "rule"
            self._apply_boundary_response_policy(state)
            self._apply_guard(state)
            return state

        if state.intent == "image_recipe_query" and not wants_image_recommendations(state.user_input):
            state.meta["image_reference_recipe_count"] = len(state.retrieved_docs)
            state.retrieved_docs = []
        prompt = self._build_prompt(state)
        llm_answer = self.llm_client.generate(prompt)
        if llm_answer:
            state.final_answer = llm_answer
            state.generator = self.llm_client.provider
            self._apply_boundary_response_policy(state)
            self._apply_guard(state)
            return state

        state.final_answer = self._template_answer(state)
        state.generator = "template"
        self._apply_boundary_response_policy(state)
        self._apply_guard(state)
        return state

    @staticmethod
    def _apply_boundary_response_policy(state: AgentState) -> None:
        boundary = state.meta.get("query_boundary")
        if not isinstance(boundary, dict) or boundary.get("decision") != "caution":
            state.meta.setdefault("response_policy", "standard")
            return
        if boundary.get("reason_code") != "HEALTH_SENSITIVE":
            state.meta["response_policy"] = "caution_generic_v1"
            return

        state.meta["response_policy"] = "health_sensitive_v1"
        unsafe_patterns = (
            r"(可以|建议|应该|需要)\s*(立即)?停药",
            r"(无需|不用)\s*(服药|吃药|就医)",
            r"(替代|取代)\s*(处方|药物|治疗)",
            r"(根治|保证治愈|百分之百治愈)",
        )
        if any(re.search(pattern, state.final_answer, flags=re.I) for pattern in unsafe_patterns):
            state.final_answer = (
                "这个问题涉及疾病、用药或其他健康风险。当前菜谱和营养数据只能提供一般饮食参考，"
                "不能用于诊断、调整药物或替代治疗。请不要自行停药或改变处方，并咨询医生或注册营养师。"
            )
            state.meta["response_policy_action"] = "replaced_high_risk_health_claim"
            return

        notice = (
            "\n\n健康提示：以上仅为一般饮食参考，不构成诊断或治疗建议，也不能替代处方或医生意见。"
            "如涉及疾病控制、用药、孕期、婴幼儿或严重过敏，请结合个人情况咨询医生或注册营养师；"
            "过敏人群还应核对配料标签和交叉污染风险。"
        )
        if "不构成诊断或治疗建议" not in state.final_answer:
            state.final_answer = state.final_answer.rstrip() + notice
            state.meta["response_policy_action"] = "appended_health_notice"
        else:
            state.meta["response_policy_action"] = "health_notice_present"

    def _apply_guard(self, state: AgentState) -> None:
        if os.getenv("ENABLE_ANSWER_GUARD", "true").strip().lower() in {"0", "false", "no", "off"}:
            state.meta["answer_guard"] = "disabled"
            return
        max_retries = max(0, int(os.getenv("ANSWER_GUARD_MAX_RETRIES", "2")))
        verification = verify_answer(state)
        state.meta["answer_guard_initial_status"] = verification.status
        state.meta["answer_guard_retry_count"] = 0

        if verification.corrected_answer:
            state.final_answer = verification.corrected_answer
            self._apply_boundary_response_policy(state)
            state.meta["answer_guard"] = verification.status
            return

        while (
            verification.status == "retryable_unsupported_claims"
            and state.meta["answer_guard_retry_count"] < max_retries
            and self.llm_client.available
        ):
            state.meta["answer_guard_retry_count"] += 1
            state.meta["unsupported_claims"] = list(verification.unsupported_claims)
            repaired = self.llm_client.generate(
                self._build_repair_prompt(state, verification.unsupported_claims),
                max_tokens=900,
            )
            if not repaired:
                break
            state.final_answer = repaired.strip()
            self._apply_boundary_response_policy(state)
            verification = verify_answer(state)

        if verification.status == "retryable_unsupported_claims":
            state.meta["unsupported_claims"] = list(verification.unsupported_claims)
            state.final_answer = safe_repair_fallback(state, verification.unsupported_claims)
            self._apply_boundary_response_policy(state)
            state.meta["answer_guard"] = "safe_fallback_after_retry"
            return
        state.meta["unsupported_claims"] = []
        state.meta["answer_guard"] = verification.status

    @staticmethod
    def _build_repair_prompt(state: AgentState, unsupported_claims: tuple[str, ...]) -> str:
        return (
            "你是 SmartRecipe 回答纠错器。只输出修正后的最终中文回答。\n"
            "必须删除或改写所有无法由证据直接支持的数字、菜名和事实；不得补充常识或猜测。\n"
            f"用户问题：{state.user_input}\n"
            f"待修正回答：{state.final_answer}\n"
            f"未支持声明：{list(unsupported_claims)}\n"
            f"允许使用的证据：\n{build_evidence_text(state)}\n"
            f"边界回答要求：{AnswerAgent._boundary_prompt_instruction(state)}\n"
        )

    def _llm_recipe_fallback_answer(self, state: AgentState) -> str:
        target_name = str(state.meta.get("recipe_detail_target") or state.user_input).strip()
        notice = (
            f"当前菜谱库中暂未收录「{target_name}」的标准菜谱。\n\n"
            "以下内容不来自本地菜谱库，而是基于通用烹饪知识生成的参考做法：\n"
        )
        cached = self._get_cached_llm_recipe_fallback(target_name)
        if cached:
            state.meta["llm_fallback_cache_hit"] = True
            return cached
        state.meta["llm_fallback_cache_hit"] = False

        if not self.llm_client.available:
            return (
                notice
                + "\n"
                + "LLM 当前不可用，暂时无法生成通用做法参考。你可以配置 API Key 后重试，"
                + "或让我推荐本地菜谱库里相近的菜。"
            )

        prompt = self._build_llm_recipe_fallback_prompt(state, target_name)
        generated = self.llm_client.generate(prompt, max_tokens=900)
        if not generated:
            return (
                notice
                + "\n"
                + "LLM 暂时没有返回可用做法。你可以稍后重试，或让我推荐本地菜谱库里相近的菜。"
            )
        answer = notice + "\n" + generated.strip()
        self._set_cached_llm_recipe_fallback(target_name, answer)
        return answer

    def _llm_recipe_fallback_cache_key(self, target_name: str) -> str:
        return stable_cache_key(
            "llm_fallback_recipe",
            {
                "target_name": target_name.strip(),
                "prompt_version": "recipe-fallback-v1",
                "model": getattr(self.llm_client, "model", ""),
                "provider": getattr(self.llm_client, "provider", ""),
                "data_version": cache_data_version(),
            },
        )

    def _get_cached_llm_recipe_fallback(self, target_name: str) -> str:
        data = self.cache.get_json(self._llm_recipe_fallback_cache_key(target_name))
        if not isinstance(data, dict):
            return ""
        answer = data.get("answer")
        return str(answer) if answer else ""

    def _set_cached_llm_recipe_fallback(self, target_name: str, answer: str) -> None:
        ttl = cache_ttl_seconds("CACHE_LLM_FALLBACK_TTL_SECONDS", 24 * 60 * 60)
        self.cache.set_json(
            self._llm_recipe_fallback_cache_key(target_name),
            {"answer": answer},
            ttl_seconds=ttl,
        )

    def _llm_query_fallback_answer(self, state: AgentState) -> str:
        question = state.user_input.strip()
        preferences = self._fallback_preferences(state)
        notice = "下面按你的问题和当前对话偏好给出通用参考推荐（由 LLM 生成，不视为本地菜谱库命中）：\n"
        cached = self._get_cached_llm_query_fallback(question, state.intent, preferences)
        if cached:
            state.meta["llm_fallback_cache_hit"] = True
            return cached
        state.meta["llm_fallback_cache_hit"] = False

        if not self.llm_client.available:
            state.meta["llm_fallback_attempted"] = False
            return self._generic_safe_recommendations(
                state,
                "下面按你的问题和当前对话偏好给出通用参考推荐（本地安全模板，不视为菜谱库命中）：\n",
            )

        prompt = self._build_llm_query_fallback_prompt(state)
        state.meta["llm_fallback_attempted"] = True
        generated = self.llm_client.generate(prompt, max_tokens=900)
        if not generated:
            return self._generic_safe_recommendations(
                state,
                "下面按你的问题和当前对话偏好给出通用参考推荐（LLM 本次未返回，已切换本地安全模板）：\n",
            )
        blocked_hits = self._blocked_terms_in_answer(generated, preferences)
        if blocked_hits:
            state.meta["llm_fallback_rejected_blocked_terms"] = blocked_hits
            return self._generic_safe_recommendations(
                state,
                "下面按你的问题和当前对话偏好给出通用参考推荐（LLM 草案触及禁用食材，已切换本地安全模板）：\n",
            )
        answer = notice + "\n" + generated.strip()
        self._set_cached_llm_query_fallback(question, state.intent, preferences, answer)
        return answer

    def _llm_query_fallback_cache_key(self, question: str, intent: str, preferences: dict) -> str:
        return stable_cache_key(
            "llm_fallback_query",
            {
                "question": question.strip(),
                "intent": intent,
                "preferences": preferences,
                "prompt_version": "query-fallback-v2-preference-safe",
                "model": getattr(self.llm_client, "model", ""),
                "provider": getattr(self.llm_client, "provider", ""),
                "data_version": cache_data_version(),
            },
        )

    def _get_cached_llm_query_fallback(self, question: str, intent: str, preferences: dict) -> str:
        data = self.cache.get_json(self._llm_query_fallback_cache_key(question, intent, preferences))
        if not isinstance(data, dict):
            return ""
        answer = data.get("answer")
        return str(answer) if answer else ""

    def _set_cached_llm_query_fallback(self, question: str, intent: str, preferences: dict, answer: str) -> None:
        ttl = cache_ttl_seconds("CACHE_LLM_FALLBACK_TTL_SECONDS", 24 * 60 * 60)
        self.cache.set_json(
            self._llm_query_fallback_cache_key(question, intent, preferences),
            {"answer": answer},
            ttl_seconds=ttl,
        )

    @staticmethod
    def _build_llm_query_fallback_prompt(state: AgentState) -> str:
        preferences = AnswerAgent._fallback_preferences(state)
        boundary_instruction = AnswerAgent._boundary_prompt_instruction(state)
        return (
            "你是 SmartRecipe 的通用饮食与烹饪建议兜底生成器。"
            "当前本地菜谱库、SQL/RAG 检索没有命中用户条件，所以你不能声称内容来自数据库、RAG、检索结果或本地菜谱库。\n"
            "请结合用户本轮问题和当前session偏好，直接给出至少2个可执行的推荐方案。"
            "过敏和不吃的食材是绝对禁用项，不得出现在主料、辅料、调味料或替代建议中；"
            "普通口味偏好用于选择和排序。不得只说没有结果，不得要求用户改用其他筛选条件。"
            "必须使用中文并严格按下面格式输出，不要添加markdown表格，不要说你检索到了菜谱。\n\n"
            f"用户问题：{state.user_input}\n"
            f"意图：{state.intent}\n"
            f"当前偏好：{preferences['preferences']}\n"
            f"绝对禁用-过敏：{preferences['allergies']}\n"
            f"绝对禁用-不吃：{preferences['dislikes']}\n"
            f"SQL/RAG 观察：{state.agent_output}\n"
            f"对话历史：\n{state.chat_history}\n\n"
            f"安全边界要求：{boundary_instruction}\n\n"
            "输出格式必须为：\n"
            "建议方向：\n"
            "- ...\n"
            "推荐方案：\n"
            "1. 名称：...\n"
            "   食材：...\n"
            "   做法：...\n"
            "   适合原因：...\n"
            "2. 名称：...\n"
            "   食材：...\n"
            "   做法：...\n"
            "   适合原因：...\n"
            "调整建议：\n"
            "- ...\n"
            "注意事项：\n"
            "- 如有疾病、过敏、孕期、儿童或老人饮食限制，请按实际情况减少油盐糖，并咨询专业人士。\n"
        )

    @staticmethod
    def _fallback_preferences(state: AgentState) -> dict[str, list[str]]:
        raw = state.meta.get("user_preferences")
        if not isinstance(raw, dict):
            raw = {}
        return {
            "preferences": [str(item) for item in raw.get("preferences", []) if str(item).strip()],
            "allergies": [str(item) for item in raw.get("allergies", []) if str(item).strip()],
            "dislikes": [str(item) for item in raw.get("dislikes", []) if str(item).strip()],
        }

    @staticmethod
    def _blocked_terms_in_answer(answer: str, preferences: dict[str, list[str]]) -> list[str]:
        blocked = expand_preference_terms(preferences["allergies"] + preferences["dislikes"])
        return [term for term in blocked if term and term in answer]

    @staticmethod
    def _generic_safe_recommendations(state: AgentState, notice: str) -> str:
        preferences = AnswerAgent._fallback_preferences(state)
        blocked = set(expand_preference_terms(preferences["allergies"] + preferences["dislikes"]))
        plans = [
            ("鸡胸肉西兰花碗", ["鸡胸肉", "西兰花", "糙米"], "鸡胸肉煎熟，与焯熟西兰花和少量糙米组合。"),
            ("香煎豆腐时蔬碗", ["豆腐", "西兰花", "胡萝卜"], "豆腐少油煎香，与焯熟时蔬拌匀。"),
            ("清蒸鱼蔬菜餐", ["鱼", "青菜", "姜"], "鱼加姜清蒸，搭配焯青菜。"),
            ("燕麦菌菇蔬菜粥", ["燕麦", "蘑菇", "青菜"], "燕麦煮软后加入菌菇和青菜煮熟。"),
        ]
        safe = []
        for name, ingredients, method in plans:
            if any(term in ingredient or ingredient in term for term in blocked for ingredient in ingredients):
                continue
            safe.append((name, ingredients, method))
            if len(safe) == 2:
                break
        if not safe:
            safe = [("自选安全食材套餐", ["确认不过敏的主食", "确认不过敏的蛋白质食材", "时令蔬菜"], "分别彻底加热后组合，少油少盐调味。")]
        lines = [notice, "推荐方案："]
        for index, (name, ingredients, method) in enumerate(safe, start=1):
            lines.extend(
                [
                    f"{index}. 名称：{name}",
                    f"   食材：{'、'.join(ingredients)}",
                    f"   做法：{method}",
                    "   适合原因：优先满足本轮饮食目标，并避开当前会话已记录的过敏和不吃食材。",
                ]
            )
        lines.append("注意事项：营养值和份量需按实际食材调整；严重过敏请再次核对配料和交叉污染风险。")
        return "\n".join(lines)

    @staticmethod
    def _build_llm_recipe_fallback_prompt(state: AgentState, target_name: str) -> str:
        boundary_instruction = AnswerAgent._boundary_prompt_instruction(state)
        return (
            "你是 SmartRecipe 的通用烹饪知识兜底生成器。"
            "当前本地菜谱库没有命中用户要查的目标菜，所以你不能声称内容来自数据库、RAG、检索结果或本地菜谱库。\n"
            "请基于通用烹饪知识，为用户生成参考做法。必须使用中文，必须严格按下面格式输出，"
            "不要添加 markdown 表格，不要添加额外标题，不要说你检索到了菜谱。\n\n"
            f"目标菜名：{target_name}\n"
            f"用户原问题：{state.user_input}\n"
            f"对话历史：\n{state.chat_history}\n\n"
            f"安全边界要求：{boundary_instruction}\n\n"
            "输出格式必须为：\n"
            f"菜名：{target_name}\n"
            "参考食材：\n"
            "- 主料：...\n"
            "- 辅料：...\n"
            "- 调味：...\n"
            "做法步骤：\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "4. ...\n"
            "火候与时间：\n"
            "- ...\n"
            "口味调整：\n"
            "- ...\n"
            "注意事项：\n"
            "- ...\n"
            "- 如有疾病、过敏、孕期、儿童或老人饮食限制，请按实际情况减少油盐糖，并咨询专业人士。\n"
        )

    def _build_prompt(self, state: AgentState) -> str:
        intent_instruction = self._intent_instruction(state.intent)
        boundary_instruction = self._boundary_prompt_instruction(state)
        return (
            "You are SmartRecipe, a Chinese recipe assistant. Answer in Chinese.\n"
            "Use only the retrieved recipes as factual basis. Do not invent unavailable recipe details.\n\n"
            "Only output the final answer for the user. Do not reveal reasoning, analysis, hidden thoughts, or planning.\n\n"
            f"Conversation history:\n{state.chat_history}\n\n"
            f"User question: {state.user_input}\n"
            f"Intent: {state.intent}\n"
            f"Agent observation: {state.agent_output}\n"
            f"Retrieved recipes:\n{self._format_recipes(state.retrieved_docs)}\n\n"
            f"Intent-specific instruction:\n{intent_instruction}\n\n"
            f"Safety boundary instruction:\n{boundary_instruction}\n\n"
            "Please provide a helpful, concise, structured final answer."
        )

    @staticmethod
    def _boundary_prompt_instruction(state: AgentState) -> str:
        boundary = state.meta.get("query_boundary")
        if not isinstance(boundary, dict) or boundary.get("decision") != "caution":
            return "Standard evidence-grounded answer policy."
        if boundary.get("reason_code") == "HEALTH_SENSITIVE":
            return (
                "这是健康敏感问题。只提供由当前证据支持的一般饮食信息，不诊断疾病，不承诺疗效，"
                "不建议停药、改药或用饮食替代处方和治疗。区分数据库事实与一般建议，明确不确定性。"
                "涉及疾病、孕期、婴幼儿、严重过敏或用药时，提醒咨询医生或注册营养师；"
                "涉及过敏时必须提示核对配料标签和交叉污染风险。"
            )
        return "Use a conservative answer, state uncertainty, and avoid instructions that could increase risk."

    @staticmethod
    def _intent_instruction(intent: str) -> str:
        if intent == "nutrition_query":
            return (
                "Focus on calories, macro tendency, dietary goal fit, and practical portion advice. "
                "Do not provide medical diagnosis. Add a caution for disease, pregnancy, children, older adults, or allergies."
            )
        if intent == "ingredient_replace":
            return (
                "Focus on ingredient substitution. Include substitute options, approximate ratio or usage, "
                "taste/texture impact, when not to substitute, and allergy/safety notes."
            )
        if intent == "recipe_detail":
            return "Focus on the selected dish's ingredients, step-by-step method, heat control, timing, and seasoning notes."
        if intent == "recipe_search":
            return "Recommend the best matching recipe first, then mention alternatives and why they match the user's constraints."
        if intent == "structured_recipe_query":
            return "Answer from the SQL Agent observation. Keep numeric values and ranking order unchanged."
        if intent == "relationship_query":
            return "Answer from the Cypher Agent graph observation. Explain the relationship briefly and keep result order unchanged."
        if intent == "multi_source_query":
            return "Answer from the Fusion Agent observation. Explain that results were merged from multiple sources and keep ranking order unchanged."
        if intent == "tool_query":
            return "Answer from the Tool Agent observations. Treat tool outputs as the only factual basis, and mention stored preferences only when the tool returned them."
        if intent == "image_recipe_query":
            instruction = (
                "Answer in this exact order: first state the image recognition result with confidence; "
                "second introduce the recognized dish itself when the dish name is known; third state whether the local recipe library has an exact match; "
                "fourth recommend similar recipes only as secondary references from retrieved evidence. "
                "If the image result is unknown or confidence is low, do not present similar recipes as the main answer; say they are only weak auxiliary references. "
                "Do not start the answer with similar recipe recommendations."
            )
            return instruction + (
                " If the user only asks what the dish is and does not explicitly ask for recommendations, similar recipes, or cooking methods, "
                "only identify and introduce the dish; do not discuss local recipe-library matches or similar recipe references."
            )
        return "If the question is out of scope, politely explain the system scope."

    def _template_answer(self, state: AgentState) -> str:
        if state.intent == "image_recipe_query":
            return self._template_image_answer(state)
        if not state.retrieved_docs:
            if state.intent in {
                "recipe_search",
                "recipe_detail",
                "ingredient_replace",
                "nutrition_query",
                "structured_recipe_query",
                "relationship_query",
                "multi_source_query",
                "tool_query",
            }:
                return self._generic_safe_recommendations(
                    state,
                    "下面按你的问题和当前对话偏好给出通用参考推荐（本地安全模板）：\n",
                )
            if state.agent_output:
                return state.agent_output
            return "你好，我可以继续帮你处理菜谱、食材和饮食相关问题。"

        recipe = state.retrieved_docs[0][0]
        lines = [
            f"根据你的问题，我推荐「{recipe.name}」。",
            f"它属于{recipe.category}，难度{recipe.difficulty}，大约需要{recipe.cooking_time}，热量约 {recipe.calories} 千卡。",
            f"主要食材：{'、'.join(recipe.ingredients)}。",
            f"做法：{recipe.steps}",
        ]
        if state.intent == "nutrition_query":
            lines.append("营养建议：这道菜可作为一般饮食参考；如有疾病、过敏、孕期或特殊身体状况，建议咨询医生或营养师。")
        elif state.intent == "ingredient_replace":
            lines.append("替换建议：可以根据口味和库存调整配料，但关键食材替换会影响口感；如涉及过敏食材，请优先避开。")
        else:
            lines.append("小提示：如果想吃得清淡一些，可以适当减少油和盐。")
        if len(state.retrieved_docs) > 1:
            alternatives = "、".join(recipe.name for recipe, _ in state.retrieved_docs[1:])
            lines.append(f"另外也可以考虑：{alternatives}。")
        return "\n".join(lines)

    def _template_image_answer(self, state: AgentState) -> str:
        vision = state.vision_result or state.meta.get("vision_result", {})
        dish_name = str(vision.get("dish_name") or "未知菜品")
        confidence = safe_float(vision.get("confidence"), 0.0)
        ingredients = [str(item) for item in vision.get("ingredients", []) if str(item).strip()]
        cooking_method = str(vision.get("cooking_method") or "待确认")
        description = str(vision.get("description") or "")
        is_unknown = dish_name in {"未知菜品", "未识别菜品", "未知"} or confidence < 0.5

        lines = [f"图片识别结果：这张图可能是「{dish_name}」，置信度约 {confidence:.0%}。"]
        if description:
            lines.append(f"识别说明：{description}")

        if is_unknown:
            lines.extend(
                [
                    "菜品介绍：当前图片识别不够明确，我不能把某一道菜当成确定结果来介绍。",
                    "本地菜谱库匹配：因为菜名不明确，暂时无法判断是否有完全匹配菜谱。",
                ]
            )
        else:
            ingredient_text = "、".join(ingredients) if ingredients else "图像结果未给出明确食材"
            lines.extend(
                [
                    f"菜品介绍：「{dish_name}」通常可以从食材、烹饪方式和口味特征来理解；本次识别到的主要食材是 {ingredient_text}，烹饪方式可能是{cooking_method}。",
                ]
            )

        if not wants_image_recommendations(state.user_input):
            lines.append("提示：如果想继续看做法或相似菜谱，可以直接告诉我。")
            return "\n".join(lines)

        if not is_unknown:
            lines.append(self._local_image_match_line(dish_name, state.retrieved_docs))

        if state.retrieved_docs:
            if is_unknown:
                lines.append("相似菜谱参考：下面只是基于图片提示和文字检索得到的弱相关参考，不代表图片中一定是这些菜。")
            else:
                lines.append("相似菜谱参考：如果本地库没有完全匹配，可以参考下面这些做法接近的菜。")
            for index, (recipe, _) in enumerate(state.retrieved_docs[: state.top_k], start=1):
                lines.append(
                    f"{index}. {recipe.name}：主要食材 {recipe_ingredients_text(recipe)}；"
                    f"用时 {recipe.cooking_time}；做法参考：{recipe.steps}"
                )
        else:
            lines.append("相似菜谱参考：本地菜谱库暂时没有检索到可参考的相似菜谱。")

        lines.append("提示：图片识别结果只作为参考，如果菜品和实际不符，可以补充菜名或主要食材，我会重新匹配。")
        return "\n".join(lines)

    @staticmethod
    def _local_image_match_line(dish_name: str, recipes: list[tuple[Recipe, float]]) -> str:
        for recipe, _ in recipes:
            if recipe.name == dish_name:
                return f"本地菜谱库匹配：已找到完全匹配的「{dish_name}」，下面优先参考库内做法。"
        return f"本地菜谱库匹配：暂时没有找到完全匹配的「{dish_name}」，下面只给相似做法参考。"

    @staticmethod
    def _format_recipes(recipes: list[tuple[Recipe, float]]) -> str:
        if not recipes:
            return "No retrieved recipes."
        blocks = []
        for recipe, score in recipes:
            blocks.append(
                f"- {recipe.name} | score={score:.3f} | ingredients={','.join(recipe.ingredients)} | "
                f"time={recipe.cooking_time} | difficulty={recipe.difficulty} | calories={recipe.calories} | "
                f"tags={','.join(recipe.tags)} | steps={recipe.steps}"
            )
        return "\n".join(blocks)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def recipe_ingredients_text(recipe: Recipe) -> str:
    return "、".join(recipe.ingredients) if recipe.ingredients else "暂无明确食材"


def wants_image_recommendations(message: str) -> bool:
    text = message.strip()
    return any(keyword in text for keyword in ("推荐", "类似", "相似", "做法", "怎么做", "菜谱", "食谱"))
