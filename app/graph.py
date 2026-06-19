from __future__ import annotations

import time
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app.agents.answer_agent import AnswerAgent
from app.agents.cypher_agent import CypherAgent
from app.agents.fusion_agent import FusionAgent
from app.agents.nutrition_agent import NutritionAgent
from app.agents.preference_agent import PreferenceAgent
from app.agents.rerank_agent import RerankAgent
from app.agents.recipe_agent import RecipeAgent
from app.agents.router_agent import RouterAgent
from app.agents.sql_agent import SQLAgent
from app.agents.support_agents import GeneralAgent, SafetyAgent
from app.agents.tool_agent import ToolAgent
from app.agents.vision_agent import VisionAgent
from app.retriever import Recipe
from app.services.memory import MemoryStore
from app.services.logger import get_logger
from app.state import AgentState
from app.tools.registry import ToolRegistry


logger = get_logger("graph")


class GraphState(TypedDict, total=False):
    user_input: str
    session_id: str
    top_k: int
    chat_history: str
    intent: str
    target_agent: str
    retrieved_docs: list[tuple[Recipe, float]]
    fusion_results: list[dict[str, Any]]
    vision_result: dict[str, Any]
    agent_output: str
    final_answer: str
    generator: str
    meta: dict[str, Any]


class SmartRecipeGraph:
    def __init__(
        self,
        router_agent: RouterAgent,
        safety_agent: SafetyAgent,
        preference_agent: PreferenceAgent,
        recipe_agent: RecipeAgent,
        nutrition_agent: NutritionAgent,
        general_agent: GeneralAgent,
        sql_agent: SQLAgent,
        cypher_agent: CypherAgent,
        fusion_agent: FusionAgent,
        vision_agent: VisionAgent | None,
        rerank_agent: RerankAgent,
        answer_agent: AnswerAgent,
        memory_store: MemoryStore,
        tool_agent: ToolAgent | None = None,
    ) -> None:
        self.router_agent = router_agent
        self.safety_agent = safety_agent
        self.preference_agent = preference_agent
        self.recipe_agent = recipe_agent
        self.nutrition_agent = nutrition_agent
        self.general_agent = general_agent
        self.sql_agent = sql_agent
        self.cypher_agent = cypher_agent
        self.fusion_agent = fusion_agent
        self.tool_agent = tool_agent or ToolAgent(ToolRegistry())
        self.vision_agent = vision_agent
        self.rerank_agent = rerank_agent
        self.answer_agent = answer_agent
        self.memory_store = memory_store
        self.app = self._build_graph()

    def run(self, message: str, session_id: str, top_k: int) -> AgentState:
        initial_state: GraphState = {
            "user_input": message,
            "session_id": session_id,
            "top_k": top_k,
            "chat_history": self.memory_store.format_history(session_id),
            "meta": {},
        }
        result = self.app.invoke(initial_state)
        final_state = self._to_agent_state(result)
        self.memory_store.add_turn(session_id, message, final_state.final_answer)
        final_state.meta.update(
            {
                "workflow": "langgraph_final_version",
                "router": "rule",
                "memory_messages": len(self.memory_store.get_history(session_id)),
            }
        )
        return final_state

    def run_image(
        self,
        message: str,
        session_id: str,
        top_k: int,
        image_bytes: bytes,
        filename: str,
    ) -> AgentState:
        initial_state: GraphState = {
            "user_input": message,
            "session_id": session_id,
            "top_k": top_k,
            "chat_history": self.memory_store.format_history(session_id),
            "intent": "image_recipe_query",
            "target_agent": "vision_agent",
            "meta": {
                "image_bytes": image_bytes,
                "image_filename": filename,
                "input_modality": "image",
            },
        }
        result = self.app.invoke(initial_state)
        final_state = self._to_agent_state(result)
        self.memory_store.add_turn(session_id, f"[image:{filename}] {message}", final_state.final_answer)
        final_state.meta.update(
            {
                "workflow": "langgraph_multimodal",
                "memory_messages": len(self.memory_store.get_history(session_id)),
            }
        )
        return final_state

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("safety", self._safety_node)
        graph.add_node("preference_agent", self._preference_node)
        graph.add_node("router", self._router_node)
        graph.add_node("recipe_agent", self._recipe_node)
        graph.add_node("nutrition_agent", self._nutrition_node)
        graph.add_node("general_agent", self._general_node)
        graph.add_node("sql_agent", self._sql_node)
        graph.add_node("cypher_agent", self._cypher_node)
        graph.add_node("fusion_agent", self._fusion_node)
        graph.add_node("tool_agent", self._tool_node)
        if self.vision_agent is not None:
            graph.add_node("vision_agent", self._vision_node)
        graph.add_node("rerank_agent", self._rerank_node)
        graph.add_node("answer_agent", self._answer_node)

        graph.set_entry_point("safety")
        graph.add_conditional_edges(
            "safety",
            self._route_after_safety,
            {
                "preference_agent": "preference_agent",
                "answer_agent": "answer_agent",
            },
        )
        graph.add_conditional_edges(
            "preference_agent",
            self._route_after_preference,
            {
                "router": "router",
                "vision_agent": "vision_agent",
            },
        )
        graph.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "recipe_agent": "recipe_agent",
                "nutrition_agent": "nutrition_agent",
                "general_agent": "general_agent",
                "sql_agent": "sql_agent",
                "cypher_agent": "cypher_agent",
                "fusion_agent": "fusion_agent",
                "tool_agent": "tool_agent",
                "vision_agent": "vision_agent",
            },
        )
        graph.add_edge("recipe_agent", "rerank_agent")
        graph.add_edge("nutrition_agent", "rerank_agent")
        if self.vision_agent is not None:
            graph.add_edge("vision_agent", "rerank_agent")
        graph.add_edge("sql_agent", "answer_agent")
        graph.add_edge("cypher_agent", "answer_agent")
        graph.add_edge("fusion_agent", "answer_agent")
        graph.add_edge("tool_agent", "answer_agent")
        graph.add_edge("rerank_agent", "answer_agent")
        graph.add_edge("general_agent", "answer_agent")
        graph.add_edge("answer_agent", END)
        return graph.compile()

    def _safety_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("safety_agent", self.safety_agent, state)

    def _preference_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("preference_agent", self.preference_agent, state)

    def _router_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("router_agent", self.router_agent, state)

    def _recipe_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("recipe_agent", self.recipe_agent, state)

    def _nutrition_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("nutrition_agent", self.nutrition_agent, state)

    def _general_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("general_agent", self.general_agent, state)

    def _sql_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("sql_agent", self.sql_agent, state)

    def _cypher_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("cypher_agent", self.cypher_agent, state)

    def _fusion_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("fusion_agent", self.fusion_agent, state)

    def _tool_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("tool_agent", self.tool_agent, state)

    def _vision_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("vision_agent", self.vision_agent, state)

    def _rerank_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("rerank_agent", self.rerank_agent, state)

    def _answer_node(self, state: GraphState) -> GraphState:
        return self._run_agent_node("answer_agent", self.answer_agent, state)

    def _run_agent_node(self, name: str, agent, state: GraphState) -> GraphState:
        start = time.perf_counter()
        agent_state = self._to_agent_state(state)
        result_state = agent_state
        try:
            result = agent.run(agent_state)
            result_state = result
            return self._from_agent_state(result)
        except Exception:
            logger.exception("Agent 执行失败 名称=%s", name)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Agent完成 名称=%s 耗时=%.2fms 意图=%s 目标=%s 输出=%s",
                name,
                elapsed_ms,
                result_state.intent,
                result_state.target_agent,
                summarize_agent_output(name, result_state),
            )

    @staticmethod
    def _route_after_safety(state: GraphState) -> Literal["preference_agent", "answer_agent"]:
        if state.get("meta", {}).get("safety_status") == "blocked":
            return "answer_agent"
        return "preference_agent"

    @staticmethod
    def _route_after_preference(state: GraphState) -> Literal["router", "vision_agent"]:
        if state.get("meta", {}).get("input_modality") == "image":
            return "vision_agent"
        return "router"

    @staticmethod
    def _route_after_router(
        state: GraphState,
    ) -> Literal["recipe_agent", "nutrition_agent", "general_agent", "sql_agent", "cypher_agent", "fusion_agent", "tool_agent", "vision_agent"]:
        target = state.get("target_agent", "general_agent")
        if target in {"recipe_agent", "nutrition_agent", "sql_agent", "cypher_agent", "fusion_agent", "tool_agent", "vision_agent"}:
            return target
        return "general_agent"

    @staticmethod
    def _to_agent_state(state: GraphState) -> AgentState:
        return AgentState(
            user_input=state.get("user_input", ""),
            session_id=state.get("session_id", "default"),
            top_k=state.get("top_k", 3),
            chat_history=state.get("chat_history", ""),
            intent=state.get("intent", "out_of_scope"),
            target_agent=state.get("target_agent", "general_agent"),
            retrieved_docs=state.get("retrieved_docs", []),
            fusion_results=state.get("fusion_results", []),
            vision_result=state.get("vision_result", {}),
            agent_output=state.get("agent_output", ""),
            final_answer=state.get("final_answer", ""),
            generator=state.get("generator", "template"),
            meta=state.get("meta", {}),
        )

    @staticmethod
    def _from_agent_state(state: AgentState) -> GraphState:
        return {
            "user_input": state.user_input,
            "session_id": state.session_id,
            "top_k": state.top_k,
            "chat_history": state.chat_history,
            "intent": state.intent,
            "target_agent": state.target_agent,
            "retrieved_docs": state.retrieved_docs,
            "fusion_results": state.fusion_results,
            "vision_result": state.vision_result,
            "agent_output": state.agent_output,
            "final_answer": state.final_answer,
            "generator": state.generator,
            "meta": state.meta,
        }


def summarize_agent_output(name: str, state: AgentState, limit: int = 500) -> str:
    """Return a compact Chinese log summary for each LangGraph node."""
    if name == "answer_agent" and state.final_answer:
        return compact_text(state.final_answer, limit)
    if state.agent_output:
        return compact_text(state.agent_output, limit)
    if name == "router_agent":
        return f"路由到 {state.target_agent}，意图={state.intent}，模式={state.meta.get('router_mode', '未知')}"
    if name == "preference_agent":
        preferences = state.meta.get("user_preferences")
        if preferences:
            return compact_text(f"用户偏好={preferences}", limit)
        return "未提取到新的用户偏好"
    if name == "safety_agent":
        return f"安全状态={state.meta.get('safety_status', '通过')}"
    if name == "rerank_agent":
        names = [recipe.name for recipe, _score in state.retrieved_docs[:5]]
        return "重排候选=" + ("、".join(names) if names else "无")
    if name == "vision_agent" and state.vision_result:
        return compact_text(f"视觉识别={state.vision_result}", limit)
    return "无直接文本输出"


def compact_text(value, limit: int = 500) -> str:
    text = " ".join(str(value).split())
    if not text:
        return "空"
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."
