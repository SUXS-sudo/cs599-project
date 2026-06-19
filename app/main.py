import os
import time
from pathlib import Path

from fastapi import File, Form, Request, UploadFile
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.answer_agent import AnswerAgent
from app.agents.cypher_agent import CypherAgent
from app.agents.fusion_agent import FusionAgent
from app.agents.nutrition_agent import NutritionAgent
from app.agents.preference_agent import PreferenceAgent
from app.agents.rerank_agent import RerankAgent
from app.agents.recipe_agent import RecipeAgent
from app.agents.router_agent import RouterAgent
from app.agents.sql_agent import SQLAgent
from app.agents.support_agents import DataAgent, GeneralAgent, SafetyAgent
from app.agents.tool_agent import ToolAgent
from app.agents.vision_agent import VisionAgent
from app.graph import SmartRecipeGraph
from app.models import ChatRequest, ChatResponse, RecipeHit
from app.retriever import RecipeRetriever
from app.services.ablation import ablation_options, env_bool, load_ablation_config, metric_definitions
from app.services.database_browser import DatabaseBrowser
from app.services.llm_client import LLMClient, load_dotenv
from app.services.graph_rag import GraphRAG
from app.services.image_analyzer import ImageAnalyzer
from app.services.logger import configure_logging, get_logger
from app.services.redis_memory import build_memory_store
from app.tools.registry import build_default_tool_registry


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "recipes.json"
STATIC_DIR = BASE_DIR / "app" / "static"
configure_logging()
logger = get_logger("main")
load_dotenv()
ENABLE_DATABASE_AGENTS = env_bool("ENABLE_DATABASE_AGENTS", "ENABLE_V2", default=False)
ABLATION_CONFIG = load_ablation_config()

app = FastAPI(
    title="SmartRecipe Multi-Agent API",
    description="SmartRecipe final-version hybrid Multi-Agent RAG API.",
    version="2.0.0",
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    logger.info("请求开始 方法=%s 路径=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("请求失败 方法=%s 路径=%s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "请求完成 方法=%s 路径=%s 状态码=%s 耗时=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response

llm_client = LLMClient()
vision_client = LLMClient(env_prefix="VISION")
retriever = RecipeRetriever(DATA_PATH, llm_client=llm_client)
memory_store = build_memory_store(max_messages=10)
sql_agent = SQLAgent(llm_client=llm_client)
cypher_agent = CypherAgent(llm_client=llm_client)
tool_registry = build_default_tool_registry(retriever, memory_store, sql_agent=sql_agent, cypher_agent=cypher_agent)
workflow = SmartRecipeGraph(
    router_agent=RouterAgent(llm_client, enable_database_agents=ENABLE_DATABASE_AGENTS, enable_fusion=ABLATION_CONFIG.enable_fusion),
    safety_agent=SafetyAgent(),
    preference_agent=PreferenceAgent(memory_store),
    recipe_agent=RecipeAgent(
        retriever,
        GraphRAG(cypher_agent.store) if ABLATION_CONFIG.enable_graph_rag else None,
        mysql_store=sql_agent.store,
    ),
    nutrition_agent=NutritionAgent(retriever),
    general_agent=GeneralAgent(),
    sql_agent=sql_agent,
    cypher_agent=cypher_agent,
    fusion_agent=FusionAgent(retriever=retriever, mysql_store=sql_agent.store, neo4j_store=cypher_agent.store),
    tool_agent=ToolAgent(tool_registry, llm_client),
    vision_agent=VisionAgent(ImageAnalyzer(vision_client), retriever),
    rerank_agent=RerankAgent(),
    answer_agent=AnswerAgent(llm_client),
    memory_store=memory_store,
)
data_agent = DataAgent()
database_browser = DatabaseBrowser(mysql_store=sql_agent.store, neo4j_store=cypher_agent.store)
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "final-version-multi-agent",
        "retriever": retriever.backend,
        "embedding": retriever.embedding_backend,
        "rerank": "enabled",
        "memory": getattr(memory_store, "backend", "memory"),
        "enable_database_agents": str(ENABLE_DATABASE_AGENTS).lower(),
    }


@app.get("/debug/session/{session_id}")
def debug_session(session_id: str) -> dict:
    data = memory_store.debug_session(session_id)
    data["mysql_preferences"] = safe_debug_call(lambda: workflow.sql_agent.store.get_user_preferences(session_id))
    return data


@app.get("/debug/stats")
def debug_stats() -> dict:
    return {
        "mysql": safe_debug_call(workflow.sql_agent.store.stats),
        "neo4j": safe_debug_call(workflow.cypher_agent.store.stats),
        "redis": {
            "ok": True,
            "backend": getattr(memory_store, "backend", "memory"),
            "active_sessions": memory_store.active_session_count(),
        },
        "retriever": {
            "ok": True,
            **retriever.status(),
        },
    }


@app.get("/debug/evaluation")
def debug_evaluation() -> dict:
    eval_dir = BASE_DIR / "data" / "evals"
    eval_files = {}
    for path in sorted(eval_dir.glob("*.jsonl")):
        try:
            rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            eval_files[path.name] = len(rows)
        except Exception as exc:
            eval_files[path.name] = f"error:{type(exc).__name__}"
    return {
        "ok": True,
        "recipe_count": len(retriever.recipes),
        "eval_files": eval_files,
        "capabilities": {
            "text_chat": True,
            "image_chat": True,
            "graph_rag": True,
            "text2sql": True,
            "text2cypher": True,
            "fusion": True,
            "data_agent": True,
            "memory_summary": True,
            "answer_guard": True,
            "redis_memory": getattr(memory_store, "backend", "memory") == "redis",
            "enable_database_agents": ENABLE_DATABASE_AGENTS,
            "vision_llm": os.getenv("ENABLE_VISION_LLM", "false").strip().lower() in {"1", "true", "yes", "on"},
            "graph_rag_enabled": ABLATION_CONFIG.enable_graph_rag,
            "fusion_enabled": ABLATION_CONFIG.enable_fusion,
            "answer_guard_enabled": ABLATION_CONFIG.enable_answer_guard,
            "memory_summary_enabled": ABLATION_CONFIG.enable_memory_summary,
        },
    }


@app.get("/debug/ablation")
def debug_ablation() -> dict:
    return {
        "ok": True,
        "active_config": load_ablation_config().to_dict(),
        "options": ablation_options(),
        "metrics": metric_definitions(),
    }


@app.get("/debug/database/overview")
def debug_database_overview() -> dict:
    return database_browser.overview()


@app.get("/debug/database/mysql/{table}")
def debug_database_mysql_table(table: str, limit: int = 50) -> dict:
    return safe_debug_call(lambda: database_browser.mysql_table(table, limit=limit))


@app.get("/debug/database/redis/key")
def debug_database_redis_key(key: str, limit: int = 50) -> dict:
    return safe_debug_call(lambda: database_browser.redis_key(key, limit=limit))


@app.get("/debug/database/neo4j/{label}")
def debug_database_neo4j_nodes(label: str, limit: int = 50) -> dict:
    return safe_debug_call(lambda: database_browser.neo4j_nodes(label, limit=limit))


@app.get("/debug/pipeline")
def debug_pipeline() -> dict:
    source_path = BASE_DIR / "data" / "recipes.json"
    output_path = BASE_DIR / "data" / "processed" / "recipes_clean.json"
    return {
        "ok": True,
        "agent": "data_agent",
        "report": data_agent.run_pipeline(source_path, output_path),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    session_id = payload.normalized_session_id()
    state = workflow.run(payload.message, session_id, payload.top_k)
    return ChatResponse(
        answer=state.final_answer,
        intent=state.intent,
        agent=state.target_agent,
        session_id=session_id,
        recipes=[
            RecipeHit(
                name=recipe.name,
                score=round(score, 4),
                ingredients=recipe.ingredients,
                category=recipe.category,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                calories=recipe.calories,
            )
            for recipe, score in state.retrieved_docs
        ],
        meta={
            **state.meta,
            "generator": state.generator,
            "recipe_count": len(state.retrieved_docs),
        },
    )


@app.post("/chat/image", response_model=ChatResponse)
async def chat_image(
    image: UploadFile = File(...),
    message: str = Form("这是什么菜？推荐类似做法。"),
    session_id: str | None = Form(default=None),
    top_k: int = Form(default=3),
) -> ChatResponse:
    normalized_session_id = session_id or ChatRequest(message=message, top_k=top_k).normalized_session_id()
    content = await image.read()
    state = workflow.run_image(message, normalized_session_id, min(max(top_k, 1), 5), content, image.filename or "uploaded-image")
    return build_chat_response(state, normalized_session_id)


def build_chat_response(state, session_id: str) -> ChatResponse:
    return ChatResponse(
        answer=state.final_answer,
        intent=state.intent,
        agent=state.target_agent,
        session_id=session_id,
        recipes=[
            RecipeHit(
                name=recipe.name,
                score=round(score, 4),
                ingredients=recipe.ingredients,
                category=recipe.category,
                cooking_time=recipe.cooking_time,
                difficulty=recipe.difficulty,
                tags=recipe.tags,
                calories=recipe.calories,
            )
            for recipe, score in state.retrieved_docs
        ],
        meta={
            **state.meta,
            "vision_result": state.vision_result,
            "generator": state.generator,
            "recipe_count": len(state.retrieved_docs),
        },
    )


def safe_debug_call(func) -> dict:
    try:
        return {"ok": True, "data": func()}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
