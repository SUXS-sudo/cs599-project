"""SmartRecipe 连通性检查工具。

子命令:
  mysql     检查 MySQL 表行数
  redis     检查 Redis 记忆读写
  neo4j     检查 Neo4j 图谱统计和样例查询
  llm       检查文本模型和视觉模型连通性
  schema    初始化或打印 MySQL schema
"""
from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ── mysql ──────────────────────────────────────────────────────────────────

def cmd_mysql(_args: argparse.Namespace) -> int:
    from src.services.mysql_store import MySQLConfig, MySQLStore

    config = MySQLConfig.from_env()
    store = MySQLStore(config)
    try:
        stats = store.stats()
    except Exception as exc:
        print(f"mysql_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_target={config.host}:{config.port}/{config.database}")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


# ── redis ──────────────────────────────────────────────────────────────────

def cmd_redis(args: argparse.Namespace) -> int:
    from src.services.redis_memory import RedisMemoryStore

    try:
        store = RedisMemoryStore(max_messages=10)
        store.add_turn(args.session_id, "我不吃牛肉，对虾过敏，喜欢清淡。", "已记录你的偏好。")
        prefs = store.update_preferences(
            args.session_id,
            preferences=["清淡"],
            allergies=["虾"],
            dislikes=["牛肉"],
        )
        history = store.get_history(args.session_id)
    except Exception as exc:
        print(f"redis_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"redis_backend={store.backend}")
    print(f"redis_url={store.redis_url}")
    print(f"session_id={args.session_id}")
    print(f"history_messages={len(history)}")
    print(f"preferences={','.join(prefs.preferences)}")
    print(f"allergies={','.join(prefs.allergies)}")
    print(f"dislikes={','.join(prefs.dislikes)}")
    return 0


# ── neo4j ──────────────────────────────────────────────────────────────────

def cmd_neo4j(args: argparse.Namespace) -> int:
    from src.services.neo4j_store import Neo4jConfig, Neo4jStore

    config = Neo4jConfig.from_env()
    store = Neo4jStore(config)
    try:
        stats = store.stats()
        ingredient_rows = store.execute_read(
            """
            MATCH (recipe:Recipe)-[:USES]->(:Ingredient {name: $ingredient})
            RETURN recipe.name AS name, recipe.calories_per_100g AS calories
            ORDER BY recipe.calories_per_100g ASC
            LIMIT 5
            """,
            {"ingredient": args.ingredient},
        )
        goal_rows = store.execute_read(
            """
            MATCH (recipe:Recipe)-[:SUITABLE_FOR]->(:Goal {name: $goal})
            RETURN recipe.name AS name, recipe.calories_per_100g AS calories
            ORDER BY recipe.calories_per_100g ASC
            LIMIT 5
            """,
            {"goal": args.goal},
        )
    except Exception as exc:
        print(f"neo4j_check_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"neo4j_target={config.uri}")
    if config.database:
        print(f"neo4j_database={config.database}")
    print("graph_stats:")
    for key, value in stats.items():
        print(f"- {key}={value}")
    print(f"sample_ingredient={args.ingredient}")
    for row in ingredient_rows:
        print(f"- {row['name']} | calories_per_100g={row['calories']}")
    print(f"sample_goal={args.goal}")
    for row in goal_rows:
        print(f"- {row['name']} | calories_per_100g={row['calories']}")
    return 0


# ── llm ────────────────────────────────────────────────────────────────────

DEFAULT_IMAGE = ROOT_DIR / "data" / "images" / "地三鲜.png"
TEXT_PROMPT = "这是 SmartRecipe 文本模型连通性检查。请只回复：正常"
VISION_PROMPT = (
    "这是 SmartRecipe 视觉模型连通性检查。"
    "请用中文简短识别图片里的菜品，并列出两三个可能食材。"
)


def cmd_llm(args: argparse.Namespace) -> int:
    from src.services.llm_client import LLMClient, load_dotenv
    from src.services.hyde import HyDEGenerator

    load_dotenv(override=True)
    logging.getLogger("smart_recipe.services.llm").setLevel(logging.ERROR)

    if args.vision_only:
        client = LLMClient(env_prefix="VISION")
    else:
        client = LLMClient()
    _print_llm_config(client)

    if not client.available:
        print("错误：LLM 配置不完整，请检查 .env 里的 BASE_URL、API_KEY 和 MODEL。")
        return 1

    text_ok = True
    vision_ok = True

    if args.hyde_only:
        text_ok = _check_hyde(client, args.timeout)
    elif not args.vision_only:
        text_ok = _check_text(client, args.timeout)

    if not args.text_only and not args.hyde_only:
        vision_client = client if args.vision_only else LLMClient(env_prefix="VISION")
        if not args.vision_only:
            _print_llm_config(vision_client, title="SmartRecipe 视觉模型配置：")
        vision_ok = _check_vision(vision_client, args.image, args.timeout)

    if text_ok and vision_ok:
        print("通过：选中的 LLM 检查全部成功。")
        return 0
    if not text_ok and not vision_ok:
        return 4
    if not text_ok:
        return 2
    return 3


def _print_llm_config(client, title: str = "SmartRecipe LLM 配置：") -> None:
    enable_vision = os.getenv("ENABLE_VISION_LLM", "false").strip().lower()
    print(title)
    print(f"- provider={client.provider}")
    print(f"- base_url={client.base_url}")
    print(f"- model={client.model}")
    print(f"- vision_model={client.vision_model}")
    print(f"- enable_vision_llm={enable_vision}")


def _check_text(client, timeout: int) -> bool:
    print("\n[1/2] 正在检查文本模型...")
    text = client.generate(TEXT_PROMPT, max_tokens=120, timeout=timeout)
    if not text:
        print(f"错误：文本模型{_failure_summary(client)}。model={client.model}")
        return False
    print(f"通过：文本模型已返回。model={client.model}, response={_preview(text)}")
    return True


def _check_hyde(client, timeout: int) -> bool:
    from src.services.hyde import HyDEGenerator

    print("\n[HyDE] 正在检查主文本模型的 HyDE 生成...")
    _ = timeout
    result = HyDEGenerator(client).generate("低脂晚餐推荐，少油少盐，高蛋白")
    if not result.enabled or not result.hypothetical_document:
        print(f"错误：HyDE {_failure_summary(client)}。error={result.error or 'disabled_or_empty'}, model={client.model}")
        return False
    print(f"通过：HyDE 已生成假设检索文本。model={client.model}")
    print(f"hyde={_preview(result.hypothetical_document, limit=600)}")
    return True


def _check_vision(client, image_path: Path, timeout: int) -> bool:
    print("\n[2/2] 正在检查视觉模型...")
    if os.getenv("ENABLE_VISION_LLM", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        print("错误：ENABLE_VISION_LLM 未开启，应用不会调用 VISION_MODEL。")
        return False

    resolved = image_path if image_path.is_absolute() else ROOT_DIR / image_path
    if not resolved.exists():
        print(f"错误：视觉测试图片不存在：{resolved}")
        return False

    image_bytes = resolved.read_bytes()
    mime_type = mimetypes.guess_type(str(resolved))[0] or "image/png"
    text = client.generate_with_image(
        VISION_PROMPT,
        image_bytes=image_bytes,
        mime_type=mime_type,
        max_tokens=500,
        timeout=timeout,
        model=client.vision_model,
    )
    if not text:
        print(f"错误：视觉模型{_failure_summary(client)}。vision_model={client.vision_model}, image={resolved}")
        return False
    print(f"通过：视觉模型已返回。vision_model={client.vision_model}, image={resolved}")
    print(f"response={_preview(text, limit=600)}")
    return True


def _preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _failure_summary(client) -> str:
    kind = getattr(client, "last_failure_kind", "") or "unknown_failure"
    detail = getattr(client, "last_failure_detail", "") or "no detail"
    labels = {
        "config_incomplete": "配置不完整",
        "call_exception": "调用失败",
        "response_parse_failed": "返回格式无法解析",
        "empty_response": "返回为空",
        "unknown_failure": "测试失败",
    }
    return f"{labels.get(kind, kind)}. failure={kind}, detail={detail}"


# ── schema ─────────────────────────────────────────────────────────────────

def cmd_schema(args: argparse.Namespace) -> int:
    from src.services.mysql_store import MySQLConfig, MySQLStore, schema_sql

    config = MySQLConfig.from_env()
    if args.print_sql:
        print(schema_sql(config.database))
        return 0

    store = MySQLStore(config)
    try:
        store.ensure_schema()
    except Exception as exc:
        print(f"mysql_schema_init_failed={type(exc).__name__}: {exc}")
        return 1

    print(f"mysql_schema_ready={config.host}:{config.port}/{config.database}")
    return 0


# ── main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="SmartRecipe 连通性检查工具。")
    sub = parser.add_subparsers(dest="command", required=True)

    p_mysql = sub.add_parser("mysql", help="检查 MySQL 表行数。")
    p_mysql.set_defaults(func=cmd_mysql)

    p_redis = sub.add_parser("redis", help="检查 Redis 记忆读写。")
    p_redis.add_argument("--session-id", default="redis-smoke", help="Smoke test 使用的 session id。")
    p_redis.set_defaults(func=cmd_redis)

    p_neo4j = sub.add_parser("neo4j", help="检查 Neo4j 图谱统计和样例查询。")
    p_neo4j.add_argument("--ingredient", default="鸡胸肉", help="食材样例查询。")
    p_neo4j.add_argument("--goal", default="减脂", help="饮食目标样例查询。")
    p_neo4j.set_defaults(func=cmd_neo4j)

    p_llm = sub.add_parser("llm", help="检查文本模型和视觉模型连通性。")
    llm_mode = p_llm.add_mutually_exclusive_group()
    llm_mode.add_argument("--text-only", action="store_true", help="只测试 MODEL 文本生成。")
    llm_mode.add_argument("--vision-only", action="store_true", help="只测试 VISION_MODEL 图片理解。")
    llm_mode.add_argument("--hyde-only", action="store_true", help="只测试主文本 MODEL 的 HyDE 生成。")
    p_llm.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="VISION_MODEL 测试使用的图片路径。")
    p_llm.add_argument("--timeout", type=int, default=45, help="请求超时时间，单位秒。")
    p_llm.set_defaults(func=cmd_llm)

    p_schema = sub.add_parser("schema", help="初始化或打印 MySQL schema。")
    p_schema.add_argument("--print-sql", action="store_true", help="只打印 SQL，不连接数据库。")
    p_schema.set_defaults(func=cmd_schema)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
