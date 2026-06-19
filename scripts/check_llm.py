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

from app.services.llm_client import LLMClient, load_dotenv
from app.services.hyde import HyDEGenerator


DEFAULT_IMAGE = ROOT_DIR / "data" / "images" / "地三鲜.png"
TEXT_PROMPT = "这是 SmartRecipe 文本模型连通性检查。请只回复：正常"
VISION_PROMPT = (
    "这是 SmartRecipe 视觉模型连通性检查。"
    "请用中文简短识别图片里的菜品，并列出两三个可能食材。"
)


def main() -> int:
    args = parse_args()
    load_dotenv(override=True)
    quiet_llm_logs()

    if args.vision_only:
        client = LLMClient(env_prefix="VISION")
    else:
        client = LLMClient()
    print_config(client)

    if not client.available:
        print("错误：LLM 配置不完整，请检查 .env 里的 BASE_URL、API_KEY 和 MODEL。")
        return 1

    text_ok = True
    vision_ok = True

    if args.hyde_only:
        text_ok = check_hyde_model(client, args.timeout)
    elif not args.vision_only:
        text_ok = check_text_model(client, args.timeout)

    if not args.text_only and not args.hyde_only:
        vision_client = client if args.vision_only else LLMClient(env_prefix="VISION")
        if not args.vision_only:
            print_config(vision_client, title="SmartRecipe 视觉模型配置：")
        vision_ok = check_vision_model(vision_client, args.image, args.timeout)

    if text_ok and vision_ok:
        print("通过：选中的 LLM 检查全部成功。")
        return 0
    if not text_ok and not vision_ok:
        return 4
    if not text_ok:
        return 2
    return 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 SmartRecipe 文本模型和视觉模型连通性。")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text-only", action="store_true", help="只测试 MODEL 文本生成。")
    mode.add_argument("--vision-only", action="store_true", help="只测试 VISION_MODEL 图片理解。")
    mode.add_argument("--hyde-only", action="store_true", help="只测试主文本 MODEL 的 HyDE 生成。")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="VISION_MODEL 测试使用的图片路径。",
    )
    parser.add_argument("--timeout", type=int, default=45, help="请求超时时间，单位秒。")
    return parser.parse_args()


def print_config(client: LLMClient, title: str = "SmartRecipe LLM 配置：") -> None:
    enable_vision = os.getenv("ENABLE_VISION_LLM", "false").strip().lower()
    print(title)
    print(f"- provider={client.provider}")
    print(f"- base_url={client.base_url}")
    print(f"- model={client.model}")
    print(f"- vision_model={client.vision_model}")
    print(f"- enable_vision_llm={enable_vision}")


def quiet_llm_logs() -> None:
    logging.getLogger("smart_recipe.services.llm").setLevel(logging.ERROR)


def check_text_model(client: LLMClient, timeout: int) -> bool:
    print("\n[1/2] 正在检查文本模型...")
    text = client.generate(TEXT_PROMPT, max_tokens=120, timeout=timeout)
    if not text:
        print(f"错误：文本模型{failure_summary(client)}。model={client.model}")
        return False
    print(f"通过：文本模型已返回。model={client.model}, response={preview(text)}")
    return True


def check_hyde_model(client: LLMClient, timeout: int) -> bool:
    print("\n[HyDE] 正在检查主文本模型的 HyDE 生成...")
    _ = timeout
    result = HyDEGenerator(client).generate("低脂晚餐推荐，少油少盐，高蛋白")
    if not result.enabled or not result.hypothetical_document:
        print(f"错误：HyDE {failure_summary(client)}。error={result.error or 'disabled_or_empty'}, model={client.model}")
        return False
    print(f"通过：HyDE 已生成假设检索文本。model={client.model}")
    print(f"hyde={preview(result.hypothetical_document, limit=600)}")
    return True


def check_vision_model(client: LLMClient, image_path: Path, timeout: int) -> bool:
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
        print(f"错误：视觉模型{failure_summary(client)}。vision_model={client.vision_model}, image={resolved}")
        return False
    print(f"通过：视觉模型已返回。vision_model={client.vision_model}, image={resolved}")
    print(f"response={preview(text, limit=600)}")
    return True


def preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def failure_summary(client: LLMClient) -> str:
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


if __name__ == "__main__":
    sys.exit(main())
