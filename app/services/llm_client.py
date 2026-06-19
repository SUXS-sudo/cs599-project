from __future__ import annotations

import json
import os
import ssl
import urllib.request
import base64
from pathlib import Path

from app.services.logger import get_logger


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env"
logger = get_logger("services.llm")


def load_dotenv(path: Path = ENV_PATH, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def env_value(prefix: str, name: str, fallback_name: str, default: str) -> str:
    if prefix:
        value = os.getenv(f"{prefix}_{name}")
        if value is not None:
            return value
    return os.getenv(fallback_name, default)


class LLMClient:
    def __init__(self, env_prefix: str = "") -> None:
        load_dotenv()
        self.env_prefix = env_prefix.strip().upper()
        self.provider = env_value(self.env_prefix, "PROVIDER", "SMART_RECIPE_PROVIDER", "anthropic").strip().lower()
        self.base_url = env_value(self.env_prefix, "BASE_URL", "BASE_URL", "").strip()
        self.api_key = env_value(self.env_prefix, "API_KEY", "API_KEY", "").strip()
        self.model = env_value(self.env_prefix, "MODEL", "MODEL", "mimo-v2.5-pro").strip()
        self.vision_model = env_value(self.env_prefix, "VISION_MODEL", "VISION_MODEL", self.model).strip()
        self.last_failure_kind = ""
        self.last_failure_detail = ""

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def generate(self, prompt: str, max_tokens: int = 800, timeout: int = 45) -> str | None:
        self._clear_failure()
        if not self.available:
            self._record_failure("config_incomplete", "BASE_URL, API_KEY or MODEL is empty", model=self.model)
            return None
        logger.info("LLM调用 provider=%s model=%s max_tokens=%s", self.provider, self.model, max_tokens)
        if self.provider == "openai":
            return self._generate_openai(prompt, max_tokens, timeout)
        return self._generate_anthropic(prompt, max_tokens, timeout)

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
        max_tokens: int = 800,
        timeout: int = 45,
        model: str | None = None,
    ) -> str | None:
        self._clear_failure()
        if not self.available:
            self._record_failure("config_incomplete", "BASE_URL, API_KEY or MODEL is empty", model=self.vision_model)
            return None
        selected_model = model or self.vision_model or self.model
        logger.info("视觉LLM调用 provider=%s model=%s max_tokens=%s", self.provider, selected_model, max_tokens)
        if self.provider == "openai":
            return self._generate_openai_with_image(prompt, image_bytes, mime_type, max_tokens, timeout, selected_model)
        return self._generate_anthropic_with_image(prompt, image_bytes, mime_type, max_tokens, timeout, selected_model)

    def _clear_failure(self) -> None:
        self.last_failure_kind = ""
        self.last_failure_detail = ""

    def _record_failure(self, kind: str, detail: str, model: str) -> None:
        self.last_failure_kind = kind
        self.last_failure_detail = detail
        logger.warning(
            "LLM异常 类型=%s provider=%s model=%s 详情=%s",
            kind,
            self.provider,
            model,
            detail,
        )

    def _generate_openai(self, prompt: str, max_tokens: int, timeout: int) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
            if is_chat_completions_model(self.model):
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only the final answer requested by the user."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
                text = extract_openai_chat_text(response)
                if not text:
                    self._record_failure("empty_response", openai_chat_empty_detail(response), model=self.model)
                    return None
                return text
            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_tokens,
            )
            text = response.output_text.strip()
            if not text:
                self._record_failure("empty_response", "OpenAI-compatible response output_text is empty", model=self.model)
                return None
            return text
        except Exception as exc:
            self._record_failure("call_exception", f"{type(exc).__name__}: {exc}", model=self.model)
            return None

    def _generate_openai_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        max_tokens: int,
        timeout: int,
        model: str,
    ) -> str | None:
        try:
            from openai import OpenAI

            image_b64 = base64.b64encode(image_bytes).decode("ascii")
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"},
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
            )
            text = response.output_text.strip()
            if not text:
                self._record_failure("empty_response", "OpenAI-compatible vision response output_text is empty", model=model)
                return None
            return text
        except Exception as exc:
            self._record_failure("call_exception", f"{type(exc).__name__}: {exc}", model=model)
            return None

    def _generate_anthropic(self, prompt: str, max_tokens: int, timeout: int) -> str | None:
        url = anthropic_messages_url(self.base_url)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=build_ssl_context()) as response:
                raw = response.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._record_failure("response_parse_failed", f"{type(exc).__name__}: {exc}", model=self.model)
                return None
            text = extract_anthropic_text(data)
            if not text:
                self._record_failure("empty_response", "Anthropic-compatible response has no text content", model=self.model)
                return None
            return text
        except Exception as exc:
            self._record_failure("call_exception", f"{type(exc).__name__}: {exc}", model=self.model)
            return None

    def _generate_anthropic_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        max_tokens: int,
        timeout: int,
        model: str,
    ) -> str | None:
        url = anthropic_messages_url(self.base_url)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=build_ssl_context()) as response:
                raw = response.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._record_failure("response_parse_failed", f"{type(exc).__name__}: {exc}", model=model)
                return None
            text = extract_anthropic_text(data)
            if not text:
                self._record_failure("empty_response", "Anthropic-compatible vision response has no text content", model=model)
                return None
            return text
        except Exception as exc:
            self._record_failure("call_exception", f"{type(exc).__name__}: {exc}", model=model)
            return None


def anthropic_messages_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1/messages"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/messages"
    return f"{cleaned}/v1/messages"


def is_chat_completions_model(model: str) -> bool:
    lowered = model.strip().lower()
    return lowered.startswith(("deepseek", "ds-", "qwen", "glm", "kimi"))


def extract_openai_chat_text(response) -> str:
    try:
        message = response.choices[0].message
    except Exception:
        return ""
    candidates = [
        getattr(message, "content", None),
        getattr(message, "reasoning_content", None),
    ]
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        candidates.extend(
            [
                extra.get("content"),
                extra.get("reasoning_content"),
                extra.get("answer"),
                extra.get("text"),
            ]
        )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = "\n".join(part.get("text", "") for part in value if isinstance(part, dict))
            if text.strip():
                return text.strip()
    return ""


def openai_chat_empty_detail(response) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        finish_reason = getattr(choice, "finish_reason", "")
        fields = []
        for name in ("content", "reasoning_content", "tool_calls", "function_call"):
            value = getattr(message, name, None)
            if value:
                fields.append(name)
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            fields.extend(f"extra.{key}" for key, value in extra.items() if value)
        return f"OpenAI-compatible chat response has no text. finish_reason={finish_reason or 'unknown'}, message_fields={fields or 'none'}"
    except Exception as exc:
        return f"OpenAI-compatible chat response has no text and could not be inspected: {type(exc).__name__}: {exc}"


def build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def extract_anthropic_text(data: dict) -> str | None:
    text_parts = []
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
        elif isinstance(item, dict) and "text" in item:
            text_parts.append(item.get("text", ""))
        elif isinstance(item, str):
            text_parts.append(item)
    if isinstance(data.get("completion"), str):
        text_parts.append(data["completion"])
    if isinstance(data.get("text"), str):
        text_parts.append(data["text"])
    if isinstance(data.get("response"), str):
        text_parts.append(data["response"])
    text = "\n".join(part for part in text_parts if part).strip()
    return text or None
