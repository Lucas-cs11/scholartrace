"""LLM 客户端（OpenAI 兼容协议：DeepSeek / OpenAI / 各类代理）。

设计原则：
- 只实现 OpenAI 兼容 /v1/chat/completions 一个协议——DeepSeek、OpenAI、多数代理均支持。
- 解析类调用失败应快速降级而非长阻塞：重试次数少、超时短。
- 所有调用通过 telemetry 记账（llm_calls / tokens），成本进入 RunReport。
- 失败抛 LLMError，上层（parser/planner）决定降级策略（如退回纯词法 QueryIR）。
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config.settings import Settings, settings as _settings
from src.telemetry import Telemetry


class LLMError(RuntimeError):
    """LLM 调用失败（网络/限流/协议/解析），上层可捕获后降级。"""


def _is_retryable(exc: BaseException) -> bool:
    """只有可重试错误触发 tenacity：超时/网络/429/5xx。"""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def _chat_url(base_url: str) -> str:
    """把用户填的 base_url 规整成 /chat/completions 地址。

    接受三种写法：https://api.deepseek.com / https://api.deepseek.com/v1 / 完整地址。
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class LLMClient:
    """轻量 LLM 客户端：complete() 文本 / complete_json() 结构化输出。

    tier 路由（LLM Router 分层）："fast" 用 llm_fast_model（解析/规划等低成本步骤），
    "strong"（默认）用 llm_model（精排/复杂推理）。两者都在 settings 配置。
    """

    def __init__(self, cfg: Settings | None = None, model: str | None = None, tier: str = "strong"):
        cfg = cfg or _settings
        self.cfg = cfg
        self.api_key = cfg.openai_api_key
        self.base_url = _chat_url(cfg.openai_base_url)
        if model:
            self.model = model
        elif tier == "fast":
            self.model = cfg.llm_fast_model or cfg.llm_model or "deepseek-chat"
        else:
            self.model = cfg.llm_model or "deepseek-chat"
        self.tier = tier
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self.client.aclose()

    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=1.5, max=20),
        reraise=True,  # 重试耗尽后抛最后一次原始异常，由 _call 统一包装
    )
    async def _post(self, payload: dict) -> dict:
        try:
            resp = await self.client.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 可重试的由 tenacity 接管；其余直接抛 LLMError（含响应体便于排查）
            if e.response.status_code not in (429, 500, 502, 503, 504):
                body = e.response.text[:300]
                raise LLMError(f"LLM HTTP {e.response.status_code}: {body}") from e
            raise
        return resp.json()

    async def _call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str | None,
        telemetry: Telemetry | None,
        note: str,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        try:
            data = await self._post(payload)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
            # 可重试错误耗尽后仍失败：统一包装成 LLMError，上层只处理一种失败类型
            raise LLMError(f"LLM 请求失败（重试后仍失败）: {type(e).__name__} {e}") from e
        latency_ms = round((time.time() - t0) * 1000, 1)

        if telemetry:
            usage = data.get("usage") or {}
            telemetry.add_llm(
                model=payload["model"],
                latency_ms=latency_ms,
                in_tokens=usage.get("prompt_tokens", 0),
                out_tokens=usage.get("completion_tokens", 0),
                note=note,
            )

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"LLM 响应缺少 choices/content: {str(data)[:300]}") from e

        # 兼容部分代理/聚合器返回结构化 content（多模态内容块或 {text: ...}），
        # 统一规整为纯文本。不做规整时 json.loads(dict) 会抛 TypeError，
        # 而 TypeError 不被 parser/planner/ranker 的 LLMError 捕获，直接冒泡为 500。
        if isinstance(content, str):
            pass
        elif isinstance(content, dict) and isinstance(content.get("text"), str):
            content = content["text"]
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            content = "\n".join(parts)
        else:
            raise LLMError(f"LLM 响应 content 类型异常: {type(content).__name__}: {str(content)[:200]}")
        return {"content": content, "data": data}

    # ------------------------------------------------------------------
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        telemetry: Telemetry | None = None,
        note: str = "llm",
    ) -> str:
        """普通文本补全。"""
        r = await self._call(messages, temperature, max_tokens, False, None, telemetry, note)
        return r["content"]

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        telemetry: Telemetry | None = None,
        note: str = "llm_json",
    ) -> dict:
        """结构化 JSON 输出（response_format=json_object）。

        解析失败抛 LLMError——调用方决定是否重试一次或降级。
        """
        r = await self._call(messages, temperature, max_tokens, True, None, telemetry, note)
        text = r["content"]
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise LLMError(f"LLM 返回非 JSON: {text[:200]}") from e
        if not isinstance(data, dict):
            raise LLMError(f"LLM JSON 不是对象: {str(data)[:200]}")
        return data
