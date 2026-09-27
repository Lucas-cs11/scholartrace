"""LLM 客户端单元测试（mock httpx，不产生真实调用）。"""
import json

import httpx
import pytest

from config.settings import Settings
from src.llm import LLMClient, LLMError, _chat_url


def _cfg(base_url: str = "https://api.deepseek.com", api_key: str = "sk-test") -> Settings:
    return Settings(openai_base_url=base_url, openai_api_key=api_key, llm_model="deepseek-chat")


def _ok_response(content: str, usage: dict | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}], "usage": usage or {}},
    )


# ------------------------------------------------------------------ URL 规整
def test_chat_url_normalization():
    assert _chat_url("https://api.deepseek.com") == "https://api.deepseek.com/v1/chat/completions"
    assert _chat_url("https://api.deepseek.com/") == "https://api.deepseek.com/v1/chat/completions"
    assert _chat_url("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1/chat/completions"
    assert _chat_url("https://x.com/v1/chat/completions") == "https://x.com/v1/chat/completions"


# ------------------------------------------------------------------ 成功路径
@pytest.mark.asyncio
async def test_complete_success(monkeypatch):
    async def fake_post(self, payload):
        assert payload["model"] == "deepseek-chat"
        assert payload["messages"][0]["role"] == "user"
        return {"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    text = await client.complete([{"role": "user", "content": "hi"}])
    assert text == "OK"
    await client.close()


@pytest.mark.asyncio
async def test_complete_json_success(monkeypatch):
    async def fake_post(self, payload):
        assert payload.get("response_format") == {"type": "json_object"}
        return {"choices": [{"message": {"content": '{"a": 1}'}}], "usage": {}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    data = await client.complete_json([{"role": "user", "content": "make json"}])
    assert data == {"a": 1}
    await client.close()


@pytest.mark.asyncio
async def test_telemetry_records_usage(monkeypatch):
    from src.telemetry import Telemetry

    async def fake_post(self, payload):
        return {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    telemetry = Telemetry()
    client = LLMClient(cfg=_cfg())
    await client.complete([{"role": "user", "content": "q"}], telemetry=telemetry, note="parse")
    assert telemetry.llm_calls == 1
    assert telemetry.input_tokens == 10
    assert telemetry.output_tokens == 3
    assert telemetry.events[0]["kind"] == "llm"
    assert telemetry.events[0]["note"] == "parse"
    await client.close()


# ------------------------------------------------------------------ 错误路径
@pytest.mark.asyncio
async def test_complete_json_bad_content_raises(monkeypatch):
    async def fake_post(self, payload):
        return {"choices": [{"message": {"content": "not json at all"}}], "usage": {}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    with pytest.raises(LLMError):
        await client.complete_json([{"role": "user", "content": "q"}])
    await client.close()


@pytest.mark.asyncio
async def test_complete_json_struct_content_dict_text(monkeypatch):
    """部分代理把 content 包成 {type: text, text: ...} 结构，需规整为字符串。"""
    async def fake_post(self, payload):
        return {
            "choices": [{"message": {"content": {"type": "text", "text": "{\"a\": 1}"}}}],
            "usage": {},
        }

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    data = await client.complete_json([{"role": "user", "content": "q"}])
    assert data == {"a": 1}
    await client.close()


@pytest.mark.asyncio
async def test_complete_json_struct_content_list_blocks(monkeypatch):
    """多模态格式：content 为内容块列表，提取 text 字段拼接。"""
    async def fake_post(self, payload):
        return {
            "choices": [{"message": {"content": [{"type": "text", "text": "{\"b\": 2}"}]}}],
            "usage": {},
        }

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    data = await client.complete_json([{"role": "user", "content": "q"}])
    assert data == {"b": 2}
    await client.close()


@pytest.mark.asyncio
async def test_complete_json_unparseable_content_type_raises_llm_error(monkeypatch):
    """content 为无法规整的类型时抛 LLMError（而非 TypeError 冒泡）。"""
    async def fake_post(self, payload):
        return {"choices": [{"message": {"content": {"image": "base64..."}}}], "usage": {}}

    monkeypatch.setattr(LLMClient, "_post", fake_post)
    client = LLMClient(cfg=_cfg())
    with pytest.raises(LLMError):
        await client.complete_json([{"role": "user", "content": "q"}])
    await client.close()


@pytest.mark.asyncio
async def test_http_400_raises_llm_error(monkeypatch):
    """400 不可重试：包装成 LLMError 直接抛出。"""
    async def fake_httpx_post(self, url, **kwargs):
        return httpx.Response(400, text="bad request body", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_httpx_post)
    client = LLMClient(cfg=_cfg())
    with pytest.raises(LLMError) as exc:
        await client.complete([{"role": "user", "content": "q"}])
    assert "400" in str(exc.value)
    await client.close()


@pytest.mark.asyncio
async def test_http_500_retried_then_raises(monkeypatch):
    """500 触发 tenacity 重试（3 次），最终仍失败则冒泡原始异常。"""
    calls = {"n": 0}

    async def fake_httpx_post(self, url, **kwargs):
        calls["n"] += 1
        return httpx.Response(500, text="err", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_httpx_post)
    client = LLMClient(cfg=_cfg())
    with pytest.raises(LLMError):
        await client.complete([{"role": "user", "content": "q"}])
    assert calls["n"] == 3  # 1 次原始 + 2 次退避重试
    await client.close()


@pytest.mark.asyncio
async def test_bad_key_403_passthrough_llm_error():
    """真实网络路径：错误 key 时 LLMError 带响应体（不触发重试，403 不可重试）。"""
    cfg = _cfg(api_key="sk-invalid")
    client = LLMClient(cfg=cfg)
    with pytest.raises(LLMError) as exc:
        await client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert "401" in str(exc.value) or "400" in str(exc.value) or "403" in str(exc.value)
    await client.close()


# ------------------------------------------------------------------ LLM Router 分层
def test_tier_strong_uses_llm_model():
    cfg = Settings(llm_model="deepseek-chat", llm_fast_model="deepseek-fast", openai_api_key="k")
    client = LLMClient(cfg=cfg, tier="strong")
    assert client.model == "deepseek-chat"
    assert client.tier == "strong"


def test_tier_fast_uses_fast_model():
    cfg = Settings(llm_model="deepseek-chat", llm_fast_model="deepseek-fast", openai_api_key="k")
    client = LLMClient(cfg=cfg, tier="fast")
    assert client.model == "deepseek-fast"


def test_tier_fast_falls_back_to_llm_model():
    cfg = Settings(llm_model="deepseek-chat", llm_fast_model="", openai_api_key="k")
    client = LLMClient(cfg=cfg, tier="fast")
    assert client.model == "deepseek-chat"


def test_explicit_model_overrides_tier():
    cfg = Settings(llm_model="m1", llm_fast_model="m2", openai_api_key="k")
    client = LLMClient(cfg=cfg, model="custom", tier="fast")
    assert client.model == "custom"


def test_default_tier_is_strong():
    cfg = Settings(llm_model="m1", llm_fast_model="m2", openai_api_key="k")
    client = LLMClient(cfg=cfg)
    assert client.model == "m1"


def test_parser_uses_fast_tier():
    """解析走 fast 模型（低成本步骤）。"""
    from src.parser import QueryIRParser
    p = QueryIRParser(llm=None)
    assert p.llm.tier == "fast"
    assert p.llm.model  # 模型名由全局 settings 决定（llm_fast_model 或兜底 llm_model）


def test_ranker_uses_strong_tier():
    """精排走 strong 模型（复杂步骤）。"""
    from src.ranker import LLMReranker
    r = LLMReranker(llm=None)
    assert r.llm.tier == "strong"
