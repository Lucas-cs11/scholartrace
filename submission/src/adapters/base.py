"""学术搜索 API 适配层抽象基类。

契约：
- search(): 关键词检索，返回候选论文（带证据）。
- get_by_doi(): 按 DOI 校验论文身份（防止幻觉/错 DOI）。
- 所有调用必须通过 telemetry 记账。
- cache 可选：传入 ResponseCache 后 HTTP 走缓存层（write/replay），
  未传入时行为与基线完全一致（observability 基础设施，不改算法）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.observability.response_cache import ResponseCache, cache_key
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry


async def _http_get(url: str, params: dict | None = None, headers: dict | None = None,
                    timeout: float = 20.0, follow_redirects: bool = False) -> httpx.Response:
    """单次 HTTP GET（无缓存路径的共享实现）。"""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=follow_redirects) as client:
        kwargs: dict = {}
        if params:
            kwargs["params"] = params
        if headers:
            kwargs["headers"] = headers
        return await client.get(url, **kwargs)


def _is_retryable(exc: BaseException) -> bool:
    """瞬时错误重试判定：超时/网络/连接错误 + 限流与 5xx。"""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (409, 429, 500, 502, 503, 504)
    return False


def http_retry(func=None, *, on_exhausted=None):
    """共享 HTTP 重试装饰器：指数退避，最多 8 次，与 OpenAlex 对齐。

    只包裹瞬时错误（超时/网络/429/5xx），不改变成功路径的任何逻辑；
    重试期间每次物理尝试都会被 ResponseCache 计数（physical_attempts 含重试）。

    on_exhausted: 提供后在重试耗尽时调用（返回其返回值），而非 reraise——
    用于「上游持续故障时静默降级」的调用（如 get_by_doi 返回 None），
    保持与基线（非 200 返回 None）一致的结果语义。
    """
    def deco(f):
        kwargs = dict(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(8),
            wait=wait_exponential(multiplier=2, min=5, max=60),
        )
        if on_exhausted is None:
            kwargs["reraise"] = True
        else:
            kwargs["retry_error_callback"] = on_exhausted
        return retry(**kwargs)(f)

    if func is None:
        return deco
    return deco(func)


class AcademicSearchAdapter(ABC):
    name: str = "base"

    def __init__(self, cache: ResponseCache | None = None):
        self._cache = cache

    async def _http(self, endpoint: str, query_or_seed: str, method: str,
                    url: str, params: dict | None, fetch) -> httpx.Response:
        """统一 HTTP 入口：可选 ResponseCache 包装（write/replay），默认直连。"""
        if self._cache is not None:
            ck = cache_key(self.name, endpoint, query_or_seed)
            return await self._cache.get_or_fetch(ck, method, url, params, fetch)
        return await fetch()

    @abstractmethod
    async def search(self, query: str, limit: int, telemetry: Telemetry) -> list[PaperEvidence]:
        """按关键词检索。返回按 API 默认相关度排序的候选论文。"""

    @abstractmethod
    async def get_by_doi(self, doi: str, telemetry: Telemetry) -> PaperIdentity | None:
        """按 DOI 获取论文身份，用于身份校验/去重。"""

    async def get_citations(self, paper_id: str, limit: int, telemetry: Telemetry) -> list[PaperIdentity]:
        """引文扩展（Round 2 用）。默认返回空，子类按需实现。"""
        return []
