"""Semantic Scholar API 适配器（引文/推荐，备用召回）。

免费无 key 限流较严（约 100 req/5min）。建议申请免费 key 提高限流：
https://www.semanticscholar.org/product/api#api-key-form
"""
from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.adapters.base import AcademicSearchAdapter, _http_get, _is_retryable
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry

API_BASE = "https://api.semanticscholar.org/graph/v1"


class S2RateLimited(RuntimeError):
    pass


def _search_once_exhausted(state):
    """_search_once 重试耗尽回调：保持基线异常语义。

    基线无重试——429 直接抛 S2RateLimited（上层 search 的 except 接住后优雅降级），
    其余非 200 返回 []。重试耗尽后同样处理：429 恢复抛 S2RateLimited，
    其它瞬时故障（超时/5xx）静默返回 []（不把上游故障当论文数据）。
    """
    exc = state.outcome.exception()
    if isinstance(exc, S2RateLimited):
        raise exc
    return []


def _graph_exhausted(state):
    """_graph_call 重试耗尽回调：与 _search_once_exhausted 一致。

    429 耗尽后恢复抛 S2RateLimited（上层 get_citations/get_references/get_by_doi
    的 except S2RateLimited 接住后优雅降级返回 []/None），其它瞬时故障静默返回 []
    （对 citations/references 是空集、对 get_by_doi 是 falsy → None，均安全）。
    """
    exc = state.outcome.exception() if state.outcome is not None else None
    if isinstance(exc, S2RateLimited):
        raise exc
    return []


class SemanticScholarAdapter(AcademicSearchAdapter):
    name = "semantic_scholar"

    def __init__(self, api_key: str = "", cache=None):
        super().__init__(cache)
        self._api_key = api_key

    async def search(self, query: str, limit: int = 20, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        results = []
        try:
            data = await self._search_once(query, limit)
        except S2RateLimited:
            if telemetry is not None:
                telemetry.add_fallback("s2_rate_limited")
            return results
        for p in data:
            ev = self._parse_paper(p, source="search")
            if ev is not None:
                results.append(ev)
        return results

    async def get_citations(self, paper_id: str, limit: int = 10, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        """引用该论文的后续论文（citations）。"""
        try:
            data = await self._graph_call(f"/paper/{paper_id}/citations", limit, fields="title,authors,year,venue,externalIds,abstract")
        except S2RateLimited:
            if telemetry is not None:
                telemetry.add_fallback("s2_rate_limited")
            return []
        results = []
        for entry in data or []:
            p = entry.get("citingPaper", {}) if isinstance(entry, dict) else {}
            ev = self._parse_paper(p, source="citation")
            if ev is not None:
                results.append(ev)
        return results

    async def get_references(self, paper_id: str, limit: int = 10, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        """该论文引用的论文（references）。"""
        try:
            data = await self._graph_call(f"/paper/{paper_id}/references", limit, fields="title,authors,year,venue,externalIds,abstract")
        except S2RateLimited:
            if telemetry is not None:
                telemetry.add_fallback("s2_rate_limited")
            return []
        results = []
        for entry in data or []:
            p = entry.get("citedPaper", {}) if isinstance(entry, dict) else {}
            ev = self._parse_paper(p, source="reference")
            if ev is not None:
                results.append(ev)
        return results

    async def get_by_doi(self, doi: str, telemetry: Telemetry | None = None) -> PaperIdentity | None:
        clean = doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
        try:
            data = await self._graph_call(f"/paper/DOI:{clean}", 1, fields="title,authors,year,venue,externalIds,abstract")
        except S2RateLimited:
            if telemetry is not None:
                telemetry.add_fallback("s2_rate_limited")
            return None
        if not data:
            return None
        ev = self._parse_paper(data, source="doi")
        return ev.identity if ev else None

    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception(lambda e: isinstance(e, S2RateLimited) or _is_retryable(e)),
        reraise=False,
        retry_error_callback=_graph_exhausted,
    )
    async def _graph_call(self, path: str, limit: int, fields: str) -> list[dict] | dict:
        import time

        headers = {"User-Agent": "scholartrace-contest/0.1 (research use)"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        params = {"fields": fields, "limit": min(limit, 100)}
        t0 = time.time()
        url = f"{API_BASE}{path}"
        resp = await self._http(path, path, "GET", url, params,
                                lambda: _http_get(url, params, headers))
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 429:
            raise S2RateLimited()
        if resp.status_code != 200:
            return [] if path.endswith("/citations") or path.endswith("/references") or "search" in path else {}

        data = resp.json()
        # search 接口返回 {"data": [...]}；response body 为 null/空时 data 可能是 None，
        # 统一规整为列表，避免上层 `for entry in data` 抛 TypeError。
        if isinstance(data, dict) and "data" in data:
            return data["data"] or []
        if isinstance(data, list):
            return data
        return [] if path.endswith("/citations") or path.endswith("/references") or "search" in path else {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=3, max=20),
        retry=retry_if_exception(lambda e: isinstance(e, S2RateLimited) or _is_retryable(e)),
        retry_error_callback=_search_once_exhausted,
    )
    async def _search_once(self, query: str, limit: int) -> list[dict]:
        import time

        headers = {"User-Agent": "scholartrace-contest/0.1 (research use)"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        t0 = time.time()
        url = f"{API_BASE}/paper/search"
        # tldr: 方向4 语义特征（行为中性，仅存元数据供精排用）
        params = {"query": query, "limit": min(limit, 100), "fields": "title,authors,year,venue,externalIds,abstract,tldr"}
        resp = await self._http("search", query, "GET", url, params,
                                lambda: _http_get(url, params, headers))
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 429:
            raise S2RateLimited()
        if resp.status_code >= 500:
            resp.raise_for_status()  # 5xx 交给重试
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("data", [])

    # ------------------------------------------------------------------
    def _parse_paper(self, p: dict, source: str) -> PaperEvidence | None:
        title = p.get("title")
        if not title:
            return None
        ext = p.get("externalIds") or {}
        s2_id = p.get("paperId") or ""
        doi = ext.get("DOI")
        # ArXiv ID 是 gold 对齐的关键身份键（PASA gold 仅含 title + arxiv_id）
        arxiv = ext.get("ArXiv") or (ext.get("arXiv") or "")
        source_ids = {"s2": s2_id, **({"doi": doi} if doi else {})}
        if arxiv:
            source_ids["arxiv"] = arxiv
        # 方向4 语义特征（行为中性元数据）：search 响应自带的 relevance + tldr 摘要
        if p.get("relevance") is not None:
            source_ids["s2_relevance"] = str(round(float(p["relevance"]), 4))
        tldr = p.get("tldr") or {}
        if isinstance(tldr, dict) and tldr.get("text"):
            source_ids["s2_tldr"] = tldr["text"]
        identity = PaperIdentity(
            paper_id=s2_id or f"doi:{doi}",
            title=title,
            doi=doi,
            authors=[a.get("name", "") for a in (p.get("authors") or []) if a.get("name")],
            venue=p.get("venue"),
            year=p.get("year"),
            source_ids=source_ids,
        )
        abstract = p.get("abstract")
        return PaperEvidence(
            identity=identity,
            abstract_scope="abstract" if abstract else "none",
            abstract=abstract,
            source=source,
        )
