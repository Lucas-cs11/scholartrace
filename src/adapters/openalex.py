"""OpenAlex 学术 API 适配器（主召回）。

无需 API key；建议配置 OPENALEX_MAILTO 进入 polite pool（降低 429 概率）。
并行检索会触发限流，所有请求统一走 _get_json（tenacity 指数退避重试）。
API: https://docs.openalex.org/api
"""
from __future__ import annotations

import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.adapters.base import AcademicSearchAdapter
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry

API_BASE = "https://api.openalex.org"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex 摘要用倒排索引存储，这里还原为原始文本。"""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in range(max(positions) + 1) if i in positions)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (409, 429, 500, 502, 503, 504)
    return False


class OpenAlexAdapter(AcademicSearchAdapter):
    name = "openalex"

    def __init__(self, mailto: str = "", cache=None):
        super().__init__(cache)
        self._mailto = mailto
        self._client = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    async def _get_json(self, path: str, params: dict, telemetry: Telemetry | None, note: str) -> httpx.Response:
        if self._mailto:
            params = {**params, "mailto": self._mailto}
        endpoint, seed = self._classify(path, params)
        t0 = time.time()
        url = f"{API_BASE}{path}"
        resp = await self._http(endpoint, seed, "GET", url, params,
                                lambda: self._client.get(url, params=params))
        latency_ms = round((time.time() - t0) * 1000, 1)
        if telemetry is not None:
            telemetry.add_api(self.name, latency_ms, note=note)
        resp.raise_for_status()  # 429/5xx 由 tenacity 接管；其余异常冒泡
        return resp

    @staticmethod
    def _classify(path: str, params: dict) -> tuple[str, str]:
        """把 HTTP 请求归类为逻辑端点 + 检索种子（供 recorder/cache 分桶）。"""
        if params.get("search"):
            return "search", str(params["search"])
        if params.get("filter"):
            return "citations", str(params["filter"])
        if "doi:" in path:
            return "get_by_doi", path
        return "get_by_id", path

    # ------------------------------------------------------------------
    async def search(self, query: str, limit: int = 20, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        try:
            resp = await self._get_json(
                "/works",
                {"search": query, "per-page": min(limit, 200)},
                telemetry,
                f"search limit={limit}",
            )
        except httpx.HTTPError:
            # 上游限流（429）/5xx/网络故障在重试耗尽后：不把上游故障当论文数据，
            # 静默降级为空（与 S2/OpenCitations/arXiv 适配器一致）。
            if telemetry is not None:
                telemetry.add_fallback("openalex_search_http_error")
            return []
        results = []
        for w in resp.json().get("results", []):
            ev = self._parse_work(w, source="search")
            if ev is not None:
                results.append(ev)
        return results

    async def get_by_id(self, openalex_id: str, telemetry: Telemetry | None = None) -> PaperEvidence | None:
        try:
            resp = await self._get_json(f"/works/{openalex_id}", {}, telemetry, f"get_by_id {openalex_id}")
        except httpx.HTTPError:
            return None  # 404 未收录 / 上游故障：不把上游故障当论文数据
        return self._parse_work(resp.json(), source="detail")

    async def get_by_doi(self, doi: str, telemetry: Telemetry | None = None) -> PaperIdentity | None:
        clean = doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
        try:
            resp = await self._get_json(f"/works/doi:{clean}", {}, telemetry, f"get_by_doi {doi}")
        except httpx.HTTPError:
            return None  # DOI 未索引 / 上游故障：不把上游故障当论文数据
        w = resp.json()
        return self._parse_work(w, source="doi").identity if w.get("id") else None

    async def get_citations(self, paper_id: str, limit: int = 10, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        """被引论文（OpenAlex 用 filter=cites:...）。"""
        try:
            resp = await self._get_json(
                "/works",
                {"filter": f"cites:{paper_id}", "per-page": min(limit, 200)},
                telemetry,
                f"citations {paper_id}",
            )
        except httpx.HTTPError:
            # 上游故障重试耗尽：静默降级为空，不中断整个检索链路
            if telemetry is not None:
                telemetry.add_fallback("openalex_citations_http_error")
            return []
        results = []
        for w in resp.json().get("results", []):
            ev = self._parse_work(w, source="citation")
            if ev is not None:
                results.append(ev)
        return results

    # ------------------------------------------------------------------
    def _parse_work(self, w: dict, source: str) -> PaperEvidence | None:
        openalex_id = w.get("id", "").rsplit("/", 1)[-1]
        if not openalex_id:
            return None

        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name")

        authors = []
        for a in w.get("authorships", []):
            name = ((a.get("author") or {}).get("display_name")) or ""
            if name:
                authors.append(name)

        identity = PaperIdentity(
            paper_id=openalex_id,
            title=w.get("title") or "",
            doi=doi or None,
            authors=authors,
            venue=venue,
            year=w.get("publication_year"),
            source_ids={"openalex": openalex_id},
        )
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        return PaperEvidence(
            identity=identity,
            abstract_scope="fulltext" if abstract else "none",
            abstract=abstract,
            source=source,
        )
