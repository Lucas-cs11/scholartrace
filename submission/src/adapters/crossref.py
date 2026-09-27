"""Crossref 学术 API 适配器（DOI 身份校验 / 辅助检索）。

无需 API key。API: https://api.crossref.org
"""
from __future__ import annotations

import httpx

from src.adapters.base import AcademicSearchAdapter, _http_get, http_retry
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry

API_BASE = "https://api.crossref.org"


class CrossrefAdapter(AcademicSearchAdapter):
    name = "crossref"

    def __init__(self, mailto: str = "", cache=None):
        super().__init__(cache)
        self._mailto = mailto

    @http_retry
    async def search(self, query: str, limit: int = 20, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        import time

        params = {
            "query.bibliographic": query,
            "rows": min(limit, 200),
            "select": "DOI,title,author,container-title,issued,abstract,reference",
        }
        if self._mailto:
            params["mailto"] = self._mailto

        t0 = time.time()
        url = f"{API_BASE}/works"
        try:
            resp = await self._http("search", query, "GET", url, params,
                                    lambda: _http_get(url, params))
        except httpx.HTTPError:
            # 上游限流/5xx/网络故障在重试耗尽后：不把上游故障当论文数据
            if telemetry is not None:
                telemetry.add_fallback("crossref_search_http_error")
            return []
        latency_ms = (time.time() - t0) * 1000
        if telemetry is not None:
            telemetry.add_api(self.name, latency_ms, note=f"search limit={limit}")
        if resp.status_code != 200:
            return []

        results = []
        for item in resp.json().get("message", {}).get("items", []):
            ev = self._parse_item(item, source="search")
            if ev is not None:
                results.append(ev)
        return results

    @http_retry(on_exhausted=lambda state: None)
    async def get_by_doi(self, doi: str, telemetry: Telemetry | None = None) -> PaperIdentity | None:
        import time

        clean = doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
        t0 = time.time()
        url = f"{API_BASE}/works/{clean}"
        params = {"select": "DOI,title,author,container-title,issued"}
        resp = await self._http("get_by_doi", clean, "GET", url, params,
                                lambda: _http_get(url, params))
        latency_ms = (time.time() - t0) * 1000
        if telemetry is not None:
            telemetry.add_api(self.name, latency_ms, note=f"get_by_doi {doi}")
        if resp.status_code >= 500:
            resp.raise_for_status()  # 5xx 交给重试；404/400 等非 200 正常返回 None
        if resp.status_code != 200:
            return None
        item = resp.json().get("message", {})
        if not item.get("DOI"):
            return None
        return self._parse_item(item, source="doi").identity

    # ------------------------------------------------------------------
    def _parse_item(self, item: dict, source: str) -> PaperEvidence | None:
        doi = item.get("DOI", "")
        if not doi:
            return None

        authors = []
        for a in item.get("author", []):
            name = " ".join(x for x in (a.get("given", ""), a.get("family", "")) if x)
            if name:
                authors.append(name)

        issued = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        venue = (item.get("container-title") or [None])[0]

        identity = PaperIdentity(
            paper_id=f"doi:{doi}",
            title=(item.get("title") or [""])[0],
            doi=doi,
            authors=authors,
            venue=venue,
            year=issued,
            source_ids={"crossref": doi},
        )
        return PaperEvidence(identity=identity, abstract_scope="none", source=source)
