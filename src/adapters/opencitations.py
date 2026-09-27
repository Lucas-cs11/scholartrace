"""OpenCitations 引文图适配器（无 key，免费）。

按 DOI 查引用/被引，用于 B2 引文扩展（S2 之外的免 key 兜底源）。
API: https://opencitations.net/index/coci/api/v1/
注意：只返回 DOI 关系，不含标题/作者——身份字段 title 为空，由上层
（search._expand_citations）用 Crossref 补全或依赖 DOI/title 匹配。
"""
from __future__ import annotations

import httpx

from src.adapters.base import AcademicSearchAdapter, _http_get, http_retry
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry

API_BASE = "https://opencitations.net/index/coci/api/v1"


def _clean_doi(doi: str) -> str:
    """OpenCitations 返回的 DOI 带 doi: 前缀，剥离并清理 URL 前缀。"""
    doi = doi.strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(p):
            doi = doi[len(p):]
    return doi


class OpenCitationsAdapter(AcademicSearchAdapter):
    """引文图扩展：citations（被引）/ references（引用）。"""

    name = "opencitations"

    async def search(self, query: str, limit: int = 20, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        return []  # 仅用于引文扩展，不做关键词检索

    async def get_by_doi(self, doi: str, telemetry: Telemetry | None = None) -> PaperIdentity | None:
        return None  # 不提供论文元数据

    async def get_citations(self, paper_id: str, limit: int = 10, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        """被引该论文的后续论文（citing DOIs）。paper_id 为 DOI 或 DOI:...。"""
        return await self._coci(paper_id, "citations", limit, telemetry)

    async def get_references(self, paper_id: str, limit: int = 10, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        """该论文引用的论文（cited DOIs）。"""
        return await self._coci(paper_id, "references", limit, telemetry)

    # ------------------------------------------------------------------
    @http_retry
    async def _coci(
        self, paper_id: str, kind: str, limit: int, telemetry: Telemetry | None
    ) -> list[PaperEvidence]:
        import time

        doi = _clean_doi(paper_id)
        if not doi:
            return []
        url = f"{API_BASE}/{kind}/{doi}"
        t0 = time.time()
        try:
            resp = await self._http(kind, doi, "GET", url, None,
                                    lambda: _http_get(url, follow_redirects=True))
        except httpx.HTTPStatusError:
            # 限流/5xx 已由重试接管并耗尽，返回空（不把上游故障当论文数据）
            return []
        except httpx.HTTPError:
            return []
        latency_ms = round((time.time() - t0) * 1000, 1)
        if telemetry is not None:
            telemetry.add_api(self.name, latency_ms, note=f"{kind} {doi[:40]}")
        if resp.status_code != 200:
            return []

        key = "citing" if kind == "citations" else "cited"
        results: list[PaperEvidence] = []
        seen: set[str] = set()
        for item in resp.json():
            raw = item.get(key)
            if not raw:
                continue
            clean = _clean_doi(raw)
            if not clean or clean in seen:
                continue
            seen.add(clean)
            results.append(
                PaperEvidence(
                    identity=PaperIdentity(
                        paper_id=f"doi:{clean}",
                        title="",
                        doi=clean,
                        source_ids={"opencitations": clean},
                    ),
                    abstract_scope="none",
                    source=kind,
                )
            )
            if len(results) >= limit:
                break
        return results
