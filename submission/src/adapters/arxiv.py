"""arXiv API 适配器（免费，无需 key，CS/物理预印本补充召回）。

API: http://export.arxiv.org/api/query（Atom XML，约 3 req/s 限流）。
用于 OpenAlex 之外的补充召回源：赛题的 ML/CS 经典论文大量首发于 arXiv。
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from src.adapters.base import AcademicSearchAdapter
from src.schemas import PaperEvidence, PaperIdentity
from src.telemetry import Telemetry

API_BASE = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


class ArxivAdapter(AcademicSearchAdapter):
    """arXiv 预印本关键词检索（补充召回）。"""

    name = "arxiv"

    async def search(self, query: str, limit: int = 20, telemetry: Telemetry | None = None) -> list[PaperEvidence]:
        import time

        # arXiv 检索：分词后空格连接（arXiv 将空格视为 AND），避免长短语精确匹配返回 0
        tokens = re.findall(r"[a-z0-9]+", query.lower())[:6]
        if not tokens:
            return []
        params = {"search_query": f"all:{' '.join(tokens)}", "start": 0, "max_results": min(limit, 50)}

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(API_BASE, params=params)
        except httpx.HTTPError:
            return []
        latency_ms = round((time.time() - t0) * 1000, 1)
        if telemetry is not None:
            telemetry.add_api(self.name, latency_ms, note=f"search limit={limit}")
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        results: list[PaperEvidence] = []
        for entry in root.findall("atom:entry", _NS):
            ev = self._parse_entry(entry)
            if ev is not None:
                results.append(ev)
            if len(results) >= limit:
                break
        return results

    async def get_by_doi(self, doi: str, telemetry: Telemetry | None = None) -> PaperIdentity | None:
        return None  # arXiv 预印本不一定有 DOI，不提供 DOI 查询

    # ------------------------------------------------------------------
    def _parse_entry(self, entry: ET.Element) -> PaperEvidence | None:
        title = _text(entry.find("atom:title", _NS))
        if not title:
            return None
        authors = [
            _text(a.find("atom:name", _NS))
            for a in entry.findall("atom:author", _NS)
            if _text(a.find("atom:name", _NS))
        ]
        arxiv_id = ""
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf" and link.get("href"):
                arxiv_id = link.get("href").rsplit("/", 1)[-1].replace(".pdf", "")
                break
        if not arxiv_id:
            m = re.search(r"arXiv:([\w.-]+)", entry.find("atom:id", _NS).text or "")
            arxiv_id = m.group(1) if m else ""

        year = None
        published = _text(entry.find("atom:published", _NS))
        if len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                year = None

        identity = PaperIdentity(
            paper_id=f"arxiv:{arxiv_id}" if arxiv_id else title[:40],
            title=title,
            authors=authors,
            year=year,
            source_ids={"arxiv": arxiv_id} if arxiv_id else {},
        )
        abstract = _text(entry.find("atom:summary", _NS))
        return PaperEvidence(
            identity=identity,
            abstract_scope="abstract" if abstract else "none",
            abstract=abstract,
            source="arxiv",
        )
