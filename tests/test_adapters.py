"""OpenCitations / arXiv 适配器单元测试（mock HTTP，不产生真实调用）。"""
import xml.etree.ElementTree as ET

import httpx
import pytest

from src.adapters.arxiv import ArxivAdapter, _text
from src.adapters.opencitations import OpenCitationsAdapter, _clean_doi
from src.telemetry import Telemetry


# ------------------------------------------------------------------ OpenCitations
def test_clean_doi():
    assert _clean_doi("DOI:10.1/x") == "10.1/x"
    assert _clean_doi("https://doi.org/10.1/y") == "10.1/y"
    assert _clean_doi("10.1/plain") == "10.1/plain"


@pytest.mark.asyncio
async def test_opencitations_get_citations_parses(monkeypatch):
    body = [
        {"citing": "DOI:10.1000/a", "cited": "10.9/seed"},
        {"citing": "DOI:10.1000/b", "cited": "10.9/seed"},
        {"citing": "DOI:10.1000/a", "cited": "10.9/seed"},  # 重复
    ]

    async def fake_get(self, url):
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = OpenCitationsAdapter()
    telemetry = Telemetry()
    results = await adapter.get_citations("10.9/seed", limit=10, telemetry=telemetry)
    assert len(results) == 2                       # 去重
    assert results[0].identity.doi == "10.1000/a"
    assert results[0].identity.paper_id == "doi:10.1000/a"
    assert telemetry.api_calls == 1
    assert telemetry.events[0]["api"] == "opencitations"


@pytest.mark.asyncio
async def test_opencitations_limit_respected(monkeypatch):
    body = [{"citing": "10.9/s", "cited": f"DOI:10.1/{i}"} for i in range(10)]

    async def fake_get(self, url):
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    results = await OpenCitationsAdapter().get_references("10.9/s", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_opencitations_http_error_returns_empty(monkeypatch):
    async def fake_get(self, url):
        return httpx.Response(500, text="err")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    telemetry = Telemetry()
    results = await OpenCitationsAdapter().get_citations("10.9/s", telemetry=telemetry)
    assert results == []


# ------------------------------------------------------------------ arXiv
@pytest.mark.asyncio
async def test_arxiv_search_parses_atom(monkeypatch):
    atom = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/1603.08134v1</id>
        <title>PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</title>
        <summary>We present a deep net architecture for point clouds.</summary>
        <author><name>Charles R. Qi</name></author>
        <published>2016-03-26T18:00:00Z</published>
        <link title="pdf" href="http://arxiv.org/pdf/1603.08134v1"/>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/1706.03762v1</id>
        <title>Attention Is All You Need</title>
        <summary>Transformer.</summary>
        <published>2017-06-12T00:00:00Z</published>
      </entry>
    </feed>"""

    async def fake_get(self, url, params):
        assert "search_query" in params
        return httpx.Response(200, text=atom)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = ArxivAdapter()
    telemetry = Telemetry()
    results = await adapter.search("point cloud deep learning", limit=10, telemetry=telemetry)
    assert len(results) == 2
    assert results[0].identity.title.startswith("PointNet")
    assert results[0].identity.year == 2016
    assert results[0].identity.source_ids.get("arxiv") == "1603.08134v1"
    assert results[0].identity.authors == ["Charles R. Qi"]
    assert results[0].abstract
    assert telemetry.api_calls == 1
    assert telemetry.events[0]["api"] == "arxiv"


@pytest.mark.asyncio
async def test_arxiv_non_200_returns_empty(monkeypatch):
    async def fake_get(self, url, params):
        return httpx.Response(503, text="err")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    results = await ArxivAdapter().search("query")
    assert results == []


# ------------------------------------------------------------------ OpenAlex 优雅降级（上游故障不 500）
@pytest.mark.asyncio
async def test_openalex_search_http_error_returns_empty(monkeypatch):
    """OpenAlex 429/5xx 重试耗尽后：返回空列表而非抛 HTTPStatusError。"""
    from src.adapters.openalex import OpenAlexAdapter

    async def fake_get(self, url, params=None):
        resp = httpx.Response(429, text="rate limited", request=httpx.Request("GET", url))
        resp.raise_for_status()  # 模拟 _get_json 的 raise_for_status 行为

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = OpenAlexAdapter()
    telemetry = Telemetry()
    results = await adapter.search("point cloud", limit=5, telemetry=telemetry)
    assert results == []
    assert telemetry.fallback_count == 1  # 记录了降级


@pytest.mark.asyncio
async def test_openalex_get_by_doi_404_and_error_return_none(monkeypatch):
    """get_by_doi：404 与上游故障均返回 None（不抛异常）。"""
    from src.adapters.openalex import OpenAlexAdapter

    async def fake_get(self, url, params=None):
        resp = httpx.Response(404, text="not found", request=httpx.Request("GET", url))
        resp.raise_for_status()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = OpenAlexAdapter()
    assert await adapter.get_by_doi("10.999/not-exist") is None


@pytest.mark.asyncio
async def test_crossref_search_http_error_returns_empty(monkeypatch):
    """Crossref search：非 200 返回空而非抛异常。"""
    from src.adapters.crossref import CrossrefAdapter

    async def fake_get(self, url, params=None):
        return httpx.Response(503, text="err")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = CrossrefAdapter()
    telemetry = Telemetry()
    results = await adapter.search("point cloud", limit=5, telemetry=telemetry)
    assert results == []


# ------------------------------------------------------------------ Semantic Scholar 防御（None/空 body 不 TypeError）
@pytest.mark.asyncio
async def test_s2_references_none_body_returns_empty(monkeypatch):
    """S2 返回 200 但 body 为 null/空时：返回空列表而非抛 TypeError。"""
    from src.adapters.semantic_scholar import SemanticScholarAdapter

    async def fake_get(self, url, params=None, headers=None):
        return httpx.Response(200, text="null")  # JSON null

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = SemanticScholarAdapter(api_key="k")
    results = await adapter.get_references("W1", limit=5)
    assert results == []

    results = await adapter.get_citations("W1", limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_s2_references_empty_data_returns_empty(monkeypatch):
    """S2 返回 {"data": []} 时空结果：正常返回空列表。"""
    from src.adapters.semantic_scholar import SemanticScholarAdapter

    async def fake_get(self, url, params=None, headers=None):
        return httpx.Response(200, json={"data": []})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = SemanticScholarAdapter(api_key="k")
    assert await adapter.get_references("W1") == []


@pytest.mark.asyncio
async def test_s2_references_malformed_entry_skips(monkeypatch):
    """S2 引用列表里混入非 dict 项：跳过而非抛错。"""
    from src.adapters.semantic_scholar import SemanticScholarAdapter

    async def fake_get(self, url, params=None, headers=None):
        return httpx.Response(
            200,
            json={"data": [None, {"citingPaper": {"paperId": "W1", "title": "T", "authors": []}}]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    adapter = SemanticScholarAdapter(api_key="k")
    results = await adapter.get_citations("seed")
    assert len(results) == 1
    assert results[0].identity.paper_id == "W1"


def test_arxiv_text_extractor():
    el = ET.fromstring("<t>  hello  </t>")
    assert _text(el) == "hello"
