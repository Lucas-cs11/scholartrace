"""后端 API 层测试（mock SearchEngine 方法，不产生真实检索）。"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import PaperIdentity, RankLabel, RankResult
from src.telemetry import Telemetry


def _rank_result(pid: str, title: str, score: float, label: RankLabel) -> RankResult:
    return RankResult(paper=PaperIdentity(paper_id=pid, title=title, doi=f"10.1/{pid}"), score=score, label=label)


@pytest.fixture
def client(monkeypatch, tmp_path):
    from api import main as api

    # R3：隔离 DB 到临时文件，避免污染真实 data/scholartrace.db
    monkeypatch.setattr(api, "_DB_PATH", tmp_path / "test.db")
    api._init_db()

    async def fake_search_b3(query, top_k=20):
        from src.schemas import SearchTrace

        t = Telemetry()
        t.add_api("openalex", 500.0)
        t.add_llm("deepseek-chat", 800.0, 100, 20)
        r = _rank_result("W1", "PointNet Deep Learning", 0.95, RankLabel.HIGH)
        r.constraint_coverage = {"must:深度学习": "命中"}
        return [
            r,
            _rank_result("W2", "Some Survey", 0.10, RankLabel.NO),
        ], t, [
            SearchTrace(round=1, query="point cloud", api="openalex", candidate_delta=2),
        ]

    class FakeSummarizer:
        async def summarize(self, query, ir, results, telemetry=None, relation_links=None):
            return type("S", (), {"model_dump": lambda self: {"query_summary": "mock summary", "groups": []}})()

    monkeypatch.setattr(api._engine, "summarizer", FakeSummarizer())
    monkeypatch.setattr(api, "MODE_MAP", {"b3": fake_search_b3, "b1": fake_search_b3, "full": fake_search_b3})
    return TestClient(api.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_ok(client):
    resp = client.post("/search", json={"query": "point cloud", "mode": "b3", "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "point cloud"
    assert body["mode"] == "b3"
    assert len(body["results"]) == 2
    top = body["results"][0]
    assert top["paper_id"] == "W1"
    assert top["label"] == "high"
    assert top["doi"] == "10.1/W1"
    assert top["constraint_coverage"] == {"must:深度学习": "命中"}  # 证据链透出
    assert body["telemetry"]["api_calls"] == 1
    assert body["telemetry"]["llm_calls"] == 1
    assert len(body["traces"]) == 1                                  # 搜索轨迹返回
    assert body["traces"][0]["api"] == "openalex"
    assert "endpoint_latency_ms" in body["telemetry"]


def test_search_default_mode_is_full(client):
    resp = client.post("/search", json={"query": "point cloud"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "full"


def test_search_unknown_mode_422(client):
    """mode 用 Literal 校验，非法值返回 422。"""
    resp = client.post("/search", json={"query": "x", "mode": "b9"})
    assert resp.status_code == 422


def test_search_empty_query_422(client):
    resp = client.post("/search", json={"query": "", "mode": "b3"})
    assert resp.status_code == 422


def test_search_invalid_top_k_422(client):
    resp = client.post("/search", json={"query": "x", "mode": "b3", "top_k": 0})
    assert resp.status_code == 422


def test_search_summarize_returns_summary(client):
    """summarize=true 时响应含结果归纳（R1 结构化展示）。"""
    resp = client.post("/search", json={"query": "point cloud", "mode": "b3", "summarize": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["query_summary"] == "mock summary"


def test_search_no_summary_by_default(client):
    resp = client.post("/search", json={"query": "point cloud", "mode": "b3"})
    assert resp.json()["summary"] == {}


# ------------------------------------------------------------------ R2 前端 Demo
def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ScholarTrace" in resp.text
    assert "app.js" in resp.text


def test_static_assets_served(client):
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    assert "search" in resp.text
    resp_css = client.get("/static/style.css")
    assert resp_css.status_code == 200


# ------------------------------------------------------------------ R3 搜索历史
def test_search_records_history(client):
    client.post("/search", json={"query": "point cloud", "mode": "b3"})
    resp = client.get("/history")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows, "历史应为空到有记录"
    assert rows[0]["query"] == "point cloud"
    assert rows[0]["mode"] == "b3"
    assert rows[0]["result_count"] == 2
    assert rows[0]["created_at"]


def test_history_limit(client):
    for i in range(3):
        client.post("/search", json={"query": f"query {i}", "mode": "b1"})
    resp = client.get("/history?limit=2")
    assert len(resp.json()) == 2
