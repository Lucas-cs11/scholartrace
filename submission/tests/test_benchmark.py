"""外部基准评测适配单元测试。"""
import asyncio
import json

import pytest

from scripts.eval_benchmark import load_pasa, norm_title, to_harness_gold


def test_norm_title():
    assert norm_title("PointNet: Deep Learning on Point Sets!") == "pointnetdeeplearningonpointsets"
    assert norm_title("  Attention  Is All  ") == "attentionisall"


def test_load_pasa(tmp_path):
    """真实 PaSa jsonl 格式：question/answer 在顶层 + answer_arxiv_id + qid。"""
    p = tmp_path / "real.jsonl"
    p.write_text(
        json.dumps({
            "question": "papers about point cloud classification",
            "answer": ["PointNet: Deep Learning", "Survey"],
            "answer_arxiv_id": ["1612.00593", ""],
            "qid": "RealScholarQuery_0",
        })
        + "\n"
        + json.dumps({"question": "second", "answer": [], "qid": "RealScholarQuery_1"})
        + "\n",
        encoding="utf-8",
    )
    queries = load_pasa(str(p))
    assert len(queries) == 2
    assert queries[0]["query_id"] == "RealScholarQuery_0"
    assert queries[0]["query"] == "papers about point cloud classification"
    assert queries[0]["gold"][0] == {"title": "PointNet: Deep Learning", "arxiv_id": "1612.00593"}
    assert "arxiv_id" not in queries[0]["gold"][1]  # 空 arxiv_id -> 不带该键
    assert queries[1]["gold"] == []  # 空 answer -> 无 gold


def test_load_pasa_missing_qid(tmp_path):
    """无 qid 时回退 qNNN。"""
    p = tmp_path / "legacy.jsonl"
    p.write_text(json.dumps({"question": "q", "answer": ["T"]}) + "\n", encoding="utf-8")
    queries = load_pasa(str(p))
    assert queries[0]["query_id"] == "q001"


def test_to_harness_gold(tmp_path):
    queries = [{"query_id": "q001", "query": "q", "gold": [{"title": "T"}]}]
    path = to_harness_gold(queries)
    import os
    try:
        data = [json.loads(l) for l in open(path, encoding="utf-8")]
        assert data[0]["query"] == "q"
        assert data[0]["gold"][0]["title"] == "T"
    finally:
        os.unlink(path)


def test_filter_existing_skips_done_queries(tmp_path):
    from scripts.eval_benchmark import filter_existing
    (tmp_path / "EXP_q1.json").write_text("{}")
    (tmp_path / "EXP_q3.json").write_text("{}")
    queries = [
        {"query_id": "q1"}, {"query_id": "q2"}, {"query_id": "q3"}, {"query_id": "q4"},
    ]
    out = filter_existing(queries, str(tmp_path), "EXP")
    assert [q["query_id"] for q in out] == ["q2", "q4"]


def test_filter_existing_no_done(tmp_path):
    from scripts.eval_benchmark import filter_existing
    queries = [{"query_id": "q1"}, {"query_id": "q2"}]
    out = filter_existing(queries, str(tmp_path), "EXP")
    assert len(out) == 2


def test_is_openalex_quota_error():
    import httpx
    from scripts.eval_benchmark import _is_openalex_quota_error

    def _mk(status, url):
        req = httpx.Request("GET", url)
        return httpx.HTTPStatusError(f"{status}", request=req, response=httpx.Response(status, request=req))

    assert _is_openalex_quota_error(_mk(429, "https://api.openalex.org/works?search=x")) is True
    assert _is_openalex_quota_error(_mk(503, "https://api.openalex.org/works")) is True
    # 非 openalex 源的 429 不等待（如 crossref/opencitations）
    assert _is_openalex_quota_error(_mk(429, "https://api.crossref.org/works")) is False
    # 非配额错误（404 等）不等待
    assert _is_openalex_quota_error(_mk(404, "https://api.openalex.org/works")) is False


def test_evaluate_reraises_selected_error(tmp_path):
    from eval.harness import evaluate

    gold = tmp_path / "gold.jsonl"
    gold.write_text(json.dumps({"query": "q", "gold": []}) + "\n", encoding="utf-8")

    class StopHere(RuntimeError):
        pass

    async def failing_search(query, top_k):
        raise StopHere("quota")

    with pytest.raises(StopHere):
        asyncio.run(evaluate(
            failing_search,
            str(gold),
            str(tmp_path / "runs"),
            skip_on_error=True,
            raise_on_error=lambda e: isinstance(e, StopHere),
        ))
