"""评测错误分析工具单元测试。"""
from scripts.analyze_runs import classify_query, _load_recall_ids, _openalex_keys


def test_openalex_keys():
    assert _openalex_keys(["W1", "W2"]) == {"openalex:W1", "openalex:W2"}
    assert _openalex_keys(["W1", "", None]) == {"openalex:W1"}


def test_classify_hit():
    gold = [{"openalex:W1", "doi:10.1/a"}]
    assert classify_query(gold, ["W1"], ["W1", "W9"]) == "hit"


def test_classify_rank_fail():
    """gold 在召回候选但不在预测结果 -> 排序失败。"""
    gold = [{"openalex:W2", "title:pointnet"}]
    assert classify_query(gold, ["W1"], ["W1", "W2"]) == "rank_fail"


def test_classify_recall_fail():
    """gold 不在任何召回候选 -> 召回失败。"""
    gold = [{"openalex:W9"}]
    assert classify_query(gold, ["W1"], ["W1", "W2"]) == "recall_fail"


def test_classify_title_only_gold():
    """工具仅做 ID 匹配：gold 仅含标题标识（无 openalex ID）时无法与候选 ID 对齐，
    视为未命中——真实命中判定以 harness（含 title/doi 归一化）为准。"""
    gold = [{"title:pointnet deep learning"}]
    assert classify_query(gold, ["W1"], ["W1"]) == "recall_fail"  # ID 无法对齐 title 标识


def test_load_recall_ids(tmp_path):
    p = tmp_path / "cache.jsonl"
    p.write_text(
        '{"q": "sub1", "evs": [{"identity": {"paper_id": "W1"}}]}\n'
        '{"q": "sub2", "evs": [{"identity": {"paper_id": "W2"}}]}\n',
        encoding="utf-8",
    )
    cache = _load_recall_ids(str(p))
    assert cache["sub1"] == {"openalex:W1"}
    assert cache["sub2"] == {"openalex:W2"}


def test_load_recall_ids_missing():
    assert _load_recall_ids("/nonexistent/path.jsonl") == {}
