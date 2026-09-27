"""核心 schema 单元测试。"""
from src.schemas import PaperIdentity, QueryIR, RankLabel, RankResult, RunReport


def test_queryir_defaults():
    q = QueryIR(raw_query="test")
    assert q.entities == []
    assert q.must_constraints == []
    assert q.unknown == []


def test_queryir_fields():
    q = QueryIR(raw_query="x", topic="3D", year_min=2015, year_max=2025, venues=["CVPR"])
    assert q.year_min == 2015
    assert q.venues == ["CVPR"]


def test_paper_identity_doi_required_fields():
    p = PaperIdentity(paper_id="W123", title="Test Paper", doi="10.1/x")
    assert p.paper_id == "W123"
    assert p.doi == "10.1/x"
    assert p.authors == []


def test_rank_result_label_default():
    r = RankResult(paper=PaperIdentity(paper_id="W1", title="t"), score=0.9)
    assert r.label == RankLabel.NO


def test_run_report_tokens():
    r = RunReport(query_id="q1", input_tokens=100, output_tokens=50)
    assert r.total_tokens == 150
