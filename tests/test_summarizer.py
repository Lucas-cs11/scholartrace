"""搜索结果归纳模块单元测试（mock LLM，不产生真实调用）。"""
import asyncio

from src.llm import LLMError
from src.schemas import PaperIdentity, QueryIR, RankLabel, RankResult
from src.summarizer import SearchSummarizer
from src.telemetry import Telemetry


def _run(awaitable):
    return asyncio.run(awaitable)


class FakeLLM:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def complete_json(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._result


def _rank(pid: str, title: str, label: RankLabel = RankLabel.HIGH, cov: dict | None = None) -> RankResult:
    return RankResult(
        paper=PaperIdentity(paper_id=pid, title=title),
        score=0.9,
        label=label,
        constraint_coverage=cov or {},
    )


def _ir() -> QueryIR:
    return QueryIR(raw_query="point cloud classification", topic="3D point cloud")


# ------------------------------------------------------------------ 归纳
def test_summarize_populates_groups():
    llm = FakeLLM(result={
        "query_summary": "3D point cloud classification and segmentation research",
        "groups": [
            {"name": "Deep learning point clouds", "description": "深度学习点云方法", "paper_ids": ["W1", "W2"]},
            {"name": "Segmentation", "description": "分割方法", "paper_ids": ["W3"]},
        ],
        "overall_summary": "找到 PointNet 等关键论文",
    })
    results = [_rank("W1", "PointNet"), _rank("W2", "Survey"), _rank("W3", "Segmentation")]
    summary = _run(SearchSummarizer(llm=llm).summarize("point cloud", _ir(), results))
    assert summary.query_summary.startswith("3D point cloud")
    assert len(summary.groups) == 2
    assert summary.groups[0].paper_ids == ["W1", "W2"]
    assert summary.overall_summary == "找到 PointNet 等关键论文"


def test_summarize_filters_fabricated_paper_ids():
    """LLM 编造不在结果里的 paper_id 被过滤。"""
    llm = FakeLLM(result={
        "query_summary": "q",
        "groups": [
            {"name": "g1", "description": "", "paper_ids": ["W1", "W99"]},  # W99 编造
            {"name": "g2", "description": "", "paper_ids": []},
            {"name": "g3", "description": "", "paper_ids": ["W2"]},
        ],
    })
    results = [_rank("W1", "A"), _rank("W2", "B")]
    summary = _run(SearchSummarizer(llm=llm).summarize("q", _ir(), results))
    assert len(summary.groups) == 2            # g1（含编造）保留有效，g2（空）丢弃
    ids = [pid for g in summary.groups for pid in g.paper_ids]
    assert "W99" not in ids
    assert "W1" in ids and "W2" in ids


def test_summarize_llm_error_falls_back():
    telemetry = Telemetry()
    llm = FakeLLM(error=LLMError("boom"))
    results = [_rank("W1", "A")]
    summary = _run(SearchSummarizer(llm=llm).summarize("my query", _ir(), results, telemetry=telemetry))
    assert summary.query_summary == "my query"  # 降级
    assert summary.groups == []
    assert telemetry.fallback_count >= 1


def test_summarize_empty_results():
    summary = _run(SearchSummarizer(llm=FakeLLM(result={})).summarize("q", _ir(), []))
    assert summary.query_summary == "q"
    assert summary.groups == []


# ------------------------------------------------------------------ 关系图
def test_relation_graph_built():
    links = [("W1", "W2", "citation"), ("W2", "W3", "reference")]
    summary = _run(SearchSummarizer(llm=FakeLLM(result={})).summarize("q", _ir(), [], relation_links=links))
    assert summary.relation_graph
    assert len(summary.relation_graph["nodes"]) == 3
    assert len(summary.relation_graph["edges"]) == 2
    assert summary.relation_graph["edges"][0].source == "W1"
    assert summary.relation_graph["edges"][1].kind == "reference"


def test_relation_graph_empty_when_no_links():
    summary = _run(SearchSummarizer(llm=FakeLLM(result={})).summarize("q", _ir(), [], relation_links=[]))
    assert summary.relation_graph == {}
