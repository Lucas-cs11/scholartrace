"""FULL 全链路模块测试（mock，不产生真实调用）。"""
import asyncio

from config.settings import Settings
from src.schemas import PaperEvidence, PaperIdentity, QueryIR, RankLabel, RankResult, SubQuery
from src.search import SearchEngine
from src.telemetry import Telemetry


def _run(awaitable):
    return asyncio.run(awaitable)


def _ev(pid: str, title: str, doi: str = "", s2: str = "", abstract: str = "") -> PaperEvidence:
    source = {}
    if s2:
        source["s2"] = s2
    if doi:
        source["doi"] = doi
    return PaperEvidence(
        identity=PaperIdentity(paper_id=pid, title=title, doi=doi or None, source_ids=source),
        abstract=abstract,
    )


class RoundS2:
    """按 seed 返回不同引文，模拟多轮扩展链。resp: {paper_id: (citations, references)}"""

    def __init__(self, resp: dict):
        self.resp = resp
        self.calls: list[str] = []

    async def get_citations(self, paper_id, limit=10, telemetry=None):
        self.calls.append(f"cite:{paper_id}")
        return self.resp.get(paper_id, ([], []))[0]

    async def get_references(self, paper_id, limit=10, telemetry=None):
        self.calls.append(f"ref:{paper_id}")
        return self.resp.get(paper_id, ([], []))[1]


def _engine(s2, cfg=None, **kw):
    from src.parser import QueryIRParser
    from src.planner import SubQueryPlanner
    from src.ranker import LLMReranker

    class NoopParser(QueryIRParser):
        async def parse(self, query, telemetry=None):
            return QueryIR(raw_query=query)

    class NoopPlanner(SubQueryPlanner):
        async def plan(self, ir, telemetry=None):
            return [SubQuery(id="sq1", query_text=ir.raw_query, intent="原始查询", priority=1)]

    class NoopReranker(LLMReranker):
        async def rerank(self, query, ir, candidates, telemetry=None):
            return []

    engine = SearchEngine(parser=NoopParser(), planner=NoopPlanner(), reranker=NoopReranker(), cfg=cfg)
    engine.s2 = s2
    return engine


# ------------------------------------------------------------------ F1 多轮引文扩展
def test_multi_round_expansion_chains():
    """round1 W1->W2，round2 W2->W3，W1 不重复扩展。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")
    w2 = _ev("W2", "PointNet++ Point Cloud Hierarchical Learning", s2="s2:W2")
    w3 = _ev("W3", "Point Cloud Transformer Segmentation", s2="s2:W3")
    s2 = RoundS2({
        "s2:W1": ([w2], []),
        "s2:W2": ([w3], []),
    })
    engine = _engine(s2, cfg=cfg)

    telemetry = Telemetry()
    traces = []
    result = _run(engine._expand_citations_multi("point cloud classification", [w1], telemetry, traces, max_rounds=2))

    ids = {e.identity.paper_id for e in result}
    assert ids == {"W1", "W2", "W3"}          # 多轮链完整
    # W1 只扩展一次（calls 里 cite:s2:W1 仅 1 次）
    assert s2.calls.count("cite:s2:W1") == 1
    assert "cite:s2:W2" in s2.calls
    assert len(traces) == 2                    # 2 轮 SearchTrace
    assert traces[0].round == 2 and traces[1].round == 3


def test_multi_round_stops_on_no_new_seed():
    """无新 seed 时提前停止。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")
    s2 = RoundS2({"s2:W1": ([], [])})  # W1 引文为空 -> 无新候选
    engine = _engine(s2, cfg=cfg)

    result = _run(engine._expand_citations_multi("point cloud", [w1], Telemetry(), [], max_rounds=3))
    assert len(result) == 1                    # 只 W1，无扩展
    assert s2.calls.count("cite:s2:W1") == 1   # 只跑了一轮


def test_multi_round_respects_max_rounds():
    """max_rounds=1 只扩展 1 轮。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")
    w2 = _ev("W2", "PointNet++ Point Cloud Hierarchical Learning", s2="s2:W2")
    s2 = RoundS2({
        "s2:W1": ([w2], []),
        "s2:W2": ([_ev("W9", "Unrelated Survey")], []),
    })
    engine = _engine(s2, cfg=cfg)

    traces = []
    result = _run(engine._expand_citations_multi("point cloud", [w1], Telemetry(), traces, max_rounds=1))
    assert {e.identity.paper_id for e in result} == {"W1", "W2"}  # 第二轮没跑
    assert len(traces) == 1


def test_multi_round_budget_stops():
    """budgeted=True 且预算极小时提前停止。"""
    cfg = Settings(semantic_scholar_api_key="sk-test", budget_max_api_calls=1)
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")
    s2 = RoundS2({"s2:W1": ([_ev("W2", "PointNet++ Learning")], [])})
    engine = _engine(s2, cfg=cfg)

    telemetry = Telemetry()
    telemetry.add_api("openalex", 1.0)  # 已用 1 次 -> 预算耗尽
    result = _run(engine._expand_citations_multi("point cloud", [w1], telemetry, [], budgeted=True))
    assert len(result) == 1  # 未扩展


def test_multi_round_opencitations_enrich():
    """无 S2 key：走 OpenCitations，title 经 Crossref 补全。"""
    cfg = Settings(semantic_scholar_api_key="")  # 显式清除：本测试意图验证无 S2 key 的 OpenCitations 路径
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", doi="10.1/w1")
    cite = PaperEvidence(identity=PaperIdentity(paper_id="doi:10.2/c1", title="", doi="10.2/c1"))
    s2 = RoundS2({})  # 不会调用 S2

    class FakeOC:
        async def get_citations(self, doi, limit=10, telemetry=None):
            return [cite]

        async def get_references(self, doi, limit=10, telemetry=None):
            return []

    class FakeCrossref:
        async def get_by_doi(self, doi, telemetry=None):
            return PaperIdentity(paper_id=f"doi:{doi}", title=f"Enriched {doi}", doi=doi)

    engine = _engine(s2, cfg=cfg)
    engine.opencitations = FakeOC()
    engine.crossref = FakeCrossref()

    result = _run(engine._expand_citations_multi("point cloud", [w1], Telemetry(), [], max_rounds=1))
    ids = {e.identity.paper_id for e in result}
    assert "doi:10.2/c1" in ids
    enriched = next(e for e in result if e.identity.paper_id == "doi:10.2/c1")
    assert enriched.identity.title == "Enriched 10.2/c1"  # 补全生效


# ------------------------------------------------------------------ F2 早停决策
def test_should_stop_matrix():
    from src.search import SearchEngine

    assert SearchEngine._should_stop(1, 3, 1, 0, False) is True    # 高相关为 0
    assert SearchEngine._should_stop(3, 3, 1, 2, False) is True    # 达轮数上限
    assert SearchEngine._should_stop(1, 3, 2, 2, False) is True    # 高相关不增长（收敛）
    assert SearchEngine._should_stop(1, 3, 1, 2, False) is False   # 增长 -> 继续
    assert SearchEngine._should_stop(1, 3, 1, 2, True) is True     # 预算耗尽


def test_high_relevance_counts():
    engine = _engine(RoundS2({}))
    evs = [
        _ev("W1", "PointNet 3D Point Cloud Classification"),
        _ev("W2", "Quantum Computing Survey"),
    ]
    assert engine._high_relevance("point cloud classification", evs) == 1


def test_multi_round_stops_on_high_relevance_saturation():
    """第二轮扩展出的候选不相关（高相关不增长）-> 早停，不再扩展。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")
    w1 = _ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")
    w2 = _ev("W2", "PointNet++ Point Cloud Hierarchical Learning", s2="s2:W2")
    w3 = _ev("W3", "Quantum Computing Survey", s2="s2:W3")  # 不相关 -> 高相关不增
    s2 = RoundS2({
        "s2:W1": ([w2], []),
        "s2:W2": ([w3], []),
    })
    engine = _engine(s2, cfg=cfg)

    traces = []
    result = _run(engine._expand_citations_multi("point cloud classification", [w1], Telemetry(), traces, max_rounds=3))
    assert len(traces) == 2                      # 只在 round2 停（高相关饱和）
    assert {e.identity.paper_id for e in result} == {"W1", "W2", "W3"}
    assert s2.calls.count("cite:s2:W3") == 0     # W3 未被作为 seed 继续扩展


# ------------------------------------------------------------------ F4 search_full 集成
class FullOpenAlex:
    """mock 子查询召回，返回固定候选。"""

    def __init__(self):
        self.calls: list[str] = []

    async def search(self, query, limit=20, telemetry=None):
        self.calls.append(query)
        if telemetry:
            telemetry.add_api("openalex", 5.0)
        return [_ev("W1", "PointNet 3D Point Cloud Classification", s2="s2:W1")]


def _full_engine(s2, cfg=None):
    from src.ranker import LLMReranker

    class FullParser:
        async def parse(self, query, telemetry=None):
            if telemetry:
                telemetry.add_llm("fake", 1.0, 100, 20)
            return QueryIR(raw_query=query)

    class FullPlanner:
        async def plan(self, ir, telemetry=None):
            return [SubQuery(id="sq1", query_text=ir.raw_query, intent="原始查询", priority=1)]

    class FullReranker(LLMReranker):
        def __init__(self, results):
            self._results = results
            self.calls = []

        async def rerank(self, query, ir, candidates, telemetry=None):
            self.calls.append(1)
            return self._results

    reranker = FullReranker([RankResult(paper=PaperIdentity(paper_id="W2", title="PointNet++"), score=0.9,
                                        label=RankLabel.HIGH, constraint_coverage={"must:x": "命中"})])
    engine = SearchEngine(parser=FullParser(), planner=FullPlanner(), reranker=reranker, cfg=cfg)
    engine.openalex = FullOpenAlex()
    engine.s2 = s2
    return engine


def test_search_full_full_pipeline():
    """FULL 全链路：召回 -> 引文扩展 -> 精排（含证据链）。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")
    w2 = _ev("W2", "PointNet++ Point Cloud Hierarchical", s2="s2:W2")
    s2 = RoundS2({"s2:W1": ([w2], [])})
    engine = _full_engine(s2, cfg=cfg)

    results, telemetry, _ = _run(engine.search_full("point cloud classification", top_k=10))
    assert results                       # 有结果
    assert results[0].paper.paper_id == "W2"  # 精排把引文扩展的 W2 排前
    assert results[0].constraint_coverage == {"must:x": "命中"}  # 证据链
    assert "cite:s2:W1" in s2.calls       # 引文扩展执行
    assert engine.reranker.calls           # 精排被调


def test_search_full_rerank_failure_falls_back():
    """精排失败回退词法。"""
    cfg = Settings(semantic_scholar_api_key="sk-test")

    class EmptyReranker:
        async def rerank(self, query, ir, candidates, telemetry=None):
            return []

    s2 = RoundS2({"s2:W1": ([], [])})
    engine = _full_engine(s2, cfg=cfg)
    engine.reranker = EmptyReranker()

    results, _, _ = _run(engine.search_full("point cloud classification", top_k=10))
    assert results                       # 词法回退仍有结果
    assert results[0].paper.paper_id == "W1"


def test_search_full_empty_recall():
    """召回为空返回空。"""

    class EmptyOA(FullOpenAlex):
        async def search(self, query, limit=20, telemetry=None):
            if telemetry:
                telemetry.add_api("openalex", 5.0)
            return []

    engine = _full_engine(RoundS2({}), Settings())
    engine.openalex = EmptyOA()
    results, _, _ = _run(engine.search_full("nothing"))
    assert results == []
