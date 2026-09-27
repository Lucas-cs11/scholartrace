"""搜索编排基础逻辑测试（不触发外部 API）。"""
import asyncio

from src.schemas import PaperEvidence, PaperIdentity, QueryIR, RankLabel, RankResult, SubQuery
from src.search import SearchEngine, _tokens
from src.telemetry import Telemetry

engine = SearchEngine()


def test_tokens():
    assert _tokens("Deep 3D PointNet, V2!") == {"deep", "3d", "pointnet", "v2"}


def test_lexical_rank_orders_by_overlap():
    candidates = [
        PaperEvidence(identity=PaperIdentity(paper_id="W1", title="PointNet Deep Learning Point Sets"), abstract="point cloud"),
        PaperEvidence(identity=PaperIdentity(paper_id="W2", title="Random Image Stitching Survey"), abstract="images stitching"),
    ]
    ranked = engine._lexical_rank("pointnet deep learning point cloud", candidates)
    assert ranked[0].paper.paper_id == "W1"
    assert ranked[0].label == RankLabel.HIGH


def test_lexical_rank_empty_query():
    candidates = [PaperEvidence(identity=PaperIdentity(paper_id="W1", title="Anything"), abstract=None)]
    ranked = engine._lexical_rank("!!!", candidates)
    assert ranked[0].score == 0.0


# ------------------------------------------------------------------ B1 链路（mock parser/planner/openalex）
class FakeParser:
    async def parse(self, query, telemetry=None):
        if telemetry:
            telemetry.add_llm("fake", 1.0, 100, 20)  # 模拟真实 parse 的 token 消耗
        return QueryIR(raw_query=query, entities=["PointNet"])


class FakePlanner:
    async def plan(self, ir, telemetry=None):
        return [
            SubQuery(id="sq1", query_text="PointNet classification", intent="术语", priority=5),
            SubQuery(id="sq2", query_text="point cloud deep learning", intent="主题", priority=2),
        ]


class FakeOpenAlex:
    """按子查询关键词返回固定候选（模拟去重场景）。"""

    def __init__(self):
        self.calls: list[str] = []

    async def search(self, query, limit=20, telemetry=None):
        self.calls.append(query)
        if telemetry:
            telemetry.add_api("openalex", 5.0)  # 模拟真实适配器的记账行为
        if "PointNet" in query:
            return [
                PaperEvidence(identity=PaperIdentity(paper_id="W1", title="PointNet 3D Point Cloud Classification Network")),
            ]
        return [
            PaperEvidence(identity=PaperIdentity(paper_id="W1", title="PointNet 3D Point Cloud Classification Network")),  # 与上重复
            PaperEvidence(identity=PaperIdentity(paper_id="W2", title="3D Point Cloud Segmentation Survey"), abstract="deep learning"),
        ]


def _run(awaitable):
    return asyncio.run(awaitable)


def test_search_b1_full_flow():
    fake_oa = FakeOpenAlex()
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner())
    engine.openalex = fake_oa  # 替换真实适配器

    results, telemetry, _ = _run(engine.search_b1("3D point cloud classification with PointNet", top_k=10))

    # 两个子查询都被执行
    assert fake_oa.calls == ["PointNet classification", "point cloud deep learning"]
    # 跨子查询去重：W1 只出现一次
    ids = [r.paper.paper_id for r in results]
    assert ids.count("W1") == 1
    assert "W2" in ids
    # telemetry 记账：2 次 OpenAlex API 调用
    assert telemetry.api_calls == 2
    # 排序：原始 query 命中多的 W1 排最前
    assert results[0].paper.paper_id == "W1"
    assert results[0].label == RankLabel.HIGH


def test_search_b1_respects_subquery_budget():
    """超过 B1_MAX_SUBQUERIES 个子查询时按优先级截断。"""
    import src.search as search_mod
    orig = search_mod.B1_MAX_SUBQUERIES
    search_mod.B1_MAX_SUBQUERIES = 2

    class ManyPlanner(FakePlanner):
        async def plan(self, ir, telemetry=None):
            return [
                SubQuery(id=f"sq{i}", query_text=f"q{i}", intent="i", priority=p)
                for i, p in enumerate([5, 4, 3, 2, 1, 0], 1)  # 6 个，优先级 5..0
            ]

    try:
        fake_oa = FakeOpenAlex()
        engine = SearchEngine(parser=FakeParser(), planner=ManyPlanner())
        engine.openalex = fake_oa
        _run(engine.search_b1("test"))
        # 只执行了 2 个子查询
        assert len(fake_oa.calls) == 2
    finally:
        search_mod.B1_MAX_SUBQUERIES = orig


def test_search_b1_empty_plan_falls_back_to_raw():
    """planner 返回空列表时用原始 query 兜底，链路不中断。"""

    class EmptyPlanner(FakePlanner):
        async def plan(self, ir, telemetry=None):
            return []

    fake_oa = FakeOpenAlex()
    engine = SearchEngine(parser=FakeParser(), planner=EmptyPlanner())
    engine.openalex = fake_oa
    results, telemetry, _ = _run(engine.search_b1("PointNet 3D classification"))
    assert fake_oa.calls == ["PointNet 3D classification"]
    assert results  # 不中断


# ------------------------------------------------------------------ B3 精排链路
class FakeReranker:
    def __init__(self, result: list | None, empty: bool = False):
        self._result = result or []
        self._empty = empty
        self.calls: list[tuple] = []

    async def rerank(self, query, ir, candidates, telemetry=None):
        self.calls.append((query, ir, candidates))
        if self._empty:
            return []
        # 构造与候选对应的 RankResult（默认全 high，模拟精排）
        from src.schemas import RankResult
        return [RankResult(paper=ev.identity, score=0.95, label=RankLabel.HIGH) for ev in candidates]


def test_search_b3_reranks_and_returns():
    fake_oa = FakeOpenAlex()
    reranker = FakeReranker(result=[])
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=reranker)
    engine.openalex = fake_oa

    results, telemetry, _ = _run(engine.search_b3("PointNet 3D classification", top_k=10))
    assert results  # 精排结果返回
    assert telemetry.api_calls == 2  # 召回 2 次 API
    # 精排输入是词法粗筛后的候选（含 PointNet 对应 W1 和主题对应 W2）
    assert len(reranker.calls) == 1
    cand_ids = {c.identity.paper_id for c in reranker.calls[0][2]}
    assert "W1" in cand_ids and "W2" in cand_ids


def test_search_b3_rerank_failure_falls_back_to_lexical():
    fake_oa = FakeOpenAlex()
    reranker = FakeReranker(result=[], empty=True)  # 精排返回空 -> 回退词法
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=reranker)
    engine.openalex = fake_oa

    results, _, _ = _run(engine.search_b3("PointNet 3D classification", top_k=10))
    assert results  # 回退到词法排序仍有结果
    assert results[0].paper.paper_id == "W1"  # 词法排序 W1 最高


def test_search_b3_empty_recall_returns_empty():
    class EmptyOpenAlex(FakeOpenAlex):
        async def search(self, query, limit=20, telemetry=None):
            if telemetry:
                telemetry.add_api("openalex", 5.0)
            return []

    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=FakeReranker(result=[]))
    engine.openalex = EmptyOpenAlex()
    results, telemetry, _ = _run(engine.search_b3("nothing relevant"))
    assert results == []
    assert telemetry.api_calls == 2


# ------------------------------------------------------------------ B2 引文扩展
class FakeS2:
    def __init__(self, citations=None, references=None):
        self._citations = citations or []
        self._references = references or []
        self.calls: list[str] = []

    async def get_citations(self, paper_id, limit=10, telemetry=None):
        self.calls.append(f"cite:{paper_id}")
        return self._citations

    async def get_references(self, paper_id, limit=10, telemetry=None):
        self.calls.append(f"ref:{paper_id}")
        return self._references


def _mk_engine(s2, cfg=None):
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=FakeReranker(result=[]), cfg=cfg)
    engine.openalex = FakeOpenAlex()
    engine.s2 = s2
    engine.opencitations = FakeOC()      # 无 S2 key 时的兜底引文源（mock，不发真实请求）
    engine.crossref = FakeCrossref()
    return engine


class FakeOC:
    def __init__(self, citations=None, references=None):
        self._c = citations or []
        self._r = references or []
        self.calls: list[str] = []

    async def get_citations(self, doi, limit=10, telemetry=None):
        self.calls.append(f"cite:{doi}")
        return self._c

    async def get_references(self, doi, limit=10, telemetry=None):
        self.calls.append(f"ref:{doi}")
        return self._r


class FakeCrossref:
    async def get_by_doi(self, doi, telemetry=None):
        return PaperIdentity(paper_id=f"doi:{doi}", title=f"Enriched {doi}", doi=doi)


class FakeOpenAlexWithDoi(FakeOpenAlex):
    """召回候选带 DOI，使 seed 可被 S2 寻址（DOI:...）。"""

    async def search(self, query, limit=20, telemetry=None):
        if telemetry:
            telemetry.add_api("openalex", 5.0)
        self.calls.append(query)
        if "PointNet" in query:
            return [PaperEvidence(identity=PaperIdentity(paper_id="W1", title="PointNet 3D Point Cloud Classification Network", doi="10.1/pn"))]
        return [PaperEvidence(identity=PaperIdentity(paper_id="W2", title="3D Point Cloud Segmentation Survey", doi="10.1/seg", abstract="deep learning"))]


def test_search_b2_skipped_without_s2_key():
    """无 S2 key 时不调用 S2；无 DOI seed 时 OpenCitations 也不调用。"""
    s2 = FakeS2()
    engine = _mk_engine(s2)
    _run(engine.search_b2("PointNet 3D classification"))
    assert s2.calls == []  # 未调用 S2
    assert engine.opencitations.calls == []  # FakeOpenAlex 候选无 DOI -> 无 seed，不触发引文


def test_search_b2_opencitations_fallback_enriches_title():
    """无 S2 key：OpenCitations 引文候选并入，标题经 Crossref 补全。"""
    from config.settings import Settings
    cfg = Settings(semantic_scholar_api_key="")  # 显式清除：本测试意图验证无 S2 key 的 OpenCitations 兜底
    cite_ev = PaperEvidence(identity=PaperIdentity(paper_id="doi:10.99/c1", title="", doi="10.99/c1"))  # title 空 -> 需补全
    oc = FakeOC(citations=[cite_ev])
    engine = _mk_engine(FakeS2(), cfg=cfg)
    engine.openalex = FakeOpenAlexWithDoi()
    engine.opencitations = oc

    results, _, _ = _run(engine.search_b2("PointNet 3D classification", top_k=10))
    ids = [r.paper.paper_id for r in results]
    assert "doi:10.99/c1" in ids          # 引文候选并入
    assert oc.calls                       # OpenCitations 被调用
    enriched = next(r for r in results if r.paper.paper_id == "doi:10.99/c1")
    assert enriched.paper.title == "Enriched 10.99/c1"  # Crossref 补全标题


def test_search_b2_expands_and_dedupes():
    """有 S2 key 时，seed 的引文候选并入结果且去重。"""
    from config.settings import Settings
    cfg = Settings(semantic_scholar_api_key="sk-test")
    cite_ev = PaperEvidence(identity=PaperIdentity(paper_id="W9", title="PointNet++ hierarchical"), abstract="deep learning")
    ref_ev = PaperEvidence(identity=PaperIdentity(paper_id="W1", title="PointNet 3D Point Cloud Classification Network"))  # 与召回重复
    s2 = FakeS2(citations=[cite_ev], references=[ref_ev])
    engine = _mk_engine(s2, cfg=cfg)
    engine.openalex = FakeOpenAlexWithDoi()

    results, telemetry, _ = _run(engine.search_b2("PointNet 3D classification", top_k=10))
    ids = [r.paper.paper_id for r in results]
    assert "W9" in ids          # 引文扩展的新候选
    assert ids.count("W1") == 1  # 跨来源去重
    assert len(s2.calls) == 4    # 2 个 seed × (citations + references)
    assert sum(1 for c in s2.calls if c.startswith("cite:")) == 2
    assert sum(1 for c in s2.calls if c.startswith("ref:")) == 2


def test_search_b3_no_citation_expansion():
    """B3 不含引文扩展（引文扩展是 B2 的特性）。"""
    from config.settings import Settings
    cfg = Settings(semantic_scholar_api_key="sk-test")
    s2 = FakeS2(citations=[PaperEvidence(identity=PaperIdentity(paper_id="W9", title="PointNet++ hierarchical"), abstract="x")])
    engine = _mk_engine(s2, cfg=cfg)
    engine.openalex = FakeOpenAlexWithDoi()
    results, _, _ = _run(engine.search_b3("PointNet 3D classification", top_k=10))
    assert "W9" not in [r.paper.paper_id for r in results]  # 不做引文扩展
    assert s2.calls == []


# ------------------------------------------------------------------ B4 缓存 + 预算
def test_search_b4_cache_reuses_subquery_recall():
    """同一子查询文本第二次调用命中缓存：cache_hits+1，API 调用不增加。"""
    fake_oa = FakeOpenAlex()
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=FakeReranker(result=[]))
    engine.openalex = fake_oa

    _run(engine.search_b4("PointNet 3D classification", top_k=10))
    assert fake_oa.calls == ["PointNet classification", "point cloud deep learning"]
    assert engine._recall_cache  # 已填充缓存

    fake_oa.calls.clear()
    telemetry2 = Telemetry()
    results, telemetry2, _ = _run(engine.search_b4("PointNet 3D classification", top_k=10))
    assert fake_oa.calls == []          # 全部命中缓存，无新 API 调用
    assert telemetry2.cache_hits == 2   # 2 个子查询都命中
    assert results                     # 结果仍正常


def test_search_b4_budget_early_stops_api():
    """API 预算极小（1）时：只允许 1 次调用，其余子查询跳过。"""
    from config.settings import Settings
    cfg = Settings(budget_max_api_calls=1)
    fake_oa = FakeOpenAlex()
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=FakeReranker(result=[]), cfg=cfg)
    engine.openalex = fake_oa

    results, telemetry, _ = _run(engine.search_b4("PointNet 3D classification", top_k=10))
    assert len(fake_oa.calls) == 1      # 预算 1：只执行 1 个子查询召回
    assert telemetry.api_calls == 1
    assert results  # 不崩溃


def test_search_b4_skips_rerank_when_token_budget_exhausted():
    """token 预算耗尽时跳过 LLM 精排（省 token），返回词法结果。"""
    from config.settings import Settings
    cfg = Settings(budget_max_tokens_per_query=10)
    reranker = FakeReranker(result=[])
    fake_oa = FakeOpenAlex()
    engine = SearchEngine(parser=FakeParser(), planner=FakePlanner(), reranker=reranker, cfg=cfg)
    engine.openalex = fake_oa

    results, _, _ = _run(engine.search_b4("PointNet 3D classification", top_k=10))
    assert reranker.calls == []          # 未调用精排
    assert [r.paper.paper_id for r in results] == ["W1", "W2"]  # 词法排序结果


# ------------------------------------------------------------------ 可插拔召回源（RECALL_SOURCE）
def test_recall_source_default_openalex():
    from config.settings import Settings
    engine = SearchEngine(cfg=Settings())
    assert engine.recall.name == "openalex"
    assert engine.recall is engine.openalex


def test_recall_source_crossref():
    from config.settings import Settings
    engine = SearchEngine(cfg=Settings(recall_source="crossref"))
    assert engine.recall is engine.crossref
    assert engine.recall.name == "crossref"


def test_recall_source_s2():
    from config.settings import Settings
    engine = SearchEngine(cfg=Settings(recall_source="s2"))
    assert engine.recall is engine.s2
    assert engine.recall.name == "semantic_scholar"


def test_recall_source_invalid_raises():
    from config.settings import Settings
    import pytest
    with pytest.raises(ValueError):
        SearchEngine(cfg=Settings(recall_source="bogus"))


def test_build_rerank_pool_preserves_assoc():
    """联想词召回候选（source=assoc）不受 B3_LEX_PREKEEP 词法粗筛截断，保送进精排池。"""
    ids = [PaperIdentity(paper_id=f"W{i}", title=f"title {i}") for i in range(50)]
    evs = [PaperEvidence(identity=ids[i], source="search") for i in range(50)]
    # 第 48 个是联想词召回（词法排名靠后，不在 top-B3_LEX_PREKEEP 内）
    evs[48] = PaperEvidence(identity=ids[48], source="assoc")
    lex_ranked = [RankResult(paper=ids[i], score=0.5) for i in range(10)]
    pool = engine._build_rerank_pool(evs, lex_ranked)
    pool_ids = [p.identity.paper_id for p in pool]
    assert "W48" in pool_ids  # assoc 候选保送
    assert len(pool_ids) == 11  # 词法 top10 + 保送 1


def test_build_rerank_pool_dedup_assoc():
    """联想词召回与词法池重复时不去重重复项。"""
    ids = [PaperIdentity(paper_id=f"W{i}", title=f"title {i}") for i in range(5)]
    evs = [PaperEvidence(identity=ids[i], source="assoc" if i == 1 else "search") for i in range(5)]
    lex_ranked = [RankResult(paper=ids[1], score=0.9), RankResult(paper=ids[0], score=0.8)]
    pool = engine._build_rerank_pool(evs, lex_ranked)
    pool_ids = [p.identity.paper_id for p in pool]
    assert pool_ids.count("W1") == 1  # 已在词法池的 assoc 不重复加入
    assert len(pool_ids) == 2
