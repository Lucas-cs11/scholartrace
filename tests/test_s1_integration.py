"""S1 Contest Engine 集成测试（全部离线、无网络、无 LLM）。

覆盖 S1 gate：
- planner schema / frozen plan 完整性
- identity 去重（canonical_from_identity）
- cache replay（round1 pool 离线重建）
- 无 silent fallback（offline 缺 plan/cache → LookupError）
- SearchTrace / StructuredResult schema
- Gold 泄漏（静态 + 运行时）
- FAST/DEEP offline smoke（mock reranker 避免 LLM）
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from s1.config import load_config
from s1.leakage import GoldLeakError, check_static, check_runtime_no_gold
from s1.pipeline import ContestEngine, FROZEN_PLAN, M5A_PLANS, RECALL_CACHE, M5A_R2_CACHE
from s1.schemas import S1Config, S1Result, S1SearchTrace, StructuredResult

REPO = Path(__file__).resolve().parent.parent

Q29 = "what role do zero-shot and few-shot prompting play in pre-training large language models, and how does prompt-based learning contribute?"
KNOWN_QID = "RealScholarQuery_29"

# 冻结输入 eval/cache/ 未随代码同步（源端 rsync 显式 --exclude=eval/cache），本机缺失。
# 缺失时整块跳过依赖它的离线重放测试，不放宽断言、不删除用例——文件补齐后自动恢复执行。
requires_recall_cache = pytest.mark.skipif(
    not Path(RECALL_CACHE).exists(),
    reason=f"冻结输入缺失：{RECALL_CACHE}（同步时被 --exclude=eval/cache 排除）",
)


def _noop_reranker(monkeypatch):
    """替换生产 LLM Reranker 为空操作，测试不触发网络/LLM。"""

    async def fake_rerank(self, question, ir, pool, telemetry=None):
        return []

    from src.ranker import LLMReranker
    monkeypatch.setattr(LLMReranker, "rerank", fake_rerank)
    return fake_rerank


# ------------------------------------------------------------------ 冻结数据完整性
def test_frozen_plan_integrity():
    plans = [json.loads(l) for l in Path(FROZEN_PLAN).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(plans) == 22
    for p in plans:
        assert p["query_id"]
        assert p["query"]
        assert isinstance(p.get("subs"), list) and p["subs"], f"{p['query_id']} 缺 subs"
    assert len({p["query_id"] for p in plans}) == 22  # query_id 唯一


def test_m5a_plan_integrity():
    plans = [json.loads(l) for l in Path(M5A_PLANS).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(plans) == 22
    for p in plans:
        fups = p.get("follow_up_queries", [])
        for fu in fups:
            assert fu["query"]
            assert fu["source"] in {"gap", "entity", "terminology"}


@requires_recall_cache
def test_recall_caches_load():
    r1 = [json.loads(l) for l in Path(RECALL_CACHE).read_text(encoding="utf-8").splitlines() if l.strip()]
    r2 = [json.loads(l) for l in Path(M5A_R2_CACHE).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert r1 and all(d["q"] for d in r1)
    assert r2 and all(d["q"] and d["evs"] for d in r2)
    # r1 允许少量空 evs（M5A OpenAlex 限流遗留）；但不能全部为空
    assert sum(1 for d in r1 if not d["evs"]) <= 1


# ------------------------------------------------------------------ identity 去重
def test_canonical_doi_precedence():
    from s1.pipeline import canonical_from_identity
    from src.schemas import PaperIdentity

    # DOI 优先于 paper_id / title_n
    a = PaperIdentity(paper_id="W1", doi="10.1000/xyz", title="Some Paper")
    assert canonical_from_identity(a) == "doi:10.1000/xyz"
    # 无 DOI → paper_id
    b = PaperIdentity(paper_id="W2", title="Another Paper")
    assert canonical_from_identity(b) == "W2"
    # 无 DOI/paper_id → title_n（keep_letters 归一化）
    c = PaperIdentity(paper_id="", title="Deep Learning for X: A Study!")
    assert canonical_from_identity(c) == "title_n:deeplearningforxastudy"


def test_identity_dedup_same_canonical_space():
    from s1.pipeline import canonical_from_identity
    from src.schemas import PaperIdentity

    # paper_id 空间：同一论文不同标题文本 → 同一 canonical
    a = PaperIdentity(paper_id="W99", title="Alpha")
    b = PaperIdentity(paper_id="W99", title="Alpha (2020)")
    assert canonical_from_identity(a) == canonical_from_identity(b) == "W99"
    # title_n 空间：仅标题（无 paper_id），大小写/符号差异 → 同一 canonical
    c = PaperIdentity(paper_id="", title="Beta Model")
    d = PaperIdentity(paper_id="", title="beta model!")
    assert canonical_from_identity(c) == canonical_from_identity(d)


# ------------------------------------------------------------------ cache replay（无网络）
@requires_recall_cache
def test_offline_round1_replay_pool(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="fast", offline=True)
    engine = ContestEngine(cfg)
    telemetry = _fake_telemetry()
    ir, evs, subs, src = asyncio.run(engine._round1(Q29, KNOWN_QID, telemetry))
    assert evs and telemetry.api_calls == 0  # offline：0 OpenAlex 调用
    assert subs and all(isinstance(s, dict) and s["query_text"] for s in subs)
    assert isinstance(src, dict)
    pool = asyncio.run(_build_pool(engine, Q29, evs))
    assert pool, "离线重建 pool 为空"


# ------------------------------------------------------------------ 无 silent fallback
def test_offline_missing_plan_raises(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="fast", offline=True)
    engine = ContestEngine(cfg)
    with pytest.raises(LookupError):
        asyncio.run(engine.search("some totally unknown question not in frozen set", query_id=None, mode="fast"))


@requires_recall_cache
def test_offline_missing_round2_cache_raises(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="deep", offline=True)
    engine = ContestEngine(cfg)
    # 构造一个不在 r2 cache 的 follow-up，验证 offline 拒绝而非静默降级
    from s1.schemas import FollowUpRecord, Round2Decision
    decision = Round2Decision(continue_search=True,
                              followups=[FollowUpRecord(query="definitely-not-in-cache-zzz-123", status="keep")])
    ir, evs, subs, src = asyncio.run(engine._round1(Q29, KNOWN_QID, _fake_telemetry()))
    with pytest.raises(LookupError):
        asyncio.run(engine._round2(Q29, KNOWN_QID, decision, evs, src, _fake_telemetry(),
                                   cfg.budget_max_new_searches))


# ------------------------------------------------------------------ schema
def test_searchtrace_schema():
    tr = S1SearchTrace(original_question="q", planner_version="v")
    assert tr.original_question == "q"
    assert tr.api_calls == 0 and tr.final_papers == []


def test_structured_result_schema():
    r = StructuredResult(title="T", doi="10.1/a", relevance_score=0.9, relevance_label="high")
    assert r.title == "T" and r.retrieval_round == 1
    assert r.authors == [] and r.canonical_id == ""


def test_s1result_wraps_trace():
    res = S1Result(question="q", mode="fast", results=[StructuredResult(title="T")])
    assert res.question == "q"
    assert isinstance(res.trace, S1SearchTrace)
    assert res.results[0].title == "T"


# ------------------------------------------------------------------ Gold 泄漏
def test_gold_leakage_static():
    assert check_static() == []


@requires_recall_cache
def test_gold_leakage_runtime(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="deep", offline=True)
    engine = ContestEngine(cfg)
    calls = []
    result = {}

    def run():
        result["res"] = asyncio.run(engine.search(Q29, query_id=KNOWN_QID, mode="deep"))

    violations = check_runtime_no_gold(run)
    assert violations == [], f"search 打开了 gold 文件: {violations}"


# ------------------------------------------------------------------ FAST/DEEP offline smoke
@requires_recall_cache
def test_fast_offline_smoke(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="fast", offline=True)
    engine = ContestEngine(cfg)
    res = asyncio.run(engine.search(Q29, query_id=KNOWN_QID, mode="fast"))
    assert isinstance(res, S1Result)
    assert res.mode == "fast"
    assert res.trace.returned_paper_count > 0
    assert res.trace.round2_decision.continue_search is False
    assert res.trace.newly_discovered_papers == []


@requires_recall_cache
def test_deep_offline_smoke(monkeypatch):
    _noop_reranker(monkeypatch)
    cfg = S1Config(mode="deep", offline=True)
    engine = ContestEngine(cfg)
    res = asyncio.run(engine.search(Q29, query_id=KNOWN_QID, mode="deep"))
    assert res.mode == "deep"
    assert res.trace.round2_decision.continue_search is True
    assert len(res.trace.round2_decision.followups) <= 3
    # round2 follow-ups 至少保留 keep 用于执行；offline 从 frozen cache 读取
    assert res.trace.llm_calls >= 0


# ------------------------------------------------------------------ 工具
def _fake_telemetry():
    from src.telemetry import Telemetry
    return Telemetry()


async def _build_pool(engine, question, evs):
    return engine._build_pool(question, evs, engine.engine)
