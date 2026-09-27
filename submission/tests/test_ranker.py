"""B3 精排器单元测试（mock LLM，不产生真实调用）。"""
import asyncio

import pytest

from src.llm import LLMError
from src.ranker import LLMReranker, _label_map, _truncate
from src.schemas import PaperEvidence, PaperIdentity, QueryIR, RankLabel
from src.telemetry import Telemetry


def _run(awaitable):
    return asyncio.run(awaitable)


class FakeLLM:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error
        self.calls: list[dict] = []

    async def complete_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self._error:
            raise self._error
        return self._result


def _ev(pid: str, title: str, abstract: str = "") -> PaperEvidence:
    return PaperEvidence(identity=PaperIdentity(paper_id=pid, title=title), abstract=abstract)


def _ir(**kw) -> QueryIR:
    base = dict(raw_query="test")
    base.update(kw)
    return QueryIR(**base)


# ------------------------------------------------------------------ 基础映射
def test_truncate():
    assert _truncate("x" * 300, 200).endswith("...")
    assert _truncate("short") == "short"


def test_label_map():
    assert _label_map("high") == RankLabel.HIGH
    assert _label_map("PARTIAL") == RankLabel.PARTIAL
    assert _label_map("No") == RankLabel.NO
    assert _label_map("garbage") == RankLabel.NO


# ------------------------------------------------------------------ 正常打分排序
def test_rerank_orders_by_score_and_filters_below_threshold():
    llm = FakeLLM(result={"scores": [
        {"paper_id": "W1", "score": 0.95, "label": "high", "reason": "直接命中"},
        {"paper_id": "W2", "score": 0.50, "label": "partial", "reason": "部分相关"},
        {"paper_id": "W3", "score": 0.10, "label": "no", "reason": "不相关"},
    ]})
    cands = [_ev("W1", "A"), _ev("W2", "B"), _ev("W3", "C")]
    results = _run(LLMReranker(llm=llm, min_keep=1).rerank("q", _ir(), cands))
    assert [r.paper.paper_id for r in results] == ["W1", "W2"]  # W3 (0.10<0.35) 被截断
    assert results[0].label == RankLabel.HIGH
    assert results[0].reason == "直接命中"
    assert results[1].label == RankLabel.PARTIAL


def test_rerank_min_keep_prevents_empty():
    """全部低于阈值时，保底保留前 min_keep 篇。"""
    llm = FakeLLM(result={"scores": [
        {"paper_id": "W1", "score": 0.10, "label": "no", "reason": "r"},
        {"paper_id": "W2", "score": 0.05, "label": "no", "reason": "r"},
        {"paper_id": "W3", "score": 0.02, "label": "no", "reason": "r"},
    ]})
    cands = [_ev("W1", "A"), _ev("W2", "B"), _ev("W3", "C")]
    results = _run(LLMReranker(llm=llm, min_keep=2).rerank("q", _ir(), cands))
    assert len(results) == 2  # 保底 2 篇


def test_rerank_max_results_cap():
    llm = FakeLLM(result={"scores": [
        {"paper_id": f"W{i}", "score": 0.9, "label": "high", "reason": "r"} for i in range(1, 8)
    ]})
    cands = [_ev(f"W{i}", str(i)) for i in range(1, 8)]
    results = _run(LLMReranker(llm=llm, min_keep=1, max_results=3).rerank("q", _ir(), cands))
    assert len(results) == 3


def test_rerank_llm_score_overflow_clamped():
    llm = FakeLLM(result={"scores": [{"paper_id": "W1", "score": 99.0, "label": "high", "reason": "r"}]})
    cands = [_ev("W1", "A")]
    results = _run(LLMReranker(llm=llm, min_keep=1).rerank("q", _ir(), cands))
    assert results[0].score == 1.0


# ------------------------------------------------------------------ 降级路径
def test_rerank_uncovered_papers_fallback_to_end():
    """LLM 未覆盖的候选排在有分数之后，score=0（保底保留时）。"""
    llm = FakeLLM(result={"scores": [{"paper_id": "W2", "score": 0.8, "label": "high", "reason": "r"}]})
    cands = [_ev("W1", "A"), _ev("W2", "B"), _ev("W3", "C")]
    results = _run(LLMReranker(llm=llm, min_keep=3).rerank("q", _ir(), cands))
    assert results[0].paper.paper_id == "W2"
    assert results[0].score == 0.8
    tail = [r.paper.paper_id for r in results[1:]]
    assert "W1" in tail and "W3" in tail
    assert all(r.score == 0.0 for r in results[1:])


def test_rerank_llm_error_falls_back_to_empty():
    telemetry = Telemetry()
    llm = FakeLLM(error=LLMError("boom"))
    cands = [_ev("W1", "A"), _ev("W2", "B")]
    results = _run(LLMReranker(llm=llm).rerank("q", _ir(), cands, telemetry=telemetry))
    assert telemetry.fallback_count >= 1
    assert results == []  # 精排失败 -> 空（上层 search_b3 会回退到词法排序）


def test_rerank_empty_candidates():
    results = _run(LLMReranker(llm=FakeLLM(result={"scores": []})).rerank("q", _ir(), []))
    assert results == []


def test_rerank_batches_across_calls():
    """候选超过 batch_size 时分批调用，每批独立。"""
    llm = FakeLLM(result={"scores": [
        {"paper_id": f"W{i}", "score": 0.9, "label": "high", "reason": "r"} for i in range(1, 6)
    ]})
    cands = [_ev(f"W{i}", str(i)) for i in range(1, 6)]
    _run(LLMReranker(llm=llm, batch_size=2).rerank("q", _ir(), cands))
    assert len(llm.calls) == 3  # 5 篇 / 2 一批 = 3 批


# ------------------------------------------------------------------ F3 证据链（constraint_coverage）
def test_rerank_populates_constraint_coverage():
    """LLM 输出 constraints -> RankResult.constraint_coverage 填充。"""
    llm = FakeLLM(result={"scores": [
        {"paper_id": "W1", "score": 0.95, "label": "high", "reason": "r",
         "constraints": ["must:使用深度学习", "should:ShapeNet"]},
        {"paper_id": "W2", "score": 0.40, "label": "partial", "reason": "r",
         "constraints": ["should:ShapeNet"]},
    ]})
    cands = [_ev("W1", "A"), _ev("W2", "B")]
    results = _run(LLMReranker(llm=llm, min_keep=1).rerank("q", _ir(), cands))
    top = results[0]
    assert top.constraint_coverage == {"must:使用深度学习": "命中", "should:ShapeNet": "命中"}
    assert results[1].constraint_coverage == {"should:ShapeNet": "命中"}


def test_rerank_no_constraints_stays_empty():
    """LLM 未输出 constraints（兼容旧输出）-> coverage 空 dict。"""
    llm = FakeLLM(result={"scores": [
        {"paper_id": "W1", "score": 0.9, "label": "high", "reason": "r"},
    ]})
    cands = [_ev("W1", "A")]
    results = _run(LLMReranker(llm=llm, min_keep=1).rerank("q", _ir(), cands))
    assert results[0].constraint_coverage == {}


def test_rerank_invalid_constraints_ignored():
    """constraints 里的非字符串/空值被忽略。"""
    llm = FakeLLM(result={"scores": [
        {"paper_id": "W1", "score": 0.9, "label": "high", "reason": "r",
         "constraints": ["must:有效", None, 123, "  "]},
    ]})
    cands = [_ev("W1", "A")]
    results = _run(LLMReranker(llm=llm, min_keep=1).rerank("q", _ir(), cands))
    assert results[0].constraint_coverage == {"must:有效": "命中"}
