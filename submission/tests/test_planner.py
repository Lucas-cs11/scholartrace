"""子查询分解器单元测试（mock LLM，不产生真实调用）。"""
import asyncio

import pytest

from src.llm import LLMError
from src.planner import ASSOC_INTENT, MAX_ASSOC_TERMS, SubQueryPlanner, fallback_subqueries
from src.schemas import QueryIR
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


def _ir(**kwargs) -> QueryIR:
    base = dict(raw_query="test query")
    base.update(kwargs)
    return QueryIR(**base)


# ------------------------------------------------------------------ LLM 路径
def test_plan_ok():
    llm = FakeLLM(result={"subqueries": [
        {"query_text": "PointNet 3D point cloud classification", "intent": "代表术语", "priority": 5},
        {"query_text": "point cloud deep learning", "intent": "主题覆盖", "priority": 2},
    ]})
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir(entities=["PointNet"])))
    assert len(subs) == 2
    assert subs[0].id == "sq1" and subs[0].priority == 5
    assert subs[1].id == "sq2" and subs[1].priority == 2
    assert subs[0].query_text == "PointNet 3D point cloud classification"


def test_plan_dedup_and_filter():
    llm = FakeLLM(result={"subqueries": [
        {"query_text": "  BERT  fine-tuning ", "intent": "a", "priority": 3},
        {"query_text": "bert fine-tuning", "intent": "b", "priority": 9},   # 重复（归一化后）+ 越界 priority
        {"query_text": "   ", "intent": "c", "priority": 1},               # 空文本 -> 丢弃
        {"query_text": "NLP", "intent": "d", "priority": -3},              # 负 priority -> 截到 1
    ]})
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir()))
    assert len(subs) == 2
    assert subs[0].priority == 3
    assert subs[1].priority == 1


def test_plan_empty_result_falls_back():
    telemetry = Telemetry()
    llm = FakeLLM(result={"subqueries": []})
    ir = _ir(entities=["PointNet"], methods=["deep learning"])
    subs = _run(SubQueryPlanner(llm=llm).plan(ir, telemetry=telemetry))
    assert subs and subs[0].query_text == "PointNet"  # 降级用 IR 拼凑
    assert telemetry.fallback_count == 1


# ------------------------------------------------------------------ 降级路径
def test_plan_llm_error_falls_back():
    telemetry = Telemetry()
    llm = FakeLLM(error=LLMError("boom"))
    ir = _ir(entities=["PointNet", "ShapeNet"], methods=["deep learning"])
    subs = _run(SubQueryPlanner(llm=llm).plan(ir, telemetry=telemetry))
    texts = [s.query_text for s in subs]
    assert "PointNet" in texts and "ShapeNet" in texts
    assert "deep learning" in texts
    assert telemetry.fallback_count == 1


def test_fallback_empty_ir_uses_raw_query():
    ir = _ir(raw_query="long query text that has no entities")
    subs = fallback_subqueries(ir)
    assert len(subs) == 1
    assert subs[0].query_text.startswith("long query text")
    assert subs[0].priority == 1


def test_prompt_passes_ir_json():
    llm = FakeLLM(result={"subqueries": [{"query_text": "x", "intent": "i", "priority": 1}]})
    ir = _ir(entities=["PointNet"], year_min=2018)
    _run(SubQueryPlanner(llm=llm).plan(ir))
    user_msg = llm.calls[0]["messages"][1]["content"]
    assert "PointNet" in user_msg
    assert "year_min" in user_msg  # IR 以 JSON 传给 LLM


# ------------------------------------------------------------------ 约束链（parent_constraint_ids）
def test_plan_parent_constraint_ids():
    llm = FakeLLM(result={"subqueries": [
        {"query_text": "PointNet classification", "intent": "a", "priority": 5, "parent_constraint_ids": [0, 1]},
        {"query_text": "point cloud", "intent": "b", "priority": 2},
    ]})
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir()))
    assert subs[0].parent_constraint_ids == [0, 1]
    assert subs[1].parent_constraint_ids == []  # 未指定 -> 空


def test_plan_parent_ids_filtered():
    """非法 parent ids（负数/非 int/字符串/bool）被过滤，重复去重。"""
    llm = FakeLLM(result={"subqueries": [
        {"query_text": "x", "intent": "i", "priority": 1,
         "parent_constraint_ids": [0, 0, -1, "2", 3, True, None]},
    ]})
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir()))
    assert subs[0].parent_constraint_ids == [0, 2, 3]  # -1/True/None/重复被过滤，"2"转 2


# ------------------------------------------------------------------ 联想论文名（assoc_terms）
def test_plan_assoc_terms_to_subqueries():
    """assoc_terms 转高优先级子查询：去重、空串丢弃、priority 5、id 前缀 assoc。"""
    llm = FakeLLM(result={
        "subqueries": [{"query_text": "video generation", "intent": "主题覆盖", "priority": 2}],
        "assoc_terms": [
            "InstructVideo human feedback",
            "  InstructVideo Human Feedback ",  # 归一化后重复
            "video diffusion reward gradients",
            "",                                  # 空串丢弃
            "DPOK",
        ],
    })
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir()))
    assoc = [s for s in subs if s.intent == ASSOC_INTENT]
    assert len(assoc) == 3
    assert assoc[0].query_text == "InstructVideo human feedback"
    assert assoc[0].priority == 5
    assert assoc[0].id == "assoc1"
    assert [s.query_text for s in assoc] == [
        "InstructVideo human feedback", "video diffusion reward gradients", "DPOK",
    ]
    # 常规子查询仍保留在返回里
    assert any(s.intent == "主题覆盖" for s in subs)


def test_plan_assoc_max_terms():
    """联想词超过 MAX_ASSOC_TERMS 时截断。"""
    llm = FakeLLM(result={"subqueries": [], "assoc_terms": [f"term{i}" for i in range(10)]})
    subs = _run(SubQueryPlanner(llm=llm).plan(_ir()))
    assert len(subs) == MAX_ASSOC_TERMS
    assert all(s.intent == ASSOC_INTENT for s in subs)


def test_plan_no_assoc_falls_back():
    """LLM 失败时降级 IR 拼凑，无联想词（无法联想）。"""
    telemetry = Telemetry()
    llm = FakeLLM(error=LLMError("boom"))
    ir = _ir(entities=["PointNet"])
    subs = _run(SubQueryPlanner(llm=llm).plan(ir, telemetry=telemetry))
    assert all(s.intent != ASSOC_INTENT for s in subs)
    assert telemetry.fallback_count == 1
