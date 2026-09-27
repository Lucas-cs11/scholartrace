"""QueryIR 解析器单元测试（mock LLM，不产生真实调用）。"""
import asyncio

import pytest

from src.llm import LLMError
from src.parser import QueryIRParser, _clean
from src.schemas import QueryIR
from src.telemetry import Telemetry


def _run(awaitable):
    """同步测试里跑 async parse（每个调用独立事件循环，线程安全）。"""
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


# ------------------------------------------------------------------ 解析映射
def test_parse_maps_fields():
    llm = FakeLLM(result={
        "topic": "3D point cloud understanding",
        "entities": ["PointNet"],
        "methods": ["deep learning"],
        "datasets": ["ShapeNet"],
        "domain": "computer vision",
        "year_min": 2017,
        "year_max": 2020,
        "venues": ["CVPR"],
        "authors": ["Qi"],
        "exclusions": ["survey"],
        "must_constraints": ["3D point cloud input"],
        "should_constraints": ["robustness"],
        "unknown": ["maybe transformer"],
    })
    ir = QueryIRParser(llm=llm).parse("3D point cloud classification using deep learning on ShapeNet after 2017")
    ir = _run(ir)
    assert ir.topic == "3D point cloud understanding"
    assert "PointNet" in ir.entities
    assert "ShapeNet" in ir.datasets
    assert ir.year_min == 2017
    assert ir.year_max == 2020
    assert "CVPR" in ir.venues
    assert "survey" in ir.exclusions
    assert len(ir.must_constraints) == 1
    assert "maybe transformer" in ir.unknown
    assert ir.raw_query.startswith("3D point cloud")


# ------------------------------------------------------------------ 清洗
def test_clean_drops_empty_and_invalid():
    raw = {
        "topic": "   ",                       # 空白 topic -> 丢弃
        "entities": [None, "", "PointNet", 123],  # 非字符串转 str，空值丢弃
        "datasets": [],                       # 空列表 -> 丢弃
        "year_min": "2018",                   # 字符串年份 -> int
        "year_max": False,                    # bool -> 丢弃
        "year_bad": "abc",                    # 非年份字段不在 schema，直接忽略
    }
    out = _clean(raw)
    assert "topic" not in out
    assert out["entities"] == ["PointNet", "123"]
    assert "datasets" not in out
    assert out["year_min"] == 2018
    assert "year_max" not in out


def test_parse_year_str_converted():
    llm = FakeLLM(result={"year_min": "2018", "year_max": "2022", "entities": ["A"]})
    ir = _run(QueryIRParser(llm=llm).parse("q"))
    assert ir.year_min == 2018
    assert ir.year_max == 2022


# ------------------------------------------------------------------ 降级
def test_parse_llm_error_falls_back():
    telemetry = Telemetry()
    llm = FakeLLM(error=LLMError("boom"))
    ir = _run(QueryIRParser(llm=llm).parse("some query", telemetry=telemetry))
    assert ir.raw_query == "some query"
    assert ir.entities == []
    assert telemetry.fallback_count == 1


def test_parse_invalid_types_cleaned_not_crash():
    """LLM 返回非法类型（dict/list/嵌套）时 _clean 防御：不崩溃、不产生 fallback。"""
    telemetry = Telemetry()
    llm = FakeLLM(result={"topic": {"nested": "dict"}, "entities": [{"x": 1}, "BERT"], "year_min": "abc"})
    ir = _run(QueryIRParser(llm=llm).parse("q", telemetry=telemetry))
    assert ir.raw_query == "q"
    assert ir.topic is None                # dict 类型被丢弃
    assert ir.entities == ["{'x': 1}", "BERT"]  # 非字符串被 str 化保留
    assert ir.year_min is None             # 非数字年份被丢弃
    assert telemetry.fallback_count == 0   # 清洗成功，无需降级


def test_parse_prompt_mentions_json_and_schema():
    llm = FakeLLM(result={})
    parser = QueryIRParser(llm=llm)
    _run(parser.parse("q"))
    assert llm.calls[0]["messages"][0]["role"] == "system"
    assert "unknown" in llm.calls[0]["messages"][0]["content"]
    assert "year_min" in llm.calls[0]["messages"][0]["content"]
