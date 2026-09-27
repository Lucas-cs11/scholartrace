"""QueryIR 解析器：自然语言查询 -> 结构化 QueryIR。

原则（开发计划 §4.1）：
- 论文事实（作者/标题/年份/venue）不由 LLM 生成——解析只负责把「用户约束」显式化。
- 不确定的约束进 QueryIR.unknown，禁止模型擅自补齐/编造。
- LLM 失败/输出非法时降级为仅含 raw_query 的 QueryIR（B0 行为），并计入 fallback。
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import QueryIR
from src.telemetry import Telemetry

SYSTEM_PROMPT = """你是学术论文检索系统的查询解析器。把用户的自然语言研究查询解析成结构化 JSON，用于后续论文检索。

输出 JSON 对象，字段如下（全部可选，没有把握的字段省略）：
- topic: str, 研究主题一句话概括
- entities: list[str], 关键实体（模型名/方法名/算法，如 "PointNet"、"BERT"）
- methods: list[str], 用户要求使用或对比的方法
- datasets: list[str], 数据或领域限定（如 "ShapeNet"、"ImageNet"）
- domain: str, 研究领域（如 "computer vision"、"NLP"）
- year_min / year_max: int, 明确给出的年份边界（"after 2018" 对应 year_min=2018）
- venues: list[str], 明确指定的会议/期刊（如 "NeurIPS"）
- authors: list[str], 明确指定的作者
- exclusions: list[str], 用户明确排除的内容（如 "not survey"、"不含综述"）
- must_constraints: list[str], 必须满足的硬约束，逐条列出
- should_constraints: list[str], 加分项软约束
- unknown: list[str], 原文出现但你无法确定如何归类的信息；不确定就放这里，不要编造

规则：
1. 只提取用户明确表达的信息，不要推断、不要补充常识。
2. 原文是英文就输出英文，中文就输出中文。
3. 只输出 JSON 对象本身，不要 markdown 代码块、不要任何解释。"""


def _clean(data: dict) -> dict:
    """清洗 LLM 输出：只保留非空合法字段，避免 pydantic 校验失败。"""
    out: dict[str, Any] = {}
    for key in ("topic", "domain"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            out[key] = v.strip()
    for key in (
        "entities", "methods", "datasets", "venues", "authors",
        "exclusions", "must_constraints", "should_constraints", "unknown",
    ):
        v = data.get(key)
        if isinstance(v, list):
            items = [str(x).strip() for x in v if x is not None and str(x).strip()]
            if items:
                out[key] = items
    for key in ("year_min", "year_max"):
        v = data.get(key)
        if isinstance(v, bool):  # JSON true/false 不是年份
            continue
        try:
            n = int(v) if v is not None else 0
        except (TypeError, ValueError):
            continue
        if n > 0:
            out[key] = n
    return out


class QueryIRParser:
    """LLM 驱动的 QueryIR 解析，失败自动降级为纯 raw_query。"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(tier="fast")

    async def parse(self, query: str, telemetry: Telemetry | None = None) -> QueryIR:
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                telemetry=telemetry,
                note="parse_queryir",
            )
            return QueryIR(raw_query=query, **_clean(data))
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"parse_queryir: {type(e).__name__}")
            return QueryIR(raw_query=query)
