"""核心数据对象（contest 版）。

设计原则（来自开发计划 §4.1）：
- PaperIdentity / DOI / Authors / Venue / Year 一律来自学术 API，禁止 LLM 生成。
- 不确定项保留在 QueryIR.unknown，禁止模型擅自补齐。
- 每次搜索动作都要能回答"为什么搜"（SubQuery.intent / SearchTrace）。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RankLabel(str, Enum):
    HIGH = "high"        # 高度相关
    PARTIAL = "partial"  # 部分相关
    NO = "no"            # 不相关


# --------------------------------------------------------------------------
# L1 查询理解
# --------------------------------------------------------------------------
class QueryIR(BaseModel):
    """把自然语言约束显式化，避免后续检索漂移。"""

    raw_query: str
    topic: Optional[str] = None
    entities: list[str] = Field(default_factory=list)   # 关键实体（方法名/模型/工具）
    methods: list[str] = Field(default_factory=list)     # 方法约束
    datasets: list[str] = Field(default_factory=list)    # 数据/领域限定
    domain: Optional[str] = None
    year_min: Optional[int] = None                       # 时间范围
    year_max: Optional[int] = None
    venues: list[str] = Field(default_factory=list)      # venue 限定
    authors: list[str] = Field(default_factory=list)     # 作者限定
    exclusions: list[str] = Field(default_factory=list)  # 排除项
    must_constraints: list[str] = Field(default_factory=list)    # 硬约束
    should_constraints: list[str] = Field(default_factory=list)  # 软约束
    unknown: list[str] = Field(default_factory=list)     # 解析不确定项，禁止补齐


# --------------------------------------------------------------------------
# L2 查询规划
# --------------------------------------------------------------------------
class SubQuery(BaseModel):
    """一个可独立检索的子问题。"""

    id: str
    parent_constraint_ids: list[int] = Field(default_factory=list)
    query_text: str
    intent: str = ""              # 为什么搜：缺失的子问题 / 代表术语 / 引文链
    priority: int = 1             # 越大越优先
    budget: Optional[int] = None  # 该子问题允许的 API 调用上限


# --------------------------------------------------------------------------
# L4 论文统一身份
# --------------------------------------------------------------------------
class PaperIdentity(BaseModel):
    """论文身份与多 API 去重主键。身份字段禁止由 LLM 生成。"""

    paper_id: str                 # 主键（OpenAlex ID 或 DOI）
    title: str
    doi: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[int] = None
    source_ids: dict[str, str] = Field(default_factory=dict)  # {"openalex": "W...", "crossref": "...", "s2": "..."}


class PaperEvidence(BaseModel):
    """候选论文的证据与相关性依据，支持细粒度相关性与可追溯解释。"""

    identity: PaperIdentity
    abstract_scope: str = "abstract"    # none | abstract | fulltext，全文不可用时不伪装
    matched_constraints: list[str] = Field(default_factory=list)  # 命中了哪些约束
    citation_links: list[str] = Field(default_factory=list)       # references/citations 的 paper_id
    source: str = ""                    # 从哪个 API 召回
    abstract: Optional[str] = None      # 供重排使用的摘要原文（可选保留）


# --------------------------------------------------------------------------
# L7 搜索轨迹 / L8 输出
# --------------------------------------------------------------------------
class SearchTrace(BaseModel):
    """效率优化与失败分析的事实日志。"""

    round: int
    query: str
    api: str
    latency_ms: float = 0
    tokens: int = 0
    candidate_delta: int = 0    # 本轮新增候选数
    relevant_delta: int = 0     # 本轮新增相关数


class RankResult(BaseModel):
    """把排序结果和"匹配了什么约束"绑定。"""

    paper: PaperIdentity
    score: float
    label: RankLabel = RankLabel.NO
    reason: str = ""
    constraint_coverage: dict[str, str] = Field(default_factory=dict)  # constraint -> 证据片段


# --------------------------------------------------------------------------
# R1 搜索结果归纳整理（赛题功能点 4）
# --------------------------------------------------------------------------
class PaperGroup(BaseModel):
    """按主题/方法归纳的论文分组。"""

    name: str
    description: str = ""
    paper_ids: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    title: str = ""


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: str = "citation"   # citation | reference


class SearchSummary(BaseModel):
    """搜索结果归纳整理：查询总结 + 分组 + 关系图 + 总体结论。"""

    query_summary: str = ""          # 查询理解总结
    groups: list[PaperGroup] = Field(default_factory=list)      # 论文分组
    relation_graph: dict = Field(default_factory=dict)          # {"nodes": [...], "edges": [...]}
    overall_summary: str = ""         # 搜索总体结论


# --------------------------------------------------------------------------
# L9 运行报告（一次实验的可比较记录）
# --------------------------------------------------------------------------
class RunReport(BaseModel):
    experiment_id: str = "B0"
    query_id: str = ""
    raw_query: str = ""
    gold_ids: list[str] = Field(default_factory=list)
    predicted_ids: list[str] = Field(default_factory=list)
    precision: float = 0
    recall: float = 0
    f1: float = 0
    api_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    schema_valid: bool = True
    traces: list[SearchTrace] = Field(default_factory=list)
    created_at: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
