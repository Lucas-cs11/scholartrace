"""S1 Contest Engine 数据对象（FAST/DEEP 统一结构化输出 + SearchTrace）。

对应 prompt_1.md §5/§6：
- StructuredResult：每篇最终论文统一输出，metadata 一律来自学术数据，禁止 LLM 生成 title/author/DOI/year。
- S1SearchTrace：每次运行完整轨迹（original question → ... → final papers + total latency）。
- continue_reason 保留为未来 adaptive trigger 的接口（§4），S1 不实现 Gold-aware trigger。
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StructuredResult(BaseModel):
    """一篇最终论文的统一结构化输出（§5）。"""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    openalex_id: Optional[str] = None
    abstract: Optional[str] = None
    relevance_score: float = 0.0
    relevance_label: str = "no"          # high | partial | no
    relevance_explanation: str = ""      # 相关性解释（LLM 给，基于候选信息）
    retrieval_round: int = 1             # 1 | 2（round2 发现）
    source_query: str = ""               # 召回该论文的查询（round1 sub / round2 follow-up）
    canonical_id: str = ""


class FollowUpRecord(BaseModel):
    """一条 Round-2 follow-up 查询的轨迹。"""

    query: str = ""
    source: str = ""                     # gap | entity | terminology
    reason: str = ""
    status: str = "planned"              # planned | kept | filtered | executed | budget
    newly_discovered_papers: list[str] = Field(default_factory=list)


class Round1Observation(BaseModel):
    """Search Observation（§6 Round-1 observation）。"""

    evidence_papers: list[dict] = Field(default_factory=list)   # top-8 evidence snapshot
    total_retrieved: int = 0
    deduplicated_candidates: int = 0


class Round2Decision(BaseModel):
    """Agent 的 Round-2 决策（§6 Round-2 decision）。"""

    continue_search: bool = False
    continue_reason: str = ""            # 为 future adaptive trigger 保留接口
    followups: list[FollowUpRecord] = Field(default_factory=list)


class S1SearchTrace(BaseModel):
    """每次运行的完整 SearchTrace（§6）。"""

    original_question: str = ""
    planner_version: str = ""
    prompt_hash: str = ""
    generated_queries: list[dict] = Field(default_factory=list)  # {query, intent, priority, round}
    api_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    returned_paper_count: int = 0
    deduplicated_candidate_count: int = 0
    round1_observation: Round1Observation = Field(default_factory=Round1Observation)
    round2_decision: Round2Decision = Field(default_factory=Round2Decision)
    newly_discovered_papers: list[str] = Field(default_factory=list)
    reranker_calls: int = 0
    final_papers: list[str] = Field(default_factory=list)
    total_latency_ms: float = 0.0


class S1Config(BaseModel):
    """运行配置（fast/deep）。"""

    mode: str = "fast"                   # fast | deep
    top_k: int = 20
    max_evidence: int = 8
    max_followup: int = 3
    budget_max_new_searches: int = 66
    recall_per_subquery: int = 20
    planner_version: str = "m5a-planner-v1"
    offline: bool = True                 # True=只读 frozen plan/cache，缺失即报错；False=在线 Planner+OpenAlex
    assoc_safepass: bool = True
    citation_expansion: bool = False
    # production 冻结参数（来自 M4 封板，不改）
    reranker_keep_threshold: float = 0.35
    reranker_min_keep: int = 3
    reranker_max_results: int = 20
    lex_prekeep: int = 40


class S1Result(BaseModel):
    """一次运行的完整输出。"""

    question: str = ""
    mode: str = "fast"
    query_id: str = ""
    results: list[StructuredResult] = Field(default_factory=list)
    trace: S1SearchTrace = Field(default_factory=S1SearchTrace)
