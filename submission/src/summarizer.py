"""搜索结果归纳整理（R1，赛题功能点 4）。

对精排后的结果做三件事：
1. query_summary：查询意图总结
2. groups：按主题/方法把 top 论文分组归纳
3. overall_summary：搜索总体结论
关系图（relation_graph）由调用方传入引文链（source/target/kind）构建，论文事实不由 LLM 生成。
"""
from __future__ import annotations

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import GraphEdge, GraphNode, PaperGroup, QueryIR, RankResult, SearchSummary
from src.telemetry import Telemetry

SYSTEM_PROMPT = """你是学术论文检索系统的结果归纳器。对搜索结果做结构化归纳，输出 JSON。

查询：{query}
查询解析（JSON）：{ir_json}

Top 论文（paper_id / title / authors / venue / year / score / label / 命中约束）：
{items}

输出 JSON 对象：
{{
  "query_summary": "该查询的核心研究意图与检索范围的一句话总结",
  "groups": [
    {{"name": "主题/方法分组名", "description": "该组论文的共同点", "paper_ids": ["paper_id", ...]}}
  ],
  "overall_summary": "基于检索结果的总体结论：关键论文、覆盖哪些方面、检索是否充分"
}}

规则：
1. paper_ids 必须原样引用给出的 paper_id，不要编造论文。
2. groups 覆盖所有相关论文；无关论文可归入"其他"。
3. 查询为英文则输出英文，中文则输出中文。
4. 只输出 JSON 对象本身，不要 markdown 代码块。"""

# 传给 LLM 归纳的 top 论文数
SUMMARIZE_TOP = 15


class SearchSummarizer:
    """LLM 驱动的搜索结果归纳，失败降级为基础结构。"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(tier="fast")

    async def summarize(
        self,
        query: str,
        ir: QueryIR,
        results: list[RankResult],
        telemetry: Telemetry | None = None,
        relation_links: list[tuple[str, str, str]] | None = None,
    ) -> SearchSummary:
        """归纳结果。relation_links: (source, target, kind) 引文关系（论文事实，来自检索证据）。"""
        summary = self._build_graph(relation_links or [])
        if not results:
            summary.query_summary = query
            return summary

        items = self._format_items(results[:SUMMARIZE_TOP])
        prompt = SYSTEM_PROMPT.format(query=query, ir_json=ir.model_dump_json(exclude_none=True), items=items)
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"请归纳以上 {len(results[:SUMMARIZE_TOP])} 篇论文的搜索结果。"},
                ],
                max_tokens=1200,
                telemetry=telemetry,
                note="summarize_results",
            )
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"summarize: {type(e).__name__}")
            summary.query_summary = query
            return summary

        valid_ids = {r.paper.paper_id for r in results[:SUMMARIZE_TOP]}
        summary.query_summary = str(data.get("query_summary", query)).strip() or query
        summary.overall_summary = str(data.get("overall_summary", "")).strip()
        for g in data.get("groups") or []:
            if not isinstance(g, dict):
                continue
            name = str(g.get("name", "")).strip()
            if not name:
                continue
            ids = [str(p) for p in (g.get("paper_ids") or []) if str(p) in valid_ids]
            if not ids:
                continue
            summary.groups.append(
                PaperGroup(name=name, description=str(g.get("description", "")).strip(), paper_ids=ids)
            )
        return summary

    # ------------------------------------------------------------------
    @staticmethod
    def _build_graph(links: list[tuple[str, str, str]]) -> SearchSummary:
        nodes: dict[str, str] = {}
        edges: list[GraphEdge] = []
        for source, target, kind in links:
            nodes.setdefault(source, "")
            nodes.setdefault(target, "")
            edges.append(GraphEdge(source=source, target=target, kind=kind))
        if not edges:
            return SearchSummary()
        return SearchSummary(
            relation_graph={
                "nodes": [GraphNode(id=nid, title=title) for nid, title in nodes.items()],
                "edges": edges,
            }
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _format_items(results: list[RankResult]) -> str:
        lines = []
        for r in results:
            ident = r.paper
            cov = ",".join(r.constraint_coverage.keys()) or "-"
            lines.append(
                f"[{ident.paper_id}]\n"
                f"  title: {ident.title}\n"
                f"  authors: {', '.join(ident.authors[:3]) or 'N/A'}\n"
                f"  venue: {ident.venue or 'N/A'}, year: {ident.year or 'N/A'}\n"
                f"  score: {r.score}, label: {r.label.value}, constraints: {cov}"
            )
        return "\n".join(lines)
