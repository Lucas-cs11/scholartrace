"""B3 二阶段精排器：LLM 对宽召回候选做相关性精排 + 动态截断。

动机（B1 结论）：词法排序在 top20 混入大量不相关候选，单篇 gold 的 query
被 precision 天花板（1/20=0.05）卡死。LLM 精排能判断语义相关性并过滤不相关项。

设计：
- 每批 batch_size 篇，abstract 截断，控制 token。
- LLM 只做相关性判断（输出 score/label/reason），论文身份一律来自候选，不生成事实。
- 动态截断：按 LLM 分数排序后，保留 score >= keep_threshold 的项；但至少保留
  min_keep 篇（兜底防全滤掉），最多 max_results 篇（评测 top_k）。
- LLM 失败 / 输出非法 / 未覆盖的候选：降级为词法顺序，排在有 LLM 分数的之后。
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import PaperEvidence, QueryIR, RankLabel, RankResult
from src.telemetry import Telemetry

SYSTEM_PROMPT = """你是学术论文检索系统的相关性精排器。判断候选论文与查询的相关程度，输出结构化 JSON。

查询解析约束（JSON）：
{ir_json}

候选论文列表（每篇含 paper_id / title / authors / venue / year / abstract）：
{items}

判断规则：
1. 三档相关性：high = 高度相关（直接命中主题/方法/实体）；partial = 部分相关（相关但非核心）；no = 不相关。
2. 硬约束 must_constraints 未满足的不能标 high；exclusions 命中的必须标 no。
3. 只依据提供的论文信息判断，不要编造论文内容。
4. score 为 0.0-1.0 连续相关度：high >= 0.7，partial 0.3-0.7，no < 0.3。
5. 每篇都给出判断，不要遗漏。
6. constraints 字段列出该论文明确命中的 must/should 约束（原样引用约束文本）；未命中任何约束或约束列表为空则省略该字段。

输出 JSON 对象（每个 paper_id 一条）：
{{"scores": [{{"paper_id": "...", "score": 0.0, "label": "high", "reason": "一句话理由", "constraints": ["must:<约束原文>", "should:<约束原文>"]}}]}}
只输出 JSON 对象本身，不要 markdown 代码块。"""

# 截断 / 保底策略
KEEP_THRESHOLD = 0.35      # score 低于此值视为不相关，截断
MIN_KEEP = 3               # 至少保留的篇数（防全滤掉）
MAX_RESULTS = 20           # 最多返回（评测 top_k）


def _truncate(text: str | None, n: int = 200) -> str:
    text = (text or "").strip()
    return text[:n] + ("..." if len(text) > n else "")


def _label_map(label: str) -> RankLabel:
    m = {"high": RankLabel.HIGH, "partial": RankLabel.PARTIAL, "no": RankLabel.NO}
    return m.get((label or "").strip().lower(), RankLabel.NO)


class LLMReranker:
    """LLM 相关性精排。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        batch_size: int = 15,
        max_abstract_chars: int = 200,
        keep_threshold: float = KEEP_THRESHOLD,
        min_keep: int = MIN_KEEP,
        max_results: int = MAX_RESULTS,
        use_tldr: bool = False,  # 方向4：S2 tldr 语义特征进精排（默认关，行为与基线一致）
    ):
        self.llm = llm or LLMClient(tier="strong")
        self.batch_size = batch_size
        self.max_abstract_chars = max_abstract_chars
        self.keep_threshold = keep_threshold
        self.min_keep = min_keep
        self.max_results = max_results
        self.use_tldr = use_tldr

    # ------------------------------------------------------------------
    async def rerank(
        self,
        query: str,
        ir: QueryIR,
        candidates: list[PaperEvidence],
        telemetry: Telemetry | None = None,
    ) -> list[RankResult]:
        """精排候选。返回按相关性降序的 RankResult 列表（已动态截断）。"""
        if not candidates:
            return []

        # 每批打分（串行，稳定优先）
        all_scores: dict[str, dict[str, Any]] = {}
        fallback_order: list[tuple[int, PaperEvidence]] = []
        scored_any = False
        for i, batch in enumerate(self._chunks(candidates, self.batch_size)):
            batch_scores = await self._score_batch(query, ir, batch, telemetry)
            if batch_scores:
                scored_any = True
            for ev in batch:
                s = batch_scores.get(ev.identity.paper_id)
                if s is not None:
                    all_scores[ev.identity.paper_id] = s
                else:
                    fallback_order.append((i, ev))  # 未覆盖 -> 词法兜底

        # 全部打分失败：返回空，由上层（search_b3）回退到词法排序
        if not scored_any:
            return []

        # 有 LLM 分数的按分数降序
        ranked: list[RankResult] = []
        for pid, s in sorted(all_scores.items(), key=lambda kv: -kv[1]["score"]):
            ranked.append(
                RankResult(
                    paper=next(ev.identity for ev in candidates if ev.identity.paper_id == pid),
                    score=round(float(s["score"]), 4),
                    label=s["label"],
                    reason=s["reason"],
                    constraint_coverage=s.get("coverage") or {},
                )
            )
        # 未覆盖的按词法顺序排在最后
        for _, ev in sorted(fallback_order, key=lambda x: x[0]):
            ranked.append(RankResult(paper=ev.identity, score=0.0, label=RankLabel.NO, reason="LLM 未覆盖"))

        # 动态截断：score >= 阈值 或 保底 min_keep，上限 max_results
        kept = [r for r in ranked if r.score >= self.keep_threshold]
        if len(kept) < self.min_keep and ranked:
            kept = ranked[:self.min_keep]
        return kept[:self.max_results]

    # ------------------------------------------------------------------
    async def _score_batch(
        self,
        query: str,
        ir: QueryIR,
        batch: list[PaperEvidence],
        telemetry: Telemetry | None,
    ) -> dict[str, dict[str, Any]]:
        """对一批候选打分。失败返回空 dict（该批降级为词法兜底）。"""
        items = []
        for ev in batch:
            ident = ev.identity
            lines = (
                f"[{ident.paper_id}]\n"
                f"  title: {ident.title}\n"
                f"  authors: {', '.join(ident.authors[:3]) or 'N/A'}\n"
                f"  venue: {ident.venue or 'N/A'}, year: {ident.year or 'N/A'}\n"
                f"  abstract: {_truncate(ev.abstract, self.max_abstract_chars)}"
            )
            # 方向4：S2 tldr 语义摘要（仅 use_tldr 时注入，弥补长摘要截断/无摘要候选的信号缺失）
            if self.use_tldr:
                tldr = (ident.source_ids or {}).get("s2_tldr")
                if tldr:
                    lines += f"\n  tldr: {_truncate(tldr, 160)}"
            items.append(lines)
        prompt = SYSTEM_PROMPT.format(ir_json=ir.model_dump_json(exclude_none=True), items="\n".join(items))
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"查询: {query}\n请判断以上 {len(batch)} 篇候选论文与查询的相关性。"},
                ],
                max_tokens=1500,
                telemetry=telemetry,
                note=f"rerank_batch_{len(batch)}",
            )
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"rerank: {type(e).__name__}")
            return {}

        scores: dict[str, dict[str, Any]] = {}
        for item in data.get("scores") or []:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("paper_id", "")).strip()
            if not pid:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            score = max(0.0, min(1.0, score))
            # F3 证据链：命中的 must/should 约束 -> constraint_coverage
            coverage: dict[str, str] = {}
            for c in item.get("constraints") or []:
                if isinstance(c, str) and c.strip():
                    coverage[c.strip()] = "命中"
            scores[pid] = {
                "score": score,
                "label": _label_map(str(item.get("label", "no"))),
                "reason": str(item.get("reason", "")).strip(),
                "coverage": coverage,
            }
        return scores

    # ------------------------------------------------------------------
    @staticmethod
    def _chunks(items: list, n: int) -> list[list]:
        return [items[i : i + n] for i in range(0, len(items), n)]
