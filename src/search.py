"""搜索编排器。

B0：单 query -> OpenAlex 宽召回 -> 词法相关度排序 -> Top-K。
B1：Round0 查询理解（QueryIR 解析 + 子查询分解）-> Round1 子查询并行宽召回
     -> 去重 -> 词法排序 -> Top-K。
B3：B1 召回后 -> 词法粗筛 -> LLM 二阶段精排 -> 动态截断（突破 precision 天花板）。
后续 B2+ 在此扩展：引文扩展、预算控制、早停。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from config.settings import Settings, settings as _settings

from src.adapters import ArxivAdapter, CrossrefAdapter, OpenAlexAdapter, OpenCitationsAdapter, SemanticScholarAdapter
from src.observability.response_cache import ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.parser import QueryIRParser
from src.planner import ASSOC_INTENT, PLANNER_VERSION, SubQueryPlanner
from src.ranker import LLMReranker
from src.schemas import PaperEvidence, QueryIR, RankLabel, RankResult, SearchTrace, SubQuery
from src.summarizer import SearchSummarizer
from src.telemetry import Telemetry

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# B1 Round1 预算：子查询数上限 × 每子查询 OpenAlex 召回上限 × 并发上限
B1_MAX_SUBQUERIES = 5
B1_RECALL_PER_SUBQUERY = 20
B1_RECALL_CONCURRENCY = 2  # 限并发避免触发 OpenAlex 429

# B3 精排预算：词法粗筛保留的候选数（再交给 LLM 精排）
B3_LEX_PREKEEP = 40

# B2 引文扩展预算：seed 数 × 每个 seed 前向/后向引文数
B2_SEED_COUNT = 5
B2_CITATIONS_PER_SEED = 15
B2_REFERENCES_PER_SEED = 15

# OpenCitations 兜底（无 S2 key）：每 seed 引文上限 + 每 query 标题补全上限
B2_OC_LIMIT = 8
B2_OC_ENRICH_LIMIT = 12


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class SearchEngine:
    def __init__(
        self,
        cfg: Settings | None = None,
        parser: QueryIRParser | None = None,
        planner: SubQueryPlanner | None = None,
        reranker: LLMReranker | None = None,
        recorder: TraceRecorder | None = None,
        response_cache: ResponseCache | None = None,
        enable_citation_expansion: bool = True,
        assoc_safepass: bool = True,
        citation_max_rounds: int = 2,  # 方向5：引文扩展深度
    ):
        cfg = cfg or _settings
        self.cfg = cfg
        self.recorder = recorder  # Phase 2 可选观测层；None 时行为与基线一致
        self.response_cache = response_cache
        self.openalex = OpenAlexAdapter(mailto=cfg.openalex_mailto, cache=response_cache)
        self.crossref = CrossrefAdapter(mailto=cfg.openalex_mailto, cache=response_cache)
        self.s2 = SemanticScholarAdapter(api_key=cfg.semantic_scholar_api_key, cache=response_cache)
        self.opencitations = OpenCitationsAdapter(cache=response_cache)
        # 主召回源（可插拔）：openalex（默认，配额受限）/ crossref（免配额）/ s2（无 key 1 req/s）
        if cfg.recall_source not in ("openalex", "crossref", "s2"):
            raise ValueError(f"未知 recall_source: {cfg.recall_source}（可选 openalex/crossref/s2）")
        self.parser = parser or QueryIRParser()
        self.planner = planner or SubQueryPlanner()
        self.reranker = reranker or LLMReranker()
        self.arxiv = ArxivAdapter()
        self.summarizer = SearchSummarizer()
        self.top_k = cfg.top_k
        # Phase 2 controlled-eval 开关：默认 True 保持 FULL 行为逐字节一致；
        # NO_CITATION eval 置 False 跳过 _expand_citations_multi（citation/reference/metadata 三阶段整体关闭）。
        # 纯 observability/开关层，不触碰任何冻结算法参数（top-k/prekeep/reranker/召回数量）。
        self.enable_citation_expansion = enable_citation_expansion
        # 方向5：引文扩展深度（默认2=两轮；提为3 探索引文-of-引文的第2跳）
        self.citation_max_rounds = citation_max_rounds
        # M2_QUERY_DENSITY 实验开关：False 时关闭 assoc 精排保送（_build_rerank_pool 不再把
        # source=="assoc" 的候选强制加入精排池），从而隔离「查询密度」与「保送保留」两个杠杆。
        # assoc 子查询的「不被 5 条截断」不受影响（_plan_and_recall 仍对 assoc 不截断）。
        self.assoc_safepass = assoc_safepass
        self._recall_cache: dict[str, list[PaperEvidence]] = {}  # subquery_text -> 候选（缓存复用）
        self._plan_cache: dict[str, dict] = {}  # query -> {"ir": ..., "subs": [...]}（评测一致性）

    @property
    def recall(self):
        """当前主召回适配器（动态解析，允许测试注入替换 openalex/crossref/s2）。"""
        return {
            "openalex": self.openalex,
            "crossref": self.crossref,
            "s2": self.s2,
        }[self.cfg.recall_source]

    # ------------------------------------------------------------------
    async def search(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """B0：单 query -> OpenAlex 召回 -> 词法排序 -> 去重 -> Top-K。"""
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        # Round 1: broad recall（召回源由 cfg.recall_source 决定）
        candidates = await self.recall.search(query, limit=50, telemetry=telemetry)
        traces.append(SearchTrace(round=1, query=query, api=self.cfg.recall_source, candidate_delta=len(candidates)))

        # 去重（按 paper_id / doi）
        unique: dict[str, PaperEvidence] = {}
        for ev in candidates:
            key = ev.identity.paper_id or ev.identity.doi
            if key and key not in unique:
                unique[key] = ev

        # 基础词法排序（B0 占位；B3 换成二阶段重排）
        ranked = self._lexical_rank(query, list(unique.values()))
        return ranked[:top_k], telemetry, traces

    def load_plan_cache(self, path: str | None) -> None:
        """加载子查询规划缓存（query -> ir+subs），保证多实验查询理解一致。

        版本不匹配的条目（planner 策略变更后）跳过，强制用新 planner 重计划。
        """
        if not path or not Path(path).exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("v") != PLANNER_VERSION:
                    continue
                self._plan_cache[d["query"]] = {"v": d["v"], "ir": d["ir"], "subs": d["subs"]}

    def save_plan_cache(self, path: str | None) -> None:
        """把子查询规划缓存落盘。"""
        if not path or not self._plan_cache:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for query, plan in self._plan_cache.items():
                f.write(json.dumps({"query": query, "v": PLANNER_VERSION, **plan}, ensure_ascii=False) + "\n")

    def load_recall_cache(self, path: str | None) -> None:
        """从磁盘加载召回缓存（jsonl: {q, evs}），供多实验复用省 OpenAlex 配额。"""
        if not path or not Path(path).exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self._recall_cache[d["q"]] = [PaperEvidence(**e) for e in d["evs"]]

    def save_recall_cache(self, path: str | None) -> None:
        """把本引擎的召回缓存落盘，供后续实验复用。"""
        if not path or not self._recall_cache:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for q, evs in self._recall_cache.items():
                f.write(
                    json.dumps({"q": q, "evs": [e.model_dump() for e in evs]}, ensure_ascii=False)
                    + "\n"
                )

    def _api_budget_exceeded(self, telemetry: Telemetry) -> bool:
        """API 调用预算超限（召回/S2 前检查）。"""
        return telemetry.api_calls >= self.cfg.budget_max_api_calls

    def _token_budget_exceeded(self, telemetry: Telemetry) -> bool:
        """token 预算超限（LLM 精排前检查）。"""
        return telemetry.total_tokens >= self.cfg.budget_max_tokens_per_query

    # ------------------------------------------------------------------
    async def _plan_and_recall(
        self,
        query: str,
        telemetry: Telemetry,
        traces: list[SearchTrace],
        use_cache: bool = False,
    ) -> tuple[QueryIR, list[PaperEvidence]]:
        """Round0 查询理解 + Round1 子查询并行宽召回 + 去重（B1/B2/B3/B4 共用）。

        use_cache=True（B4）时：重复子查询命中缓存；每次 API 调用前检查预算。
        """
        # 子查询规划缓存：评测时复用同一查询理解（保证跨实验一致 + 省 LLM）
        cached = self._plan_cache.get(query)
        if cached is not None:
            ir = QueryIR(**cached["ir"])
            subs = [SubQuery(**s) for s in cached["subs"]]
        else:
            ir = await self.parser.parse(query, telemetry=telemetry)
            subs = await self.planner.plan(ir, telemetry=telemetry)
            self._plan_cache[query] = {
                "v": PLANNER_VERSION,
                "ir": ir.model_dump(),
                "subs": [s.model_dump() for s in subs],
            }
        if not subs:
            subs = [SubQuery(id="sq1", query_text=query, intent="原始查询", priority=1)]
        # 预算控制：常规子查询只保留优先级最高的 B1_MAX_SUBQUERIES 个；
        # 联想论文名子查询（LLM 先验专名词）不受截断，全部参与召回（召回修复关键杠杆）。
        assoc_subs = [s for s in subs if s.intent == ASSOC_INTENT]
        regular = sorted(
            (s for s in subs if s.intent != ASSOC_INTENT), key=lambda s: -s.priority
        )[:B1_MAX_SUBQUERIES]
        subs = regular + assoc_subs
        traces.append(
            SearchTrace(round=0, query=query, api="llm_parse_plan", tokens=telemetry.total_tokens)
        )
        # Phase 2 observability：记录 planner 状态与 plan hash（frozen-plan 门禁）
        if self.recorder:
            plan_for_hash = cached if cached is not None else self._plan_cache.get(query, {})
            self.recorder.record_planner(
                status="cache_hit" if cached is not None else "generated",
                version=PLANNER_VERSION,
                plan=plan_for_hash,
                regular_count=len(regular),
                assoc_count=len(assoc_subs),
                cache_hit=cached is not None,
            )

        # Round 1: 子查询并行宽召回（限并发，避免触发 OpenAlex 限流）
        sem = asyncio.Semaphore(B1_RECALL_CONCURRENCY)

        async def _recall(sq: SubQuery) -> list[PaperEvidence]:
            async with sem:
                stage = "assoc_recall" if sq.intent == ASSOC_INTENT else "regular_recall"
                if use_cache:
                    cached = self._recall_cache.get(sq.query_text)
                    if cached is not None:
                        telemetry.add_cache_hit(f"subquery:{sq.query_text[:40]}")
                        if sq.intent == ASSOC_INTENT:
                            # 旧缓存可能无 assoc 标记（联想词功能前落盘），补标保证精排保送生效
                            cached = [ev.model_copy(update={"source": "assoc"}) for ev in cached]
                        # 方向1 多源：缓存即主源层（离线复跑/续跑时主源命中缓存不重调 API），
                        # 开多源则补一路实时 S2 合并——使"OpenAlex 缓存 + S2"离线端到端可跑。
                        if self.cfg.enable_s2_recall and self.cfg.semantic_scholar_api_key:
                            s2_evs = await self.s2.search(
                                sq.query_text, limit=B1_RECALL_PER_SUBQUERY, telemetry=telemetry)
                            s2_evs = [ev.model_copy(update={"source": "s2_recall"}) for ev in s2_evs]
                            cached = cached + s2_evs
                        if self.recorder:
                            self.recorder.record_api_call(
                                stage, self.cfg.recall_source, "search", sq.query_text,
                                candidates=cached, cache_hit=True,
                            )
                        return cached
                if use_cache and self._api_budget_exceeded(telemetry):
                    return []  # API 预算超限，跳过该子查询
                evs = await self.recall.search(sq.query_text, limit=B1_RECALL_PER_SUBQUERY, telemetry=telemetry)
                if sq.intent == ASSOC_INTENT:
                    # 联想词召回打 assoc 标记：精排池保送，避免专名命中被词法粗筛淹没
                    evs = [ev.model_copy(update={"source": "assoc"}) for ev in evs]
                # 方向1：多源召回——主源之外并行补一路 S2 search（仅 S2 key 就绪时）
                if self.cfg.enable_s2_recall and self.cfg.semantic_scholar_api_key:
                    s2_evs = await self.s2.search(
                        sq.query_text, limit=B1_RECALL_PER_SUBQUERY, telemetry=telemetry)
                    # S2 结果打 source=s2_recall（区别于主源/assoc，供归因与精排池保送决策）
                    s2_evs = [ev.model_copy(update={"source": "s2_recall"}) for ev in s2_evs]
                    evs = evs + s2_evs
                if self.recorder:
                    self.recorder.record_api_call(
                        stage, self.cfg.recall_source, "search", sq.query_text, candidates=evs,
                    )
                if use_cache:
                    self._recall_cache[sq.query_text] = evs
                return evs

        batches = await asyncio.gather(*(_recall(s) for s in subs))

        # 去重（按 paper_id / doi）
        unique: dict[str, PaperEvidence] = {}
        for i, cands in enumerate(batches):
            for ev in cands:
                key = ev.identity.paper_id or ev.identity.doi
                if key and key not in unique:
                    unique[key] = ev
            traces.append(
                SearchTrace(round=1, query=subs[i].query_text, api=self.cfg.recall_source, candidate_delta=len(cands))
            )
        if self.recorder:
            self.recorder.snapshot("after_raw_recall", list(unique.values()))
        return ir, list(unique.values())

    # ------------------------------------------------------------------
    async def _expand_citations(
        self,
        query: str,
        evs: list[PaperEvidence],
        telemetry: Telemetry,
        traces: list[SearchTrace],
        budgeted: bool = False,
    ) -> list[PaperEvidence]:
        """B2 Round2：从词法高分 seed 出发，引文图扩展候选（前向+后向）。

        S2 key 配置后走 S2（元数据完整）；无 key 时走 OpenCitations 兜底
        （免 key，标题经 Crossref 补全）。引文候选与原有合并去重。
        budgeted=True（B4）时每次调用前检查预算。
        """
        if not self.cfg.semantic_scholar_api_key:
            return await self._expand_citations_opencitations(query, evs, telemetry, traces, budgeted)

        lex = self._lexical_rank(query, evs)
        seeds: list[str] = []
        for r in lex[:B2_SEED_COUNT]:
            ident = r.paper
            key = ident.source_ids.get("s2") or (f"DOI:{ident.doi}" if ident.doi else "")
            if key and key not in seeds:
                seeds.append(key)
        if not seeds:
            return evs

        new: dict[str, PaperEvidence] = {}
        for key in seeds:
            if budgeted and self._api_budget_exceeded(telemetry):
                break  # API 预算超限，停止引文扩展
            for cands in (
                await self.s2.get_citations(key, limit=B2_CITATIONS_PER_SEED, telemetry=telemetry),
                await self.s2.get_references(key, limit=B2_REFERENCES_PER_SEED, telemetry=telemetry),
            ):
                for ev in cands:
                    pid = ev.identity.paper_id or ev.identity.doi
                    if pid:
                        new[pid] = ev
        if not new:
            return evs

        traces.append(
            SearchTrace(round=2, query=query, api="s2_citation", candidate_delta=len(new))
        )
        merged: dict[str, PaperEvidence] = {}
        for ev in evs + list(new.values()):
            pid = ev.identity.paper_id or ev.identity.doi
            if pid and pid not in merged:
                merged[pid] = ev
        return list(merged.values())

    # ------------------------------------------------------------------
    async def _expand_citations_opencitations(
        self,
        query: str,
        evs: list[PaperEvidence],
        telemetry: Telemetry,
        traces: list[SearchTrace],
        budgeted: bool,
    ) -> list[PaperEvidence]:
        """OpenCitations 兜底引文扩展（无 S2 key 时）。

        OpenCitations 只返回 DOI 关系，标题经 Crossref 补全（有限额，避免调用爆炸）。
        """
        lex = self._lexical_rank(query, evs)
        seeds = [r.paper for r in lex[:B2_SEED_COUNT] if r.paper.doi]
        if not seeds:
            return evs

        new: dict[str, PaperEvidence] = {}
        enrich_budget = B2_OC_ENRICH_LIMIT
        for ident in seeds:
            if budgeted and self._api_budget_exceeded(telemetry):
                break
            for cands in (
                await self.opencitations.get_citations(ident.doi, limit=B2_OC_LIMIT, telemetry=telemetry),
                await self.opencitations.get_references(ident.doi, limit=B2_OC_LIMIT, telemetry=telemetry),
            ):
                for ev in cands:
                    if not ev.identity.title and ev.identity.doi and enrich_budget > 0:
                        got = await self.crossref.get_by_doi(ev.identity.doi, telemetry=telemetry)
                        enrich_budget -= 1
                        if got:
                            ev.identity = got
                    if ev.identity.doi:
                        new[ev.identity.doi] = ev
        if not new:
            return evs

        traces.append(
            SearchTrace(round=2, query=query, api="opencitations_citation", candidate_delta=len(new))
        )
        merged: dict[str, PaperEvidence] = {}
        for ev in evs + list(new.values()):
            pid = ev.identity.paper_id or ev.identity.doi
            if pid and pid not in merged:
                merged[pid] = ev
        return list(merged.values())

    # ------------------------------------------------------------------
    def _high_relevance(self, query: str, evs: list[PaperEvidence]) -> int:
        """候选集中词法相关（PARTIAL 及以上）的篇数，作为收敛信号。"""
        return sum(1 for r in self._lexical_rank(query, evs) if r.label != RankLabel.NO)

    @staticmethod
    def _should_stop(round_num: int, max_rounds: int, prev_high: int, new_high: int, budget_exceeded: bool) -> bool:
        """FULL 早停决策：预算耗尽 / 达轮数上限 / 高相关候选无增长 / 高相关为 0。"""
        if budget_exceeded:
            return True
        if round_num >= max_rounds:
            return True
        if new_high == 0:
            return True
        if new_high <= prev_high:
            return True  # 高相关候选不再增长：收敛
        return False

    # ------------------------------------------------------------------
    async def _expand_citations_multi(
        self,
        query: str,
        evs: list[PaperEvidence],
        telemetry: Telemetry,
        traces: list[SearchTrace],
        budgeted: bool = False,
        max_rounds: int = 2,
    ) -> list[PaperEvidence]:
        """FULL：多轮引文扩展。

        每轮以当前词法高分、且未扩展过的候选为 seed，经 S2（有 key）/ OpenCitations
        （免 key）扩展后合并去重；迭代至轮数上限、无新 seed、无新增或预算耗尽。
        返回扩展后的候选集（含每轮 SearchTrace）。
        """
        current = list(evs)
        expanded: set[str] = set()
        enrich_budget = [B2_OC_ENRICH_LIMIT]  # 跨轮共享的 Crossref 补全预算

        def _seed_key(ident) -> str | None:
            """S2 可寻址 key（s2 id 或 DOI），无则 None。"""
            return ident.source_ids.get("s2") or (f"DOI:{ident.doi}" if ident.doi else "")

        async def _one_round(seeds: list) -> dict[str, PaperEvidence]:
            new: dict[str, PaperEvidence] = {}
            for ident in seeds:
                if budgeted and self._api_budget_exceeded(telemetry):
                    break
                if self.cfg.semantic_scholar_api_key and _seed_key(ident):
                    skey = _seed_key(ident)
                    cands_lists = (
                        await self.s2.get_citations(skey, limit=B2_CITATIONS_PER_SEED, telemetry=telemetry),
                        await self.s2.get_references(skey, limit=B2_REFERENCES_PER_SEED, telemetry=telemetry),
                    )
                    provider = "s2"
                    need_enrich = False
                elif ident.doi:
                    skey = ident.doi
                    cands_lists = (
                        await self.opencitations.get_citations(ident.doi, limit=B2_OC_LIMIT, telemetry=telemetry),
                        await self.opencitations.get_references(ident.doi, limit=B2_OC_LIMIT, telemetry=telemetry),
                    )
                    provider = "opencitations"
                    need_enrich = True
                else:
                    continue
                for i, cands in enumerate(cands_lists):
                    kind = "citations" if i == 0 else "references"
                    for ev in cands:
                        eid = ev.identity.paper_id or ev.identity.doi
                        if not eid:
                            continue
                        if need_enrich and not ev.identity.title and ev.identity.doi and enrich_budget[0] > 0:
                            got = await self.crossref.get_by_doi(ev.identity.doi, telemetry=telemetry)
                            enrich_budget[0] -= 1
                            if got:
                                ev.identity = got
                            if self.recorder:
                                self.recorder.record_api_call(
                                    "metadata", "crossref", "get_by_doi", ev.identity.doi,
                                    candidates=[ev],
                                )
                        new[eid] = ev
                    if self.recorder:
                        self.recorder.record_api_call(
                            "citation" if kind == "citations" else "reference",
                            provider, kind, skey, candidates=cands,
                        )
            return new

        prev_high = self._high_relevance(query, current)
        for rnd in range(max_rounds):
            if budgeted and self._api_budget_exceeded(telemetry):
                break
            lex = self._lexical_rank(query, current)
            seeds: list = []
            for r in lex[:B2_SEED_COUNT]:
                ident = r.paper
                pid = ident.paper_id
                if not pid or pid in expanded:
                    continue
                if _seed_key(ident) or ident.doi:  # 可寻址才做 seed
                    expanded.add(pid)
                    seeds.append(ident)
            if not seeds:
                break  # 无新 seed：收敛
            new = await _one_round(seeds)
            if not new:
                break  # 本轮无新增：收敛
            merged: dict[str, PaperEvidence] = {}
            for ev in current + list(new.values()):
                pid = ev.identity.paper_id or ev.identity.doi
                if pid and pid not in merged:
                    merged[pid] = ev
            current = list(merged.values())
            new_high = self._high_relevance(query, current)
            traces.append(
                SearchTrace(round=2 + rnd, query=query, api="citation_multi", candidate_delta=len(new))
            )
            if self.recorder:
                self.recorder.snapshot(f"after_citation_round{rnd}", current)
            # F2 早停：预算 / 轮数 / 高相关收敛
            if self._should_stop(rnd + 1, max_rounds, prev_high, new_high,
                                 budgeted and self._api_budget_exceeded(telemetry)):
                break
            prev_high = new_high
        return current

    # ------------------------------------------------------------------
    async def search_b1(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """B1：Round0 查询理解 -> Round1 子查询并行宽召回 -> 去重 -> 词法排序 -> Top-K。"""
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        _, evs = await self._plan_and_recall(query, telemetry, traces, use_cache=True)
        ranked = self._lexical_rank(query, evs)
        return ranked[:top_k], telemetry, traces

    # ------------------------------------------------------------------
    async def search_b2(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """B2：B1 召回 + S2 引文扩展 + 词法排序。"""
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        _, evs = await self._plan_and_recall(query, telemetry, traces, use_cache=True)
        evs = await self._expand_citations(query, evs, telemetry, traces)
        ranked = self._lexical_rank(query, evs)
        return ranked[:top_k], telemetry, traces

    # ------------------------------------------------------------------
    async def search_b3(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """B3：B1 召回 + S2 引文扩展 + 词法粗筛 -> LLM 精排 -> 动态截断。精排失败回退词法。"""
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        ir, evs = await self._plan_and_recall(query, telemetry, traces, use_cache=True)
        if not evs:
            return [], telemetry, traces

        # 词法粗筛（保留 top-B3_LEX_PREKEEP，控制精排 token 成本）
        lex_ranked = self._lexical_rank(query, evs)
        ev_by_id = {ev.identity.paper_id: ev for ev in evs}
        pre = self._build_rerank_pool(evs, lex_ranked)

        # LLM 精排（动态截断，突破 precision 天花板）
        ranked = await self.reranker.rerank(query, ir, pre, telemetry=telemetry)
        if not ranked:
            ranked = lex_ranked[:top_k]  # 精排失败回退词法排序
        return ranked[:top_k], telemetry, traces

    # ------------------------------------------------------------------
    async def search_b4(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """B4：B3 链路 + 召回缓存 + 预算早停（效率分优化）。

        相比 B3：子查询召回命中缓存（跨 query 复用）、每次 API/S2 调用前检查
        per-query 预算（api_calls / tokens），超限提前停止检索或精排。
        """
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        ir, evs = await self._plan_and_recall(query, telemetry, traces, use_cache=True)
        if not evs:
            return [], telemetry, traces

        lex_ranked = self._lexical_rank(query, evs)

        # 预算超限（token 已超或 API 调用过多）：跳过精排，直接词法返回省 token
        if self._api_budget_exceeded(telemetry) or self._token_budget_exceeded(telemetry):
            return lex_ranked[:top_k], telemetry, traces

        ev_by_id = {ev.identity.paper_id: ev for ev in evs}
        pre = self._build_rerank_pool(evs, lex_ranked)
        ranked = await self.reranker.rerank(query, ir, pre, telemetry=telemetry)
        if not ranked:
            ranked = lex_ranked[:top_k]
        return ranked[:top_k], telemetry, traces

    # ------------------------------------------------------------------
    async def search_full(self, query: str, top_k: int | None = None) -> tuple[list[RankResult], Telemetry]:
        """FULL 全链路：查询理解 + 子查询召回 + 多轮引文扩展（早停）+ LLM 精排（证据链）。

        相比 B4：Round2 起多轮引文扩展（F1）带早停收敛（F2），精排输出约束
        证据链（F3）。预算/token 超限时跳过精排回退词法。
        """
        top_k = top_k or self.top_k
        telemetry = Telemetry()
        traces: list[SearchTrace] = []

        ir, evs = await self._plan_and_recall(query, telemetry, traces, use_cache=True)
        if self.enable_citation_expansion:
            evs = await self._expand_citations_multi(query, evs, telemetry, traces,
                                                     budgeted=True,
                                                     max_rounds=self.citation_max_rounds)
        if not evs:
            return [], telemetry, traces

        lex_ranked = self._lexical_rank(query, evs)
        if self._api_budget_exceeded(telemetry) or self._token_budget_exceeded(telemetry):
            out = lex_ranked[:top_k]
            if self.recorder:
                out_evs = self._rank_results_to_evs(out, evs)
                self.recorder.snapshot("final_topk", out_evs)
                self.recorder.finalize(out_evs, drop_stage="lexical_topk_budget")
            return out, telemetry, traces

        ev_by_id = {ev.identity.paper_id: ev for ev in evs}
        pre = self._build_rerank_pool(evs, lex_ranked)
        if self.recorder:
            self.recorder.snapshot("rerank_pool", pre)
            self.recorder.record_pool(pre)
        ranked = await self.reranker.rerank(query, ir, pre, telemetry=telemetry)
        if not ranked:
            ranked = lex_ranked[:top_k]
        out = ranked[:top_k]
        if self.recorder:
            out_evs = self._rank_results_to_evs(out, evs)
            self.recorder.snapshot("final_reranked", out_evs)
            self.recorder.finalize(out_evs, drop_stage="reranker_truncation",
                                   drop_reason="below_keep_threshold_or_topk")
        return out, telemetry, traces

    @staticmethod
    def _rank_results_to_evs(results: list[RankResult], evs: list[PaperEvidence]) -> list[PaperEvidence]:
        """RankResult 列表 -> 原始 PaperEvidence 列表（供 recorder finalize 用）。"""
        by_id = {ev.identity.paper_id: ev for ev in evs}
        return [by_id[r.paper.paper_id] for r in results if r.paper.paper_id in by_id]

    # ------------------------------------------------------------------
    def _build_rerank_pool(self, evs: list[PaperEvidence], lex_ranked: list[RankResult]) -> list[PaperEvidence]:
        """精排候选池：词法 top-B3_LEX_PREKEEP + （可选）联想词召回候选保送。

        联想词子查询为 LLM 先验专名（如 "FinBen"、"Q-Align"），命中论文与
        query 词法重叠可能很低（benchmark/专名），会被 B3_LEX_PREKEEP 词法
        粗筛挤出精排池——专名命中价值不应被词法淹没，故保送进精排。
        当 assoc_safepass=False（M2_QUERY_DENSITY）时关闭保送，隔离该杠杆。
        """
        ev_by_id = {ev.identity.paper_id: ev for ev in evs}
        pre = [ev_by_id[r.paper.paper_id] for r in lex_ranked[:B3_LEX_PREKEEP] if r.paper.paper_id in ev_by_id]
        if self.assoc_safepass:
            in_pool = {ev.identity.paper_id for ev in pre}
            for ev in evs:
                if ev.source == "assoc" and ev.identity.paper_id not in in_pool:
                    pre.append(ev)
                    in_pool.add(ev.identity.paper_id)
        return pre

    # ------------------------------------------------------------------
    def _lexical_rank(self, query: str, candidates: list[PaperEvidence]) -> list[RankResult]:
        """词法相关度：查询 token 与 title(权重3)/abstract(权重1) 的重叠。"""
        q_tokens = _tokens(query)
        if not q_tokens:
            return [RankResult(paper=c.identity, score=0.0) for c in candidates]

        results = []
        for ev in candidates:
            title_t = _tokens(ev.identity.title or "")
            abs_t = _tokens(ev.abstract or "")
            title_hits = len(q_tokens & title_t)
            abs_hits = len(q_tokens & abs_t)
            score = (title_hits * 3.0 + abs_hits * 1.0) / (len(q_tokens) * 3.0 + 1e-9)
            label = RankLabel.HIGH if score >= 0.5 else (RankLabel.PARTIAL if score >= 0.25 else RankLabel.NO)
            results.append(
                RankResult(
                    paper=ev.identity,
                    score=round(score, 4),
                    label=label,
                    reason=f"title_hits={title_hits} abs_hits={abs_hits}",
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results
