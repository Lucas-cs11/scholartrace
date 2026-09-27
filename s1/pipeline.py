"""S1 Contest Engine 统一 pipeline（FAST / DEEP）。

组合已验证模块：Query Planner(M3-R frozen) → Regular+Assoc Retrieval → OpenAlex →
Identity Normalization → Candidate Pool → Search Observation → [Optional Evidence-Guided Round 2
(M5A frozen planner)] → Candidate Merge → Production LLM Reranker → Structured Results → SearchTrace。

约束（S1 规格）：
- Production baseline = M3-R Preserve+Augment；Production reranker = M3-R LLM Reranker；不再修改算法。
- FAST：Round1 → Rerank → Results（无 Round 2）。
- DEEP：Round1 → SearchObservation → Round2 Planner（冻结 M5A prompt）→ Follow-up Retrieval → Merge → Rerank。
- 不实现 Gold-aware trigger；continue_reason 仅为 future adaptive trigger 留接口。
- Offline replay 不允许 silent fallback：缺失 frozen plan/cache → 明确报错。
- 本模块不 import eval/harness（无 Gold 泄漏）。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from config.settings import settings
from scripts.run_m5a import filter_followup

from src.adapters import OpenAlexAdapter
from src.llm import LLMClient
from src.observability.canonical import clean_doi, norm_title
from src.planner import ASSOC_INTENT
from src.ranker import LLMReranker
from src.schemas import PaperEvidence, QueryIR
from src.search import B1_MAX_SUBQUERIES, SearchEngine
from src.telemetry import Telemetry

from s1.schemas import (  # noqa: E402
    FollowUpRecord, Round1Observation, Round2Decision, S1Config, S1Result, S1SearchTrace, StructuredResult,
)

REPO = Path(__file__).resolve().parent.parent
FROZEN_PLAN = REPO / "eval/runs/m3r_append/m3r_query_plans.jsonl"
RECALL_CACHE = REPO / "eval/cache/m3r_append/recall_cache.jsonl"
M5A_PLANS = REPO / "eval/runs/m5a_two_round/m5a_round2_plans.jsonl"
M5A_R2_CACHE = REPO / "eval/runs/m5a_two_round/m5a_round2_recall_cache.jsonl"
M5A_META = REPO / "eval/runs/m5a_two_round/m5a_plan_meta.json"

VALID_FUP_SOURCES = {"gap", "entity", "terminology"}


def canonical_from_identity(ident) -> str:
    doi = clean_doi(ident.doi)
    if doi:
        return f"doi:{doi}"
    if ident.paper_id:
        return ident.paper_id
    t = norm_title(ident.title)
    return f"title_n:{t}" if t else "unknown"


def _tokens(q: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", q.lower()))


def _normq(q: str) -> str:
    return " ".join(q.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _executed_subs(plan: dict) -> list[dict]:
    subs = plan.get("subs", [])
    assoc = [s for s in subs if s.get("intent") == ASSOC_INTENT]
    regular = sorted((s for s in subs if s.get("intent") != ASSOC_INTENT),
                     key=lambda s: -s.get("priority", 0))[:B1_MAX_SUBQUERIES]
    return regular + assoc


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少冻结数据文件（offline 不允许 silent fallback）: {path}")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _paper_evidence(d: dict) -> PaperEvidence:
    return PaperEvidence(**d)


class ContestEngine:
    """统一参赛引擎。"""

    def __init__(self, cfg: S1Config | None = None):
        self.cfg = cfg or S1Config()
        self.engine = SearchEngine(enable_citation_expansion=self.cfg.citation_expansion,
                                   assoc_safepass=self.cfg.assoc_safepass)
        self.llm = LLMClient()
        self.openalex = OpenAlexAdapter(mailto=settings.openalex_mailto)
        self.reranker = LLMReranker(keep_threshold=self.cfg.reranker_keep_threshold,
                                    min_keep=self.cfg.reranker_min_keep,
                                    max_results=self.cfg.reranker_max_results)
        self._frozen_plans: dict[str, dict] = {}
        self._m5a_plans: dict[str, dict] = {}
        self._prompt_hash = ""
        self._load_frozen()

    def _load_frozen(self) -> None:
        for d in _load_jsonl(FROZEN_PLAN):
            self._frozen_plans[d["query_id"]] = d
        for d in _load_jsonl(M5A_PLANS):
            self._m5a_plans[d["query_id"]] = d
        if M5A_META.exists():
            self._prompt_hash = json.loads(M5A_META.read_text(encoding="utf-8")).get("prompt_hash", "")

    # ------------------------------------------------------------------
    def _find_frozen_plan(self, question: str, query_id: str | None) -> dict | None:
        if query_id and query_id in self._frozen_plans:
            return self._frozen_plans[query_id]
        for qid, plan in self._frozen_plans.items():
            if plan.get("query") == question:
                return plan
        return None

    # ------------------------------------------------------------------
    async def _round1(self, question: str, query_id: str | None, telemetry: Telemetry) -> tuple[QueryIR, list[PaperEvidence], list[dict], dict[str, str]]:
        """Round-1：plan（frozen/online）+ recall（cache/openalex）。返回 (ir, raw_evs, subs, query_text_by_canonical)。"""
        src_map: dict[str, str] = {}
        if self.cfg.offline:
            plan = self._find_frozen_plan(question, query_id)
            if plan is None:
                raise LookupError(
                    f"offline 模式缺少 frozen plan（question={question[:60]!r}）。"
                    f"offline replay 不允许 silent fallback，请提供 query_id 或改为 online。"
                )
            subs = _executed_subs(plan)
            recall = _load_jsonl(RECALL_CACHE)
            recall_map = {d["q"]: [_paper_evidence(e) for e in d["evs"]] for d in recall}
            merged: dict[str, PaperEvidence] = {}
            for s in subs:
                for ev in recall_map.get(s["query_text"], []):
                    cid = canonical_from_identity(ev.identity)
                    merged.setdefault(cid, ev)
                    src_map.setdefault(cid, s["query_text"])
            ir = QueryIR(**plan["ir"]) if plan.get("ir") else QueryIR(raw_query=question)
            return ir, list(merged.values()), subs, src_map
        # online：真实 Planner + OpenAlex 召回
        ir, evs = await self.engine._plan_and_recall(question, telemetry, [], use_cache=False)
        plan = self.engine._plan_cache.get(question, {})
        subs_raw = plan.get("subs", [])
        subs = [s if isinstance(s, dict) else s.model_dump() for s in subs_raw]
        for s in subs:
            for ev in evs:
                if ev.identity.title and _tokens(s["query_text"]) & _tokens(ev.identity.title or ""):
                    src_map.setdefault(canonical_from_identity(ev.identity), s["query_text"])
        return ir, evs, subs, src_map

    # ------------------------------------------------------------------
    @staticmethod
    def _build_pool(question: str, raw: list[PaperEvidence], engine: SearchEngine) -> list[PaperEvidence]:
        lex = engine._lexical_rank(question, raw)
        return engine._build_rerank_pool(raw, lex)

    # ------------------------------------------------------------------
    async def _round2(self, question: str, query_id: str | None, decision: Round2Decision,
                      raw1: list[PaperEvidence], src1: dict[str, str],
                      telemetry: Telemetry, budget_left: int) -> tuple[list[PaperEvidence], dict[str, tuple[int, str]], int]:
        """执行 follow-up 检索（offline cache / online openalex），返回 (round2_evs, 更新后 src_map, 剩余预算)。"""
        r2: dict[str, PaperEvidence] = {}
        r2_src: dict[str, str] = {}
        r2cache_map: dict[str, list[PaperEvidence]] = {}
        if self.cfg.offline:
            r2cache_map = {d["q"]: [_paper_evidence(e) for e in d["evs"]] for d in _load_jsonl(M5A_R2_CACHE)}
        raw1_canons = {canonical_from_identity(e.identity) for e in raw1}
        for fu in decision.followups:
            if not fu.status.startswith("keep") and fu.status != "executed":
                continue
            fq = fu.query
            if self.cfg.offline:
                if fq not in r2cache_map:
                    raise LookupError(f"offline 模式缺少 frozen round2 cache: {fq!r}（不允许 silent fallback）")
                evs = r2cache_map[fq]
            else:
                if budget_left <= 0:
                    fu.status = "budget"
                    break
                evs = await self.openalex.search(fq, limit=self.cfg.recall_per_subquery, telemetry=telemetry)
                budget_left -= 1
            fu.status = "executed"
            for ev in evs:
                cid = canonical_from_identity(ev.identity)
                if cid not in raw1_canons:
                    r2.setdefault(cid, ev)
                r2_src.setdefault(cid, fq)
        r2_new = list(r2.values())
        src_map: dict[str, tuple[int, str]] = {cid: (1, q) for cid, q in src1.items()}
        for cid, fq in r2_src.items():
            src_map[cid] = (2, fq)
        return r2_new, src_map, budget_left

    # ------------------------------------------------------------------
    async def search(self, question: str, query_id: str | None = None, mode: str | None = None,
                     bq: dict | None = None) -> S1Result:
        mode = mode or self.cfg.mode
        t0 = time.time()
        telemetry = Telemetry()
        trace = S1SearchTrace(original_question=question, planner_version=self.cfg.planner_version,
                              prompt_hash=self._prompt_hash)

        # ---- Round-1 ----
        ir, raw1, subs, src1 = await self._round1(question, query_id, telemetry)
        for s in subs:
            trace.generated_queries.append({"query": s["query_text"], "intent": s.get("intent", ""),
                                            "priority": s.get("priority", 0), "round": 1})
        trace.returned_paper_count = len(raw1)
        trace.deduplicated_candidate_count = len({canonical_from_identity(e.identity) for e in raw1})

        pool1 = self._build_pool(question, raw1, self.engine)
        obs = Round1Observation(
            evidence_papers=[{
                "E": f"E{i}", "title": ev.identity.title or "",
                "abstract": (ev.abstract or "")[:500],
                "year": ev.identity.year, "venue": ev.identity.venue or "",
                "pre_rerank_rank": i, "source_query": src1.get(canonical_from_identity(ev.identity), ""),
            } for i, ev in enumerate(pool1[: self.cfg.max_evidence], 1)],
            total_retrieved=len(raw1),
            deduplicated_candidates=trace.deduplicated_candidate_count,
        )
        trace.round1_observation = obs

        # ---- continue decision（FAST 停 / DEEP 继续；S1 不做 Gold-aware trigger）----
        decision = Round2Decision()
        if mode == "deep":
            decision.continue_search = True
            decision.continue_reason = "deep mode：深度研究模式执行 evidence-guided Round-2（future adaptive trigger 接口预留）"
            fups = []
            m5a = self._m5a_plans.get(query_id or "")
            if m5a and m5a.get("follow_up_queries"):
                round1_queries = [s["query_text"] for s in subs]
                for fu in m5a["follow_up_queries"][: self.cfg.max_followup]:
                    # 复用 M5A 的 follow-up 过滤：dup/generic → filtered，否则 keep（_round2 才执行 keep）
                    status = filter_followup(fu["query"], round1_queries)
                    fups.append(FollowUpRecord(query=fu["query"], source=fu.get("source", "gap"),
                                               reason=fu.get("reason", ""), status=status))
            else:
                raise LookupError(f"DEEP 缺少冻结 M5A plan（query_id={query_id!r}）；不允许 silent fallback")
            decision.followups = fups
        else:
            decision.continue_search = False
            decision.continue_reason = "fast mode：Round-1 → Rerank → Results"
        trace.round2_decision = decision

        # ---- Round-2 ----
        r2_evs, src_map, _ = await self._round2(question, query_id, decision, raw1, src1, telemetry, self.cfg.budget_max_new_searches)
        trace.newly_discovered_papers = [canonical_from_identity(e.identity) for e in r2_evs]

        # ---- Merge + pool ----
        merged: dict[str, PaperEvidence] = {canonical_from_identity(e.identity): e for e in raw1}
        for e in r2_evs:
            merged[canonical_from_identity(e.identity)] = e
        merged_evs = list(merged.values())
        lex = self.engine._lexical_rank(question, merged_evs)
        pool = self.engine._build_rerank_pool(merged_evs, lex)
        pool_canons = {canonical_from_identity(e.identity) for e in pool}
        for e in r2_evs:  # 保送 agent 发现的论文进入精排池（不被词法粗筛挤出）
            cid = canonical_from_identity(e.identity)
            if cid not in pool_canons:
                pool.append(e)
                pool_canons.add(cid)

        # ---- Rerank（production M3-R LLM Reranker）----
        ranked = await self.reranker.rerank(question, ir, pool, telemetry=telemetry)
        trace.reranker_calls = telemetry.llm_calls

        # ---- Structured results ----
        by_cid = {canonical_from_identity(e.identity): e for e in merged_evs}
        results: list[StructuredResult] = []
        for r in ranked[: self.cfg.top_k]:
            cid = canonical_from_identity(r.paper)
            ev = by_cid.get(cid)
            round_no, srcq = src_map.get(cid, (1, ""))
            results.append(StructuredResult(
                title=r.paper.title,
                authors=r.paper.authors,
                year=r.paper.year,
                venue=r.paper.venue,
                doi=r.paper.doi,
                openalex_id=r.paper.paper_id if str(r.paper.paper_id).startswith("W") else None,
                abstract=(ev.abstract if ev else None),
                relevance_score=r.score,
                relevance_label=r.label.value,
                relevance_explanation=r.reason,
                retrieval_round=round_no,
                source_query=srcq,
                canonical_id=cid,
            ))
        trace.final_papers = [r.canonical_id for r in results]
        trace.api_calls = telemetry.api_calls
        trace.llm_calls = telemetry.llm_calls
        trace.input_tokens = telemetry.input_tokens
        trace.output_tokens = telemetry.output_tokens
        trace.total_latency_ms = round((time.time() - t0) * 1000, 1)

        return S1Result(question=question, mode=mode, query_id=query_id or "", results=results, trace=trace)
