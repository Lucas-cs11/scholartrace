"""Phase 2 stage-level TraceRecorder：逐 query 记录 planner/API/funnel/gold lifecycle。

设计约束（Phase 2 用户明令）：
- 不改变生产检索逻辑：所有记录点都在 search.py 以 `if self.recorder:` 守卫存在，
  未传入 recorder 时行为与基线完全一致。
- Gold 只进诊断统计，绝不进搜索逻辑（gold-derived query 禁止进入正式系统）。
- 逻辑 API call（算法意图，record_api_call 一次）与物理 HTTP call（含重试，
  ResponseCache 按 logical_key 计数）分层记录。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.observability.canonical import canonical_paper_id, is_gold_by_title, norm_title
from src.observability.response_cache import ResponseCache, cache_key
from src.schemas import PaperEvidence


class TraceRecorder:
    def __init__(
        self,
        query_id: str,
        run_id: str,
        baseline_reference_id: str = "",
        response_cache: ResponseCache | None = None,
    ):
        self.query_id = query_id
        self.run_id = run_id
        self.baseline_reference_id = baseline_reference_id
        self._cache = response_cache

        self.api_calls: list[dict] = []
        self.candidate_snapshots: list[dict] = []
        self.gold_lifecycle: dict[str, dict] = {}  # canonical_id -> lifecycle dict

        self.planner_status: str = "unknown"
        self.planner_fallback_reason: str = ""
        self.planner_version: int = 0
        self.plan_hash: str = ""
        self.regular_subquery_count: int = 0
        self.assoc_subquery_count: int = 0
        self.plan_cache_hit: bool = False
        self.external_drift_detected: bool = False

        self._gold_title_ns: set[str] = set()
        self._cumulative_canonical: set[str] = set()
        self._ordered_canonical: list[str] = []

    # ------------------------------------------------------------------
    # gold 集合（只用于诊断统计，不参与搜索）
    # ------------------------------------------------------------------
    def set_gold_titles(self, gold_titles: list[str]) -> None:
        """设置 gold 论文标题（keep_letters 归一化集合）。"""
        self._gold_title_ns = {norm_title(t) for t in gold_titles if norm_title(t)}

    def set_gold_title_ns(self, gold_title_ns: set[str]) -> None:
        self._gold_title_ns = set(gold_title_ns)

    # ------------------------------------------------------------------
    def record_planner(
        self,
        status: str,
        version: int,
        plan: dict,
        regular_count: int,
        assoc_count: int,
        cache_hit: bool,
        fallback_reason: str = "",
    ) -> None:
        """记录 planner 状态与 plan hash（实验一致性门禁）。"""
        self.planner_status = status
        self.planner_version = version
        self.planner_fallback_reason = fallback_reason
        self.regular_subquery_count = regular_count
        self.assoc_subquery_count = assoc_count
        self.plan_cache_hit = cache_hit
        canonical = {"v": version, "ir": plan.get("ir", {}), "subs": plan.get("subs", [])}
        self.plan_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    # ------------------------------------------------------------------
    def _physical_attempts(self, ck: str | None) -> int:
        if self._cache is not None and ck:
            return self._cache.physical_attempts(ck)
        return 1

    def _add_gold_hits(self, evs: list[PaperEvidence], stage: str, query_or_seed: str, start_rank: int) -> tuple[int, int]:
        """登记 gold hits 的 lifecycle（first_seen 等）。返回 (gold_hit_count, new_gold_count)。"""
        hits = 0
        new_gold = 0
        for i, ev in enumerate(evs):
            if not is_gold_by_title(ev, self._gold_title_ns):
                continue
            cid = canonical_paper_id(ev)
            hits += 1
            if cid not in self.gold_lifecycle:
                new_gold += 1
                self.gold_lifecycle[cid] = {
                    "canonical_id": cid,
                    "title_n": norm_title(ev.identity.title),
                    "first_seen_stage": stage,
                    "first_seen_query": (query_or_seed or "")[:80],
                    "first_seen_rank": start_rank + i,
                    "survived_prekeep": None,
                    "pre_rerank_rank": None,
                    "reranker_rank": None,
                    "final_rank": None,
                    "drop_stage": None,
                    "drop_reason": None,
                }
        return hits, new_gold

    # ------------------------------------------------------------------
    def record_api_call(
        self,
        stage: str,
        provider: str,
        endpoint: str,
        query_or_seed: str,
        candidates: list[PaperEvidence] | None = None,
        cache_key_str: str | None = None,
        cache_hit: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        """记录一次逻辑 API 调用（一次算法意图，可能含多次物理重试）。

        candidates: 该调用的原始候选（PaperEvidence）。内部算 canonical 去重、
        gold 命中，并维护累积唯一集合。
        """
        candidates = candidates or []
        ck = cache_key_str or cache_key(provider, endpoint, query_or_seed)
        canon_ids = [canonical_paper_id(ev) for ev in candidates]
        new_unique = [c for c in canon_ids if c not in self._cumulative_canonical]
        self._cumulative_canonical.update(canon_ids)
        for c in canon_ids:
            if c not in self._ordered_canonical:
                self._ordered_canonical.append(c)

        gold_hits, new_gold = self._add_gold_hits(
            candidates, stage, query_or_seed, start_rank=len(self._ordered_canonical) - len(canon_ids)
        )
        self.api_calls.append({
            "stage": stage,
            "provider": provider,
            "endpoint": endpoint,
            "query_or_seed": (query_or_seed or "")[:160],
            "logical_call_id": f"{stage}:{provider}:{endpoint}",
            "physical_http_calls": self._physical_attempts(ck),
            "retry_count": max(0, self._physical_attempts(ck) - 1),
            "cache_hit": cache_hit,
            "cache_key": ck,
            "latency_ms": round(latency_ms, 1),
            "candidate_count": len(candidates),
            "new_unique_count": len(new_unique),
            "gold_hit_count": gold_hits,
            "new_gold_count": new_gold,
            "candidate_summary": [
                {
                    "paper_id": ev.identity.paper_id,
                    "title": (ev.identity.title or "")[:200],
                    "doi": ev.identity.doi,
                }
                for ev in candidates
            ],
        })

    # ------------------------------------------------------------------
    def snapshot(self, stage: str, candidates: list[PaperEvidence]) -> None:
        """记录候选池快照（raw/prekeep/rerank/final 各边界）。"""
        ids = [canonical_paper_id(ev) for ev in candidates]
        gold_ids = sorted({cid for cid in ids if cid in self.gold_lifecycle})
        self.candidate_snapshots.append({
            "stage": stage,
            "candidate_count": len(ids),
            "unique_candidate_count": len(set(ids)),
            "gold_count": len(gold_ids),
            "gold_ids": gold_ids,
            "candidate_ids": ids,
        })

    # ------------------------------------------------------------------
    def record_pool(self, pool_evs: list[PaperEvidence]) -> None:
        """prekeep（词法粗筛 + assoc 保送）后的精排池：更新 gold 存活信息。"""
        in_pool = {canonical_paper_id(ev) for ev in pool_evs}
        for cid, g in self.gold_lifecycle.items():
            g["survived_prekeep"] = cid in in_pool
            if cid in in_pool:
                pool_ids = [canonical_paper_id(ev) for ev in pool_evs]
                g["pre_rerank_rank"] = pool_ids.index(cid) + 1

    # ------------------------------------------------------------------
    def finalize(self, final_evs: list[PaperEvidence], drop_stage: str = "final_output", drop_reason: str = "not_in_final_topk") -> None:
        """最终输出：标记 gold 的 final_rank / drop 阶段。"""
        final_ids = [canonical_paper_id(ev) for ev in final_evs]
        for cid, g in self.gold_lifecycle.items():
            if cid in final_ids:
                g["final_rank"] = final_ids.index(cid) + 1
                g["reranker_rank"] = g.get("reranker_rank") or g["final_rank"]
            else:
                g["drop_stage"] = drop_stage
                g["drop_reason"] = drop_reason

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "run_id": self.run_id,
            "baseline_reference_id": self.baseline_reference_id,
            "planner_status": self.planner_status,
            "planner_fallback_reason": self.planner_fallback_reason,
            "planner_version": self.planner_version,
            "plan_hash": self.plan_hash,
            "regular_subquery_count": self.regular_subquery_count,
            "assoc_subquery_count": self.assoc_subquery_count,
            "plan_cache_hit": self.plan_cache_hit,
            "external_drift_detected": self.external_drift_detected,
            "logical_api_calls": len(self.api_calls),
            "physical_http_calls": sum(c["physical_http_calls"] for c in self.api_calls),
            "retry_count": sum(c["retry_count"] for c in self.api_calls),
            "api_calls": self.api_calls,
            "candidate_snapshots": self.candidate_snapshots,
            "gold_lifecycle": list(self.gold_lifecycle.values()),
        }

    def save(self, out_dir: str | Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"trace_{self.query_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
