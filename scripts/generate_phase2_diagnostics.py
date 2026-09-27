"""Phase 2 offline diagnostics：从 instrumented replay 的 trace + response cache 生成 9 个产物。

全部离线：不调用任何外部 API / LLM。数据来源：
- eval/diagnostics/v2_instrumented/trace_*.json（instrumented run 的 recorder 输出）
- eval/runs/instrumented/PASA_ASSOC_INSTR_*.json（instrumented run reports）
- eval/runs/PASA_ASSOC_*.json（v2 baseline reports，历史参考，不修改）
- eval/runs/_pasa_plan_cache.jsonl / _pasa_recall_cache.jsonl（v2 冻结计划与 recall）

产物（eval/diagnostics/v2_instrumented/）：
1  INSTRUMENTED_BASELINE_REPORT.md   综合分析 + drift 分类 + 决策建议
2  api_cost_breakdown.csv            per query × stage 的 logical/physical/unique/gold
3  query_stage_trace.jsonl           recorder 全量 trace（每 query 一行）
4  candidate_funnel.jsonl            per query × stage 候选池快照
5  gold_lifecycle.csv                per gold 论文检索生命周期
6  raw_recall_failure_taxonomy.csv   F1=0 且 raw recall miss 的失败分类
7  gold_reachability.csv             per query × gold 各阶段可达性
8  citation_seed_value.csv           per citation seed 的边际价值
9  citation_ablation.json            Full/Top-1/Top-3/High-confidence/No Citation 离线对比
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import compute_p_r_f1, match_gold
from src.observability.canonical import clean_doi, norm_title
from src.schemas import PaperIdentity, RankLabel, RankResult

TRACE_DIR = Path("eval/diagnostics/v2_instrumented")
INSTR_RUNS = Path("eval/runs/instrumented")
BASELINE_RUNS = Path("eval/runs")
PLAN_CACHE = Path("eval/runs/_pasa_plan_cache.jsonl")
BASELINE_RECALL = Path("eval/runs/_pasa_recall_cache.jsonl")
PASA_DATA = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")

ASSOC_INTENT = "联想论文名"
RECALL_STAGES = {"regular_recall", "assoc_recall"}
CITATION_STAGES = {"citation", "reference"}
FROZEN_MEAN_F1 = 0.0412


def load_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_json_files(dir: Path, pattern: str) -> dict[str, dict]:
    out = {}
    if dir.exists():
        for p in sorted(dir.glob(pattern)):
            d = json.loads(p.read_text(encoding="utf-8"))
            qid = d.get("query_id") or d.get("raw_query", "")
            out[qid] = d
    return out


def title_ns_of(report: dict) -> set[str]:
    return {x.removeprefix("title_n:") for x in report.get("gold_ids", []) if x.startswith("title_n:")}


def gold_groups_from_titles(titles: list[str]) -> list[set[str]]:
    return [{f"title_n:{norm_title(t)}"} for t in titles if norm_title(t)]


def summary_to_ev(summary: dict) -> dict:
    """候选摘要 -> 用于词法排序/F1 的轻量对象。"""
    return {
        "paper_id": summary.get("paper_id"),
        "title": summary.get("title") or "",
        "doi": summary.get("doi"),
        "title_n": norm_title(summary.get("title") or ""),
    }


def lex_score(query_tokens: set[str], ev: dict) -> float:
    title_t = set(re.findall(r"[a-z0-9]+", (ev["title"] or "").lower()))
    title_hits = len(query_tokens & title_t)
    return (title_hits * 3.0) / (len(query_tokens) * 3.0 + 1e-9)


def lexical_topk(summaries: list[dict], query: str, top_k: int = 20) -> list[dict]:
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_tokens:
        return summaries[:top_k]
    ranked = sorted(summaries, key=lambda s: lex_score(q_tokens, s), reverse=True)
    return ranked[:top_k]


def pool_f1(summaries: list[dict], query: str, gold_groups: list[set[str]], top_k: int = 20) -> dict:
    top = lexical_topk(summaries, query, top_k)
    results = [RankResult(paper=PaperIdentity(
                              paper_id=s["paper_id"] or s.get("doi") or s["title_n"] or f"anon-{i}",
                              title=s["title"] or "",
                              doi=s.get("doi")),
                          score=0.0, label=RankLabel.NO)
               for i, s in enumerate(top)]
    m = compute_p_r_f1(results, gold_groups)
    return {"topk": top_k, "F1": m["f1"], "P": m["precision"], "R": m["recall"],
            "tp": m["tp"], "gold_covered": m["tp"]}


def union_pool(api_calls: list[dict], stages: set[str]) -> tuple[dict[str, dict], dict[str, int]]:
    """按 canonical id 合并指定 stage 的候选摘要。返回 (pool, id->stage)。"""
    pool: dict[str, dict] = {}
    origin: dict[str, int] = {}  # canonical -> stage bucket label
    for call in api_calls:
        if call["stage"] not in stages:
            continue
        for s in call.get("candidate_summary", []):
            doi = clean_doi(s.get("doi") or "")
            cid = f"doi:{doi}" if doi else (s.get("paper_id") or f"title_n:{norm_title(s.get('title') or '')}")
            if cid not in pool:
                pool[cid] = s
                origin[cid] = call["stage"]
    return pool, origin


def seed_order(api_calls: list[dict], raw_pool: dict[str, dict], query: str) -> list[str]:
    """citation/reference 调用的 seed 顺序：按词法排序（生产 = lex top-5）近似。"""
    seed_keys = []
    seen = set()
    for call in api_calls:
        if call["stage"] in CITATION_STAGES:
            key = call.get("query_or_seed", "")
            if key and key not in seen:
                seen.add(key)
                seed_keys.append(key)
    # 生产 seed = 候选池词法 top-5；seed key 通常为 doi，与 raw_pool 的 candidate 关联较弱，
    # 因此直接按调用出现顺序返回（该顺序即生产每轮 seed 执行顺序）。
    return seed_keys


def group_seed_candidates(api_calls: list[dict]) -> dict[str, dict[str, dict]]:
    """seed key -> {canonical_id: summary}（citations+references 合并）。"""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for call in api_calls:
        if call["stage"] not in CITATION_STAGES:
            continue
        seed = call.get("query_or_seed", "")
        for s in call.get("candidate_summary", []):
            doi = clean_doi(s.get("doi") or "")
            cid = f"doi:{doi}" if doi else (s.get("paper_id") or f"title_n:{norm_title(s.get('title') or '')}")
            out[seed][cid] = s
    return out


def canonical_of(s: dict) -> str:
    doi = clean_doi(s.get("doi") or "")
    if doi:
        return f"doi:{doi}"
    pid = s.get("paper_id")
    if pid:
        return pid
    return f"title_n:{norm_title(s.get('title') or '')}"


# ---------------------------------------------------------------------------
# 9 个产物
# ---------------------------------------------------------------------------
def api_cost_rows(traces: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        for call in tr["api_calls"]:
            rows.append({
                "query_id": qid,
                "stage": call["stage"],
                "provider": call["provider"],
                "endpoint": call["endpoint"],
                "query_or_seed": call["query_or_seed"],
                "logical_calls": 1,
                "physical_http_calls": call["physical_http_calls"],
                "retry_count": call["retry_count"],
                "cache_hit": call["cache_hit"],
                "candidate_count": call["candidate_count"],
                "new_unique_count": call["new_unique_count"],
                "gold_hit_count": call["gold_hit_count"],
                "new_gold_count": call["new_gold_count"],
            })
    return rows


def funnel_rows(traces: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        for snap in tr["candidate_snapshots"]:
            rows.append({
                "query_id": qid,
                "stage": snap["stage"],
                "candidate_count": snap["candidate_count"],
                "unique_candidate_count": snap["unique_candidate_count"],
                "gold_count": snap["gold_count"],
            })
    return rows


def gold_lifecycle_rows(traces: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        for g in tr["gold_lifecycle"]:
            rows.append({
                "query_id": qid,
                "canonical_id": g["canonical_id"],
                "title_n": g.get("title_n", ""),
                "first_seen_stage": g.get("first_seen_stage"),
                "first_seen_query": g.get("first_seen_query"),
                "first_seen_rank": g.get("first_seen_rank"),
                "survived_prekeep": g.get("survived_prekeep"),
                "pre_rerank_rank": g.get("pre_rerank_rank"),
                "reranker_rank": g.get("reranker_rank"),
                "final_rank": g.get("final_rank"),
                "drop_stage": g.get("drop_stage"),
                "drop_reason": g.get("drop_reason"),
            })
    return rows


def reachability_rows(traces: dict[str, dict], instr_reports: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        gold_ns = title_ns_of(instr_reports.get(qid, {}))
        if not gold_ns:
            continue
        seen_stages: dict[str, str] = {}
        for call in tr["api_calls"]:
            for s in call.get("candidate_summary", []):
                if norm_title(s.get("title") or "") in gold_ns:
                    seen_stages.setdefault(canonical_of(s), call["stage"])
        # 池/最终可达性
        raw_covered = len(seen_stages)
        for cid in list(seen_stages):
            rows.append({
                "query_id": qid,
                "canonical_id": cid,
                "title_n": next((g.get("title_n") for g in tr["gold_lifecycle"] if g["canonical_id"] == cid), ""),
                "first_seen_stage": seen_stages[cid],
                "gold_total": len(gold_ns),
            })
    return rows


def seed_value_rows(traces: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        for call in tr["api_calls"]:
            if call["stage"] not in CITATION_STAGES:
                continue
            rows.append({
                "query_id": qid,
                "stage": call["stage"],
                "seed": call["query_or_seed"],
                "provider": call["provider"],
                "candidate_count": call["candidate_count"],
                "new_unique_count": call["new_unique_count"],
                "gold_hit_count": call["gold_hit_count"],
                "new_gold_count": call["new_gold_count"],
            })
    return rows


def failure_taxonomy_rows(traces: dict[str, dict], instr_reports: dict[str, dict],
                          recall_by_query: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        rep = instr_reports.get(qid, {})
        if rep.get("f1", 0) != 0:
            continue
        gold_ns = title_ns_of(rep)
        if not gold_ns:
            continue
        # 任何 subquery 候选里是否含 gold（instrumented）
        instr_hits = set()
        for call in tr["api_calls"]:
            for s in call.get("candidate_summary", []):
                if norm_title(s.get("title") or "") in gold_ns:
                    instr_hits.add(canonical_of(s))
        # baseline recall cache 里是否含 gold（统一 canonical 口径）
        base_hits = set()
        for evs in recall_by_query.values():
            for ev in evs:
                ident = ev.get("identity") or {}
                if norm_title(ident.get("title") or "") in gold_ns:
                    base_hits.add(canonical_of({
                        "paper_id": ident.get("paper_id"),
                        "doi": ident.get("doi"),
                        "title": ident.get("title") or "",
                    }))
        rows.append({
            "query_id": qid,
            "gold_total": len(gold_ns),
            "instrumented_raw_gold_hits": len(instr_hits),
            "baseline_recall_cache_gold_hits": len(base_hits),
            "instr_eq_base_cache": instr_hits == base_hits,
            "taxonomy": _taxonomy(len(instr_hits), len(base_hits)),
        })
    return rows


def _taxonomy(instr: int, base: int) -> str:
    if instr == 0 and base == 0:
        return "RAW_RECALL_COVERAGE_MISS"
    if instr > 0 and base == 0:
        return "RAW_RECALL_GAINED_VS_CACHE"
    if instr == 0 and base > 0:
        return "EXTERNAL_API_DRIFT_OR_CACHE_STALE"
    return "REACHED_RAW_BUT_LOST_BEFORE_FINAL"


def gold_metrics_summary(traces: dict[str, dict], instr_reports: dict[str, dict]) -> dict:
    """权威 Gold 口径（phase2_decision_report §0）：直接读 trace 快照，不重算匹配。

    三个口径 + 派生：
    - total_query_gold_instances: 各 query 基准 gold 实例数之和（title_n 口径，与 ablation gold_total 一致）
    - query_gold_instances_reached_{raw,pool,final}: 各阶段快照 gold_count 之和（实例级）
    - unique_gold_reached: rerank_pool 快照去重 gold_ids 论文数（论文级，跨 query）
    - gold_hit_events / new_gold_events: api_calls 求和（事件级）

    注意：snapshot 的 gold_count 由 recorder 在真实 pipeline 中用 canonical 匹配得到，
    是权威口径；离线 ablation（§citation_ablation）是 title_n 近似，可能与之不一致
    （例如 q6 的 DOI 识别候选），22/22 统一以本函数为准。
    """
    rows = []
    totals = {"instances": 0, "raw": 0, "pool": 0, "final": 0, "events": 0, "new_events": 0}
    unique_pool: set[str] = set()
    for qid in sorted(traces):
        tr = traces[qid]
        rep = instr_reports.get(qid, {})
        n_total = len(title_ns_of(rep))
        snaps = {s["stage"]: s for s in tr.get("candidate_snapshots", [])}
        raw = snaps.get("after_raw_recall", {}).get("gold_count", 0)
        pool_snap = snaps.get("rerank_pool") or snaps.get("final_reranked")
        pool_n = pool_snap.get("gold_count", 0) if pool_snap else 0
        final = snaps.get("final_reranked", {}).get("gold_count", 0)
        unique_pool.update(pool_snap.get("gold_ids", []) if pool_snap else [])
        ev = sum(a.get("gold_hit_count", 0) for a in tr.get("api_calls", []))
        nev = sum(a.get("new_gold_count", 0) for a in tr.get("api_calls", []))
        totals["instances"] += n_total
        totals["raw"] += raw
        totals["pool"] += pool_n
        totals["final"] += final
        totals["events"] += ev
        totals["new_events"] += nev
        rows.append({
            "query_id": qid,
            "gold_instances_total": n_total,
            "instances_reached_raw": raw,
            "instances_reached_pool": pool_n,
            "instances_reached_final": final,
            "gold_hit_events": ev,
            "new_gold_events": nev,
        })
    summary = {
        "total_query_gold_instances": totals["instances"],
        "query_gold_instances_reached_raw": totals["raw"],
        "query_gold_instances_reached_pool": totals["pool"],
        "query_gold_instances_reached_final": totals["final"],
        "unique_gold_reached": len(unique_pool),
        "gold_hit_events": totals["events"],
        "new_gold_events": totals["new_events"],
    }
    return {"per_query": rows, "summary": summary}


def pool_gold_coverage(pool: dict[str, dict], gold_groups: list[set[str]]) -> int:
    """pool 中覆盖的 gold 论文数（跨 paper_id/doi/title_n 匹配，与 harness 语义一致）。

    这是「检索贡献」指标：不依赖排序器，只回答『该 stage 的候选池是否捞到了 gold』。
    词法 top-20 F1 是排序近似（LLM reranker 非确定性 + abstract 证据缺失），会系统性低估，
    因此 coverage 才是 ablation 的主指标。

    注意：本函数是**离线 ablation 口径**（按候选摘要 title_n 匹配），与
    gold_metrics_summary 的 trace 快照权威口径可能不一致，全量统计以 trace 为准。
    """
    matched: set[int] = set()
    for s in pool.values():
        keys: set[str] = set()
        if s.get("paper_id"):
            keys.add(f"openalex:{s['paper_id']}")
        doi = clean_doi(s.get("doi") or "")
        if doi:
            keys.add(f"doi:{doi}")
        tn = norm_title(s.get("title") or "")
        if tn:
            keys.add(f"title_n:{tn}")
        for gi, gkeys in enumerate(gold_groups):
            if gi in matched:
                continue
            if keys & gkeys:
                matched.add(gi)
                break
    return len(matched)


def citation_ablation(traces: dict[str, dict], instr_reports: dict[str, dict]) -> dict:
    strategies = {}
    per_query = []
    for qid, tr in traces.items():
        rep = instr_reports.get(qid, {})
        gold_groups = gold_groups_from_titles(
            [t for g in rep.get("gold_ids", []) if g.startswith("title_n:") for t in [g.removeprefix("title_n:")]]
        )
        raw_pool, _ = union_pool(tr["api_calls"], RECALL_STAGES)
        all_cit = union_pool(tr["api_calls"], CITATION_STAGES)[0]
        seeds = seed_order(tr["api_calls"], raw_pool, rep.get("raw_query", ""))
        seed_cands = group_seed_candidates(tr["api_calls"])
        query = rep.get("raw_query", "")

        # 高相关 seed：与生产 _high_relevance 语义对齐（词法相关 seed = 首个）
        hi_seeds = seeds[:1] if seeds else []

        variants = {
            "no_citation": dict(raw_pool),
            "top1": {**raw_pool, **{k: v for seed in seeds[:1] for k, v in seed_cands.get(seed, {}).items()}},
            "top3": {**raw_pool, **{k: v for seed in seeds[:3] for k, v in seed_cands.get(seed, {}).items()}},
            "high_confidence": {**raw_pool, **{k: v for seed in hi_seeds for k, v in seed_cands.get(seed, {}).items()}},
            "full": {**raw_pool, **all_cit},
        }
        row = {"query_id": qid, "seeds": seeds, "gold_total": len(gold_groups)}
        base_covered = None
        for name, pool in variants.items():
            covered = pool_gold_coverage(pool, gold_groups)
            f1 = pool_f1(list(pool.values()), query, gold_groups)
            entry = {
                "gold_in_pool": covered,
                "gold_total": len(gold_groups),
                "coverage_ratio": round(covered / len(gold_groups), 4) if gold_groups else 0.0,
                "lex_f1_approx": f1["F1"],          # title-only 词法近似，LLM reranker 下界
                "lex_tp": f1["tp"],
            }
            if name == "no_citation":
                base_covered = covered
            entry["new_gold_vs_no_citation"] = max(covered - (base_covered or covered), 0) if name != "no_citation" else 0
            row[name] = entry
            strategies.setdefault(name, []).append({"query_id": qid, **entry})
        per_query.append(row)
    summary = {}
    for name, lst in strategies.items():
        mean_f1 = round(sum(r["lex_f1_approx"] for r in lst) / len(lst), 4) if lst else 0.0
        mean_cov = round(sum(r["coverage_ratio"] for r in lst) / len(lst), 4) if lst else 0.0
        gold_in_pool = sum(r["gold_in_pool"] for r in lst)
        new_gold = sum(r["new_gold_vs_no_citation"] for r in lst)
        summary[name] = {"mean_lex_f1_approx": mean_f1, "mean_pool_coverage": mean_cov,
                         "total_gold_in_pool": gold_in_pool, "total_new_gold_vs_no_citation": new_gold,
                         "queries": len(lst)}
    return {"per_query": per_query, "summary": summary}


def drift_analysis(traces: dict[str, dict], instr_reports: dict[str, dict],
                   baseline_reports: dict[str, dict], plans: dict[str, dict]) -> list[dict]:
    rows = []
    for qid, tr in traces.items():
        rep = instr_reports.get(qid, {})
        base = baseline_reports.get(qid, {})
        plan = plans.get(rep.get("raw_query", ""), {})
        plan_hash_ok = tr.get("plan_hash", "") == plan_hash_of(plan) if plan else None
        f1_diff = round(rep.get("f1", 0) - base.get("f1", 0), 4) if base else None
        api_diff = (rep.get("api_calls", 0) - base.get("api_calls", 0)) if base else None
        cls = "MATCH" if f1_diff == 0 else ("EXTERNAL_API_DRIFT" if f1_diff is not None else "UNKNOWN")
        if f1_diff and f1_diff > 0.02:
            cls = "RECALL_IMPROVED_DRIFT"
        rows.append({
            "query_id": qid,
            "plan_hash_ok": plan_hash_ok,
            "baseline_f1": base.get("f1"),
            "instrumented_f1": rep.get("f1"),
            "f1_diff": f1_diff,
            "baseline_api": base.get("api_calls"),
            "instrumented_api": rep.get("api_calls"),
            "api_diff": api_diff,
            "classification": cls,
            "logical_calls": tr.get("logical_api_calls"),
            "physical_calls": tr.get("physical_http_calls"),
        })
    return rows


def plan_hash_of(plan: dict) -> str:
    return hashlib.sha256(
        json.dumps({"v": plan["v"], "ir": plan["ir"], "subs": plan["subs"]},
                   sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
def write_md(traces, api_rows, funnel_rows, lifecycle_rows, fail_rows, reach_rows,
             seed_rows, ablation, drift, gold, instr_reports, baseline_reports) -> str:
    lines = []
    lines.append("# ScholarTrace Phase 2 Instrumented Baseline Report")
    lines.append("")
    lines.append(f"**Frozen baseline reference:** mean F1 = {FROZEN_MEAN_F1} (22 queries, v2 plan). "
                 "Instrumented replay preserves the frozen plan (plan_hash validated per query).")
    lines.append("")
    n = len(instr_reports)
    mean_instr = round(sum(r.get("f1", 0) for r in instr_reports.values()) / max(n, 1), 4)
    logical = sum(r.get("logical_api_calls", 0) for r in traces.values())
    physical = sum(r.get("physical_http_calls", 0) for r in traces.values())
    lines.append(f"## 1. Instrumented run aggregate")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| queries | {n} |")
    lines.append(f"| instrumented mean F1 | {mean_instr} |")
    lines.append(f"| frozen baseline mean F1 | {FROZEN_MEAN_F1} |")
    lines.append(f"| F1 diff | {round(mean_instr - FROZEN_MEAN_F1, 4)} |")
    lines.append(f"| logical API calls | {logical} |")
    lines.append(f"| physical HTTP attempts (incl. retries) | {physical} |")
    lines.append(f"| retries | {sum(r.get('retry_count', 0) for r in traces.values())} |")
    lines.append("")
    g = gold["summary"]
    lines.append("## 1.5 Gold metrics（权威 trace 口径，见 phase2_decision_report §0）")
    lines.append("")
    lines.append("| 口径 | 定义 | 值 |")
    lines.append("|---|---|---|")
    lines.append(f"| total_query_gold_instances | 基准 gold 实例总数 | {g['total_query_gold_instances']} |")
    lines.append(f"| query_gold_instances_reached (raw) | after_raw_recall 快照实例 | {g['query_gold_instances_reached_raw']} |")
    lines.append(f"| query_gold_instances_reached (pool) | rerank_pool 快照实例 | {g['query_gold_instances_reached_pool']} |")
    lines.append(f"| query_gold_instances_reached (final) | final_reranked 快照实例 | {g['query_gold_instances_reached_final']} |")
    lines.append(f"| unique_gold_reached | 去重论文数（rerank_pool，跨 query） | {g['unique_gold_reached']} |")
    lines.append(f"| gold_hit_events | 事件级命中（api_calls 求和） | {g['gold_hit_events']} |")
    lines.append(f"| new_gold_events | 首次带回事件 | {g['new_gold_events']} |")
    lines.append("")
    lines.append("## 2. API cost breakdown (aggregate, gold 列 = 事件级 gold_hit_events / new_gold_events)")
    lines.append("")
    aggr = defaultdict(lambda: {"logical": 0, "physical": 0, "unique": 0, "gold": 0, "new_gold": 0})
    for r in api_rows:
        a = aggr[r["stage"]]
        a["logical"] += r["logical_calls"]
        a["physical"] += r["physical_http_calls"]
        a["unique"] += r["new_unique_count"]
        a["gold"] += r["gold_hit_count"]
        a["new_gold"] += r["new_gold_count"]
    lines.append("| stage | logical calls | physical attempts | new unique papers | gold hits | new gold |")
    lines.append("|---|---|---|---|---|---|")
    for stage, a in sorted(aggr.items()):
        lines.append(f"| {stage} | {a['logical']} | {a['physical']} | {a['unique']} | {a['gold']} | {a['new_gold']} |")
    lines.append("")
    lines.append("## 3. Drift classification (instrumented vs frozen baseline)")
    lines.append("")
    lines.append("| query | plan_hash | base F1 | instr F1 | F1 diff | base API | instr API | class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in drift:
        lines.append(f"| {r['query_id']} | {r['plan_hash_ok']} | {r['baseline_f1']} | {r['instrumented_f1']} "
                     f"| {r['f1_diff']} | {r['baseline_api']} | {r['instrumented_api']} | {r['classification']} |")
    lines.append("")
    lines.append("## 4. Raw recall failure taxonomy")
    lines.append("")
    lines.append("| query | gold total | instr raw hits | cache raw hits | taxonomy |")
    lines.append("|---|---|---|---|---|")
    for r in fail_rows:
        lines.append(f"| {r['query_id']} | {r['gold_total']} | {r['instrumented_raw_gold_hits']} "
                     f"| {r['baseline_recall_cache_gold_hits']} | {r['taxonomy']} |")
    lines.append("")
    lines.append("## 5. Citation ablation (offline replay, same response cache)")
    lines.append("")
    lines.append("| strategy | mean pool coverage | total gold in pool (离线口径) | new gold vs no-citation | mean lex-F1(approx) | queries |")
    lines.append("|---|---|---|---|---|---|")
    for name, s in ablation["summary"].items():
        lines.append(f"| {name} | {s['mean_pool_coverage']} | {s['total_gold_in_pool']} | "
                     f"{s['total_new_gold_vs_no_citation']} | {s['mean_lex_f1_approx']} | {s['queries']} |")
    lines.append("")
    lines.append("> 注：本表是**离线 ablation 口径**（prekeep 前的 raw+expansion 候选集，title_n 匹配近似），"
                 "与 §1.5 的 trace 权威口径可能不一致（权威统计以 §1.5 / phase2_decision_report §0 为准）。"
                 "主指标是 **pool coverage**（该策略候选池捞到几篇 gold，不依赖排序器）。"
                 "lex-F1 用 title-only 词法排序近似（无 abstract/LLM rerank），是真实 F1 的保守下界，"
                 "仅用于比较 citation 策略的候选池边际贡献，**pool recall equivalence ≠ final F1 equivalence**，"
                 "不证明关闭 Citation 后真实 reranker F1 相同（须 NO_CITATION controlled eval）。")
    lines.append("")
    lines.append("## 6. Recommended next mechanism")
    lines.append("")
    lines.append("详见 `phase2_decision_report.md`。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", type=Path, default=TRACE_DIR)
    ap.add_argument("--instr-runs", type=Path, default=INSTR_RUNS)
    ap.add_argument("--baseline-runs", type=Path, default=BASELINE_RUNS)
    args = ap.parse_args()

    traces = load_json_files(args.trace_dir, "trace_*.json")
    instr_reports = load_json_files(args.instr_runs, "PASA_ASSOC_INSTR_*.json")
    baseline_reports = load_json_files(args.baseline_runs, "PASA_ASSOC_*.json")
    plans = {d["query"]: d for d in load_jsonl(PLAN_CACHE)}
    recall_by_query: dict[str, list[dict]] = defaultdict(list)
    for r in load_jsonl(BASELINE_RECALL):
        recall_by_query[r["q"]].extend(r["evs"])

    print(f"traces={len(traces)} instr={len(instr_reports)} baseline={len(baseline_reports)}")
    if not traces:
        raise SystemExit("无 trace：先运行 scripts/instrumented_replay.py")

    api_rows = api_cost_rows(traces)
    funnel = funnel_rows(traces)
    life = gold_lifecycle_rows(traces)
    fail = failure_taxonomy_rows(traces, instr_reports, recall_by_query)
    reach = reachability_rows(traces, instr_reports)
    seeds = seed_value_rows(traces)
    ablation = citation_ablation(traces, instr_reports)
    drift = drift_analysis(traces, instr_reports, baseline_reports, plans)
    gold = gold_metrics_summary(traces, instr_reports)

    out = args.trace_dir
    out.mkdir(parents=True, exist_ok=True)

    # 3: query_stage_trace.jsonl
    with (out / "query_stage_trace.jsonl").open("w", encoding="utf-8") as f:
        for qid in sorted(traces):
            f.write(json.dumps(traces[qid], ensure_ascii=False) + "\n")

    def write_csv(name, rows, fieldnames):
        if not rows:
            (out / name).write_text("", encoding="utf-8")
            return
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_csv("api_cost_breakdown.csv", api_rows, list(api_rows[0].keys()))
    write_csv("candidate_funnel.csv", funnel, list(funnel[0].keys()))
    write_csv("gold_lifecycle.csv", life, list(life[0].keys()))
    fail_cols = list(fail[0].keys()) if fail else ["query_id", "f1", "recall", "raw_gold_count", "stage", "reason"]
    write_csv("raw_recall_failure_taxonomy.csv", fail, fail_cols)
    write_csv("gold_reachability.csv", reach, list(reach[0].keys()))
    write_csv("citation_seed_value.csv", seeds, list(seeds[0].keys()))
    write_csv("gold_metrics.csv", gold["per_query"],
              ["query_id", "gold_instances_total", "instances_reached_raw",
               "instances_reached_pool", "instances_reached_final",
               "gold_hit_events", "new_gold_events"])
    (out / "gold_metrics_summary.json").write_text(
        json.dumps(gold["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    # candidate_funnel.jsonl + citation_ablation.json
    with (out / "candidate_funnel.jsonl").open("w", encoding="utf-8") as f:
        for row in funnel:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "citation_ablation.json").write_text(json.dumps(ablation, ensure_ascii=False, indent=2), encoding="utf-8")

    md = write_md(traces, api_rows, funnel, life, fail, reach, seeds, ablation, drift, gold,
                  instr_reports, baseline_reports)
    (out / "INSTRUMENTED_BASELINE_REPORT.md").write_text(md, encoding="utf-8")
    print(f"写入 {out}：9 个产物完成")
    print(json.dumps(ablation["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
