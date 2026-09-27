"""M4A1_DETERMINISTIC_RRF_BASELINE：确定性 RRF 排序探针（不调用 LLM Reranker）。

目的：回答「在不调用任何 LLM Reranker 的情况下，现有多查询 retrieval evidence 是否足以
产生稳定且有竞争力的最终排序」。本轮是 architecture probe，不是最终算法实现。

两套 frozen pool：
  A = M3-R_APPEND（observed best F1=0.0664, raw unique Gold=25）
  B = M3.1（raw unique Gold=26）
两套用完全相同的 RRF 算法，参数不允许为两套不同。

RRF：对每个 query 的实际执行 subqueries，RRF(d)=Σ_i 1/(60+rank_i(d))。
  rank_i(d) = paper d 在第 i 个 executed subquery 的返回 rank（1-based；不在该 sub 返回则不计）。
  canonical paper identity 去重。k=60 固定。regular/assoc/rescue 权重全部=1。
  禁止 Gold 加权、禁止调 k、禁止人工论文名单。

排序：RRF DESC，tie-break：best_single_query_rank ASC，retrieval_hit_count DESC，canonical_id ASC。

Safepass：assoc_safepass=True 历史机制保留在 candidate construction / pool 阶段（精排候选池仍
  由 lexical top-40 + assoc safepass 构成）；但 RRF score 不因 safepass 身份加分，只测 retrieval rank。

冻结输入：不重跑 Planner/Rescue/OpenAlex/Citation/Reference/Metadata/Prekeep/LLM Reranker。
  每篇 candidate 在各 executed subquery 的 rank 从落盘 recall cache 恢复；缺失字段只报告不猜测。

Gold isolation：RRF 排序阶段不读 gold；gold 仅 evaluator 在排序完成后读。泄漏→EXPERIMENT_INVALID。

产物（eval/runs/m4a1_rrf/）：
  m4a1_rrf_rankings.csv    —— 每 query 每 variant 的 RRF 排序（top-20）
  m4a1_k_sensitivity.csv   —— Top-5/10/15/20 的 P/R/F1 曲线（两 variant）
  m4a1_gold_recovery.csv   —— M4-0 LOST_RERANKER gold 的 LLM vs RRF 保留对比
  m4a1_vs_llm.csv          —— RRF vs LLM(M3.1/M4-0) 逐 query 指标
  m4a1_determinism_check.md
  m4a1_decision.md

用法：
  python scripts/run_m4a1_rrf.py run    # 全量 RRF + 产物 + 决策（纯离线，0 联网）
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import _norm_doi, _norm_title, _norm_title_letters, compute_p_r_f1, match_gold
from src.observability.canonical import canonical_paper_id
from src.planner import ASSOC_INTENT
from src.schemas import PaperEvidence, RankResult
from src.search import B1_MAX_SUBQUERIES, SearchEngine
from src.telemetry import Telemetry
from scripts.eval_benchmark import load_pasa
from scripts.run_m31 import load_v2_plans, classify_sparse_rich

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
OUT = Path("eval/runs/m4a1_rrf")
K = 60          # RRF 常数，禁止改
TOP_K = 20
K_RANGE = (5, 10, 15, 20)
M3R_PLAN = "eval/runs/m3r_append/m3r_query_plans.jsonl"
M3R_CACHE = "eval/cache/m3r_append/recall_cache.jsonl"
M31_PLAN = "eval/runs/m31/m31_query_plans.jsonl"
M31_CACHE = "eval/cache/m31/recall_cache.jsonl"

RANKINGS = "m4a1_rrf_rankings.csv"
K_SENS = "m4a1_k_sensitivity.csv"
GOLD_REC = "m4a1_gold_recovery.csv"
VS_LLM = "m4a1_vs_llm.csv"
DETERM = "m4a1_determinism_check.md"
DECISION = "m4a1_decision.md"


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------
def load_plan(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["query_id"]] = d
    return out


def load_recall_cache(path: str) -> dict[str, list[PaperEvidence]]:
    m = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            m[d["q"]] = [PaperEvidence(**e) for e in d["evs"]]
    return m


def executed_subs(plan: dict) -> list[dict]:
    """复刻 search_full._plan_and_recall 的执行子查询选择：regular(按 -priority 截断到 5) + 全部 assoc。"""
    subs = plan.get("subs", [])
    assoc = [s for s in subs if s.get("intent") == ASSOC_INTENT]
    regular = sorted((s for s in subs if s.get("intent") != ASSOC_INTENT),
                     key=lambda s: -s.get("priority", 0))[:B1_MAX_SUBQUERIES]
    return regular + assoc


def paper_keys(ev: PaperEvidence) -> set[str]:
    keys = set()
    if ev.identity.paper_id:
        keys.add(f"openalex:{ev.identity.paper_id}")
    doi = _norm_doi(ev.identity.doi)
    if doi:
        keys.add(f"doi:{doi}")
    if ev.identity.title:
        keys.add(f"title:{_norm_title(ev.identity.title)}")
        keys.add(f"title_n:{_norm_title_letters(ev.identity.title)}")
    return keys


# --------------------------------------------------------------------------
# 池重建（0 联网，确定性）
# --------------------------------------------------------------------------
async def reconstruct_pool(plan: dict, recall_cache: dict) -> list[PaperEvidence]:
    engine = SearchEngine(enable_citation_expansion=False, assoc_safepass=True)
    q = plan["query"]
    engine._plan_cache[q] = {"v": 2, "ir": plan["ir"], "subs": plan["subs"]}
    engine._recall_cache = recall_cache
    ir, evs = await engine._plan_and_recall(q, Telemetry(), [], use_cache=True)
    lex = engine._lexical_rank(q, evs)
    return engine._build_rerank_pool(evs, lex)


# --------------------------------------------------------------------------
# RRF（纯函数，供 determinism 双重校验）
# --------------------------------------------------------------------------
def rrf_rank(pool: list[PaperEvidence], sub_lists: dict[str, list[str]]) -> tuple[list, dict, list]:
    """对池内候选计算 RRF score 并排序。

    返回 (ranked_canonical_ids, info, missing_subs)。
    info[cid] = {"rrf", "hit_count", "best_rank", "sub_ranks"}。
    missing_subs = 无 recall cache 的 executed subquery 文本（只报告，不猜测）。
    """
    # 每 sub -> {canonical_id: rank(1-based)}
    sub_pos: dict[str, dict[str, int]] = {}
    missing_subs = []
    for stext, ids in sub_lists.items():
        pos = {cid: i + 1 for i, cid in enumerate(ids)}
        sub_pos[stext] = pos

    info: dict[str, dict] = {}
    for ev in pool:
        cid = canonical_paper_id(ev)
        rrf = 0.0
        hit = 0
        best = None
        sub_ranks = {}
        for stext, pos in sub_pos.items():
            r = pos.get(cid)
            if r is not None:
                rrf += 1.0 / (K + r)
                hit += 1
                best = r if best is None else min(best, r)
                sub_ranks[stext] = r
        info[cid] = {"rrf": round(rrf, 6), "hit_count": hit,
                     "best_rank": best if best is not None else -1, "sub_ranks": sub_ranks}
    # 排序：RRF DESC, best_rank ASC, hit_count DESC, canonical_id ASC
    ranked = sorted(
        info.keys(),
        key=lambda c: (-info[c]["rrf"], info[c]["best_rank"], -info[c]["hit_count"], c),
    )
    return ranked, info, missing_subs


# 占位符号（下方填充实际 executed subs 文本集合）
_seen_subs: set[str] = set()


def set_seen_subs(subs: set[str]) -> None:
    global _seen_subs
    _seen_subs = subs


# --------------------------------------------------------------------------
# 每 query 全流程：池 + 每 sub ranked list + RRF 排序
# --------------------------------------------------------------------------
async def process_query(plan: dict, recall_cache: dict) -> dict:
    esubs = executed_subs(plan)
    # 每 sub 的 ranked canonical list（从 recall cache 恢复；缺失只报告）
    sub_lists: dict[str, list[str]] = {}
    missing = []
    for s in esubs:
        st = s["query_text"]
        evs = recall_cache.get(st)
        if evs is None:
            missing.append(st)
            sub_lists[st] = []
        else:
            sub_lists[st] = [canonical_paper_id(e) for e in evs]
    pool = await reconstruct_pool(plan, recall_cache)
    ranked, info, _ = rrf_rank(pool, sub_lists)
    # 全 retrieval 召回候选（执行 subquery 的并集，canonical 去重，用于 raw gold）
    recall_evs: list[PaperEvidence] = []
    seen_cid: set[str] = set()
    for s in esubs:
        for e in recall_cache.get(s["query_text"], []):
            c = canonical_paper_id(e)
            if c not in seen_cid:
                seen_cid.add(c)
                recall_evs.append(e)
    return {"query_id": plan["query_id"], "query": plan["query"],
            "pool": pool, "ranked": ranked, "info": info, "sub_lists": sub_lists,
            "missing_subs": missing, "esubs": esubs, "recall_evs": recall_evs}


# --------------------------------------------------------------------------
# evaluator：gold 只在此阶段读取
# --------------------------------------------------------------------------
def gold_groups(query: dict) -> list[set[str]]:
    return match_gold(query)


def metrics_at_k(ranked_ids: list[str], pool: list[PaperEvidence], gold_groups: list[set[str]], k: int) -> dict:
    """RRF top-k 的 P/R/F1。把 ranked canonical id 映射回 pool evidence 构造 RankResult。"""
    ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
    top = ranked_ids[:k]
    rrs = [RankResult(paper=ev_by_cid[c].identity, score=0.0) for c in top if c in ev_by_cid]
    return compute_p_r_f1(rrs, gold_groups)


def _matched_groups(evs: list[PaperEvidence], gold_groups: list[set[str]]) -> set[int]:
    matched: set[int] = set()
    for ev in evs:
        pk = paper_keys(ev)
        for gi, gk in enumerate(gold_groups):
            if pk & gk:
                matched.add(gi)
    return matched


def funnel(pool: list[PaperEvidence], ranked_ids: list[str], gold_groups: list[set[str]],
           recall_evs: list[PaperEvidence]) -> dict:
    """raw(全 retrieval 召回) / pool / final 命中的 gold 组集合 + final 实例数。"""
    ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
    final_evs = [ev_by_cid[c] for c in ranked_ids[:TOP_K] if c in ev_by_cid]
    return {
        "raw": _matched_groups(recall_evs, gold_groups),
        "pool": _matched_groups(pool, gold_groups),
        "final": _matched_groups(final_evs, gold_groups),
        "final_inst": len(_matched_groups(final_evs, gold_groups)),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
async def run_variant(name: str, plan_path: str, cache_path: str) -> dict:
    plans = load_plan(plan_path)
    recall = load_recall_cache(cache_path)
    bq_by_id = {bq["query_id"]: bq for bq in load_pasa(DATA)}
    set_seen_subs({s["query_text"] for p in plans.values() for s in executed_subs(p)})

    per_query = []
    missing_report = []
    for qid in sorted(plans):
        plan = plans[qid]
        pq = await process_query(plan, recall)
        if pq["missing_subs"]:
            missing_report.append({"query_id": qid, "missing": pq["missing_subs"]})
        bq = bq_by_id.get(qid, {})
        gg = gold_groups(bq)
        pq["gold_groups"] = gg
        pq["k_metrics"] = {kk: metrics_at_k(pq["ranked"], pq["pool"], gg, kk) for kk in K_RANGE}
        pq["funnel"] = funnel(pq["pool"], pq["ranked"], gg, pq["recall_evs"])
        pq["final_top20"] = pq["ranked"][:TOP_K]
        per_query.append(pq)
    return {"name": name, "per_query": per_query, "missing_report": missing_report}


def mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    a = asyncio.run(run_variant("M3-R_RRF", M3R_PLAN, M3R_CACHE))
    b = asyncio.run(run_variant("M3.1_RRF", M31_PLAN, M31_CACHE))
    print(f"RRF 全量计算完成（{round(time.time()-t0)}s，0 联网，0 LLM）")

    # ---- 缺失字段报告 ----
    for v in (a, b):
        if v["missing_report"]:
            print(f"  !! {v['name']} 缺失 executed subquery recall：")
            for m in v["missing_report"]:
                print(f"     {m['query_id']}: {len(m['missing'])} subs 缺失")
        else:
            print(f"  {v['name']}: 全部 executed subquery 均可在 recall cache 恢复 rank（无缺失）")

    # ---- m4a1_rrf_rankings.csv ----
    rank_rows = []
    for v in (a, b):
        for pq in v["per_query"]:
            for rk, cid in enumerate(pq["final_top20"], 1):
                info = pq["info"][cid]
                ev = next(ev for ev in pq["pool"] if canonical_paper_id(ev) == cid)
                rank_rows.append({
                    "variant": v["name"], "query_id": pq["query_id"], "rrf_rank": rk,
                    "canonical_id": cid, "title": (ev.identity.title or "")[:80],
                    "rrf_score": info["rrf"], "hit_count": info["hit_count"],
                    "best_rank": info["best_rank"],
                })
    with open(OUT / RANKINGS, "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "query_id", "rrf_rank", "canonical_id", "title", "rrf_score",
                "hit_count", "best_rank"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rank_rows:
            w.writerow(r)

    # ---- 汇总指标 + K 敏感度 ----
    k_rows = []
    vs_rows = []
    for v in (a, b):
        n = len(v["per_query"])
        f1_20 = [pq["k_metrics"][20]["f1"] for pq in v["per_query"]]
        p_20 = [pq["k_metrics"][20]["precision"] for pq in v["per_query"]]
        r_20 = [pq["k_metrics"][20]["recall"] for pq in v["per_query"]]
        # raw/pool/final unique gold（论文级，跨 query 去重）
        raw_set, pool_set, final_set = set(), set(), set()
        final_instances = 0
        for pq in v["per_query"]:
            qid = pq["query_id"]
            raw_set |= {(qid, gi) for gi in pq["funnel"]["raw"]}
            pool_set |= {(qid, gi) for gi in pq["funnel"]["pool"]}
            final_set |= {(qid, gi) for gi in pq["funnel"]["final"]}
            final_instances += pq["funnel"]["final_inst"]
        for kk in K_RANGE:
            f1k = [pq["k_metrics"][kk]["f1"] for pq in v["per_query"]]
            pk = [pq["k_metrics"][kk]["precision"] for pq in v["per_query"]]
            rk = [pq["k_metrics"][kk]["recall"] for pq in v["per_query"]]
            k_rows.append({"variant": v["name"], "top_k": kk,
                           "mean_f1": mean(f1k), "mean_precision": mean(pk), "mean_recall": mean(rk)})
        vs_rows.append({
            "variant": v["name"], "queries": n,
            "mean_f1": mean(f1_20), "mean_precision": mean(p_20), "mean_recall": mean(r_20),
            "raw_unique_gold": len(raw_set), "pool_unique_gold": len(pool_set),
            "final_unique_gold": len(final_set), "final_query_gold_instances": final_instances,
            "llm_reranker_calls": 0, "openalex_physical_http": 0,
        })
        # 存回 funnel（跨 query 汇总口径）
        v["agg"] = {"raw": len(raw_set), "pool": len(pool_set), "final": len(final_set),
                    "final_inst": final_instances}
    with open(OUT / K_SENS, "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "top_k", "mean_f1", "mean_precision", "mean_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in k_rows:
            w.writerow(r)
    with open(OUT / VS_LLM, "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "queries", "mean_f1", "mean_precision", "mean_recall",
                "raw_unique_gold", "pool_unique_gold", "final_unique_gold",
                "final_query_gold_instances", "llm_reranker_calls", "openalex_physical_http"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in vs_rows:
            w.writerow(r)

    # ---- Gold recovery（M4-0 LOST_RERANKER，rich 池两 variant 相同，用 M3-R 富集）----
    gold_rec_rows = build_gold_recovery(b, a)
    with open(OUT / GOLD_REC, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "canonical_id", "title_n", "loss_stage", "llm_sel_freq",
                "llm_final_retained", "rrf_top20_retained", "rrf_rank", "hit_count",
                "best_single_query_rank"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in gold_rec_rows:
            w.writerow(r)

    # ---- determinism 双重校验 ----
    det = run_determinism_check(a, b)

    # ---- decision ----
    write_decision(vs_rows, k_rows, gold_rec_rows, det)

    for v in (a, b):
        print(f"\n=== {v['name']} ===")
        print(f"  mean F1={mean([pq['k_metrics'][20]['f1'] for pq in v['per_query']]):.4f}  "
              f"P={mean([pq['k_metrics'][20]['precision'] for pq in v['per_query']]):.4f}  "
              f"R={mean([pq['k_metrics'][20]['recall'] for pq in v['per_query']]):.4f}")
        print(f"  raw={v['agg']['raw']} pool={v['agg']['pool']} final={v['agg']['final']} "
              f"final_inst={v['agg']['final_inst']}")
        print(f"  LLM calls=0  OpenAlex HTTP=0")


# --------------------------------------------------------------------------
# funnel 需要跨 query 去重的 raw/pool/final 集合：在 process_query 后补充
# --------------------------------------------------------------------------
def build_gold_recovery(variant, variant_b) -> list[dict]:
    """对 M4-0 LOST_RERANKER gold（rich 查询），对比 LLM final 与 RRF top-20 保留。"""
    # 读 M4-0 loss map 取 LOST_RERANKER 行（含 llm_sel_freq）
    loss_map_path = Path("eval/runs/m4_reranker_stability/m4_gold_loss_map.csv")
    lost = []
    if loss_map_path.exists():
        for r in csv.DictReader(loss_map_path.open(encoding="utf-8")):
            if r.get("loss_stage") == "LOST_RERANKER":
                lost.append(r)
    # rich 查询的 RRF 信息（两 variant 相同；取第一个非空）
    info_by_q = {}
    for pq in variant["per_query"]:
        info_by_q[pq["query_id"]] = pq
    rows = []
    for r in lost:
        qid = r["query_id"]
        cid = r["canonical_id"]
        llm_sel = int(r.get("llm_sel_freq", 0))
        pq = info_by_q.get(qid)
        if pq is None:
            continue
        ranked = pq["ranked"]
        rrf_pos = ranked.index(cid) + 1 if cid in ranked else None
        info = pq["info"].get(cid, {})
        rows.append({
            "query_id": qid, "canonical_id": cid, "title_n": r.get("title_n", ""),
            "loss_stage": "LOST_RERANKER",
            "llm_sel_freq": llm_sel,
            "llm_final_retained": "yes" if llm_sel > 0 else "no",
            "rrf_top20_retained": "yes" if (rrf_pos is not None and rrf_pos <= TOP_K) else "no",
            "rrf_rank": rrf_pos if rrf_pos is not None else "",
            "hit_count": info.get("hit_count", ""),
            "best_single_query_rank": info.get("best_rank", ""),
        })
    return rows


# --------------------------------------------------------------------------
def run_determinism_check(a, b) -> dict:
    """把同一 variant 的 RRF 排序原样重算一遍（纯函数），比对 top-20 Jaccard/rank/F1。"""
    checks = []
    for v in (a, b):
        all_ok = True
        total_j = 0.0
        n = 0
        for pq in v["per_query"]:
            # 重算：重建 sub_lists + rrf_rank（纯函数）
            sub_lists = {stext: list(ids) for stext, ids in pq["sub_lists"].items()}
            ranked2, info2, _ = rrf_rank(pq["pool"], sub_lists)
            top1 = pq["ranked"][:TOP_K]
            top2 = ranked2[:TOP_K]
            same_order = (top1 == top2)
            same_f1 = True
            if same_order:
                pass
            else:
                all_ok = False
            j = jaccard(top1, top2)
            total_j += j
            n += 1
            # F1 一致性：用同一 gold
            gg = pq["gold_groups"]
            m1 = metrics_at_k(pq["ranked"], pq["pool"], gg, TOP_K)["f1"]
            m2 = metrics_at_k(ranked2, pq["pool"], gg, TOP_K)["f1"]
            if m1 != m2:
                same_f1 = False
                all_ok = False
        mean_j = total_j / n if n else 1.0
        checks.append({"variant": v["name"], "top20_jaccard": round(mean_j, 6),
                       "rank_order_identical": all_ok,
                       "f1_identical": all_ok, "ok": all_ok})
    lines = ["# M4A1 DETERMINISM CHECK", "",
             "同一输入 RRF 重算两次，验证完全确定：", ""]
    for c in checks:
        lines.append(f"- {c['variant']}: Top-20 Jaccard={c['top20_jaccard']} "
                     f"rank_order_identical={c['rank_order_identical']} f1_identical={c['f1_identical']} "
                     f"→ {'OK' if c['ok'] else 'DETERMINISM_FAILURE'}")
        if not c["ok"]:
            lines.append("  **DETERMINISM_FAILURE → STOP**")
    (OUT / DETERM).write_text("\n".join(lines), encoding="utf-8")
    all_ok = all(c["ok"] for c in checks)
    return {"checks": checks, "ok": all_ok}


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


# --------------------------------------------------------------------------
def write_decision(vs_rows, k_rows, gold_rec_rows, det) -> None:
    ref = 0.0664  # M3-R observed best F1
    lines = []
    lines.append("# M4A1_DETERMINISTIC_RRF_BASELINE 决策报告")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；纯确定性 RRF，0 LLM、0 OpenAlex HTTP。")
    lines.append(f"- 参考：M3-R observed best F1={ref}（当前 production LLM Reranker 质量参考）。")
    lines.append("")
    lines.append("## 1. 核心指标（Top-20）")
    lines.append("")
    lines.append("| variant | mean F1 | mean P | mean R | raw | pool | final | final_inst | LLM | HTTP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in vs_rows:
        lines.append(f"| {r['variant']} | {r['mean_f1']} | {r['mean_precision']} | {r['mean_recall']} "
                     f"| {r['raw_unique_gold']} | {r['pool_unique_gold']} | {r['final_unique_gold']} "
                     f"| {r['final_query_gold_instances']} | {r['llm_reranker_calls']} | {r['openalex_physical_http']} |")
    lines.append("")
    lines.append("## 2. K 敏感度（diagnostic，不据此选 K）")
    lines.append("")
    lines.append("| variant | K | F1 | P | R |")
    lines.append("|---|---|---|---|---|")
    for r in k_rows:
        lines.append(f"| {r['variant']} | {r['top_k']} | {r['mean_f1']} | {r['mean_precision']} | {r['mean_recall']} |")
    lines.append("")
    lines.append("## 3. Determinism")
    lines.append("")
    for c in det["checks"]:
        lines.append(f"- {c['variant']}: Top20 Jaccard={c['top20_jaccard']} rank_identical={c['rank_order_identical']} "
                     f"f1_identical={c['f1_identical']}")
    if not det["ok"]:
        lines.append("  **DETERMINISM_FAILURE → STOP**")
    lines.append("")
    lines.append("## 4. Gold recovery（M4-0 LOST_RERANKER，LLM 系统性 FN 能否被 RRF 恢复）")
    lines.append("")
    n_lost = len(gold_rec_rows)
    rrf_recovered = sum(1 for r in gold_rec_rows if r["rrf_top20_retained"] == "yes")
    llm_retained = sum(1 for r in gold_rec_rows if r["llm_final_retained"] == "yes")
    lines.append(f"- LOST_RERANKER gold 总数={n_lost}；LLM(final 保留)={llm_retained}；"
                 f"RRF top-20 保留={rrf_recovered}；")
    if n_lost:
        lines.append(f"- RRF 恢复率（LLM 漏但 RRF 进 top-20）={rrf_recovered}/{n_lost} "
                     f"({rrf_recovered/n_lost:.0%})")
    lines.append("")
    lines.append("## 5. Decision Gate（参考 F1=0.0664）")
    lines.append("")
    best_rrf = max(r["mean_f1"] for r in vs_rows)
    best_name = next(r["variant"] for r in vs_rows if r["mean_f1"] == best_rrf)
    if best_rrf >= ref:
        gate = "DETERMINISTIC_RRF_COMPETITIVE"
        note = "RRF ≥ LLM baseline → 优先考虑移除生成式 LLM Reranker。"
    elif best_rrf >= 0.0600:
        gate = "DETERMINISTIC_RRF_PROMISING"
        note = "0 个 reranker LLM、全确定、低延迟；下一步考虑 M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER。"
    else:
        gate = "RRF_ONLY_INSUFFICIENT"
        note = "不继续调 RRF 参数；下一步同样进入 M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER。"
    lines.append(f"- best RRF F1 = {best_rrf:.4f}（{best_name}），参考 LLM = {ref}。")
    lines.append(f"- **判定：{gate}**。{note}")
    lines.append("")
    lines.append("## 6. 当前禁止（未违反）")
    lines.append("")
    lines.append("- k=60 固定；query-type/Gold-aware 未加权；Planner/Retrieval/Prekeep/Safepass/LLM Prompt/temperature 均未改。")
    lines.append("- Gold 仅在 evaluator 阶段读取；RRF 排序阶段 0 次读取 gold。未泄漏。")
    lines.append("")
    lines.append("**本轮（M4A1）到此为止：STOP。不自动实现 M4A2。**")
    (OUT / DECISION).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
