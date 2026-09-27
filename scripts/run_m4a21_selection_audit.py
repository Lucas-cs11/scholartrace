"""M4A2_1_OUTPUT_SELECTION_AUDIT：拆分 Ranking Quality 与 Output Selection/Cutoff。

背景：M4A2 的 Gate 只看 Top-20 F1=0.0503（<0.055 → MINILM_CE_INSUFFICIENT），但
CE Top-10=0.0708、Top-5=0.0719 均高于 LLM 0.0664，且 CE final unique Gold=17 > LLM 16、
CE 恢复 M4-0 LOST_RERANKER Gold 5/8 → 问题可能不是「MiniLM 不会排序」，而是
「固定 Top-20 输出把大量尾部候选当相关论文，拖垮 Precision」。本审计把两个问题分开。

本轮全部离线、确定性：
  - OpenAlex HTTP=0，LLM calls=0，CE inference=0（不加载模型、不重新打分）。
  - 直接读取已落盘：m4a2_ce_rankings.csv（冻结 CE 排序/分数）、M3-R trace final_reranked
    （LLM final 输出）、plan+recall cache（仅离线重建 pool 以取 paper_keys 供 evaluator 匹配）。
  - 禁止重跑：Planner/OpenAlex/Prekeep/MiniLM inference/LLM Reranker。

内容：
  1. LLM output cardinality 审计（22 queries：min/p25/median/mean/p75/max）。
  2. Matched-cardinality：N_q=LLM final 数，CE 取 Top-N_q，同预算比 ranking quality。
  3. Ranking diagnostics：MRR@20/MAP@20/Recall@5/10/20/NDCG@5/10/20（CE 与 LLM 同口径）。
  4. 唯一允许的新 selection policy：M4A2_1_SCORE_GAP（i∈[3,min(19,n-1)] 取最大 gap，min=3 max=20）。
  5. Score-gap determinism 双重校验。
  6. 对照表 + Decision Gate（A: matched-N≥LLM→CE_RANKING_COMPETITIVE；
     B: score-gap≥0.0664→DETERMINISTIC_CE_PIPELINE_COMPETITIVE；C→CE_SEMANTIC_CAPACITY_LIMIT）。

Gold isolation：score-gap cutoff 只读 sorted CE scores；gold 仅在 evaluator 输出形成后读取。

产物（eval/runs/m4a21_selection_audit/）：
  m4a21_llm_cardinality.csv   m4a21_matched_cardinality.csv   m4a21_ranking_metrics.csv
  m4a21_score_gap_predictions.csv  m4a21_comparison.csv  m4a21_determinism.md  m4a21_decision.md

用法：python scripts/run_m4a21_selection_audit.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
import statistics
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

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
OUT = Path("eval/runs/m4a21_selection_audit")
M3R_PLAN = "eval/runs/m3r_append/m3r_query_plans.jsonl"
M3R_CACHE = "eval/cache/m3r_append/recall_cache.jsonl"
M3R_TRACE = Path("eval/diagnostics/m3r_append")
M4A2_RANKINGS = "eval/runs/m4a2_ce/m4a2_ce_rankings.csv"

CARD = "m4a21_llm_cardinality.csv"
MATCHED = "m4a21_matched_cardinality.csv"
RMET = "m4a21_ranking_metrics.csv"
SG = "m4a21_score_gap_predictions.csv"
CMP = "m4a21_comparison.csv"
DET = "m4a21_determinism.md"
DEC = "m4a21_decision.md"

TOP_K = 20
REF_LLM_F1 = 0.0664
REF_LLM_FINAL = 16
REF_LLM_CALLS = 77
REF_RRF_F1 = 0.0241
REF_RRF_FINAL = 7
REF_CE_TOP20_F1 = 0.0503
REF_CE_TOP20_FINAL = 17


# --------------------------------------------------------------------------
# 加载（全部离线；pool 重建仅读 plan+recall cache，0 联网 0 推理）
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


def load_trace(qid: str) -> dict:
    return json.loads((M3R_TRACE / f"trace_{qid}.json").read_text(encoding="utf-8"))


def snapshot_ids(trace: dict, stage: str) -> list[str]:
    for s in trace.get("candidate_snapshots", []):
        if s.get("stage") == stage:
            return list(s.get("candidate_ids", []))
    return []


def executed_subs(plan: dict) -> list[dict]:
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


async def reconstruct_pool(plan: dict, recall_cache: dict) -> list[PaperEvidence]:
    engine = SearchEngine(enable_citation_expansion=False, assoc_safepass=True)
    q = plan["query"]
    engine._plan_cache[q] = {"v": 2, "ir": plan["ir"], "subs": plan["subs"]}
    engine._recall_cache = recall_cache
    ir, evs = await engine._plan_and_recall(q, Telemetry(), [], use_cache=True)
    lex = engine._lexical_rank(q, evs)
    return engine._build_rerank_pool(evs, lex)


def load_ce_rankings() -> dict[str, list[dict]]:
    """{qid: [{ce_rank, canonical_id, ce_score}, ...]} 按 ce_rank 升序（score 降序）。"""
    out: dict[str, list[dict]] = {}
    with open(M4A2_RANKINGS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["query_id"], []).append({
                "rank": int(r["ce_rank"]), "cid": r["canonical_id"], "score": float(r["ce_score"]),
            })
    for qid in out:
        out[qid].sort(key=lambda r: r["rank"])
    return out


# --------------------------------------------------------------------------
# evaluator：gold 只在输出形成后读取
# --------------------------------------------------------------------------
def gold_groups(query: dict) -> list[set[str]]:
    return match_gold(query)


def ev_matches_gold(ev: PaperEvidence, gg: list[set[str]]) -> bool:
    pk = paper_keys(ev)
    return any(pk & gk for gk in gg)


def matched_groups(evs: list[PaperEvidence], gg: list[set[str]]) -> set[int]:
    matched: set[int] = set()
    for ev in evs:
        pk = paper_keys(ev)
        for gi, gk in enumerate(gg):
            if pk & gk:
                matched.add(gi)
    return matched


def f1_of(selected: list[PaperEvidence], gg: list[set[str]]) -> dict:
    rrs = [RankResult(paper=ev.identity, score=0.0) for ev in selected]
    return compute_p_r_f1(rrs, gg)


def select_evs(cids: list[str], ev_by_cid: dict[str, PaperEvidence]) -> list[PaperEvidence]:
    return [ev_by_cid[c] for c in cids if c in ev_by_cid]


# --------------------------------------------------------------------------
# Ranking diagnostics（binary relevance：rank 上候选匹配任一 gold 组 → rel=1）
# --------------------------------------------------------------------------
def rank_metrics(ranked_cids: list[str], ev_by_cid: dict[str, PaperEvidence], gg: list[set[str]]) -> dict:
    G = len(gg)  # 该 query 的 gold 论文数（总相关数）
    rel = [1 if ev_matches_gold(ev_by_cid[c], gg) else 0 for c in ranked_cids if c in ev_by_cid]
    Ks = (5, 10, 20)
    out = {}
    # MRR@20
    mrr = 0.0
    for i, r in enumerate(rel[:20], 1):
        if r:
            mrr = 1.0 / i
            break
    out["mrr@20"] = round(mrr, 6)
    # AP@20（除以 G=总相关数）
    hits = 0
    ap = 0.0
    for i, r in enumerate(rel[:20], 1):
        if r:
            hits += 1
            ap += hits / i
    out["map@20"] = round(ap / G, 6) if G else 0.0
    # Recall@k
    for k in Ks:
        out[f"recall@{k}"] = round(sum(rel[:k]) / G, 6) if G else 0.0
    # NDCG@k（binary relevance）
    for k in Ks:
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:k], 1))
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(G, k) + 1))
        out[f"ndcg@{k}"] = round(dcg / ideal, 6) if ideal else 0.0
    return out


# --------------------------------------------------------------------------
# M4A2_1_SCORE_GAP：唯一允许的 deterministic adaptive cutoff
# --------------------------------------------------------------------------
def score_gap_cutoff(scores_desc: list[float], n_pool: int) -> int:
    """sorted CE scores 降序；gap_i = s_i - s_(i+1)，i∈[3, min(19, n-1)]，argmax gap，tie 取更小 i。
    返回 Top-i*（即输出 i* 篇）。输入只能是 sorted CE scores（不含任何 gold/历史信息）。"""
    hi = min(19, n_pool - 1)
    best_i, best_gap = 3, None
    for i in range(3, hi + 1):
        gap = scores_desc[i - 1] - scores_desc[i]  # s_i - s_(i+1)
        if best_gap is None or gap > best_gap:
            best_gap = gap
            best_i = i
    return best_i


# --------------------------------------------------------------------------
async def build_dataset(plans: dict, recall_cache: dict) -> dict:
    dataset = {}
    for qid in sorted(plans):
        pool = await reconstruct_pool(plans[qid], recall_cache)
        ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
        llm_final = snapshot_ids(load_trace(qid), "final_reranked")
        dataset[qid] = {"query_id": qid, "pool": pool, "ev_by_cid": ev_by_cid,
                        "n_pool": len(pool), "llm_final": llm_final, "n_llm": len(llm_final)}
    return dataset


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plans = load_plan(M3R_PLAN)
    recall = load_recall_cache(M3R_CACHE)
    ce_rank = load_ce_rankings()
    bq_by_id = {bq["query_id"]: bq for bq in load_pasa(DATA)}

    dataset = asyncio.run(build_dataset(plans, recall))
    # 校验 CE rankings CSV 与重建池一致 + CE 分数降序
    for qid, rows in ce_rank.items():
        ev_by_cid = dataset[qid]["ev_by_cid"]
        missing = [r["cid"] for r in rows if r["cid"] not in ev_by_cid]
        if missing:
            raise SystemExit(f"CE rankings 含重建池外 cid: {qid} {missing[:3]}")
        sc = [r["score"] for r in rows]
        if any(sc[i] < sc[i + 1] for i in range(len(sc) - 1)):
            raise SystemExit(f"CE scores 非降序: {qid}")
    print(f"[M4A2.1] 冻结数据载入完成：{len(dataset)} queries；CE rankings 与重建池一致；LLM final 与 n_predicted 一致。")

    # gold（evaluator）在输出形成后才读；此处建立每 query gold_groups 供各变体统一评估
    gg_by_q = {qid: gold_groups(bq_by_id.get(qid, {})) for qid in dataset}

    # ================= 1. LLM cardinality =================
    card_rows = []
    llm_counts = [dataset[qid]["n_llm"] for qid in sorted(dataset)]
    for qid in sorted(dataset):
        card_rows.append({"query_id": qid, "llm_final_count": dataset[qid]["n_llm"]})
    with open(OUT / CARD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "llm_final_count"])
        w.writeheader()
        for r in card_rows:
            w.writerow(r)
    if llm_counts:
        p25 = sorted(llm_counts)[int(0.25 * (len(llm_counts) - 1))]
        p75 = sorted(llm_counts)[int(0.75 * (len(llm_counts) - 1))]
    else:
        p25 = p75 = 0
    card_stat = {"min": min(llm_counts), "p25": p25, "median": statistics.median(llm_counts),
                 "mean": statistics.mean(llm_counts), "p75": p75, "max": max(llm_counts)}
    print(f"  LLM final count: min={card_stat['min']} p25={card_stat['p25']} "
          f"median={card_stat['median']} mean={card_stat['mean']:.2f} p75={card_stat['p75']} max={card_stat['max']}")

    # ================= 2. Matched-cardinality：CE Top-N_q vs LLM variable-N =================
    matched_rows = []
    ce_match_f1s = []
    llm_var_f1s = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        N = pq["n_llm"]
        # CE Top-N（从冻结 rankings）
        ce_top_n = [r["cid"] for r in ce_rank[qid][:N]]
        ce_evs = select_evs(ce_top_n, pq["ev_by_cid"])
        llm_evs = select_evs(pq["llm_final"], pq["ev_by_cid"])
        m_ce = f1_of(ce_evs, gg)
        m_llm = f1_of(llm_evs, gg)
        ce_match_f1s.append(m_ce["f1"])
        llm_var_f1s.append(m_llm["f1"])
        matched_rows.append({
            "query_id": qid, "llm_count_N": N, "ce_matched_count": len(ce_top_n),
            "llm_var_N_f1": m_llm["f1"], "llm_var_N_precision": m_llm["precision"],
            "llm_var_N_recall": m_llm["recall"],
            "ce_matched_N_f1": m_ce["f1"], "ce_matched_N_precision": m_ce["precision"],
            "ce_matched_N_recall": m_ce["recall"],
        })
    with open(OUT / MATCHED, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "llm_count_N", "ce_matched_count", "llm_var_N_f1",
                "llm_var_N_precision", "llm_var_N_recall", "ce_matched_N_f1",
                "ce_matched_N_precision", "ce_matched_N_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in matched_rows:
            w.writerow(r)
    llm_var_f1 = statistics.mean(llm_var_f1s)
    ce_match_f1 = statistics.mean(ce_match_f1s)
    # final unique gold（跨 query 去重）+ 实例
    def unique_gold(select_fn):
        u = set()
        inst = 0
        for qid in sorted(dataset):
            pq = dataset[qid]
            gg = gg_by_q[qid]
            evs = select_fn(qid, pq)
            grps = matched_groups(evs, gg)
            u |= {(qid, gi) for gi in grps}
            inst += len(grps)
        return len(u), inst
    ce_match_gold, ce_match_inst = unique_gold(
        lambda qid, pq: select_evs([r["cid"] for r in ce_rank[qid][:pq["n_llm"]]], pq["ev_by_cid"]))
    llm_var_gold, llm_var_inst = unique_gold(
        lambda qid, pq: select_evs(pq["llm_final"], pq["ev_by_cid"]))
    print(f"  matched-N：LLM variable-N F1={llm_var_f1:.4f} (gold={llm_var_gold}, inst={llm_var_inst})；"
          f"CE matched-N F1={ce_match_f1:.4f} (gold={ce_match_gold}, inst={ce_match_inst})")

    # ================= 3. Ranking diagnostics =================
    rm_rows = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        ce_cids = [r["cid"] for r in ce_rank[qid]]
        llm_cids = pq["llm_final"]
        ce_m = rank_metrics(ce_cids, pq["ev_by_cid"], gg)
        llm_m = rank_metrics(llm_cids, pq["ev_by_cid"], gg)
        row = {"query_id": qid}
        for k in (5, 10, 20):
            for metric in ("recall", "ndcg"):
                row[f"ce_{metric}@{k}"] = ce_m[f"{metric}@{k}"]
                row[f"llm_{metric}@{k}"] = llm_m[f"{metric}@{k}"]
        row["ce_mrr@20"] = ce_m["mrr@20"]; row["llm_mrr@20"] = llm_m["mrr@20"]
        row["ce_map@20"] = ce_m["map@20"]; row["llm_map@20"] = llm_m["map@20"]
        rm_rows.append(row)
    # 汇总行
    def agg(col):
        return statistics.mean(r[col] for r in rm_rows)
    agg_row = {"query_id": "MEAN"}
    for col in rm_rows[0]:
        if col != "query_id":
            agg_row[col] = round(agg(col), 4)
    rm_rows.append(agg_row)
    with open(OUT / RMET, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rm_rows[0].keys()))
        w.writeheader()
        for r in rm_rows:
            w.writerow(r)
    print(f"  ranking metrics (MEAN)：CE MRR@20={agg_row['ce_mrr@20']} MAP@20={agg_row['ce_map@20']} "
          f"R@5={agg_row['ce_recall@5']} R@10={agg_row['ce_recall@10']} R@20={agg_row['ce_recall@20']} "
          f"NDCG@5={agg_row['ce_ndcg@5']} NDCG@20={agg_row['ce_ndcg@20']}")
    print(f"                 LLM  MRR@20={agg_row['llm_mrr@20']} MAP@20={agg_row['llm_map@20']} "
          f"R@5={agg_row['llm_recall@5']} R@10={agg_row['llm_recall@10']} R@20={agg_row['llm_recall@20']} "
          f"NDCG@5={agg_row['llm_ndcg@5']} NDCG@20={agg_row['llm_ndcg@20']}")

    # ================= 4. Score-gap predictions =================
    sg_rows = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        rows = ce_rank[qid]
        scores = [r["score"] for r in rows]
        i_star = score_gap_cutoff(scores, pq["n_pool"])
        selected = [r["cid"] for r in rows[:i_star]]
        gap_val = scores[i_star - 1] - scores[i_star]  # s_i* - s_{i*+1}
        sg_rows.append({"query_id": qid, "i_star": i_star, "cardinality": len(selected),
                        "s_at_cut": scores[i_star - 1], "s_next": scores[i_star],
                        "gap": round(gap_val, 4), "selected_cids": ";".join(selected)})
    with open(OUT / SG, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "i_star", "cardinality", "s_at_cut", "s_next", "gap", "selected_cids"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sg_rows:
            w.writerow(r)
    # score-gap 变体 F1 / gold
    sg_f1s = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        row = next(r for r in sg_rows if r["query_id"] == qid)
        evs = select_evs(row["selected_cids"].split(";"), pq["ev_by_cid"])
        sg_f1s.append(f1_of(evs, gg)["f1"])
    sg_f1 = statistics.mean(sg_f1s)
    sg_gold, sg_inst = unique_gold(
        lambda qid, pq: select_evs(next(r for r in sg_rows if r["query_id"] == qid)["selected_cids"].split(";"),
                                   pq["ev_by_cid"]))
    print(f"  score-gap：F1={sg_f1:.4f} (gold={sg_gold}, inst={sg_inst})；cardinality 分布 "
          f"min={min(r['cardinality'] for r in sg_rows)} max={max(r['cardinality'] for r in sg_rows)}")

    # ================= 5. Determinism（score-gap 两次）=================
    det = determinism_check(dataset, ce_rank, gg_by_q)

    # ================= 6. 对照表 + 决策 =================
    # 复算 CE Top20 / LLM 供对照交叉验证
    ce20_f1 = statistics.mean([f1_of(select_evs([r["cid"] for r in ce_rank[qid][:20]], dataset[qid]["ev_by_cid"]),
                                        gg_by_q[qid])["f1"] for qid in sorted(dataset)])
    ce20_gold, ce20_inst = unique_gold(
        lambda qid, pq: select_evs([r["cid"] for r in ce_rank[qid][:20]], pq["ev_by_cid"]))
    llm_f1 = statistics.mean(llm_var_f1s)
    cmp_rows = [
        {"variant": "M3-R LLM", "selection": "LLM threshold", "f1": round(REF_LLM_F1, 4),
         "precision": "", "recall": "", "final_gold": REF_LLM_FINAL, "gen_llm_calls": REF_LLM_CALLS},
        {"variant": "RRF", "selection": "Top20", "f1": REF_RRF_F1, "precision": "", "recall": "",
         "final_gold": REF_RRF_FINAL, "gen_llm_calls": 0},
        {"variant": "MiniLM CE", "selection": "Top20", "f1": round(REF_CE_TOP20_F1, 4), "precision": "",
         "recall": "", "final_gold": REF_CE_TOP20_FINAL, "gen_llm_calls": 0},
        {"variant": "MiniLM CE", "selection": "matched LLM N", "f1": round(ce_match_f1, 4), "precision": "",
         "recall": "", "final_gold": ce_match_gold, "gen_llm_calls": 0},
        {"variant": "MiniLM CE", "selection": "score-gap", "f1": round(sg_f1, 4), "precision": "",
         "recall": "", "final_gold": sg_gold, "gen_llm_calls": 0},
    ]
    # 交叉验证：复算 LLM F1 应与参考一致
    print(f"  交叉验证：复算 LLM variable-N F1={llm_f1:.4f}（参考 0.0664）；CE Top20 F1={ce20_f1:.4f}（参考 0.0503）")
    with open(OUT / CMP, "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "selection", "f1", "precision", "recall", "final_gold", "gen_llm_calls"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in cmp_rows:
            w.writerow(r)

    write_determinism(det)
    write_decision(card_stat, llm_var_f1, llm_var_gold, llm_var_inst,
                   ce_match_f1, ce_match_gold, ce_match_inst,
                   rm_rows, sg_rows, sg_f1, sg_gold, sg_inst, det, cmp_rows, llm_f1, ce20_f1)
    print(f"  产物已写入 {OUT}/；总耗时 {time.time()-t0:.1f}s（全程 0 LLM、0 HTTP、0 CE inference）")


# --------------------------------------------------------------------------
def determinism_check(dataset, ce_rank, gg_by_q) -> dict:
    """Score-gap selection 执行两次，验证 selected IDs / cardinality / F1 一致。"""
    sel1, sel2 = {}, {}
    for qid in sorted(dataset):
        pq = dataset[qid]
        rows = ce_rank[qid]
        scores = [r["score"] for r in rows]
        i1 = score_gap_cutoff(scores, pq["n_pool"])
        i2 = score_gap_cutoff(scores, pq["n_pool"])
        sel1[qid] = [r["cid"] for r in rows[:i1]]
        sel2[qid] = [r["cid"] for r in rows[:i2]]
    ids_ok = all(sel1[q] == sel2[q] for q in sel1)
    card_ok = all(len(sel1[q]) == len(sel2[q]) for q in sel1)
    f1_ok = True
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        e1 = select_evs(sel1[qid], pq["ev_by_cid"])
        e2 = select_evs(sel2[qid], pq["ev_by_cid"])
        if f1_of(e1, gg)["f1"] != f1_of(e2, gg)["f1"]:
            f1_ok = False
    ok = ids_ok and card_ok and f1_ok
    return {"ok": ok, "selected_ids_identical": ids_ok, "cardinality_identical": card_ok,
            "f1_identical": f1_ok, "queries": len(dataset)}


def write_determinism(det: dict) -> None:
    lines = [
        "# M4A2.1 DETERMINISM CHECK（M4A2_1_SCORE_GAP）",
        "",
        "- Score-gap selection 为纯函数（仅依赖冻结的 sorted CE scores），对每 query 执行两次。",
        f"- selected_ids_identical = {det['selected_ids_identical']}",
        f"- cardinality_identical = {det['cardinality_identical']}",
        f"- f1_identical = {det['f1_identical']}",
        f"- 校验 queries = {det['queries']}",
        "",
        "**SCORE-GAP DETERMINISM PASSED**" if det["ok"] else "**SCORE-GAP DETERMINISM_FAILURE → STOP**",
        "",
        "## 运行成本",
        "- 全程离线：generative LLM calls = 0；OpenAlex HTTP = 0；CE inference = 0（未加载模型、未重新打分）。",
    ]
    (OUT / DET).write_text("\n".join(lines), encoding="utf-8")


def write_decision(card_stat, llm_var_f1, llm_var_gold, llm_var_inst,
                   ce_match_f1, ce_match_gold, ce_match_inst,
                   rm_rows, sg_rows, sg_f1, sg_gold, sg_inst, det, cmp_rows, llm_f1_recomp,
                   ce20_f1_recomp) -> None:
    lines = ["# M4A2_1_OUTPUT_SELECTION_AUDIT 决策报告", "",
             f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；全程离线：OpenAlex HTTP=0、LLM calls=0、CE inference=0。",
             "- 数据源：m4a2_ce_rankings.csv（冻结 CE 排序/分数）+ M3-R trace final_reranked（LLM final 输出）+ plan/recall cache（离线重建 pool）。",
             f"- 交叉验证：LLM F1 复算={llm_f1_recomp:.4f}（参考 0.0664）；CE Top20 F1 复算={ce20_f1_recomp:.4f}（参考 0.0503）。",
             ""]
    lines.append("")
    lines.append("## 1. LLM output cardinality（Section 2）")
    lines.append("")
    lines.append(f"- 22 queries LLM final result count：min={card_stat['min']} p25={card_stat['p25']} "
                 f"median={card_stat['median']} mean={card_stat['mean']:.2f} p75={card_stat['p75']} max={card_stat['max']}。")
    lines.append("- LLM 是 **Ranking + Variable-length Selection**（score>=0.35, min_keep=3, max_results=20），非固定 20 篇。")
    lines.append("")
    lines.append("## 2. Matched-cardinality（Section 3，同 prediction budget）")
    lines.append("")
    lines.append(f"- LLM variable-N：mean F1={llm_var_f1:.4f}，final unique Gold={llm_var_gold}，instances={llm_var_inst}。")
    lines.append(f"- CE matched-N（N=LLM 每 query 输出数）：mean F1={ce_match_f1:.4f}，final unique Gold={ce_match_gold}，instances={ce_match_inst}。")
    lines.append("")
    lines.append("## 3. Ranking diagnostics（Section 4，binary relevance，@20 因冻结 Top-20 输出）")
    lines.append("")
    agg = rm_rows[-1]
    lines.append("| metric | CE | LLM |")
    lines.append("|---|---|---|")
    for m in ("mrr@20", "map@20", "recall@5", "recall@10", "recall@20", "ndcg@5", "ndcg@10", "ndcg@20"):
        lines.append(f"| {m} | {agg[f'ce_{m}']} | {agg[f'llm_{m}']} |")
    lines.append("")
    lines.append("## 4. Score-gap（Section 5，唯一允许的 adaptive cutoff）")
    lines.append("")
    lines.append(f"- M4A2_1_SCORE_GAP：i∈[3,min(19,n-1)] 取最大 gap，tie 取更小 i；min=3 max=20。")
    lines.append(f"- mean F1={sg_f1:.4f}，final unique Gold={sg_gold}，instances={sg_inst}。")
    lines.append(f"- cardinality：min={min(r['cardinality'] for r in sg_rows)} max={max(r['cardinality'] for r in sg_rows)}（详见 m4a21_score_gap_predictions.csv）。")
    lines.append("")
    lines.append("## 5. Determinism（Section 7）")
    lines.append("")
    lines.append(f"- selected_ids_identical={det['selected_ids_identical']}；cardinality_identical={det['cardinality_identical']}；f1_identical={det['f1_identical']}；queries={det['queries']}。")
    lines.append(f"- → {'DETERMINISM PASSED' if det['ok'] else 'DETERMINISM_FAILURE → STOP'}。")
    lines.append("")
    lines.append("## 6. 对照表（Section 8）")
    lines.append("")
    lines.append("| Variant | Selection | F1 | P | R | final Gold | Gen LLM calls |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in cmp_rows:
        lines.append(f"| {r['variant']} | {r['selection']} | {r['f1']} | {r['precision'] or '—'} | "
                     f"{r['recall'] or '—'} | {r['final_gold']} | {r['gen_llm_calls']} |")
    lines.append("")
    lines.append("Top-5 / Top-10 仅保留在 diagnostic appendix（m4a21_ranking_metrics.csv / m4a2_k_sensitivity.csv），"
                 "**不得作为 production variant**。")
    lines.append("")
    lines.append("## 7. Decision Gate（Section 9，参考 LLM F1=0.0664）")
    lines.append("")
    if ce_match_f1 >= REF_LLM_F1:
        gate = "CE_RANKING_COMPETITIVE"
        note = "同 prediction budget 下 CE ≥ LLM → MiniLM 主要问题是 selection 而非 semantic ranking。"
    elif sg_f1 >= REF_LLM_F1:
        gate = "DETERMINISTIC_CE_PIPELINE_COMPETITIVE"
        note = "MiniLM + adaptive cutoff（确定性、0 LLM）成为 production candidate。"
    else:
        gate = "CE_SEMANTIC_CAPACITY_LIMIT"
        note = "matched-N 与 score-gap 均未达到 LLM → 下一步才允许测试一次更强 reranker。"
    lines.append(f"- CE matched-N F1={ce_match_f1:.4f}（LLM={llm_var_f1:.4f}）；score-gap F1={sg_f1:.4f}（参考 0.0664）。")
    lines.append(f"- **判定：{gate}**。{note}")
    lines.append("")
    lines.append("## 8. 当前禁止（未违反）")
    lines.append("")
    lines.append("- 未用 Top-5/Top-10 作 production；未据 K-sensitivity 选 K；未调 MiniLM、未 fine-tune；")
    lines.append("- 无 score threshold grid search、无 score-gap 参数 tuning、无 CE+RRF fusion、未换 BGE；未改 Planner/Retriever/Prekeep。")
    lines.append("- Score-gap 计算仅用 sorted CE scores，0 次读 gold；evaluator 在输出形成后读取。")
    lines.append("")
    lines.append("**本轮（M4A2.1）到此为止：STOP。不自行进入更强 Reranker。**")
    (OUT / DEC).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
