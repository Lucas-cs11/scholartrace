"""M4A3_STRONGER_RERANKER：BAAI/bge-reranker-v2-m3（Reranker 模型升级的最后一次实验）。

M4A2.1 已冻结（CE_SEMANTIC_CAPACITY_LIMIT）。MiniLM 不再做 threshold/K/score-gap tuning、不 fine-tune、
不做 CE+RRF fusion。本轮唯一模型：BAAI/bge-reranker-v2-m3，inference only，model.eval()。

上游完全冻结：M3-R_APPEND frozen pre-rerank pool（raw=25, pool=22）。禁止重跑 Planner/Rescue/OpenAlex/
Citation/Reference/Metadata/Prekeep。必须验证 candidate IDs/order 与 M3-R snapshot 一致，否则 EXPERIMENT_INVALID。

主比较用 matched cardinality：N_q = M3-R LLM final output count（冻结，不读 Gold）；BGE 输出 Top-N_q。
Top-20 仅辅助。全程确定性两次打分。

产物（eval/runs/m4a3_bge/）：
  m4a3_bge_rankings.csv m4a3_matched_cardinality.csv m4a3_ranking_metrics.csv m4a3_gold_comparison.csv
  m4a3_efficiency.csv m4a3_determinism.md m4a3_vs_llm_vs_minilm_vs_rrf.csv m4a3_decision.md
完成后 STOP（RERANKER_RESEARCH_FREEZE=true，下一主线 Iterative/Agentic Retrieval）。

用法：python3 scripts/run_m4a3_bge.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder

from eval.harness import _norm_doi, _norm_title, _norm_title_letters, compute_p_r_f1, match_gold
from src.observability.canonical import canonical_paper_id
from src.planner import ASSOC_INTENT
from src.schemas import PaperEvidence, RankResult
from src.search import B1_MAX_SUBQUERIES, SearchEngine
from src.telemetry import Telemetry
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
OUT = Path("eval/runs/m4a3_bge")
M3R_PLAN = "eval/runs/m3r_append/m3r_query_plans.jsonl"
M3R_CACHE = "eval/cache/m3r_append/recall_cache.jsonl"
M3R_TRACE = Path("eval/diagnostics/m3r_append")
M4A2_RANKINGS = "eval/runs/m4a2_ce/m4a2_ce_rankings.csv"
M4_LOSS = "eval/runs/m4_reranker_stability/m4_gold_loss_map.csv"
M4A1_RECOVERY = "eval/runs/m4a1_rrf/m4a1_gold_recovery.csv"

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MAX_LEN = 512  # 模型官方推荐值（HF 模型卡 transformers 示例 max_length=512）；非测试集定制
BATCH_SIZE = 4  # 568M CPU 推理，小批控制峰值内存
DEVICE = "cpu"
SCORE_P1 = OUT / "bge_pass1_scores.npy"
SCORE_P2 = OUT / "bge_pass2_scores.npy"

TOP_K = 20
# 冻结基线（M4A2.1 / M4A2 / M4A1）
REF_LLM = {"f1": 0.0664, "final_gold": 16, "mrr": 0.369, "map": 0.120, "ndcg5": 0.207, "calls": 77}
REF_MINI = {"f1": 0.0598, "final_gold": 14, "mrr": 0.184, "map": 0.051, "ndcg5": 0.099, "calls": 0}
REF_RRF = {"f1": 0.0241, "final_gold": 7, "calls": 0}


# --------------------------------------------------------------------------
# 加载（上游冻结）
# --------------------------------------------------------------------------
def load_plan(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["query_id"]] = d
    return out


def load_recall_cache(path: str) -> dict[str, list[PaperEvidence]]:
    m = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
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


# --------------------------------------------------------------------------
# passage + token stats（输入定义：title + "\\n" + abstract；缺 abstract 仅 title）
# --------------------------------------------------------------------------
def build_passage(ev: PaperEvidence) -> str:
    title = ev.identity.title or ""
    abstract = ev.abstract or ""
    return f"{title}\n{abstract}" if abstract else title


def token_stats(tok, ev: PaperEvidence) -> dict:
    title = ev.identity.title or ""
    abstract = ev.abstract or ""
    passage = f"{title}\n{abstract}" if abstract else title
    n_full = len(tok(passage, add_special_tokens=False, verbose=False)["input_ids"])
    n_trunc = len(tok(passage, add_special_tokens=False, max_length=MAX_LEN,
                      truncation=True, verbose=False)["input_ids"])
    n_title = len(tok(title, add_special_tokens=False, verbose=False)["input_ids"])
    n_abs = len(tok(abstract, add_special_tokens=False, verbose=False)["input_ids"]) if abstract else 0
    return {"tokens": n_full, "was_truncated": n_full > MAX_LEN, "n_trunc_tokens": n_trunc,
            "title_tokens": n_title, "abstract_tokens": n_abs, "has_abstract": bool(abstract)}


# --------------------------------------------------------------------------
# evaluator
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


def rank_metrics(ranked_cids: list[str], ev_by_cid: dict[str, PaperEvidence], gg: list[set[str]]) -> dict:
    G = len(gg)
    rel = [1 if ev_matches_gold(ev_by_cid[c], gg) else 0 for c in ranked_cids if c in ev_by_cid]
    Ks = (5, 10, 20)
    out = {}
    mrr = 0.0
    for i, r in enumerate(rel[:20], 1):
        if r:
            mrr = 1.0 / i
            break
    out["mrr@20"] = round(mrr, 6)
    hits = 0
    ap = 0.0
    for i, r in enumerate(rel[:20], 1):
        if r:
            hits += 1
            ap += hits / i
    out["map@20"] = round(ap / G, 6) if G else 0.0
    for k in Ks:
        out[f"recall@{k}"] = round(sum(rel[:k]) / G, 6) if G else 0.0
    for k in Ks:
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel[:k], 1))
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(G, k) + 1))
        out[f"ndcg@{k}"] = round(dcg / ideal, 6) if ideal else 0.0
    return out


# --------------------------------------------------------------------------
async def build_dataset(plans: dict, recall_cache: dict) -> dict:
    dataset = {}
    for qid in sorted(plans):
        pool = await reconstruct_pool(plans[qid], recall_cache)
        ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
        trace = load_trace(qid)
        snap_pool = snapshot_ids(trace, "rerank_pool")
        llm_final = snapshot_ids(trace, "final_reranked")
        dataset[qid] = {"query_id": qid, "pool": pool, "ev_by_cid": ev_by_cid, "n_pool": len(pool),
                        "snap_pool": snap_pool, "llm_final": llm_final, "n_llm": len(llm_final)}
    return dataset


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plans = load_plan(M3R_PLAN)
    recall = load_recall_cache(M3R_CACHE)
    bq_by_id = {bq["query_id"]: bq for bq in load_pasa(DATA)}

    dataset = asyncio.run(build_dataset(plans, recall))

    # ---- 池一致性验证（EXPERIMENT_INVALID 守卫）----
    for qid in sorted(dataset):
        pq = dataset[qid]
        got_ids = [canonical_paper_id(ev) for ev in pq["pool"]]
        if set(got_ids) != set(pq["snap_pool"]):
            raise SystemExit(f"EXPERIMENT_INVALID: {qid} 池 set 不一致 "
                             f"(got {len(got_ids)} vs snapshot {len(pq['snap_pool'])})")
        if got_ids != pq["snap_pool"]:
            raise SystemExit(f"EXPERIMENT_INVALID: {qid} 池 order 不一致")
    print(f"[M4A3] 22/22 pool set+order 与 M3-R rerank_pool snapshot 逐位一致 → EXPERIMENT_INVALID 未触发。")

    # ---- 构建输入 ----
    items = []  # (qid, cid, ev, pre_rerank_rank)
    for qid in sorted(dataset):
        pq = dataset[qid]
        for rank, cid in enumerate([canonical_paper_id(ev) for ev in pq["pool"]], 1):
            items.append((qid, cid, pq["ev_by_cid"][cid], rank))
    pairs = [[bq_by_id[qid]["query"], build_passage(ev)] for qid, cid, ev, _ in items]
    total_pairs = len(pairs)
    print(f"[M4A3] 输入构建完成：{len(dataset)} queries，{total_pairs} (query, passage) pairs。")

    # ---- token 统计（诊断）----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ts = [token_stats(tok, ev) for _, _, ev, _ in items]
    n_trunc = sum(1 for t in ts if t["was_truncated"])
    n_abs = sum(1 for t in ts if t["has_abstract"])
    mean_tokens = statistics.mean(t["tokens"] for t in ts)
    print(f"  token: mean={mean_tokens:.1f} trunc={n_trunc}/{total_pairs} abstract={n_abs}/{total_pairs}")

    # ---- 模型加载（失败→RESOURCE_BLOCKED, STOP）----
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    t_load0 = time.time()
    try:
        model = CrossEncoder(MODEL_NAME, max_length=MAX_LEN, device=DEVICE, activation_fn=nn.Identity())
        model.model.eval()
    except Exception as e:  # noqa: BLE001
        print(f"STRONG_RERANKER_RESOURCE_BLOCKED: 模型加载失败 → {e}")
        sys.exit(0)
    t_load = time.time() - t_load0
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[M4A3] BGE 加载完成：load={t_load:.1f}s，params={n_params/1e6:.0f}M，device={DEVICE}（CUDA={model.model.device.type}）")

    # ---- 打分两次（determinism，带断点：pass 完成即存盘，被杀可续跑）----
    def run_pass(path: Path) -> tuple[list[float], float]:
        t0p = time.time()
        scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False,
                               processing_kwargs={"text": {"max_length": MAX_LEN, "truncation": True}})
        dt = time.time() - t0p
        np.save(path, np.asarray(scores))
        path.with_suffix(".txt").write_text(f"{dt:.3f}")
        print(f"  pass saved → {path.name}（{dt:.0f}s）", flush=True)
        return list(scores), dt

    def load_pass(path: Path):
        arr = list(np.load(path))
        t = float(path.with_suffix(".txt").read_text()) if path.with_suffix(".txt").exists() else 0.0
        print(f"[M4A3] 续跑：加载 {path.name} 已存分数（{len(arr)}，time={t:.0f}s）", flush=True)
        return arr, t

    s1 = s2 = None
    t_pass1 = t_pass2 = 0.0
    if SCORE_P1.exists():
        s1, t_pass1 = load_pass(SCORE_P1)
    if SCORE_P2.exists():
        s2, t_pass2 = load_pass(SCORE_P2)
    if s1 is None:
        s1, t_pass1 = run_pass(SCORE_P1)
    if s2 is None:
        s2, t_pass2 = run_pass(SCORE_P2)
    assert len(s1) == len(s2) == total_pairs, f"分数长度异常 {len(s1)}/{len(s2)}/{total_pairs}"
    t_pass1 = t_pass1 or 0.0
    t_pass2 = t_pass2 or 0.0
    max_diff = max(abs(a - b) for a, b in zip(s1, s2))
    print(f"[M4A3] 打分完成：pass1={t_pass1:.1f}s pass2={t_pass2:.1f}s max|Δscore|={max_diff:.3e}")

    # ---- 排序（bge_score DESC, pre_rerank_rank ASC, canonical_id ASC）----
    def rank_rows(scores: list[float]) -> dict[str, list[dict]]:
        out = {}
        for (qid, cid, ev, pre), sc in zip(items, scores):
            out.setdefault(qid, []).append({"cid": cid, "bge_score": sc, "pre": pre})
        for qid in out:
            out[qid].sort(key=lambda r: (-r["bge_score"], r["pre"], r["cid"]))
        return out

    r1 = rank_rows(s1)
    r2 = rank_rows(s2)
    rank_identical = all([r["cid"] for r in r1[q]] == [r["cid"] for r in r2[q]] for q in r1)
    top20_jaccard = all(
        set(r["cid"] for r in r1[q][:TOP_K]) == set(r["cid"] for r in r2[q][:TOP_K]) for q in r1)

    # ---- evaluator（gold 只在排序后读取）----
    gg_by_q = {qid: gold_groups(bq_by_id.get(qid, {})) for qid in dataset}

    def selected_evs(qid, n):
        pq = dataset[qid]
        return select_evs([r["cid"] for r in r1[qid][:n]], pq["ev_by_cid"])

    # 主比较：matched-N（N_q=LLM final count，冻结，不读 gold）
    matched_rows = []
    bge_mn_f1s, bge_top20_f1s = [], []
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        N = pq["n_llm"]
        m_bge = f1_of(selected_evs(qid, N), gg)
        m_bge20 = f1_of(selected_evs(qid, TOP_K), gg)
        bge_mn_f1s.append(m_bge["f1"])
        bge_top20_f1s.append(m_bge20["f1"])
        matched_rows.append({"query_id": qid, "llm_count_N": N, "bge_matched_count": N,
                             "bge_matched_N_f1": m_bge["f1"], "bge_matched_N_precision": m_bge["precision"],
                             "bge_matched_N_recall": m_bge["recall"],
                             "bge_top20_f1": m_bge20["f1"], "bge_top20_precision": m_bge20["precision"],
                             "bge_top20_recall": m_bge20["recall"]})
    with open(OUT / "m4a3_matched_cardinality.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "llm_count_N", "bge_matched_count", "bge_matched_N_f1", "bge_matched_N_precision",
                "bge_matched_N_recall", "bge_top20_f1", "bge_top20_precision", "bge_top20_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in matched_rows:
            w.writerow(r)
    bge_mn_f1 = statistics.mean(bge_mn_f1s)
    bge_top20_f1 = statistics.mean(bge_top20_f1s)

    def unique_gold(select_n):
        u = set()
        inst = 0
        for qid in sorted(dataset):
            pq = dataset[qid]
            evs = select_evs([r["cid"] for r in r1[qid][:select_n(qid)]], pq["ev_by_cid"])
            grps = matched_groups(evs, gg_by_q[qid])
            u |= {(qid, gi) for gi in grps}
            inst += len(grps)
        return len(u), inst

    bge_mn_gold, bge_mn_inst = unique_gold(lambda q: dataset[q]["n_llm"])
    bge_top20_gold, bge_top20_inst = unique_gold(lambda q: TOP_K)
    print(f"[M4A3] matched-N F1={bge_mn_f1:.4f} (gold={bge_mn_gold}, inst={bge_mn_inst}) | "
          f"Top-20 F1={bge_top20_f1:.4f} (gold={bge_top20_gold})")

    # ---- ranking metrics（@20，与 LLM/MiniLM 同口径）----
    rm_rows = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        gg = gg_by_q[qid]
        ce_ids = [r["cid"] for r in r1[qid]]
        m = rank_metrics(ce_ids, pq["ev_by_cid"], gg)
        rm_rows.append({"query_id": qid, **m})
    agg_row = {"query_id": "MEAN"}
    for col in rm_rows[0]:
        if col != "query_id":
            agg_row[col] = round(statistics.mean(r[col] for r in rm_rows), 4)
    rm_rows.append(agg_row)
    with open(OUT / "m4a3_ranking_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rm_rows[0].keys()))
        w.writeheader()
        for r in rm_rows:
            w.writerow(r)
    print(f"[M4A3] BGE MRR@20={agg_row['mrr@20']} MAP@20={agg_row['map@20']} R@5={agg_row['recall@5']} "
          f"NDCG@5={agg_row['ndcg@5']} NDCG@20={agg_row['ndcg@20']}")

    # ---- BGE rankings CSV（matched-N 输出全量排序，冻结分数）----
    with open(OUT / "m4a3_bge_rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "bge_rank", "canonical_id", "title", "bge_score", "pre_rerank_rank"])
        for qid in sorted(dataset):
            pq = dataset[qid]
            for i, r in enumerate(r1[qid], 1):
                w.writerow([qid, i, r["cid"], pq["ev_by_cid"][r["cid"]].identity.title or "",
                            r["bge_score"], r["pre"]])

    # ---- determinism ----
    f1_identical = all(
        f1_of(select_evs([r["cid"] for r in r1[q][:dataset[q]["n_llm"]]], dataset[q]["ev_by_cid"]), gg_by_q[q])["f1"] ==
        f1_of(select_evs([r["cid"] for r in r2[q][:dataset[q]["n_llm"]]], dataset[q]["ev_by_cid"]), gg_by_q[q])["f1"]
        for q in dataset)
    det_ok = rank_identical and top20_jaccard and f1_identical
    print(f"[M4A3] determinism: max|Δscore|={max_diff:.3e} rank_identical={rank_identical} "
          f"Top20_Jaccard={1.0 if top20_jaccard else 0.0} F1_identical={f1_identical}")

    # ---- gold comparison（重点案例，只解释）----
    gold_rows = build_gold_rows(dataset, r1, bq_by_id, gg_by_q)

    # ---- efficiency ----
    eff = {"model": MODEL_NAME, "n_params": n_params, "device": DEVICE, "model_load_s": round(t_load, 1),
           "pass1_infer_s": round(t_pass1, 1), "pass2_infer_s": round(t_pass2, 1),
           "total_infer_s": round(t_pass1 + t_pass2, 1), "pairs": total_pairs,
           "pairs_per_sec": round(total_pairs / max(t_pass1, 1e-6), 1),
           "mean_query_latency_s": round(max(t_pass1, 1e-6) / len(dataset), 2),
           "gen_llm_calls": 0, "openalex_http": 0,
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
           "peak_rss_note": "本进程（续跑加载模型+后处理）峰值；打分进程 ps 观测 ≈1.5GB"}
    with open(OUT / "m4a3_efficiency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(eff.keys()))
        w.writeheader()
        w.writerow(eff)
    print(f"[M4A3] efficiency: load={t_load:.1f}s pass1={t_pass1:.1f}s "
          f"pairs/s={total_pairs/t_pass1:.0f} mean_query={t_pass1/len(dataset):.1f}s peakRSS={eff['peak_rss_mb']}MB")

    # ---- vs 表 + decision ----
    vs_rows = [
        {"variant": "M3-R LLM", "selection": "variable-N (threshold)", "f1": REF_LLM["f1"], "final_gold": REF_LLM["final_gold"],
         "mrr@20": REF_LLM["mrr"], "map@20": REF_LLM["map"], "ndcg@5": REF_LLM["ndcg5"], "gen_llm_calls": REF_LLM["calls"]},
        {"variant": "RRF", "selection": "Top20", "f1": REF_RRF["f1"], "final_gold": REF_RRF["final_gold"],
         "mrr@20": "", "map@20": "", "ndcg@5": "", "gen_llm_calls": REF_RRF["calls"]},
        {"variant": "MiniLM CE", "selection": "matched LLM N", "f1": REF_MINI["f1"], "final_gold": REF_MINI["final_gold"],
         "mrr@20": REF_MINI["mrr"], "map@20": REF_MINI["map"], "ndcg@5": REF_MINI["ndcg5"], "gen_llm_calls": REF_MINI["calls"]},
        {"variant": "BGE v2-m3", "selection": "matched LLM N", "f1": round(bge_mn_f1, 4), "final_gold": bge_mn_gold,
         "mrr@20": agg_row["mrr@20"], "map@20": agg_row["map@20"], "ndcg@5": agg_row["ndcg@5"], "gen_llm_calls": 0},
        {"variant": "BGE v2-m3", "selection": "Top20 (辅助)", "f1": round(bge_top20_f1, 4), "final_gold": bge_top20_gold,
         "mrr@20": "", "map@20": "", "ndcg@5": "", "gen_llm_calls": 0},
    ]
    with open(OUT / "m4a3_vs_llm_vs_minilm_vs_rrf.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "selection", "f1", "final_gold", "mrr@20", "map@20", "ndcg@5", "gen_llm_calls"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in vs_rows:
            w.writerow(r)

    write_determinism(max_diff, rank_identical, top20_jaccard, f1_identical, det_ok, eff)
    write_decision(bge_mn_f1, bge_mn_gold, bge_mn_inst, bge_top20_f1, agg_row, det_ok,
                   eff, gold_rows, vs_rows, t0)
    print(f"[M4A3] 产物已写入 {OUT}/；总耗时 {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------
def build_gold_rows(dataset, r1, bq_by_id, gg_by_q) -> list[dict]:
    loss_rows = {f"{r['query_id']}|{r['canonical_id']}": r for r in csv.DictReader(open(M4_LOSS, encoding="utf-8"))
                 if r["loss_stage"] == "LOST_RERANKER"}
    rrf = {f"{r['query_id']}|{r['canonical_id']}": r["rrf_rank"] for r in csv.DictReader(open(M4A1_RECOVERY, encoding="utf-8"))}
    mini = {f"{r['query_id']}|{r['canonical_id']}": int(r["ce_rank"])
            for r in csv.DictReader(open(M4A2_RANKINGS, encoding="utf-8"))}
    rows = []
    for key, lr in sorted(loss_rows.items()):
        qid, cid = key.split("|")
        pq = dataset[qid]
        bge_rank = next((i for i, r in enumerate(r1[qid], 1) if r["cid"] == cid), None)
        llm_rank = next((i for i, c in enumerate(pq["llm_final"], 1) if c == cid), None)
        rows.append({"query_id": qid, "canonical_id": cid, "title_n": lr["title_n"],
                     "pre_rerank_rank": lr["pre_rerank_rank"],
                     "llm_rank": llm_rank or 0, "llm_selected": 1 if llm_rank else 0,
                     "rrf_top20_rank": rrf.get(key, ""),
                     "minilm_rank": mini.get(key, ""),
                     "bge_rank": bge_rank or 0,
                     "bge_selected": 1 if bge_rank and bge_rank <= pq["n_llm"] else 0})
    with open(OUT / "m4a3_gold_comparison.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "canonical_id", "title_n", "pre_rerank_rank", "llm_rank", "llm_selected",
                "rrf_top20_rank", "minilm_rank", "bge_rank", "bge_selected"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def write_determinism(max_diff, rank_identical, top20_jaccard, f1_identical, det_ok, eff) -> None:
    lines = [
        "# M4A3 DETERMINISM CHECK", "",
        f"- model: {MODEL_NAME}（device={eff['device']}，inference-only，model.eval()，params={eff['n_params']/1e6:.0f}M）",
        f"- activation_fn=nn.Identity → bge_score 为 positive-class 原始 logit（单调等价，仅用于排序）。",
        f"- 同一 frozen pool 完整打分两次（pass1=正式结果，pass2=校验）。",
        "",
        f"- max |score1 - score2| = {max_diff:.3e}  (tolerance <=1e-6 → {'PASS' if max_diff <= 1e-6 else 'FAIL'})",
        f"- rank_order_identical = {rank_identical}",
        f"- Top-20 Jaccard = {1.0 if top20_jaccard else 0.0}",
        f"- matched-N F1 identical = {f1_identical}",
        f"- 校验 queries = 22",
        "",
        "**BGE DETERMINISM PASSED**" if det_ok else "**BGE DETERMINISM_FAILURE → STOP**",
        "",
        "## 运行成本",
        f"- model load = {eff['model_load_s']}s；两遍 inference = {eff['pass1_infer_s']}s + {eff['pass2_infer_s']}s",
        f"- pairs/s = {eff['pairs_per_sec']}；per-query mean latency = {eff['mean_query_latency_s']}s；peak RSS = {eff['peak_rss_mb']} MB",
        f"- generative LLM calls = {eff['gen_llm_calls']}；OpenAlex physical HTTP = {eff['openalex_http']}",
    ]
    (OUT / "m4a3_determinism.md").write_text("\n".join(lines), encoding="utf-8")


def write_decision(bge_mn_f1, bge_mn_gold, bge_mn_inst, bge_top20_f1, agg_row, det_ok,
                   eff, gold_rows, vs_rows, t0) -> None:
    lines = ["# M4A3_STRONGER_RERANKER 决策报告", "",
             f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；模型：BAAI/bge-reranker-v2-m3（inference-only, eval, CPU）。",
             "- 上游完全冻结：M3-R_APPEND frozen pool（raw=25, pool=22）；Planner/Rescue/OpenAlex/Citation/Reference/Metadata/Prekeep 未重跑。",
             "- 池一致性：22/22 set+order 与 M3-R rerank_pool snapshot 逐位一致 → EXPERIMENT_INVALID 未触发。",
             "- 输入：query=原始问题；passage=title+\"\\n\"+abstract（缺 abstract 仅 title）；max_length=512=模型官方推荐值；0 次读 Gold/Oracle/历史 loss map。",
             ""]
    lines.append(f"- 主比较（matched-N，N_q=LLM final count）：mean F1={bge_mn_f1:.4f}，final unique Gold={bge_mn_gold}，instances={bge_mn_inst}。")
    lines.append(f"- Top-20（辅助）：mean F1={bge_top20_f1:.4f}。")
    lines.append(f"- Ranking metrics：MRR@20={agg_row['mrr@20']} MAP@20={agg_row['map@20']} R@5={agg_row['recall@5']} "
                 f"R@10={agg_row['recall@10']} R@20={agg_row['recall@20']} NDCG@5={agg_row['ndcg@5']} "
                 f"NDCG@10={agg_row['ndcg@10']} NDCG@20={agg_row['ndcg@20']}。")
    lines.append("")
    lines.append("## 对照表（vs LLM / MiniLM / RRF）")
    lines.append("")
    lines.append("| variant | selection | F1 | final Gold | MRR@20 | MAP@20 | NDCG@5 | gen LLM calls |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in vs_rows:
        lines.append(f"| {r['variant']} | {r['selection']} | {r['f1']} | {r['final_gold']} | {r['mrr@20']} | "
                     f"{r['map@20']} | {r['ndcg@5']} | {r['gen_llm_calls']} |")
    lines.append("")
    lines.append("## Determinism")
    lines.append("")
    lines.append(f"- det_ok={det_ok}（max|Δscore| / rank / Jaccard / F1 详见 m4a3_determinism.md）。")
    lines.append("")
    lines.append("## 系统性 FN 对照（Q6×3/5、Q15 RLHF、Q47 FinEval ×2，只解释不改模型）")
    lines.append("")
    lines.append("| query | gold | pre_rank | LLM rank | LLM sel | RRF rank | MiniLM rank | BGE rank | BGE sel |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in gold_rows:
        lines.append(f"| {r['query_id']} | {r['title_n'][:24]} | {r['pre_rerank_rank']} | {r['llm_rank']} | {r['llm_selected']} "
                     f"| {r['rrf_top20_rank']} | {r['minilm_rank']} | {r['bge_rank']} | {r['bge_selected']} |")
    lines.append("")
    lines.append("## Efficiency")
    lines.append("")
    lines.append(f"- model load={eff['model_load_s']}s；pass1 infer={eff['pass1_infer_s']}s（pass2={eff['pass2_infer_s']}s）；"
                 f"pairs/s={eff['pairs_per_sec']}；per-query={eff['mean_query_latency_s']}s；peak RSS={eff['peak_rss_mb']}MB。")
    lines.append(f"- generative LLM calls = 0；OpenAlex physical HTTP = 0。")
    lines.append("")
    lines.append("## Decision Gate（参考 LLM matched-N F1=0.0664）")
    lines.append("")
    if bge_mn_f1 >= 0.0664:
        gate = "STRONG_DETERMINISTIC_RERANKER_COMPETITIVE"
        note = "BGE 达到/超过 LLM 且确定性、0 LLM calls → BGE 成为 production candidate。"
    elif bge_mn_f1 >= 0.060:
        gate = "STRONG_RERANKER_NEAR_COMPETITIVE"
        note = "BGE 接近 LLM。不再测试其他 reranker。综合 F1/determinism/latency/deployment cost 决定 production 用 LLM 或 BGE。"
    else:
        gate = "DETERMINISTIC_RERANKER_QUALITY_LIMIT"
        note = "BGE 未达 LLM 且 < 0.060。不再测试其他 reranker。production 暂时保留 M3-R LLM Reranker。"
    lines.append(f"- BGE matched-N F1={bge_mn_f1:.4f}（LLM=0.0664，MiniLM=0.0598）。")
    lines.append(f"- **判定：{gate}**。{note}")
    lines.append("")
    lines.append("## RERANKER_RESEARCH_FREEZE")
    lines.append("")
    lines.append("- 本轮后冻结 M4：禁止自动进入第三个 reranker / ensemble / fine-tuning / threshold search / K tuning / LLM voting / CE+RRF fusion。")
    lines.append("- 下一研发主线转向：Iterative / Agentic Retrieval。")
    lines.append("")
    lines.append(f"**本轮（M4A3）到此为止：STOP。总耗时 {time.time()-t0:.0f}s。**")
    (OUT / "m4a3_decision.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
