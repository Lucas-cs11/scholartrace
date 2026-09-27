"""M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER：确定性轻量 Cross-Encoder 精排探针。

目的：用一个确定性的 Cross-Encoder（cross-encoder/ms-marco-MiniLM-L6-v2）替代生成式 LLM
Reranker，验证 complex academic query + paper semantic content 能否在
  0 generative LLM reranker calls；完全可复现；较低计算成本；
条件下达到或接近 M3-R LLM baseline（F1=0.0664）。本轮是 architecture probe，不 fine-tune。

冻结上游：主实验只用 M3-R_APPEND frozen pool（observed best production baseline，
F1=0.0664，raw unique Gold=25，pool unique Gold=22）。禁止重跑 Planner/Rescue/OpenAlex/
Citation/Reference/Metadata/Prekeep。必须读取已冻结 pre-rerank 池；若候选 ID/顺序与 M3-R
trace 快照不一致 → EXPERIMENT_INVALID=true → STOP。M3.1 不作为主实验。

输入：query = 原始用户问题；passage = title + "\\n" + abstract（abstract 缺失则仅 title）。
tokenizer max_length=512，truncation=True（title 在前，天然优先保留，剩余 token 给 abstract）。
不使用 Gold/Oracle/evaluator label/citation count/historical loss。

打分：ce_score = CrossEncoder(query, passage) 的 positive-class score；排序 ce_score DESC，
tie-break pre_rerank_rank ASC，canonical_id ASC。禁止 RRF/LLM/retrieval score fusion、
safepass bonus、handcrafted score。

Determinism：同一 frozen pool 完整打分两次。验证 candidate scores 数值差 <=1e-6、
rank order 相同、Top20 Jaccard=1.0、F1 相同；任一失败 → CROSS_ENCODER_DETERMINISM_FAILURE → STOP。

Gold isolation：CE 打分/排序阶段 0 次读 gold；gold 仅 evaluator 在排序完成后读取。泄漏 → EXPERIMENT_INVALID。

产物（eval/runs/m4a2_ce/）：
  m4a2_ce_rankings.csv        —— 每 query 每 pool 候选的 CE 排序（top-20）+ ce_score/pre_rerank_rank/abstract 统计
  m4a2_k_sensitivity.csv      —— Top-5/10/15/20 的 mean P/R/F1 曲线
  m4a2_gold_comparison.csv    —— M4-0 LOST_RERANKER gold 的 LLM vs RRF vs CE 保留对比
  m4a2_abstract_coverage.csv  —— pool/gold abstract coverage + 有无 abstract 的 mean CE rank + truncated 数
  m4a2_determinism_check.md
  m4a2_vs_llm_vs_rrf.csv      —— M3-R LLM / M3-R RRF / M4A2 CE 三方对照
  m4a2_decision.md

用法：
  python scripts/run_m4a2_ce.py run           # 全量 22 查询
  python scripts/run_m4a2_ce.py run --limit 2 # 前 2 条 query（smoke）
"""
from __future__ import annotations

import asyncio
import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import warnings

import torch
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import _norm_doi, _norm_title, _norm_title_letters, compute_p_r_f1, match_gold
from src.observability.canonical import canonical_paper_id
from src.planner import ASSOC_INTENT
from src.schemas import PaperEvidence, RankResult
from src.search import B1_MAX_SUBQUERIES, SearchEngine
from src.telemetry import Telemetry
from scripts.eval_benchmark import load_pasa

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
MAX_LEN = 512
TOP_K = 20
K_RANGE = (5, 10, 15, 20)
BATCH_SIZE = 32
DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
OUT = Path("eval/runs/m4a2_ce")
M3R_PLAN = "eval/runs/m3r_append/m3r_query_plans.jsonl"
M3R_CACHE = "eval/cache/m3r_append/recall_cache.jsonl"
M3R_TRACE = Path("eval/diagnostics/m3r_append")

RANKINGS = "m4a2_ce_rankings.csv"
K_SENS = "m4a2_k_sensitivity.csv"
GOLD_CMP = "m4a2_gold_comparison.csv"
ABS_CVG = "m4a2_abstract_coverage.csv"
DETERM = "m4a2_determinism_check.md"
VS = "m4a2_vs_llm_vs_rrf.csv"
DECISION = "m4a2_decision.md"

REF_LLM_F1 = 0.0664   # M3-R observed best production baseline
REF_LLM_FINAL = 16
REF_LLM_CALLS = 77
REF_RRF_F1 = 0.0241   # M4A1 M3-R_RRF top-20
REF_RRF_FINAL = 7
REF_RRF_CALLS = 0


# --------------------------------------------------------------------------
# 加载（全部离线，确定性）
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


def snapshot_pool_ids(trace: dict) -> list[str]:
    for s in trace.get("candidate_snapshots", []):
        if s.get("stage") == "rerank_pool":
            return list(s.get("candidate_ids", []))
    return []


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
# 池重建 + 快照校验（0 联网，确定性）
# --------------------------------------------------------------------------
async def reconstruct_pool(plan: dict, recall_cache: dict) -> list[PaperEvidence]:
    engine = SearchEngine(enable_citation_expansion=False, assoc_safepass=True)
    q = plan["query"]
    engine._plan_cache[q] = {"v": 2, "ir": plan["ir"], "subs": plan["subs"]}
    engine._recall_cache = recall_cache
    ir, evs = await engine._plan_and_recall(q, Telemetry(), [], use_cache=True)
    lex = engine._lexical_rank(q, evs)
    return engine._build_rerank_pool(evs, lex)


async def build_dataset(plans: dict, recall_cache: dict) -> dict:
    """每 query：pool + recall_evs（executed subs 召回并集，raw gold 依据）+ missing_subs。"""
    dataset = {}
    for qid in sorted(plans):
        plan = plans[qid]
        pool = await reconstruct_pool(plan, recall_cache)
        # 快照校验：canonical id set+order 必须与 M3-R trace 一致
        snap = snapshot_pool_ids(load_trace(qid))
        cand_ids = [canonical_paper_id(ev) for ev in pool]
        match = (cand_ids == snap)
        esubs = executed_subs(plan)
        missing = [s["query_text"] for s in esubs if s["query_text"] not in recall_cache]
        recall_evs: list[PaperEvidence] = []
        seen_cid: set[str] = set()
        for s in esubs:
            for e in recall_cache.get(s["query_text"], []):
                c = canonical_paper_id(e)
                if c not in seen_cid:
                    seen_cid.add(c)
                    recall_evs.append(e)
        dataset[qid] = {"query_id": qid, "query": plan["query"], "pool": pool,
                        "recall_evs": recall_evs, "missing_subs": missing,
                        "snapshot_match": match, "snapshot_len": len(snap), "pool_len": len(pool)}
    return dataset


# --------------------------------------------------------------------------
# Cross-Encoder 打分（确定性；同一输入两次打分用于 determinism 校验）
# --------------------------------------------------------------------------
def build_passage(ev: PaperEvidence) -> tuple[str, bool]:
    title = ev.identity.title or ""
    abstract = ev.abstract or ""
    has_abstract = bool(abstract.strip())
    passage = (title + "\n" + abstract) if has_abstract else title
    return passage, has_abstract


def token_stats(tok, query: str, passage: str, title: str, abstract: str) -> dict:
    enc = tok(query, passage, truncation=True, max_length=MAX_LEN)
    # 未截断长度只用于 was_truncated 判定；verbose=False 抑制 tokenizer 的超长 logger 警告（非错误，属预期）
    enc_full = tok(query, passage, truncation=False, verbose=False)
    return {
        "input_token_count": len(enc["input_ids"]),
        "was_truncated": len(enc_full["input_ids"]) > MAX_LEN,
        "title_tokens": len(tok(title, add_special_tokens=False, verbose=False).input_ids),
        "abstract_tokens": len(tok(abstract, add_special_tokens=False, verbose=False).input_ids) if abstract else 0,
    }


def flatten_items(dataset: dict) -> list[tuple]:
    """(qid, cid, query, passage, ev, has_abstract) 固定顺序。"""
    items = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        for ev in pq["pool"]:
            cid = canonical_paper_id(ev)
            passage, has_abs = build_passage(ev)
            items.append((qid, cid, pq["query"], passage, ev, has_abs))
    return items


def ce_pass(ce, tok, dataset: dict, items: list[tuple]) -> dict[tuple, dict]:
    """一次完整打分 pass → stats[(qid, cid)] = {ce_score, input_token_count, was_truncated, ...}。"""
    pairs = [(it[2], it[3]) for it in items]
    # processing_kwargs 显式强制 truncation=True, max_length=512（title 在 passage 前，天然优先保留）
    scores = ce.predict(
        pairs, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True,
        processing_kwargs={"text": {"max_length": MAX_LEN, "truncation": True}},
    )
    stats = {}
    for it, sc in zip(items, scores):
        qid, cid, query, passage, ev, has_abs = it
        st = token_stats(tok, query, passage, ev.identity.title or "", ev.abstract or "")
        st["has_abstract"] = has_abs
        st["ce_score"] = float(sc)
        stats[(qid, cid)] = st
    return stats


def _rank_pool(qid: str, pool: list[PaperEvidence], stats: dict[tuple, dict]) -> tuple[list[str], dict[str, int]]:
    cids = [canonical_paper_id(ev) for ev in pool]
    pre = {c: i + 1 for i, c in enumerate(cids)}
    ranked = sorted(cids, key=lambda c: (-stats[(qid, c)]["ce_score"], pre[c], c))
    return ranked, pre


# --------------------------------------------------------------------------
# evaluator：gold 只在此阶段读取（排序完成后）
# --------------------------------------------------------------------------
def gold_groups(query: dict) -> list[set[str]]:
    return match_gold(query)


def metrics_at_k(ranked_ids: list[str], pool: list[PaperEvidence], gold_groups: list[set[str]], k: int) -> dict:
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
    ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
    final_evs = [ev_by_cid[c] for c in ranked_ids[:TOP_K] if c in ev_by_cid]
    return {
        "raw": _matched_groups(recall_evs, gold_groups),
        "pool": _matched_groups(pool, gold_groups),
        "final": _matched_groups(final_evs, gold_groups),
        "final_inst": len(_matched_groups(final_evs, gold_groups)),
    }


def mean(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def run(limit: int = 0) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    plans = load_plan(M3R_PLAN)
    recall = load_recall_cache(M3R_CACHE)
    bq_by_id = {bq["query_id"]: bq for bq in load_pasa(DATA)}
    if limit > 0:
        plans = {k: plans[k] for k in sorted(plans)[:limit]}

    print(f"[M4A2] 冻结输入：M3-R plan={len(plans)} queries, recall cache={len(recall)} keys, "
          f"trace dir={len(list(M3R_TRACE.glob('trace_*.json')))} files")

    dataset = asyncio.run(build_dataset(plans, recall))
    bad = [qid for qid, pq in dataset.items() if not pq["snapshot_match"]]
    for qid, pq in dataset.items():
        mark = "OK" if pq["snapshot_match"] else "MISMATCH"
        print(f"  verify {qid}: pool={pq['pool_len']} snapshot={pq['snapshot_len']} {mark}")
    if bad:
        with open(OUT / DECISION, "w", encoding="utf-8") as f:
            f.write(f"# M4A2 EXPERIMENT_INVALID\n\n重建池与 M3-R 快照不一致：{bad}\nSTOP。\n")
        print(f"!! EXPERIMENT_INVALID: {bad} → STOP（不评分）。")
        return
    print("  pool 快照校验全部通过（set+order 一致）。")
    miss_q = [(qid, pq["missing_subs"]) for qid, pq in dataset.items() if pq["missing_subs"]]
    for qid, m in miss_q:
        print(f"  !! {qid}: {len(m)} executed subs 无 recall cache（只报告，不猜测）：{m}")

    # ---- 模型加载 + 两次打分 ----
    t_load0 = time.time()
    ce = CrossEncoder(MODEL_NAME, max_length=MAX_LEN)
    ce.model.eval()  # inference only
    t_load = time.time() - t_load0

    items = flatten_items(dataset)
    tok = ce.tokenizer

    t_inf0 = time.time()
    pass1 = ce_pass(ce, tok, dataset, items)
    pass2 = ce_pass(ce, tok, dataset, items)
    t_inf = time.time() - t_inf0

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024  # macOS: bytes; Linux: KB
    print(f"  模型加载 {t_load:.1f}s；两次打分 {t_inf:.1f}s（{len(items)} 候选 ×2）；peak RSS={rss_bytes/1e6:.0f} MB")

    # ---- per-query CE 排序（pass1 为正式结果）----
    for qid, pq in dataset.items():
        ranked, pre = _rank_pool(qid, pq["pool"], pass1)
        pq["ranked"] = ranked
        pq["pre_rerank"] = pre
        ranked2, _ = _rank_pool(qid, pq["pool"], pass2)
        pq["ranked2"] = ranked2

    # ---- determinism（分数 + 排序 + Jaccard，gold 无关）----
    det = determinism_check(dataset, pass1, pass2)

    # ---- evaluator（gold 在此之后读取）----
    for qid, pq in dataset.items():
        bq = bq_by_id.get(qid, {})
        gg = gold_groups(bq)
        pq["gold_groups"] = gg
        pq["k_metrics"] = {kk: metrics_at_k(pq["ranked"], pq["pool"], gg, kk) for kk in K_RANGE}
        pq["k_metrics2"] = {kk: metrics_at_k(pq["ranked2"], pq["pool"], gg, kk) for kk in K_RANGE}
        pq["funnel"] = funnel(pq["pool"], pq["ranked"], gg, pq["recall_evs"])
        pq["funnel2"] = funnel(pq["pool"], pq["ranked2"], gg, pq["recall_evs"])
        pq["final_top20"] = pq["ranked"][:TOP_K]
    det = finalize_determinism(det, dataset)

    write_rankings(dataset, pass1)
    k_rows = write_k_sensitivity(dataset)
    vs_row, per_query_f1 = write_vs(dataset, bq_by_id)
    gold_rows = write_gold_comparison(dataset, pass1)
    abs_rows = write_abstract_coverage(dataset, pass1)
    write_determinism(det, ce, t_load, t_inf, rss_bytes)
    write_decision(vs_row, k_rows, gold_rows, abs_rows, det, dataset, per_query_f1,
                   t_load, t_inf, rss_bytes, miss_q)

    # ---- 控制台汇总 ----
    n = len(dataset)
    f1_20 = mean([pq["k_metrics"][20]["f1"] for pq in dataset.values()])
    p_20 = mean([pq["k_metrics"][20]["precision"] for pq in dataset.values()])
    r_20 = mean([pq["k_metrics"][20]["recall"] for pq in dataset.values()])
    raw_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["raw"]}
    pool_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["pool"]}
    fin_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["final"]}
    fin_inst = sum(pq["funnel"]["final_inst"] for pq in dataset.values())
    print(f"\n=== M4A2 CE（M3-R frozen pool, {n} queries）===")
    print(f"  mean F1={f1_20:.4f}  P={p_20:.4f}  R={r_20:.4f}")
    print(f"  raw={len(raw_set)} pool={len(pool_set)} final={len(fin_set)} final_inst={fin_inst}")
    print(f"  CE 打分 determinism: {det['ok']}  LLM calls=0  OpenAlex HTTP=0")
    print(f"  total {time.time()-t_start:.0f}s")


# --------------------------------------------------------------------------
def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def determinism_check(dataset: dict, pass1: dict, pass2: dict) -> dict:
    """分数 + 排序 + Jaccard（gold 无关，可在 evaluator 前执行）。F1 一致性由调用方在 evaluator 后补齐。"""
    max_diff = 0.0
    all_rank_ok = True
    total_j = 0.0
    n = 0
    for qid, pq in dataset.items():
        for cid in {c for (q, c) in pass1 if q == qid}:
            d = abs(pass1[(qid, cid)]["ce_score"] - pass2[(qid, cid)]["ce_score"])
            max_diff = max(max_diff, d)
        if pq["ranked"] != pq["ranked2"]:
            all_rank_ok = False
        j = jaccard(pq["ranked"][:TOP_K], pq["ranked2"][:TOP_K])
        total_j += j
        n += 1
    return {"ok": False, "max_abs_score_diff": max_diff, "rank_order_identical": all_rank_ok,
            "f1_identical": None, "mean_top20_jaccard": round(total_j / n, 6) if n else 1.0,
            "queries": n}


def finalize_determinism(det: dict, dataset: dict) -> dict:
    """evaluator 后补齐 F1 一致性并计算最终 ok。"""
    all_f1_ok = all(pq["k_metrics"][20]["f1"] == pq["k_metrics2"][20]["f1"]
                    for pq in dataset.values())
    det["f1_identical"] = all_f1_ok
    ok = (det["max_abs_score_diff"] <= 1e-6 and det["rank_order_identical"] and all_f1_ok
          and abs(det["mean_top20_jaccard"] - 1.0) < 1e-9)
    det["ok"] = ok
    return det


# --------------------------------------------------------------------------
# 产物
# --------------------------------------------------------------------------
def write_rankings(dataset: dict, stats: dict) -> None:
    rows = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        for rk, cid in enumerate(pq["final_top20"], 1):
            st = stats[(qid, cid)]
            ev = next(ev for ev in pq["pool"] if canonical_paper_id(ev) == cid)
            rows.append({
                "query_id": qid, "ce_rank": rk, "canonical_id": cid,
                "title": (ev.identity.title or "")[:80],
                "ce_score": st["ce_score"], "pre_rerank_rank": pq["pre_rerank"][cid],
                "has_abstract": st["has_abstract"], "input_token_count": st["input_token_count"],
                "was_truncated": st["was_truncated"],
            })
    with open(OUT / RANKINGS, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "ce_rank", "canonical_id", "title", "ce_score", "pre_rerank_rank",
                "has_abstract", "input_token_count", "was_truncated"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_k_sensitivity(dataset: dict) -> list[dict]:
    k_rows = []
    for kk in K_RANGE:
        f1k = [pq["k_metrics"][kk]["f1"] for pq in dataset.values()]
        pk = [pq["k_metrics"][kk]["precision"] for pq in dataset.values()]
        rk = [pq["k_metrics"][kk]["recall"] for pq in dataset.values()]
        k_rows.append({"top_k": kk, "mean_f1": mean(f1k), "mean_precision": mean(pk),
                       "mean_recall": mean(rk)})
    with open(OUT / K_SENS, "w", newline="", encoding="utf-8") as f:
        cols = ["top_k", "mean_f1", "mean_precision", "mean_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in k_rows:
            w.writerow(r)
    return k_rows


def write_vs(dataset: dict, bq_by_id: dict) -> tuple[dict, dict]:
    n = len(dataset)
    f1_20 = mean([pq["k_metrics"][20]["f1"] for pq in dataset.values()])
    p_20 = mean([pq["k_metrics"][20]["precision"] for pq in dataset.values()])
    r_20 = mean([pq["k_metrics"][20]["recall"] for pq in dataset.values()])
    raw_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["raw"]}
    pool_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["pool"]}
    fin_set = {(qid, gi) for qid, pq in dataset.items() for gi in pq["funnel"]["final"]}
    fin_inst = sum(pq["funnel"]["final_inst"] for pq in dataset.values())
    ce_row = {
        "variant": "M4A2_CE", "queries": n, "mean_f1": f1_20, "mean_precision": p_20,
        "mean_recall": r_20, "raw_unique_gold": len(raw_set), "pool_unique_gold": len(pool_set),
        "final_unique_gold": len(fin_set), "final_query_gold_instances": fin_inst,
        "llm_reranker_calls": 0, "openalex_physical_http": 0,
    }
    llm_row = {"variant": "M3-R_LLM", "queries": 22, "mean_f1": REF_LLM_F1,
               "mean_precision": "", "mean_recall": "", "raw_unique_gold": 25,
               "pool_unique_gold": 22, "final_unique_gold": REF_LLM_FINAL,
               "final_query_gold_instances": "", "llm_reranker_calls": REF_LLM_CALLS,
               "openalex_physical_http": ""}
    rrf_row = {"variant": "M3-R_RRF", "queries": 22, "mean_f1": REF_RRF_F1,
               "mean_precision": "", "mean_recall": "", "raw_unique_gold": 25,
               "pool_unique_gold": 22, "final_unique_gold": REF_RRF_FINAL,
               "final_query_gold_instances": "", "llm_reranker_calls": REF_RRF_CALLS,
               "openalex_physical_http": 0}
    with open(OUT / VS, "w", newline="", encoding="utf-8") as f:
        cols = ["variant", "queries", "mean_f1", "mean_precision", "mean_recall",
                "raw_unique_gold", "pool_unique_gold", "final_unique_gold",
                "final_query_gold_instances", "llm_reranker_calls", "openalex_physical_http"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in (llm_row, rrf_row, ce_row):
            w.writerow(r)
    # per-query CE F1（供对照）
    per_query = {qid: pq["k_metrics"][20]["f1"] for qid, pq in dataset.items()}
    return ce_row, per_query


def write_gold_comparison(dataset: dict, stats: dict) -> list[dict]:
    """M4-0 LOST_RERANKER gold：LLM 最终保留 vs RRF rank vs CE rank/score/pre_rerank_rank。"""
    loss_map_path = Path("eval/runs/m4_reranker_stability/m4_gold_loss_map.csv")
    m4a1_path = Path("eval/runs/m4a1_rrf/m4a1_gold_recovery.csv")
    lost = []
    if loss_map_path.exists():
        for r in csv.DictReader(loss_map_path.open(encoding="utf-8")):
            if r.get("loss_stage") == "LOST_RERANKER":
                lost.append(r)
    rrf_by_key = {}
    if m4a1_path.exists():
        for r in csv.DictReader(m4a1_path.open(encoding="utf-8")):
            rrf_by_key[(r.get("query_id"), r.get("canonical_id"))] = r
    rows = []
    focus_q = {"RealScholarQuery_47": "Q47 FinEval x2",
               "RealScholarQuery_15": "Q15 RLHF Gold",
               "RealScholarQuery_6": "Q6 systematic FN"}
    for r in lost:
        qid = r["query_id"]
        cid = r["canonical_id"]
        llm_sel = int(r.get("llm_sel_freq", 0))
        pq = dataset.get(qid)
        if pq is None:
            continue
        ranked = pq["ranked"]
        ce_pos = ranked.index(cid) + 1 if cid in ranked else None
        st = stats.get((qid, cid), {})
        rrf = rrf_by_key.get((qid, cid), {})
        rows.append({
            "query_id": qid, "canonical_id": cid, "title_n": r.get("title_n", ""),
            "focus": focus_q.get(qid, ""),
            "llm_final_retained": "yes" if llm_sel > 0 else "no",
            "llm_sel_freq": llm_sel,
            "rrf_top20_retained": rrf.get("rrf_top20_retained", ""),
            "rrf_rank": rrf.get("rrf_rank", ""),
            "ce_top20_retained": "yes" if (ce_pos is not None and ce_pos <= TOP_K) else "no",
            "ce_rank": ce_pos if ce_pos is not None else "",
            "ce_score": st.get("ce_score", ""),
            "pre_rerank_rank": pq["pre_rerank"].get(cid, ""),
        })
    with open(OUT / GOLD_CMP, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "canonical_id", "title_n", "focus", "llm_final_retained",
                "llm_sel_freq", "rrf_top20_retained", "rrf_rank", "ce_top20_retained",
                "ce_rank", "ce_score", "pre_rerank_rank"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def write_abstract_coverage(dataset: dict, stats: dict) -> list[dict]:
    rows = []
    for qid in sorted(dataset):
        pq = dataset[qid]
        pool = pq["pool"]
        n_pool = len(pool)
        n_abs = sum(1 for ev in pool if stats[(qid, canonical_paper_id(ev))]["has_abstract"])
        n_trunc = sum(1 for ev in pool if stats[(qid, canonical_paper_id(ev))]["was_truncated"])
        ranks = {c: i + 1 for i, c in enumerate(pq["ranked"])}
        r_with = [ranks[canonical_paper_id(ev)] for ev in pool
                  if stats[(qid, canonical_paper_id(ev))]["has_abstract"]]
        r_without = [ranks[canonical_paper_id(ev)] for ev in pool
                     if not stats[(qid, canonical_paper_id(ev))]["has_abstract"]]
        # gold abstract coverage（evaluator 阶段，仅诊断）：gold 组在池内有匹配且匹配候选有 abstract 的比例
        ev_by_cid = {canonical_paper_id(ev): ev for ev in pool}
        gg = pq["gold_groups"]
        g_matched = 0
        g_abs = 0
        for gi, gk in enumerate(gg):
            matched_evs = [ev_by_cid[c] for c in pq["ranked"] if c in ev_by_cid and paper_keys(ev_by_cid[c]) & gk]
            if matched_evs:
                g_matched += 1
                if any(stats[(qid, canonical_paper_id(e))]["has_abstract"] for e in matched_evs):
                    g_abs += 1
        rows.append({
            "query_id": qid, "pool_size": n_pool, "pool_abstract_coverage": round(n_abs / n_pool, 4),
            "pool_candidates_with_abstract": n_abs, "truncated_candidates": n_trunc,
            "mean_ce_rank_with_abstract": round(mean(r_with), 2),
            "mean_ce_rank_without_abstract": round(mean(r_without), 2),
            "n_without_abstract": len(r_without),
            "gold_groups_matched_in_pool": g_matched,
            "gold_abstract_coverage": round(g_abs / g_matched, 4) if g_matched else "",
        })
    with open(OUT / ABS_CVG, "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "pool_size", "pool_abstract_coverage", "pool_candidates_with_abstract",
                "truncated_candidates", "mean_ce_rank_with_abstract", "mean_ce_rank_without_abstract",
                "n_without_abstract", "gold_groups_matched_in_pool", "gold_abstract_coverage"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def write_determinism(det: dict, ce, t_load: float, t_inf: float, rss_bytes: int) -> None:
    lines = ["# M4A2 DETERMINISM CHECK", "",
             f"- model: {MODEL_NAME} (max_length={MAX_LEN}, device=cpu, inference-only, model.eval())",
             f"- sentence-transformers/transformers/torch: "
             f"{ce.__class__.__module__.split('.')[0]} / transformers / torch {torch.__version__}",
             f"- 同一 frozen pool 完整打分两次（pass1=正式结果，pass2=校验）。",
             "",
             f"- max |score1 - score2| = {det['max_abs_score_diff']:.2e}  "
             f"(tolerance <=1e-6 → {'PASS' if det['max_abs_score_diff'] <= 1e-6 else 'FAIL'})",
             f"- rank_order_identical = {det['rank_order_identical']}",
             f"- Top-20 Jaccard = {det['mean_top20_jaccard']} (all queries)",
             f"- F1 identical (top-20) = {det['f1_identical']}",
             f"- 校验 queries = {det['queries']}",
             ""]
    if det["ok"]:
        lines.append("**CROSS-ENCODER DETERMINISM PASSED**（无 CROSS_ENCODER_DETERMINISM_FAILURE）。")
    else:
        lines.append("**CROSS_ENCODER_DETERMINISM_FAILURE → STOP**")
    lines.append("")
    lines.append("## 运行成本")
    lines.append(f"- model load time = {t_load:.1f}s")
    lines.append(f"- CE inference (2 passes, {det['queries']} queries) = {t_inf:.1f}s")
    lines.append(f"- peak RSS = {rss_bytes/1e6:.0f} MB (CPU, no VRAM)")
    lines.append(f"- generative LLM reranker calls = 0；OpenAlex physical HTTP = 0")
    (OUT / DETERM).write_text("\n".join(lines), encoding="utf-8")


def write_decision(vs_row, k_rows, gold_rows, abs_rows, det, dataset, per_query_f1,
                   t_load, t_inf, rss_bytes, miss_q) -> None:
    f1 = vs_row["mean_f1"]
    fin = vs_row["final_unique_gold"]
    lines = ["# M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER 决策报告", ""]
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；模型：{MODEL_NAME}（max_length={MAX_LEN}，CPU，inference-only）。")
    lines.append(f"- 冻结输入：M3-R_APPEND frozen pool（22 queries）；Planner/Rescue/OpenAlex/Citation/Reference/")
    lines.append("  Metadata/Prekeep 全未重跑；CE 排序阶段 0 次读 gold（gold 仅 evaluator）。")
    lines.append("")
    lines.append("## 1. 三方对照（Top-20）")
    lines.append("")
    lines.append("| variant | mean F1 | mean P | mean R | raw | pool | final | final_inst | LLM calls | HTTP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| M3-R LLM | {REF_LLM_F1} | — | — | 25 | 22 | {REF_LLM_FINAL} | — | {REF_LLM_CALLS} | — |")
    lines.append(f"| M3-R RRF | {REF_RRF_F1} | — | — | 25 | 22 | {REF_RRF_FINAL} | — | {REF_RRF_CALLS} | 0 |")
    lines.append(f"| **M4A2 CE** | **{f1}** | {vs_row['mean_precision']} | {vs_row['mean_recall']} "
                 f"| {vs_row['raw_unique_gold']} | {vs_row['pool_unique_gold']} | **{fin}** "
                 f"| {vs_row['final_query_gold_instances']} | 0 | 0 |")
    lines.append("")
    lines.append(f"- 运行成本：model load {t_load:.1f}s；CE 打分（2 passes）{t_inf:.1f}s；peak RSS={rss_bytes/1e6:.0f} MB。")
    lines.append("")
    lines.append("## 2. K 敏感度（diagnostic，不据此选 K）")
    lines.append("")
    lines.append("| K | F1 | P | R |")
    lines.append("|---|---|---|---|")
    for r in k_rows:
        lines.append(f"| {r['top_k']} | {r['mean_f1']} | {r['mean_precision']} | {r['mean_recall']} |")
    lines.append("")
    lines.append("## 3. Determinism")
    lines.append("")
    lines.append(f"- max |score1-score2| = {det['max_abs_score_diff']:.2e}；rank_order_identical={det['rank_order_identical']}；"
                 f"Top-20 Jaccard={det['mean_top20_jaccard']}；F1 identical={det['f1_identical']}。")
    lines.append(f"- → {'CROSS-ENCODER DETERMINISM PASSED' if det['ok'] else 'CROSS_ENCODER_DETERMINISM_FAILURE → STOP'}。")
    lines.append("")
    lines.append("## 4. Gold comparison（M4-0 LOST_RERANKER + M4A1 recovery）")
    lines.append("")
    n_lost = len(gold_rows)
    ce_rec = sum(1 for r in gold_rows if r["ce_top20_retained"] == "yes")
    llm_ret = sum(1 for r in gold_rows if r["llm_final_retained"] == "yes")
    rrf_rec = sum(1 for r in gold_rows if r["rrf_top20_retained"] == "yes")
    lines.append(f"- LOST_RERANKER gold={n_lost}；LLM 最终保留={llm_ret}；RRF top-20 保留={rrf_rec}；CE top-20 保留={ce_rec}。")
    for tag in ("Q47 FinEval x2", "Q15 RLHF Gold", "Q6 systematic FN"):
        sub = [r for r in gold_rows if tag in r["focus"]]
        if sub:
            lines.append(f"- **{tag}**：LLM={[r['llm_sel_freq'] for r in sub]}，"
                         f"RRF rank={[r['rrf_rank'] for r in sub]}，CE rank={[r['ce_rank'] for r in sub]}，"
                         f"CE score={[round(float(r['ce_score']),3) if r['ce_score']!='' else '' for r in sub]}，"
                         f"pre_rerank_rank={[r['pre_rerank_rank'] for r in sub]}。")
    lines.append("")
    lines.append("## 5. Abstract coverage（诊断，不据此调参）")
    lines.append("")
    rows_abs = abs_rows
    if rows_abs:
        covs = [r["pool_abstract_coverage"] for r in rows_abs if r["pool_abstract_coverage"] != ""]
        gcovs = [float(r["gold_abstract_coverage"]) for r in rows_abs if r["gold_abstract_coverage"] != ""]
        trunc = sum(r["truncated_candidates"] for r in rows_abs)
        n_with = [r["n_without_abstract"] for r in rows_abs]
        rw = [float(r["mean_ce_rank_with_abstract"]) for r in rows_abs]
        rwo = [float(r["mean_ce_rank_without_abstract"]) for r in rows_abs if r["n_without_abstract"] > 0]
        lines.append(f"- pool abstract coverage：mean={mean(covs):.2%}（跨 {len(rows_abs)} queries）。")
        if gcovs:
            lines.append(f"- gold abstract coverage（池内匹配 gold 组）：mean={mean(gcovs):.2%}。")
        lines.append(f"- truncated candidates 总数={trunc}；mean CE rank 有 abstract={mean(rw):.1f} vs 无 abstract={mean(rwo):.1f} "
                     f"（无 abstract 候选数合计={sum(n_with)}）。")
    lines.append("")
    lines.append("## 6. Decision Gate（参考：M3-R LLM F1=0.0664）")
    lines.append("")
    if f1 >= REF_LLM_F1:
        gate = "LIGHTWEIGHT_CE_COMPETITIVE"
        note = "CE ≥ LLM baseline → 优先用 Cross-Encoder 替代生成式 LLM Reranker。"
    elif f1 >= 0.055:
        gate = "LIGHTWEIGHT_CE_PROMISING"
        note = "下一阶段允许 M4A3_CE_RETRIEVAL_FUSION（只做一次确定性融合实验）。"
    else:
        gate = "MINILM_CE_INSUFFICIENT"
        note = "不调 MiniLM、不 fine-tune、不改阈值；下一步 STOP，由用户决定是否测试更强 reranker。"
    lines.append(f"- CE F1 = {f1}（final unique Gold={fin}），参考 LLM={REF_LLM_F1}。")
    lines.append(f"- **判定：{gate}**。{note}")
    lines.append("")
    lines.append("## 7. 当前禁止（未违反）")
    lines.append("")
    lines.append("- 无 fine-tune、无 Gold 训练、无 hard-negative mining、无多模型 ensemble；")
    lines.append("- 无 RRF/LLM/retrieval score fusion、无 safepass bonus、无 handcrafted score；")
    lines.append("- 未修改 Planner/Retriever/Safepass/Prekeep；未根据 Gold 选 K；未针对 Q6/Q15/Q47 写特殊规则。")
    lines.append("")
    if miss_q:
        lines.append(f"## 8. 已知数据缺失（如实报告，不猜测）")
        lines.append("")
        for qid, m in miss_q:
            lines.append(f"- {qid}: {len(m)} 个 executed sub 无 recall cache，CE 不受影响（CE 只对冻结池打分）。")
        lines.append("")
    lines.append("**本轮（M4A2）到此为止：STOP。不自动进入 M4A3。**")
    (OUT / DECISION).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "run":
        run(limit=args.limit)
    else:
        print("usage: python scripts/run_m4a2_ce.py run [--limit N]")
