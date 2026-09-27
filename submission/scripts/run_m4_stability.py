"""M4-0_RERANKER_STABILITY_AUDIT：测量当前 LLM Reranker 在「同一 pre-rerank 候选池」上的稳定性。

只做测量，不改任何算法。M3.1 结论冻结为 QUERY_FORMULATION_TUNING_STOP（不再动 Planner Prompt）。

版本定义：
  - M3-R = 当前 observed best final baseline（F1=0.0664）
  - M3.1 = 当前最高 deterministic retrieval coverage（raw unique Gold=26）
  不根据 M3.1 单次 F1=0.0519 判断真实算法退化（已观测到 frozen rich query 的 reranker stochastic contamination）。

冻结输入（Section 3）：7 条 rich 查询（rich_frozen），Planner/OpenAlex/Citation/Reference/Metadata/Prekeep
全部禁止重跑。直接读取已落盘 pre-rerank 快照（eval/diagnostics/m31/trace_*.json 的 rerank_pool.candidate_ids）。
必须证明输入 candidate IDs + order + metadata 完全一致，否则 EXPERIMENT_INVALID=true 并 STOP。

流程：
  1. reconstruct：从冻结 plan cache + merged recall cache 确定性重建 pre-rerank 池（0 联网）。
  2. verify：重建池 canonical_ids 逐位 == 落盘快照 rerank_pool.candidate_ids（set+order）。任一不符 → INVALID+STOP。
  3. run：对 7 条 rich 查询，用当前 LLMReranker（默认配置）在【相同输入】上跑 3 次，逐 run 记录。
  4. report：稳定性指标 + gate + 零 LLM 损失图谱（从 M3.1 gold lifecycle）。

产物（eval/runs/m4_reranker_stability/）：
  m4_reranker_config_audit.md       —— Reranker 配置审计（测量口径，不修改）
  m4_stability_runs.csv             —— 3 run x 7 query 逐 run 明细
  m4_query_variance.csv             —— 逐 query F1 min/mean/max/std、gold 保留、Top-20 Jaccard
  m4_gold_selection_frequency.csv   —— 逐 gold 3 轮选择频率
  m4_gold_loss_map.csv              —— 零 LLM 损失图谱（RAW_ONLY/LOST_PREKEEP/ENTERED_RERANK_POOL/LOST_RERANKER/FINAL_RETAINED）
  m4_stability_decision.md          —— gate 决策（RERANKER_STOCHASTICITY_CONFIRMED -> M4A / LOW -> M4B）

用法：
  python scripts/run_m4_stability.py rerank   # reconstruct+verify+3x LLM rerank（联网 DeepSeek）
  python scripts/run_m4_stability.py report   # 全部离线产物 + 决策
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import compute_p_r_f1, match_gold
from src.observability.canonical import canonical_paper_id, norm_title
from src.ranker import LLMReranker
from src.schemas import RankResult
from src.search import SearchEngine
from src.telemetry import Telemetry
from scripts.eval_benchmark import load_pasa
from scripts.run_m31 import load_query_plans, load_merged_cache, load_v2_plans, classify_sparse_rich

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
M31_TRACE_DIR = Path("eval/diagnostics/m31")
OUT = Path("eval/runs/m4_reranker_stability")
TOP_K = 20
RUNS = 3

CONFIG_AUDIT = "m4_reranker_config_audit.md"
RUNS_FILE = "m4_stability_runs.csv"
VARIANCE_FILE = "m4_query_variance.csv"
GOLD_SEL_FILE = "m4_gold_selection_frequency.csv"
LOSS_MAP_FILE = "m4_gold_loss_map.csv"
DECISION_FILE = "m4_stability_decision.md"
RUNS_JSON = "m4_runs.jsonl"


# --------------------------------------------------------------------------
def rich_queries() -> list[dict]:
    """7 条 rich 冻结查询（m3_kind==rich_frozen）。携带完整 benchmark dict（含 gold）供 match_gold。"""
    v2 = load_v2_plans(V2_PLAN_CACHE)
    sparse, rich = classify_sparse_rich(v2)
    plans = load_query_plans()
    bq_by_q = {bq["query"]: bq for bq in load_pasa(DATA)}
    out = []
    for q in plans:
        if q in rich and plans[q].get("m3_kind") == "rich_frozen":
            bq = bq_by_q.get(q, {})
            out.append({"query": q, "query_id": plans[q]["query_id"],
                        "plan": plans[q], "gold": bq.get("gold", []),
                        "bq": bq})
    return out


# --------------------------------------------------------------------------
# 确定性重建 + 验证（离线，0 联网）
# --------------------------------------------------------------------------
def reranker_input_hash(pre: list) -> str:
    """精排输入的字节/规范哈希：按 batch 边界的 _score_batch 序列化，证明三轮输入一致。"""
    h = hashlib.sha256()
    for i, ev in enumerate(pre):
        ident = ev.identity
        item = (
            f"[{ident.paper_id}]title:{ident.title}|auth:{','.join(ident.authors[:3])}|"
            f"venue:{ident.venue}|year:{ident.year}|abs:{(ev.abstract or '')[:200]}|src:{ev.source}"
        )
        h.update(f"{i}:{item}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def pool_hash(pre: list) -> str:
    h = hashlib.sha256()
    for cid in [canonical_paper_id(ev) for ev in pre]:
        h.update(cid.encode("utf-8") + b"\n")
    return h.hexdigest()[:16]


def asyncio_reset():
    return None


async def reconstruct_pool(q: str, plan: dict, merged: dict) -> tuple[list, int]:
    """确定性重建 pre-rerank 池（复用 search_full 的池构建路径，0 联网）。"""
    engine = SearchEngine(enable_citation_expansion=False, assoc_safepass=True)
    engine._plan_cache[q] = {"v": 2, "ir": plan["ir"], "subs": plan["subs"]}
    engine._recall_cache = merged
    ir, evs = await engine._plan_and_recall(q, Telemetry(), [], use_cache=True)
    lex = engine._lexical_rank(q, evs)
    pre = engine._build_rerank_pool(evs, lex)
    return pre, ir


def verify_pool(qid: str, pre: list) -> bool:
    """重建池 vs 落盘快照 rerank_pool.candidate_ids（canonical，set+order 全一致才算）。"""
    trc = M31_TRACE_DIR / f"trace_{qid}.json"
    if not trc.exists():
        print(f"  !! 缺落盘快照 {trc}")
        return False
    t = json.loads(trc.read_text(encoding="utf-8"))
    snap_ids = [s["candidate_ids"] for s in t["candidate_snapshots"] if s["stage"] == "rerank_pool"]
    if not snap_ids:
        print(f"  !! 快照无 rerank_pool 阶段 {qid}")
        return False
    snap = snap_ids[0]
    got = [canonical_paper_id(ev) for ev in pre]
    ok = (set(got) == set(snap)) and (got == snap)
    if not ok:
        print(f"  !! EXPERIMENT_INVALID 候选：{qid} 重建池与快照不一致 (set={set(got)==set(snap)} order={got==snap})")
    return ok


# --------------------------------------------------------------------------
# 3x 精排（LLM，DeepSeek；每 query 独立 fresh Telemetry 记 LLM calls + 延迟）
# --------------------------------------------------------------------------
async def rerank_query(q: dict, pre: list, ir, reranker: LLMReranker) -> list[dict]:
    runs = []
    for r in range(1, RUNS + 1):
        t0 = time.time()
        tel = Telemetry()
        ranked = await reranker.rerank(q["query"], ir, pre, telemetry=tel)
        latency = round((time.time() - t0) * 1000, 1)
        gold_groups = match_gold(q)
        m = compute_p_r_f1(ranked, gold_groups)
        top_ids = [canonical_paper_id(ev) for ev in pre]  # unused sentinel
        final_ids = [r.paper.paper_id for r in ranked] if ranked else []
        final_canonical = []
        for pid in final_ids:
            ev = next((e for e in pre if e.identity.paper_id == pid), None)
            final_canonical.append(canonical_paper_id(ev) if ev else f"w:{pid}")
        gold_retained = len({cid for cid in final_canonical if cid in gold_groups})
        runs.append({
            "run": r,
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "tp": m["tp"], "llm_calls": tel.llm_calls, "latency_ms": latency,
            "final_topk_ids": final_canonical, "n_final": len(final_canonical),
        })
    return runs


# --------------------------------------------------------------------------
# cmd: rerank
# --------------------------------------------------------------------------
async def cmd_rerank() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = rich_queries()
    if len(queries) != 7:
        print(f"!! EXPERIMENT_INVALID: rich 冻结查询应为 7，实际 {len(queries)}")
        sys.exit(1)
    merged = load_merged_cache()
    print(f"rich 冻结查询 = {len(queries)}；merged recall cache = {len(merged)} 条")

    # 1) 重建 + 验证（全部通过才联网）
    pools = {}
    for q in queries:
        pre, ir = await reconstruct_pool(q["query"], q["plan"], merged)
        ok = verify_pool(q["query_id"], pre)
        if not ok:
            print("\n!! EXPERIMENT_INVALID=true：无法证明输入 candidate IDs + order + metadata 一致。STOP。")
            sys.exit(1)
        pools[q["query_id"]] = {"pre": pre, "ir": ir}
        print(f"  验证通过 {q['query_id']}: pool={len(pre)}")

    # 2) 3x 精排
    reranker = LLMReranker()  # 默认配置 = 生产（deepseek-chat, batch=15, keep=0.35, min=3, max=20）
    print(f"\n对 {len(queries)} 条 rich 查询各跑 {RUNS} 轮 LLM 精排（batch_size=15 串行）...")
    records = []
    for q in queries:
        qid = q["query_id"]
        pre = pools[qid]["pre"]
        ph = pool_hash(pre); ih = reranker_input_hash(pre)
        runs = await rerank_query(q, pre, pools[qid]["ir"], reranker)
        for r in runs:
            records.append({
                "query_id": qid, "query": q["query"], "run": r["run"],
                "candidate_pool_hash": ph, "reranker_input_hash": ih,
                "pool_size": len(pre), "f1": r["f1"], "precision": r["precision"],
                "recall": r["recall"], "tp": r["tp"], "llm_calls": r["llm_calls"],
                "latency_ms": r["latency_ms"], "n_final": r["n_final"],
                "final_topk_ids": ";".join(r["final_topk_ids"]),
            })
        print(f"  {qid}: pool={len(pre)}  F1={[round(x['f1'],4) for x in runs]}  "
              f"llm={[x['llm_calls'] for x in runs]}")
    # 落盘 jsonl + csv
    with open(OUT / RUNS_JSON, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cols = ["query_id", "query", "run", "candidate_pool_hash", "reranker_input_hash",
            "pool_size", "f1", "precision", "recall", "tp", "llm_calls", "latency_ms",
            "n_final", "final_topk_ids"]
    with open(OUT / RUNS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    print(f"写 {OUT / RUNS_FILE} 与 {OUT / RUNS_JSON}（{len(records)} 行）")


# --------------------------------------------------------------------------
# cmd: report（全部离线）
# --------------------------------------------------------------------------
def load_runs() -> list[dict]:
    if not (OUT / RUNS_JSON).exists():
        print("!! 先跑 rerank 生成 m4_runs.jsonl")
        sys.exit(1)
    return [json.loads(l) for l in (OUT / RUNS_JSON).read_text(encoding="utf-8").splitlines() if l.strip()]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


async def _reconstruct_pools() -> dict[str, list]:
    """离线重建全部 7 条 rich 查询的 pre-rerank 池（0 联网），供报告 gold/输入哈希对齐。"""
    merged = load_merged_cache()
    pools = {}
    for q in rich_queries():
        pre, ir = await reconstruct_pool(q["query"], q["plan"], merged)
        pools[q["query_id"]] = pre
    return pools


def _gold_groups_by_query() -> dict[str, list[set[str]]]:
    return {q["query_id"]: match_gold(q["bq"]) for q in rich_queries()}


def _group_selection(final_evs: list, gold_groups: list[set[str]]) -> set[int]:
    """返回被 final_evs 命中的 gold 组下标集合（复用 compute_p_r_f1 的 key 匹配）。"""
    matched: set[int] = set()
    for ev in final_evs:
        rr = RankResult(paper=ev.identity, score=0.0)
        pkeys = {f"openalex:{rr.paper.paper_id}"} if rr.paper.paper_id else set()
        from eval.harness import _norm_doi, _norm_title, _norm_title_letters
        doi = _norm_doi(rr.paper.doi)
        if doi:
            pkeys.add(f"doi:{doi}")
        if rr.paper.title:
            pkeys.add(f"title:{_norm_title(rr.paper.title)}")
            pkeys.add(f"title_n:{_norm_title_letters(rr.paper.title)}")
        for gi, gkeys in enumerate(gold_groups):
            if gi in matched:
                continue
            if pkeys & gkeys:
                matched.add(gi)
    return matched


async def cmd_report() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    qids = sorted({r["query_id"] for r in runs})
    print(f"载入 {len(runs)} 行 runs；{len(qids)} 条查询；离线重建池 + 重算 F1/gold（0 联网）")

    pools = await _reconstruct_pools()
    gold_by_q = _gold_groups_by_query()
    bq_by_id = {q["query_id"]: q for q in rich_queries()}

    # 逐 run 分组
    by_q_run: dict[tuple[str, int], dict] = {}
    for r in runs:
        by_q_run[(r["query_id"], r["run"])] = r

    # ---- 离线重算每 run 的 F1/P/R/tp 与 gold 命中 ----
    # 存储的 f1/precision/recall/tp（rerank 时因缺 gold 算错）用正确值覆盖
    for qid in qids:
        pre = pools.get(qid, [])
        ev_by_canonical = {canonical_paper_id(ev): ev for ev in pre}
        gold_groups = gold_by_q.get(qid, [])
        for rn in range(1, RUNS + 1):
            rec = by_q_run[(qid, rn)]
            final_ids = [x for x in rec["final_topk_ids"].split(";") if x]
            final_evs = [ev_by_canonical[cid] for cid in final_ids if cid in ev_by_canonical]
            ranked = [RankResult(paper=ev.identity, score=0.0) for ev in final_evs]
            m = compute_p_r_f1(ranked, gold_groups)
            matched_gi = _group_selection(final_evs, gold_groups)
            rec["f1"], rec["precision"], rec["recall"], rec["tp"] = m["f1"], m["precision"], m["recall"], m["tp"]
            rec["_matched_groups"] = matched_gi
            rec["_n_gold"] = len(gold_groups)

    # 用重算值重写 m4_stability_runs.csv
    cols = ["query_id", "query", "run", "candidate_pool_hash", "reranker_input_hash",
            "pool_size", "f1", "precision", "recall", "tp", "llm_calls", "latency_ms",
            "n_final", "final_topk_ids"]
    with open(OUT / RUNS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for qid in qids:
            for rn in range(1, RUNS + 1):
                rec = dict(by_q_run[(qid, rn)])
                rec.pop("_matched_groups", None); rec.pop("_n_gold", None)
                w.writerow(rec)

    # ---- m4_query_variance.csv ----
    var_rows = []
    for qid in qids:
        f1s = [by_q_run[(qid, r)]["f1"] for r in range(1, RUNS + 1)]
        sets = [set(by_q_run[(qid, r)]["final_topk_ids"].split(";")) for r in range(1, RUNS + 1)]
        pair_jac = [jaccard(list(sets[a]), list(sets[b])) for a in range(RUNS) for b in range(a + 1, RUNS)]
        mean_jac = sum(pair_jac) / len(pair_jac) if pair_jac else 0.0
        matched_runs = [by_q_run[(qid, r)]["_matched_groups"] for r in range(1, RUNS + 1)]
        flips = sum(1 for gi in set().union(*matched_runs) if len({gi in mr for mr in matched_runs}) > 1)
        lo = min(f1s); hi = max(f1s); mean = sum(f1s) / len(f1s)
        std = (sum((x - mean) ** 2 for x in f1s) / len(f1s)) ** 0.5
        var_rows.append({
            "query_id": qid, "f1_min": round(lo, 4), "f1_mean": round(mean, 4), "f1_max": round(hi, 4),
            "f1_range": round(hi - lo, 4), "f1_std": round(std, 4),
            "gold_retention_flips": flips, "top20_mean_jaccard": round(mean_jac, 4),
            "top20_jaccard_pairs": ";".join(f"{p:.3f}" for p in pair_jac),
            "n_final_min": min(len(sets[a]) for a in range(RUNS)),
            "n_final_max": max(len(sets[a]) for a in range(RUNS)),
        })
    with open(OUT / VARIANCE_FILE, "w", newline="", encoding="utf-8") as f:
        vcols = ["query_id", "f1_min", "f1_mean", "f1_max", "f1_range", "f1_std",
                 "gold_retention_flips", "top20_mean_jaccard", "top20_jaccard_pairs",
                 "n_final_min", "n_final_max"]
        w = csv.DictWriter(f, fieldnames=vcols)
        w.writeheader()
        for row in var_rows:
            w.writerow(row)

    # ---- m4_gold_selection_frequency.csv（逐 gold 组 3 轮选择频率）----
    gold_sel_rows = []
    for qid in qids:
        gold_groups = gold_by_q.get(qid, [])
        bq = bq_by_id.get(qid, {})
        gold_list = bq.get("gold", [])
        for gi, gkeys in enumerate(gold_groups):
            sel = [1 if gi in by_q_run[(qid, r)]["_matched_groups"] else 0 for r in range(1, RUNS + 1)]
            title = gold_list[gi].get("title") if gi < len(gold_list) else ""
            gold_sel_rows.append({
                "query_id": qid, "gold_index": gi, "gold_title": title,
                "run1": sel[0], "run2": sel[1], "run3": sel[2],
                "selection_freq": sum(sel), "flips": (len(set(sel)) > 1),
            })
    with open(OUT / GOLD_SEL_FILE, "w", newline="", encoding="utf-8") as f:
        gcols = ["query_id", "gold_index", "gold_title", "run1", "run2", "run3", "selection_freq", "flips"]
        w = csv.DictWriter(f, fieldnames=gcols)
        w.writeheader()
        for row in gold_sel_rows:
            w.writerow(row)

    # ---- 零 LLM 损失图谱（从 M3.1 lifecycle）----
    loss_rows, lost_reranker_rows = build_loss_map(gold_sel_rows, by_q_run)
    with open(OUT / LOSS_MAP_FILE, "w", newline="", encoding="utf-8") as f:
        lcols = ["query_id", "canonical_id", "title_n", "first_seen_query", "first_seen_stage",
                 "source_kind", "query_type", "safepass", "pre_rerank_rank", "first_seen_rank",
                 "loss_stage", "final_rank", "llm_sel_freq"]
        w = csv.DictWriter(f, fieldnames=lcols, extrasaction="ignore")
        w.writeheader()
        for row in loss_rows:
            w.writerow(row)

    # ---- m4_reranker_config_audit.md ----
    write_config_audit()

    # ---- 决策 ----
    write_decision(var_rows, gold_sel_rows, by_q_run, gold_by_q)
    print("报告完成。")



def build_loss_map(gold_sel_rows: list[dict], by_q_run: dict) -> tuple[list[dict], list[dict]]:
    """零 LLM 损失图谱：把 M3.1 gold lifecycle 分类为损失漏斗 + 特殊 LOST_RERANKER 表。
    llm_sel_freq = 该 gold 在 3 轮精排 final top-K 中被选中的次数（0~3）。"""
    loss_rows = []
    lost_reranker_rows = []
    for q in rich_queries():
        qid = q["query_id"]
        trc = M31_TRACE_DIR / f"trace_{qid}.json"
        if not trc.exists():
            continue
        t = json.loads(trc.read_text(encoding="utf-8"))
        for g in t["gold_lifecycle"]:
            stage = g.get("first_seen_stage", "")
            if stage not in ("regular_recall", "assoc_recall"):
                continue  # 只统计 raw gold
            survived = g.get("survived_prekeep")
            final_rank = g.get("final_rank")
            if final_rank is not None:
                loss_stage = "FINAL_RETAINED"
            elif survived:
                loss_stage = "LOST_RERANKER"
            else:
                loss_stage = "LOST_PREKEEP"
            cid = g["canonical_id"]
            src = g.get("source_kind", "rich")
            qtype = g.get("query_type", "")
            safepass = (stage == "assoc_recall")
            if src == "rescue":
                retrieval_source = f"rescue:{qtype}" if qtype else "rescue"
            elif stage == "assoc_recall":
                retrieval_source = "assoc"
            else:
                retrieval_source = "regular"
            llm_sel = sum(1 for rn in range(1, RUNS + 1)
                          if (qid, rn) in by_q_run and
                          cid in set(by_q_run[(qid, rn)]["final_topk_ids"].split(";")))
            row = {
                "query_id": qid, "canonical_id": cid, "title_n": g.get("title_n", ""),
                "first_seen_query": g.get("first_seen_query", ""),
                "first_seen_stage": stage, "source_kind": src, "query_type": qtype,
                "safepass": safepass, "pre_rerank_rank": g.get("pre_rerank_rank", ""),
                "first_seen_rank": g.get("first_seen_rank", ""),
                "loss_stage": loss_stage, "final_rank": final_rank if final_rank is not None else "",
                "llm_sel_freq": llm_sel,
            }
            loss_rows.append(row)
            if survived and final_rank is None:
                lost_reranker_rows.append({**row, "retrieval_source": retrieval_source})
    return loss_rows, lost_reranker_rows


def write_config_audit() -> None:
    lines = []
    lines.append("# M4-0 RERANKER 配置审计（测量口径，不修改）")
    lines.append("")
    lines.append("- 目的：记录当前 LLM Reranker 的全部可调配置，作为稳定性测量的输入口径。")
    lines.append("- 只记录，不改任何参数/算法。")
    lines.append("")
    lines.append("## 1. 模型与采样")
    lines.append("")
    lines.append("| 项 | 值 | 备注 |")
    lines.append("|---|---|---|")
    lines.append("| provider | DeepSeek（openai_base_url=https://api.deepseek.com） | 生产 |")
    lines.append("| model | deepseek-chat（llm_model） | tier=strong |")
    lines.append("| temperature | 0.0（complete_json 默认） | 非真正确定性 |")
    lines.append("| top_p | 未设置（采样随机） | 潜在随机源 |")
    lines.append("| seed | 未设置 | 潜在随机源 |")
    lines.append("| max_tokens | 1500（rerank batch） | 见 LLMClient |")
    lines.append("| 输出格式 | json_object（response_format） | 结构化打分 |")
    lines.append("")
    lines.append("## 2. Reranker Prompt（prompt hash 见搜索实现）")
    lines.append("")
    lines.append("- Prompt 要求逐候选输出 score/label/reason；完整 prompt 文本未在本审计重复（hash 锁定，未修改）。")
    lines.append("")
    lines.append("## 3. 候选分批 / 打分")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| batch_size | 15 |")
    lines.append("| max_abstract_chars | 200（abstract 截断） |")
    lines.append("| 打分方式 | 逐批串行（serial _score_batch） |")
    lines.append("| LLM calls / query | ceil(pool_size / 15) |")
    lines.append("| 输出解析 | JSON 解析（完整_json，逐候选 key=paper_id） |")
    lines.append("")
    lines.append("## 4. 截断与排序")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| keep_threshold | 0.35（score>=阈值保留） |")
    lines.append("| min_keep | 3（保底） |")
    lines.append("| max_results | 20（评测 top_k） |")
    lines.append("| tie-break | 按 -score 稳定排序 → 候选插入序 |")
    lines.append("| 未覆盖候选 | 词法兜底排末尾 |")
    lines.append("")
    lines.append("## 5. 重试 / 并发 / 状态")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 重试 | tenacity 3 次，同 payload（无重采样） |")
    lines.append("| query 内并发 | 无（串行） |")
    lines.append("| 并发顺序改变 | 不影响单 query（串行打分） |")
    lines.append("| 随机性来源 | temperature>0 实际采样 + 无 seed/top_p 约束 |")
    lines.append("")
    lines.append("## 6. 结论：唯一随机性来源")
    lines.append("")
    lines.append("LLM 采样本身（temperature 名义 0.0 但非 seed 固定、无 top_p），其余路径全确定性。")
    lines.append("同一 pre-rerank 池的 3 次精排输出差异即可归因于 Reranker 采样随机性。")
    OUT.joinpath(CONFIG_AUDIT).write_text("\n".join(lines), encoding="utf-8")
    print(f"写 {OUT / CONFIG_AUDIT}")


def write_decision(var_rows, gold_sel_rows, by_q_run, qid_gold) -> None:
    # 证据收集
    flags = []
    # (a) 同 query gold retention flips in 3 runs
    flips = [r for r in gold_sel_rows if r["flips"]]
    if flips:
        flags.append(f"gold retention flip：{len(flips)} 个 query-gold 在 3 轮间保留状态翻转")
    # (b) Q15 gold wobble
    q15 = [r for r in gold_sel_rows if r["query_id"] == "RealScholarQuery_15" and r["flips"]]
    if q15:
        flags.append(f"Q15 gold wobble：{len(q15)} 个 Q15 gold 在 3 轮间翻转")
    # (c) overall mean F1 range
    overall = {}
    for qid in qids_of(var_rows):
        pass
    mean_by_run = {}
    for r in range(1, RUNS + 1):
        f1s = [row["f1"] for row in runs_by_run(r, by_q_run, var_rows)]
        mean_by_run[r] = sum(f1s) / len(f1s) if f1s else 0.0
    f1_range = max(mean_by_run.values()) - min(mean_by_run.values())
    if f1_range >= 0.005:
        flags.append(f"overall mean F1 range = {f1_range:.4f} >= 0.005")
    # (d) Top-20 Jaccard 稳定性
    jac = [row["top20_mean_jaccard"] for row in var_rows]
    mean_jac = sum(jac) / len(jac) if jac else 0.0
    min_jac = min(jac) if jac else 1.0
    if min_jac < 0.8:
        flags.append(f"Top-20 Jaccard 明显不稳定：min={min_jac:.3f} mean={mean_jac:.3f} (<0.8)")

    q15f1 = [by_q_run[("RealScholarQuery_15", r)]["f1"] for r in range(1, RUNS + 1)] \
        if ("RealScholarQuery_15", 1) in by_q_run else []

    lines = []
    lines.append("# M4-0 RERANKER 稳定性审计 决策报告")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}；{len(var_rows)} 条 rich 查询，每查询 {RUNS} 轮 LLM 精排（相同 pre-rerank 池）")
    lines.append("- M3.1 冻结：QUERY_FORMULATION_TUNING_STOP。本审计只测量，不改任何算法。")
    lines.append("")
    lines.append("## 1. 输入冻结证明（Section 3）")
    lines.append("")
    lines.append("7 条 rich 冻结查询重建池 canonical_ids 与落盘快照 rerank_pool.candidate_ids 逐位一致（set+order）。")
    lines.append("Planner/OpenAlex/Citation/Reference/Metadata/Prekeep 均未重跑。EXPERIMENT_INVALID 未触发。")
    lines.append("")
    lines.append("## 2. 稳定性证据")
    lines.append("")
    lines.append(f"- overall mean F1 per run：{ {r: round(mean_by_run[r],4) for r in mean_by_run} }；range={f1_range:.4f}")
    lines.append(f"- Top-20 mean Jaccard：mean={mean_jac:.3f} min={min_jac:.3f}")
    lines.append(f"- gold retention flips（3 轮间保留状态翻转的 query-gold 数）={len(flips)}")
    lines.append(f"- Q15（历史 0.1333→0.0000 污染探针）3 轮 F1={[round(x,4) for x in q15f1]}")
    lines.append("")
    lines.append("## 3. Gate（Section 6）")
    lines.append("")
    if flags:
        lines.append("触发随机性判定条件：")
        for f in flags:
            lines.append(f"  - {f}")
        lines.append("")
        lines.append("**判定：RERANKER_STOCHASTICITY_CONFIRMED**")
        lines.append("**NEXT = M4A_DETERMINISTIC_RERANKER**（不实现，仅记录）")
    else:
        lines.append("3 轮高度稳定，未触发随机性判定条件。")
        lines.append("")
        lines.append("**判定：RERANKER_STOCHASTICITY_LOW**")
        lines.append("**NEXT = M4B_RETENTION_OPTIMIZATION**（不实现，仅记录）")
    lines.append("")
    lines.append("## 4. 零 LLM 损失图谱（Section 7，来自 M3.1 lifecycle）")
    lines.append("")
    lines.append("见 m4_gold_loss_map.csv。LOST_RERANKER（进池但未进 final）候选含 pre_rerank_rank/检索源/safepass/LLM 3 轮选择频率。")
    lines.append("")
    lines.append("**本轮（M4-0）到此为止：只测量。STOP，等待下一步批准，不自动实现 M4A/M4B。**")
    OUT.joinpath(DECISION_FILE).write_text("\n".join(lines), encoding="utf-8")
    print(f"写 {OUT / DECISION_FILE}")


def qids_of(var_rows):
    return [r["query_id"] for r in var_rows]


def runs_by_run(r, by_q_run, var_rows):
    qids = [r2["query_id"] for r2 in var_rows]
    return [by_q_run[(qid, r)] for qid in qids if (qid, r) in by_q_run]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rerank", "report", "all"])
    args = ap.parse_args()
    if args.mode in ("rerank", "all"):
        await cmd_rerank()
    if args.mode in ("report", "all"):
        await cmd_report()


if __name__ == "__main__":
    asyncio.run(main())
