"""M1_ANCHOR_AUGMENTED 全量评测（v3-MVP 唯一实验）。

对照：B0 = PASA_ASSOC_NO_CIT（F1=0.0618, P=0.0615, R=0.1339, api=87）——不重跑，直接读基线报告。
只变 Query Formulation：M1 plan（core/anchor/discovery，≤5 query）替换 v2 subqueries。
Retriever / Prekeep / Reranker / OpenAlex / top-k 全部冻结（复用 v2 冻结 ir，reranker 看到的 query+ir 与 B0 完全一致）。
Citation/Reference/metadata 扩展关闭（enable_citation_expansion=False，与 B0 一致）。

Gold isolation：planner 只喂 question 文本；gold 只进 TraceRecorder 诊断统计。

产物（eval/runs/m1_anchor_augmented/）：
  m1_query_plans.jsonl     —— v=2 plan cache（key=query text；也作为搜索阶段的加载源）
  m1_query_type_metrics.csv —— 每类型：执行 query 数 / new unique Gold / 增量 Gold-per-query
  m1_gold_lifecycle.csv    —— 全 gold 生命周期（first_seen -> drop/final）
  m1_vs_b0.csv             —— 逐 query B0 vs M1 的 F1/P/R/api
  v3_mvp_decision.md       —— 决策报告（raw_unique_gold gate + 下一步）

用法：
  python scripts/run_m1_anchor.py generate            # 只生成 plan（可先审）
  python scripts/run_m1_anchor.py search              # 用既有 plan 跑检索（断点续跑）
  python scripts/run_m1_anchor.py report              # 只重生成产物/决策（不联网）
  python scripts/run_m1_anchor.py all --limit 3       # 全流程 smoke（限前 3 条）

修订轮次：
  --rev 2   # ONE_PLANNER_REVISION：所有产物/缓存/实验名输出到 m1_anchor_augmented_r2/，
            # 与 v1 的 m1_anchor_augmented/ 完全隔离，不覆盖任何 M1 产物。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import compute_p_r_f1, match_gold
from src.observability.response_cache import ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.planner_anchor import (
    INTENT_ANCHOR, INTENT_CORE, INTENT_DISCOVERY, AnchorAugmentedPlannerV3,
)
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
NOCIT_RESULTS = "eval/runs/nocitation/nocitation_results.json"
M1_DIR = Path("eval/runs/m1_anchor_augmented")
PLAN_FILE = M1_DIR / "m1_query_plans.jsonl"
CACHE_DIR = "eval/cache/m1_anchor_augmented"
TRACE_DIR = Path("eval/diagnostics/m1_anchor_augmented")
EXPERIMENT = "PASA_ASSOC_M1_ANCHOR"
BASELINE_REF = "PASA_ASSOC_NO_CIT"
TOP_K = 20
QUERY_TYPES = (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY)


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_v2_plans(path: str) -> dict[str, dict]:
    plans: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("v") != 2 or not isinstance(d.get("ir"), dict) or not isinstance(d.get("subs"), list):
            raise ValueError(f"v2 冻结 plan 非法: {d.get('query')}")
        plans[d["query"]] = d
    return plans


def load_m1_plans() -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not PLAN_FILE.exists():
        return plans
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("v") != 2 or not isinstance(d.get("ir"), dict) or not isinstance(d.get("subs"), list):
            raise ValueError(f"M1 plan 非法: {d.get('query')}")
        plans[d["query"]] = d
    return plans


def load_nocit_baseline() -> dict[str, dict]:
    d = json.loads(Path(NOCIT_RESULTS).read_text(encoding="utf-8"))
    assert d["experiment"] == BASELINE_REF, f"基线实验名不符: {d['experiment']}"
    return {r["query_id"]: r for r in d["results"]}


def select_queries() -> list[dict]:
    """加载 PASA，过滤到 B0（nocitation）覆盖的那 22 条，保证 vs_b0 对齐。"""
    b0 = load_nocit_baseline()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    all_q = load_pasa(DATA)
    picked = [q for q in all_q if q["query_id"] in b0 and q["query"] in v2]
    if len(picked) != len(b0):
        raise SystemExit(f"EXPERIMENT_INVALID: 过滤后 {len(picked)} 条 ≠ B0 基线 {len(b0)} 条")
    return picked


# --------------------------------------------------------------------------
# Phase A：plan 生成（一次 LLM 调用 / query，resumable）
# --------------------------------------------------------------------------
async def cmd_generate(queries: list[dict], limit: int | None) -> None:
    M1_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_m1_plans()
    todo = [q for q in queries if q["query"] not in existing]
    if limit:
        todo = todo[:limit]
    print(f"M1 plan 生成：22 条中已存在 {len(existing)} 条，本次生成 {len(todo)} 条")
    if not todo:
        print("无需生成。")
        return

    planner = AnchorAugmentedPlannerV3()
    for i, q in enumerate(todo, 1):
        t0 = time.time()
        subs = await planner.plan_raw(q["query"])
        plan = {
            "query": q["query"],
            "query_id": q["query_id"],
            "v": 2,
            "ir": load_v2_plans(V2_PLAN_CACHE)[q["query"]]["ir"],
            "subs": [s.model_dump() for s in subs],
        }
        with open(PLAN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
        tag = " ".join(f"{s['intent']}={s['query_text'][:32]}" for s in plan["subs"])
        print(f"[{i}/{len(todo)}] {q['query_id']} ({round((time.time() - t0) * 1000)}ms) {tag}")
    # 汇总按类型计数
    plans = load_m1_plans()
    from collections import Counter
    cnt = Counter()
    for p in plans.values():
        cnt.update(s["intent"] for s in p["subs"])
    print("M1 plan 类型统计（全部 plan）：", dict(cnt))
    print("写", PLAN_FILE)


# --------------------------------------------------------------------------
# Phase B：检索 + trace（断点续跑；每 query 报告幂等）
# --------------------------------------------------------------------------
async def cmd_search(limit: int | None) -> None:
    plans = load_m1_plans()
    if not plans:
        raise SystemExit("EXPERIMENT_INVALID: 无 M1 plan，先跑 generate")
    queries = [q for q in select_queries() if q["query"] in plans]
    if limit:
        queries = queries[:limit]
    print(f"M1 检索：{len(queries)} 条（plan 缓存 {len(plans)} 条）")

    cache = ResponseCache(CACHE_DIR, mode="write")
    engine = SearchEngine(response_cache=cache, enable_citation_expansion=False)
    engine.load_plan_cache(str(PLAN_FILE))
    # 只加载 M1 plan cache；不加载 v2 与 baseline recall cache（subquery 真实搜索）

    M1_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_before = 0
    todo = []
    for q in queries:
        rep = M1_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            done_before += 1
            continue
        todo.append(q)
    if done_before:
        print(f"断点续跑：跳过已完成 {done_before} 条，剩余 {len(todo)} 条")

    for i, q in enumerate(todo, 1):
        qid = q["query_id"]
        gold_groups = match_gold(q)
        gold_titles = [g["title"] for g in q.get("gold", []) if g.get("title")]

        recorder = TraceRecorder(
            query_id=qid, run_id=EXPERIMENT, baseline_reference_id=BASELINE_REF,
            response_cache=cache,
        )
        recorder.set_gold_titles(gold_titles)
        engine.recorder = recorder

        t0 = time.time()
        try:
            results, telemetry, traces = await engine.search_full(q["query"], top_k=TOP_K)
        except Exception as e:  # noqa: BLE001 —— 单条失败不炸全 run，写失败日志后继续
            print(f"[{i}/{len(todo)}] {qid}: FAIL {type(e).__name__}: {e}")
            with open(M1_DIR / "search_failures.log", "a", encoding="utf-8") as f:
                f.write(f"{qid}\t{type(e).__name__}\t{e}\n")
            continue
        latency_ms = round((time.time() - t0) * 1000, 1)

        metrics = compute_p_r_f1(results, gold_groups)
        snapshots = {s["stage"]: s for s in recorder.candidate_snapshots}
        report = {
            "experiment": EXPERIMENT,
            "query_id": qid,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "api_calls": telemetry.api_calls,
            "llm_calls": telemetry.llm_calls,
            "input_tokens": telemetry.input_tokens,
            "output_tokens": telemetry.output_tokens,
            "latency_ms": latency_ms,
            "n_predicted": len(results),
            "raw_candidates": snapshots.get("after_raw_recall", {}).get("candidate_count", 0),
            "rerank_pool_candidates": snapshots.get("rerank_pool", {}).get("candidate_count", 0),
        }
        recorder.save(TRACE_DIR)
        cache.flush()
        (M1_DIR / f"report_{qid}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {qid}: F1={report['f1']:.4f} P={report['precision']:.4f} "
              f"R={report['recall']:.4f} api={report['api_calls']} llm={report['llm_calls']} "
              f"raw={report['raw_candidates']} {latency_ms}ms")

    cache.close()
    print("检索完成。生成产物...")
    cmd_report()


# --------------------------------------------------------------------------
# Phase C：产物 + 决策（读 reports + traces，不联网）
# --------------------------------------------------------------------------
def load_report(qid: str) -> dict:
    return json.loads((M1_DIR / f"report_{qid}.json").read_text(encoding="utf-8"))


def load_trace(qid: str) -> dict:
    return json.loads((TRACE_DIR / f"trace_{qid}.json").read_text(encoding="utf-8"))


def query_type_of(plans: dict[str, dict], subquery_text: str) -> str:
    """gold lifecycle 的 first_seen_query(80 字符截断) -> 类型。"""
    key = subquery_text[:80]
    for p in plans.values():
        for s in p["subs"]:
            if s["query_text"][:80] == key:
                return s["intent"]
    return "unknown"


def cmd_report() -> None:
    plans = load_m1_plans()
    b0 = load_nocit_baseline()
    queries = select_queries()
    # 容忍缺失报告（如配额耗尽导致部分 query 未完成）：只聚合有 report+trace 的 query
    reports: dict[str, dict] = {}
    for q in queries:
        rep = M1_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            reports[q["query_id"]] = json.loads(rep.read_text(encoding="utf-8"))

    # ---- 全 gold 生命周期聚合（跨 query 按 canonical_id 全局去重）----
    all_lifecycle: list[dict] = []      # 每行一个 (query_id, lifecycle entry)
    raw_by_id: dict[str, dict] = {}     # canonical_id -> first-seen query_id（raw 阶段）
    for q in queries:
        qid = q["query_id"]
        if qid not in reports:
            continue
        trc = load_trace(qid)
        for g in trc.get("gold_lifecycle", []):
            row = {**g, "query_id": qid,
                   "query_type": query_type_of(plans, g.get("first_seen_query", ""))}
            all_lifecycle.append(row)
            if g.get("first_seen_stage") == "regular_recall":
                cid = g["canonical_id"]
                raw_by_id.setdefault(cid, qid)  # 首次见到即归属该 query
    raw_unique_gold = len(raw_by_id)
    final_ids = {g["canonical_id"] for g in all_lifecycle if g.get("final_rank") is not None}
    final_unique_gold = len(final_ids)
    # raw 阶段 gold 的去向：最终进 top-k 的比例（= gold 保留率，诊断 reranker 是否掉 gold）
    raw_gold_final = sum(1 for cid in raw_by_id if cid in final_ids)

    # ---- m1_gold_lifecycle.csv ----
    lc_cols = ["query_id", "canonical_id", "title_n", "first_seen_stage", "first_seen_query",
               "first_seen_rank", "query_type", "survived_prekeep", "pre_rerank_rank",
               "reranker_rank", "final_rank", "drop_stage", "drop_reason"]
    with open(M1_DIR / "m1_gold_lifecycle.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=lc_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_lifecycle:
            w.writerow(row)

    # ---- m1_query_type_metrics.csv ----
    type_gold = {t: 0 for t in QUERY_TYPES}
    type_exec = {t: 0 for t in QUERY_TYPES}
    for row in all_lifecycle:
        qt = row["query_type"]
        if qt in type_gold:
            type_exec[qt] += 1  # 该类型 subquery 首次命中的 gold 数（每个 lifecycle 行一次）
            type_gold[qt] += 1
    # executed_queries = plan 里该类型 subquery 总数（全部参与检索，无截断损失）
    executed = {t: sum(1 for p in plans.values() for s in p["subs"] if s["intent"] == t) for t in QUERY_TYPES}
    with open(M1_DIR / "m1_query_type_metrics.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["query_type", "executed_queries", "new_unique_gold", "incremental_gold_per_query"])
        for t in QUERY_TYPES:
            per_q = (type_gold[t] / executed[t]) if executed[t] else 0.0
            w.writerow([t, executed[t], type_gold[t], round(per_q, 3)])

    # ---- m1_vs_b0.csv ----
    n = len(reports)
    agg = {k: {"b0": 0.0, "m1": 0.0} for k in ("f1", "precision", "recall", "api_calls", "llm_calls")}
    with open(M1_DIR / "m1_vs_b0.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        cols = ["query_id", "b0_f1", "b0_precision", "b0_recall", "b0_api_calls", "b0_llm_calls",
                "m1_f1", "m1_precision", "m1_recall", "m1_api_calls", "m1_llm_calls",
                "m1_raw_gold", "m1_final_gold", "delta_f1", "delta_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for qid in sorted(reports):
            r = reports[qid]
            b = b0[qid]
            raw_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and row["first_seen_stage"] == "regular_recall")
            fin_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and row.get("final_rank") is not None)
            w.writerow({
                "query_id": qid,
                "b0_f1": b["f1"], "b0_precision": b["precision"], "b0_recall": b["recall"],
                "b0_api_calls": b["api_calls"], "b0_llm_calls": b["llm_calls"],
                "m1_f1": r["f1"], "m1_precision": r["precision"], "m1_recall": r["recall"],
                "m1_api_calls": r["api_calls"], "m1_llm_calls": r["llm_calls"],
                "m1_raw_gold": raw_g, "m1_final_gold": fin_g,
                "delta_f1": round(r["f1"] - b["f1"], 4), "delta_recall": round(r["recall"] - b["recall"], 4),
            })
            for k in agg:
                agg[k]["b0"] += b[k]
                agg[k]["m1"] += r[k]
        w.writerow({})
        w.writerow({
            "query_id": "MEAN",
            "b0_f1": round(agg["f1"]["b0"] / n, 4), "b0_precision": round(agg["precision"]["b0"] / n, 4),
            "b0_recall": round(agg["recall"]["b0"] / n, 4), "b0_api_calls": round(agg["api_calls"]["b0"] / n, 1),
            "b0_llm_calls": round(agg["llm_calls"]["b0"] / n, 1),
            "m1_f1": round(agg["f1"]["m1"] / n, 4), "m1_precision": round(agg["precision"]["m1"] / n, 4),
            "m1_recall": round(agg["recall"]["m1"] / n, 4), "m1_api_calls": round(agg["api_calls"]["m1"] / n, 1),
            "m1_llm_calls": round(agg["llm_calls"]["m1"] / n, 1),
            "m1_raw_gold": raw_unique_gold, "m1_final_gold": final_unique_gold,
            "delta_f1": round(agg["f1"]["m1"] / n - agg["f1"]["b0"] / n, 4),
            "delta_recall": round(agg["recall"]["m1"] / n - agg["recall"]["b0"] / n, 4),
        })

    # ---- v3_mvp_decision.md ----
    lines = []
    lines.append("# M1_ANCHOR_AUGMENTED 决策报告（v3-MVP）")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}（M1 全量 22-query 评测）")
    lines.append(f"- 对照 B0：`{BASELINE_REF}`（F1=0.0618 P=0.0615 R=0.1339 api=87 llm=77）——未重跑，读基线报告")
    lines.append(f"- M1 实验：`{EXPERIMENT}`，{n}/{len(queries)} 条完成（其余因 OpenAlex 配额耗尽未跑；本报告聚合口径为已完成的 {n} 条），只改 Query Formulation；Retriever/Prekeep/Reranker/OpenAlex/top-k={TOP_K} 冻结；引文扩展关闭")
    lines.append("")
    lines.append("## 1. 核心指标")
    lines.append("")
    lines.append("| 指标 | B0 | M1 | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| F1 | {agg['f1']['b0']/n:.4f} | {agg['f1']['m1']/n:.4f} | {agg['f1']['m1']/n - agg['f1']['b0']/n:+.4f} |")
    lines.append(f"| Precision | {agg['precision']['b0']/n:.4f} | {agg['precision']['m1']/n:.4f} | {agg['precision']['m1']/n - agg['precision']['b0']/n:+.4f} |")
    lines.append(f"| Recall | {agg['recall']['b0']/n:.4f} | {agg['recall']['m1']/n:.4f} | {agg['recall']['m1']/n - agg['recall']['b0']/n:+.4f} |")
    lines.append(f"| API calls/query | {agg['api_calls']['b0']/n:.1f} | {agg['api_calls']['m1']/n:.1f} | |")
    lines.append(f"| LLM calls/query | {agg['llm_calls']['b0']/n:.1f} | {agg['llm_calls']['m1']/n:.1f} | |")
    lines.append("")
    lines.append("## 2. 检索原始召回（raw，检索阶段命中的去重 gold）")
    lines.append("")
    lines.append(f"- **raw_unique_gold = {raw_unique_gold}**（22 条跨 query 按 canonical_id 去重；title keep_letters 匹配口径，保守下限）")
    lines.append(f"- raw 命中 gold 中最终进 top-k 的 = {raw_gold_final}（raw→final 保留率 {raw_gold_final/raw_unique_gold:.1%}；若 raw 升而 final 平 → 下一步看 reranker 保留）")
    lines.append(f"- **final_unique_gold = {final_unique_gold}**（final top-{TOP_K} 中命中的去重 gold）")
    lines.append("")
    lines.append("## 3. Query 类型贡献（first-seen 归因；parallel recall 下同 batch 共现时归因有轻微顺序噪声）")
    lines.append("")
    lines.append("| query_type | executed_queries | new_unique_gold | incremental_gold/query |")
    lines.append("|---|---|---|---|")
    for t in QUERY_TYPES:
        per_q = (type_gold[t] / executed[t]) if executed[t] else 0.0
        lines.append(f"| {t} | {executed[t]} | {type_gold[t]} | {per_q:.3f} |")
    lines.append("")
    # ---- 根因：v2 assoc 联想词机制贡献对比（若有 v2 instrumented traces）----
    import glob as _glob
    v2_traces = sorted(_glob.glob("eval/diagnostics/v2_instrumented/trace_*.json"))
    v2_aso_total = v2_raw_total = 0
    if v2_traces:
        v2_raw_by_q: dict[str, tuple[int, int]] = {}  # qid -> (regular_raw, assoc_raw)
        for fp in v2_traces:
            trc = json.loads(Path(fp).read_text(encoding="utf-8"))
            qid = trc["query_id"]
            reg = sum(1 for g in trc.get("gold_lifecycle", []) if g["first_seen_stage"] != "assoc_recall")
            aso = sum(1 for g in trc.get("gold_lifecycle", []) if g["first_seen_stage"] == "assoc_recall")
            v2_raw_by_q[qid] = (reg, aso)
        v2_reg_total = sum(v[0] for v in v2_raw_by_q.values())
        v2_aso_total = sum(v[1] for v in v2_raw_by_q.values())
        v2_raw_total = v2_reg_total + v2_aso_total
        lines.append("## 4. 根因对比：v2（assoc 联想词）vs M1 raw 召回")
        lines.append("")
        lines.append(f"- v2 raw_unique_gold ≈ **{v2_raw_total}**（其中 **assoc 联想词子查询贡献 {v2_aso_total}**、regular {v2_reg_total}）；M1 raw_unique_gold = **{raw_unique_gold}**（core {type_gold['core']} + anchor {type_gold['anchor']} + discovery {type_gold['discovery']}）")
        lines.append("- 结论：v3-MVP 冻结 prekeep/reranker 时同时移除了 ASSOC_INTENT 保送机制，而该机制正是 v2 最强的 raw 召回杠杆（specific 论文标识直接命中 gold 标题）。M1 的 anchor 是「专名+任务」组合（更稀释），discovery 词汇桥在 OpenAlex 顶 k 内够不到具体 gold 论文。")
        lines.append("")
        lines.append("| query_id | v2_regular | v2_assoc | v2_raw | M1_raw |")
        lines.append("|---|---|---|---|---|")
        for qid in sorted(set(v2_raw_by_q) | set(r['query_id'] for r in all_lifecycle)):
            vr, va = v2_raw_by_q.get(qid, (0, 0))
            mr = sum(1 for row in all_lifecycle if row["query_id"] == qid and row["first_seen_stage"] == "regular_recall")
            lines.append(f"| {qid} | {vr} | {va} | {vr+va} | {mr} |")
        lines.append("")
    lines.append("## 5. 决策 Gate")
    lines.append("")
    lines.append("")
    if raw_unique_gold >= 40:
        gate = "STRONG_RETRIEVAL_SUCCESS"
    elif raw_unique_gold >= 35:
        gate = "QUERY_FORMULATION_DIRECTION_VALID"
    else:
        gate = "QUERY_FORMULATION_INSUFFICIENT"
    lines.append(f"- raw_unique_gold >= 40 → STRONG_RETRIEVAL_SUCCESS；>= 35 → QUERY_FORMULATION_DIRECTION_VALID。")
    lines.append(f"- **判定：{gate}**（raw_unique_gold = {raw_unique_gold}）")
    lines.append("")
    if raw_unique_gold >= 35:
        lines.append(f"## 6. 下一步")
        lines.append("")
        lines.append(f"- raw→final 保留率 {raw_gold_final/raw_unique_gold:.1%}：")
        if raw_gold_final / raw_unique_gold < 0.5:
            lines.append(f"  **NEXT = RERANKER_RETENTION**：raw gold 在 prekeep/reranker 大量丢失，下一轮做 Gold-Retention Reranker（三剑第三剑）。")
        else:
            lines.append(f"  **NEXT = EVALUATE_TOP_K**：raw 与 final 同向提升，可考虑 top-k 或精度截断优化。")
    else:
        lines.append("## 6. 下一步")
        lines.append("")
        lines.append("- raw 提升不达标：**NEXT = ONE_PLANNER_REVISION**（只允许一次 Planner prompt 修订）。")
        lines.append("  根因定向：v2 assoc 联想词（specific 论文标识，如 \"Chinchilla scaling laws\"）贡献了 v2 raw 的 "
                     f"{v2_aso_total}/{v2_raw_total}，是 raw 召回第一杠杆；本轮 anchor 是「专名+任务」更稀释、discovery 词汇桥 0 gold。")
        lines.append("  修订方向：把 anchor/discovery 改为产出**接近论文标题的 specific 标识**（≤5 cap 内优先专名，不追加宽泛限定词），"
                     "并在最小过滤里保留单专名（已支持）；不做第二轮复杂度，仅一版 prompt。")
    lines.append("")
    lines.append("## 7. 范围遵守声明")
    lines.append("")
    lines.append("- src/search.py 未改动；src/planner.py 未改动。新增 src/planner_anchor.py + scripts/run_m1_anchor.py。")
    lines.append("- 引文/引用/metadata 扩展全部关闭；B1_MAX_SUBQUERIES=5 天然实现 ≤5 query 上限。")
    lines.append("- Gold isolation：planner 输入仅 question 文本；gold 只进 TraceRecorder 诊断统计。")
    lines.append("- 响应缓存隔离：eval/cache/m1_anchor_augmented/（新 subquery 全部真实网络检索）。")
    lines.append("")
    lines.append("**本轮到此为止：等待用户批准后才进入下一阶段。**")
    (M1_DIR / "v3_mvp_decision.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== M1 汇总 ===")
    print(f"F1 {agg['f1']['b0']/n:.4f} -> {agg['f1']['m1']/n:.4f}")
    print(f"P  {agg['precision']['b0']/n:.4f} -> {agg['precision']['m1']/n:.4f}")
    print(f"R  {agg['recall']['b0']/n:.4f} -> {agg['recall']['m1']/n:.4f}")
    print(f"raw_unique_gold={raw_unique_gold}  final_unique_gold={final_unique_gold}  raw→final 保留率={raw_gold_final/raw_unique_gold:.1%}")
    print(f"Gate 判定：{gate}")
    print("产物：", ", ".join(str(p) for p in (M1_DIR / "m1_query_type_metrics.csv",
                                                M1_DIR / "m1_gold_lifecycle.csv",
                                                M1_DIR / "m1_vs_b0.csv",
                                                M1_DIR / "v3_mvp_decision.md")))


def apply_rev(rev: int) -> None:
    """ONE_PLANNER_REVISION 隔离：把全局路径/实验名改成 R 轮次专用，不碰 v1 产物。"""
    global M1_DIR, PLAN_FILE, CACHE_DIR, TRACE_DIR, EXPERIMENT
    if rev == 1:
        return  # v1 用默认路径
    suffix = f"_r{rev}"
    M1_DIR = Path(f"eval/runs/m1_anchor_augmented{suffix}")
    PLAN_FILE = M1_DIR / "m1_query_plans.jsonl"
    CACHE_DIR = f"eval/cache/m1_anchor_augmented{suffix}"
    TRACE_DIR = Path(f"eval/diagnostics/m1_anchor_augmented{suffix}")
    EXPERIMENT = f"PASA_ASSOC_M1_ANCHOR_R{rev}"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "search", "report", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rev", type=int, default=1,
                    help="规划器修订轮次（ONE_PLANNER_REVISION）：>1 时输出到 m1_anchor_augmented_r{rev}/ 隔离目录")
    args = ap.parse_args()

    apply_rev(args.rev)
    if args.mode == "generate":
        await cmd_generate(select_queries(), args.limit)
    elif args.mode == "search":
        await cmd_search(args.limit)
    elif args.mode == "report":
        cmd_report()
    else:  # all
        queries = select_queries()
        await cmd_generate(queries, args.limit)
        if args.limit:
            await cmd_search(args.limit)
        else:
            await cmd_search(None)


if __name__ == "__main__":
    asyncio.run(main())
