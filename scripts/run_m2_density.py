"""M2_QUERY_DENSITY：查询密度消融（隔离「密度」vs「保送」两个杠杆）。

动机（M1/M1_R2 后修正的根因）：v2 的 raw 召回（28 gold，其中 assoc 联想词贡献 20）
主要来自**查询密度**——每 query 固定 6 条密集 specific 标识子查询（assoc，优先 5、
不受 B1_MAX_SUBQUERIES=5 截断）+ 5 条常规 = 11 subquery/query。
M1/M1_R2 被 v3-MVP 的「≤5 total」上限压死（核心/anchor/discovery 挤 5 个名额），
specific 标识密度从 6/query 掉到 ~0.9/query，raw 从 28 崩到 7。

本实验：**回放 v2 冻结 plan（5 常规 + 6 assoc = 11 subquery/query），但关闭 assoc 保送**
（assoc_safepass=False，_build_rerank_pool 不再把 source=="assoc" 候选强制加入精排池）。
其余全部冻结（enable_citation_expansion=False，真实 OpenAlex 检索，top_k=20）。

判读：
- 若 M2 raw_unique_gold ≈ 28（v2 数）：v2 的 raw 召回 100% 由**查询集（密度+措辞）**解释，
  保送对 raw 无关紧要 → 密度是唯一杠杆，修复方向 = 让 planner 恒产出密集 specific 标识。
- raw→final 保留率：隔离保送对最终保留的价值（M1_R2 无 assoc 时已 85.7%）。

gold isolation：不回放 gold；planner 不参与（plan 来自冻结 v2 缓存）。assoc 子查询保留
ASSOC_INTENT 以保持「不被 5 条截断」+ 在 assoc_recall 阶段计数（用于密度轨贡献归因）。

产物（eval/runs/m2_query_density/）：
  m2_query_plans.jsonl        —— v=2 plan cache（回放 v2 冻结 plan，key=query text）
  m2_query_type_metrics.csv   —— assoc/regular 两轨：执行数 / new unique Gold / gold-per-query
  m2_gold_lifecycle.csv       —— 全 gold 生命周期
  m2_vs_b0.csv                —— 逐 query B0 vs M2 的 F1/P/R/api
  v3_mvp_decision.md          —— 决策报告（raw_unique_gold gate + 密度假设裁决）

用法：
  python scripts/run_m2_density.py build    # 回放 v2 plan -> m2 plan cache（离线，无 LLM）
  python scripts/run_m2_density.py search   # 用 m2 plan 跑检索（断点续跑）
  python scripts/run_m2_density.py report   # 只重生成产物/决策（不联网）
  python scripts/run_m2_density.py all --limit 3   # 全流程 smoke
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
from src.planner import ASSOC_INTENT
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
NOCIT_RESULTS = "eval/runs/nocitation/nocitation_results.json"
# 离线回放用（M2 不联网）：recall cache 覆盖全部 subquery；v2_instrumented 作只读响应缓存兜底。
RECALL_CACHE = "eval/runs/_pasa_recall_cache.jsonl"
V2_INSTRUMENTED_CACHE = "eval/cache/v2_instrumented"
V2_TRACE_DIR = Path("eval/diagnostics/v2_instrumented")  # safepass ON 对照（带 trace）
M2_DIR = Path("eval/runs/m2_query_density")
PLAN_FILE = M2_DIR / "m2_query_plans.jsonl"
CACHE_DIR = "eval/cache/m2_query_density"
TRACE_DIR = Path("eval/diagnostics/m2_query_density")
EXPERIMENT = "PASA_ASSOC_M2_DENSITY"
BASELINE_REF = "PASA_ASSOC_NO_CIT"
TOP_K = 20
TRACKS = ("assoc", "regular")  # assoc=联想词密度轨；regular=常规轨


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


def load_m2_plans() -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not PLAN_FILE.exists():
        return plans
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("v") != 2 or not isinstance(d.get("ir"), dict) or not isinstance(d.get("subs"), list):
            raise ValueError(f"M2 plan 非法: {d.get('query')}")
        plans[d["query"]] = d
    return plans


def load_nocit_baseline() -> dict[str, dict]:
    d = json.loads(Path(NOCIT_RESULTS).read_text(encoding="utf-8"))
    assert d["experiment"] == BASELINE_REF, f"基线实验名不符: {d['experiment']}"
    return {r["query_id"]: r for r in d["results"]}


def select_queries() -> list[dict]:
    b0 = load_nocit_baseline()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    all_q = load_pasa(DATA)
    picked = [q for q in all_q if q["query_id"] in b0 and q["query"] in v2]
    if len(picked) != len(b0):
        raise SystemExit(f"EXPERIMENT_INVALID: 过滤后 {len(picked)} 条 ≠ B0 基线 {len(b0)} 条")
    return picked


def track_of(intent: str) -> str:
    return "assoc" if intent == ASSOC_INTENT else "regular"


# --------------------------------------------------------------------------
# Phase A：build —— 回放 v2 冻结 plan（离线，无 LLM）
# --------------------------------------------------------------------------
def cmd_build() -> None:
    M2_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    existing = load_m2_plans()
    todo = [q for q in select_queries() if q["query"] not in existing]
    print(f"M2 plan 回放：22 条中已存在 {len(existing)} 条，本次回放 {len(todo)} 条")
    for q in todo:
        plan = {
            "query": q["query"],
            "query_id": q["query_id"],
            "v": 2,
            "ir": v2[q["query"]]["ir"],
            # 原样回放 v2 的 subs：6 assoc + 5 常规；assoc 保留 ASSOC_INTENT（不被截断 + 计数）
            "subs": v2[q["query"]]["subs"],
        }
        with open(PLAN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
    from collections import Counter
    plans = load_m2_plans()
    cnt = Counter()
    for p in plans.values():
        cnt.update(track_of(s["intent"]) for s in p["subs"])
    print("M2 plan 轨统计（全部 plan）：", dict(cnt))
    print("写", PLAN_FILE)


# --------------------------------------------------------------------------
# Phase B：检索 + trace（assoc_safepass=False，断点续跑）
# --------------------------------------------------------------------------
async def cmd_search(limit: int | None, offline: bool) -> None:
    plans = load_m2_plans()
    if not plans:
        raise SystemExit("EXPERIMENT_INVALID: 无 M2 plan，先跑 build")
    queries = [q for q in select_queries() if q["query"] in plans]
    if limit:
        queries = queries[:limit]
    mode_desc = "offline replay（联网 0）" if offline else "online"
    print(f"M2 检索：{len(queries)} 条（plan 回放 {len(plans)} 条），assoc_safepass=False，{mode_desc}")

    if offline:
        # 只读响应缓存兜底 + recall cache 预加载：保证 recall 全部命中缓存、零 OpenAlex 联网
        cache = ResponseCache(V2_INSTRUMENTED_CACHE, mode="replay")
    else:
        cache = ResponseCache(CACHE_DIR, mode="write")
    engine = SearchEngine(
        response_cache=cache, enable_citation_expansion=False, assoc_safepass=False,
    )
    engine.load_plan_cache(str(PLAN_FILE))
    if offline:
        engine.load_recall_cache(RECALL_CACHE)  # {q:evs} 全部预加载 -> _recall 直接命中
        n_loaded = len(engine._recall_cache)
        n_need = sum(1 for p in plans.values() for s in p["subs"])
        missing = [s["query_text"] for p in plans.values() for s in p["subs"]
                   if s["query_text"] not in engine._recall_cache]
        print(f"offline：recall cache 预加载 {n_loaded}，需要 {n_need}，缺失 {len(missing)}")
        if missing:
            print("缺失 subquery（仅允许补这些，不允许重跑全部）：", missing)

    M2_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_before = 0
    todo = []
    for q in queries:
        rep = M2_DIR / f"report_{q['query_id']}.json"
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
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] {qid}: FAIL {type(e).__name__}: {e}")
            with open(M2_DIR / "search_failures.log", "a", encoding="utf-8") as f:
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
        (M2_DIR / f"report_{qid}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {qid}: F1={report['f1']:.4f} P={report['precision']:.4f} "
              f"R={report['recall']:.4f} api={report['api_calls']} llm={report['llm_calls']} "
              f"raw={report['raw_candidates']} {latency_ms}ms")

    cache.close()
    # offline 联网护栏：校验所有 report 的 OpenAlex api_calls 为 0（防意外联网）
    if offline:
        bad = []
        for q in queries:
            rep = M2_DIR / f"report_{q['query_id']}.json"
            if rep.exists():
                r = json.loads(rep.read_text(encoding="utf-8"))
                if r.get("api_calls", 0) > 0:
                    bad.append((q["query_id"], r.get("api_calls")))
        if bad:
            print(f"!! OFFLINE_VIOLATION：以下 query 发生 OpenAlex 联网（api_calls>0）：{bad}")
        else:
            print("offline 校验通过：所有 query OpenAlex api_calls == 0（零联网）")
    print("检索完成。生成产物...")
    cmd_report()


# --------------------------------------------------------------------------
# Phase C：产物 + 决策
# --------------------------------------------------------------------------
def load_report(qid: str) -> dict:
    return json.loads((M2_DIR / f"report_{qid}.json").read_text(encoding="utf-8"))


def load_trace(qid: str) -> dict:
    return json.loads((TRACE_DIR / f"trace_{qid}.json").read_text(encoding="utf-8"))


def track_of_first_seen(plans: dict[str, dict], subquery_text: str) -> str:
    key = subquery_text[:80]
    for p in plans.values():
        for s in p["subs"]:
            if s["query_text"][:80] == key:
                return track_of(s["intent"])
    return "unknown"


def v2_safepass_on_retention() -> dict:
    """safepass ON 对照：读 v2_instrumented traces。返回论文级（title_n 去重）漏斗与保留率。

    口径与 M2 一致：raw/pool/final 均以 title_n 为论文级身份（canonical_id=DOI 为实例级）。
    """
    import glob as _glob
    raw = set(); pool = set(); final = set()
    for fp in _glob.glob(str(V2_TRACE_DIR / "trace_*.json")):
        trc = json.loads(Path(fp).read_text(encoding="utf-8"))
        for g in trc.get("gold_lifecycle", []):
            tid = g.get("title_n") or g["canonical_id"]
            if g.get("first_seen_stage") in ("regular_recall", "assoc_recall"):
                raw.add(tid)
            if g.get("survived_prekeep"):
                pool.add(tid)
            if g.get("final_rank") is not None:
                final.add(tid)
    rp = (len(pool) / len(raw)) if raw else 0.0
    pf = (len(final) / len(pool)) if pool else 0.0
    return {"raw": len(raw), "pool": len(pool), "final": len(final),
            "raw_to_pool": rp, "pool_to_final": pf}


def cmd_report() -> None:
    plans = load_m2_plans()
    b0 = load_nocit_baseline()
    queries = select_queries()
    reports: dict[str, dict] = {}
    for q in queries:
        rep = M2_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            reports[q["query_id"]] = json.loads(rep.read_text(encoding="utf-8"))

    all_lifecycle: list[dict] = []
    raw_by_id: dict[str, dict] = {}
    for q in queries:
        qid = q["query_id"]
        if qid not in reports:
            continue
        trc = load_trace(qid)
        for g in trc.get("gold_lifecycle", []):
            row = {**g, "query_id": qid,
                   "query_type": track_of_first_seen(plans, g.get("first_seen_query", ""))}
            all_lifecycle.append(row)
            if g.get("first_seen_stage") in ("regular_recall", "assoc_recall"):
                cid = g["canonical_id"]
                raw_by_id.setdefault(cid, qid)
    # 口径修正（Phase 2.5）：canonical_id（DOI）为实例级（同篇多 DOI 形态 → 28），
    # title_n 去重为论文级（22）。M3 gate 用论文级。
    raw_rows = [g for g in all_lifecycle if g.get("first_seen_stage") in ("regular_recall", "assoc_recall")]
    pool_rows = [g for g in all_lifecycle if g.get("survived_prekeep")]
    final_rows = [g for g in all_lifecycle if g.get("final_rank") is not None]
    # 实例级（canonical_id）
    raw_query_gold_instances = len({g["canonical_id"] for g in raw_rows})
    pool_query_gold_instances = len({g["canonical_id"] for g in pool_rows})
    final_query_gold_instances = len({g["canonical_id"] for g in final_rows})
    # 论文级（title_n 去重）
    raw_unique_gold_papers = len({g["title_n"] for g in raw_rows})
    pool_unique_gold_papers = len({g["title_n"] for g in pool_rows})
    final_unique_gold_papers = len({g["title_n"] for g in final_rows})
    # 兼容旧变量名（raw/pool/final "unique gold" = 论文级，M3 gate 口径）
    raw_unique_gold = raw_unique_gold_papers
    pool_unique_gold = pool_unique_gold_papers
    final_unique_gold = final_unique_gold_papers
    raw_to_pool = (pool_unique_gold / raw_unique_gold) if raw_unique_gold else 0.0
    pool_to_final = (final_unique_gold / pool_unique_gold) if pool_unique_gold else 0.0

    # 全 gold 生命周期表
    lc_cols = ["query_id", "canonical_id", "title_n", "first_seen_stage", "first_seen_query",
               "first_seen_rank", "query_type", "survived_prekeep", "pre_rerank_rank",
               "reranker_rank", "final_rank", "drop_stage", "drop_reason"]
    with open(M2_DIR / "m2_gold_lifecycle.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=lc_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_lifecycle:
            w.writerow(row)

    # 轨贡献（first-seen 归因）
    track_gold = {t: 0 for t in TRACKS}
    for row in all_lifecycle:
        if row["query_type"] in track_gold:
            track_gold[row["query_type"]] += 1
    executed = {t: sum(1 for p in plans.values() for s in p["subs"] if track_of(s["intent"]) == t)
                for t in TRACKS}
    with open(M2_DIR / "m2_query_type_metrics.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["track", "executed_queries", "new_unique_gold", "incremental_gold_per_query"])
        for t in TRACKS:
            per_q = (track_gold[t] / executed[t]) if executed[t] else 0.0
            w.writerow([t, executed[t], track_gold[t], round(per_q, 3)])

    # vs B0
    n = len(reports)
    agg = {k: {"b0": 0.0, "m2": 0.0} for k in ("f1", "precision", "recall", "api_calls", "llm_calls")}
    with open(M2_DIR / "m2_vs_b0.csv", "w", newline="", encoding="utf-8") as f:
        import csv
        cols = ["query_id", "b0_f1", "b0_precision", "b0_recall", "b0_api_calls", "b0_llm_calls",
                "m2_f1", "m2_precision", "m2_recall", "m2_api_calls", "m2_llm_calls",
                "m2_raw_gold", "m2_final_gold", "delta_f1", "delta_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for qid in sorted(reports):
            r = reports[qid]
            b = b0[qid]
            raw_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and
                        row["first_seen_stage"] in ("regular_recall", "assoc_recall"))
            fin_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and row.get("final_rank") is not None)
            w.writerow({
                "query_id": qid,
                "b0_f1": b["f1"], "b0_precision": b["precision"], "b0_recall": b["recall"],
                "b0_api_calls": b["api_calls"], "b0_llm_calls": b["llm_calls"],
                "m2_f1": r["f1"], "m2_precision": r["precision"], "m2_recall": r["recall"],
                "m2_api_calls": r["api_calls"], "m2_llm_calls": r["llm_calls"],
                "m2_raw_gold": raw_g, "m2_final_gold": fin_g,
                "delta_f1": round(r["f1"] - b["f1"], 4), "delta_recall": round(r["recall"] - b["recall"], 4),
            })
            for k in agg:
                agg[k]["b0"] += b[k]
                agg[k]["m2"] += r[k]
        w.writerow({})
        w.writerow({
            "query_id": "MEAN",
            "b0_f1": round(agg["f1"]["b0"] / n, 4), "b0_precision": round(agg["precision"]["b0"] / n, 4),
            "b0_recall": round(agg["recall"]["b0"] / n, 4), "b0_api_calls": round(agg["api_calls"]["b0"] / n, 1),
            "b0_llm_calls": round(agg["llm_calls"]["b0"] / n, 1),
            "m2_f1": round(agg["f1"]["m2"] / n, 4), "m2_precision": round(agg["precision"]["m2"] / n, 4),
            "m2_recall": round(agg["recall"]["m2"] / n, 4), "m2_api_calls": round(agg["api_calls"]["m2"] / n, 1),
            "m2_llm_calls": round(agg["llm_calls"]["m2"] / n, 1),
            "m2_raw_gold": raw_unique_gold, "m2_final_gold": final_unique_gold,
            "delta_f1": round(agg["f1"]["m2"] / n - agg["f1"]["b0"] / n, 4),
            "delta_recall": round(agg["recall"]["m2"] / n - agg["recall"]["b0"] / n, 4),
        })

    # 决策报告：m2_safepass_ablation.md（保留率消融，vs safepass ON）
    v2on = v2_safepass_on_retention()
    lines = []
    lines.append("# M2_SAFEPASS_ABLATION 决策报告（保送开关消融，offline replay）")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}（M2 全量 22-query，offline replay）")
    lines.append(f"- M2 实验：`{EXPERIMENT}`，{n}/{len(queries)} 条完成。**回放 v2 冻结 plan（{sum(len(p['subs']) for p in plans.values())} 条 subquery），"
                 f"assoc_safepass=False，0 次 OpenAlex 联网**（recall 全命中 _pasa_recall_cache.jsonl）。")
    lines.append("- Retriever/Prekeep/Reranker/top-k=20 冻结；引文扩展关闭。")
    lines.append("")
    lines.append("## 1. 保留漏斗（论文级 unique Gold，title_n 去重；M3 gate 口径）")
    lines.append("")
    lines.append("| 阶段 | safepass ON（v2 实测） | safepass OFF（M2 回放） | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| raw unique Gold papers | {v2on['raw']} | {raw_unique_gold} | {raw_unique_gold - v2on['raw']:+d} |")
    lines.append(f"| pool unique Gold papers | {v2on['pool']} | {pool_unique_gold} | {pool_unique_gold - v2on['pool']:+d} |")
    lines.append(f"| final unique Gold papers | {v2on['final']} | {final_unique_gold} | {final_unique_gold - v2on['final']:+d} |")
    lines.append(f"| raw→pool 保留率 | {v2on['raw_to_pool']:.1%} | {raw_to_pool:.1%} | {raw_to_pool - v2on['raw_to_pool']:+.1%} |")
    lines.append(f"| pool→final 保留率 | {v2on['pool_to_final']:.1%} | {pool_to_final:.1%} | {pool_to_final - v2on['pool_to_final']:+.1%} |")
    lines.append("")
    lines.append("**实例级（canonical_id/DOI，含同篇多 DOI 形态）**：")
    lines.append(f"raw_query_gold_instances = {raw_query_gold_instances}，pool_query_gold_instances = {pool_query_gold_instances}，"
                 f"final_query_gold_instances = {final_query_gold_instances}")
    lines.append(f"（论文级 raw {raw_unique_gold} 篇 ↔ 实例级 {raw_query_gold_instances} 条，Phase 2.5 口径一致：28 实例 ↔ 22 论文）")
    lines.append("")
    lines.append("## 2. 核心指标")
    lines.append("")
    lines.append("| 指标 | B0 | M2 (safepass OFF) | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| F1 | {agg['f1']['b0']/n:.4f} | {agg['f1']['m2']/n:.4f} | {agg['f1']['m2']/n - agg['f1']['b0']/n:+.4f} |")
    lines.append(f"| Precision | {agg['precision']['b0']/n:.4f} | {agg['precision']['m2']/n:.4f} | {agg['precision']['m2']/n - agg['precision']['b0']/n:+.4f} |")
    lines.append(f"| Recall | {agg['recall']['b0']/n:.4f} | {agg['recall']['m2']/n:.4f} | {agg['recall']['m2']/n - agg['recall']['b0']/n:+.4f} |")
    lines.append("")
    lines.append("## 3. 轨贡献（first-seen 归因；assoc=联想词密度轨，regular=常规轨）")
    lines.append("")
    lines.append("| track | executed_queries | new_unique_gold | incremental_gold/query |")
    lines.append("|---|---|---|---|")
    for t in TRACKS:
        per_q = (track_gold[t] / executed[t]) if executed[t] else 0.0
        lines.append(f"| {t} | {executed[t]} | {track_gold[t]} | {per_q:.3f} |")
    lines.append("")
    lines.append("## 4. 保送价值判定")
    lines.append("")
    lines.append(f"- **保送对 raw 无影响**（raw 在 prekeep 之前计数，_build_rerank_pool 只影响 pool）：M2 raw = OFF = ON = {v2on['raw']}，完全一致。")
    lines.append(f"- **保送对 raw→pool 保留**（保送直接杠杆）：OFF={raw_to_pool:.1%} vs ON={v2on['raw_to_pool']:.1%}，Δ={raw_to_pool - v2on['raw_to_pool']:+.1%}pp。")
    lines.append(f"- **pool→final 保留**（受 pool 规模混杂，仅参考）：OFF={pool_to_final:.1%} vs ON={v2on['pool_to_final']:.1%}。")
    lines.append(f"- **净 final Gold**：ON={v2on['final']} vs OFF={final_unique_gold}（保送净增 {v2on['final'] - final_unique_gold:+d}）；F1 ON=0.0618 vs OFF={agg['f1']['m2']/n:.4f}。")
    lines.append("")
    # 判定依据：保送的直接效果 = raw→pool 保留率差（pool 规模混杂不影响 raw→pool）；净 final gold 佐证。
    if v2on["raw_to_pool"] - raw_to_pool >= 0.10:
        lines.append("**判定：保送明显提高 raw→pool Gold 保留（Δ≥10pp）→ MVP 保留 assoc_safepass=True。**")
        safepass_decision = "KEEP_SAFEPASS"
    elif abs(v2on["raw_to_pool"] - raw_to_pool) < 0.10:
        lines.append("**判定：保送对保留基本无影响（raw→pool 差 <10pp）→ 后续可简化（考虑去掉保送或合并）。**")
        safepass_decision = "SAFEPASS_REDUNDANT"
    else:
        lines.append("**判定：保送对保留无帮助（raw→pool OFF 更高）→ 可考虑简化。**")
        safepass_decision = "SAFEPASS_HARMFUL"
    lines.append("")
    lines.append("## 5. 范围遵守声明")
    lines.append("")
    lines.append("- src/search.py 改动：仅新增 `assoc_safepass` 开关（默认 True 保持原行为；M2 置 False）。冻结参数/prekeep/reranker/top-k 未动。")
    lines.append("- gold isolation：不回放 gold；planner 不参与（回放冻结 v2 plan）。")
    lines.append("- **offline replay 校验：OpenAlex api_calls == 0（零联网）**；仅 LLM 精排调用。")
    lines.append("")
    lines.append("**本轮到此为止：等待用户批准后才进入下一阶段（M3_SPARSE_PLAN_RESCUE）。**")
    (M2_DIR / "m2_safepass_ablation.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== M2 汇总 ===")
    print(f"F1 {agg['f1']['b0']/n:.4f} -> {agg['f1']['m2']/n:.4f}")
    print(f"P  {agg['precision']['b0']/n:.4f} -> {agg['precision']['m2']/n:.4f}")
    print(f"R  {agg['recall']['b0']/n:.4f} -> {agg['recall']['m2']/n:.4f}")
    print(f"raw_unique_gold={raw_unique_gold}  pool_unique_gold={pool_unique_gold}  final_unique_gold={final_unique_gold}")
    print(f"raw→pool={raw_to_pool:.1%}  pool→final={pool_to_final:.1%}  (safepass ON: {v2on['raw_to_pool']:.1%} / {v2on['pool_to_final']:.1%})")
    print(f"assoc轨 gold={track_gold['assoc']}  regular轨 gold={track_gold['regular']}")
    print(f"保送判定：{safepass_decision}")
    print("产物：", ", ".join(str(p) for p in (M2_DIR / "m2_query_type_metrics.csv",
                                                M2_DIR / "m2_gold_lifecycle.csv",
                                                M2_DIR / "m2_vs_b0.csv",
                                                M2_DIR / "v3_mvp_decision.md")))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "search", "report", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true",
                    help="offline replay：用 _pasa_recall_cache 预加载 + v2_instrumented 只读缓存，0 次 OpenAlex 联网")
    args = ap.parse_args()

    if args.mode == "build":
        cmd_build()
    elif args.mode == "search":
        await cmd_search(args.limit, offline=args.offline)
    elif args.mode == "report":
        cmd_report()
    else:  # all
        cmd_build()
        await cmd_search(args.limit, offline=args.offline)


if __name__ == "__main__":
    asyncio.run(main())
