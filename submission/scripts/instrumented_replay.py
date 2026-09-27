"""Phase 2 instrumented replay：在冻结 v2 plan 下重跑一次 FULL（含引文）链路。

本次 run 是 Phase 2 唯一允许联网的完整 run：
- 冻结 plan：加载 eval/runs/_pasa_plan_cache.jsonl（v=2），严格校验版本与计划完整。
- 响应缓存：所有 OpenAlex/Crossref/OpenCitations HTTP 响应写入 eval/cache/v2_instrumented/
  （后续 Full/Top-1/Top-3/High-confidence/No Citation ablation 全部基于这份缓存离线 replay）。
- 逐 query TraceRecorder：planner 状态、每次逻辑 API 调用、候选 funnel、gold lifecycle。
- 输出报告写入 eval/runs/instrumented/（PASA_ASSOC_INSTR_*，不覆盖 v2 baseline）。
- 不加载 baseline recall cache 做检索（保证 subquery 真实搜索，响应进缓存，供离线重建）。

注意：gold 只进 recorder 诊断统计，绝不进入搜索逻辑（gold-derived query 禁止进入正式系统）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import compute_p_r_f1, match_gold
from src.observability.response_cache import ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.schemas import RunReport, SearchTrace
from src.search import SearchEngine
from src.telemetry import Telemetry
from scripts.eval_benchmark import load_pasa

PLAN_CACHE_PATH = "eval/runs/_pasa_plan_cache.jsonl"
BASELINE_REPORT_GLOB = "PASA_ASSOC_RealScholarQuery_*.json"
INSTR_EXPERIMENT = "PASA_ASSOC_INSTR"
BASELINE_REF = "PASA_ASSOC_v2"
TRACE_DIR = "eval/diagnostics/v2_instrumented"
CACHE_DIR = "eval/cache/v2_instrumented"
RUNS_DIR = "eval/runs/instrumented"


def plan_hash_of(plan: dict) -> str:
    """与 diagnose_v2_baseline.py 相同的 deterministic plan hash。"""
    return hashlib.sha256(
        json.dumps({"v": plan["v"], "ir": plan["ir"], "subs": plan["subs"]},
                   sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_plans(path: Path) -> dict[str, dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    plans = {}
    for r in rows:
        if r.get("v") != 2 or not isinstance(r.get("ir"), dict) or not isinstance(r.get("subs"), list):
            raise ValueError(f"non-v2 or incomplete plan: {r.get('query')}")
        plans[r["query"]] = r
    return plans


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/benchmarks/pasa/RealScholarQuery/test.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条（联调用）")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--trace-dir", default=TRACE_DIR)
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    args = ap.parse_args()

    all_queries = load_pasa(args.data)
    # 只跑 v2 baseline 覆盖的查询（v2 冻结 plan cache 里的那 22 条 PASA_ASSOC）
    plans = load_plans(Path(PLAN_CACHE_PATH))
    queries = [q for q in all_queries if q["query"] in plans]
    if not queries:
        raise SystemExit("EXPERIMENT_INVALID: v2 冻结 plan 与数据无交集")
    if args.limit:
        queries = queries[:args.limit]
    print(f"全量 PASA {len(all_queries)} 条 -> 过滤到 v2 冻结 plan 查询 {len(queries)} 条")
    print(f"冻结 v2 plan 校验通过：{len(plans)} 条，全部 v=2")

    cache = ResponseCache(args.cache_dir, mode="write")
    engine = SearchEngine(response_cache=cache)
    engine.load_plan_cache(PLAN_CACHE_PATH)
    # 故意不加载 baseline recall cache：subquery 真实搜索，响应进 response cache

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    # 断点续跑：已有 report + trace 的 query 跳过（缓存已在之前 flush 落盘，不重跑不重联网）
    skipped = 0
    resume = []
    for q in queries:
        rep = runs_dir / f"{INSTR_EXPERIMENT}_{q['query_id']}.json"
        trc = Path(args.trace_dir) / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            skipped += 1
            continue
        resume.append(q)
    queries = resume
    if skipped:
        print(f"断点续跑：跳过已完成 {skipped} 条，剩余 {len(queries)} 条")

    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        gold_groups = match_gold(q)
        gold_titles = [g["title"] for g in q.get("gold", []) if g.get("title")]

        recorder = TraceRecorder(
            query_id=qid,
            run_id=INSTR_EXPERIMENT,
            baseline_reference_id=BASELINE_REF,
            response_cache=cache,
        )
        recorder.set_gold_titles(gold_titles)
        engine.recorder = recorder  # 复用 engine（共享 plan/recall 缓存），逐 query 换 recorder

        import time
        t0 = time.time()
        results, telemetry, traces = await engine.search_full(q["query"], top_k=20)
        latency_ms = round((time.time() - t0) * 1000, 1)

        metrics = compute_p_r_f1(results, gold_groups)
        # recorder 的 finalize 已在 search_full 内调用；确保 snapshot 顺序保存
        recorder.save(args.trace_dir)
        cache.flush()  # 崩溃保护：每 query 后落盘缓存

        report = RunReport(
            experiment_id=INSTR_EXPERIMENT,
            query_id=qid,
            raw_query=q["query"],
            gold_ids=sorted(k for g in gold_groups for k in g),
            predicted_ids=[r.paper.paper_id for r in results],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            api_calls=telemetry.api_calls,
            llm_calls=telemetry.llm_calls,
            input_tokens=telemetry.input_tokens,
            output_tokens=telemetry.output_tokens,
            latency_ms=latency_ms,
            schema_valid=True,
            traces=traces,
            created_at="2026-08-24 12:00:00",
        )
        report_path = runs_dir / f"{INSTR_EXPERIMENT}_{qid}.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        reports.append(report)
        # plan hash 一致性自检
        ph = plan_hash_of(plans[q["query"]])
        ok = "OK" if recorder.plan_hash == ph else "MISMATCH"
        print(f"[{i}/{len(queries)}] {qid}: F1={report.f1:.4f} P={report.precision:.4f} "
              f"R={report.recall:.4f} api={report.api_calls} llm={report.llm_calls} "
              f"plan_hash={ok} {latency_ms}ms")

    cache.close()
    # summary 覆盖磁盘上全部已完成报告（断点续跑时含历史 query）
    all_reports = []
    for p in sorted(runs_dir.glob(f"{INSTR_EXPERIMENT}_*.json")):
        try:
            all_reports.append(RunReport.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    n = len(all_reports)
    summary = {
        "experiment": INSTR_EXPERIMENT,
        "queries": n,
        "mean_f1": round(sum(r.f1 for r in all_reports) / n, 4) if n else None,
        "total_api_calls": sum(r.api_calls for r in all_reports),
        "total_llm_calls": sum(r.llm_calls for r in all_reports),
        "physical_http_attempts": cache.total_physical_attempts(),
    }
    (Path(args.trace_dir) / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== instrumented replay 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
