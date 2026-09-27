"""Phase 2 NO_CITATION controlled evaluation（指令 C 的受控实验）。

在完全相同的冻结 v2 plan + 相同 regular/assoc response cache 下，关闭
citation/reference/metadata 三阶段（enable_citation_expansion=False），
保留相同 candidate selection 与相同 LLM reranker，比较 final F1 /
logical calls / physical calls / latency / reranker candidate count。

纯离线 replay：ResponseCache(mode="replay")，绝无网络请求（未命中抛 CacheMiss）。

用法：
  python3 scripts/replay_nocitation.py --runs-dir eval/runs/nocitation --trace-dir eval/diagnostics/v2_instrumented/nocit_traces
  # 对 F1 有差异的 query 追加 --only <qid> 配合多次运行做 3-seed/3-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import compute_p_r_f1, match_gold
from src.observability.response_cache import CacheMiss, ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

PLAN_CACHE_PATH = "eval/runs/_pasa_plan_cache.jsonl"
INSTR_EXPERIMENT = "PASA_ASSOC_NO_CIT"
CACHE_DIR = "eval/cache/v2_instrumented"
FULL_RUNS_DIR = "eval/runs/instrumented"


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
    ap.add_argument("--only", default=None, help="只跑指定 query_id（逗号分隔，3-seed 验证用）")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--runs-dir", default="eval/runs/nocitation")
    ap.add_argument("--tag", default="", help="输出文件后缀，3-seed 多次运行用（如 seed1）")
    args = ap.parse_args()

    all_queries = load_pasa(args.data)
    plans = load_plans(Path(PLAN_CACHE_PATH))
    queries = [q for q in all_queries if q["query"] in plans]
    if not queries:
        raise SystemExit("EXPERIMENT_INVALID: 与 v2 冻结 plan 无交集")

    only = set(args.only.split(",")) if args.only else None
    if only:
        queries = [q for q in queries if q["query_id"] in only]
        print(f"[3-seed] 只跑 {len(queries)} 条: {only}")

    cache = ResponseCache(args.cache_dir, mode="replay")
    engine = SearchEngine(response_cache=cache, enable_citation_expansion=False)
    engine.load_plan_cache(PLAN_CACHE_PATH)

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, q in enumerate(queries, 1):
        qid = q["query_id"]
        gold_groups = match_gold(q)
        gold_titles = [g["title"] for g in q.get("gold", []) if g.get("title")]

        recorder = TraceRecorder(
            query_id=qid,
            run_id=INSTR_EXPERIMENT,
            baseline_reference_id="PASA_ASSOC_v2",
            response_cache=cache,
        )
        recorder.set_gold_titles(gold_titles)
        engine.recorder = recorder

        t0 = time.time()
        try:
            results_, telemetry, traces = await engine.search_full(q["query"], top_k=20)
        except CacheMiss as e:
            print(f"[{i}/{len(queries)}] {qid}: CACHE_MISS {e}")
            cache.close()
            raise
        latency_ms = round((time.time() - t0) * 1000, 1)

        metrics = compute_p_r_f1(results_, gold_groups)
        # rerank_pool 权威口径：recorder.candidate_snapshots 里 rerank_pool 快照
        pool_count = None
        for snap in recorder.candidate_snapshots:
            if snap.get("stage") == "rerank_pool":
                pool_count = snap.get("candidate_count")
                break
        raw_count = None
        for snap in recorder.candidate_snapshots:
            if snap.get("stage") == "after_raw_recall":
                raw_count = snap.get("candidate_count")
                break
        row = {
            "query_id": qid,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "api_calls": telemetry.api_calls,
            "llm_calls": telemetry.llm_calls,
            "latency_ms": latency_ms,
            "n_predicted": len(results_),
            "raw_candidates": raw_count,
            "rerank_pool_candidates": pool_count,
        }
        results.append(row)
        print(f"[{i}/{len(queries)}] {qid}: F1={metrics['f1']:.4f} "
              f"api={telemetry.api_calls} llm={telemetry.llm_calls} {latency_ms}ms")

    cache.close()
    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.runs_dir) / f"nocitation_results{tag}.json"
    out.write_text(json.dumps({"experiment": INSTR_EXPERIMENT, "results": results},
                              ensure_ascii=False, indent=2))
    print(f"\n写入 {out}（{len(results)} 条）")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
