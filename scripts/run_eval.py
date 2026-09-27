"""S1 eval entrypoint：在 22 个冻结查询上运行 FAST/DEEP 引擎，输出指标 + 报告。

用法：
    python3 scripts/run_eval.py --mode fast            # offline（默认）
    python3 scripts/run_eval.py --mode deep --online --limit 2

指标（标准化术语，docs/reports/s11_metric_definition.md）：
- Precision / Recall / F1（对 final ranked StructuredResult 与 gold 匹配）
- retrieval_raw_unique_gold / final_unique_gold / total_gold_papers
- 系统成本：api_calls / llm_calls / tokens / latency
- structured-output validity（所有 final 论文必须带 title，metadata 来自学术数据）
- 无 silent fallback 校验（offline 缺 frozen plan/cache → 该查询计为 FAILED）

Gold 匹配口径复用 eval/harness：同一论文的 openalex_id / doi / title / title_n 任一命中即命中。

Corpus version: pasa_realscholar_test_b3b570411ce2399c (frozen 22 queries, 184 Gold)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.harness import _norm_doi, _norm_title, _norm_title_letters, match_gold  # noqa: E402
from scripts.eval_benchmark import load_pasa  # noqa: E402
from s1.config import load_config  # noqa: E402
from s1.pipeline import ContestEngine  # noqa: E402

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
EVAL_CORPUS_VERSION = "pasa_realscholar_test_b3b570411ce2399c"


def structured_keys(r) -> set[str]:
    keys = set()
    if r.openalex_id:
        keys.add(f"openalex:{r.openalex_id}")
    doi = _norm_doi(r.doi)
    if doi:
        keys.add(f"doi:{doi}")
    if r.title:
        keys.add(f"title:{_norm_title(r.title)}")
        keys.add(f"title_n:{_norm_title_letters(r.title)}")
    return keys


def matched_gold_ids(results: list, gg: list[set[str]]) -> set[int]:
    matched: set[int] = set()
    for r in results:
        pk = structured_keys(r)
        for gi, gk in enumerate(gg):
            if pk & gk:
                matched.add(gi)
    return matched


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


async def _run(engine: ContestEngine, bq: dict, cfg) -> dict:
    """运行单个查询，返回指标行（异常→failed=True，绝不 silent 吞错）。"""
    qid = bq["query_id"]
    gg = match_gold(bq)
    row = {"query_id": qid, "question": bq["query"], "failed": False, "error": ""}
    try:
        res = await engine.search(bq["query"], query_id=qid, mode=cfg.mode)
    except Exception as e:  # noqa: BLE001
        row["failed"] = True
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    t = res.trace
    # System cost
    row["api_calls"] = t.api_calls
    row["llm_calls"] = t.llm_calls
    row["tokens"] = t.input_tokens + t.output_tokens
    row["latency_ms"] = t.total_latency_ms

    # Retrieval stage (标准化字段)
    row["retrieval_total_candidates"] = t.returned_paper_count
    row["retrieval_deduplicated_candidates"] = t.deduplicated_candidate_count
    # retrieval_raw_unique_gold: 需要 pool 数据，当前 trace 未记录，暂时无法计算

    # Final output stage (标准化字段)
    row["final_output_size"] = len(res.results)
    matched = matched_gold_ids(res.results, gg)
    row["final_unique_gold"] = len(matched)
    row["final_gold_instances"] = len(matched)  # 理论上=final_unique_gold（dedup 成功）
    row["total_gold_papers"] = len(gg)

    # Metrics (基于 final output)
    row["precision"] = len(matched) / len(res.results) if res.results else 0.0
    row["recall"] = len(matched) / len(gg) if gg else 0.0
    row["f1"] = (2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
                 if row["precision"] + row["recall"] else 0.0)

    # Structured-output validity
    invalid = [r.title.strip() for r in res.results if not (r.title or "").strip()]
    row["invalid_empty_title"] = len(invalid)

    # Round-2 accounting (标准化字段)
    row["final_round2_papers"] = len([1 for r in res.results if r.retrieval_round == 2])
    if cfg.mode == "deep":
        dec = t.round2_decision
        row["round2_planned_queries"] = len(dec.followups)
        row["round2_executed_queries"] = sum(1 for fu in dec.followups if fu.status == "executed")
        row["round2_filtered_queries"] = sum(1 for fu in dec.followups if fu.status not in {"executed", "keep"})
        row["round2_newly_discovered_papers"] = len(t.newly_discovered_papers)

    return row


async def _main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode.lower()
    if args.offline is not None:
        cfg.offline = args.offline

    queries = load_pasa(DATA)
    bq_by_id = {q["query_id"]: q for q in queries}

    # 只跑有 frozen plan 的查询（S1 用与 M3-R/M5A 相同的 22 查询集）
    engine = ContestEngine(cfg)
    frozen_ids = list(engine._frozen_plans)
    if args.limit:
        frozen_ids = frozen_ids[: args.limit]

    rows: list[dict] = []
    for i, qid in enumerate(frozen_ids, 1):
        bq = bq_by_id.get(qid, {"query_id": qid, "query": "", "gold": []})
        r = await _run(engine, bq, cfg)
        rows.append(r)
        tag = "FAIL" if r["failed"] else "ok"
        fg = r.get("final_unique_gold", 0)
        tg = r.get("total_gold_papers", 0)
        print(f"  [{i}/{len(frozen_ids)}] {qid} {tag} "
              f"gold={fg}/{tg} "
              f"f1={fmt(r.get('f1', 0))} api={r.get('api_calls', 0)} "
              f"{r.get('error', '')[:80]}")

    ok = [r for r in rows if not r["failed"]]
    failed = [r for r in rows if r["failed"]]
    n = len(ok)
    def avg(key):
        vals = [r.get(key, 0) for r in ok]
        return sum(vals) / n if n else 0.0

    # Aggregate Gold stats (标准化字段)
    total_final_unique_gold = sum(r.get("final_unique_gold", 0) for r in ok)
    total_gold_papers = sum(r.get("total_gold_papers", 0) for r in ok)

    summary = {
        "mode": cfg.mode,
        "offline": cfg.offline,
        "eval_corpus_version": EVAL_CORPUS_VERSION,
        "queries_run": len(rows),
        "ok": n,
        "failed": len(failed),
        "failures": [r["query_id"] for r in failed],
        # Metrics (基于 final output)
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        # Gold stats (标准化字段)
        "total_final_unique_gold": total_final_unique_gold,
        "total_gold_papers": total_gold_papers,
        # System cost
        "mean_api_calls": avg("api_calls"),
        "mean_llm_calls": avg("llm_calls"),
        "mean_tokens": avg("tokens"),
        "mean_latency_ms": avg("latency_ms"),
        # Stage stats
        "mean_retrieval_total_candidates": avg("retrieval_total_candidates"),
        "mean_final_output_size": avg("final_output_size"),
        # Validity
        "invalid_empty_title_total": sum(r.get("invalid_empty_title", 0) for r in ok),
    }

    # Round-2 aggregate (DEEP only)
    if cfg.mode == "deep":
        summary["round2_planned_queries_total"] = sum(r.get("round2_planned_queries", 0) for r in ok)
        summary["round2_executed_queries_total"] = sum(r.get("round2_executed_queries", 0) for r in ok)
        summary["round2_filtered_queries_total"] = sum(r.get("round2_filtered_queries", 0) for r in ok)
        summary["round2_newly_discovered_papers_total"] = sum(r.get("round2_newly_discovered_papers", 0) for r in ok)
        summary["final_round2_papers_total"] = sum(r.get("final_round2_papers", 0) for r in ok)

    print(f"\n=== S1 EVAL {cfg.mode.upper()} ({'offline' if cfg.offline else 'online'}) ===")
    print(f"Corpus: {EVAL_CORPUS_VERSION}")
    print(f"queries ok={n} failed={len(failed)} {failed if failed else ''}")
    print(f"P={fmt(summary['precision'])} R={fmt(summary['recall'])} F1={fmt(summary['f1'])}")
    print(f"final unique gold={total_final_unique_gold}/{total_gold_papers}")
    print(f"cost: api={fmt(summary['mean_api_calls'],1)}/q llm={fmt(summary['mean_llm_calls'],1)}/q "
          f"tok={fmt(summary['mean_tokens'],1)}/q lat={fmt(summary['mean_latency_ms'],1)}ms/q")
    print(f"retrieved={fmt(summary['mean_retrieval_total_candidates'],1)}/q "
          f"final={fmt(summary['mean_final_output_size'],1)}/q "
          f"empty_title_total={summary['invalid_empty_title_total']}")

    if cfg.mode == "deep":
        print(f"round2: planned={summary['round2_planned_queries_total']} "
              f"executed={summary['round2_executed_queries_total']} "
              f"filtered={summary['round2_filtered_queries_total']} "
              f"new_papers={summary['round2_newly_discovered_papers_total']} "
              f"final_r2_papers={summary['final_round2_papers_total']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[S1:run_eval] 已写入 {args.out}")

    return 0 if not failed else 1


def main() -> int:
    p = argparse.ArgumentParser(description="S1 批量评测（FAST/DEEP × offline/online）")
    p.add_argument("--mode", choices=["fast", "deep"], default="fast")
    p.add_argument("--offline", dest="offline", action="store_true", default=None)
    p.add_argument("--online", dest="offline", action="store_false")
    p.add_argument("--config", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.config is None:
        args.config = str(REPO / "configs" / f"{args.mode}.yaml")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
