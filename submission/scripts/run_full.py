"""运行 FULL 全链路评测：查询理解 + 子查询召回 + 多轮引文(早停) + LLM 精排(证据链)。

用法：
    python scripts/run_full.py                     # 跑全部挑战 query
    python scripts/run_full.py --limit 5           # 只跑前 5 条
    python scripts/run_full.py --query "..."       # 单条调试
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from eval.harness import evaluate, summarize
from src.search import SearchEngine


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="FULL")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cache", type=str, default="eval/runs/_recall_cache.jsonl")
    parser.add_argument("--plan-cache", type=str, default="eval/runs/_plan_cache.jsonl")
    args = parser.parse_args()

    engine = SearchEngine()
    engine.load_plan_cache(args.plan_cache)
    engine.load_recall_cache(args.cache)

    async def search_fn(query: str, top_k: int):
        return await engine.search_full(query, top_k=top_k)

    if args.query:
        results, telemetry, _ = await engine.search_full(args.query, top_k=args.top_k)
        print(f"\n=== 单条调试: {args.query} ===")
        for i, r in enumerate(results[:10], 1):
            cov = ",".join(r.constraint_coverage.keys())[:30]
            print(f"  {i:2d}. [{r.label.value.upper():7s}] {r.score:.3f} | {r.paper.title[:50]} | cov={cov}")
        print(f"  api_calls={telemetry.api_calls} llm_calls={telemetry.llm_calls} "
              f"tokens={telemetry.total_tokens} fallback={telemetry.fallback_count}")
        return

    reports = await evaluate(search_fn, settings.gold_path, settings.runs_dir,
                             experiment_id=args.experiment, top_k=args.top_k, limit=args.limit)
    engine.save_recall_cache(args.cache)
    engine.save_plan_cache(args.plan_cache)
    summary = summarize(reports)
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
