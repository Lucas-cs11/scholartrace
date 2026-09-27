"""运行 B2 评测：查询理解 + 子查询召回 + S2 引文扩展 -> F1。

用法：
    python scripts/run_b2.py                     # 跑全部挑战 query
    python scripts/run_b2.py --limit 5           # 只跑前 5 条
    python scripts/run_b2.py --query "..."       # 单条调试
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
    parser.add_argument("--experiment", type=str, default="B2")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cache", type=str, default="eval/runs/_recall_cache.jsonl")
    parser.add_argument("--plan-cache", type=str, default="eval/runs/_plan_cache.jsonl")
    args = parser.parse_args()

    engine = SearchEngine()
    engine.load_recall_cache(args.cache)
    engine.load_plan_cache(args.plan_cache)

    async def search_fn(query: str, top_k: int):
        return await engine.search_b2(query, top_k=top_k)

    if args.query:
        results, telemetry, _ = await engine.search_b2(args.query, top_k=args.top_k)
        print(f"\n=== 单条调试: {args.query} ===")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. [{r.label.value.upper():7s}] {r.score:.3f} | {r.paper.title[:55]} | {r.paper.venue or ''} {r.paper.year}")
        print(f"  api_calls={telemetry.api_calls} llm_calls={telemetry.llm_calls} tokens={telemetry.total_tokens} fallback={telemetry.fallback_count}")
        return

    reports = await evaluate(search_fn, settings.gold_path, settings.runs_dir,
                             experiment_id=args.experiment, top_k=args.top_k, limit=args.limit)
    summary = summarize(reports)
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")
    engine.save_recall_cache(args.cache)
    engine.save_plan_cache(args.plan_cache)


if __name__ == "__main__":
    asyncio.run(main())
