"""运行 B0 基线评测：单 query -> OpenAlex -> 词法排序 -> F1。

用法：
    python scripts/run_b0.py                     # 跑全部挑战 query
    python scripts/run_b0.py --limit 5           # 只跑前 5 条
    python scripts/run_b0.py --query "..."       # 单条调试
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from eval.harness import evaluate, load_challenges, summarize
from src.search import SearchEngine


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="B0")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    engine = SearchEngine()

    async def search_fn(query: str, top_k: int):
        return await engine.search(query, top_k=top_k)

    if args.query:
        # 单条调试模式：打印排序明细
        results, telemetry, _ = await engine.search(args.query, top_k=args.top_k)
        print(f"\n=== 单条调试: {args.query} ===")
        for i, r in enumerate(results[:10], 1):
            print(f"  {i:2d}. [{r.label.value.upper():7s}] {r.score:.3f} | {r.paper.title} | {r.paper.venue or ''} {r.paper.year}")
        print(f"  api_calls={telemetry.api_calls} tokens={telemetry.total_tokens}")
        return

    reports = await evaluate(search_fn, settings.gold_path, settings.runs_dir,
                             experiment_id=args.experiment, top_k=args.top_k, limit=args.limit)
    summary = summarize(reports)
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
