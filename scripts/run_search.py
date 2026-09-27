"""S1 unified CLI：单问题运行 FAST/DEEP 比赛引擎，输出结构化结果 + SearchTrace。

用法：
    python3 scripts/run_search.py --question "<Q>" --mode fast [--offline|--online] [--query-id X]
    python3 scripts/run_search.py --question "<Q>" --mode deep --config configs/deep.yaml --out out.json

约束（S1 规格）：
- offline 模式只读 frozen plan/cache，缺失即报错（无 silent fallback）。
- 元数据（title/author/DOI/year/venue）一律来自学术数据，禁止 LLM 生成。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from s1.config import load_config  # noqa: E402
from s1.pipeline import ContestEngine  # noqa: E402


def _result_to_json(res) -> dict:
    return {
        "question": res.question,
        "mode": res.mode,
        "query_id": res.query_id,
        "results": [r.model_dump() for r in res.results],
        "trace": res.trace.model_dump(),
    }


async def _main(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode.lower()
    if args.offline is not None:
        cfg.offline = args.offline

    engine = ContestEngine(cfg)
    res = await engine.search(args.question, query_id=args.query_id, mode=cfg.mode)

    payload = _result_to_json(res)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[S1:run_search] 已写入 {args.out}")

    print(f"\n=== S1 {cfg.mode.upper()} result — qid={res.query_id!r} ===")
    print(f"question: {res.question}")
    t = res.trace
    print(f"trace: api_calls={t.api_calls} llm_calls={t.llm_calls} "
          f"tokens={t.input_tokens + t.output_tokens} latency_ms={t.total_latency_ms} "
          f"retrieved={t.returned_paper_count} dedup={t.deduplicated_candidate_count} "
          f"final={len(res.results)}")
    for r in res.results[:10]:
        au = ", ".join(r.authors[:2])
        print(f"  [{r.relevance_label:6} {r.relevance_score:+.2f}] R{r.retrieval_round} "
              f"{r.title[:70]} ({au})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="S1 统一比赛引擎 CLI")
    p.add_argument("--question", required=True)
    p.add_argument("--mode", choices=["fast", "deep"], default=None)
    p.add_argument("--offline", dest="offline", action="store_true", default=None)
    p.add_argument("--online", dest="offline", action="store_false")
    p.add_argument("--query-id", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.config is None:
        args.config = str(REPO / "configs" / f"{args.mode or 'fast'}.yaml")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
